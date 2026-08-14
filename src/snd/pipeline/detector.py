import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from collections import defaultdict
from snd.data.dataset import DatasetPairs
from snd.training.contrastive import ContrastiveLoss
from snd.training.trainer import Trainer
from snd.evaluation.visualizer import EmbeddingVisualizer
from snd.evaluation.tester import Tester
from snd.models.siamese import SiameseNetwork
from snd.training.hyperparams import TrainingConfig
import numpy as np
import os


class NoiseDetector:
    """Main class for training and using Siamese networks to detect noisy labels in datasets.

    Implements k-fold cross-validation training and various evaluation methods.
    """

    def __init__(self, model_class: SiameseNetwork, dataset, device, config: TrainingConfig,
                 num_folds=10, model_save_path="model_fold_{}.pth", prediction_path=''):
        """Initialize the noise detector.

        Args:
            model_class: Siamese network class to instantiate per fold
            dataset: the outer-fold training subset
            device: torch device to train on
            config: shared model/optimization hyperparameters
            num_folds: size of the inner ensemble
            model_save_path: per-fold checkpoint path, with a `{}` for the fold number
            prediction_path: per-fold prediction CSV path, with a `{}` for the fold number
        """
        if config.transform is None:
            raise ValueError('transform should be determined')
        if config.augmented_transform is None:
            raise ValueError('augmented transform should be determined')

        self.model_class = model_class
        self.dataset = dataset
        self.device = device
        self.config = config
        self.num_folds = num_folds
        self.model_save_path = model_save_path
        self.prediction_path = prediction_path

        self.kf = StratifiedKFold(n_splits=num_folds, shuffle=True)
        self.trainers = []
        self.testers = []

    def __getattr__(self, name):
        """Read hyperparameters straight off the config (`self.margin`, `self.loss`, ...)."""
        try:
            config = self.__dict__['config']
        except KeyError:
            raise AttributeError(name) from None
        try:
            return getattr(config, name)
        except AttributeError:
            raise AttributeError(
                f'{type(self).__name__!r} object has no attribute {name!r}') from None

    def get_targets(self):
        """Extract target labels from dataset, handling different dataset types."""
        dataset = self.dataset
        if hasattr(dataset, 'targets'):
            return dataset.targets
        elif isinstance(dataset, torch.utils.data.Subset):
            return [dataset.dataset.targets[i] for i in dataset.indices]
        else:
            raise ValueError(
                "The dataset does not have 'targets' attribute and is not a Subset with accessible targets.")

    def __train_a_model(self, fold, train_idx, val_idx, num_epochs, lr):
        print(f'Training fold {fold + 1}/{self.num_folds}...')
        train_subset = Subset(self.dataset, train_idx)
        val_subset = Subset(self.dataset, val_idx)

        train_dataset = DatasetPairs(train_subset, smart_count=False,
                                     num_pairs_per_epoch=self.train_pairs, transform=self.augmented_transform)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)

        val_dataset = DatasetPairs(
            val_subset, num_pairs_per_epoch=self.val_pairs, transform=self.transform)
        val_loader = DataLoader(
            val_dataset, batch_size=64, shuffle=False, num_workers=16)

        model = self.model_class(num_classes=self.num_classes, dropout_prob=self.dropout_prob, pre_trained=self.pre_trained,
                                 model=self.model, embedding_dimension=self.embedding_dimension, trainable=self.trainable,
                                 cnn_size=self.cnn_size, middle_size=self.siamese_middle_size, parallel=self.parallel).to(self.device)

        if self.optimizer == 'Adam':
            optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                  momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError('optimizer not supported')

        if self.loss == 'ce':
            criterion = nn.CrossEntropyLoss(
                label_smoothing=self.label_smoothing)
        else:
            raise ValueError('loss function not supported')
        contrastive_criterion = ContrastiveLoss(
            distance_meter=self.distance_meter, margin=self.margin)

        trainer = Trainer(model, contrastive_criterion, criterion, optimizer, train_loader, self.device,
                          val_dataloader=val_loader, patience=self.patience, checkpoint_path='val_best_model.pth',
                          contrastive_ratio=self.contrastive_ratio, freeze_epoch=self.freeze_epoch)

        trainer.train(num_epochs)

        if fold == 0:
            trainer.plot_losses()
            trainer.plot_accuracies()
            try:
                visualizer = EmbeddingVisualizer(
                    model=model, dataloader=val_loader, device=self.device, num_class=self.num_classes)
                embeddings, real_labels, predicted_labels, indices, incorrect_images = visualizer.extract_embeddings()
                visualizer.visualize(embeddings, real_labels, predicted_labels)
                unique, counts = np.unique(
                    predicted_labels, return_counts=True)
                print('value counts for predicted:')
                print(np.asarray((unique, counts)).T)
                unique, counts = np.unique(real_labels, return_counts=True)
                print('value counts for real:')
                print(np.asarray((unique, counts)).T)
            except:
                a = 2

        tester = Tester(model, val_loader, self.device)
        tester.test()
        model_save_path = self.model_save_path.format(fold + 1)
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')

        model.to('cpu')
        torch.cuda.empty_cache()

        print(f'Finished training fold {fold + 1}')

    def train_models(self, num_epochs=10, skip=0, lr=0.001):
        """Train models using k-fold cross-validation with the specified parameters."""
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(self.dataset, self.get_targets())):
            if fold <= (skip - 1):
                continue

            model_save_path = self.model_save_path.format(fold + 1)
            if os.path.exists(model_save_path):
                print(f'Model {model_save_path} already exists, skipping...')
                continue

            self.__train_a_model(fold, train_idx, val_idx, num_epochs, lr)

    def evaluate_noisy_samples(self, dataloader):
        """Evaluate potential noisy samples by counting incorrect predictions across folds."""
        wrong_predictions_count = defaultdict(int)

        for fold in range(self.num_folds):
            # Reload the model
            model = self.model_class(num_classes=self.num_classes, dropout_prob=self.dropout_prob, pre_trained=self.pre_trained,
                                     model=self.model, embedding_dimension=self.embedding_dimension, trainable=self.trainable,
                                     cnn_size=self.cnn_size, middle_size=self.siamese_middle_size, parallel=self.parallel).to(self.device)
            model_save_path = self.model_save_path.format(fold + 1)
            model.load_state_dict(torch.load(
                model_save_path, map_location=self.device))
            model.eval()

            with torch.no_grad():
                seen_indices = set()
                for img1, img2, label1, label2, i, j in tqdm(dataloader, desc=f"Evaluating Noisy Samples for fold {fold + 1}"):
                    img1, img2 = img1.to(self.device), img2.to(self.device)
                    emb1, emb2, class1, class2 = model(img1, img2)

                    _, pred1 = torch.max(class1, 1)
                    _, pred2 = torch.max(class2, 1)

                    for idx, idx_i in enumerate(i):
                        if idx_i.item() not in seen_indices:
                            if pred1[idx].item() != label1[idx].item():
                                wrong_predictions_count[idx_i.item()] += 1
                                seen_indices.add(idx_i.item())

                    for idx, idx_j in enumerate(j):
                        if idx_i.item() not in seen_indices:
                            if pred2[idx].item() != label2[idx].item():
                                wrong_predictions_count[idx_j.item()] += 1
                                seen_indices.add(idx_i.item())

            model.to('cpu')
            torch.cuda.empty_cache()

    def evaluate_noisy_samples_one_by_one(self, dataloader):
        wrong_predictions_count = defaultdict(int)
        predictions = defaultdict(list)

        for fold in tqdm(range(self.num_folds), desc='Evaluating Noisy Samples'):
            model = self.model_class(num_classes=self.num_classes, dropout_prob=self.dropout_prob, pre_trained=self.pre_trained,
                                     model=self.model, embedding_dimension=self.embedding_dimension, trainable=self.trainable,
                                     cnn_size=self.cnn_size, middle_size=self.siamese_middle_size, parallel=self.parallel).to(self.device)
            model_save_path = self.model_save_path.format(fold + 1)
            model.load_state_dict(torch.load(
                model_save_path, map_location=self.device))
            model.eval()

            with torch.no_grad():
                seen_indices = set()
                for img, label, i in dataloader:
                    img = img.to(self.device)
                    _, cls = model.classify(img)

                    _, pred = torch.max(cls, 1)

                    for idx, idx_i in enumerate(i):
                        if idx_i.item() not in seen_indices:
                            predictions[idx_i.item()].append(pred[idx].item())
                            if pred[idx].item() != label[idx].item():
                                wrong_predictions_count[idx_i.item()] += 1
                                seen_indices.add(idx_i.item())

            model.to('cpu')
            torch.cuda.empty_cache()

        return wrong_predictions_count, predictions

    def analyze_latent(self, dataloader):
        latents = defaultdict(list)

        for fold in tqdm(range(self.num_folds), desc='Evaluating Noisy Samples'):
            model: SiameseNetwork = self.model_class(num_classes=self.num_classes, dropout_prob=self.dropout_prob, pre_trained=self.pre_trained,
                                                     model=self.model, embedding_dimension=self.embedding_dimension, trainable=self.trainable,
                                                     cnn_size=self.cnn_size, middle_size=self.siamese_middle_size, parallel=self.parallel).to(self.device)
            model_save_path = self.model_save_path.format(fold + 1)
            model.load_state_dict(torch.load(
                model_save_path, map_location=self.device))
            model.eval()

            with torch.no_grad():
                seen_indices = set()
                for img, label, i in dataloader:
                    img = img.to(self.device)
                    emb, _ = model.classify(img)

                    for idx, idx_i in enumerate(i):
                        if idx_i.item() not in seen_indices:
                            latents[idx_i.item()].append(emb[idx])
                            seen_indices.add(idx_i.item())

            model.to('cpu')
            torch.cuda.empty_cache()

        return latents
