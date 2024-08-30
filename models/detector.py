import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from collections import defaultdict
from torchvision import transforms
from models.dataset import DatasetPairs
from models.contrastive import ContrastiveLoss
from models.trainer import Trainer
from models.visualizer import EmbeddingVisualizer
from models.tester import Tester
from models.siamese import SiameseNetwork
import numpy as np

class NoiseDetector:
    def __init__(self, model_class: SiameseNetwork, dataset, device, num_classes=10, model='resnet18', batch_size=256, num_folds=10,
                 model_save_path="model_fold_{}.pth", transform=None, train_pairs=12000, val_pairs=5000, embedding_dimension=128,
                 optimizer= 'Adam', patience=5, weight_decay=0.001, pre_trained=True, dropout_prob=0.5, contrastive_ratio=2,
                 distance_meter='euclidian', augmented_transform=None, trainable=True):
        self.model_class = model_class
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.num_folds = num_folds
        self.trainable = trainable
        self.model_save_path = model_save_path
        if transform is None:
            raise ValueError('transform should be determined')
        else:
            self.transform = transform
        if augmented_transform is None:
            raise ValueError('augmented transform should be determined')
        else:
            self.augmented_transform = augmented_transform
        # self.models = [self.model_class(num_classes=num_classes, dropout_prob=dropout_prob, pre_trained=pre_trained, model=model, embedding_dimension=embedding_dimension).to(self.device) for _ in range(num_folds)]
        self.num_classes=num_classes
        self.dropout_prob=dropout_prob
        self.pre_trained=pre_trained 
        self.model=model
        self.embedding_dimension=embedding_dimension
        self.distance_meter = distance_meter
        
        self.kf = StratifiedKFold(n_splits=num_folds, shuffle=True)
        self.trainers = []
        self.testers = []
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        if optimizer != 'Adam' and optimizer != 'SGD':
            raise ValueError('optimizer')
        self.optimizer = optimizer
        self.patience = patience
        self.weight_decay = weight_decay
        self.contrastive_ratio = contrastive_ratio
        
    def get_targets(self):
        dataset = self.dataset
        if hasattr(dataset, 'targets'):
            return dataset.targets
        elif isinstance(dataset, torch.utils.data.Subset):
            return [dataset.dataset.targets[i] for i in dataset.indices]
        else:
            raise ValueError("The dataset does not have 'targets' attribute and is not a Subset with accessible targets.")


    def train_models(self, num_epochs=10, skip=0, lr=0.001):
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(self.dataset, self.get_targets())):
            if fold <= (skip - 1):
                continue
            print(f'Training fold {fold + 1}/{self.num_folds}...')
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)
            train_loader = DataLoader(DatasetPairs(train_subset, smart_count=False, num_pairs_per_epoch=self.train_pairs, 
                                                   transform=self.augmented_transform), batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(DatasetPairs(val_subset, num_pairs_per_epoch=self.val_pairs, transform=self.transform),
                                    batch_size=8, shuffle=False)

            model = self.model_class(num_classes=self.num_classes, dropout_prob=self.dropout_prob, pre_trained=self.pre_trained, 
                                     model=self.model, embedding_dimension=self.embedding_dimension, trainable=self.trainable).to(self.device)
            if self.optimizer == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=self.weight_decay)
            elif self.optimizer == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=self.weight_decay)
            criterion = nn.CrossEntropyLoss()
            contrastive_criterion = ContrastiveLoss(distance_meter=self.distance_meter)

            trainer = Trainer(model, contrastive_criterion, criterion, optimizer, train_loader, self.device,
                              val_dataloader=val_loader, patience=self.patience, checkpoint_path='val_best_model.pth',
                              contrastive_ratio=self.contrastive_ratio)

            trainer.train(num_epochs)

            if fold == 0:
                trainer.plot_losses()                    
                trainer.plot_accuracies()
                try:
                    visualizer = EmbeddingVisualizer(model, val_loader, self.device)
                    embeddings, real_labels, predicted_labels, indices, incorrect_images = visualizer.extract_embeddings()
                    visualizer.visualize(embeddings, real_labels, predicted_labels)
                    unique, counts = np.unique(predicted_labels, return_counts=True)
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

    def get_predictions(self, dataloader):
        all_predictions = defaultdict(list)

        for fold in range(self.num_folds):
            model = self.model_class().to(self.device)
            model_save_path = self.model_save_path.format(fold + 1)
            model.load_state_dict(torch.load(model_save_path, map_location=self.device))
            model.eval()

            with torch.no_grad():
                seen_indices = set()
                for img1, img2, label1, label2, i, j in tqdm(dataloader, desc=f"Extracting Predictions for fold {fold + 1}"):
                    img1, img2 = img1.to(self.device), img2.to(self.device)
                    emb1, emb2, class1, class2 = model(img1, img2)
                    outputs1 = nn.functional.softmax(class1, dim=1)
                    outputs2 = nn.functional.softmax(class2, dim=1)
                    
                    for idx, idx_i in enumerate(i):
                        if idx_i.item() not in seen_indices:
                            all_predictions[idx_i.item()].append(outputs1[idx].cpu().numpy())
                            seen_indices.add(idx_i.item())
                    
                    for idx, idx_j in enumerate(j):
                        if idx_j.item() not in seen_indices:
                            all_predictions[idx_j.item()].append(outputs2[idx].cpu().numpy())
                            seen_indices.add(idx_j.item())

        return all_predictions
    
    def evaluate_noisy_samples(self, dataloader):
        wrong_predictions_count = defaultdict(int)

        for fold in range(self.num_folds):
            # Reload the model
            model = self.model_class().to(self.device)
            model_save_path = self.model_save_path.format(fold + 1)
            model.load_state_dict(torch.load(model_save_path, map_location=self.device))
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

        for fold in range(self.num_folds):
            # Reload the model
            model: SiameseNetwork = self.model_class().to(self.device)
            model_save_path = self.model_save_path.format(fold + 1)
            model.load_state_dict(torch.load(model_save_path, map_location=self.device))
            model.eval()

            with torch.no_grad():
                seen_indices = set()
                for img, label, i in tqdm(dataloader, desc=f"Evaluating Noisy Samples for fold {fold + 1}"):
                    img = img.to(self.device)
                    _, cls = model.classify(img)
                    
                    _, pred = torch.max(cls, 1)
                    
                    for idx, idx_i in enumerate(i):
                        if idx_i.item() not in seen_indices:
                            if pred[idx].item() != label[idx].item():
                                wrong_predictions_count[idx_i.item()] += 1
                                seen_indices.add(idx_i.item())
                                
            model.to('cpu')
            torch.cuda.empty_cache()

        return wrong_predictions_count