import torch
from torch.utils.data import Subset, DataLoader
from models.dataset import DatasetPairs, DatasetSingle
from models.siamese import SiameseNetwork, SimpleSiamese
from models.noise import LabelNoiseAdder
from models.detector import NoiseDetector
from models.fold import CustomKFoldSplitter
from models.predefined import InstanceDependentNoiseAdder
from torchvision import transforms
import PIL
import matplotlib.pyplot as plt
import math

class NoiseCleaner:
    def __init__(self, dataset, model_save_path, inner_folds_num, outer_folds_num, noise_type, model, train_noise_level=0.1, epochs_num=30,
                 train_pairs=6000, val_pairs=1000, transform=None, embedding_dimension=128, lr=0.001, optimizer='Adam', distance_meter='euclidian',
                 patience=5, weight_decay=0.001, training_batch_size=256, pre_trained=True, dropout_prob=0.5, contrastive_ratio=3,
                 augmented_transform=None, trainable=True, pair_validation=True, label_smoothing=0.1):
        self.dataset = dataset
        self.lr = lr
        self.weight_decay = weight_decay
        self.training_batch_size = training_batch_size
        self.pre_trained = pre_trained
        self.dropout_prob = dropout_prob
        self.contrastive_ratio = contrastive_ratio
        self.distance_meter = distance_meter
        self.augmented_transform = augmented_transform
        self.trainable = trainable
        self.pair_validation = pair_validation
        self.label_smoothing = label_smoothing
        
        if noise_type == 'idn':
            image_size = self.get_image_size()
            self.train_noise_adder = InstanceDependentNoiseAdder(dataset, image_size=image_size, ratio=train_noise_level, num_classes=10)
            self.train_noise_adder.add_noise()
        elif noise_type == 'iin':
            self.train_noise_adder = LabelNoiseAdder(dataset, noise_level=train_noise_level, num_classes=10)
            self.train_noise_adder.add_noise()
        else:
            raise ValueError('Noise type should be either "idn" or "iin"')
        
        print(f'noise count: {len(self.train_noise_adder.get_noisy_indices())} out of {len(dataset)} data')
        self.device = torch.device('cuda')
        self.model_save_path = model_save_path
        self.inner_folds_num = inner_folds_num
        self.outer_folds_num = outer_folds_num
        self.custom_kfold_splitter = CustomKFoldSplitter(dataset_size=len(dataset), labels=dataset.targets, num_folds=outer_folds_num, shuffle=True)
        self.predicted_noise_indices = []
        self.clean_dataset = None
        self.model = model
        self.epochs_num = epochs_num
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.transform = transform
        self.embedding_dimension = embedding_dimension
        self.optimzer = optimizer
        self.patience = patience
        
    def get_image_size(self):
        sample, _ = self.dataset[0]
        if isinstance(sample, PIL.Image.Image):
            sample = transforms.ToTensor()(sample)
        return sample.shape[0] * sample.shape[1] * sample.shape[2]
        
    def remove_noisy_samples(self, dataset, noisy_indices):
        clean_indices = [i for i in range(len(dataset)) if i not in noisy_indices]
        cleaned_dataset = Subset(dataset, clean_indices)
        return cleaned_dataset

    def clean(self):
        for fold in range(self.outer_folds_num):
            train_indices, val_indices = self.custom_kfold_splitter.get_fold(fold)
            self.handle_fold(fold, train_indices, val_indices)
        print('Predicted noise indices accuracy:')
        self.train_noise_adder.calculate_noised_label_percentage(self.predicted_noise_indices)
        self.clean_dataset = self.remove_noisy_samples(self.dataset, self.predicted_noise_indices)
        return self.clean_dataset
    
    def handle_fold(self, fold, train_indices, val_indices):
        print(f'handling big fold {fold + 1}/{self.outer_folds_num}')
        train_subset = Subset(self.dataset, train_indices)
        val_subset = Subset(self.dataset, val_indices)
        number_of_pairs = math.floor(len(val_subset) * (math.e - 2))
        print(f'number_of_pairs: {number_of_pairs}')
        
        noise_detector = NoiseDetector(SiameseNetwork, train_subset, self.device, model_save_path=self.model_save_path, num_folds=self.inner_folds_num, 
                                       model=self.model, train_pairs=self.train_pairs, val_pairs=self.val_pairs, transform=self.transform, 
                                       embedding_dimension=self.embedding_dimension, optimizer=self.optimzer, patience=self.patience,
                                       weight_decay=self.weight_decay, batch_size=self.training_batch_size, pre_trained=self.pre_trained,
                                       dropout_prob=self.dropout_prob, contrastive_ratio=self.contrastive_ratio, distance_meter=self.distance_meter,
                                       augmented_transform=self.augmented_transform, trainable=self.trainable, label_smoothing=self.label_smoothing)
        noise_detector.train_models(num_epochs=self.epochs_num, lr=self.lr)
       
        if self.pair_validation:
            test_dataset_pair = DatasetPairs(val_subset, num_pairs_per_epoch=number_of_pairs, transform=self.transform)
            test_loader = DataLoader(test_dataset_pair, batch_size=1024, shuffle=False)
            wrong_preds = noise_detector.evaluate_noisy_samples(test_loader)
        else:
            test_dataset = DatasetSingle(val_subset, transform=self.transform)
            test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
            wrong_preds = noise_detector.evaluate_noisy_samples_one_by_one(test_loader)
        predicted_noise_indices = [idx for (idx, count) in wrong_preds.items() if count >= self.inner_folds_num]
        counts = [count for (idx, count) in wrong_preds.items()]
        plt.hist(counts)
        plt.show()
        predicted_noise_original_indices = self.custom_kfold_splitter.get_original_indices(fold, predicted_noise_indices)
        print(f'Predicted noise indices: {predicted_noise_original_indices}')
        self.train_noise_adder.calculate_noised_label_percentage(predicted_noise_original_indices)
        self.predicted_noise_indices.extend(predicted_noise_original_indices)