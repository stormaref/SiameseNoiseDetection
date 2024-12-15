import torch
from torch.utils.data import Subset, DataLoader
from models.dataset import DatasetPairs, DatasetSingle, PositiveSamplingDatasetPairs
from models.siamese import SiameseNetwork, SimpleSiamese
from models.noise import LabelNoiseAdder
from models.detector import NoiseDetector
from models.fold import CustomKFoldSplitter
from models.predefined import InstanceDependentNoiseAdder
from torchvision import transforms
import PIL
import matplotlib.pyplot as plt
import math
import os
import pickle
import numpy as np
from tqdm import tqdm
import csv

class NoiseCleaner:
    def __init__(self, dataset, model_save_path, inner_folds_num, outer_folds_num, noise_type, model, train_noise_level=0.1, epochs_num=30,
                 train_pairs=6000, val_pairs=1000, transform=None, embedding_dimension=128, lr=0.001, optimizer='Adam', distance_meter='euclidian',
                 patience=5, weight_decay=0.001, training_batch_size=256, pre_trained=True, dropout_prob=0.5, contrastive_ratio=3,
                 augmented_transform=None, trainable=True, pair_validation=True, label_smoothing=0.1, loss='ce', cnn_size=None, margin=5,
                 freeze_epoch=10, noisy_indices_path=''):
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
        self.loss = loss
        self.cnn_size = cnn_size
        self.margin = margin
        self.freeze_epoch = freeze_epoch
        self.noisy_indices_path = noisy_indices_path
        
        if noise_type == 'idn':
            image_size = self.get_image_size()
            self.train_noise_adder = InstanceDependentNoiseAdder(dataset, image_size=image_size, ratio=train_noise_level, num_classes=10)
            self.train_noise_adder.add_noise()
        elif noise_type == 'iin':
            self.train_noise_adder = LabelNoiseAdder(dataset, noise_level=train_noise_level, num_classes=10)
            self.train_noise_adder.add_noise()
        elif noise_type == 'none':
            a = 2
        else:
            raise ValueError('Noise type should be either "idn" or "iin"')
        
        if noise_type != 'none':
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
        self.ensure_model_directory_exists()
        
    def ensure_model_directory_exists(self):
        model_dir = os.path.dirname(self.model_save_path.format(0))
        os.makedirs(model_dir, exist_ok=True)
        
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
            file_path = self.noisy_indices_path.format(fold + 1)
            if os.path.exists(file_path):
                print(f'Skipping outer fold {fold + 1} with results:')
                self.process_and_load_noisy_indices(file_path)
                continue
            train_indices, val_indices = self.custom_kfold_splitter.get_fold(fold)
            self.handle_fold(fold, train_indices, val_indices)
        print('Predicted noise indices accuracy:')
        self.train_noise_adder.calculate_noised_label_percentage(self.predicted_noise_indices)
        precision, recall = self.train_noise_adder.calculate_precision_recall(self.predicted_noise_indices)
        print(f'Precision: {precision}, Recall: {recall}')
        self.train_noise_adder.plot_confusion_matrix(self.predicted_noise_indices)
        self.clean_dataset = self.remove_noisy_samples(self.dataset, self.predicted_noise_indices)
    
    def save_cleaned_cifar_dataset(self, save_dir: str, dataset_name: str):
        if self.clean_dataset is None:
            raise ValueError("The cleaned dataset is not available. Call the `clean` method first.")

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{dataset_name}.pkl")
        with open(save_path, "wb") as f:
            for (img, label) in tqdm(self.clean_dataset):
                img_array = np.array(img)
                entry = {'data': img_array, 'label': label}
                pickle.dump(entry, f)

        print(f"Cleaned dataset saved to {save_path}")
    
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
                                       augmented_transform=self.augmented_transform, trainable=self.trainable, label_smoothing=self.label_smoothing,
                                       loss=self.loss, cnn_size=self.cnn_size, margin=self.margin, freeze_epoch=self.freeze_epoch)
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
        
        self.save_noisy_indices(fold, predicted_noise_original_indices)
        
    def process_and_load_noisy_indices(self, file_path):
        noisy_indices = []
        with open(file_path, mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                noisy_indices.extend(map(int, row))

        self.train_noise_adder.calculate_noised_label_percentage(noisy_indices)
        self.predicted_noise_indices.extend(noisy_indices)
        print(f'Loaded {len(noisy_indices)} noisy indices from {file_path}')
        
    def save_noisy_indices(self, fold, noisy_indices):
        file_path = self.noisy_indices_path.format(fold + 1)
        model_dir = os.path.dirname(file_path)
        os.makedirs(model_dir, exist_ok=True)

        with open(file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(noisy_indices)

        print(f'Noisy indices for fold {fold + 1} saved to {file_path}')