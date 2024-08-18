import torch
from torch.utils.data import Subset, DataLoader
from models.dataset import DatasetPairs
from models.siamese import SiameseNetwork
from models.noise import LabelNoiseAdder
from models.detector import NoiseDetector
from models.fold import CustomKFoldSplitter
from models.predefined import InstanceDependentNoiseAdder
from torchvision import transforms
import PIL

class NoiseCleaner:
    def __init__(self, dataset, model_save_path, folds_num, noise_type, model, train_noise_level=0.1, epochs_num=30, train_pairs=6000, val_pairs=1000, transform=None):
        self.dataset = dataset
        
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
        self.folds_num = folds_num
        self.custom_kfold_splitter = CustomKFoldSplitter(dataset_size=len(dataset), labels=dataset.targets, num_folds=folds_num, shuffle=True)
        self.predicted_noise_indices = []
        self.clean_dataset = None
        self.model = model
        self.epochs_num = epochs_num
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.transform = transform
        
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
        for fold in range(self.folds_num):
            train_indices, val_indices = self.custom_kfold_splitter.get_fold(fold)
            self.handle_fold(fold, train_indices, val_indices)
        print('Predicted noise indices accuracy:')
        self.train_noise_adder.calculate_noised_label_percentage(self.predicted_noise_indices)
        self.clean_dataset = self.remove_noisy_samples(self.dataset, self.predicted_noise_indices)
        return self.clean_dataset
    
    def handle_fold(self, fold, train_indices, val_indices):
        train_subset = Subset(self.dataset, train_indices)
        val_subset = Subset(self.dataset, val_indices)
        
        noise_detector = NoiseDetector(SiameseNetwork, train_subset, self.device, model_save_path=self.model_save_path, num_folds=self.folds_num, 
                                       model=self.model, train_pairs=self.train_pairs, val_pairs=self.val_pairs, transform=self.transform)
        noise_detector.train_models(num_epochs=self.epochs_num)
       
        test_dataset_pair = DatasetPairs(val_subset, num_pairs_per_epoch=25000, transform=self.transform)
        test_loader = DataLoader(test_dataset_pair, batch_size=1024, shuffle=False)
        wrong_preds = noise_detector.evaluate_noisy_samples(test_loader)
        predicted_noise_indices = [idx for (idx, count) in wrong_preds.items() if count >= self.folds_num]
        predicted_noise_original_indices = self.custom_kfold_splitter.get_original_indices(fold, predicted_noise_indices)
        print(f'Predicted noise indices: {predicted_noise_original_indices}')
        self.train_noise_adder.calculate_noised_label_percentage(predicted_noise_original_indices)
        self.predicted_noise_indices.extend(predicted_noise_original_indices)