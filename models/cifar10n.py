import torch
import numpy as np
from models.interfaces import NoiseAdder
from torchvision.datasets import CIFAR10
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class CIFAR10N(NoiseAdder):
    def __init__(self, dataset: CIFAR10):
        self.dataset = dataset
        self.noisy_labels = self.load_label(noise_path='data/cifar10n/data.pt', train_labels=dataset.targets, noise_type='aggre_label')
        self.noisy_indices = []
        for i in range(len(self.dataset.targets)):
            if self.dataset.targets[i] != self.noisy_labels[i]:
                self.noisy_indices.append(i)
    
    def add_noise(self):
        self.dataset.targets = np.array(self.noisy_labels, copy=True)
        
    def get_noisy_indices(self):
        return self.noisy_indices
    
    def calculate_noised_label_percentage(self, indices):
        intersection = set(indices) & set(self.noisy_indices)
        percentage = (len(intersection) / len(indices)) * 100
        print(f'{percentage}% accuracy in {len(indices)} data')
        return percentage
    
    def report(self, indices):
        predicted_labels = np.zeros(len(self.dataset))
        predicted_labels[indices] = 1
        real_labels = np.zeros(len(self.dataset))
        real_labels[self.noisy_indices] = 1
        labels = ['Clean', 'Noisy']
        print(classification_report(real_labels, predicted_labels, target_names=labels, digits=4))
        cm = confusion_matrix(real_labels, predicted_labels)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
    
    def ravel(self, indices):
        predicted_labels = np.zeros(len(self.dataset))
        predicted_labels[indices] = 1
        real_labels = np.zeros(len(self.dataset))
        real_labels[self.noisy_indices] = 1
        cm = confusion_matrix(real_labels, predicted_labels)
        return cm.ravel()
    
    def load_label(self, noise_path, train_labels, noise_type):
        noise_label = torch.load(noise_path, weights_only=False)
        if isinstance(noise_label, dict):
            if "clean_label" in noise_label.keys():
                clean_label = torch.tensor(noise_label['clean_label'])
                assert torch.sum(torch.tensor(train_labels) - clean_label) == 0 
                print(f'The overall noise rate is {1-np.mean(clean_label.numpy() == noise_label[noise_type])}')
            return noise_label[noise_type].reshape(-1)  
        else:
            raise Exception('Input Error')