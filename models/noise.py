import random
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from models.interfaces import NoiseAdder
import matplotlib.pyplot as plt
import seaborn as sns

class LabelNoiseAdder(NoiseAdder):
    def __init__(self, dataset, noise_level=0.1, num_classes=10):
        self.dataset = dataset
        self.noise_level = noise_level
        self.num_classes = num_classes
        self.noisy_indices = []
        self.orginal_labels = np.array(self.dataset.targets, copy=True)
        self.noisy_labels = None

    def add_noise(self):
        num_noisy_samples = int(len(self.dataset) * self.noise_level)
        self.noisy_indices = random.sample(range(len(self.dataset)), num_noisy_samples)
        
        for idx in self.noisy_indices:
            original_label = self.dataset.targets[idx]
            noisy_label = random.randint(0, self.num_classes - 1)
            
            while noisy_label == original_label:
                noisy_label = random.randint(0, self.num_classes - 1)
                
            self.dataset.targets[idx] = noisy_label
        
        self.noisy_labels = np.array(self.dataset.targets, copy=True)

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