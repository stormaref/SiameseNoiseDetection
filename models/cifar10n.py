import torch
import numpy as np
from models.interfaces import NoiseAdder
from torchvision.datasets import CIFAR10
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class CIFAR10N(NoiseAdder):
    """Class for handling the CIFAR-10N dataset with real-world label noise.
    
    This implements the NoiseAdder interface for the CIFAR-10N dataset,
    which contains human-annotated noisy labels for CIFAR-10.
    """
    def __init__(self, dataset: CIFAR10):
        """Initialize with CIFAR-10 dataset and load associated noisy labels.
        
        Args:
            dataset: The CIFAR-10 dataset to add noise to
        """
        self.dataset = dataset
        self.orginal_labels = np.array(self.dataset.targets, copy=True)
        self.noisy_labels = self.load_label(noise_path='data/cifar10n/data.pt', train_labels=dataset.targets, noise_type='aggre_label')
        self.noisy_indices = []
        for i in range(len(self.dataset.targets)):
            if self.dataset.targets[i] != self.noisy_labels[i]:
                self.noisy_indices.append(i)
    
    def add_noise(self):
        """Apply the noisy labels to the dataset."""
        self.dataset.targets = np.array(self.noisy_labels, copy=True)
        
    def get_noisy_indices(self):
        """Return indices of samples with noisy labels."""
        return self.noisy_indices
    
    def calculate_noised_label_percentage(self, indices):
        """Calculate percentage of detected noisy labels within given indices.
        
        Args:
            indices: Indices of samples predicted to have noisy labels
            
        Returns:
            Percentage of correctly identified noisy labels
        """
        intersection = set(indices) & set(self.noisy_indices)
        percentage = (len(intersection) / len(indices)) * 100
        print(f'{percentage}% accuracy in {len(indices)} data')
        return percentage
    
    def report(self, indices):
        """Generate classification report and confusion matrix for noise detection.
        
        Args:
            indices: Indices of samples predicted to have noisy labels
        """
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
    
    def calculate_metrics(self, indices):
        """Calculate accuracy, precision, recall and F1 score for noise detection.
        
        Args:
            indices: Indices of samples predicted to have noisy labels
            
        Returns:
            Dictionary containing accuracy, precision, recall and F1 metrics
        """
        predicted_labels = np.zeros(len(self.dataset))
        predicted_labels[indices] = 1
        real_labels = np.zeros(len(self.dataset))
        real_labels[self.noisy_indices] = 1
        accuracy = np.mean(predicted_labels == real_labels)
        tp = np.sum((predicted_labels == 1) & (real_labels == 1))
        fp = np.sum((predicted_labels == 1) & (real_labels == 0))
        fn = np.sum((predicted_labels == 0) & (real_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def ravel(self, indices):
        """Convert confusion matrix to a flattened array for analysis.
        
        Args:
            indices: Indices of samples predicted to have noisy labels
            
        Returns:
            Flattened confusion matrix (TP, FP, FN, TN)
        """
        predicted_labels = np.zeros(len(self.dataset))
        predicted_labels[indices] = 1
        real_labels = np.zeros(len(self.dataset))
        real_labels[self.noisy_indices] = 1
        cm = confusion_matrix(real_labels, predicted_labels)
        return cm.ravel()
    
    def load_label(self, noise_path, train_labels, noise_type):
        """Load CIFAR-10N noisy labels from file.
        
        Args:
            noise_path: Path to the noisy labels file
            train_labels: Original CIFAR-10 labels to verify against
            noise_type: Type of noise to use ('aggre_label', 'random_label1', etc.)
            
        Returns:
            Array of noisy labels
            
        Raises:
            Exception: If input format is invalid
        """
        noise_label = torch.load(noise_path, weights_only=False)
        if isinstance(noise_label, dict):
            if "clean_label" in noise_label.keys():
                clean_label = torch.tensor(noise_label['clean_label'])
                assert torch.sum(torch.tensor(train_labels) - clean_label) == 0 
                print(f'The overall noise rate is {1-np.mean(clean_label.numpy() == noise_label[noise_type])}')
            return noise_label[noise_type].reshape(-1)  
        else:
            raise Exception('Input Error')