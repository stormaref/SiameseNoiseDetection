import numpy as np
import torch
from math import inf
from scipy import stats
import torch.nn.functional as F
from torchvision import transforms
import PIL
import matplotlib.pyplot as plt
import seaborn as sns

class InstanceDependentNoiseAdder:
    def __init__(self, dataset, image_size, ratio, num_classes=10):
        self.dataset = dataset
        self.num_classes = num_classes
        self.feature_size = image_size
        self.noise_ratio = ratio
        self.noisy_indices = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def add_noise(self, norm_std=0.1, seed=21):
        seed = np.random.randint(0, 100)
        print(f'Seed: {seed}')
        noisy_labels = self.get_noisy_labels(norm_std=norm_std, seed=seed)
        self.noisy_indices = np.where(np.array(self.dataset.targets) != np.array(noisy_labels))[0]
        self.dataset.targets = noisy_labels
        
    def get_noisy_indices(self):
        return self.noisy_indices
    
    def calculate_noised_label_percentage(self, indices):
        intersection = set(indices) & set(self.noisy_indices)
        percentage = (len(intersection) / len(indices)) * 100
        print(f'{percentage}% accuracy in {len(indices)} data')
        return percentage
    
    def calculate_precision_recall(self, indices):
        intersection = set(indices) & set(self.noisy_indices)
        precision = len(intersection) / len(indices)
        recall = len(intersection) / len(self.noisy_indices)
        return precision, recall
    
    def plot_confusion_matrix(self, indices):
        true_positive = set(indices) & set(self.noisy_indices)
        false_positive = set(indices) - set(self.noisy_indices)
        false_negative = set(self.noisy_indices) - set(indices)
        true_negative = len(self.dataset) - len(self.noisy_indices) - len(false_negative)
        plt.figure(figsize=(10, 10))
        confusion_matrix = np.array([[len(true_positive), len(false_positive)], [len(false_negative), true_negative]])
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    
    def get_noisy_labels(self, norm_std, seed): 
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(seed))

        P = []
        ratio = self.noise_ratio
        labels = self.dataset.targets
        
        if isinstance(labels, list):
            labels = torch.FloatTensor(labels)
        labels = labels.to(self.device)
        
        flip_distribution = stats.truncnorm((0 - ratio) / norm_std, (1 - ratio) / norm_std, loc=ratio, scale=norm_std)
        flip_rate = flip_distribution.rvs(labels.shape[0])

        label_num = self.num_classes
        W = np.random.randn(label_num, self.feature_size, label_num)
        W = torch.FloatTensor(W).to(self.device)
        
        for i, (x, y) in enumerate(self.dataset):
            if isinstance(x, PIL.Image.Image):
                x = transforms.ToTensor()(x)
            x = x.to(self.device)
            A = x.contiguous().view(1, -1).mm(W[int(y)]).squeeze(0)
            A[int(y)] = -inf
            A = flip_rate[i] * F.softmax(A, dim=0)
            A[int(y)] += 1 - flip_rate[i]
            P.append(A)
        
        P = torch.stack(P, 0).cpu().numpy()
        l = [i for i in range(label_num)]
        new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
        return np.array(new_label)