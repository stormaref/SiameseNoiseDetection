import torch
from torchvision import transforms
from torch.utils.data import Dataset
import random

class DatasetPairs(Dataset):
    def __init__(self, dataset, num_pairs_per_epoch=100000, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])):
        self.dataset = dataset
        self.transform = transform
        self.length = len(dataset)
        self.num_pairs_per_epoch = num_pairs_per_epoch
        self.pairs_indices = self.generate_pairs_indices()

    def generate_pairs_indices(self):
        pairs_indices = []
        for _ in range(self.num_pairs_per_epoch):
            i, j = random.sample(range(self.length), 2)
            pairs_indices.append((i, j))
        return pairs_indices

    def __len__(self):
        return self.num_pairs_per_epoch

    def __getitem__(self, idx):
        i, j = self.pairs_indices[idx]
        img1, label1 = self.dataset[i]
        img2, label2 = self.dataset[j]
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2, torch.tensor(label1), torch.tensor(label2), i, j