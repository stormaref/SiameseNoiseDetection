import torch
from torchvision import transforms
from torch.utils.data import Dataset
import random
import os
from PIL import Image
import math

class DatasetPairs(Dataset):
    def __init__(self, dataset, num_pairs_per_epoch=100000, smart_count=True, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])):
        self.dataset = dataset
        self.transform = transform
        if transform == None:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.length = len(dataset)
        if smart_count:
            self.num_pairs_per_epoch = math.floor((math.e - 2) * len(dataset))
        else:
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

class DatasetSingle(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx]
        sample = self.transform(sample)
        return sample, label, idx

class CustomDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx]
        sample = self.transform(sample)
        return sample, label
    
class Animal10NDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.targets = []

        for image_name in os.listdir(root_dir):
            if os.path.isfile(os.path.join(root_dir, image_name)):
                self.image_paths.append(os.path.join(root_dir, image_name))
                label = int(image_name.split('_')[0])  # Extract label from filename prefix
                self.targets.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, label