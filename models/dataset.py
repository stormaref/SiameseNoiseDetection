import torch
from torchvision import transforms
from torch.utils.data import Dataset
import random
import os
from PIL import Image
import math
import pickle

class DatasetPairs(Dataset):
    def __init__(self, dataset, num_pairs_per_epoch=100000, smart_count=True, transform=None):
        self.dataset = dataset
        self.transform = transform if transform is not None else transforms.Compose([transforms.ToTensor()])
        
        self.length = len(dataset)
        if smart_count:
            self.num_pairs_per_epoch = math.floor((math.e - 2) * len(dataset))
        else:
            self.num_pairs_per_epoch = num_pairs_per_epoch
            
        # Generate pairs with a 1:1 ratio of positive to negative
        self.pairs_indices = self.generate_pairs_indices()

    def generate_pairs_indices(self):
        pairs_indices = []

        # Generate positive pairs
        num_positive_pairs = self.num_pairs_per_epoch // 2
        for _ in range(num_positive_pairs):
            while True:
                i = random.randint(0, self.length - 1)
                j = random.randint(0, self.length - 1)
                if self.dataset[i][1] == self.dataset[j][1] and i != j:  # Ensure different indices with the same label
                    pairs_indices.append((i, j))  # Positive pair
                    break

        # Generate negative pairs
        num_negative_pairs = self.num_pairs_per_epoch // 2
        for _ in range(num_negative_pairs):
            while True:
                i, j = random.sample(range(self.length), 2)
                if self.dataset[i][1] != self.dataset[j][1]:  # Ensure different labels
                    pairs_indices.append((i, j))  # Negative pair
                    break

        random.shuffle(pairs_indices)
        return pairs_indices

    def __len__(self):
        return self.num_pairs_per_epoch

    def __getitem__(self, idx):
        i, j = self.pairs_indices[idx]
        img1, label1 = self.dataset[i]
        img2, label2 = self.dataset[j]

        # Apply transformations if specified
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label1), torch.tensor(label2), i, j

    

class PositiveSamplingDatasetPairs(Dataset):
    def __init__(self, dataset, num_pairs_per_epoch=100000, smart_count=True, transform=None, augmentation=None):
        self.dataset = dataset
        self.transform = transform
        self.augmentation = augmentation
        
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        if augmentation is None:
            self.augmentation = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])
        
        self.length = len(dataset)
        if smart_count:
            self.num_pairs_per_epoch = math.floor((math.e - 2) * len(dataset))
        else:
            self.num_pairs_per_epoch = num_pairs_per_epoch

        self.pairs_indices = self.generate_pairs_indices()

    def generate_pairs_indices(self):
        pairs_indices = []
        for _ in range(self.num_pairs_per_epoch // 2):  # Half for positives
            i = random.randint(0, self.length - 1)
            pairs_indices.append((i, i))  # Positive pairs
            
        for _ in range(self.num_pairs_per_epoch // 2):  # Half for negatives
            while True:
                i, j = random.sample(range(self.length), 2)
                if self.dataset[i][1] != self.dataset[j][1]:  # Ensure different labels
                    pairs_indices.append((i, j))  # Negative pairs
                    break
                    
        return pairs_indices

    def __len__(self):
        return self.num_pairs_per_epoch

    def __getitem__(self, idx):
        i, j = self.pairs_indices[idx]
        
        img1, label1 = self.dataset[i]
        img2, label2 = self.dataset[j]
        
        img1 = self.augmentation(img1)
        img2 = self.augmentation(img2)
        
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
    

class CleanDatasetLoader(Dataset):
    def __init__(self, pkl_file, transform=None):
        self.pkl_file = pkl_file
        self.transform = transform
        self.index_map = self._build_index_map()

    def _build_index_map(self):
        index_map = []
        with open(self.pkl_file, "rb") as f:
            while True:
                try:
                    index_map.append(f.tell())
                    pickle.load(f)
                except EOFError:
                    break
        return index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        with open(self.pkl_file, "rb") as f:
            f.seek(self.index_map[idx])
            entry = pickle.load(f)
        
        img = entry['data']
        label = entry['label']
        
        if self.transform:
            img = self.transform(img)
        
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label
    
class Animal10NDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.targets = []
        self.images = []

        for image_name in os.listdir(root_dir):
            if os.path.isfile(os.path.join(root_dir, image_name)):
                self.image_paths.append(os.path.join(root_dir, image_name))
                label = int(image_name.split('_')[0])  # Extract label from filename prefix
                self.targets.append(label)
                
        for idx in range(len(self.image_paths)):
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
            label = self.targets[idx]

            if self.transform:
                image = self.transform(image)
            self.images.append(image)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # image_path = self.image_paths[idx]
        # image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
        image = self.images[idx]
        label = self.targets[idx]

        # if self.transform:
            # image = self.transform(image)

        return image, label