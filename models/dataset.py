import torch
from torchvision import transforms
from torch.utils.data import Dataset
import random
import os
from PIL import Image
import math
import pickle

class DatasetPairs(Dataset):
    """Dataset class for creating pairs of samples for Siamese network training.
    
    Creates balanced positive pairs (same class) and negative pairs (different classes).
    """
    def __init__(self, dataset, num_pairs_per_epoch=100000, smart_count=True, transform=None):
        """Initialize dataset pairs with source dataset and pair generation parameters.
        
        Args:
            dataset: Source dataset to create pairs from
            num_pairs_per_epoch: Number of pairs to generate per epoch
            smart_count: If True, calculate pairs based on dataset size using math.e formula
            transform: Transformations to apply to images
        """
        self.dataset = dataset
        self.transform = transform if transform is not None else transforms.Compose([transforms.ToTensor()])
        self.length = len(dataset)
        if smart_count:
            self.num_pairs_per_epoch = math.floor((math.e - 2) * len(dataset))
        else:
            self.num_pairs_per_epoch = num_pairs_per_epoch
            
        # Generate pairs with a 1:1 ratio of positive to negative
        self.pairs_indices = self.faster_generate_pairs_indices()
        
    def generate_pairs_indices(self):
        """Generate balanced pairs of indices with same/different labels (slow implementation)."""
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
    
    
    def faster_generate_pairs_indices(self):
        """Generate balanced pairs of indices more efficiently using label-to-indices mapping."""
        pairs_indices = []
        
        # Precompute mapping from label to list of indices.
        label_to_indices = {}
        for i, (_, label) in enumerate(self.dataset):
            label_to_indices.setdefault(label, []).append(i)
        labels = list(label_to_indices.keys())

        # Ensure that at least one label has two samples for positive pairs.
        valid_labels = [label for label in labels if len(label_to_indices[label]) > 1]
        if not valid_labels:
            raise ValueError("No label has at least two instances to form positive pairs.")
        if len(labels) < 2:
            raise ValueError("Not enough distinct labels to form negative pairs.")

        # Generate positive pairs
        num_positive_pairs = self.num_pairs_per_epoch // 2
        positive_pairs = []
        for _ in range(num_positive_pairs):
            label = random.choice(valid_labels)
            indices = label_to_indices[label]
            # Randomly select two distinct indices for the chosen label.
            i, j = random.sample(indices, 2)
            positive_pairs.append((i, j))

        # Generate negative pairs
        num_negative_pairs = self.num_pairs_per_epoch // 2
        negative_pairs = []
        for _ in range(num_negative_pairs):
            # Randomly select two different labels.
            label1, label2 = random.sample(labels, 2)
            i = random.choice(label_to_indices[label1])
            j = random.choice(label_to_indices[label2])
            negative_pairs.append((i, j))
        
        # Combine and shuffle the pairs.
        pairs_indices = positive_pairs + negative_pairs
        random.shuffle(pairs_indices)
        return pairs_indices

    def __len__(self):
        """Return the number of pairs in the dataset."""
        return len(self.pairs_indices)
    
    def __getitem__(self, idx):
        """Get a pair of images and their labels at the given index."""
        i, j = self.pairs_indices[idx]
        img1, label1 = self.dataset[i]
        img2, label2 = self.dataset[j]

        # Apply transformations if specified
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label1), torch.tensor(label2), i, j

    

class PositiveSamplingDatasetPairs(Dataset):
    """Dataset for creating pairs with special focus on positive pairs using augmentation.
    
    Creates pairs where half are from the same image (with different augmentations) 
    and half are from different classes.
    """
    def __init__(self, dataset, num_pairs_per_epoch=100000, smart_count=True, transform=None, augmentation=None):
        """Initialize dataset with augmentation capabilities for contrastive learning.
        
        Args:
            dataset: Source dataset
            num_pairs_per_epoch: Number of pairs to generate
            smart_count: Whether to calculate pairs based on dataset size
            transform: Basic transformations
            augmentation: Data augmentations for contrastive learning
        """
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
        """Generate pair indices where half use the same image with different augmentations."""
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
        """Return the number of pairs in the dataset."""
        return self.num_pairs_per_epoch

    def __getitem__(self, idx):
        """Get a pair of images with augmentations applied."""
        i, j = self.pairs_indices[idx]
        
        img1, label1 = self.dataset[i]
        img2, label2 = self.dataset[j]
        
        img1 = self.augmentation(img1)
        img2 = self.augmentation(img2)
        
        return img1, img2, torch.tensor(label1), torch.tensor(label2), i, j


class DatasetSingle(Dataset):
    """Simple dataset wrapper that returns single samples with their indices."""
    def __init__(self, data, transform):
        """Initialize with dataset and transforms."""
        self.data = data
        self.transform = transform

    def __len__(self):
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single sample, its label and index."""
        sample, label = self.data[idx]
        sample = self.transform(sample)
        return sample, label, idx

class CustomDataset(Dataset):
    """Basic dataset wrapper that applies transformations to samples."""
    def __init__(self, data, transform):
        """Initialize with dataset and transforms."""
        self.data = data
        self.transform = transform

    def __len__(self):
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get a transformed sample and its label."""
        sample, label = self.data[idx]
        sample = self.transform(sample)
        return sample, label

class CleanDatasetLoader(Dataset):
    """Dataset for loading cleaned data from pickle files."""
    def __init__(self, pkl_file, transform=None):
        """Initialize dataset loader with pickle file path and transforms."""
        self.pkl_file = pkl_file
        self.transform = transform
        self.images = []
        self.labels = []
        self.load_data()

    def load_data(self):
        """Load data from pickle file into memory."""
        with open(self.pkl_file, "rb") as f:
            while True:
                try:
                    entry = pickle.load(f)
                    self.images.append(entry['data'])
                    self.labels.append(entry['label'])
                except EOFError:
                    break
        print(f"Loaded {len(self.images)} samples from {self.pkl_file}")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Get a sample at the given index with transformations applied."""
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class CleanWrapperDataset(Dataset):
    """Wrapper dataset that selects specific indices from a base dataset."""
    def __init__(self, dataset: CleanDatasetLoader, indices, transform=None):
        """Initialize with base dataset and indices to select."""
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        """Return the number of selected indices."""
        return len(self.indices)

    def __getitem__(self, idx):
        """Get a sample from the base dataset at the specified index."""
        image, label = self.dataset[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

class Animal10NDataset(Dataset):
    """Dataset for Animal-10N dataset with file-based loading."""
    def __init__(self, root_dir, transform=None):
        """Initialize with root directory and transforms."""
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.targets = []
        
        # Class names in Animal-10N dataset
        self.classes = ['Cat', 'Lynx', 'Wolf', 'Coyote', 'Cheetah', 'Jaguar', 'Chimpanzee', 'Orangutan', 'Hamster', 'Guinea Pig']
        
        # Map class names to subdirectory names
        class_to_dir = {
            'Cat': 'cat', 'Lynx': 'lynx', 'Wolf': 'wolf', 'Coyote': 'coyote', 
            'Cheetah': 'cheetah', 'Jaguar': 'jaguar', 'Chimpanzee': 'chimp', 
            'Orangutan': 'gorilla', 'Hamster': 'hamster', 'Guinea Pig': 'guinea_pig'
        }
        
        # Load image paths and labels
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_to_dir[class_name])
            if os.path.exists(class_dir):
                image_names = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                for image_name in image_names:
                    self.image_paths.append(os.path.join(class_dir, image_name))
                    self.labels.append(label)
                    self.targets.append(label)

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get an image and its label at the given index."""
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label