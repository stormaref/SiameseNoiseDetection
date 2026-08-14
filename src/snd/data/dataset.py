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
        self.transform = transform if transform is not None else transforms.Compose([
                                                                                    transforms.ToTensor()])
        self.length = len(dataset)
        if smart_count:
            self.num_pairs_per_epoch = math.floor((math.e - 2) * len(dataset))
        else:
            self.num_pairs_per_epoch = num_pairs_per_epoch

        # Generate pairs with a 1:1 ratio of positive to negative
        self.pairs_indices = self.faster_generate_pairs_indices()

    def faster_generate_pairs_indices(self):
        """Generate balanced pairs of indices more efficiently using label-to-indices mapping."""
        pairs_indices = []

        # Precompute mapping from label to list of indices.
        label_to_indices = {}
        for i, (_, label) in enumerate(self.dataset):
            label_to_indices.setdefault(label, []).append(i)
        labels = list(label_to_indices.keys())

        # Ensure that at least one label has two samples for positive pairs.
        valid_labels = [label for label in labels if len(
            label_to_indices[label]) > 1]
        if not valid_labels:
            raise ValueError(
                "No label has at least two instances to form positive pairs.")
        if len(labels) < 2:
            raise ValueError(
                "Not enough distinct labels to form negative pairs.")

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
        self.images = []
        self.targets = []
        self.load_data()

    def load_data(self):
        """Load all image files in root_dir, extract label from filename, and store images and labels."""
        self.images = []
        self.targets = []
        for fname in os.listdir(self.root_dir):
            if fname.endswith((".jpg")):
                label = fname.split('_')[0]
                self.targets.append(int(label))
                img_path = os.path.join(self.root_dir, fname)
                image = Image.open(img_path).convert('RGB')
                self.images.append(image)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.targets)

    def __getitem__(self, idx):
        """Get a sample at the given index with transformations applied."""
        image = self.images[idx]
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
