import random
import numpy as np
import torch

def set_global_seed(seed):
    """Set random seed for all random number generators to ensure reproducibility.
    
    Args:
        seed: Integer seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
# Class names for CIFAR-10 dataset (in order corresponding to label indices)
CIFAR10_CLASSES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Class names for Fashion-MNIST dataset (in order corresponding to label indices)
FashionMNIST_CLASSES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']