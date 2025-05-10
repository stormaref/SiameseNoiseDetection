from models.utils import set_global_seed, CIFAR10_CLASSES, FashionMNIST_CLASSES
from models.cleaner import NoiseCleaner
import argparse
from models.config import *
import os
import torch
from torch.utils.data import Dataset

# Set global seed for reproducibility
set_global_seed(42)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for dataset and noise configuration.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Train and evaluate noise detection model')
    parser.add_argument('--dataset', 
                       type=str, 
                       required=True, 
                       choices=['cifar10', 'fashionmnist', 'cifar10n'],
                       help='Dataset to use: cifar10, fashionmnist, or cifar10n')
    parser.add_argument('--noise_ratio', 
                       type=str, 
                       required=True,
                       help='Noise ratio: 20, 30, 40 (or n for cifar10n)')
    parser.add_argument('--output_dir', 
                       type=str, 
                       default='./results',
                       help='Directory to save results')
    parser.add_argument('--mistakes_count',
                        type=int,
                        default=8,
                        help='Number of mistakes to count as noise')
    parser.add_argument('--relabel_threshold',
                        type=int,
                        default=9,
                        help='Relabel threshold')
    
    return parser.parse_args()

def get_dataset_config(args: argparse.Namespace) -> tuple:
    """Get dataset configuration based on command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (train_dataset, test_dataset, train_transform, test_transform, classes, params)
    """
    if args.dataset == 'cifar10':
        train_dataset = CIFAR10_TRAIN_DATASET
        test_dataset = CIFAR10_TEST_DATASET
        train_transform = CIFAR10_TRAIN_TRANSFORMS
        test_transform = CIFAR10_TEST_TRANSFORMS
        classes = CIFAR10_CLASSES
        
        if args.noise_ratio == '20':
            params = CIFAR10_20_PARAMS
        elif args.noise_ratio == '30':
            params = CIFAR10_30_PARAMS
        elif args.noise_ratio == '40':
            params = CIFAR10_40_PARAMS
        else:
            raise ValueError(f"Invalid noise ratio for CIFAR-10: {args.noise_ratio}. Choose from 20, 30, 40.")
    
    elif args.dataset == 'cifar10n':
        if args.noise_ratio != 'n':
            print(f"Warning: For CIFAR-10N, noise ratio should be 'n'. Ignoring provided value: {args.noise_ratio}")
        
        train_dataset = CIFAR10_TRAIN_DATASET
        test_dataset = CIFAR10_TEST_DATASET
        train_transform = CIFAR10_TRAIN_TRANSFORMS
        test_transform = CIFAR10_TEST_TRANSFORMS
        classes = CIFAR10_CLASSES
        params = CIFAR10N_PARAMS
    
    elif args.dataset == 'fashionmnist':
        train_dataset = FashionMNIST_TRAIN_DATASET
        test_dataset = FashionMNIST_TEST_DATASET
        train_transform = FashionMNIST_TRAIN_TRANSFORMS
        test_transform = FashionMNIST_TEST_TRANSFORMS
        classes = FashionMNIST_CLASSES
        
        if args.noise_ratio == '20':
            params = FashionMNIST_20_PARAMS
        elif args.noise_ratio == '30':
            params = FashionMNIST_30_PARAMS
        elif args.noise_ratio == '40':
            params = FashionMNIST_40_PARAMS
        else:
            raise ValueError(f"Invalid noise ratio for Fashion-MNIST: {args.noise_ratio}. Choose from 20, 30, 40.")
    
    return train_dataset, test_dataset, train_transform, test_transform, classes, params

def get_raw_dataset(args: argparse.Namespace) -> torch.utils.data.Dataset:
    """Get the raw dataset based on command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        torch.utils.data.Dataset: Raw training dataset
    """
    if args.dataset == 'cifar10':
        return CIFAR10_TRAIN_DATASET
    elif args.dataset == 'fashionmnist':
        return FashionMNIST_TRAIN_DATASET
    elif args.dataset == 'cifar10n':
        return CIFAR10_TRAIN_DATASET

def main() -> None:
    """Main function to run the noise detection and cleaning pipeline."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get dataset configuration
    train_dataset, test_dataset, train_transform, test_transform, classes, params = get_dataset_config(args)
    mistakes_count = args.mistakes_count
    relabel_threshold = args.relabel_threshold
    # Initialize noise cleaner
    noise_cleaner = NoiseCleaner(
        dataset=train_dataset,
        transform=train_transform,
        augmented_transform=train_transform,
        **params
    )
    
    # Run noise cleaning pipeline
    print(f"Starting training for {args.dataset} with noise ratio {args.noise_ratio}")
    noise_cleaner.clean()
    
    # Save cleaned dataset
    dataset = get_raw_dataset(args)
    manual_cleaned = noise_cleaner.advanced_clean(dataset=dataset, mistakes_count=mistakes_count, relabel_threshold=relabel_threshold)
    noise_cleaner.save_cleaned_cifar_dataset_manual(
        manual_cleaned, 
        args.output_dir, 
        f'cleaned_{args.dataset}_{args.noise_ratio}_{mistakes_count}_{relabel_threshold}.pth'
    )

if __name__ == "__main__":
    main()