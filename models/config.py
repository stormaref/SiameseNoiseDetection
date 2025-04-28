from models.utils import set_global_seed, CIFAR10_CLASSES, FashionMNIST_CLASSES
set_global_seed(42)
from torchvision import transforms
from torchvision.datasets import FashionMNIST, CIFAR10, CIFAR100
from models.cleaner import NoiseCleaner
from models.final_model_tester import FinalEvaluator

# CIFAR-10 default training transforms
CIFAR10_TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomRotation(degrees=15),        # Random rotation within 15 degrees
    transforms.RandomHorizontalFlip(p=0.5),      # Random horizontal flip
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Width and height shift
    transforms.RandomResizedCrop(size=32, scale=(0.9, 1.0)),  # Zoom-like effect
    transforms.ToTensor(),                        # Convert images to PyTorch tensors
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # Normalize
])

CIFAR10_TEST_TRANSFORMS = transforms.Compose([
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# CIFAR-10 default training dataset
CIFAR10_TRAIN_DATASET = CIFAR10(root='data', train=True, download=True)

CIFAR10_TEST_DATASET = CIFAR10(root='data', train=False, download=True)

CIFAR10_20_PARAMS = {
    'noise_type': 'idn',
    'model_save_path': "cifar10/resnet50/model_resnet50_cifar10_fold_{}.pth",
    'inner_folds_num': 10,
    'outer_folds_num': 10,
    'model': 'resnet50',
    'train_noise_level': 0.2,
    'epochs_num': 1000,
    'train_pairs': 200000,
    'val_pairs': 20000,
    'embedding_dimension': 64,
    'lr': 5e-5,
    'optimizer': 'Adam',
    'patience': 8,
    'weight_decay': 5e-4,
    'training_batch_size': 2048,
    'pre_trained': True,
    'dropout_prob': 0.5,
    'contrastive_ratio': 1,
    'distance_meter': 'euclidian',
    'trainable': True,
    'pair_validation': False,
    'label_smoothing': 0.1,
    'loss': 'ce',
    'margin': 2,
    'freeze_epoch': None,
    'noisy_indices_path': 'cifar10/resnet50/fold{}_noisy_indices.csv',
    'prediction_path': 'cifar10/resnet50/fold{}_analysis.csv',
    'mistakes_count': 10,
    'relabeling_range': range(6, 11)
}

CIFAR10_30_PARAMS = {
    'noise_type': 'idn',
    'model_save_path': "cifar10(30)/resnet50/model_resnet50_cifar10_fold_{}.pth",
    'inner_folds_num': 10,
    'outer_folds_num': 10,
    'model': 'resnet50',
    'train_noise_level': 0.3,
    'epochs_num': 1000,
    'train_pairs': 200000,
    'val_pairs': 20000,
    'embedding_dimension': 64,
    'lr': 5e-5,
    'optimizer': 'Adam',
    'patience': 10,
    'weight_decay': 5e-4,
    'training_batch_size': 2048,
    'pre_trained': True,
    'dropout_prob': 0.5,
    'contrastive_ratio': 1,
    'distance_meter': 'euclidian',
    'trainable': True,
    'pair_validation': False,
    'label_smoothing': 0.1,
    'loss': 'ce',
    'margin': 2,
    'freeze_epoch': None,
    'noisy_indices_path': 'cifar10(30)/resnet50/fold{}_noisy_indices.csv',
    'prediction_path': 'cifar10(30)/resnet50/fold{}_analysis.csv',
    'mistakes_count': 10,
    'relabeling_range': range(6, 11)
}

CIFAR10_40_PARAMS = {
    'noise_type': 'idn',
    'model_save_path': "cifar10(40)/resnet50/model_resnet50_cifar10_fold_{}.pth",
    'inner_folds_num': 10,
    'outer_folds_num': 10,
    'model': 'resnet50',
    'train_noise_level': 0.4,
    'epochs_num': 1000,
    'train_pairs': 200000,
    'val_pairs': 20000,
    'embedding_dimension': 64,
    'lr': 5e-5,
    'optimizer': 'Adam',
    'patience': 15,
    'weight_decay': 5e-4,
    'training_batch_size': 2048,
    'pre_trained': True,
    'dropout_prob': 0.5,
    'contrastive_ratio': 1,
    'distance_meter': 'euclidian',
    'trainable': True,
    'pair_validation': False,
    'label_smoothing': 0.1,
    'loss': 'ce',
    'margin': 2,
    'freeze_epoch': None,
    'noisy_indices_path': 'cifar10(40)/resnet50/fold{}_noisy_indices.csv',
    'prediction_path': 'cifar10(40)/resnet50/fold{}_analysis.csv',
    'mistakes_count': 10,
    'relabeling_range': range(6, 11)
}

CIFAR10N_PARAMS = {
    'noise_type': 'cifar10n',
    'model_save_path': "cifar10n/resnet50/model_resnet50_cifar10_fold_{}.pth",
    'inner_folds_num': 10,
    'outer_folds_num': 10,
    'model': 'resnet50',
    'train_noise_level': 0.2,
    'epochs_num': 1000,
    'train_pairs': 200000,
    'val_pairs': 20000,
    'embedding_dimension': 64,
    'lr': 5e-5,
    'optimizer': 'Adam',
    'patience': 8,
    'weight_decay': 5e-4,
    'training_batch_size': 2048,
    'pre_trained': True,
    'dropout_prob': 0.5,
    'contrastive_ratio': 1,
    'distance_meter': 'euclidian',
    'trainable': True,
    'pair_validation': False,
    'label_smoothing': 0.1,
    'loss': 'ce',
    'margin': 2,
    'freeze_epoch': None,
    'noisy_indices_path': 'cifar10n/resnet50/fold{}_noisy_indices.csv',
    'prediction_path': 'cifar10n/resnet50/fold{}_analysis.csv',
    'mistakes_count': 10,
    'relabeling_range': range(6, 11)
}

FashionMNIST_TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

FashionMNIST_TEST_TRANSFORMS = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

FashionMNIST_TRAIN_DATASET = FashionMNIST(root='data', train=True, download=True)

FashionMNIST_TEST_DATASET = FashionMNIST(root='data', train=False, download=True)

FashionMNIST_20_PARAMS = {
    'noise_type': 'idn',
    'model_save_path': "fmnist(20)/resnet34/model_resnet34_fmnist(20)_fold_{}.pth",
    'inner_folds_num': 10,
    'outer_folds_num': 10,
    'model': 'resnet34',
    'train_noise_level': 0.2,
    'epochs_num': 1000,
    'train_pairs': 200000,
    'val_pairs': 20000,
    'embedding_dimension': 128,
    'lr': 5e-5,
    'optimizer': 'Adam',
    'patience': 12,
    'weight_decay': 1e-3,
    'training_batch_size': 2048,
    'pre_trained': False,
    'dropout_prob': 0.5,
    'contrastive_ratio': 1,
    'distance_meter': 'euclidian',
    'trainable': True,
    'pair_validation': False,
    'label_smoothing': 0.1,
    'loss': 'ce',
    'margin': 2,
    'freeze_epoch': None,
    'noisy_indices_path': 'fmnist(20)/resnet34/fold{}_noisy_indices.csv',
    'prediction_path': 'fmnist(20)/resnet34/fold{}_analysis.csv',
    'mistakes_count': 10,
    'relabeling_range': range(6, 11)
}

FashionMNIST_30_PARAMS = {
    'noise_type': 'idn',
    'model_save_path': "fmnist(30)/resnet34/model_resnet34_fmnist(30)_fold_{}.pth",
    'inner_folds_num': 10,
    'outer_folds_num': 10,
    'model': 'resnet34',
    'train_noise_level': 0.3,
    'epochs_num': 1000,
    'train_pairs': 200000,
    'val_pairs': 20000,
    'embedding_dimension': 128,
    'lr': 5e-5,
    'optimizer': 'Adam',
    'patience': 12,
    'weight_decay': 1e-3,
    'training_batch_size': 2048,
    'pre_trained': False,
    'dropout_prob': 0.5,
    'contrastive_ratio': 1,
    'distance_meter': 'euclidian',
    'trainable': True,
    'pair_validation': False,
    'label_smoothing': 0.1,
    'loss': 'ce',
    'margin': 2,
    'freeze_epoch': None,
    'noisy_indices_path': 'fmnist(30)/resnet34/fold{}_noisy_indices.csv',
    'prediction_path': 'fmnist(30)/resnet34/fold{}_analysis.csv',
    'mistakes_count': 10,
    'relabeling_range': range(6, 11)
}

FashionMNIST_40_PARAMS = {
    'noise_type': 'idn',
    'model_save_path': "fmnist(40)/resnet34/model_resnet34_fmnist(40)_fold_{}.pth",
    'inner_folds_num': 10,
    'outer_folds_num': 10,
    'model': 'resnet34',
    'train_noise_level': 0.4,
    'epochs_num': 1000,
    'train_pairs': 200000,
    'val_pairs': 20000,
    'embedding_dimension': 128,
    'lr': 5e-5,
    'optimizer': 'Adam',
    'patience': 12,
    'weight_decay': 1e-3,
    'training_batch_size': 2048,
    'pre_trained': False,
    'dropout_prob': 0.5,
    'contrastive_ratio': 1,
    'distance_meter': 'euclidian',
    'trainable': True,
    'pair_validation': False,
    'label_smoothing': 0.1,
    'loss': 'ce',
    'margin': 2,
    'freeze_epoch': None,
    'noisy_indices_path': 'fmnist(40)/resnet34/fold{}_noisy_indices.csv',
    'prediction_path': 'fmnist(40)/resnet34/fold{}_analysis.csv',
    'mistakes_count': 10,
    'relabeling_range': range(6, 11)
}