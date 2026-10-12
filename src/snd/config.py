"""Per-dataset / per-noise-level hyperparameters, transforms and dataset handles.

The four dataset handles (``CIFAR10_TRAIN_DATASET``, ``CIFAR10_TEST_DATASET``,
``FashionMNIST_TRAIN_DATASET``, ``FashionMNIST_TEST_DATASET``) are built lazily on
first attribute access, so importing this module no longer downloads ~200 MB into
``data/``. They are still plain module attributes -- ``from snd.config import
CIFAR10_TRAIN_DATASET`` and ``config.CIFAR10_TRAIN_DATASET`` behave exactly as before.
"""
from torchvision.datasets import FashionMNIST, CIFAR10
from torchvision import transforms
from snd.utils import set_global_seed, CIFAR10_CLASSES, FashionMNIST_CLASSES
set_global_seed(42)

# CIFAR-10 default training transforms
CIFAR10_TRAIN_TRANSFORMS = transforms.Compose([
    # Random rotation within 15 degrees
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),      # Random horizontal flip
    transforms.RandomAffine(degrees=0, translate=(
        0.1, 0.1)),  # Width and height shift
    transforms.RandomResizedCrop(
        size=32, scale=(0.9, 1.0)),  # Zoom-like effect
    transforms.ToTensor(),                        # Convert images to PyTorch tensors
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))  # Normalize
])

CIFAR10_TEST_TRANSFORMS = transforms.Compose([
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# CIFAR-10 train/test datasets -- instantiated on first access, see _LAZY_DATASETS below.

CIFAR10_20_PARAMS = {
    'noise_type': 'idn',
    'model_save_path': "cifar10(20)/resnet50/model_resnet50_cifar10_fold_{}.pth",
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
    'noisy_indices_path': 'preds/cifar10(20)/resnet50/fold{}_noisy_indices.csv',
    'prediction_path': 'preds/cifar10(20)/resnet50/fold{}_analysis.csv',
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
    'noisy_indices_path': 'preds/cifar10(30)/resnet50/fold{}_noisy_indices.csv',
    'prediction_path': 'preds/cifar10(30)/resnet50/fold{}_analysis.csv',
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
    'noisy_indices_path': 'preds/cifar10(40)/resnet50/fold{}_noisy_indices.csv',
    'prediction_path': 'preds/cifar10(40)/resnet50/fold{}_analysis.csv',
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
    'noisy_indices_path': 'preds/cifar10n/resnet50/fold{}_noisy_indices.csv',
    'prediction_path': 'preds/cifar10n/resnet50/fold{}_analysis.csv',
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

# Fashion-MNIST 20/30/60%: ImageNet-pretrained ResNet34 trained with crop/flip on unnormalized
# grayscale-to-RGB images; inner validation and the scoring of held-out samples use the plain
# transform. (40% keeps the from-scratch, normalized setting above.)
FashionMNIST_PLAIN_TRANSFORMS = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()])

FashionMNIST_AUG_TRANSFORMS = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomCrop(28, padding=2),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor()])

# Fashion-MNIST train/test datasets -- instantiated on first access, see _LAZY_DATASETS below.

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
    'noisy_indices_path': 'preds/fmnist(20)/resnet34/fold{}_noisy_indices.csv',
    'prediction_path': 'preds/fmnist(20)/resnet34/fold{}_analysis.csv',
    'mistakes_count': 10,
    'relabeling_range': range(6, 11),
    'transform': FashionMNIST_PLAIN_TRANSFORMS,
    'augmented_transform': FashionMNIST_AUG_TRANSFORMS,
    'ensemble_seed': 0
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
    'noisy_indices_path': 'preds/fmnist(30)/resnet34/fold{}_noisy_indices.csv',
    'prediction_path': 'preds/fmnist(30)/resnet34/fold{}_analysis.csv',
    'mistakes_count': 10,
    'relabeling_range': range(6, 11),
    'transform': FashionMNIST_PLAIN_TRANSFORMS,
    'augmented_transform': FashionMNIST_AUG_TRANSFORMS,
    'ensemble_seed': 0
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
    'noisy_indices_path': 'preds/fmnist(40)/resnet34/fold{}_noisy_indices.csv',
    'prediction_path': 'preds/fmnist(40)/resnet34/fold{}_analysis.csv',
    'mistakes_count': 10,
    'relabeling_range': range(6, 11)
}
# Fashion-MNIST 60% (extreme-noise experiment): 15 x 15 folds, 300k training pairs, outer folds
# split with a fixed seed
FashionMNIST_60_PARAMS = {
    'noise_type': 'idn',
    'model_save_path': "fmnist(60)/resnet34/model_resnet34_fmnist(60)_fold_{}.pth",
    'inner_folds_num': 15,
    'outer_folds_num': 15,
    'model': 'resnet34',
    'train_noise_level': 0.6,
    'epochs_num': 1000,
    'train_pairs': 300000,
    'val_pairs': 20000,
    'embedding_dimension': 128,
    'lr': 5e-5,
    'optimizer': 'Adam',
    'patience': 12,
    'weight_decay': 1e-3,
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
    'noisy_indices_path': 'preds/fmnist(60)/resnet34/fold{}_noisy_indices.csv',
    'prediction_path': 'preds/fmnist(60)/resnet34/fold{}_analysis.csv',
    'mistakes_count': 15,
    'relabeling_range': range(9, 16),
    'fold_seed': 51,
    'transform': FashionMNIST_PLAIN_TRANSFORMS,
    'augmented_transform': FashionMNIST_AUG_TRANSFORMS,
    'ensemble_seed': 0
}


# Oracle thresholds (T_D, T_R) reported in the paper, per (dataset, noise ratio)
PAPER_THRESHOLDS = {
    ('cifar10', '20'): (10, 10), ('cifar10', '30'): (8, 10), ('cifar10', '40'): (8, 10),
    ('cifar10n', 'n'): (10, 10),
    ('fashionmnist', '20'): (9, 10), ('fashionmnist', '30'): (8, 10),
    ('fashionmnist', '40'): (7, 10), ('fashionmnist', '60'): (9, 11),
}


# --- downstream classifier --------------------------------------------------------------
# PreAct-ResNet34 trained on a cleaned set (scripts/train_downstream.py): Adam, linear warm-up for
# `warmup_epochs` and then a constant learning rate, label smoothing, early stopping on validation
# accuracy with the best weights restored, accuracy on the clean test set.
FashionMNIST_DOWNSTREAM_TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

DOWNSTREAM_PARAMS = {
    'cifar10': dict(lr=1e-3, weight_decay=5.46e-5, train_batch_size=256, patience=20,
                    warmup_epochs=5, smoothing=0.1, val_ratio=0.1, epochs=200, test='cifar',
                    train_transform=CIFAR10_TRAIN_TRANSFORMS, test_transform=CIFAR10_TEST_TRANSFORMS),
    'fashionmnist': dict(lr=2e-4, weight_decay=1.12e-6, train_batch_size=256, patience=20,
                         warmup_epochs=5, smoothing=0.1, val_ratio=0.05, epochs=200, test='fmnist',
                         train_transform=FashionMNIST_DOWNSTREAM_TRAIN_TRANSFORMS,
                         test_transform=FashionMNIST_TEST_TRANSFORMS),
}
DOWNSTREAM_PARAMS['cifar10n'] = DOWNSTREAM_PARAMS['cifar10']    # same images, CIFAR-10 test set


# --- lazy dataset handles ---------------------------------------------------
# Building these at import time would download ~200 MB into data/ just to read a
# hyperparameter dict. PEP 562 module __getattr__ defers construction to first use
# and caches the result, so the module-attribute API is unchanged.
_LAZY_DATASETS = {
    'CIFAR10_TRAIN_DATASET': lambda: CIFAR10(root='data', train=True, download=True),
    'CIFAR10_TEST_DATASET': lambda: CIFAR10(root='data', train=False, download=True),
    'FashionMNIST_TRAIN_DATASET': lambda: FashionMNIST(root='data', train=True, download=True),
    'FashionMNIST_TEST_DATASET': lambda: FashionMNIST(root='data', train=False, download=True),
}


def __getattr__(name):
    """Instantiate a torchvision dataset on first access, then cache it."""
    if name in _LAZY_DATASETS:
        dataset = _LAZY_DATASETS[name]()
        globals()[name] = dataset
        return dataset
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(list(globals()) + list(_LAZY_DATASETS))
