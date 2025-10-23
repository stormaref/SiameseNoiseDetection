from typing import Any, Dict, List, Tuple

import csv
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import PIL
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from models.contrastive import ContrastiveLoss
from models.dataset import DatasetPairs, DatasetSingle
from models.fold import CustomKFoldSplitter
from models.noise import LabelNoiseAdder
from models.predefined import InstanceDependentNoiseAdder
from models.siamese import SiameseNetwork
from models.tester import Tester
from models.trainer import Trainer


def create_tta_transforms(normalize_mean: Tuple[float, ...], normalize_std: Tuple[float, ...], is_grayscale: bool) -> List[transforms.Compose]:
    """Create class-preserving augmentation transforms for TTA."""

    # Common tensor + normalization part
    to_tensor_and_norm = [
        *([transforms.Grayscale(num_output_channels=3)] if is_grayscale else []),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ]

    # Define base augmentations
    tta_transforms = [
        [],  # identity / no augmentation
        [transforms.RandomHorizontalFlip(p=1.0)],
        [transforms.RandomRotation(degrees=10)],
        [transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))],
        [transforms.ColorJitter(brightness=0.2)],
        [transforms.ColorJitter(contrast=0.2)],
        [transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomRotation(degrees=5),
         transforms.ColorJitter(brightness=0.1, contrast=0.1)]
    ]

    # Combine each augmentation with the common tail using spread (*)
    return [transforms.Compose([*aug, *to_tensor_and_norm]) for aug in tta_transforms]


def create_train_transform(normalize_mean: Tuple[float, ...], normalize_std: Tuple[float, ...]) -> transforms.Compose:
    """Create training transform with augmentations."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])


def create_val_transform(normalize_mean: Tuple[float, ...], normalize_std: Tuple[float, ...]) -> transforms.Compose:
    """Create validation transform without augmentations."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])


def convert_to_pil(image: torch.Tensor):
    """Convert tensor to PIL Image."""
    if isinstance(image, PIL.Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = (image * 255).clamp(0, 255).byte().numpy()
        image = Image.fromarray(image)
        return image
    raise ValueError(f"Unsupported image type: {type(image)}")


def generate_variants(image: torch.Tensor, label: int, transforms_list: List[transforms.Compose],
                      k_variants: int) -> List[Tuple[torch.Tensor, int]]:
    """Generate k augmented variants of a single datapoint."""
    image = convert_to_pil(image)
    variants = []

    for i in range(k_variants):
        transform_idx = i % len(transforms_list)
        transform = transforms_list[transform_idx]
        augmented = transform(image)
        variants.append((augmented, label))

    return variants


def predict_variants(model: SiameseNetwork, variants: List[Tuple[torch.Tensor, int]],
                     device: str) -> Tuple[List[int], List[float]]:
    """Get predictions for all variants."""
    predictions, confidences = [], []

    for i, (variant_img, _) in enumerate(variants):
        variant_img = variant_img.unsqueeze(0).to(device)
        emb, cls_output = model.classify(variant_img)
        probabilities = torch.softmax(cls_output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        predictions.append(prediction.item())
        confidences.append(confidence.item())

        # Debug: Print first few predictions to see what's happening
        if i < 3:  # Only print first 3 variants for debugging
            print(
                f"Variant {i}: prediction={prediction.item()}, confidence={confidence.item():.4f}")
            print(f"Raw logits: {cls_output.squeeze().detach().cpu().numpy()}")
            print(
                f"Probabilities: {probabilities.squeeze().detach().cpu().numpy()}")

    return predictions, confidences


def aggregate_predictions(predictions: List[int], confidences: List[float],
                          confidence_threshold: float) -> Tuple[int, float, bool]:
    """Aggregate predictions using majority voting and confidence."""
    final_prediction = max(set(predictions), key=predictions.count)
    avg_confidence = np.mean(confidences)
    prediction_consistency = predictions.count(
        final_prediction) / len(predictions)
    is_clean = avg_confidence > confidence_threshold and prediction_consistency > 0.6

    return final_prediction, avg_confidence, is_clean


class TTACleaner:
    """Test-Time Augmentation based cleaner using Siamese networks."""

    def __init__(self, num_epochs: int, num_pairs: int, lr: float, batch_size: int,
                 num_classes: int, num_workers: int, device: str,
                 normalize_mean: Tuple[float, ...] = (0.4914, 0.4822, 0.4465),
                 normalize_std: Tuple[float, ...] = (0.2023, 0.1994, 0.2010),
                 model_type: str = 'resnet18', embedding_dim: int = 128,
                 pre_trained: bool = True, dropout_prob: float = 0.5,
                 trainable: bool = True, cnn_size: int = None,
                 middle_size: int = None, parallel: bool = False,
                 contrastive_ratio: float = 1.0, k_variants: int = 5,
                 val_split_size: float = 0.2, val_split_shuffle: bool = True,
                 results_save_path: str = None, noise_type: str = 'none',
                 train_noise_level: float = 0.1, train_transform=None,
                 val_transform=None, is_grayscale=False, patience=10):

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.patience = patience

        # Training parameters
        self.num_epochs = num_epochs
        self.num_pairs = num_pairs
        self.lr = lr
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.device = device
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.contrastive_ratio = contrastive_ratio
        self.k_variants = k_variants

        # Validation split parameters
        self.val_split_size = val_split_size
        self.val_split_shuffle = val_split_shuffle

        # Results saving
        self.results_save_path = results_save_path

        # Noise parameters
        self.noise_type = noise_type
        self.train_noise_level = train_noise_level
        self.train_noise_adder = None

        # Siamese network parameters
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.pre_trained = pre_trained
        self.dropout_prob = dropout_prob
        self.trainable = trainable
        self.cnn_size = cnn_size
        self.middle_size = middle_size
        self.parallel = parallel

        self.model = None
        self.trainer = None
        self.tester = None
        self.splitter = None
        self.trained = False
        self.tta_transforms = create_tta_transforms(
            normalize_mean, normalize_std, is_grayscale)

    def get_image_size(self, dataset):
        """Get the flattened size of images in the dataset."""
        sample, _ = dataset[0]
        if isinstance(sample, PIL.Image.Image):
            sample = transforms.ToTensor()(sample)
        return sample.shape[0] * sample.shape[1] * sample.shape[2]

    def setup_noise_adder(self, dataset):
        """Set up train noise adder based on noise type."""
        if self.noise_type == 'idn':
            image_size = self.get_image_size(dataset)

            self.train_noise_adder = InstanceDependentNoiseAdder(
                dataset, image_size=image_size, ratio=self.train_noise_level, num_classes=self.num_classes)
            self.train_noise_adder.add_noise()
        elif self.noise_type == 'iin':
            self.train_noise_adder = LabelNoiseAdder(
                dataset, noise_level=self.train_noise_level, num_classes=self.num_classes)
            self.train_noise_adder.add_noise()
        elif self.noise_type == 'none':
            self.train_noise_adder = None

        if self.noise_type != 'none' and self.train_noise_adder:
            print(
                f'Noise count: {len(self.train_noise_adder.get_noisy_indices())} out of {len(dataset)} data')

    def create_model(self) -> SiameseNetwork:
        """Create and return a Siamese network model."""
        return SiameseNetwork(
            num_classes=self.num_classes,
            model=self.model_type,
            embedding_dimension=self.embedding_dim,
            pre_trained=self.pre_trained,
            dropout_prob=self.dropout_prob,
            trainable=self.trainable,
            cnn_size=self.cnn_size,
            middle_size=self.middle_size,
            parallel=self.parallel
        )

    def train_model(self, dataset: torch.utils.data.Dataset,
                    use_validation: bool = True) -> None:
        """Train the Siamese network using stratified train/validation split."""
        print(f"Training TTACleaner with {self.model_type} backbone...")

        # Setup noise adder first
        self.setup_noise_adder(dataset)

        self.model = self.create_model()

        if use_validation:
            # Use stratified split
            self.splitter = CustomKFoldSplitter(
                dataset_size=len(dataset),
                labels=dataset.targets,
                num_folds=10,
                shuffle=True
            )

            train_indices, val_indices = self.splitter.get_fold(0)
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)

        else:
            train_dataset = dataset
            val_dataset = None
            self.splitter = None

        train_transform = self.train_transform
        if self.train_transform == None:
            train_transform = create_train_transform(
                self.normalize_mean, self.normalize_std)

        # Create dataset pairs for training
        train_pairs = DatasetPairs(
            train_dataset,
            self.num_pairs,
            False,
            train_transform
        )
        train_loader = DataLoader(
            train_pairs,
            self.batch_size,
            True,
            num_workers=self.num_workers
        )

        val_transform = self.val_transform
        if self.val_transform == None:
            val_transform = create_val_transform(
                self.normalize_mean, self.normalize_std)

        val_loader = None
        if val_dataset is not None:
            val_pairs = DatasetPairs(
                val_dataset,
                min(self.num_pairs // 4, len(val_dataset) * 2),
                False,
                val_transform
            )
            val_loader = DataLoader(
                val_pairs,
                self.batch_size,
                False,
                num_workers=self.num_workers
            )

        self.trainer = Trainer(
            model=self.model,
            contrastive_criterion=ContrastiveLoss(margin=2.0),
            classifier_criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(self.model.parameters(), lr=self.lr),
            dataloader=train_loader,
            device=self.device,
            contrastive_ratio=self.contrastive_ratio,
            val_dataloader=val_loader,
            patience=self.patience,
            checkpoint_path='tta_cleaner_best.pth',
            freeze_epoch=None
        )

        self.trainer.train(self.num_epochs)
        self.trained = True
        print("Training completed!")

    def test_model(self, test_dataset: torch.utils.data.Dataset) -> Tuple[float, float, float, float]:
        """Test the model using the Tester class."""
        if not self.trained:
            raise ValueError("Model must be trained before testing!")

        val_transform = self.val_transform
        if val_transform == None:
            val_transform = create_val_transform(
                self.normalize_mean, self.normalize_std)

        test_dataset = DatasetSingle(test_dataset, transform=val_transform)
        # test_pairs = DatasetPairs(
        #     test_dataset,
        #     len(test_dataset) * 2,
        #     False,
        #     val_transform
        # )
        test_loader = DataLoader(
            test_dataset,
            self.batch_size,
            False,
            num_workers=self.num_workers
        )

        self.tester = Tester(self.model, test_loader, self.device)
        return self.tester.test_single()

    def get_datapoint_variants(self, image: torch.Tensor, label: int = None) -> List[Tuple[torch.Tensor, int]]:
        """Generate k augmented variants of a single datapoint."""
        variants = generate_variants(
            image, label, self.tta_transforms, self.k_variants)
        # Debug: Print info about variants
        print(f"Generated {len(variants)} variants for label {label}")
        # Show first 3 shapes
        print(f"Variant shapes: {[v[0].shape for v in variants[:3]]}")
        return variants

    def evaluate_with_tta(self, test_dataset: torch.utils.data.Dataset,
                          confidence_threshold: float = 0.8) -> Dict[str, Any]:
        """Evaluate samples using Test-Time Augmentation for robust predictions."""
        if not self.trained:
            raise ValueError("Model must be trained before evaluation!")

        self.model.eval()
        results = {
            'predictions': [], 'confidences': [], 'labels': [],
            'clean_indices': [], 'noisy_indices': [],
            'accuracy': 0.0, 'clean_accuracy': 0.0,
            'detailed_results': []  # For detailed saving
        }

        print("Evaluating with Test-Time Augmentation...")

        with torch.no_grad():
            for idx in tqdm(range(len(test_dataset))):
                image, true_label = test_dataset[idx]

                # Debug: Print info for first few samples
                if idx < 3:
                    print(f"\n=== Sample {idx} (true_label={true_label}) ===")

                variants = self.get_datapoint_variants(image, true_label)
                predictions, confidences = predict_variants(
                    self.model, variants, self.device)

                # Debug: Print prediction summary for first few samples
                if idx < 3:
                    print(f"Predictions: {predictions}")
                    print(f"Confidences: {[f'{c:.4f}' for c in confidences]}")

                final_prediction, avg_confidence, is_clean = aggregate_predictions(
                    predictions, confidences, confidence_threshold)

                if idx < 3:
                    print(
                        f"Final prediction: {final_prediction}, avg_confidence: {avg_confidence:.4f}, is_clean: {is_clean}")

                results['predictions'].append(final_prediction)
                results['confidences'].append(avg_confidence)
                results['labels'].append(true_label)

                # Store detailed results using exact cleaner.py column names
                # Use exact same variable names as cleaner.py
                index = idx
                preds = np.array(predictions)
                if self.train_noise_adder and self.noise_type != 'none':
                    noisy_label = int(test_dataset.targets[idx]) if hasattr(
                        test_dataset, 'targets') else true_label
                    real_label = int(self.train_noise_adder.orginal_labels[idx]) if hasattr(
                        self.train_noise_adder, 'orginal_labels') else true_label
                    is_noisy = idx in set(self.train_noise_adder.noisy_indices) if hasattr(
                        self.train_noise_adder, 'noisy_indices') else False
                else:
                    noisy_label = true_label
                    real_label = true_label
                    is_noisy = False

                # Calculate mistakes (same logic as cleaner.py)
                mistakes_counter = 0
                for p in preds:
                    if p != noisy_label:
                        mistakes_counter += 1

                # Convert predictions to string format (same as cleaner.py)
                s = '|'.join(np.array(preds, dtype=np.str_))

                # Calculate correct_label_pred (same logic as cleaner.py)
                correct_label_pred: int
                unique, counts = np.unique(preds, return_counts=True)
                sorted = np.sort(-counts)
                if len(sorted) > 1 and sorted[0] == sorted[1]:
                    correct_label_pred = -1
                else:
                    correct_label_pred = int(unique[np.argsort(-counts)[0]])

                # Store only the columns needed for cleaner.py format
                detailed_result = {
                    'index': index,
                    'noisy_label': noisy_label,
                    'is_noisy': is_noisy,
                    'real_label': real_label,
                    'mistakes': mistakes_counter,
                    'label_pred': correct_label_pred,
                    'preds': s
                }
                results['detailed_results'].append(detailed_result)

                if is_clean:
                    results['clean_indices'].append(idx)
                else:
                    results['noisy_indices'].append(idx)

        predictions = np.array(results['predictions'])
        labels = np.array(results['labels'])
        results['accuracy'] = np.mean(predictions == labels)

        if len(results['clean_indices']) > 0:
            clean_predictions = predictions[results['clean_indices']]
            clean_labels = labels[results['clean_indices']]
            results['clean_accuracy'] = np.mean(
                clean_predictions == clean_labels)

        print(f"Overall Accuracy: {results['accuracy']:.4f}")
        print(
            f"Clean Samples: {len(results['clean_indices'])}/{len(test_dataset)}")
        print(f"Clean Accuracy: {results['clean_accuracy']:.4f}")
        print(f"Average Confidence: {np.mean(results['confidences']):.4f}")

        # Always save detailed results if path is provided
        if self.results_save_path:
            self._save_detailed_results(results['detailed_results'])

        return results

    def save_model(self, path: str) -> None:
        """Save the trained model state."""
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
            print(f"Model saved to {path}")
        else:
            raise ValueError("No trained model to save!")

    def load_model(self, path: str) -> None:
        """Load a pre-trained model state."""
        if self.model is None:
            self.model = self.create_model()

        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.trained = True
        print(f"Model loaded from {path}")

    def _save_detailed_results(self, detailed_results: List[Dict]) -> None:
        """Save detailed TTA evaluation results to CSV file using cleaner.py format."""
        if not self.results_save_path:
            return

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.results_save_path), exist_ok=True)

        with open(self.results_save_path, mode='w', newline='') as f:
            # Use exact same fieldnames as cleaner.py
            writer = csv.DictWriter(f, fieldnames=[
                                    'index', 'noisy_label', 'is_noisy', 'real_label', 'mistakes', 'label_pred', 'preds'])
            writer.writeheader()

            # Data is already in correct format, just write it directly
            for result in detailed_results:
                writer.writerow(result)

        print(f"Detailed results saved to: {self.results_save_path}")


def save_clean_indices_to_file(clean_indices: List[int], save_path: str) -> None:
    """Save clean sample indices to CSV file (similar to existing NoiseCleaner pattern)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index'])  # Header
        for idx in clean_indices:
            writer.writerow([idx])

    print(f"Clean indices saved to: {save_path}")
