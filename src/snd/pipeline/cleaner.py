import csv
import math
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from snd.data.cifar10n import CIFAR10N
from snd.data.dataset import DatasetPairs, DatasetSingle
from snd.data.fold import CustomKFoldSplitter
from snd.data.instance_dependent import InstanceDependentNoiseAdder
from snd.data.noise import LabelNoiseAdder
from snd.evaluation.cleaner_report import CleanerReportingMixin
from snd.models.siamese import SiameseNetwork
from snd.pipeline.detector import NoiseDetector


class NoiseCleaner(CleanerReportingMixin):
    """Main class for cleaning noisy labels from datasets using Siamese networks.

    Implements nested cross-validation for noise detection and dataset cleaning.
    The ground-truth analysis and plotting methods live in
    :class:`~snd.evaluation.cleaner_report.CleanerReportingMixin`.
    """

    def __init__(self, dataset, model_save_path, inner_folds_num, outer_folds_num, noise_type, model, train_noise_level=0.1, epochs_num=30,
                 train_pairs=6000, val_pairs=1000, transform=None, embedding_dimension=128, lr=0.001, optimizer='Adam', distance_meter='euclidian',
                 patience=5, weight_decay=0.001, training_batch_size=256, pre_trained=True, dropout_prob=0.5, contrastive_ratio=3,
                 augmented_transform=None, trainable=True, pair_validation=True, label_smoothing=0.1, loss='ce', cnn_size=None, margin=5,
                 freeze_epoch=10, noisy_indices_path='', prediction_path='', mistakes_count=-1, relabeling_range=range(1), num_class=10,
                 siamese_middle_size: int = None, parallel: bool = False):
        """Initialize the noise cleaner with dataset, model and noise configuration."""
        self.parallel = parallel
        self.num_class = num_class
        self.dataset = dataset
        self.lr = lr
        self.weight_decay = weight_decay
        self.training_batch_size = training_batch_size
        self.pre_trained = pre_trained
        self.dropout_prob = dropout_prob
        self.contrastive_ratio = contrastive_ratio
        self.distance_meter = distance_meter
        self.augmented_transform = augmented_transform
        self.trainable = trainable
        self.pair_validation = pair_validation
        self.label_smoothing = label_smoothing
        self.loss = loss
        self.cnn_size = cnn_size
        self.margin = margin
        self.freeze_epoch = freeze_epoch
        self.noisy_indices_path = noisy_indices_path
        self.prediction_path = prediction_path
        self.relabeling_range = relabeling_range
        self.siamese_middle_size = siamese_middle_size
        if mistakes_count == -1:
            self.mistakes_count = self.inner_folds_num
        else:
            self.mistakes_count = mistakes_count
        self.noise_type = noise_type
        if noise_type == 'idn':
            image_size = self.get_image_size()
            self.train_noise_adder = InstanceDependentNoiseAdder(
                dataset, image_size=image_size, ratio=train_noise_level, num_classes=self.num_class)
            self.train_noise_adder.add_noise()
        elif noise_type == 'iin':
            self.train_noise_adder = LabelNoiseAdder(
                dataset, noise_level=train_noise_level, num_classes=self.num_class)
            self.train_noise_adder.add_noise()
        elif noise_type == 'cifar10n':
            self.train_noise_adder = CIFAR10N(dataset)
            self.train_noise_adder.add_noise()
        elif noise_type == 'none':
            a = 2
        else:
            raise ValueError('Noise type is not defined')

        if noise_type != 'none':
            print(
                f'noise count: {len(self.train_noise_adder.get_noisy_indices())} out of {len(dataset)} data')
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model_save_path = model_save_path
        self.inner_folds_num = inner_folds_num
        self.outer_folds_num = outer_folds_num
        self.custom_kfold_splitter = CustomKFoldSplitter(dataset_size=len(
            dataset), labels=dataset.targets, num_folds=outer_folds_num, shuffle=True)
        self.predicted_noise_indices = []
        self.clean_dataset = None
        self.model = model
        self.epochs_num = epochs_num
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.transform = transform
        self.embedding_dimension = embedding_dimension
        self.optimzer = optimizer
        self.patience = patience
        self.ensure_model_directory_exists()

    def save_noisy_dataset(self, save_dir: str, dataset_name: str):
        """Save the noisy dataset to disk for later use."""
        if self.train_noise_adder is None:
            raise ValueError(
                "The noisy dataset is not available. Call the `add_noise` method first.")

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{dataset_name}.pkl")
        with open(save_path, "wb") as f:
            for (img, label) in tqdm(self.dataset):
                img_array = np.array(img)
                entry = {'data': img_array, 'label': label}
                pickle.dump(entry, f)

        print(f"Noisy dataset saved to {save_path}")

    def ensure_model_directory_exists(self):
        """Create directory for saving models if it doesn't exist."""
        model_dir = os.path.dirname(self.model_save_path.format(0))
        os.makedirs(model_dir, exist_ok=True)

    def get_image_size(self):
        """Get the flattened size of images in the dataset."""
        sample, _ = self.dataset[0]
        if isinstance(sample, PIL.Image.Image):
            sample = transforms.ToTensor()(sample)
        return sample.shape[0] * sample.shape[1] * sample.shape[2]

    def remove_noisy_samples(self, dataset, noisy_indices):
        """Create a clean dataset by removing samples with detected noisy labels."""
        clean_indices = [i for i in range(
            len(dataset)) if i not in noisy_indices]
        cleaned_dataset = Subset(dataset, clean_indices)
        return cleaned_dataset

    def clean(self):
        """Main method to detect and remove noisy labels using nested cross-validation."""
        for fold in range(self.outer_folds_num):
            file_path = self.noisy_indices_path.format(fold + 1)
            if os.path.exists(file_path):
                print(f'Skipping outer fold {fold + 1} with results:')
                self.process_and_load_noisy_indices(file_path)
                continue
            train_indices, val_indices = self.custom_kfold_splitter.get_fold(
                fold)
            self.handle_fold(fold, train_indices, val_indices)
        self.clean_dataset = self.remove_noisy_samples(
            self.dataset, self.predicted_noise_indices)

    def save_cleaned_cifar_dataset(self, save_dir: str, dataset_name: str):
        if self.clean_dataset is None:
            raise ValueError(
                "The cleaned dataset is not available. Call the `clean` method first.")

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{dataset_name}.pkl")
        with open(save_path, "wb") as f:
            for (img, label) in tqdm(self.clean_dataset):
                img_array = np.array(img)
                entry = {'data': img_array, 'label': label}
                pickle.dump(entry, f)

        print(f"Cleaned dataset saved to {save_path}")

    def save_cleaned_cifar_dataset_manual(self, clean_dataset, save_dir: str, dataset_name: str):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{dataset_name}.pkl")
        with open(save_path, "wb") as f:
            for (img, label) in tqdm(clean_dataset):
                img_array = np.array(img)
                entry = {'data': img_array, 'label': label}
                pickle.dump(entry, f)

        print(f"Cleaned dataset saved to {save_path}")

    def handle_fold(self, fold, train_indices, val_indices):
        print(f'handling big fold {fold + 1}/{self.outer_folds_num}')
        train_subset = Subset(self.dataset, train_indices)
        val_subset = Subset(self.dataset, val_indices)
        number_of_pairs = math.floor(len(val_subset) * (math.e - 2))
        print(f'number_of_pairs: {number_of_pairs}')

        noise_detector = NoiseDetector(SiameseNetwork, train_subset, self.device, model_save_path=self.model_save_path,
                                       num_folds=self.inner_folds_num, model=self.model, train_pairs=self.train_pairs,
                                       val_pairs=self.val_pairs, transform=self.transform, embedding_dimension=self.embedding_dimension,
                                       optimizer=self.optimzer, patience=self.patience, weight_decay=self.weight_decay,
                                       batch_size=self.training_batch_size, pre_trained=self.pre_trained, dropout_prob=self.dropout_prob,
                                       contrastive_ratio=self.contrastive_ratio, distance_meter=self.distance_meter,
                                       augmented_transform=self.augmented_transform, trainable=self.trainable,
                                       label_smoothing=self.label_smoothing, loss=self.loss, cnn_size=self.cnn_size, margin=self.margin,
                                       freeze_epoch=self.freeze_epoch, prediction_path=self.prediction_path, num_classes=self.num_class,
                                       siamese_middle_size=self.siamese_middle_size, parallel=self.parallel)
        noise_detector.train_models(num_epochs=self.epochs_num, lr=self.lr)

        if self.pair_validation:
            test_dataset_pair = DatasetPairs(
                val_subset, num_pairs_per_epoch=number_of_pairs, transform=self.transform)
            test_loader = DataLoader(
                test_dataset_pair, batch_size=1024, shuffle=False)
            wrong_preds = noise_detector.evaluate_noisy_samples(test_loader)
        else:
            test_dataset = DatasetSingle(val_subset, transform=self.transform)
            test_loader = DataLoader(
                test_dataset, batch_size=1024, shuffle=False)
            wrong_preds, predictions = noise_detector.evaluate_noisy_samples_one_by_one(
                test_loader)
            predictions_indices = self.custom_kfold_splitter.get_original_indices_as_dic(
                fold, predictions.keys())
            self.save_predictions(fold, predictions, predictions_indices)

        predicted_noise_indices = [
            idx for (idx, count) in wrong_preds.items() if count >= self.mistakes_count]
        counts = [count for (idx, count) in wrong_preds.items()]
        plt.hist(counts)
        plt.show()
        predicted_noise_original_indices = self.custom_kfold_splitter.get_original_indices(
            fold, predicted_noise_indices)
        print(f'Predicted noise indices: {predicted_noise_original_indices}')
        if self.noise_type != 'none':
            self.train_noise_adder.calculate_noised_label_percentage(
                predicted_noise_original_indices)
        self.predicted_noise_indices.extend(predicted_noise_original_indices)

        self.save_noisy_indices(fold, predicted_noise_original_indices)

        for i in range(self.inner_folds_num):
            if i == fold:
                continue
            path = self.model_save_path.format(i + 1)
            os.remove(path)
            print(f'Removed model {path}')

    def save_predictions(self, fold, predictions: defaultdict[int, list], predictions_indices: defaultdict[int, int]):
        dic = defaultdict()
        for i in predictions.keys():
            array = predictions[i]
            index = predictions_indices[i]
            dic[index] = array

        self.process_predictions(dic, fold)

    def process_predictions(self, dic: defaultdict[int], fold):
        correct = all = 0
        noisy_indices = set(
            self.train_noise_adder.noisy_indices) if self.noise_type != 'none' else set()
        file_path = self.prediction_path.format(fold + 1)
        model_dir = os.path.dirname(file_path)
        os.makedirs(model_dir, exist_ok=True)
        mistakes_counter = 0
        with open(file_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                                    'index', 'noisy_label', 'is_noisy', 'real_label', 'mistakes', 'label_pred', 'preds'])
            writer.writeheader()
            for index in dic.keys():
                preds = np.array(dic[index])
                noisy_label = int(self.dataset.targets[index])
                is_noisy = noisy_indices.__contains__(index)
                real_label = int(
                    self.train_noise_adder.orginal_labels[index]) if self.noise_type != 'none' else 0
                mistakes_counter = 0
                for p in preds:
                    if p != noisy_label:
                        mistakes_counter += 1
                s = '|'.join(np.array(preds, dtype=np.str_))
                correct_label_pred: int
                unique, counts = np.unique(preds, return_counts=True)
                sorted = np.sort(-counts)
                if len(sorted) > 1 and sorted[0] == sorted[1]:
                    correct_label_pred = -1
                else:
                    correct_label_pred = int(unique[np.argsort(-counts)[0]])

                writer.writerow({'index': index, 'noisy_label': noisy_label, 'is_noisy': is_noisy, 'real_label': real_label,
                                 'mistakes': mistakes_counter, 'label_pred': correct_label_pred, 'preds': s})

                if mistakes_counter >= self.mistakes_count:
                    if correct_label_pred != -1:
                        if correct_label_pred == real_label:
                            correct += 1
                        all += 1

        if all != 0:
            print(f'{correct / all * 100}% relabeling accuracy')
        else:
            print(
                f'mistakes_counter: {mistakes_counter}, self.mistakes_count: {self.mistakes_count}')

    def process_and_load_noisy_indices(self, file_path):
        noisy_indices = []
        with open(file_path, mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                noisy_indices.extend(map(int, row))

        if self.noise_type != 'none':
            self.train_noise_adder.calculate_noised_label_percentage(
                noisy_indices)
        self.predicted_noise_indices.extend(noisy_indices)
        print(f'Loaded {len(noisy_indices)} noisy indices from {file_path}')

    def save_noisy_indices(self, fold, noisy_indices):
        file_path = self.noisy_indices_path.format(fold + 1)
        model_dir = os.path.dirname(file_path)
        os.makedirs(model_dir, exist_ok=True)

        with open(file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(noisy_indices)

        print(f'Noisy indices for fold {fold + 1} saved to {file_path}')

    def read_predictions(self):
        array = []
        for fold in range(1, self.inner_folds_num + 1):
            file_path = self.prediction_path.format(fold)
            with open(file_path, mode='r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    array.append(row)
        return array

    def advanced_clean(self, dataset, mistakes_count, relabel_threshold=-1):
        dataset.targets = self.train_noise_adder.noisy_labels if self.noise_type != 'none' else dataset.targets
        array = self.read_predictions()
        predicted_noise_indices = []
        new_labels = defaultdict()
        for item in array:
            index = int(item['index'])
            mistakes = int(item['mistakes'])
            preds = np.array(str(item['preds']).split('|'), dtype=np.int32)

            if mistakes >= mistakes_count:
                predicted_noise_indices.append(index)

                if relabel_threshold != -1:
                    unique, counts = np.unique(preds, return_counts=True)
                    if relabel_threshold > 0:
                        found = unique[counts >= relabel_threshold]
                        if len(found) > 0:
                            new_labels[index] = int(found[0])
                    else:
                        sorted = np.sort(-counts)
                        if not (len(sorted) > 1 and sorted[0] == sorted[1]):
                            int(unique[np.argsort(-counts)[0]])
                            correct_label_pred = int(
                                unique[np.argsort(-counts)[0]])
                            new_labels[index] = correct_label_pred

        predicted_noise_indices_set = set(predicted_noise_indices)
        ls = set(new_labels.keys())
        should_be_removed = np.array(list(predicted_noise_indices_set - ls))

        for idx in new_labels.keys():
            new_label = new_labels[idx]
            dataset.targets[idx] = new_label
        clean_indices = [i for i in range(
            len(dataset)) if i not in should_be_removed]
        final_targets = []
        for i, item in enumerate(dataset.targets):
            if i in should_be_removed:
                final_targets.append(-1)
            else:
                final_targets.append(item)
        cleaned_dataset = Subset(dataset, clean_indices)
        if self.noise_type != 'none':
            self.train_noise_adder.report(predicted_noise_indices)
            self.train_noise_adder.report(should_be_removed)
        all = 0
        correct = 0
        for i in range(len(final_targets)):
            new = final_targets[i]
            if new == -1:
                continue
            all += 1
            real = self.train_noise_adder.orginal_labels[i] if self.noise_type != 'none' else 0
            if real == new:
                correct += 1
        print(f'{len(should_be_removed)} removed from dataset and {len(ls)} relabled')
        print(f'{100 - (correct / all * 100):.2f}% noise remained in {all} data')
        self.plot_before_after(correct, all)
        return cleaned_dataset
