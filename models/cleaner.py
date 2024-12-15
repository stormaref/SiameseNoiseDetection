import torch
from torch.utils.data import Subset, DataLoader
from models.dataset import DatasetPairs, DatasetSingle, PositiveSamplingDatasetPairs
from models.siamese import SiameseNetwork, SimpleSiamese
from models.noise import LabelNoiseAdder
from models.detector import NoiseDetector
from models.fold import CustomKFoldSplitter
from models.predefined import InstanceDependentNoiseAdder
from torchvision import transforms
import PIL
import matplotlib.pyplot as plt
import math
import os
import pickle
import numpy as np
from tqdm import tqdm
import csv
from collections import defaultdict
from sklearn.metrics import auc

class NoiseCleaner:
    def __init__(self, dataset, model_save_path, inner_folds_num, outer_folds_num, noise_type, model, train_noise_level=0.1, epochs_num=30,
                 train_pairs=6000, val_pairs=1000, transform=None, embedding_dimension=128, lr=0.001, optimizer='Adam', distance_meter='euclidian',
                 patience=5, weight_decay=0.001, training_batch_size=256, pre_trained=True, dropout_prob=0.5, contrastive_ratio=3,
                 augmented_transform=None, trainable=True, pair_validation=True, label_smoothing=0.1, loss='ce', cnn_size=None, margin=5,
                 freeze_epoch=10, noisy_indices_path='', prediction_path='', mistakes_count=-1, relabeling_range=range(1)):
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
        if mistakes_count == -1:
            self.mistakes_count = self.inner_folds_num
        else:
            self.mistakes_count = mistakes_count
        if noise_type == 'idn':
            image_size = self.get_image_size()
            self.train_noise_adder = InstanceDependentNoiseAdder(dataset, image_size=image_size, ratio=train_noise_level, num_classes=10)
            self.train_noise_adder.add_noise()
        elif noise_type == 'iin':
            self.train_noise_adder = LabelNoiseAdder(dataset, noise_level=train_noise_level, num_classes=10)
            self.train_noise_adder.add_noise()
        elif noise_type == 'none':
            a = 2
        else:
            raise ValueError('Noise type should be either "idn" or "iin"')
        
        if noise_type != 'none':
            print(f'noise count: {len(self.train_noise_adder.get_noisy_indices())} out of {len(dataset)} data')
        self.device = torch.device('cuda')
        self.model_save_path = model_save_path
        self.inner_folds_num = inner_folds_num
        self.outer_folds_num = outer_folds_num
        self.custom_kfold_splitter = CustomKFoldSplitter(dataset_size=len(dataset), labels=dataset.targets, num_folds=outer_folds_num, shuffle=True)
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
        
    def ensure_model_directory_exists(self):
        model_dir = os.path.dirname(self.model_save_path.format(0))
        os.makedirs(model_dir, exist_ok=True)
        
    def get_image_size(self):
        sample, _ = self.dataset[0]
        if isinstance(sample, PIL.Image.Image):
            sample = transforms.ToTensor()(sample)
        return sample.shape[0] * sample.shape[1] * sample.shape[2]
        
    def remove_noisy_samples(self, dataset, noisy_indices):
        clean_indices = [i for i in range(len(dataset)) if i not in noisy_indices]
        cleaned_dataset = Subset(dataset, clean_indices)
        return cleaned_dataset

    def clean(self):
        for fold in range(self.outer_folds_num):
            file_path = self.noisy_indices_path.format(fold + 1)
            if os.path.exists(file_path):
                print(f'Skipping outer fold {fold + 1} with results:')
                self.process_and_load_noisy_indices(file_path)
                continue
            train_indices, val_indices = self.custom_kfold_splitter.get_fold(fold)
            self.handle_fold(fold, train_indices, val_indices)
        self.clean_dataset = self.remove_noisy_samples(self.dataset, self.predicted_noise_indices)
        
    def report(self, mistakes_count):
        predicted_noise_indices = []
        array = self.read_predictions()
        for row in array:
            m = int(row['mistakes'])
            index = int(row['index'])
            if m >= mistakes_count:                        
                predicted_noise_indices.append(index)
        self.train_noise_adder.report(predicted_noise_indices)
        
    def analyze_relabeling(self, detected_noise: bool, preds: np.array, real_label: int):
        result = []
        for i in self.relabeling_range:
            if not detected_noise:
                result.append(-1)
                continue
            unique, counts = np.unique(preds, return_counts=True)
            found = unique[counts >= i]
            if len(found) == 0:
                result.append(-1)
                continue
            if found[0] != real_label:
                result.append(1)
            else:
                result.append(2)
        return result
        
    def analyze(self):
        array = self.read_predictions()
        tpr_list = []
        fpr_list = []
        relabeling_accuracies = []
        relabeling_ratios = []
        relabeling_accuracy_analysis = []
        relabeling_ratio_analysis = []

        for mistakes_count in range(1, self.inner_folds_num + 1):
            tpr, fpr, relabeling_accuracy, relabel_ratio, accuracy_analysis, ratio_analysis = self.analyze_with_mistakes_count(array, mistakes_count)

            tpr_list.append(tpr)
            fpr_list.append(fpr)
            
            relabeling_accuracies.append(relabeling_accuracy)
            relabeling_ratios.append(relabel_ratio)
            
            relabeling_accuracy_analysis.append(accuracy_analysis)
            relabeling_ratio_analysis.append(ratio_analysis)
            
        relabeling_accuracy_analysis = np.array(relabeling_accuracy_analysis)
        relabeling_ratio_analysis = np.array(relabeling_ratio_analysis)
        multiply = relabeling_accuracy_analysis * relabeling_ratio_analysis
        
        self.plot_relabeling_analysis(relabeling_accuracy_analysis, "Accuracy")
        self.plot_relabeling_analysis(relabeling_ratio_analysis, "Ratio")
        self.plot_relabeling_analysis(100 * multiply, "Multiplied")
        
        self.plot_roc(fpr_list, tpr_list)
        
        self.plot_relabeling(relabeling_accuracies, relabeling_ratios)
        
    def plot_relabeling_analysis(self, relabeling_results, title):
        num_mistakes, num_thresholds = relabeling_results.shape

        plt.figure(figsize=(10, 6))
        c = plt.imshow(relabeling_results, cmap='viridis', aspect='auto')

        plt.colorbar(c, label='Relabeling Metric')

        plt.xlabel('Relabeling Thresholds', fontsize=12)
        plt.ylabel('Number of Mistakes', fontsize=12)
        plt.title(f'Relabeling {title} Heatmap', fontsize=14)

        plt.xticks(ticks=np.arange(num_thresholds), labels=[f'Thresh {i+self.relabeling_range.start}' for i in range(num_thresholds)])
        plt.yticks(ticks=np.arange(num_mistakes), labels=[f'{i+1}' for i in range(num_mistakes)])

        for i in range(num_mistakes):
            for j in range(num_thresholds):
                plt.text(j, i, f'{relabeling_results[i, j]:.2f}', 
                        ha='center', va='center', color='white', fontsize=8)

        plt.tight_layout()
        plt.show()

    def analyze_with_mistakes_count(self, array, mistakes_count):
        predicted_noise_indices = []
        correct_relabel = 0
        perform_relabel = 0
        all_relabel = 0
        relabeling_analysis = []
        
        for item in array:
            index = int(item['index'])
            noisy_label = int(item['noisy_label'])
            is_noisy = bool(item['is_noisy'])
            real_label = int(item['real_label'])
            mistakes = int(item['mistakes'])
            label_pred = int(item['label_pred'])
            preds = np.array(str(item['preds']).split('|'), dtype=np.int32)
            
            if mistakes >= mistakes_count:
                predicted_noise_indices.append(index)
                
                if label_pred != -1:
                    if label_pred == real_label:
                        correct_relabel += 1
                    perform_relabel += 1
                all_relabel += 1
                
            result = self.analyze_relabeling(mistakes >= mistakes_count, preds, real_label)
            relabeling_analysis.append(result)
            
        l = self.relabeling_range.stop - self.relabeling_range.start
        relabeling_accuracy_analysis = []
        relabeling_ratio_analysis = []
        for i in range(l):
            correct_i = 0
            performed_i = 0
            all_i = 0
            for j in relabeling_analysis:
                if j[i] > 0:
                    if j[i] == 2:
                        correct_i += 1
                    performed_i += 1
                all_i += 1
            relabeling_accuracy_analysis.append(correct_i / performed_i)
            relabeling_ratio_analysis.append(performed_i / all_i)
        
        tn, fp, fn, tp = self.train_noise_adder.ravel(predicted_noise_indices)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        relabeling_accuracy = correct_relabel / perform_relabel
        relabel_ratio = perform_relabel / all_relabel
        return tpr, fpr, relabeling_accuracy, relabel_ratio, relabeling_accuracy_analysis, relabeling_ratio_analysis
    
    def plot_roc(self, fpr_list, tpr_list):
        plt.figure(figsize=(8, 6))
        for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
            plt.plot(fpr, tpr, marker='o', label=f'Mistakes Count: {i + 1}' if 1 == 2 else '')  # Plot each point
            plt.annotate(f"{i + 1}", (fpr, tpr), textcoords="offset points", xytext=(5, -5), ha='center')  # Annotate with mistakes_count

        plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')

        plt.title('ROC Curve with Mistakes Count Annotations')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.legend()
        plt.grid()
        plt.show()

        roc_auc = auc(fpr_list, tpr_list)
        print(f"Area Under the Curve (AUC): {roc_auc}")

    def plot_relabeling(self, relabeling_accuracies, relabel_ratios):
        mistake_counts = range(1, len(relabeling_accuracies) + 1)  # The position in the array represents the number of mistakes.
        bar_width = 0.35  # Width of each bar
        x = np.arange(len(mistake_counts))  # X positions for the groups

        plt.figure(figsize=(10, 6))
        plt.bar(x - bar_width / 2, relabeling_accuracies, width=bar_width, label='Accuracy', color='blue')
        plt.bar(x + bar_width / 2, relabel_ratios, width=bar_width, label='Relabel Ratio', color='orange')
        
        plt.xlabel('Mistake Count', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Relabeling Accuracy and Ratio by Mistake Count', fontsize=14)
        plt.xticks(x, labels=mistake_counts)  # Label x-axis with mistake counts
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        for i, (accuracy, ratio) in enumerate(zip(relabeling_accuracies, relabel_ratios)):
            plt.text(i - bar_width / 2, accuracy + 0.02, f'{accuracy * 100:.2f}', ha='center', fontsize=9)
            plt.text(i + bar_width / 2, ratio + 0.02, f'{ratio * 100:.2f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def save_cleaned_cifar_dataset(self, save_dir: str, dataset_name: str):
        if self.clean_dataset is None:
            raise ValueError("The cleaned dataset is not available. Call the `clean` method first.")

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
        
        noise_detector = NoiseDetector(SiameseNetwork, train_subset, self.device, model_save_path=self.model_save_path, num_folds=self.inner_folds_num, 
                                       model=self.model, train_pairs=self.train_pairs, val_pairs=self.val_pairs, transform=self.transform, 
                                       embedding_dimension=self.embedding_dimension, optimizer=self.optimzer, patience=self.patience,
                                       weight_decay=self.weight_decay, batch_size=self.training_batch_size, pre_trained=self.pre_trained,
                                       dropout_prob=self.dropout_prob, contrastive_ratio=self.contrastive_ratio, distance_meter=self.distance_meter,
                                       augmented_transform=self.augmented_transform, trainable=self.trainable, label_smoothing=self.label_smoothing,
                                       loss=self.loss, cnn_size=self.cnn_size, margin=self.margin, freeze_epoch=self.freeze_epoch, prediction_path=self.prediction_path)
        noise_detector.train_models(num_epochs=self.epochs_num, lr=self.lr)
       
        if self.pair_validation:
            test_dataset_pair = DatasetPairs(val_subset, num_pairs_per_epoch=number_of_pairs, transform=self.transform)
            test_loader = DataLoader(test_dataset_pair, batch_size=1024, shuffle=False)
            wrong_preds = noise_detector.evaluate_noisy_samples(test_loader)
        else:
            test_dataset = DatasetSingle(val_subset, transform=self.transform)
            test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
            wrong_preds, predictions = noise_detector.evaluate_noisy_samples_one_by_one(test_loader)
            predictions_indices = self.custom_kfold_splitter.get_original_indices_as_dic(fold, predictions.keys())
            self.save_predictions(fold, predictions, predictions_indices)
            
        predicted_noise_indices = [idx for (idx, count) in wrong_preds.items() if count >= self.mistakes_count]
        counts = [count for (idx, count) in wrong_preds.items()]
        plt.hist(counts)
        plt.show()
        predicted_noise_original_indices = self.custom_kfold_splitter.get_original_indices(fold, predicted_noise_indices)
        print(f'Predicted noise indices: {predicted_noise_original_indices}')
        self.train_noise_adder.calculate_noised_label_percentage(predicted_noise_original_indices)
        self.predicted_noise_indices.extend(predicted_noise_original_indices)
        
        self.save_noisy_indices(fold, predicted_noise_original_indices)
        
    def save_predictions(self, fold, predictions: defaultdict[int, list], predictions_indices: defaultdict[int, int]):
        dic = defaultdict()
        for i in predictions.keys():
            array = predictions[i]
            index = predictions_indices[i]
            dic[index] = array
        
        self.process_predictions(dic, fold)
        
    def process_predictions(self, dic: defaultdict[int], fold):
        correct = all = 0
        noisy_indices = set(self.train_noise_adder.noisy_indices)
        file_path = self.prediction_path.format(fold + 1)
        model_dir = os.path.dirname(file_path)
        os.makedirs(model_dir, exist_ok=True)
        with open(file_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['index', 'noisy_label', 'is_noisy', 'real_label', 'mistakes', 'label_pred', 'preds'])
            writer.writeheader()
            for index in dic.keys():
                preds = np.array(dic[index])
                noisy_label = int(self.dataset.targets[index])
                is_noisy = noisy_indices.__contains__(index)
                real_label = int(self.train_noise_adder.orginal_labels[index])
                mistakes_counter = 0
                for p in preds:
                    if p != noisy_label:
                        mistakes_counter += 1
                s = '|'.join(np.array(preds, dtype=np.str_))
                correct_label_pred : int
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
        
        print(f'{correct / all * 100}% relabeling accuracy')
        
    def process_and_load_noisy_indices(self, file_path):
        noisy_indices = []
        with open(file_path, mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                noisy_indices.extend(map(int, row))

        self.train_noise_adder.calculate_noised_label_percentage(noisy_indices)
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
        for fold in range(self.inner_folds_num):
            file_path = self.prediction_path.format(fold)
            with open(file_path, mode='r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    array.append(row)
        return array
        
    def advanced_clean(self, dataset, mistakes_count, relabel_threshold=-1):
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
                            correct_label_pred = int(unique[np.argsort(-counts)[0]])
                            new_labels[index] = correct_label_pred
                        
        predicted_noise_indices_set = set(predicted_noise_indices)
        should_be_removed = np.array(list(predicted_noise_indices_set - set(new_labels.keys())))
            
        for idx in new_labels.keys():
            new_label = new_labels[idx]
            dataset.targets[idx] = new_label
        clean_indices = [i for i in range(len(dataset)) if i not in should_be_removed]
        cleaned_dataset = Subset(dataset, clean_indices)
        self.train_noise_adder.report(predicted_noise_indices)
        self.train_noise_adder.report(should_be_removed)
        return cleaned_dataset