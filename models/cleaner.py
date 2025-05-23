import torch
from torch.utils.data import Subset, DataLoader
from models.dataset import DatasetPairs, DatasetSingle
from models.siamese import SiameseNetwork
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
import matplotlib.patches as patches
from models.cifar10n import CIFAR10N
import math
import random
from sklearn.manifold import TSNE

class NoiseCleaner:
    """Main class for cleaning noisy labels from datasets using Siamese networks.
    
    Implements nested cross-validation for noise detection and dataset cleaning.
    """
    def __init__(self, dataset, model_save_path, inner_folds_num, outer_folds_num, noise_type, model, train_noise_level=0.1, epochs_num=30,
                 train_pairs=6000, val_pairs=1000, transform=None, embedding_dimension=128, lr=0.001, optimizer='Adam', distance_meter='euclidian',
                 patience=5, weight_decay=0.001, training_batch_size=256, pre_trained=True, dropout_prob=0.5, contrastive_ratio=3,
                 augmented_transform=None, trainable=True, pair_validation=True, label_smoothing=0.1, loss='ce', cnn_size=None, margin=5,
                 freeze_epoch=10, noisy_indices_path='', prediction_path='', mistakes_count=-1, relabeling_range=range(1), num_class=10,
                 siamese_middle_size:int=None):
        """Initialize the noise cleaner with dataset, model and noise configuration."""
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
        if noise_type == 'idn':
            image_size = self.get_image_size()
            self.train_noise_adder = InstanceDependentNoiseAdder(dataset, image_size=image_size, ratio=train_noise_level, num_classes=self.num_class)
            self.train_noise_adder.add_noise()
        elif noise_type == 'iin':
            self.train_noise_adder = LabelNoiseAdder(dataset, noise_level=train_noise_level, num_classes=self.num_class)
            self.train_noise_adder.add_noise()
        elif noise_type == 'cifar10n':
            self.train_noise_adder = CIFAR10N(dataset)
            self.train_noise_adder.add_noise()
        elif noise_type == 'none':
            a = 2
        else:
            raise ValueError('Noise type is not defined')
        
        if noise_type != 'none':
            print(f'noise count: {len(self.train_noise_adder.get_noisy_indices())} out of {len(dataset)} data')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
        
    def save_noisy_dataset(self, save_dir: str, dataset_name: str):
        """Save the noisy dataset to disk for later use."""
        if self.train_noise_adder is None:
            raise ValueError("The noisy dataset is not available. Call the `add_noise` method first.")

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
        clean_indices = [i for i in range(len(dataset)) if i not in noisy_indices]
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
            train_indices, val_indices = self.custom_kfold_splitter.get_fold(fold)
            self.handle_fold(fold, train_indices, val_indices)
        self.clean_dataset = self.remove_noisy_samples(self.dataset, self.predicted_noise_indices)
        
    def report(self, mistakes_count, detail=False):
        """Generate report on detected noisy labels using a specified mistake threshold."""
        predicted_noise_indices = []
        array = self.read_predictions()
        for row in array:
            m = int(row['mistakes'])
            index = int(row['index'])
            if m >= mistakes_count:                        
                predicted_noise_indices.append(index)
        if not detail:
            self.train_noise_adder.report(predicted_noise_indices)
            return
        return self.train_noise_adder.calculate_metrics(predicted_noise_indices)
        
    def analyze_relabeling(self, detected_noise: bool, preds: np.array, real_label: int):
        """Analyze potential relabeling outcomes for detected noisy samples."""
        result = []
        for i in self.relabeling_range:
            if not detected_noise:
                result.append(-1)
                continue
            unique, counts = np.unique(preds, return_counts=True)
            found = unique[counts >= i]
            if len(found) == 0:
                result.append(0)
                continue
            if found[0] != real_label:
                result.append(1)
            else:
                result.append(2)
        return result
        
    def analyze(self):
        """Analyze noise detection performance with ROC curves and relabeling strategies."""
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
        """Plot heatmap of relabeling analysis results."""
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
                if j[i] >= 0:
                    if j[i] >= 1:
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
    
    def calculate_relabeling_score(self, mistakes_count, relabel_threshold, plot=True):
        array = self.read_predictions()
        score = 0
        report = {}
        report['-2'] = 0
        report['-1'] = 0
        report['0'] = 0
        report['1'] = 0
        report['2'] = 0
        for item in array:
            is_noisy = item['is_noisy'] == 'True'
            real_label = int(item['real_label'])
            mistakes = int(item['mistakes'])
            preds = np.array(str(item['preds']).split('|'), dtype=np.int32)
            
            if mistakes < mistakes_count:
                # Not detected
                continue
            
            unique, counts = np.unique(preds, return_counts=True)
            found = unique[counts >= relabel_threshold]
            if len(found) > 0:
                # Relabeling
                new_label = int(found[0])
                if is_noisy:
                    if new_label == real_label:
                        score += 2
                        report['2'] += 1
                    else:
                        score += 0
                        report['0'] += 1
                elif (not is_noisy) and new_label != real_label:
                    score -= 2
                    report['-2'] += 1
            else:
                # No relabeling
                if is_noisy:
                    score += 1
                    report['1'] += 1
                else:
                    score -= 1
                    report['-1'] += 1
        
        if plot:
            self.plot_relabeling_score_diagram(report, score)
        return score, report
    
    def plot_relabeling_score_diagram(self, report, score):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        
        # Use enhanced hex colors for a richer palette
        colors = ['#8B0000', '#FF0000', '#808080', '#32CD32', '#006400']
        labels = ['-2', '-1', '0', '1', '2']
        
        root = (0.5, 0.5)
        r = 0.4  # Slightly larger radius for better spacing
        
        # Draw branches, nodes, and annotations
        for i, (color, label) in enumerate(zip(colors, labels)):
            angle = 2 * math.pi * i / len(labels)
            node_x = root[0] + r * math.cos(angle)
            node_y = root[1] + r * math.sin(angle)
            node = (node_x, node_y)
            
            # Draw a thick branch from the root to the node
            ax.plot([root[0], node[0]], [root[1], node[1]], color='black', lw=2, zorder=1)
            
            # Compute midpoint for branch label and add a background box
            mid_x = (root[0] + node[0]) / 2
            mid_y = (root[1] + node[1]) / 2
            ax.text(mid_x, mid_y, label, fontsize=14, ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', pad=1))
            
            # Draw the node as a larger circle with an edge
            circle = patches.Circle(node, 0.06, edgecolor='black', facecolor=color, lw=2, zorder=2)
            ax.add_patch(circle)
            
            # Annotate the node with the corresponding report value
            ax.text(node[0], node[1] - 0.10, str(report[label]), fontsize=14, ha='center', va='top', fontweight='bold')
        
        # Highlight the root node with a special color (gold) and add the overall score
        root_circle = patches.Circle(root, 0.04, edgecolor='black', facecolor='gold', lw=2, zorder=3)
        ax.add_patch(root_circle)
        ax.text(root[0], root[1] + 0.07, "Score", fontsize=16, ha='center', va='bottom', fontweight='bold')
        ax.text(root[0], root[1] - 0.07, str(score), fontsize=16, ha='center', va='top', fontweight='bold')
        
        # Set title and adjust finite parameters for a cleaner look
        plt.title(f"Report Tree Diagram (Overall Score: {score})", fontsize=18, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    def plot_roc(self, fpr_list, tpr_list):
        plt.figure(figsize=(8, 6))
        for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
            print(f"FPR: {fpr}, TPR: {tpr}, Mistakes Count: {i + 1}")
            plt.plot(fpr, tpr, marker='o', label=f'Mistakes Count: {i + 1}')  # Plot each point
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
        
        noise_detector = NoiseDetector(SiameseNetwork, train_subset, self.device, model_save_path=self.model_save_path, 
                                       num_folds=self.inner_folds_num, model=self.model, train_pairs=self.train_pairs, 
                                       val_pairs=self.val_pairs, transform=self.transform, embedding_dimension=self.embedding_dimension, 
                                       optimizer=self.optimzer, patience=self.patience, weight_decay=self.weight_decay, 
                                       batch_size=self.training_batch_size, pre_trained=self.pre_trained, dropout_prob=self.dropout_prob, 
                                       contrastive_ratio=self.contrastive_ratio, distance_meter=self.distance_meter, 
                                       augmented_transform=self.augmented_transform, trainable=self.trainable, 
                                       label_smoothing=self.label_smoothing, loss=self.loss, cnn_size=self.cnn_size, margin=self.margin, 
                                       freeze_epoch=self.freeze_epoch, prediction_path=self.prediction_path, num_classes=self.num_class, 
                                       siamese_middle_size=self.siamese_middle_size)
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
        for fold in range(1, self.inner_folds_num + 1):
            file_path = self.prediction_path.format(fold)
            with open(file_path, mode='r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    array.append(row)
        return array
        
    def advanced_clean(self, dataset, mistakes_count, relabel_threshold=-1):
        dataset.targets = self.train_noise_adder.noisy_labels
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
        ls = set(new_labels.keys())
        should_be_removed = np.array(list(predicted_noise_indices_set - ls))
            
        for idx in new_labels.keys():
            new_label = new_labels[idx]
            dataset.targets[idx] = new_label
        clean_indices = [i for i in range(len(dataset)) if i not in should_be_removed]
        final_targets = []
        for i, item in enumerate(dataset.targets):
            if i in should_be_removed:
                final_targets.append(-1)
            else:
                final_targets.append(item)
        cleaned_dataset = Subset(dataset, clean_indices)
        self.train_noise_adder.report(predicted_noise_indices)
        self.train_noise_adder.report(should_be_removed)
        all = 0
        correct = 0
        for i in range(len(final_targets)):
            new = final_targets[i]
            if new == -1:
                continue
            all += 1
            real = self.train_noise_adder.orginal_labels[i]
            if real == new:
                correct += 1
        print(f'{len(should_be_removed)} removed from dataset and {len(ls)} relabled')
        print(f'{100 - (correct / all * 100):.2f}% noise remained in {all} data')
        self.plot_before_after(correct, all)
        return cleaned_dataset
    
    def plot_before_after(self, correct, all):
        before_noisy = len(self.train_noise_adder.noisy_indices)
        before_clean = len(self.dataset) - before_noisy
        after_noisy = all - correct
        after_clean = correct
        labels = ['Before', 'After']
        noisy = [before_noisy, after_noisy]
        clean = [before_clean, after_clean]
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 8))  # Increased height of the plot
        rects1 = ax.bar(x - width/2, noisy, width, label='Noisy', color='tomato')
        rects2 = ax.bar(x + width/2, clean, width, label='Clean', color='skyblue')
        ax.set_ylabel('Count')
        ax.set_title('Before and After Cleaning')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        add_labels(rects1)
        add_labels(rects2)

        plt.show()
        
    def plot_false_positives(self, dataset, mistakes_count, count, labels, min_t=5):
        array = self.read_predictions()
        fp_indices = []
        fp_labels = []
        predicted = []
        for item in array:
            index = int(item['index'])
            is_noisy = item['is_noisy'] == 'True'
            real_label = int(item['real_label'])
            mistakes = int(item['mistakes'])
            preds = np.array(str(item['preds']).split('|'), dtype=np.int32)
            
            if mistakes >= mistakes_count and not is_noisy:
                fp_indices.append(index)
                fp_labels.append(real_label)
                unique, counts = np.unique(preds, return_counts=True)
                found = unique[counts >= min_t]
                if len(found > 0):
                    predicted.append(int(found[0]))
                else:
                    predicted.append(-1)
                                    
        
        cols = min(count, 5)
        rows = math.ceil(count / cols)
        plt.figure(figsize=(15, 3 * rows))
        
        for i in range(count):
            idx = random.randint(0, len(fp_indices) - 1)
            dataset_idx = fp_indices[idx]
            label = fp_labels[idx]
            pred = predicted[idx]
            img, _ = dataset[dataset_idx]
            plt.subplot(rows, cols, i + 1)
            plt.imshow(np.array(img))
            plt.title(f"I:{dataset_idx},R:{labels[label]},P:{labels[pred] if pred!=-1 else 'Unknown'}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        
    def plot_unknown(self, dataset, mistakes_count, count, labels):
        array = self.read_predictions()
        fp_indices = []
        fp_labels = []
        for item in array:
            index = int(item['index'])
            is_noisy = item['is_noisy'] == 'True'
            real_label = int(item['real_label'])
            mistakes = int(item['mistakes'])
            preds = np.array(str(item['preds']).split('|'), dtype=np.int32)
            
            unique, counts = np.unique(preds, return_counts=True)
            found = unique[counts >= 5]
            if mistakes >= mistakes_count and not is_noisy and len(found) == 0:
                fp_indices.append(index)
                fp_labels.append(real_label)
        
        cols = min(count, 5)
        rows = math.ceil(count / cols)
        plt.figure(figsize=(15, 3 * rows))
        
        for i in range(count):
            idx = random.randint(0, len(fp_indices) - 1)
            dataset_idx = fp_indices[idx]
            label = fp_labels[idx]
            img, _ = dataset[dataset_idx]
            plt.subplot(rows, cols, i + 1)
            plt.imshow(np.array(img))
            plt.title(f"Real: {labels[label]}, Pred: Unknown")
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        
    def analyze_parameters(self, start=8, end=10):
        results = []
        for td in range(start, end + 1):
            total = len(self.dataset)
            
            report = self.report(mistakes_count=td, detail=True)
            metrics = {
                'threshold': td,
                'precision': report['precision'],
                'recall': report['recall'],
                'f1': report['f1'],
                'accuracy': report['accuracy'],
                'relabeling': []
            }
            
            for tr in range(start, end + 1):
                score, r_report = self.calculate_relabeling_score(
                    mistakes_count=td,
                    relabel_threshold=tr,
                    plot=False
                )
                relabled = r_report['-2'] + r_report['2'] + r_report['0']
                relabeling_metrics = {
                    'threshold': tr,
                    'score': score,
                    'n_score': score / relabled,
                    'report': r_report,
                    'accuracy': r_report['2'] / (r_report['-2'] + r_report['2'] + r_report['0']) * 100,
                    'count': relabled,
                    
                }
                noisy = len(self.train_noise_adder.noisy_indices)
                clean = total - noisy
                clean -= r_report['-1']
                noisy -= r_report['1']
                
                clean -= r_report['-2']
                noisy += r_report['-2']
                
                clean += r_report['2']
                noisy -= r_report['2']
                relabeling_metrics['noise_ratio'] = noisy / (noisy + clean) * 100
                relabeling_metrics['remaining'] = noisy + clean
                relabeling_metrics['clean_after'] = clean
                relabeling_metrics['noisy_after'] = noisy
                metrics['relabeling'].append(relabeling_metrics)
                
            results.append(metrics)
        return results
    
    def plot_noise_rate_vs_wrong_predictions(self):
        array = self.read_predictions()
        clean_dic = {}
        noisy_dic = {}
        counter = {}
        for item in array:
            mistakes = int(item['mistakes'])
            is_noisy = item['is_noisy'] == 'True'
            counter[mistakes] = counter.get(mistakes, 0) + 1
            if is_noisy:
                noisy_dic[mistakes] = noisy_dic.get(mistakes, 0) + 1
            else:
                clean_dic[mistakes] = clean_dic.get(mistakes, 0) + 1
            
        clean_dic = {k: v / counter[k] for k, v in clean_dic.items()}
        noisy_dic = {k: v / counter[k] for k, v in noisy_dic.items()}
        
        clean_keys = sorted(clean_dic.keys())
        clean_values = [clean_dic[k] for k in clean_keys]
        noisy_keys = sorted(noisy_dic.keys())
        noisy_values = [noisy_dic[k] for k in noisy_keys]
        
        # plt.plot(clean_keys, clean_values, 'o-', label='Clean', color='green', markersize=8)
        plt.plot(noisy_keys, noisy_values, 'o-', label='Noisy', color='red', markersize=8)
        plt.legend()
        plt.xlabel('Mistakes')
        plt.ylabel('Rate')
        plt.title('Noise Rate vs Wrong Predictions')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
        
        print(noisy_values)
        
    def analyze_fold_latent(self, fold, cmap_txt):
        train_indices, val_indices = self.custom_kfold_splitter.get_fold(fold)
        self.analyze_latent(fold, train_indices, val_indices, cmap_txt)
        
    def plot_latent_analysis(self, latents, latents_indices, noisy_indices, cmap_txt):
        all_keys = list(latents.keys())
        N = len(all_keys)
        emb0 = latents[all_keys[0]][0].cpu().numpy().ravel()
        d = emb0.size

        emb_first = np.zeros((N, d))
        true_labels = np.zeros(N, dtype=int)
        is_noisy = np.zeros(N, dtype=bool)

        for i, key in enumerate(all_keys):
            idx = latents_indices[key]
            emb = latents[key][0].cpu().numpy().ravel()
            emb_first[i] = emb
            true_labels[i] = self.train_noise_adder.noisy_labels[idx]
            is_noisy[i] = (idx in noisy_indices)

        # Compute t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        emb2d = tsne.fit_transform(emb_first)

        # Create clean plot
        plt.figure(figsize=(6, 6), dpi=300)
        cmap = plt.get_cmap(cmap_txt)

        # Base scatter: colored by class
        plt.scatter(
            emb2d[:, 0], emb2d[:, 1],
            c=true_labels, cmap=cmap, s=15, alpha=0.8, linewidth=0
        )

        # Overlay noisy circles
        plt.scatter(
            emb2d[is_noisy, 0], emb2d[is_noisy, 1],
            facecolors='none', edgecolors='black',
            s=30, linewidths=0.5, alpha=0.4
        )

        # Remove all axis elements
        plt.gca().set_axis_off()

        # Save to PDF (vectorized)
        plt.tight_layout(pad=0)
        plt.savefig("tsne_latent.pdf", format='pdf', bbox_inches='tight')
        plt.close()
        
    def analyze_latent(self, fold, train_indices, val_indices, cmap_txt):
        print(f'analyzing latent space for big fold {fold + 1}')
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
                                       siamese_middle_size=self.siamese_middle_size)

        test_dataset = DatasetSingle(val_subset, transform=self.transform)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
        latents = noise_detector.analyze_latent(test_loader)
        latents_indices = self.custom_kfold_splitter.get_original_indices_as_dic(fold, latents.keys())
        noisy_indices = set(self.train_noise_adder.noisy_indices)
        self.plot_latent_analysis(latents, latents_indices, noisy_indices, cmap_txt)