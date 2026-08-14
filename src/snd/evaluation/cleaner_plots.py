"""Figures for the cleaner's detection and relabeling results.

The numbers these plot come from :mod:`snd.evaluation.cleaner_metrics`. Mixed into
``NoiseCleaner`` via :class:`~snd.evaluation.cleaner_report.CleanerReportingMixin`.
"""
import math
import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import auc
from torch.utils.data import DataLoader, Subset

from snd.data.dataset import DatasetSingle
from snd.models.siamese import SiameseNetwork
from snd.pipeline.detector import NoiseDetector


class CleanerPlotsMixin:
    """Matplotlib output for the cleaner's analyses."""

    def plot_relabeling_analysis(self, relabeling_results, title):
        """Plot heatmap of relabeling analysis results."""
        num_mistakes, num_thresholds = relabeling_results.shape

        plt.figure(figsize=(10, 6))
        c = plt.imshow(relabeling_results, cmap='viridis', aspect='auto')

        plt.colorbar(c, label='Relabeling Metric')

        plt.xlabel('Relabeling Thresholds', fontsize=12)
        plt.ylabel('Number of Mistakes', fontsize=12)
        plt.title(f'Relabeling {title} Heatmap', fontsize=14)

        plt.xticks(ticks=np.arange(num_thresholds), labels=[
                   f'Thresh {i+self.relabeling_range.start}' for i in range(num_thresholds)])
        plt.yticks(ticks=np.arange(num_mistakes), labels=[
                   f'{i+1}' for i in range(num_mistakes)])

        for i in range(num_mistakes):
            for j in range(num_thresholds):
                plt.text(j, i, f'{relabeling_results[i, j]:.2f}',
                         ha='center', va='center', color='white', fontsize=8)

        plt.tight_layout()
        plt.show()

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
            ax.plot([root[0], node[0]], [root[1], node[1]],
                    color='black', lw=2, zorder=1)

            # Compute midpoint for branch label and add a background box
            mid_x = (root[0] + node[0]) / 2
            mid_y = (root[1] + node[1]) / 2
            ax.text(mid_x, mid_y, label, fontsize=14, ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', pad=1))

            # Draw the node as a larger circle with an edge
            circle = patches.Circle(
                node, 0.06, edgecolor='black', facecolor=color, lw=2, zorder=2)
            ax.add_patch(circle)

            # Annotate the node with the corresponding report value
            ax.text(node[0], node[1] - 0.10, str(report[label]),
                    fontsize=14, ha='center', va='top', fontweight='bold')

        # Highlight the root node with a special color (gold) and add the overall score
        root_circle = patches.Circle(
            root, 0.04, edgecolor='black', facecolor='gold', lw=2, zorder=3)
        ax.add_patch(root_circle)
        ax.text(root[0], root[1] + 0.07, "Score", fontsize=16,
                ha='center', va='bottom', fontweight='bold')
        ax.text(root[0], root[1] - 0.07, str(score), fontsize=16,
                ha='center', va='top', fontweight='bold')

        # Set title and adjust finite parameters for a cleaner look
        plt.title(
            f"Report Tree Diagram (Overall Score: {score})", fontsize=18, fontweight='bold')
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
            # Plot each point
            plt.plot(fpr, tpr, marker='o', label=f'Mistakes Count: {i + 1}')
            plt.annotate(f"{i + 1}", (fpr, tpr), textcoords="offset points",
                         xytext=(5, -5), ha='center')  # Annotate with mistakes_count

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
        # The position in the array represents the number of mistakes.
        mistake_counts = range(1, len(relabeling_accuracies) + 1)
        bar_width = 0.35  # Width of each bar
        x = np.arange(len(mistake_counts))  # X positions for the groups

        plt.figure(figsize=(10, 6))
        plt.bar(x - bar_width / 2, relabeling_accuracies,
                width=bar_width, label='Accuracy', color='blue')
        plt.bar(x + bar_width / 2, relabel_ratios, width=bar_width,
                label='Relabel Ratio', color='orange')

        plt.xlabel('Mistake Count', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Relabeling Accuracy and Ratio by Mistake Count', fontsize=14)
        # Label x-axis with mistake counts
        plt.xticks(x, labels=mistake_counts)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        for i, (accuracy, ratio) in enumerate(zip(relabeling_accuracies, relabel_ratios)):
            plt.text(i - bar_width / 2, accuracy + 0.02,
                     f'{accuracy * 100:.2f}', ha='center', fontsize=9)
            plt.text(i + bar_width / 2, ratio + 0.02,
                     f'{ratio * 100:.2f}', ha='center', fontsize=9)

        plt.tight_layout()
        plt.show()

    def plot_before_after(self, correct, all):
        before_noisy = len(
            self.train_noise_adder.noisy_indices) if self.noise_type != 'none' else 0
        before_clean = len(self.dataset) - before_noisy
        after_noisy = all - correct
        after_clean = correct
        labels = ['Before', 'After']
        noisy = [before_noisy, after_noisy]
        clean = [before_clean, after_clean]
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 8))  # Increased height of the plot
        rects1 = ax.bar(x - width/2, noisy, width,
                        label='Noisy', color='tomato')
        rects2 = ax.bar(x + width/2, clean, width,
                        label='Clean', color='skyblue')
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
            plt.title(
                f"I:{dataset_idx},R:{labels[label]},P:{labels[pred] if pred!=-1 else 'Unknown'}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

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
        plt.plot(noisy_keys, noisy_values, 'o-',
                 label='Noisy', color='red', markersize=8)
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
            true_labels[i] = self.train_noise_adder.noisy_labels[idx] if self.noise_type != 'none' else 0
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

        noise_detector = NoiseDetector(SiameseNetwork, train_subset, self.device, self.config,
                                       num_folds=self.inner_folds_num,
                                       model_save_path=self.model_save_path,
                                       prediction_path=self.prediction_path)

        test_dataset = DatasetSingle(val_subset, transform=self.transform)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
        latents = noise_detector.analyze_latent(test_loader)
        latents_indices = self.custom_kfold_splitter.get_original_indices_as_dic(
            fold, latents.keys())
        noisy_indices = set(
            self.train_noise_adder.noisy_indices) if self.noise_type != 'none' else set()
        self.plot_latent_analysis(
            latents, latents_indices, noisy_indices, cmap_txt)
