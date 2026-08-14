"""Ground-truth scoring of the cleaner's detection and relabeling decisions.

Everything here reads the per-fold prediction CSVs plus the noise adder's ground
truth and returns numbers; the figures that visualize them live in
:mod:`snd.evaluation.cleaner_plots`. Mixed into ``NoiseCleaner`` via
:class:`~snd.evaluation.cleaner_report.CleanerReportingMixin`.
"""
import numpy as np


class CleanerMetricsMixin:
    """Detection metrics and the ground-truth-free relabeling score."""

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

    def analyze_with_mistakes_count(self, array, mistakes_count):
        predicted_noise_indices = []
        correct_relabel = 0
        perform_relabel = 0
        all_relabel = 0
        relabeling_analysis = []

        for item in array:
            index = int(item['index'])
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

            result = self.analyze_relabeling(
                mistakes >= mistakes_count, preds, real_label)
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
        detected_count = 0
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

            detected_count += 1
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
                else:
                    if new_label != real_label:
                        score -= 2
                        report['-2'] += 1
            else:
                # No relabeling (removal)
                if is_noisy:
                    score += 1
                    report['1'] += 1
                else:
                    score -= 1
                    report['-1'] += 1

        normalized_score = score / detected_count if detected_count > 0 else 0
        if plot:
            self.plot_relabeling_score_diagram(report, normalized_score)
        return normalized_score, report

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
            tpr, fpr, relabeling_accuracy, relabel_ratio, accuracy_analysis, ratio_analysis = self.analyze_with_mistakes_count(
                array, mistakes_count)

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
                    'score': score,  # already normalized by detected_count in calculate_relabeling_score
                    'report': r_report,
                    'accuracy': r_report['2'] / (r_report['-2'] + r_report['2'] + r_report['0']) * 100,
                    'count': relabled,

                }
                noisy = len(
                    self.train_noise_adder.noisy_indices) if self.noise_type != 'none' else 0
                clean = total - noisy
                clean -= r_report['-1']
                noisy -= r_report['1']

                clean -= r_report['-2']
                noisy += r_report['-2']

                clean += r_report['2']
                noisy -= r_report['2']
                relabeling_metrics['noise_ratio'] = noisy / \
                    (noisy + clean) * 100
                relabeling_metrics['remaining'] = noisy + clean
                relabeling_metrics['clean_after'] = clean
                relabeling_metrics['noisy_after'] = noisy
                metrics['relabeling'].append(relabeling_metrics)

            results.append(metrics)
        return results
