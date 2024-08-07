import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

class Tester:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.wrong_indices = []
        self.wrong_predictions = []

    def test(self):
        self.model.to(self.device)
        self.model.eval()
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for img1, img2, label1, label2, i, j in tqdm(self.dataloader, desc="Testing"):
                img1, img2, label1, label2 = img1.to(self.device), img2.to(self.device), label1.to(self.device), label2.to(self.device)
                
                emb1, emb2, class1, class2 = self.model(img1, img2)
                
                _, pred1 = torch.max(class1, 1)
                _, pred2 = torch.max(class2, 1)
                
                all_labels.extend(label1.cpu().numpy())
                all_labels.extend(label2.cpu().numpy())
                all_predictions.extend(pred1.cpu().numpy())
                all_predictions.extend(pred2.cpu().numpy())

                # Store wrong predictions and indices
                for idx, (p1, l1) in enumerate(zip(pred1, label1)):
                    if p1 != l1:
                        self.wrong_indices.append(i[idx].item())
                        self.wrong_predictions.append((p1.item(), l1.item()))

                for idx, (p2, l2) in enumerate(zip(pred2, label2)):
                    if p2 != l2:
                        self.wrong_indices.append(j[idx].item())
                        self.wrong_predictions.append((p2.item(), l2.item()))

        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print(f"Test Precision: {precision:.2f}")
        print(f"Test Recall: {recall:.2f}")
        print(f"Test F1 Score: {f1:.2f}")

        return accuracy, precision, recall, f1

    def get_wrong_predictions(self):
        """
        Returns the indices where the predictions were wrong along with their wrong predictions.
        """
        return self.wrong_indices, self.wrong_predictions