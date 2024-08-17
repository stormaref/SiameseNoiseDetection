import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.resnet import Resnet
from models.dataset import CustomDataset
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

class Predictor:
    def __init__(self, train_dataset, test_dataset, transform, num_classes=10, model_type='resnet18', batch_size=256, device=None):
        train_dataset, val_dataset = self.get_train_val_test_datasets(train_dataset)
        self.train_dataset = CustomDataset(train_dataset, transform)
        self.val_dataset = CustomDataset(val_dataset, transform)
        self.test_dataset = CustomDataset(test_dataset, transform)
        self.batch_size = batch_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model = Resnet(num_classes=num_classes, model=model_type).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), momentum=0.9, lr=0.001)

    def get_train_val_test_datasets(self, train_dataset):
        targets = self.get_targets(train_dataset)
        train_idx, val_idx = train_test_split(range(len(train_dataset)), test_size=0.1, stratify=targets)
        train_dataset = Subset(train_dataset, train_idx)
        val_dataset = Subset(train_dataset, val_idx)
        train_dataset = self.reset_subset_indices(train_dataset)
        val_dataset = self.reset_subset_indices(val_dataset)
        return train_dataset, val_dataset

    def reset_subset_indices(self, subset):
        return Subset(subset.dataset, list(range(len(subset))))
    
    def get_targets(self, dataset):
        if hasattr(dataset, 'targets'):
            return dataset.targets
        elif isinstance(dataset, torch.utils.data.Subset):
            return [dataset.dataset.targets[i] for i in dataset.indices]
        else:
            raise ValueError("The dataset does not have 'targets' attribute and is not a Subset with accessible targets.")
        
    def train(self, num_epochs):
        best_val_accuracy = 0.0
        self.model.train()
        for epoch in range(num_epochs):
            # Training
            running_loss = 0.0
            for inputs, labels in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for val_inputs, val_labels in self.val_loader:
                    val_inputs, val_labels = val_inputs.to(self.device), val_labels.to(self.device)
                    val_outputs = self.model(val_inputs)
                    val_loss += self.criterion(val_outputs, val_labels).item()
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_total += val_labels.size(0)
                    val_correct += (val_predicted == val_labels).sum().item()

            val_accuracy = 100 * val_correct / val_total
            print(f"Validation Accuracy: {val_accuracy:.2f}%")

            # Save model if it has the best validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), 'best_model.pth')
                
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(self.train_loader)}")
        self.model.load_state_dict(torch.load('best_model.pth'))
    
    def free_model(self):
        self.model.to('cpu')
        torch.cuda.empty_cache()
    
    def evaluate(self):
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Evaluating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())
        
        # Calculate accuracy
        accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        return accuracy