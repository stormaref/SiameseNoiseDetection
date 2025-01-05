import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from models.dataset import CleanDatasetLoader, CleanWrapperDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

class CIFAR10FinalModelTester:
    def __init__(self, train_dataset_path: str, train_transform: transforms.transforms, test_transform: transforms.transforms, 
                 train_batch_size=256, val_batch_size=256, test_batch_size=64, pretrained=True, lr=0.001, patience=5):
        
        self.pretrained = pretrained
        self.cleaned_dataset = CleanDatasetLoader(train_dataset_path, None)
        
        train_indices, val_indices = train_test_split(
            range(len(self.cleaned_dataset)),
            test_size=0.1,
        )

        self.train_dataset = CleanWrapperDataset(self.cleaned_dataset, train_indices, train_transform)
        self.train_loader = DataLoader(self.cleaned_dataset, batch_size=train_batch_size)
        
        self.val_dataset = CleanWrapperDataset(self.cleaned_dataset, val_indices, test_transform)
        self.val_loader = DataLoader(self.cleaned_dataset, batch_size=val_batch_size)
        
        self.test_dataset = CIFAR10(root='data', train=False, download=True, transform=test_transform)
        self.test_loader = DataLoader(self.test_dataset, test_batch_size)
        
        self.model = self.get_model()
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.patience = patience
        self.best_val_accuracy = 0
        self.early_stop_counter = 0
        
    def get_model(self):
        base = resnet18(weights=ResNet18_Weights.DEFAULT) if self.pretrained else resnet18()
        base.fc = torch.nn.Linear(base.fc.in_features, 10)
        return base
    
    def train(self, epochs):
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            for data in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs, labels = data
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            accuracy = correct / total
            
            self.train_losses.append(avg_epoch_loss)
            self.train_accuracies.append(accuracy)
            tqdm.set_postfix_str(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_epoch_loss}, Training Accuracy: {accuracy}')
            
            val_loss, val_accuracy = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            tqdm.set_postfix_str(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
            
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            if self.early_stop_counter >= self.patience:
                tqdm.write("Early stopping triggered")
                break
                
    def validate(self):
        epoch_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.val_loader:
                inputs, labels = data
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_epoch_loss = epoch_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_epoch_loss, accuracy
    
    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy}')
        
    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Training Accuracy')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy')
        
        plt.show()