import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.resnet import Resnet
import importlib
import models.dataset
importlib.reload(models.dataset)
from models.dataset import CustomDataset
import numpy as np
from torch.utils.data import Dataset

class Predictor:
    def __init__(self, train_dataset, test_dataset, transform, num_classes=10, model_type='resnet18', batch_size=256, num_epochs=50, device=None):
        self.train_dataset = CustomDataset(train_dataset, transform)
        self.test_dataset = CustomDataset(test_dataset, transform)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # DataLoader for training and testing
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize the custom Resnet model
        self.model = Resnet(num_classes=num_classes, model=model_type).to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), momentum=0.9, lr=0.001)

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for inputs, labels in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}/{self.num_epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {running_loss/len(self.train_loader)}")
    
    def evaluate(self):
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

    def run(self):
        self.train()
        return self.evaluate()
    
