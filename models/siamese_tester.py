import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.contrastive import ContrastiveLoss
from models.dataset import *
import numpy as np

class SiameseTester:
    def __init__(self, train_dataset, model_class, transform, augmented_transform, num_classes=10, dropout_prob=0.5, embedding_dimension=128,
                 pre_trained=True, lr=0.001, weight_decay=1e-5, batch_size=64, device='cuda', patience=15, 
                 checkpoint_path='best_siamese_model.pth'):
        # Initialize parameters
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.embedding_dimension = embedding_dimension
        self.pre_trained = pre_trained
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.device = device
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.model_class = model_class

        # Split dataset into training and validation sets
        train_idx, val_idx = train_test_split(range(len(train_dataset)), stratify=train_dataset.targets, test_size=0.2)
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)


        # Dataloaders
        self.train_loader = DataLoader(DatasetPairs(train_subset, 4000, augmented_transform), batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(DatasetPairs(val_subset, 1000, transform), batch_size=self.batch_size, shuffle=False)

        # Model, optimizer, and loss functions
        self.model = self.model_class(num_classes=self.num_classes, dropout_prob=self.dropout_prob, 
                                      pre_trained=self.pre_trained, embedding_dimension=self.embedding_dimension).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.classifier_criterion = nn.CrossEntropyLoss()
        self.contrastive_criterion = ContrastiveLoss(distance_meter='euclidian')

        # For early stopping
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0
        self.epochs_no_improve = 0
        self.early_stop = False

        # To store metrics
        self.epoch_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def train(self, num_epochs):
        self.model.train()
        progress_bar = tqdm(range(num_epochs))
        for epoch in progress_bar:
            correct = 0
            total = 0
            
            progress_bar.set_description(f'Epoch {epoch+1}/{num_epochs}')
            epoch_loss = 0

            for img1, img2, label1, label2, i, j in self.train_loader:
                img1, img2 = img1.to(self.device), img2.to(self.device)
                label1, label2 = label1.to(self.device), label2.to(self.device)

                self.optimizer.zero_grad()

                emb1, emb2, class1, class2 = self.model(img1, img2)

                # Calculate if the labels are the same
                same_label = (label1 == label2).float()

                # Calculate losses
                contrastive_loss = self.contrastive_criterion(emb1, emb2, same_label)
                classifier_loss1 = self.classifier_criterion(class1, label1)
                classifier_loss2 = self.classifier_criterion(class2, label2)

                # Total loss using plain sum
                total_loss = contrastive_loss + classifier_loss1 + classifier_loss2
                
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += total_loss.item()
                
                _, pred1 = torch.max(class1, 1)
                _, pred2 = torch.max(class2, 1)
                correct += (pred1 == label1).sum().item() + (pred2 == label2).sum().item()
                total += label1.size(0) + label2.size(0)
            
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            self.epoch_losses.append(avg_epoch_loss)
            epoch_accuracy = 100 * correct / total
            self.train_accuracies.append(epoch_accuracy)

            # Validation
            if self.val_loader:
                val_loss, val_accuracy = self.validate()
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)

                # Save the best model based on validation accuracy
                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.patience:
                        print("Early stopping triggered")
                        self.early_stop = True
                        break

            # Update progress bar
            progress_bar.set_postfix({'train_loss': avg_epoch_loss, 'val_loss': val_loss, 'val_accuracy': val_accuracy})
            
        # Load the best model at the end of training
        if self.val_loader:
            print("Loading best model from checkpoint...")
            self.model.load_state_dict(torch.load(self.checkpoint_path))

        # Plot t-SNE visualization after training
        print(f'best accuracy: {self.best_val_accuracy}')
        self.visualize_embeddings()

    def validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for img1, img2, label1, label2, i, j in self.val_loader:
                img1, img2 = img1.to(self.device), img2.to(self.device)
                label1, label2 = label1.to(self.device), label2.to(self.device)

                emb1, emb2, class1, class2 = self.model(img1, img2)

                same_label = (label1 == label2).float()

                contrastive_loss = self.contrastive_criterion(emb1, emb2, same_label)
                classifier_loss1 = self.classifier_criterion(class1, label1)
                classifier_loss2 = self.classifier_criterion(class2, label2)

                total_loss = contrastive_loss + classifier_loss1 + classifier_loss2
                val_loss += total_loss.item()

                _, pred1 = torch.max(class1, 1)
                _, pred2 = torch.max(class2, 1)
                correct += (pred1 == label1).sum().item() + (pred2 == label2).sum().item()
                total += label1.size(0) + label2.size(0)

        avg_val_loss = val_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        return avg_val_loss, accuracy

    def visualize_embeddings(self):
        self.model.eval()
        embeddings = []
        real_labels = []
        predicted_labels = []

        with torch.no_grad():
            for img1, img2, label1, label2, i, j in self.val_loader:
                img1, img2 = img1.to(self.device), img2.to(self.device)

                emb1, emb2, class1, class2 = self.model(img1, img2)

                embeddings.append(emb1.cpu().numpy())
                embeddings.append(emb2.cpu().numpy())

                real_labels.extend(label1.cpu().numpy())
                real_labels.extend(label2.cpu().numpy())

                _, pred1 = torch.max(class1, 1)
                _, pred2 = torch.max(class2, 1)
                predicted_labels.extend(pred1.cpu().numpy())
                predicted_labels.extend(pred2.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        real_labels = np.array(real_labels)
        predicted_labels = np.array(predicted_labels)

        # t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(embeddings)

        plt.figure(figsize=(20, 8))

        # Plot real labels
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=real_labels, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, ticks=range(self.num_classes))
        plt.title('t-SNE visualization with Real Labels')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

        # Plot predicted labels
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=predicted_labels, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, ticks=range(self.num_classes))
        plt.title('t-SNE visualization with Predicted Labels')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

        plt.show()
