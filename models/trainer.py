import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.ols import OnlineLabelSmoothing
from models.siamese import SiameseNetwork

class Trainer:
    def __init__(self, model: SiameseNetwork, contrastive_criterion, classifier_criterion, optimizer, dataloader, device, contrastive_ratio,
                 val_dataloader=None, patience=5, checkpoint_path='best_model.pth', freeze_epoch=10):
        self.model = model
        self.contrastive_criterion = contrastive_criterion
        self.classifier_criterion = classifier_criterion
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.epoch_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.patience = patience
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0
        self.early_stop = False
        self.epochs_no_improve = 0
        self.checkpoint_path = checkpoint_path
        self.contrastive_ratio = contrastive_ratio
        self.freeze_epoch = freeze_epoch
        
        
    def calc_loss(self, img1, img2, label1, label2, epoch):
        emb1, emb2, class1, class2 = self.model(img1, img2)
        same_label = (label1 == label2).float()
        
        # Calculate losses
        contrastive_loss = self.contrastive_criterion(emb1, emb2, same_label)
        classifier_loss1 = self.classifier_criterion(class1, label1)
        classifier_loss2 = self.classifier_criterion(class2, label2)
        
        if self.freeze_epoch == None:
            total_loss = classifier_loss1 + classifier_loss2 + contrastive_loss
            return total_loss, class1, class2
        if epoch < self.freeze_epoch:
            total_loss = classifier_loss1 + classifier_loss2
        else:
            total_loss = classifier_loss1 + classifier_loss2 + contrastive_loss
        # total_loss = classifier_loss1 + classifier_loss2
        return total_loss, class1, class2
        
    def train(self, num_epochs, normal_optimizer=True):
        self.model.to(self.device)
        self.model.train()
        progress_bar = tqdm(range(num_epochs))
        for epoch in progress_bar:
            correct = 0
            total = 0
            if self.early_stop:
                print("Early stopping triggered")
                break
            
            progress_bar.set_description(f'Epoch {epoch}/{num_epochs}')
            epoch_loss = 0
            
            if self.freeze_epoch != None and epoch == self.freeze_epoch:
                self.model.freeze_feature_extractor()
                print('freezed')

            for img1, img2, label1, label2, i, j in self.dataloader:
                img1, img2, label1, label2 = img1.to(self.device), img2.to(self.device), label1.to(self.device), label2.to(self.device)
                
                if normal_optimizer:
                    self.optimizer.zero_grad()
                
                total_loss, class1, class2 = self.calc_loss(img1, img2, label1, label2, epoch)
                total_loss.backward()
                
                if normal_optimizer:
                    self.optimizer.step()
                else:
                    self.optimizer.first_step(zero_grad=True)
                    
                    total_loss, class1, class2 = self.calc_loss(img1, img2, label1, label2, epoch)
                    total_loss.backward()
                    
                    self.optimizer.second_step(zero_grad=True)
                
                epoch_loss += total_loss.item()
                
                _, pred1 = torch.max(class1, 1)
                _, pred2 = torch.max(class2, 1)
                correct += (pred1 == label1).sum().item() + (pred2 == label2).sum().item()
                total += label1.size(0) + label2.size(0)
            
            avg_epoch_loss = epoch_loss / len(self.dataloader)
            self.epoch_losses.append(avg_epoch_loss)
            epoch_accuracy = 100 * correct / total
            self.train_accuracies.append(epoch_accuracy)
            
            # progress_bar.set_postfix(loss=avg_epoch_loss)
            
            # Validation
            if self.val_dataloader:
                val_loss, val_accuracy = self.validate(epoch)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)
                
                # Save the best model based on validation accuracy
                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    # print(f"Best model saved with accuracy: {val_accuracy:.2f}%")
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.patience:
                        self.early_stop = True
            
            # Print Epoch Summary
            # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
            if self.val_dataloader:
                # print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
                progress_bar.set_postfix({'val_loss': val_loss, 'val_accuracy': val_accuracy, 'train_loss': avg_epoch_loss, 
                                          'best_accuracy': self.best_val_accuracy})                


            if isinstance(self.classifier_criterion, OnlineLabelSmoothing):
                self.classifier_criterion.next_epoch()

        # Load the best model at the end of training
        if self.val_dataloader:
            print("Loading best model from checkpoint...")
            self.model.load_state_dict(torch.load(self.checkpoint_path))

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for img1, img2, label1, label2, i, j in self.val_dataloader:
                img1, img2, label1, label2 = img1.to(self.device), img2.to(self.device), label1.to(self.device), label2.to(self.device)
                
                total_loss, class1, class2 = self.calc_loss(img1, img2, label1, label2, epoch)
                
                # emb1, emb2, class1, class2 = self.model(img1, img2)

                # # Calculate same_label dynamically
                # same_label = (label1 == label2).float()
                
                # # Calculate losses
                # contrastive_loss = self.contrastive_criterion(emb1, emb2, same_label)
                # classifier_loss1 = self.classifier_criterion(class1, label1)
                # classifier_loss2 = self.classifier_criterion(class2, label2)
                
                # total_loss = contrastive_loss + classifier_loss1 + classifier_loss2
                
                val_loss += total_loss.item()
                
                # Calculate accuracy
                _, pred1 = torch.max(class1, 1)
                _, pred2 = torch.max(class2, 1)
                correct += (pred1 == label1).sum().item() + (pred2 == label2).sum().item()
                total += label1.size(0) + label2.size(0)
        
        avg_val_loss = val_loss / len(self.val_dataloader)
        accuracy = 100 * correct / total
        return avg_val_loss, accuracy
    
    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.epoch_losses[10:], label='Training Loss')
        if self.val_dataloader:
            plt.plot(self.val_losses[10:], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.show()
        
    def plot_accuracies(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_accuracies, label='Training Accuracy')
        if self.val_dataloader:
            plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Accuracy Over Epochs')
        plt.legend()
        plt.show()