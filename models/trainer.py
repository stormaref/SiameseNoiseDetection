from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, contrastive_criterion, classifier_criterion, optimizer, dataloader, device):
        self.model = model
        self.contrastive_criterion = contrastive_criterion
        self.classifier_criterion = classifier_criterion
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.epoch_losses = []

    def train(self, num_epochs):
        self.model.to(self.device)
        self.model.train()
        progress_bar = tqdm(range(num_epochs))
        for epoch in progress_bar:
            progress_bar.set_description(f'Epoch {epoch}/{num_epochs}')
            epoch_loss = 0
            # progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for img1, img2, label1, label2, i, j in self.dataloader:
                img1, img2, label1, label2 = img1.to(self.device), img2.to(self.device), label1.to(self.device), label2.to(self.device)
                
                self.optimizer.zero_grad()
                
                emb1, emb2, class1, class2 = self.model(img1, img2)
                
                # Calculate same_label dynamically
                same_label = (label1 == label2).float()
                
                # Calculate losses
                contrastive_loss = self.contrastive_criterion(emb1, emb2, same_label)
                classifier_loss1 = self.classifier_criterion(class1, label1)
                classifier_loss2 = self.classifier_criterion(class2, label2)
                total_loss = contrastive_loss + classifier_loss1 + classifier_loss2
                
                total_loss.backward()
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
            avg_epoch_loss = epoch_loss / len(self.dataloader)
            self.epoch_losses.append(avg_epoch_loss)
            
            progress_bar.set_postfix(loss=avg_epoch_loss)
            # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss}")

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.epoch_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.show()
