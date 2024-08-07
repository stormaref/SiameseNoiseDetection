import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

class EmbeddingVisualizer:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def extract_embeddings(self):
        self.model.to(self.device)
        self.model.eval()
        embeddings = []
        predcitions = []
        labels = []
        indices = []
        wrong_images = []

        with torch.no_grad():
            for img1, img2, label1, label2, i, j in tqdm(self.dataloader, desc="Extracting Embeddings"):
                img1, img2, label1, label2 = img1.to(self.device), img2.to(self.device), label1.to(self.device), label2.to(self.device)
                
                emb1, emb2, class1, class2 = self.model(img1, img2)
                
                _, pred1 = torch.max(class1, 1)
                _, pred2 = torch.max(class2, 1)
                
                embeddings.append(emb1.cpu().numpy())
                embeddings.append(emb2.cpu().numpy())
                
                labels.append(label1.cpu().numpy())
                labels.append(label2.cpu().numpy())
                
                predcitions.append(pred1.cpu().numpy())                
                predcitions.append(pred2.cpu().numpy())
                
                indices.extend(i.cpu().numpy())
                indices.extend(j.cpu().numpy())


                # Collect wrong images
                for k in range(label1.size(0)):
                    if label1[k].item() != class1[k].argmax().item():
                        wrong_images.append((img1[k].cpu(), label1[k].cpu(), class1[k].argmax().cpu(), i))
                    if label2[k].item() != class2[k].argmax().item():
                        wrong_images.append((img2[k].cpu(), label2[k].cpu(), class2[k].argmax().cpu(), j))

        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)
        return embeddings, labels, predcitions, indices, wrong_images

    def visualize(self, embeddings, real_labels, predicted_labels):
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(embeddings)

        plt.figure(figsize=(20, 8))

        # Plot real labels
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=real_labels, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, ticks=range(10))
        plt.title('t-SNE visualization with Real Labels')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

        # Plot predicted labels
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=predicted_labels, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, ticks=range(10))
        plt.title('t-SNE visualization with Predicted Labels')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

        plt.show()


    def plot_incorrect_images(self, incorrect_images, class_names):
        incorrect_images = [(img.squeeze(0), real, pred) for img, real, pred, idx in incorrect_images if real.item() != pred.item()]
        num_images = len(incorrect_images)
        print(num_images)
        plt.figure(figsize=(20, 20))

        for j in range(int(num_images/25)):
            plt.figure(figsize=(20, 20))
            plt.title(f'{j}')
            for i, (img, real, pred) in enumerate(incorrect_images[j*25:(j+1)*25]):
                plt.subplot(5, 5, i + 1)
                plt.imshow(img.permute(1, 2, 0).numpy())
                plt.title(f"Real: {class_names[real.item()]}, Pred: {class_names[pred.item()]}")
                plt.axis('off')

            plt.show()