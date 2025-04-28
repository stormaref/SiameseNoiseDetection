import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    """Custom Convolutional Neural Network for feature extraction.
    
    A simple CNN architecture with two convolutional blocks followed by a fully-connected 
    classifier that outputs embeddings of specified dimension.
    """
    def __init__(self, input_channels=3, embedding_dim=256):
        """Initialize the CNN with configurable input channels and embedding dimension.
        
        Args:
            input_channels: Number of input channels (3 for RGB images, 1 for grayscale)
            embedding_dim: Dimension of the output embedding vector
        """
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(8192, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        """Forward pass through the network to generate embeddings.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Embedding vectors of shape [batch_size, embedding_dim]
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = self.classifier(x)
        return x
