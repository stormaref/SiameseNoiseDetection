import torch
import torch.functional as F
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from models.preact import *
from models.cnn import CustomCNN
from models.dla import DLA

class SiameseNetwork(nn.Module):
    def __init__(self, num_classes=10, model='resnet18', embedding_dimension=128, pre_trained=True, dropout_prob=0.5, trainable=True):
        super(SiameseNetwork, self).__init__()
        cnn_output = -1
        if model == 'resnet18':
            cnn_output = 512
            base_model = resnet18(weights=ResNet18_Weights.DEFAULT if pre_trained else None)
        elif model == 'preact-resnet18':
            cnn_output = 512
            if pre_trained:
                raise ValueError('Pre-trained weights are not available for PreActResNet18.')
            else:
                base_model = PreActResNet18()
        elif model == 'preact-resnet34':
            cnn_output = 512
            if pre_trained:
                raise ValueError('Pre-trained weights are not available for PreActResNet18.')
            else:
                base_model = PreActResNet34()
        elif model == 'resnet34':
            cnn_output = 512
            base_model = resnet34(weights=ResNet34_Weights.DEFAULT if pre_trained else None)
        elif model == 'resnet50':
            cnn_output = 2048
            base_model = resnet50(weights=ResNet50_Weights.DEFAULT if pre_trained else None)
        elif model == 'dla':
            cnn_output = 512
            base_model = DLA()
        else:
            raise ValueError('Model not supported')

        # Set whether the ResNet model is trainable or not
        if not trainable:
            for param in base_model.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p=dropout_prob)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.fc_embedding = nn.Linear(cnn_output, embedding_dimension)
        self.fc_classifier = nn.Linear(embedding_dimension, num_classes)  # Classifier layer

    def forward_once(self, input):
        feat = self.feature_extractor(input)
        feat = feat.view(feat.size(0), -1)
        emb = torch.sigmoid(self.fc_embedding(feat))
        emb = self.dropout(emb)
        return emb, self.fc_classifier(emb)
        
    def forward(self, input1, input2):
        emb1, class1 = self.forward_once(input1)        
        emb2, class2 = self.forward_once(input2)
        return emb1, emb2, class1, class2
    
    def classify(self, input):
        emb, cls = self.forward_once(input)
        return emb, cls

    def extract_features(self, input):
        with torch.no_grad():
            features = self.feature_extractor(input)
            features = features.view(features.size(0), -1)
            features = self.fc_embedding(features)
        return features

    
class SimpleSiamese(nn.Module):
    def __init__(self, num_classes=10, model='resnet18', embedding_dimension=128, pre_trained=True, dropout_prob=0.5):
        super(SimpleSiamese, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),#out -> b, 8, 5, 5
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.lin = nn.Sequential(
            nn.Linear(512, 200),
            nn.ReLU(),
            nn.Linear(200, num_classes),
            nn.Sigmoid()
        )
        
    def extract_features(self, x):
        self.lin.eval()
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(x)
            z = z.view(x.size()[0], -1)
        self.lin.train()
        self.encoder.train()
        return self.lin(z)
    
    def forward(self, inp1, inp2):
        z1 = self.extract_features(inp1)
        z2 = self.extract_features(inp2)

        # Get predicted class indices
        y1_indices = torch.argmax(z1, dim=1)
        y2_indices = torch.argmax(z2, dim=1)

        # Convert indices to one-hot encoded vectors
        y1_onehot = F.one_hot(y1_indices, num_classes=z1.size(1)).float()
        y2_onehot = F.one_hot(y2_indices, num_classes=z2.size(1)).float()
        
        return z1, z2, y1_onehot, y2_onehot