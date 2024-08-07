import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights

class SiameseNetwork(nn.Module):
    def __init__(self, num_classes=10, model='resnet18'):
        super(SiameseNetwork, self).__init__()
        if model == 'resnet18':
            base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model == 'resnet50':
            base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise ValueError('Model not supported')
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.fc_embedding = nn.Linear(512, 128)  # Embedding layer for Siamese Network
        self.fc_classifier = nn.Linear(128, num_classes)  # Classifier layer

    def forward(self, input1, input2):
        # Feature extraction
        feat1 = self.feature_extractor(input1)
        feat2 = self.feature_extractor(input2)

        # Flatten feature maps
        feat1 = feat1.view(feat1.size(0), -1)
        feat2 = feat2.view(feat2.size(0), -1)

        # Embedding
        emb1 = self.fc_embedding(feat1)
        emb2 = self.fc_embedding(feat2)

        # Classification
        class1 = self.fc_classifier(emb1)
        class2 = self.fc_classifier(emb2)

        return emb1, emb2, class1, class2

    def extract_features(self, input):
        with torch.no_grad():
            features = self.feature_extractor(input)
            features = features.view(features.size(0), -1)
            features = self.fc_embedding(features)
        return features