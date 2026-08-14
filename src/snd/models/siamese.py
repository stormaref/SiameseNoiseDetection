import torch
import torch.functional as F
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from snd.models.preact import *
import timm


def initialize_weights(m):
    """Initialize neural network weights using Kaiming initialization."""
    if isinstance(m, nn.Linear):
        # Use Kaiming initialization for linear layers
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class SiameseNetwork(nn.Module):
    """Siamese neural network architecture for comparing image pairs.

    Uses various backbone architectures (ResNet, PreAct-ResNet, DLA, EfficientNetV2)
    as feature extractors followed by embedding and classification layers.
    """

    def __init__(self, num_classes=10, model='resnet18', embedding_dimension=128, pre_trained=True, dropout_prob=0.5, trainable=True,
                 cnn_size=None, middle_size: int = None, parallel: bool = False):
        """Initialize Siamese network with configurable backbone and embedding dimensions."""
        super(SiameseNetwork, self).__init__()
        self.parallel = parallel
        cnn_output = -1
        if model == 'resnet18':
            cnn_output = 512
            base_model = resnet18(
                weights=ResNet18_Weights.DEFAULT if pre_trained else None)
        elif model == 'preact-resnet18':
            cnn_output = 512
            if pre_trained:
                raise ValueError(
                    'Pre-trained weights are not available for PreActResNet18.')
            else:
                base_model = PreActResNet18(num_class=num_classes)
        elif model == 'preact-resnet34':
            cnn_output = 512
            if pre_trained:
                raise ValueError(
                    'Pre-trained weights are not available for PreActResNet34.')
            else:
                base_model = PreActResNet34(num_class=num_classes)
        elif model == 'preact-resnet50':
            cnn_output = 2048
            if pre_trained:
                raise ValueError(
                    'Pre-trained weights are not available for PreActResNet50.')
            else:
                base_model = PreActResNet50(num_class=num_classes)
        elif model == 'resnet34':
            cnn_output = 512
            base_model = resnet34(
                weights=ResNet34_Weights.DEFAULT if pre_trained else None)
        elif model == 'resnet50':
            cnn_output = 2048
            base_model = resnet50(
                weights=ResNet50_Weights.DEFAULT if pre_trained else None)
        elif model == 'resnet101':
            cnn_output = 2048
            base_model = resnet101(
                weights=ResNet101_Weights.DEFAULT if pre_trained else None)
        elif model == 'efficientnetv2':
            cnn_output = 1792
            base_model = timm.create_model(
                'efficientnetv2_rw_s.ra2_in1k',
                pretrained=True,
            )
        elif model == 'custom':
            cnn_output = 256
            base_model = nn.Sequential(
                # out ->  b, 16, 14, 14
                nn.Conv2d(3, 32, 3, stride=1, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # out -> b, 16, 8, 8

                nn.Conv2d(32, 64, 3, stride=1, padding=1),  # out -> b, 8, 8, 8
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2,
                             padding=1),  # out -> b, 8, 5, 5
                nn.Flatten(),

                nn.Linear(5184, 256),
                nn.ReLU(),
            )
        else:
            raise ValueError('Model not supported')

        if cnn_size != None:
            cnn_output = cnn_size

        # self.dropout = nn.Dropout(p=dropout_prob)
        if model == 'custom':
            self.feature_extractor = base_model
        else:
            if hasattr(base_model, 'fc'):
                base_model.fc = nn.Flatten()
                self.feature_extractor = base_model
            else:
                self.feature_extractor = nn.Sequential(
                    *list(base_model.children())[:-1])

        # Set whether the ResNet model is trainable or not
        if not trainable:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # self.fc_embedding = nn.Linear(cnn_output, embedding_dimension)
        layer1 = int(cnn_output / 2)
        layer2 = int(layer1 / 2)
        layer3 = int(layer2 / 2)
        self.fc_embedding = nn.Sequential(
            nn.Linear(cnn_output, layer1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(layer1, layer2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(layer2, embedding_dimension),
            nn.Sigmoid()
        )

        if middle_size != None and middle_size > 0:
            middle1 = middle_size
        elif parallel:
            middle1 = int(cnn_output // 3)
        else:
            middle1 = int(embedding_dimension / 3)
        
        if parallel:
            self.fc_classifier = nn.Sequential(
                nn.Linear(cnn_output, middle1),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(middle1, num_classes),
            )
        else:
            self.fc_classifier = nn.Sequential(
                nn.Linear(embedding_dimension, middle1),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(middle1, num_classes),
            )

        self.apply(initialize_weights)

    def forward_once(self, input):
        """Process a single input through the network to get embedding and class prediction."""
        feat = self.feature_extractor(input)
        emb = self.fc_embedding(feat)
        cls = self.fc_classifier(emb)
        return emb, cls

    def parallel_forward_once(self, input):
        """Process a single input through the network to get embedding and class prediction."""
        feat = self.feature_extractor(input)
        emb = self.fc_embedding(feat)
        cls = self.fc_classifier(feat)
        return emb, cls

    def forward(self, input1, input2):
        """Process a pair of inputs through the siamese network."""
        if self.parallel:
            emb1, class1 = self.parallel_forward_once(input1)
            emb2, class2 = self.parallel_forward_once(input2)
        else:
            emb1, class1 = self.forward_once(input1)
            emb2, class2 = self.forward_once(input2)
        return emb1, emb2, class1, class2

    def classify(self, input):
        """Classify a single input and return its embedding and class prediction."""
        if self.parallel:
            emb, cls = self.parallel_forward_once(input)
        else:
            emb, cls = self.forward_once(input)
        return emb, cls

    def extract_features(self, input):
        """Extract features from input without gradient computation (for inference)."""
        with torch.no_grad():
            features = self.feature_extractor(input)
            features = features.view(features.size(0), -1)
            features = self.fc_embedding(features)
        return features

    def freeze_classifier(self):
        """Freeze parameters in classifier layers to prevent updating during training."""
        for param in self.fc_classifier.parameters():
            param.requires_grad = False

    def unfreeze_classifier(self):
        """Unfreeze parameters in classifier layers to allow updating during training."""
        for param in self.fc_classifier.parameters():
            param.requires_grad = True

    def freeze_feature_extractor(self):
        """Freeze parameters in feature extractor and embedding layers."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.fc_embedding.parameters():
            param.requires_grad = False
