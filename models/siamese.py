import torch
import torch.functional as F
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torchvision.models import vgg19_bn, VGG19_BN_Weights
from models.preact import *
from models.cnn import CustomCNN
from models.dla import DLA
# from torchsummary import summary

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # Use Kaiming initialization for linear layers
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
class SiameseNetwork(nn.Module):
    def __init__(self, num_classes=10, model='resnet18', embedding_dimension=128, pre_trained=True, dropout_prob=0.5, trainable=True,
                 cnn_size=None):
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
                base_model = PreActResNet18(num_class=num_classes)
        elif model == 'preact-resnet34':
            cnn_output = 512
            if pre_trained:
                raise ValueError('Pre-trained weights are not available for PreActResNet34.')
            else:
                base_model = PreActResNet34(num_class=num_classes)
        elif model == 'preact-resnet50':
            cnn_output = 2048
            if pre_trained:
                raise ValueError('Pre-trained weights are not available for PreActResNet50.')
            else:
                base_model = PreActResNet50(num_class=num_classes)
        elif model == 'wresnet50':
            cnn_output = 2048
            if pre_trained:
                raise ValueError('Pre-trained weights are not available for WideResNet50.')
            else:
                base_model = wide_resnet50_2()
        elif model == 'resnet34':
            cnn_output = 512
            base_model = resnet34(weights=ResNet34_Weights.DEFAULT if pre_trained else None)
        elif model == 'resnet50':
            cnn_output = 2048
            base_model = resnet50(weights=ResNet50_Weights.DEFAULT if pre_trained else None)
        elif model == 'resnet101':
            cnn_output = 2048
            base_model = resnet101(weights=ResNet101_Weights.DEFAULT if pre_trained else None)
        elif model == 'dla':
            cnn_output = 512
            base_model = DLA()
        elif model == 'vgg16-bn':
            cnn_output = 4096
            base_model = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT if pre_trained else None)
        elif model == 'vgg19-bn':
            cnn_output = 4096
            base_model = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT if pre_trained else None)
        elif model == 'custom':
            cnn_output = 256
            base_model = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=1, padding=1),  #out ->  b, 16, 14, 14
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2),  #out -> b, 16, 8, 8
                
                nn.Conv2d(32, 64, 3, stride=1, padding=1),  #out -> b, 8, 8, 8
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  #out -> b, 8, 5, 5
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
        elif model.__contains__('vgg'):
            base_model.classifier = base_model.classifier[:-1]
            self.feature_extractor = base_model
        else:
            if hasattr(base_model, 'fc'):
                base_model.fc = nn.Flatten()
                self.feature_extractor = base_model
            else:
                self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
                
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

        middle1 = int(embedding_dimension / 3)
        middle2 = int(middle1 / 2)
        self.fc_classifier = nn.Sequential(
            nn.Linear(embedding_dimension, middle1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(middle1, num_classes),
            # nn.ReLU(),
            # nn.Dropout(dropout_prob),
            # nn.Linear(middle2, num_classes)
            )
        
        self.apply(initialize_weights)

    def forward_once(self, input):
        feat = self.feature_extractor(input)
        emb = self.fc_embedding(feat)
        # emb = F.normalize(emb, dim=1) 
        # emb = F.sigmoid(emb)
        cls = self.fc_classifier(emb)
        return emb, cls
        
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
    
    def freeze_classifier(self):
        for param in self.fc_classifier.parameters():
            param.requires_grad = False
            
    def unfreeze_classifier(self):
        for param in self.fc_classifier.parameters():
            param.requires_grad = True

    def freeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.fc_embedding.parameters():
            param.requires_grad = False
    
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