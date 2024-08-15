import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights

class Resnet(nn.Module):
    def __init__(self, num_classes=10, model='resnet18'):
        super(Resnet, self).__init__()
        if model == 'resnet18':
            base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model == 'resnet34':
            base_model = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif model == 'resnet50':
            base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise ValueError('Model not supported')
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.output = nn.Linear(512, num_classes)

    def forward(self, input):
        out = self.feature_extractor(input)
        out = out.view(out.size(0), -1)
        out = self.output(out)

        return out