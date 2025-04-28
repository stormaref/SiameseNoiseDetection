import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
import models
import importlib
importlib.reload(models)

class Resnet(nn.Module):
    """Wrapper class for different ResNet architectures with customizable output layer.
    
    Uses pre-trained ResNet models from torchvision and adapts them for classification.
    """
    def __init__(self, num_classes=10, model='resnet18', pre_trained=True):
        """Initialize ResNet model with specified architecture and pre-training options.
        
        Args:
            num_classes: Number of output classes for classification
            model: ResNet architecture variant to use ('resnet18', 'resnet34', etc.)
            pre_trained: Whether to use pre-trained weights
        """
        super(Resnet, self).__init__()
        if model == 'resnet18':
            if pre_trained:
                base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
            else:
                base_model = resnet18()
        elif model == 'preact-resnet18':
            if pre_trained:
                raise ValueError('Not supported')
            else:
                base_model = models.preact.PreActResNet18()
        elif model == 'resnet34':
            if pre_trained:
                base_model = resnet34(weights=ResNet34_Weights.DEFAULT)
            else:
                base_model = resnet34()
        elif model == 'resnet50':
            if pre_trained:
                base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            else:
                base_model = resnet50()
        else:
            raise ValueError('Model not supported')
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        if model.__contains__('preact'):
            self.output = nn.Linear(8192, num_classes)
        else:
            self.output = nn.Linear(512, num_classes)

    def forward(self, input):
        """Forward pass through the network, extracting features and classifying.
        
        Args:
            input: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Classification logits
        """
        out = self.feature_extractor(input)
        out = out.view(out.size(0), -1)
        out = self.output(out)

        return out