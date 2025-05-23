import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """Initialize a pre-activation residual block with batch normalization before convolutions.
        
        Args:
            in_planes: Number of input channels
            planes: Number of output channels
            stride: Stride for the first convolution
        """
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        """Forward pass with pre-activation and shortcut connection."""
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        """Initialize a pre-activation bottleneck block with 3 convolutions.
        
        Args:
            in_planes: Number of input channels
            planes: Number of intermediate channels (expanded by 4 for output)
            stride: Stride for the second convolution
        """
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        """Forward pass with pre-activation and shortcut connection."""
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    """Pre-activation ResNet architecture with batch normalization before convolutions."""
    
    def __init__(self, block, num_blocks, num_classes=10):
        """Initialize PreActResNet with specified block type and number of blocks per layer.
        
        Args:
            block: The building block class (PreActBlock or PreActBottleneck)
            num_blocks: List containing number of blocks in each of the 4 layers
            num_classes: Number of output classes
        """
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer5 = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Flatten()
        )
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Create a layer with the specified number of blocks.
        
        Args:
            block: Block type to use
            planes: Number of output channels
            num_blocks: How many blocks to stack in this layer
            stride: Stride for the first block
            
        Returns:
            Sequential container with the blocks
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        out = self.layer5(out)
        out = self.linear(out)
        return out

def PreActResNet9(num_class=10):
    """Constructs a PreActResNet-9 model with 1 block in each layer."""
    return PreActResNet(PreActBlock, [1,1,1,1], num_classes=num_class)

def PreActResNet18(num_class=10):
    """Constructs a PreActResNet-18 model with basic blocks."""
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=num_class)

def PreActResNet34(num_class=10):
    """Constructs a PreActResNet-34 model with basic blocks."""
    return PreActResNet(PreActBlock, [3,4,6,3], num_classes=num_class)

def PreActResNet50(num_class=10):
    """Constructs a PreActResNet-50 model with bottleneck blocks."""
    return PreActResNet(PreActBottleneck, [3,4,6,3], num_classes=num_class)

def PreActResNet101(num_class=10):
    """Constructs a PreActResNet-101 model with bottleneck blocks."""
    return PreActResNet(PreActBottleneck, [3,4,23,3], num_classes=num_class)

def PreActResNet152(num_class=10):
    """Constructs a PreActResNet-152 model with bottleneck blocks."""
    return PreActResNet(PreActBottleneck, [3,8,36,3], num_classes=num_class)


def test():
    """Test function to verify the network's output dimensions."""
    net = PreActResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())