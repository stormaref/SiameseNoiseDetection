import torch
import torch.nn as nn
import torch.nn.functional as F

class LossAttentionLayer(nn.Module):
    def __init__(self, input_dim, num_losses):
        super(LossAttentionLayer, self).__init__()
        
        self.attention_layer = nn.Linear(input_dim, num_losses)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.attention_layer(x)
        x = F.relu(x)
        x = self.dropout(x)
        attention_weights = F.softmax(x, dim=1)
        return attention_weights
