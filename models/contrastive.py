import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1, distance_meter='euclidian'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance_meter = distance_meter
        
    def compute_covariance_matrix(self, x):
        # Subtract the mean of each feature
        x = x - torch.mean(x, dim=0)
        # Calculate covariance matrix
        cov_matrix = torch.mm(x.T, x) / (x.size(0) - 1)
        return cov_matrix

    def forward(self, output1, output2, label):
        if self.distance_meter == 'euclidian':
            distance = nn.functional.pairwise_distance(output1, output2)
        elif self.distance_meter == 'cosine':
            distance = 1 - nn.functional.cosine_similarity(output1, output2)
        elif self.distance_meter == 'manhattan':
            distance = torch.sum(torch.abs(output1 - output2), dim=1)
        loss = torch.mean((1 - label) * torch.pow(distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss