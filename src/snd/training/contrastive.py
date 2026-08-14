import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """Contrastive loss function for Siamese networks.

    Brings similar samples closer and pushes dissimilar samples apart in the embedding space.
    """

    def __init__(self, margin=1, distance_meter='euclidian'):
        """Initialize with margin and distance metric type.

        Only the Euclidean metric is supported; `distance_meter` is kept so the
        per-dataset config dicts and notebook call sites stay valid.
        """
        super(ContrastiveLoss, self).__init__()
        if distance_meter != 'euclidian':
            raise ValueError(
                f"Unsupported distance_meter {distance_meter!r}; only 'euclidian' is supported.")
        self.margin = margin
        self.distance_meter = distance_meter

    def forward(self, output1, output2, same_label):
        """Calculate contrastive loss between pairs of embeddings.

        Args:
            output1: First embedding vector
            output2: Second embedding vector
            same_label: Binary label (1 for same class, 0 for different class)

        Returns:
            Loss value encouraging similar pairs to be close and dissimilar pairs to be distant
        """
        distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((same_label) * torch.pow(distance, 2) +
                          (1 - same_label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss
