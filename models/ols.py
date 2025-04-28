import torch
import torch.nn as nn
from torch import Tensor


class OnlineLabelSmoothing(nn.Module):
    """
    Implements Online Label Smoothing as described in the paper:
    "Online Label Smoothing for Deep Learning"
    https://arxiv.org/pdf/2011.12562.pdf
    
    This technique adaptively updates label smoothing values based on model predictions,
    improving generalization and robustness to label noise.
    """

    def __init__(self, alpha: float, n_classes: int, smoothing: float = 0.1):
        """Initialize the Online Label Smoothing module.
        
        Args:
            alpha: Weight balancing factor between soft_loss and hard_loss (0-1)
            n_classes: Number of classes in the classification problem
            smoothing: Initial smoothing factor for first epoch (standard label smoothing)
        """
        super(OnlineLabelSmoothing, self).__init__()
        assert 0 <= alpha <= 1, 'Alpha must be in range [0, 1]'
        self.a = alpha
        self.n_classes = n_classes
        # Initialize soft labels with normal Label Smoothing for first epoch
        self.register_buffer('supervise', torch.zeros(n_classes, n_classes))
        self.supervise.fill_(smoothing / (n_classes - 1))
        self.supervise.fill_diagonal_(1 - smoothing)

        # Update matrix is used to supervise next epoch
        self.register_buffer('update', torch.zeros_like(self.supervise))
        # For normalizing we need a count for each class
        self.register_buffer('idx_count', torch.zeros(n_classes))
        self.hard_loss = nn.CrossEntropyLoss()

    def forward(self, y_h: Tensor, y: Tensor):
        """Calculate the combined loss using both hard and soft components.
        
        Args:
            y_h: Predicted logits from the model
            y: Ground truth labels
            
        Returns:
            Weighted sum of hard cross-entropy loss and soft loss
        """
        # Calculate the final loss
        soft_loss = self.soft_loss(y_h, y)
        hard_loss = self.hard_loss(y_h, y)
        return self.a * hard_loss + (1 - self.a) * soft_loss

    def soft_loss(self, y_h: Tensor, y: Tensor):
        """Calculate the soft loss and update the supervision matrix when training.
        
        Args:
            y_h: Predicted logits from the model
            y: Ground truth labels
            
        Returns:
            Soft loss based on current supervision matrix
        """
        y_h = y_h.log_softmax(dim=-1)
        if self.training:
            with torch.no_grad():
                self.step(y_h.exp(), y)
        true_dist = torch.index_select(self.supervise, 1, y).swapaxes(-1, -2)
        return torch.mean(torch.sum(-true_dist * y_h, dim=-1))

    def step(self, y_h: Tensor, y: Tensor) -> None:
        """Update the supervision matrix using current model predictions.
        
        This method accumulates the probability distributions of correctly classified
        examples to be used for supervising the next epoch.
        
        Args:
            y_h: Predicted probabilities (after softmax)
            y: Ground truth labels
        
        Steps:
            1. Find correctly classified examples
            2. Extract their predicted probability distributions
            3. Accumulate these distributions in the update matrix
            4. Keep count of samples added for each class for normalization
        """
        # 1. Calculate predicted classes
        y_h_idx = y_h.argmax(dim=-1)
        # 2. Filter only correct
        mask = torch.eq(y_h_idx, y)
        y_h_c = y_h[mask]
        y_h_idx_c = y_h_idx[mask]
        # 3. Add y_h probabilities rows as columns to `update`
        self.update.index_add_(1, y_h_idx_c, y_h_c.swapaxes(-1, -2))
        # 4. Update `idx_count`
        self.idx_count.index_add_(0, y_h_idx_c, torch.ones_like(y_h_idx_c, dtype=torch.float32))

    def next_epoch(self) -> None:
        """Prepare for the next epoch by updating the supervision matrix.
        
        This method should be called at the end of each epoch. It:
        1. Normalizes the accumulated update matrix by the count of samples per class
        2. Sets the supervision matrix to this normalized update
        3. Resets the update matrix and counters for the next epoch
        """
        # 5. Divide memory by `idx_count` to obtain average (column-wise)
        self.idx_count[torch.eq(self.idx_count, 0)] = 1  # Avoid 0 denominator
        # Normalize by taking the average
        self.update /= self.idx_count
        self.idx_count.zero_()
        self.supervise = self.update
        self.update = self.update.clone().zero_()