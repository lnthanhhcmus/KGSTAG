import torch.nn as nn
import torch
from torch.autograd import Variable


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""
    
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
    
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = torch.full_like(x, self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = (target == self.padding_idx)
        true_dist[mask] = 0
        self.true_dist = true_dist.detach()
        return self.criterion(x, true_dist)