import torch
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma = 2.0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.register_buffer(name='alpha', tensor=alpha)
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6
    
    def forward(self, pred, target):
        focal_pos = -self.alpha[1] * torch.pow(1-pred, self.gamma) * torch.log(pred + self.eps)
        focal_neg = -self.alpha[0] * torch.pow(pred, self.gamma)  * torch.log(1-pred + self.eps)
        loss = target*focal_pos + (1-target)*focal_neg

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
