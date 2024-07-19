import torch
from torch import nn

class RankLoss(nn.Module):
    def __init__(self, tau=0.5):
        super(RankLoss, self).__init__()
        self.tau = tau

    def forward(self, pred, target, mask=None):
        pos_ind = torch.where(target==1)[0]
        neg_ind = torch.where((target==0)&mask if mask is not None else (target==0))[0]
        pos_num ,neg_num = pos_ind.shape[0], neg_ind.shape[0]
        if neg_num == 0:
            return 0
        pos_pred, neg_pred = pred[pos_ind], pred[neg_ind]
        rank_loss = torch.sum(torch.clamp(self.tau-(pos_pred.unsqueeze(-1)-neg_pred.unsqueeze(-2)), min=0))/(pos_num*neg_num)
        return rank_loss