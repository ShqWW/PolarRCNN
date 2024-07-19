import torch
from torch import nn
from .roi_loss.build import build_roi_loss
from .rpn_loss.build import build_rpn_loss

class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.roi_loss = build_roi_loss(cfg)
        self.rpn_loss = build_rpn_loss(cfg)
    
    def forward(self, pred_dict, target_dict):
        loss_roi, loss_roi_msg = self.roi_loss(pred_dict, target_dict)
        if self.rpn_loss is not None:
            loss_rpn, loss_rpn_msg = self.rpn_loss(pred_dict, target_dict)
            loss_roi = loss_roi + loss_rpn
            loss_roi_msg.update(loss_rpn_msg)
        
        loss = loss_roi
        loss_msg = loss_roi_msg
        loss_msg['loss'] = loss
        return loss, loss_msg

        