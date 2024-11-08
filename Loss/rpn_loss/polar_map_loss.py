import torch 
from torch import nn


class PolarMapLoss(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.loss_weight = cfg.polarmap_loss_weight
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='mean')
    def forward(self, pred_dict, target_dict):
        pred_polar_map = pred_dict['polar_map']
        target_polar_map = target_dict['polar_map']

        pred_polar_map_cls = pred_polar_map[:, 0, ...]
        pred_polar_map_reg = pred_polar_map[:, 1:, ...]
        target_polar_map_cls = target_polar_map[:, 0, ...]
        target_polar_map_reg = target_polar_map[:, 1:, ...]

        valid_mask = (target_polar_map_cls==1).unsqueeze(1)
        polar_map_reg_loss = ((target_polar_map_reg-pred_polar_map_reg).abs()*valid_mask).sum()/(valid_mask.sum()+1e-4)*self.loss_weight

        polar_map_cls_loss = self.cls_loss(pred_polar_map_cls, target_polar_map_cls)*self.loss_weight
        loss_msg = {'polarmap_reg_loss': polar_map_reg_loss, 'polarmap_cls_loss': polar_map_cls_loss}
        loss = polar_map_reg_loss + polar_map_cls_loss
        return loss, loss_msg
