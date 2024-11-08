import math
import torch
from torch import nn
import torch.nn.functional as F
from .focal_loss import FocalLoss
from .rank_loss import RankLoss
from .lineiou_loss import liou_loss
from .assign import Assigner
from utils.coord_transform import CoordTrans_torch

class TriBranchLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cls_loss_weight = cfg.cls_loss_weight
        self.iou_loss_weight = cfg.iou_loss_weight
        self.end_loss_weight = cfg.end_loss_weight
        self.aux_loss_weight = cfg.aux_loss_weight
        self.rank_loss_weight = cfg.rank_loss_weight
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.center_w, self.center_h = cfg.center_w, cfg.center_h
        self.num_offsets = cfg.num_offsets
        self.n_strips = self.num_offsets - 1
        self.offset_stride = cfg.offset_stride
        self.conf_thres = cfg.conf_thres
        self.conf_thres_o2o = cfg.conf_thres_o2o
        self.loss_iou_width = cfg.loss_iou_width
        self.g_weight = cfg.g_weight

        alpha = torch.tensor([1-cfg.cls_loss_alpha, cfg.cls_loss_alpha])
        alpha_o2o = torch.tensor([1-cfg.cls_loss_alpha_o2o, cfg.cls_loss_alpha_o2o])

        self.coord_trans = CoordTrans_torch(cfg)
        self.assigner = Assigner(cfg)
        self.cls_criterion = FocalLoss(alpha=alpha, gamma=2.)
        self.cls_criterion_o2o = FocalLoss(alpha=alpha_o2o, gamma=2.)
        self.rank_loss = RankLoss(tau=0.5)
        self.y_stride = self.offset_stride*((cfg.ori_img_h-cfg.cut_height)/cfg.ori_img_w)/(self.img_h/self.img_w)
    
    def forward(self, pred_dict, target_dict):
        cls_pred_batch = pred_dict['cls']
        end_points_batch = pred_dict['end_points']
        lanereg_xs_offset_batch = pred_dict['lanereg_xs_offset']
        cls_pred_batch_o2o = pred_dict['cls_o2o']
        line_paras_group_reg_batch = pred_dict['line_paras_group_reg']
        lanereg_base_car_batch = pred_dict['lanereg_base_car']
        anchor_embeddings_batch = pred_dict['anchor_embeddings']

        line_paras_group_gt_batch = target_dict['line_paras_group']
        group_validmask_batch = target_dict['group_validmask']
        lane_valids_batch = target_dict['lane_valid']
        lane_point_xs_gt_batch =  target_dict['lane_point_xs']
        lane_point_validmask_batch =  target_dict['lane_point_validmask']
        end_point_gt_batch = target_dict['end_point']
        line_paras_batch=  target_dict['line_paras']
        
        cls_loss = torch.tensor([0]).cuda()
        cls_o2o_loss = torch.tensor([0]).cuda()
        iou_loss = torch.tensor([0]).cuda()
        end_point_loss = torch.tensor([0]).cuda()
        aux_reg_loss = torch.tensor([0]).cuda()
        rank_loss = torch.tensor([0]).cuda()
        batch_size = cls_pred_batch.shape[0]

        prior_idx_list = []
        gt_idx_list = []
        batch_idx_list = []
        
        batch_idx = 0
        for cls_pred, cls_pred_o2o, lanereg_xs_offset, lanereg_base_car, lane_valids, lane_point_xs_gt, lane_point_validmask, anchor_embeddings, line_paras in zip(cls_pred_batch, cls_pred_batch_o2o, lanereg_xs_offset_batch, lanereg_base_car_batch, lane_valids_batch, lane_point_xs_gt_batch, lane_point_validmask_batch, anchor_embeddings_batch, line_paras_batch):

            num_gt = lane_valids.sum() 
            lane_point_xs_gt = lane_point_xs_gt[lane_valids]
            lane_point_validmask = lane_point_validmask[lane_valids]
            line_paras = line_paras[lane_valids]
            cls_target = cls_pred.new_zeros(cls_pred.shape[0]).long()
            cls_target_o2o = cls_pred.new_zeros(cls_pred.shape[0]).long() 
            if num_gt==0:
                cls_loss = cls_loss + self.cls_criterion(cls_pred, cls_target).sum()
                cls_mask_o2o = (cls_pred>self.conf_thres_o2o)
                cls_o2o_loss = cls_o2o_loss + (self.cls_criterion_o2o(cls_pred_o2o, cls_target_o2o)*cls_mask_o2o).sum()
                batch_idx += 1
                continue
            with torch.no_grad():
                prior_idx, prior_idx_o2o, prior_idx_reg, gt_idx_reg = self.assigner(cls_pred, cls_pred_o2o, lanereg_xs_offset, lanereg_base_car, lane_point_xs_gt, lane_point_validmask, anchor_embeddings, line_paras) 
                prior_idx_list.append(prior_idx_reg)
                gt_idx_list.append(gt_idx_reg)
                batch_idx_list.append(batch_idx*torch.ones_like(prior_idx_reg))
                    
                cls_target[prior_idx] = 1
                cls_target_o2o[prior_idx_o2o] = 1
            cls_loss = cls_loss + self.cls_criterion(cls_pred, cls_target).sum()
            cls_mask_o2o = (cls_pred>self.conf_thres_o2o)
            cls_o2o_loss = cls_o2o_loss + (self.cls_criterion_o2o(cls_pred_o2o, cls_target_o2o)*cls_mask_o2o).sum()
            rank_loss = rank_loss + self.rank_loss(cls_pred_o2o, cls_target_o2o, mask=cls_mask_o2o)
            batch_idx += 1

        prior_idx_batch = (torch.cat(batch_idx_list, dim=0), torch.cat((prior_idx_list), dim=0))
        gt_idx_batch = (torch.cat(batch_idx_list, dim=0), torch.cat((gt_idx_list), dim=0))

        with torch.no_grad():
            line_paras_group_gt = line_paras_group_gt_batch[gt_idx_batch].detach().clone()
            group_validmask = group_validmask_batch[gt_idx_batch].detach().clone().unsqueeze(-1)
            
            line_paras_group_gt = line_paras_group_gt * group_validmask

            end_points_gt = end_point_gt_batch[gt_idx_batch]
            x_samples_car = lanereg_base_car_batch[prior_idx_batch][..., 0].detach().clone()
            
            lane_points_target = lane_point_xs_gt_batch[gt_idx_batch]
            lane_points_validmask = lane_point_validmask_batch[gt_idx_batch].bool()
            
            end_points_gt = end_points_gt/self.img_h*self.n_strips
            line_paras_group_gt[..., 0] *= (180/math.pi)


        end_points = end_points_batch[prior_idx_batch]
        lanereg_xs_offset = lanereg_xs_offset_batch[prior_idx_batch]
        line_paras_group_reg = line_paras_group_reg_batch[prior_idx_batch]
        line_paras_group_reg = line_paras_group_reg * group_validmask
        line_paras_group_reg[..., 0] *= 180
        line_paras_group_reg[..., 1] *= self.img_w

        iou_loss = iou_loss + liou_loss(lanereg_xs_offset*self.img_w+x_samples_car, lane_points_target, lane_points_validmask, width=self.loss_iou_width, y_stride=self.y_stride, g_weight=self.g_weight).mean()
        end_point_loss = end_point_loss + F.smooth_l1_loss(end_points*self.n_strips, end_points_gt).mean()
        aux_reg_loss = aux_reg_loss + F.smooth_l1_loss(line_paras_group_reg.flatten(-2, -1), line_paras_group_gt.flatten(-2, -1)).mean()


        cls_loss /= batch_size
        cls_o2o_loss /= batch_size

        loss = cls_loss * self.cls_loss_weight\
             + iou_loss * self.iou_loss_weight\
             + end_point_loss * self.end_loss_weight\
             + aux_reg_loss * self.aux_loss_weight\
             + cls_o2o_loss *self.cls_loss_weight\
             + rank_loss * self.rank_loss_weight
             
        loss_msg = {
            'loss': loss,
            'cls_loss': cls_loss * self.cls_loss_weight,
            'reg_loss': (end_point_loss * self.end_loss_weight + aux_reg_loss * self.aux_loss_weight),
            'iou_loss': iou_loss * self.iou_loss_weight,
            'cls_loss_o2o': cls_o2o_loss *self.cls_loss_weight,
            'rank_loss': rank_loss * self.rank_loss_weight
        }
        return loss, loss_msg