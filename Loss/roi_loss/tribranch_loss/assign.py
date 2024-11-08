import torch
import math
from scipy.optimize import linear_sum_assignment
from .lineiou_loss import Lane_iou

def dynamic_k_assign(cost_matrix, ious_matrix, n_candidate_k=4):
    matching_matrix = torch.zeros_like(cost_matrix)
    ious_matrix[ious_matrix < 0] = 0.
    topk_ious, _ = torch.topk(ious_matrix, n_candidate_k, dim=1)
    dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
    num_gt = cost_matrix.shape[0]
    for gt_idx in range(num_gt):
        _, pos_idx = torch.topk(cost_matrix[gt_idx],
                                k=dynamic_ks[gt_idx].item(),
                                largest=False)
        matching_matrix[gt_idx, pos_idx] = 1.0
    del topk_ious, dynamic_ks, pos_idx

    matched_gt = matching_matrix.sum(0)
    if (matched_gt > 1).sum() > 0:
        _, cost_argmin = torch.min(cost_matrix[:, matched_gt > 1], dim=0)
        matching_matrix[:, matched_gt > 1] *= 0.0
        matching_matrix[cost_argmin, matched_gt > 1] = 1.0

    prior_idx = matching_matrix.sum(0).nonzero()
    gt_idx = matching_matrix[:, prior_idx].argmax(0)
    return prior_idx.flatten(), gt_idx.flatten()

def hungarian_assign(cost_matrix):
    device = cost_matrix.device
    cost_matrix = cost_matrix.cpu().numpy()
    gt_idx, prior_idx = linear_sum_assignment(cost_matrix)
    prior_idx, gt_idx = torch.from_numpy(prior_idx).to(device), torch.from_numpy(gt_idx).to(device)
    return prior_idx, gt_idx

class Assigner():
    def __init__(self, cfg):
        super().__init__()
        self.reg_cost_weight = cfg.reg_cost_weight
        self.reg_cost_weight_o2o = cfg.reg_cost_weight_o2o
        self.cls_cost_weight = cfg.cls_cost_weight
        self.img_h, self.img_w = cfg.img_h, cfg.img_w
        self.num_offsets = cfg.num_offsets
        self.offset_stride = cfg.offset_stride
        self.n_stripes = self.num_offsets - 1
        self.center_w, self.center_h = cfg.center_w, cfg.center_h
        self.y_stride = self.offset_stride*((cfg.ori_img_h-cfg.cut_height)/cfg.ori_img_w)/(self.img_h/self.img_w)
        self.angle_prior_thres = cfg.angle_prior_thres
        self.rho_prior_thres = cfg.rho_prior_thres
        self.cost_iou_width = cfg.cost_iou_width
        self.ota_iou_width = cfg.ota_iou_width

    def __call__(self, cls_pred, cls_pred_o2o, lane_points, anchor_samples, lane_point_xs_gt, lane_point_validmask, anchor_embeddings, line_paras):
        cls_pred = cls_pred.detach().clone()
        cls_pred_o2o = cls_pred_o2o.detach().clone()
        lane_points = lane_points.detach().clone()
        lane_point_validmask = lane_point_validmask.bool()
        anchors=  anchor_embeddings.detach().clone()
        line_paras = line_paras.detach().clone()
        anchors[..., 0] *= math.pi
        anchors[..., 1] *= self.img_w
        
        angle_dis_matrix = torch.abs(anchors[..., 0].unsqueeze(-2) - line_paras[..., 0].unsqueeze(-1))
        rho_dis_matrix = torch.abs(anchors[..., 1].unsqueeze(-2) - line_paras[..., 1].unsqueeze(-1))
        is_in_box = (angle_dis_matrix<self.angle_prior_thres) & (rho_dis_matrix<self.rho_prior_thres)

        lane_points *= self.img_w
        num_gt = lane_point_xs_gt.shape[0]

        cls_score = -torch.log(cls_pred+1e-8)
        cls_score = cls_score.unsqueeze(0).repeat(num_gt, 1)

        cls_score_o2o = -torch.log(cls_pred_o2o+1e-8)
        cls_score_o2o = cls_score_o2o.unsqueeze(0).repeat(num_gt, 1)

        cost_iou = Lane_iou(lane_points+anchor_samples[..., 0], lane_point_xs_gt, lane_point_validmask, width=self.cost_iou_width, y_stride=self.y_stride, align=False) 
        iou_score = torch.log(cost_iou+1e-8)
        
        cost_matrix = -iou_score*self.reg_cost_weight + cls_score*self.cls_cost_weight + 1e6*(~is_in_box)
        cost_matrix_o2o = -iou_score*self.reg_cost_weight_o2o + cls_score_o2o*self.cls_cost_weight + 1e6*(~is_in_box)
        
        iou_matrix = Lane_iou(lane_points+anchor_samples[..., 0], lane_point_xs_gt, lane_point_validmask, width=self.ota_iou_width, y_stride=self.y_stride, align=False)

        prior_idx, gt_idx = dynamic_k_assign(cost_matrix, iou_matrix, n_candidate_k=4)
        prior_idx_o2o, _ = hungarian_assign(cost_matrix_o2o)

        prior_idx_reg, gt_idx_reg = prior_idx, gt_idx
        return prior_idx, prior_idx_o2o, prior_idx_reg, gt_idx_reg

