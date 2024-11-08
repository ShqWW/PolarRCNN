import torch
from torch import nn
from .triplet_head import TripletHead
from FastNMS.fastnms import nms
import numpy as np
from utils.coord_transform import CoordTrans_torch
import math

class GlobalPolarHead(nn.Module):
    def __init__(self, cfg=None):
        super(GlobalPolarHead, self).__init__()
        self.cfg = cfg
        self.img_w = self.cfg.img_w
        self.img_h = self.cfg.img_h
        self.ori_img_w = self.cfg.ori_img_w
        self.ori_img_h = self.cfg.ori_img_h
        self.max_laness = cfg.max_lanes
        self.cut_height = cfg.cut_height
        self.num_offsets = cfg.num_offsets
        self.num_strips = cfg.num_offsets - 1
        self.num_feat_samples = cfg.num_feat_samples
        
        self.offset_stride = cfg.offset_stride

        self.fc_hidden_dim = cfg.fc_hidden_dim
        self.prior_feat_channels = cfg.prior_feat_channels
        self.num_line_groups = cfg.num_line_groups

        self.nms_thres = cfg.nms_thres
        self.is_nmsfree = cfg.is_nmsfree
        self.conf_thres = cfg.conf_thres_nmsfree if self.is_nmsfree else cfg.conf_thres
        
        sample_index = torch.linspace(0, 1, steps=self.num_feat_samples, dtype=torch.float32)
        sample_index = torch.flip(self.num_strips-(sample_index*self.num_strips).long(), dims=[-1])
        self.register_buffer(name='sample_index', tensor=sample_index)

        self.rcnn_head = TripletHead(cfg=cfg)
        self.coord_transform = CoordTrans_torch(cfg)

    def forward(self, x, rpn_dict):
        feat_list = list(x)
        feat_list.reverse()
        anchor_embeddings = rpn_dict['anchor_embeddings']
        anchor_id = rpn_dict['anchor_id']
        pred_dict = self.forward_function(anchor_embeddings, feat_list, anchor_id)

        if self.training:
            return pred_dict
        else:
            lane_list = self.pred_lanes_batch(pred_dict)
            anchor_embeddings = anchor_embeddings.detach().cpu().numpy()
            result_dict = {'lane_list':lane_list, 'anchor_embeddings': anchor_embeddings}
            return result_dict

    def forward_function(self, anchor_embeddings, feat_list, anchor_id_embeddings):
        pred_dict = {}
        anchor_embeddings = anchor_embeddings.detach().clone()
        anchor_embeddings_group_base = anchor_embeddings.unsqueeze(-2).repeat(1, 1, self.num_line_groups, 1)
        
        feat_samples_grid_norm, lanereg_base_car = self.sample_from_anchor(anchor_embeddings)
        cls_pred, reg_pred, reg_pred_aux, cls_pred_o2o = self.rcnn_head(feat_list, feat_samples_grid_norm, anchor_id_embeddings, anchor_embeddings)

        end_points, lanereg_xs_offset = reg_pred[..., 0:2], reg_pred[..., 2:]
        line_paras_group_reg = reg_pred_aux.view(reg_pred.shape[0], reg_pred.shape[1], self.num_line_groups, 2) + anchor_embeddings_group_base

        pred_dict['cls'] = cls_pred
        pred_dict['cls_o2o'] = cls_pred_o2o
        pred_dict['end_points'] = end_points
        pred_dict['lanereg_xs_offset'] = lanereg_xs_offset
        pred_dict['lanereg_base_car'] = lanereg_base_car
        pred_dict['line_paras_group_reg'] = line_paras_group_reg
        return pred_dict
    
    def sample_from_anchor(self, anchor_embeddings):
        anchor_paras = anchor_embeddings.detach().clone()
        anchor_paras[..., 1] *= self.img_w
        anchor_paras[..., 0] *= math.pi
        with torch.no_grad():
            samples_car = self.coord_transform.sample_xs_by_fix_ys(anchor_paras.view(-1, 2))
            img_samples = self.coord_transform.cartesian2img(samples_car)
            anchor_samples = torch.flip(samples_car, dims=[-2]).contiguous()
            anchor_samples = anchor_samples.view(anchor_embeddings.shape[0], anchor_embeddings.shape[1], -1, 2)
            feat_samples_grid = img_samples[..., self.sample_index, :]
            feat_samples_grid[..., 0] = (feat_samples_grid[..., 0]/self.img_w)
            feat_samples_grid[..., 1] = (feat_samples_grid[..., 1]/self.img_h)
            feat_samles_grid_norm = feat_samples_grid.view(anchor_embeddings.shape[0], anchor_embeddings.shape[1], -1, 2)
        return feat_samles_grid_norm, anchor_samples
        
    def pred_lanes_batch(self, pred_dict):
        cls_pred_batch = pred_dict['cls_o2o'] if self.is_nmsfree else pred_dict['cls']
        end_points_batch = pred_dict['end_points']
        lanereg_xs_offset = pred_dict['lanereg_xs_offset']
        lanereg_base_car = pred_dict['lanereg_base_car']

        batch_size, num_anchor = lanereg_base_car.shape[0], lanereg_base_car.shape[1]
        lanereg_base_car[..., 0] = lanereg_base_car[..., 0] + lanereg_xs_offset*self.img_w
        lanereg_car = lanereg_base_car 

        img_samples = self.coord_transform.cartesian2img(lanereg_car)
        lane_points_img_batch = img_samples.view(batch_size, num_anchor, -1, 2)
        
        lane_list = []
        for cls_pred, end_points, lane_points_img in zip(cls_pred_batch, end_points_batch, lane_points_img_batch):
            keep_inds = (cls_pred>=self.conf_thres)
            cls_pred, end_points, lane_points_img = cls_pred[keep_inds], end_points[keep_inds], lane_points_img[keep_inds]
            if cls_pred.shape[0] == 0:
                lane_list.append([])
                continue

            #nms per image
            if not self.is_nmsfree:
                sample_nms = torch.cat((end_points, lane_points_img[..., 0]), dim=-1)
                keep, num_to_keep, _ = nms(sample_nms.clone(), cls_pred.clone(), overlap=self.nms_thres, top_k=self.max_laness)
                keep_ind = keep[:num_to_keep]
                cls_pred, end_points, lane_points_img = cls_pred[keep_ind], end_points[keep_ind], lane_points_img[keep_ind]

            lane_prediction = self.get_lane_point(end_points, lane_points_img)
            lane_list.append(lane_prediction)
        return lane_list

    def get_lane_point(self, end_points, points_lanes):
        lanes = []
        for end_point, points_lane in zip(end_points, points_lanes):
            lane_xs = points_lane[..., 0]
            lane_ys = points_lane[..., 1]
            start = min(max(0, int(round(end_point[0].item() * self.num_strips))), self.num_strips)
            end = min(int(round(end_point[1].item() * self.num_strips))-1, self.num_offsets - 1)
            mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = lane_ys[lane_xs >= 0].flip(0)
            lane_xs = lane_xs[lane_xs >= 0].flip(0)
            lane_ys = (lane_ys*((self.ori_img_h-self.cut_height)/self.img_h) + self.cut_height) / self.ori_img_h
            lane_xs = lane_xs / self.img_w
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
            points = points.cpu().numpy().astype(np.double)  
            lane = {"points": points, "conf": 1}
            lanes.append(lane)
        return lanes


