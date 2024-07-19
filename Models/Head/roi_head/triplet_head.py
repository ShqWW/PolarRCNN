import torch
from torch import nn
import torch.nn.functional as F
import math
# import time

class LaneGNN(nn.Module):
    def __init__(self, feat_dim, feat_inter_dim=128, iou_dim=1):
        super(LaneGNN, self).__init__()
        self.pos_linear = nn.Linear(2, feat_inter_dim)
        self.linear_in = nn.Linear(feat_dim, feat_inter_dim)
        self.linear_out = nn.Linear(feat_dim, feat_inter_dim)

        self.edge_layers = nn.Sequential(nn.Linear(feat_inter_dim, feat_inter_dim),
                                         nn.GELU(),
                                         nn.Linear(feat_inter_dim, iou_dim))
        
        self.node_layers = nn.Sequential(nn.Linear(iou_dim, feat_dim), 
                                         nn.ReLU(),
                                         nn.Linear(feat_dim, feat_dim),
                                         nn.ReLU())

    def forward(self, node_features, supress_mat, pos_emb_mat):
        node_features_in = self.linear_in(node_features)
        node_features_out = self.linear_out(node_features)

        edge_emb_mat = self.pos_linear(pos_emb_mat)
        edge_features_mat = node_features_in.unsqueeze(2) - node_features_out.unsqueeze(1) + edge_emb_mat

        edge_features_mat = self.edge_layers(edge_features_mat)

        edge_features_mat = edge_features_mat * supress_mat.unsqueeze(-1)
        node_features_max, _ = torch.max(edge_features_mat, dim=1)

        node_features = self.node_layers(node_features_max)
        return node_features


class O2OClsHead(nn.Module):
    def __init__(self, cfg=None):
        super(O2OClsHead, self).__init__()
        self.img_w = cfg.img_w
        self.img_h = cfg.img_h
        self.center_h = cfg.center_h
        self.fc_hidden_dim = cfg.fc_hidden_dim
        self.o2o_angle_thres = cfg.o2o_angle_thres 
        self.o2o_rho_thres = cfg.o2o_rho_thres 
        self.conf_thres = cfg.conf_thres
        self.conf_thres_o2o = cfg.conf_thres_o2o
        
        self.cls_block = nn.Sequential(nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim), nn.ReLU())
        self.lane_gnn = LaneGNN(self.fc_hidden_dim, cfg.gnn_inter_dim, cfg.iou_dim)
        self.cls_layer = nn.Linear(self.fc_hidden_dim, 1) 
        self.sigmoid = nn.Sigmoid()

        sample_car_y = self.center_h - torch.linspace(0, 1-1e-5, steps=cfg.num_offsets, dtype=torch.float32)*self.img_h
        self.register_buffer(name='sample_car_y', tensor=sample_car_y[0:2])

    def forward(self, batch_features, cls_pred, anchor_id, anchor_embeddings): 
        batch_features = batch_features.detach().clone()
        anchor_id = anchor_id.detach().clone()
        anchor_embeddings = anchor_embeddings.detach().clone()
        cls_pred = cls_pred.detach().clone()

        with torch.no_grad():
            nbr_masks = self.get_nbr_mask(anchor_embeddings.clone())
            cls_pred_ = cls_pred.clone()

            cls_mat = cls_pred_.unsqueeze(-1) - cls_pred_.unsqueeze(-2)
            id_mat = anchor_id.unsqueeze(-1) - anchor_id.unsqueeze(-2)

            supress_mat = ((cls_mat>0)|((cls_mat==0)&(id_mat>0))).float()
            supress_mat[torch.where(nbr_masks==0)] = 0 #mask the invalid edge

            pos_emb = self.get_sample_point(anchor_embeddings.clone())
            pos_emb_mat = pos_emb.unsqueeze(2)-pos_emb.unsqueeze(1)

        cls_features = self.cls_block(batch_features)
        cls_features = self.lane_gnn(cls_features, supress_mat, pos_emb_mat)
        
        logits = self.cls_layer(cls_features).squeeze(-1)
        logits[torch.where(cls_pred<self.conf_thres_o2o)] = (-1e6)
        cls_pred_o2o = self.sigmoid(logits)
        return cls_pred_o2o

    def get_sample_point(self, anchor_embeddings):
        batch_size, num_anchor = anchor_embeddings.shape[0], anchor_embeddings.shape[1]
        anchor_embeddings = anchor_embeddings.view(-1, 2)
        angle = anchor_embeddings[..., 0] * math.pi
        rho = anchor_embeddings[..., 1] * self.img_w
        sample_car_y = self.sample_car_y.unsqueeze(0).repeat(angle.shape[0], 1)
        sample_car_x = -torch.tan(angle).unsqueeze(-1)*sample_car_y + (rho/torch.cos(angle)).unsqueeze(-1)
        fea_point = sample_car_x.view(batch_size, num_anchor, 2)/self.img_w
        return fea_point

    def get_nbr_mask(self, anchor_embeddings):
        angle = anchor_embeddings[..., 0] * math.pi
        rho = anchor_embeddings[..., 1] * self.img_w
        angle_dis_matrix = torch.abs(angle.unsqueeze(-1)-angle.unsqueeze(-2))
        rho_dis_matrix = torch.abs(rho.unsqueeze(-1)-rho.unsqueeze(-2))

        angle_dis_matrix = torch.triu(angle_dis_matrix, diagonal=1)
        angle_dis_matrix = angle_dis_matrix + angle_dis_matrix.transpose(-1, -2)

        rho_dis_matrix = torch.triu(angle_dis_matrix, diagonal=1)
        rho_dis_matrix = rho_dis_matrix + rho_dis_matrix.transpose(-1, -2)

        dis_matrix = torch.stack((angle_dis_matrix, rho_dis_matrix), dim=-1)

        masks = (dis_matrix[..., 0]<self.o2o_angle_thres) & (dis_matrix[..., 1]<self.o2o_rho_thres)
        return masks
    
class TripletHead(nn.Module):
    def __init__(self, cfg=None):
        super(TripletHead, self).__init__()
        self.img_w = cfg.img_w
        self.num_offsets = cfg.num_offsets
        self.num_strips = self.num_offsets - 1
        self.num_feat_samples = cfg.num_feat_samples
        self.fc_hidden_dim = cfg.fc_hidden_dim
        self.num_line_groups = cfg.num_line_groups
        self.prior_feat_channels = cfg.prior_feat_channels

        
        self.layer_logits = nn.Parameter(torch.zeros(3, self.num_feat_samples), requires_grad=True)
        self.softmax = nn.Softmax(dim=0)
        self.fc = nn.Linear(self.num_feat_samples*self.prior_feat_channels, self.fc_hidden_dim)

        self.cls_block = nn.Sequential(nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim), nn.ReLU())
        self.reg_block = nn.Sequential(nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim), nn.ReLU())
        
        self.cls_layer = nn.Linear(self.fc_hidden_dim, 1)
        self.reg_layer = nn.Linear(self.fc_hidden_dim, self.num_offsets + 2)
        self.aux_reg_layer = nn.Linear(self.fc_hidden_dim, 2*self.num_line_groups)

        self.o2o_cls_head = O2OClsHead(cfg)
        self.sigmoid = nn.Sigmoid()
        
        self.init_weights()
    
    def init_weights(self):
        for p in self.cls_layer.parameters():
            nn.init.normal_(p, mean=0., std=1e-3)
        for p in self.reg_layer.parameters():
            nn.init.normal_(p, mean=0., std=1e-3)
        for p in self.aux_reg_layer.parameters():
            nn.init.normal_(p, mean=0., std=1e-3)
        nn.init.normal_(self.layer_logits, mean=0, std=0.3)

    def pool_roi_features(self, feat_list, sample_points):
        batch_num, num_priors, num_points, _ = sample_points.shape
        nun_channel = feat_list[0].shape[1]
        num_levels = len(feat_list)
        sample_points_xy = (sample_points * 2.0 - 1.0).float()
        layer_weight = self.softmax(self.layer_logits)
        out = sample_points.new_zeros(batch_num, num_priors, num_points, nun_channel)
        for i in range(num_levels):
            sample_feat = F.grid_sample(feat_list[i], sample_points_xy, mode="bilinear", padding_mode="zeros", align_corners=True)
            sample_feat = sample_feat * layer_weight[i]
            out += sample_feat.permute(0, 2, 3, 1)
        out = out.view(batch_num, num_priors, -1)
        out = self.fc(out)
        return out

    def forward(self, feat_list, lane_points_img, anchor_id, anchor_embeddings):
        batch_anchor_features = self.pool_roi_features(feat_list, lane_points_img)

        cls_features = self.cls_block(batch_anchor_features)
        logits= self.cls_layer(cls_features).squeeze(-1)
        cls_pred = self.sigmoid(logits)
        
        reg_features = self.reg_block(batch_anchor_features)
        reg_pred = self.reg_layer(reg_features)
        reg_pred_aux = self.aux_reg_layer(reg_features)

        cls_pred_o2o = self.o2o_cls_head(batch_anchor_features, cls_pred, anchor_id, anchor_embeddings)
        return cls_pred, reg_pred, reg_pred_aux, cls_pred_o2o
    

    

    

        

