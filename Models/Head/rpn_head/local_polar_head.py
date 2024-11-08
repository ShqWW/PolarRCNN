import math
import torch
from torch import nn
from utils.coord_transform import CoordTrans_torch

class LocalPolarHead(nn.Module):
    def __init__(self, cfg):
        super(LocalPolarHead, self).__init__()
        self.n_offsets = cfg.num_offsets
        self.in_channel = cfg.rpn_inchannel
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.center_h, self.center_w = cfg.center_h, cfg.center_w
        self.polar_map_h, self.polar_map_w = cfg.polar_map_size
        self.num_priors = self.polar_map_h * self.polar_map_w
        self.num_training_priors = cfg.num_training_priors
        self.num_testing_priors = cfg.num_testing_priors
        self.angle_noise_p, self.rho_noise_p = cfg.angle_noise_p, cfg.rho_noise_p

        self.coord_trans = CoordTrans_torch(cfg)
        self.upsample = nn.UpsamplingBilinear2d(size=[self.polar_map_h, self.polar_map_w])
        self.reg_layers = nn.Conv2d(self.in_channel, 2, 1, 1, 0, bias=False)
        self.cls_layers = nn.Sequential(nn.Conv2d(self.in_channel, 64, 1, 1, 0),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 1, 1, 1, 0))

        grid_y, grid_x = torch.meshgrid(torch.arange(0.5, self.polar_map_h, 1, dtype=torch.float32),
                                        torch.arange(0.5, self.polar_map_w, 1, dtype=torch.float32),
                                        indexing="ij")
        grid = torch.stack((grid_x*self.img_w/self.polar_map_w, grid_y*self.img_h/self.polar_map_h), dim=-1)
        grid_car = self.coord_trans.img2cartesian(grid)
        self.register_buffer(name="grid_car", tensor=grid_car)
        self.register_buffer(name='prior_id', tensor=torch.linspace(0, self.num_priors-1, self.num_priors, dtype=torch.int32))
        
    def forward(self, feats):
        x = self.upsample(feats[-1])
        polar_map_reg = self.reg_layers(x)
        polar_map_cls = self.cls_layers(x.detach())  # stop the grad of the cls block
        polar_map_reg = torch.arctan(polar_map_reg)/math.pi
        polar_map = torch.cat((polar_map_cls, polar_map_reg), dim=1)
        
        if self.training:
            # add noise to the local anchor parameters
            polar_map_reg_rand = polar_map_reg.detach().clone()
            polar_map_reg_rand[:, 0, ...] += (torch.rand_like(polar_map_reg_rand[:, 0, ...])*self.angle_noise_p-self.angle_noise_p/2)
            polar_map_reg_rand[:, 1, ...] += (torch.rand_like(polar_map_reg_rand[:, 1, ...])*self.rho_noise_p - self.rho_noise_p/2)
            anchor_embeddings, anchor_id = self.local2global(polar_map_reg_rand.detach().clone(), polar_map_cls.detach().sigmoid().squeeze(1), top_k=self.num_training_priors)
            pred_dict = {'polar_map':polar_map, 'anchor_embeddings':anchor_embeddings, 'anchor_id': anchor_id}
        else:
            anchor_embeddings, anchor_id = self.local2global(polar_map_reg.detach().clone(), polar_map_cls.detach().sigmoid().squeeze(1), top_k=self.num_testing_priors)
            pred_dict = {'anchor_embeddings':anchor_embeddings, 'anchor_id': anchor_id}
        return pred_dict

    def local2global(self, polar_map_reg, polar_map_cls, top_k):
        angle, local_rho = polar_map_reg[:, 0, ...]* math.pi , polar_map_reg[:, 1, ...] * self.img_w/self.polar_map_w
        global_rho = local_rho + self.grid_car[..., 0] * torch.cos(angle) + self.grid_car[..., 1]*torch.sin(angle)
        angle = angle/math.pi
        rho = global_rho/self.img_w
        angle = angle.clamp_(min=-0.45, max=0.45)
        anchor_embeddings = torch.stack((angle, rho), axis=-1).flatten(1, 2)
        anchor_id = self.prior_id.unsqueeze(0).repeat(anchor_embeddings.shape[0], 1)

        if top_k == self.num_priors:
            return anchor_embeddings, anchor_id
        
        cls_score = polar_map_cls.flatten(-2, -1)
        _, batch_top_k_ind = torch.topk(cls_score, k=top_k,  dim=-1, largest=True, sorted=False)
        
        # to be optimized
        anchor_embeddings = torch.stack([anchor_embedding[top_k_ind] for anchor_embedding, top_k_ind in zip(anchor_embeddings, batch_top_k_ind)], dim=0)
        anchor_id = torch.stack([anchor_id[top_k_ind] for anchor_id, top_k_ind in zip(anchor_id, batch_top_k_ind)], dim=0)
        return anchor_embeddings, anchor_id

