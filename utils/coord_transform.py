import numpy as np
import torch

class CoordTrans():
    def __init__(self, cfg):
        self.img_h, self.img_w = cfg.img_h, cfg.img_w
        self.center_h = cfg.center_h
        self.center_w = cfg.center_w
        self.num_offsets = cfg.num_offsets
    
    def img2cartesian(self, points):
        points_new = np.zeros_like(points)
        points_new[..., 0] = points[..., 0] - self.center_w
        points_new[..., 1] = self.center_h - points[..., 1]
        return points_new
    
    def cartesian2img(self, points):
        points_new = np.zeros_like(points)
        points_new[..., 0] = points[..., 0] + self.center_w
        points_new[..., 1] = self.center_h - points[..., 1]
        return points_new

class CoordTrans_torch(CoordTrans):
    def __init__(self, cfg):
        super(CoordTrans_torch, self).__init__(cfg)
        sample_img_y = torch.linspace(0, 1-1e-5, steps=self.num_offsets, dtype=torch.float32)*self.img_h
        sample_car_y = self.center_h - sample_img_y
        self.sample_img_y = sample_img_y.cuda()
        self.sample_car_y = sample_car_y.cuda()

    def img2cartesian(self, points):
        points_new = torch.zeros_like(points)
        points_new[..., 0] = points[..., 0] - self.center_w
        points_new[..., 1] = self.center_h - points[..., 1]
        return points_new
    
    def cartesian2img(self, points):
        points_new = torch.zeros_like(points)
        points_new[..., 0] = points[..., 0] + self.center_w
        points_new[..., 1] = self.center_h - points[..., 1]
        return points_new

    def sample_xs_by_fix_ys(self, line_paras):
        angles, rs = line_paras[..., 0], line_paras[..., 1]
        sample_car_y = self.sample_car_y.unsqueeze(0).repeat(angles.shape[0], 1)
        sample_car_x = -torch.tan(angles).unsqueeze(-1)*sample_car_y + (rs/torch.cos(angles)).unsqueeze(-1)
        sample_car = torch.stack((sample_car_x, sample_car_y), dim=-1)
        return sample_car
    
    
        
