import torch
from torch.utils.data import Dataset
import math
import random
import numpy as np
import albumentations as A
from scipy.interpolate import InterpolatedUnivariateSpline as interp
from utils.coord_transform import CoordTrans
from utils.lane_utils import clipline_out_of_image, points_to_lineseg

from utils.ploter import Ploter
import cv2
import os 

class BaseTrSet(Dataset):
    def __init__(self, cfg=None, transforms=None):
        self.cfg = cfg
        random.seed(cfg.random_seed)
        self.img_h, self.img_w = cfg.img_h, cfg.img_w
        self.center_h, self.center_w = cfg.center_h, cfg.center_w
        self.max_lanes = cfg.max_lanes
        self.num_offsets = cfg.num_offsets
        self.num_line_groups = cfg.num_line_groups
        self.polar_map_size = cfg.polar_map_size

        self.ploter = Ploter(cfg=cfg)
        self.coord_trans = CoordTrans(cfg) 
        self.transforms = transforms

        img_transforms = []
        self.aug_names = cfg.train_augments
        for aug in self.aug_names:
            if aug['name'] != 'OneOf':
                img_transforms.append(getattr(A, aug['name'])(**aug['parameters']))
            else:
                img_transforms.append(A.OneOf([getattr(A, aug_['name'])(**aug_['parameters'])
                                      for aug_ in aug['transforms']], p=aug['p']))
        self.train_augments = A.Compose(img_transforms, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        self.sample_y_car = self.center_h - np.linspace(0, 1-1e-5, num = self.num_offsets)*self.img_h
        self.sample_y_car_reverse = self.sample_y_car[::-1]

        self.plot_no = 1
        
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):
        img, lanes = self.get_sample(index)
        img, lanes = self.augment(img, lanes)
        lanes = self.extend_lane2boundary(lanes)
        num_lanes = len(lanes)
        plot_img = img
        
        data_dict = dict()
        data_dict['img'] = self.transforms(img)

        lane_valid = np.zeros((self.max_lanes), dtype=np.bool_)
        lane_valid[:len(lanes)] = True
        end_points_pad = np.zeros((self.max_lanes, 2), dtype=np.float32)
        line_paras_pad = np.zeros((self.max_lanes, 2), dtype=np.float32)
        line_paras_group_pad = np.zeros((self.max_lanes, self.num_line_groups, 2))
        group_validmask_pad = np.zeros((self.max_lanes, self.num_line_groups))
        polar_map = np.zeros((3, self.polar_map_size[0], self.polar_map_size[1]))
        lane_point_xs_pad = np.zeros((self.max_lanes, self.num_offsets), dtype=np.float32)
        lane_point_validmask_pad = np.zeros((self.max_lanes, self.num_offsets), dtype=np.float32)
        

        if num_lanes>0:
            lanes_car = [self.coord_trans.img2cartesian(lane) for lane in lanes]
            _, lane_point_xs, lane_point_validmask, end_points, lane_dense_sample_car = self.fit_lane(lanes_car, is_sample=True)
            line_paras, _ = self.curve2line_group(lane_point_xs, lane_point_validmask, 1)
            line_paras = np.squeeze(line_paras, axis=1)
            line_paras_group, group_valid_masks = self.curve2line_group(lane_point_xs, lane_point_validmask, self.num_line_groups)
            polar_map = self.get_polar_map(lane_dense_sample_car)

            line_paras_pad[:num_lanes] = line_paras
            line_paras_group_pad[:num_lanes] = line_paras_group
            group_validmask_pad[:num_lanes] = group_valid_masks
            lane_point_xs_pad[:num_lanes] = lane_point_xs
            lane_point_validmask_pad[:num_lanes] = lane_point_validmask
            end_points_pad[:num_lanes] = end_points

            # img = self.ploter.plot_lines(plot_img, line_paras=line_paras_group.reshape(-1, 2))
            # img = self.ploter.plot_lines_group(plot_img, line_paras_group2, num_group=self.num_line_groups)
            # img = self.ploter.plot_lanes(img, lanes_car, color=(0, 0, 255))
            # img = self.ploter.plot_lanes_xs(img, lane_point_xs, lane_point_validmask, color=(0, 0, 255))

            # print(img.shape)
            # cv2.imwrite(os.path.join('./img', f'{str(self.plot_no)}.jpg'), img)
            # self.plot_no+=1                        
            
        data_dict['line_paras'] = line_paras_pad 
        data_dict['lane_valid'] = lane_valid
        data_dict['plot_img'] = plot_img
        data_dict['end_point'] = end_points_pad
        data_dict['line_paras_group'] = line_paras_group_pad
        data_dict['group_validmask'] = group_validmask_pad
        data_dict['polar_map'] = polar_map
        data_dict['lane_point_xs'] = lane_point_xs_pad
        data_dict['lane_point_validmask'] = lane_point_validmask_pad

        return data_dict

    def get_lane_endpoints(self, fit_paras):
        return fit_paras[:, (1, 0), 1]-self.center_h+self.img_h

    def get_polar_map(self, xy_car):
        rho_thres = self.img_w/self.polar_map_size[1]/2
        xy_car_diff = -np.diff(xy_car, axis=1)
        angle = np.arctan2(xy_car_diff[..., 1], xy_car_diff[..., 0])/math.pi-0.5
        xy_car = xy_car[..., :-1, :]
        angle, xy_car = angle.reshape(-1), xy_car.reshape(-1, 2)
        grid_x, grid_y = np.meshgrid(np.arange(0.5, self.polar_map_size[1], 1, dtype=np.float32), np.arange(0.5, self.polar_map_size[0], 1, dtype = np.float32))
        grid_x, grid_y = grid_x.flatten()/self.polar_map_size[1]*self.img_w, grid_y.flatten()/self.polar_map_size[0]*self.img_h
        grid_car = self.coord_trans.img2cartesian(np.stack((grid_x, grid_y), axis=-1))

        min_ind = np.argmin(np.linalg.norm(grid_car[:, np.newaxis, :] - xy_car[np.newaxis, ...], axis=-1), axis=-1)

        min_xy_car, min_angle = xy_car[min_ind], angle[min_ind]
        local_xy_car = min_xy_car - grid_car
        rho = np.linalg.norm(local_xy_car, axis=-1)
        rho[local_xy_car[..., 0]<0] *= -1

        rho, angle = rho.reshape(self.polar_map_size), min_angle.reshape(self.polar_map_size)
        valid_mask = np.abs(rho)<rho_thres
        
        polar_map = np.zeros((3, self.polar_map_size[0], self.polar_map_size[1]))
        polar_map[0, valid_mask] = 1
        polar_map[1, valid_mask] = angle[valid_mask] 
        polar_map[2, valid_mask] = rho[valid_mask]/(self.img_w/self.polar_map_size[1])
         
        return polar_map

    
    def augment(self, img, lanes):
        if len(lanes)>0:
            lane_lengths = [len(lane) for lane in lanes]
            keypoints = np.concatenate(lanes, axis=0)
            content = self.train_augments(image=img, keypoints=keypoints)
            keypoints = np.array(content['keypoints'])
            start_dim = 0
            lanes = []
            for lane_length in lane_lengths:
                lane = keypoints[start_dim:start_dim+lane_length]
                lanes.append(lane)
                start_dim += lane_length
        else:
            content = self.train_augments(image=img)
        img = content['image']
        clip_lanes = []
        img_shape = (img.shape[0], img.shape[1])
        for lane in lanes:
            lane = clipline_out_of_image(line_coords=lane, img_shape=img_shape)
            if lane is not None:
                clip_lanes.append(lane)
        lanes = clip_lanes
        return img, lanes
    
    def extend_lane2boundary(self, lanes):
        bound_h, bound_w = self.img_h-1, self.img_w-1
        extend_lanes = []
        for lane in lanes:
            if lane.shape[0]<2:
                continue
            point1, point2 = lane[-2, :], lane[-1, :]
            x1, x2 = point1[0], point2[0]
            y1, y2 = point1[1], point2[1]
            if x1==x2:
                end_axis_x, end_axis_y = x1, bound_h
            elif x2>x1:
                end_axis_y = min(bound_h, (bound_w-x2)/(x2-x1)*(y2-y1)+y2)
                end_axis_x = min(bound_w, ((bound_h-y2)/(y2-y1)*(x2-x1)+x2))
            else:
                end_axis_y = min(bound_h, (-x2)/(x2-x1)*(y2-y1)+y2)
                end_axis_x = max(0, ((bound_h-y2)/(y2-y1)*(x2-x1)+x2))
                
            end_point = np.array([end_axis_x, end_axis_y])
            margin = np.linalg.norm((end_point-point2))
            if len(lane)>5:
                if margin<3 or margin>1000:
                    extend_lanes.append(lane)
                else:
                    extend_lanes.append(np.vstack((lane, end_point)))
        return extend_lanes

    def fit_lane(self, lanes, is_sample=False):
        curves = []
        lane_point_xs_list = []
        lane_point_validmask_list = []
        end_points_list = []
        lane_dense_sample_list = []

        for lane in lanes:
            lane = lane[np.unique(lane[:, 1], return_index=True)[1]]
            lane = lane[lane[:, 1].argsort()]

            x, y = lane[..., 0], lane[..., 1]
            y_start, y_end = y[0], y[-1]
            spline = interp(y, x, k=1)
            
            x_fit = spline(self.sample_y_car_reverse)
            lane_point_validmask = (y_start<self.sample_y_car_reverse)&(self.sample_y_car_reverse<y_end)

            y_dense_sample = np.linspace(y_end, y_start, 101)
            x_dense_sample = spline(y_dense_sample)
            lane_dense_sample = np.stack((x_dense_sample, y_dense_sample), axis=-1)            

            curves.append(spline)
            lane_point_xs_list.append(x_fit)
            lane_point_validmask_list.append(lane_point_validmask)
            end_points_list.append([y_start, y_end])
            lane_dense_sample_list.append(lane_dense_sample)

        lane_point_xs = np.stack(lane_point_xs_list, axis=0)
        lane_point_validmask = np.stack(lane_point_validmask_list, axis=0)
        end_points = np.array(end_points_list, dtype=np.float32)-self.center_h + self.img_h
        lane_dense_sample = np.stack(lane_dense_sample_list, axis=0)
        return curves, lane_point_xs, lane_point_validmask, end_points, lane_dense_sample
            
    
    def curve2line_group(self, lane_point_xs, lane_point_validmask, num_group=1):
        lane_point_xs = lane_point_xs[..., ::-1]
        lane_point_validmask = lane_point_validmask[..., ::-1]
        sample_y_car = np.repeat(self.sample_y_car[np.newaxis, ...], lane_point_xs.shape[0], axis=0)
        samples_car = np.stack((lane_point_xs, sample_y_car), axis=-1)
        samples_car = samples_car.astype(np.float32)
        line_paras_group, seg_valid_masks = points_to_lineseg(samples_car, lane_point_validmask, num_group=num_group)
        return line_paras_group, seg_valid_masks 
    
    def collate_fn(self, data_dict_list):
        batch_list_dict = {}
        batch_dict = {}
        for key in data_dict_list[0].keys():
            if 'list' in key:
                batch_list_dict[key] = [[] for _ in range(len(data_dict_list[0][key]))]
            else:
                batch_list_dict[key] = []
        for data_dict in data_dict_list:
            for key in data_dict.keys():
                if key=='img' and self.transforms is not None:
                    batch_list_dict[key].append(data_dict[key].unsqueeze(0))
                elif 'list' in key:
                    for i in range(len(data_dict[key])):
                        batch_list_dict[key][i].append(torch.from_numpy(data_dict[key][i]).unsqueeze(0))
                else:
                    batch_list_dict[key].append(torch.from_numpy(data_dict[key]).unsqueeze(0))
        for key in batch_list_dict.keys():
            if 'list' in key:
                batch_dict[key] = [torch.cat(data, dim=0) for data in batch_list_dict[key]] 
            else: 
                batch_dict[key] = torch.cat(batch_list_dict[key], dim=0)
        return batch_dict





class BaseTsSet(Dataset):
    def __init__(self, cfg=None, transforms=None):
        self.data_root = cfg.data_root
        self.cut_height = cfg.cut_height
        self.transforms = transforms
        self.cut_height = cfg.cut_height
        self.is_val = cfg.is_val
        self.is_view = cfg.is_view
        self.img_path_list = []
        self.file_name_list = []

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        ori_img = cv2.imread(self.img_path_list[index])
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img = ori_img[self.cut_height:]
        if self.transforms is not None:
            img = self.transforms(img)
        if not self.is_view:
            ori_img = None
        return img, self.file_name_list[index], ori_img

    def collate_fn(self, samples):
        img_list = []
        ori_img_list = []
        file_name_list = []
        for img, file_name, ori_img in samples:
            img_list.append(img.unsqueeze(0))
            file_name_list.append(file_name)
            if ori_img is not None:
                ori_img_list.append(torch.from_numpy(ori_img))
        imgs = torch.cat(img_list, dim=0)
        return imgs, file_name_list, ori_img_list


    
    
        