import torch 
import numpy as np
import os
import cv2
import copy
import json
from scipy.interpolate import InterpolatedUnivariateSpline as interp
from .base_dataset import BaseTrSet, BaseTsSet
# from tqdm import tqdm
# from random import shuffle

stno = 0

def inter(lanes):
    lanes_new_list = []
    for lane in lanes:
        lane = lane[np.unique(lane[:, 1], return_index=True)[1]]
        lane = lane[lane[:, 1].argsort()]
        x, y = lane[..., 0], lane[..., 1]
        if y.shape[0]<=1:
            continue
        y_start, y_end = y[0], y[-1]
        spline = interp(y, x, k=1)
        y_dense_sample = np.linspace(y_start, y_end, 20)
        x_dense_sample = spline(y_dense_sample)
        lane_dense_sample = np.stack((x_dense_sample, y_dense_sample), axis=-1)            
        lanes_new_list.append(lane_dense_sample)
    return lanes_new_list


class CurveLanesTrSet(BaseTrSet):
    def __init__(self, cfg=None, transforms=None):
        super().__init__(cfg=cfg, transforms=transforms)
        self.data_root = cfg.data_root
        self.img_path_list, self.label_list = self.get_data_list()
        self.cut_height = cfg.cut_height
        self.ori_img_h = cfg.ori_img_h
        self.ori_img_w = cfg.ori_img_w
        self.cut_height_dict = cfg.cut_height_dict
 
    def get_sample(self, index):
        img_path, lanes = self.img_path_list[index], copy.deepcopy(self.label_list[index])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, lanes = self.cut_img(img, lanes)
        return img, lanes

    def get_data_list(self):
        img_path_list = []
        label_list = []
        train_list_txt = os.path.join(self.data_root, 'train/train.txt')
        with open(train_list_txt, 'r') as f:
            data_path_list = f.readlines()

        # data_path_list = data_path_list[stno:stno+96]
        # path_list_new = []
        # for i in range(300):
        #     shuffle(data_path_list)
        #     path_list_new += data_path_list
        # data_path_list = path_list_new

        for data_path in data_path_list:
            img_path = os.path.join(self.data_root, 'train', data_path).strip('\n')
            label_path = img_path.replace('images', 'labels').replace('jpg', 'lines.json')
            label = self.get_label(label_path)
            label_list.append(label)
            img_path_list.append(img_path)
        # for i in tqdm(range(len(label_list))):
        #     label_list[i] = inter(label_list[i])

        from multiprocessing import Pool, cpu_count
        with Pool(cpu_count()) as p:
            label_list = p.starmap(inter, zip(label_list))
            
        return img_path_list, label_list
    
    def get_label(self, label_path):
        with open(label_path, 'r') as f:
            lanes_json = json.load(f)['Lines']
        lane_list = []
        for lane_json in lanes_json:
            point_list = []
            for point_json in lane_json:
                x, y = float(point_json['x']), float(point_json['y'])
                point_list.append([x, y])
            lane = np.array(point_list)[::-1]
            lane_list.append(lane)
        return lane_list
    
    # def cut_img(self, img, lanes):
    #     img = img[self.cut_height:]
    #     lanes = lanes.copy()
    #     for lane in lanes:
    #         lane[:, 1] = lane[:, 1] - self.cut_height
    #     return img, lanes

    def cut_img(self, img, lanes):
        cut_height = self.cut_height_dict[img.shape]
        img = img[cut_height:]
        for lane in lanes:
            lane[:, 1] = lane[:, 1] - cut_height
        return img, lanes


class CurveLanesTsSet(BaseTsSet):
    def __init__(self, cfg=None, transforms=None):
        super().__init__(cfg=cfg, transforms=transforms) 
        self.cut_height_dict = cfg.cut_height_dict
        if not self.is_val:
            print('The test set is not available, use the valid set instead!')
        self.img_root = os.path.join(self.data_root, 'valid')
        self.img_path_list = self.get_data_list()
        self.file_name_list = [img_path.replace(self.img_root + '/', '') for img_path in self.img_path_list]
 
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        ori_img = cv2.imread(img_path)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        ori_img_shape = ori_img.shape
        cut_height = self.cut_height_dict[ori_img_shape]
        img = ori_img[cut_height:]
        if self.transforms is not None:
            img = self.transforms(img)
        file_name = self.file_name_list[index]
        if not self.is_view:
            ori_img = None
        return img, (file_name, ori_img_shape), ori_img
    
    def get_data_list(self):
        img_path_list = []
        data_list_txt = os.path.join(self.data_root, 'valid/valid.txt')
        with open(data_list_txt, 'r') as f:
            data_path_list = f.readlines()

        for data_path in data_path_list:
            img_path = os.path.join(self.data_root, 'valid', data_path).strip('\n')
            img_path_list.append(img_path)
            
        return img_path_list 



class CurveLanesTrSetView():
    def __init__(self, cfg=None, transforms=None):
        super().__init__()
        self.data_root = cfg.data_root
        self.img_path_list, self.label_list = self.get_data_list()
        self.cut_height = cfg.cut_height
        self.transforms = transforms
        self.ori_img_h = cfg.ori_img_h
        self.ori_img_w = cfg.ori_img_w
        self.cut_height_dict = cfg.cut_height_dict

    
    def __getitem__(self, index):
        img_path, lanes = self.img_path_list[index], self.label_list[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lanes = self.inter(lanes)
        ori_img = img
        img = self.cut_img(img)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, lanes, ori_img
    
    # def process_small_img(self, img, lanes):
    #     img_h, img_w = img.shape[0], img.shape[1]
    #     img = cv2.resize(img, dsize=(self.ori_img_w, self.ori_img_h), interpolation=cv2.INTER_CUBIC)
    #     lanes_resize = []
    #     for lane in lanes:
    #         lane[..., 0] *= (self.ori_img_w/img_w)
    #         lane[..., 1] *= (self.ori_img_h/img_h)
    #         lanes_resize.append(lane)
    #     return img, lanes_resize
    
    def cut_img(self, img):
        cut_height = self.cut_height_dict[img.shape]
        img = img[cut_height:]
        return img
        

    def inter(self, lanes):
        lanes_new_list = []
        for lane in lanes:
            lane = lane[np.unique(lane[:, 1], return_index=True)[1]]
            lane = lane[lane[:, 1].argsort()]
            x, y = lane[..., 0], lane[..., 1]
            y_start, y_end = y[0], y[-1]
            spline = interp(y, x, k=1)
            y_dense_sample = np.linspace(y_start, y_end, 40)
            x_dense_sample = spline(y_dense_sample)
            lane_dense_sample = np.stack((x_dense_sample, y_dense_sample), axis=-1)            
            lanes_new_list.append(lane_dense_sample)
        return lanes_new_list
    
    def __len__(self):
        return 4*24#len(self.img_path_list)
    
    def get_data_list(self):
        img_path_list = []
        label_list = []
        train_list_txt = os.path.join(self.data_root, 'valid/valid.txt')
        with open(train_list_txt, 'r') as f:
            data_path_list = f.readlines()

        data_path_list = data_path_list[stno:stno+96]

        for data_path in data_path_list:
            img_path = os.path.join(self.data_root, 'valid', data_path).strip('\n')
            label_path = img_path.replace('images', 'labels').replace('jpg', 'lines.json')
            with open(label_path, 'r') as f:
                lanes_json = json.load(f)['Lines']
            label = self.get_lane(lanes_json)
            label_list.append(label)
            img_path_list.append(img_path)
        return img_path_list, label_list
    def get_lane(self, lanes_json):
        lane_list = []
        for lane_json in lanes_json:
            point_list = []
            for point_json in lane_json:
                x, y = float(point_json['x']), float(point_json['y'])
                point_list.append([x, y])
            lane = np.array(point_list)[::-1]
            lane_list.append(lane)
        return lane_list
    
    def get_label(self, sample_dict):
        lane_xs = sample_dict['lanes']
        ys = sample_dict['h_samples']
        label = []

        for lane_x in lane_xs:
            lane_array = np.array([lane_x, ys]).transpose()
            ind = np.where((lane_array[:, 0] >=0))
            lane_array = lane_array[ind]
            label.append(lane_array)
        return label
    
    
    def collate_fn(self, item_list):
        img_list = []
        ori_img_list = []
        lane_gt_list = [] 
        for img, lanes, ori_img in item_list:
            img_list.append(img.unsqueeze(0))
            if ori_img is not None:
                ori_img_list.append(ori_img)
            lane_gt_list.append(lanes)
        imgs = torch.cat(img_list, dim=0)
        return imgs, lane_gt_list, ori_img_list


