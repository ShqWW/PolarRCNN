import torch 
import numpy as np
import os
import cv2
from .base_dataset import BaseTrSet, BaseTsSet
# from random import shuffle
# from tqdm import tqdm


class CULaneTrSet(BaseTrSet):
    def __init__(self, cfg=None, transforms=None):
        super().__init__(cfg=cfg, transforms=transforms)
        self.data_root = cfg.data_root
        self.img_path_list, self.label_path_list = self.get_data_list()
        self.cut_height = cfg.cut_height
    
    def get_sample(self, index):
        img_path, label_path = self.img_path_list[index], self.label_path_list[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lanes = self.get_label(label_path)
        img, lanes = self.cut_img(img, lanes)
        return img, lanes
    
    def get_data_list(self):
        list_path = os.path.join(self.data_root, 'list/train_gt_new.txt')
        with open(list_path, 'r') as f:
            path_list = [line.strip(' \n').split(' ')[0][1:] for line in f.readlines()]

        # start = 1000
        # path_list_ori = path_list[start:start+96]#?????????????????????????????????????????1
        # path_list = path_list_ori

        # path_list_new = []
        # for i in range(300):
        #     shuffle(path_list)
        #     path_list_new += path_list
        # path_list = path_list_new

        img_path_list = [os.path.join(self.data_root, path) for path in path_list]
        # mask_path_list = [os.path.join(self.data_root, 'laneseg_label_w16/', path.strip(' \n').replace('.jpg', '.png')) for path in path_list]
        label_path_list = [os.path.join(self.data_root, path.replace('.jpg', '.lines.txt')) for path in path_list]
        return img_path_list, label_path_list

    def get_label(self, label_path):
        with open(label_path, 'r') as f:
            lane_strs = f.readlines()
        lane_arrays = []
        for lane_str in lane_strs:
            lane_array = np.array(lane_str.strip(' \n').split(' ')).astype(np.float32)
            lane_array_size = int(len(lane_array)/2)
            lane_array = lane_array.reshape(lane_array_size, 2)[::-1, :]
            ind = np.where((lane_array[:, 0] >=0)&(lane_array[:, 1] >=0))
            lane_array = lane_array[ind]
            if lane_array.shape[0]>2:
                lane_arrays.append(lane_array)
        return lane_arrays
    
    def cut_img(self, img, lanes):
        img = img[self.cut_height:]
        for lane in lanes:
            lane[:, 1] = lane[:, 1] - self.cut_height
        return img, lanes

class CULaneTsSet(BaseTsSet):
    def __init__(self, cfg=None, transforms=None):
        super().__init__(cfg=cfg, transforms=transforms) 
        if self.is_val:
            self.txt_path = os.path.join(self.data_root, 'list/val.txt')
        else:
            self.txt_path = os.path.join(self.data_root, 'list/test.txt')
        with open(self.txt_path, 'r') as f:
            for line in f.readlines():
                self.file_name_list.append(line.strip('\n')[1:])
        self.img_path_list = [os.path.join(self.data_root, file_name) for file_name in self.file_name_list]


class CULaneTrSetView():
    def __init__(self, cfg=None, transforms=None):
        self.data_root = cfg.data_root
        self.img_path_list, self.mask_path_list, self.label_path_list = self.get_data_list()
        self.cut_height = cfg.cut_height
        self.img_h = cfg.img_h
        self.img_w = cfg.img_w
        self.transforms = transforms
        
    
    def __getitem__(self, index):
        img_path, mask_path, label_path = self.img_path_list[index], self.mask_path_list[index], self.label_path_list[index]
        # print(img_path)
        img = cv2.imread(img_path)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ori_img = img.copy()
        # mask = np.zeros_like(img)
        lanes = self.get_label(label_path)
        img, lanes = self.cut_img(img, lanes)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, lanes, ori_img
    def __len__(self):
        return 4*24#len(self.img_path_list)
    
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
    
    def get_data_list(self):
        list_path = os.path.join(self.data_root, 'list/train_gt_new.txt')
        with open(list_path, 'r') as f:
            path_list = [line.strip(' \n').split(' ')[0][1:] for line in f.readlines()]
        path_list = path_list[1000:]#???????????????????????????????????
        img_path_list = [os.path.join(self.data_root, path) for path in path_list]
        mask_path_list = [os.path.join(self.data_root, 'laneseg_label_w16/', path.strip(' \n').replace('.jpg', '.png')) for path in path_list]
        label_path_list = [os.path.join(self.data_root, path.replace('.jpg', '.lines.txt')) for path in path_list]
        # print(img_path_list[100:124])
        # print(len(img_path_list))
        # return img_path_list[stno:], mask_path_list[stno:], label_path_list[stno:]
        return img_path_list, mask_path_list, label_path_list
    
    def get_label(self, label_path):
        with open(label_path, 'r') as f:
            lane_strs = f.readlines()
        lane_arrays = []
        for lane_str in lane_strs:
            lane_array = np.array(lane_str.strip(' \n').split(' ')).astype(np.float32)
            lane_array_size = int(len(lane_array)/2)
            lane_array = lane_array.reshape(lane_array_size, 2)[::-1, :]
            ind = np.where((lane_array[:, 0] >=0)&(lane_array[:, 1] >=0))
            lane_array = lane_array[ind]
            if lane_array.shape[0]>2:
                lane_arrays.append(lane_array)
        return lane_arrays
    
    
    def cut_img(self, img, lanes):
        img = img[self.cut_height:]
        return img, lanes
    
    



    
    

    
