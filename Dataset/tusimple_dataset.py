import torch 
import numpy as np
import os
import cv2
import copy
from .base_dataset import BaseTrSet, BaseTsSet
# from random import shuffle

stno = 1000
class TuSimpleTrSet(BaseTrSet):
    def __init__(self, cfg=None, transforms=None):
        super().__init__(cfg=cfg, transforms=transforms)
        self.data_root = cfg.data_root
        self.label_jsons = ['label_data_0313.json', 'label_data_0531.json', 'label_data_0601.json']
        self.img_path_list, self.mask_path_list, self.label_list = self.get_data_list()
        self.cut_height = cfg.cut_height
 
    def get_sample(self, index):
        img_path, lanes = self.img_path_list[index], copy.deepcopy(self.label_list[index])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, lanes = self.cut_img(img, lanes)
        return img, lanes
    
    def get_data_list(self):
        img_path_list = []
        label_list = []

        for label_json in self.label_jsons:
            with open(os.path.join(self.data_root, 'train_set/' + label_json)) as f:
                line_strs = f.readlines()
                for line_str in line_strs:
                    sample_dict = eval(line_str)
                    img_path = os.path.join(self.data_root, 'train_set/' + sample_dict['raw_file'])
                    img_path_list.append(img_path) 
                    label = self.get_label(sample_dict)
                    label_list.append(label)
        mask_path_list = [img_path.replace('clips', 'seg_label').replace('jpg', 'png') for img_path in img_path_list]

        # img_path_list, mask_path_list, label_list = img_path_list[stno:stno+24*4], mask_path_list[stno:stno+24*4], label_list[stno:stno+24*4]

        # all_list = []
        # for a, b, c in zip(img_path_list, mask_path_list, label_list):
        #     all_list.append((a, b, c))

        # path_list_new = []
        # for i in range(300):
        #     shuffle(all_list)
        #     path_list_new += all_list
        # all_list = path_list_new

        # img_path_list = []
        # mask_path_list = []
        # label_list = []

        # for a, b, c in all_list:
        #     img_path_list.append(a)
        #     mask_path_list.append(b)
        #     label_list.append(c)

        return img_path_list, mask_path_list, label_list
    
    def get_label(self, sample_dict):
        lane_xs = sample_dict['lanes']
        ys = sample_dict['h_samples']
        label = []

        for lane_x in lane_xs:
            lane_array = np.array([lane_x, ys]).transpose()
            ind = np.where((lane_array[:, 0] >=0))
            lane_array = lane_array[ind]
            if lane_array.shape[0]>0:
                label.append(lane_array)
        return label
    
    def cut_img(self, img, lanes):
        img = img[self.cut_height:]
        for lane in lanes:
            lane[:, 1] = lane[:, 1] - self.cut_height
        return img, lanes
    

class TusimpleTsSet(BaseTsSet):
    def __init__(self, cfg=None, transforms=None):
        super().__init__(cfg=cfg, transforms=transforms) 
        self.label_json = 'test_tasks_0627.json'
        self.file_name_list = self.get_data_list()        
        self.img_path_list = [os.path.join(self.data_root, 'test_set/' + file_name) for file_name in self.file_name_list]
        if self.is_val:
            print('No validset, use testset instead!')
    
    def get_data_list(self):
        file_name_list = []
        with open(os.path.join(self.data_root, 'test_set/' + self.label_json)) as f:
            line_strs = f.readlines()
            for line_str in line_strs:
                sample_dict = eval(line_str)
                img_path = sample_dict['raw_file']
                file_name_list.append(img_path.replace('\\', '')) 
        return file_name_list


class TuSimpleTrSetView():
    def __init__(self, cfg=None, transforms=None):
        self.data_root = cfg.data_root
        self.label_jsons = ['label_data_0313.json', 'label_data_0531.json', 'label_data_0601.json']
        self.img_path_list, self.mask_path_list, self.label_list = self.get_data_list()
        self.cut_height = cfg.cut_height
        self.img_h = cfg.img_h
        self.img_w = cfg.img_w
        self.transforms = transforms
        
    
    def __getitem__(self, index):
        img_path, mask_path, lanes = self.img_path_list[index], self.mask_path_list[index], copy.deepcopy(self.label_list[index])
        ori_img = cv2.imread(img_path)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

        img, lanes = self.cut_img(ori_img, lanes)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, lanes, ori_img
    def __len__(self):
        return 24*4#len(self.img_path_list)
    
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
        img_path_list = []
        label_list = []

        for label_json in self.label_jsons:
            with open(os.path.join(self.data_root, 'train_set/' + label_json)) as f:
                line_strs = f.readlines()
                for line_str in line_strs:
                    sample_dict = eval(line_str)
                    img_path = os.path.join(self.data_root, 'train_set/' + sample_dict['raw_file'])
                    img_path_list.append(img_path) 
                    label = self.get_label(sample_dict)
                    label_list.append(label)
        mask_path_list = [img_path.replace('clips', 'seg_label') for img_path in img_path_list]
        img_path_list, mask_path_list, label_list = img_path_list[stno:stno+24*4], mask_path_list[stno:stno+24*4], label_list[stno:stno+24*4]
        return img_path_list, mask_path_list, label_list
    
    def get_label(self, sample_dict):
        lane_xs = sample_dict['lanes']
        ys = sample_dict['h_samples']
        label = []

        for lane_x in lane_xs:
            lane_array = np.array([lane_x, ys]).transpose()
            ind = np.where((lane_array[:, 0] >=0))
            lane_array = lane_array[ind]
            if lane_array.shape[0]>0:
                label.append(lane_array)
        return label
    def cut_img(self, img, lanes):
        img = img[self.cut_height:]
        # for lane in lanes:
        #     lane[:, 1] = lane[:, 1] - self.cut_height
        return img, lanes
        


    