import torch 
import numpy as np
import os
import cv2
import copy
from .base_dataset import BaseTrSet, BaseTsSet
from utils.llamas_utils import get_horizontal_values_for_four_lanes
# from tqdm import tqdm
# from random import shuffle


stno = 1000

class LLAMASTrSet(BaseTrSet):
    def __init__(self, cfg=None, transforms=None):
        super().__init__(cfg=cfg, transforms=transforms)
        self.data_root = cfg.data_root
        self.ori_img_h = cfg.ori_img_h
        self.img_path_list, self.label_list = self.get_data_list()
        self.cut_height = cfg.cut_height
 
    def get_sample(self, index):
        img_path, lanes = self.img_path_list[index], copy.deepcopy(self.label_list[index])
        img = cv2.imread(img_path)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, lanes = self.cut_img(img, lanes)
        return img, lanes
    
    def get_data_list(self):
        img_path_list = []
        label_path_list = []
        label_list = []



        if True:   #the origin data list read
            label_root_path = os.path.join(self.data_root, 'labels/train/')
            sub_label_path_name_list = os.listdir(label_root_path)
            for sub_label_path_name in sub_label_path_name_list:
                sub_label_path = os.path.join(label_root_path, sub_label_path_name)
                label_name_list = os.listdir(sub_label_path)
                for label_name in label_name_list:
                    label_path = os.path.join(sub_label_path, label_name)
                    img_path = label_path.replace('labels', 'color_images').replace('.json', '_color_rect.png')
                    label_path_list.append(label_path)
                    img_path_list.append(img_path)
        
        # # get the data list by read the txt file
        # list_path = os.path.join(self.data_root, 'train_new.txt')
        # with open(list_path, 'r') as f:
        #     path_list = [line.strip(' \n')[1:] for line in f.readlines()]
        # img_path_list = [os.path.join(self.data_root, path) for path in path_list]
        # label_path_list = [os.path.join(self.data_root, path.replace('color_images', 'labels').replace('_color_rect.png', '.json')) for path in path_list]
        

        # img_path_list, label_path_list = img_path_list[stno:stno+24*4], label_path_list[stno:stno+24*4]



        ys = np.arange(0, self.ori_img_h, 1)
        for label_path in label_path_list:
            lanes = []
            xs_list = [np.array(xs) for xs in get_horizontal_values_for_four_lanes(label_path)]
            for xs in xs_list:
                mask = (xs>0)
                lane = np.stack((xs[mask], ys[mask]), axis=-1)
                lane = lane[np.unique(lane[:, 1], return_index=True)[1]]
                if lane.shape[0]>=2:
                    lane = lane[lane[:, 1].argsort()]
                    lanes.append(lane)
            label_list.append(lanes)

        # all_list = []
        # for a, c in zip(img_path_list, label_list):
        #     all_list.append((a, c))

        # path_list_new = []
        # for i in range(300):
        #     shuffle(all_list)
        #     path_list_new += all_list
        # all_list = path_list_new

        # img_path_list = []
        # label_list = []
        # for a, c in all_list:
        #     img_path_list.append(a)
        #     label_list.append(c)

        return img_path_list, label_list
    
    def cut_img(self, img, lanes):
        img = img[self.cut_height:]
        for lane in lanes:
            lane[:, 1] = lane[:, 1] - self.cut_height
        return img, lanes
    

class LLAMASTsSet(BaseTsSet):
    def __init__(self, cfg=None, transforms=None):
        super().__init__(cfg=cfg, transforms=transforms) 
        if self.is_val:
            self.img_root_path = os.path.join(self.data_root, 'color_images/valid')
        else:
            self.img_root_path = os.path.join(self.data_root, 'color_images/test')
        self.img_path_list = self.get_data_list()
        self.file_name_list = [img_path.replace(self.img_root_path+'/', '') for img_path in self.img_path_list]
    
    def get_data_list(self):
        img_path_list = []
        
        sub_img_path_name_list = os.listdir(self.img_root_path)
        for sub_img_path_name in sub_img_path_name_list:
            sub_img_path = os.path.join(self.img_root_path, sub_img_path_name)
            img_name_list = os.listdir(sub_img_path)
            for img_name in img_name_list:
                img_path = os.path.join(sub_img_path, img_name)
                img_path_list.append(img_path)

        return img_path_list


class LLAMASTrSetView():
    def __init__(self, cfg=None, transforms=None):
        self.data_root = cfg.data_root
        self.label_jsons = ['label_data_0313.json', 'label_data_0531.json', 'label_data_0601.json']
        self.ori_img_h = cfg.ori_img_h
        self.img_path_list, self.label_list = self.get_data_list()
        self.cut_height = cfg.cut_height
        self.img_h = cfg.img_h
        self.img_w = cfg.img_w
        self.transforms = transforms
        
    
    def __getitem__(self, index):
        img_path, lanes = self.img_path_list[index], copy.deepcopy(self.label_list[index])
        img = cv2.imread(img_path)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ori_img = img.copy()
        img, lanes = self.cut_img(img, lanes)
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
        label_path_list = []
        label_list = []

        label_root_path = os.path.join(self.data_root, 'labels/train/')
        sub_label_path_name_list = os.listdir(label_root_path)
        for sub_label_path_name in sub_label_path_name_list:
            sub_label_path = os.path.join(label_root_path, sub_label_path_name)
            label_name_list = os.listdir(sub_label_path)
            for label_name in label_name_list:
                label_path = os.path.join(sub_label_path, label_name)
                img_path = label_path.replace('labels', 'color_images').replace('.json', '_color_rect.png')
                label_path_list.append(label_path)
                img_path_list.append(img_path)
        

        img_path_list, label_path_list = img_path_list[stno:stno+24*4], label_path_list[stno:stno+24*4]



        ys = np.arange(0, self.ori_img_h, 1)
        for label_path in tqdm(label_path_list):
            lanes = []
            xs_list = [np.array(xs) for xs in get_horizontal_values_for_four_lanes(label_path)]
            for xs in xs_list:
                mask = (xs>0)
                lane = np.stack((xs[mask], ys[mask]), axis=-1)
                lane = lane[np.unique(lane[:, 1], return_index=True)[1]]
                if lane.shape[0]>=2:
                    lane = lane[lane[:, 1].argsort()]
                    lanes.append(lane)
            label_list.append(lanes)

        return img_path_list, label_list
    
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


# class LLAMASTsSet(torch.utils.data.Dataset):
#     def __init__(self, cfg=None, transforms=None, is_view=False):
#         super().__init__()
#         self.data_root = cfg.data_root
#         self.ori_img_h = cfg.ori_img_h
#         split='valid'
#         folder = 'test' if split=='test' else 'valid'
#         self.img_root_path = os.path.join(self.data_root, 'color_images/', folder)
#         self.img_path_list, self.foler_list = self.get_data_list()
#         self.cut_height = cfg.cut_height
#         self.transform = transforms
#         self.is_view = is_view

#     def __len__(self):
#         return len(self.img_path_list)
 
#     def __getitem__(self, index):
#         img_path = self.img_path_list[index]
#         img_file_name = img_path.replace(self.img_root_path+'/', '')
#         ori_img = cv2.imread(img_path)
#         ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
#         img = ori_img[self.cut_height:]
#         if self.transform is not None:
#             img = self.transform(img)
#         if not self.is_view:
#             ori_img = None
#         return img, img_file_name, ori_img
    
#     def collate_fn(self, samples):
#         img_list = []
#         ori_img_list = []
#         file_name_list = []
#         for batch_no, (img, file_name, ori_img) in enumerate(samples):
#             img_list.append(img.unsqueeze(0))
#             file_name_list.append(file_name)
#             if ori_img is not None:
#                 ori_img_list.append(ori_img)
#         imgs = torch.cat(img_list, dim=0)
#         return imgs, file_name_list, ori_img_list
    
#     def get_data_list(self):
#         img_path_list = []
#         folder_list = []
        
#         sub_img_path_name_list = os.listdir(self.img_root_path)
#         for sub_img_path_name in sub_img_path_name_list:
#             sub_img_path = os.path.join(self.img_root_path, sub_img_path_name)
#             img_name_list = os.listdir(sub_img_path)
#             folder_list.append(sub_img_path_name)
#             for img_name in img_name_list:
#                 img_path = os.path.join(sub_img_path, img_name)
#                 img_path_list.append(img_path)
#         folder_list = list(set(folder_list))

#         return img_path_list, folder_list
    
#     def get_folder(self):
#         return self.foler_list


