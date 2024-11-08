import cv2
import numpy as np
import collections
import torch
point_color = (0, 255, 0)

def plot_lane(img, lanes, img_h, img_w):
    for lane in lanes:
        # x, y = lane['points'][-1, ...]
        # x = int(x * img_w)
        # y = int(y * img_h)
        # cv2.circle(img, center=(x, y), radius=30, color=(255, 0, 0), thickness=30)

        for point in lane["points"]:
        
            x, y = point
            x = int(x * img_w)
            y = int(y * img_h)
            # print(x, y)
            try:
                cv2.circle(img, center=(x, y), radius=2, color=point_color, thickness=3)
            except:
                pass
    return img


def deresize_output(cfg, lanes, img_shape):
    cut_height = cfg.cut_height
    ocut_height = cfg.cut_height_dict[img_shape]
    
    lanes_new = []
    oimg_h = img_shape[0]
    img_h = cfg.img_h
    ori_img_h = cfg.ori_img_h
    for lane in lanes:
        lane['points'][..., 1] = ((lane['points'][..., 1]*ori_img_h-cut_height)/(ori_img_h-cut_height)*(oimg_h-ocut_height) + ocut_height)/oimg_h
        lanes_new.append(lane)
    return lanes_new
    



        

# def plot_anchor(img, line_paras, center_h, center_w):
#     for theta, rho in line_paras:
#         ws = rho*np.cos(theta)
#         hs = rho*np.sin(theta)
#         w1, w2 = ws + rs1*np.sin(theta), ws - rs2*np.sin(theta)
#         h1, h2 = hs - rs1*np.cos(theta), hs + rs2*np.cos(theta)
#         pts1 = np.array([w1, h1])
#         pts2 = np.array([w2, h2])
#         points = np.vstack((pts1, pts2))
#         points = self.coord_trans.cartesian2img(points)
#         cv2.line(img, np.int_(points[0, :]), np.int_(points[-1, :]), colors, 2)
#     return img


def plot_gt(img, lanes, img_h, img_w):
    gt_color = (0, 0, 255)
    for lane in lanes:
        for point in lane:
            x, y = np.int_(point)
            
            cv2.circle(img, center=(x, y), radius=2, color=gt_color, thickness=3)
    return img

def load_weight(net, model_dir):
    pretrained_model = torch.load(model_dir)
    weights = pretrained_model['net']
    new_weights = collections.OrderedDict()

    for key, value in weights.items():
        new_key = key.replace('module.', '')
        if new_key.startswith('neck.'):
            new_key = new_key.replace('.conv', '') 
        new_weights[new_key] = value
    net.load_state_dict(new_weights)
    return net



def load_weight2(net, model_dir):
    pretrained_model = torch.load(model_dir)
    weights = pretrained_model
    new_weights = collections.OrderedDict()

    for key, value in weights.items():
        new_key = key.replace('module.', '')
        if new_key.startswith('neck.'):
            new_key = new_key.replace('.conv', '') 
        new_weights[new_key] = value
    # import copy
    # backbonew = copy.deepcopy(net.backbone.state_dict())
    # neckw = copy.deepcopy(net.neck.state_dict())
    # headw = copy.deepcopy(net.heads.state_dict())

    # ww = net.state_dict()
    
    # net2 = copy.deepcopy(net)

    net.load_state_dict(new_weights)
    # net.load_state_dict(ww)
    # net.backbone.load_state_dict(backbonew)
    # net.neck.load_state_dict(neckw)
    # net.heads.load_state_dict(headw)
    return net