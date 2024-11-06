cfg_name = 'polarrcnn_curvelane_dla34'
############### import package ######################
import math
import cv2

############### dataset choise ######################
dataset =  'curvelanes'
data_root = './dataset/Curvelanes'

############### image parameter #########################
ori_img_h =  1440
ori_img_w =  2560
cut_height =  640
cut_height_dict = {(ori_img_h, ori_img_w, 3): cut_height,
                   (660, 1570, 3): 180,
                   (720, 1280, 3): 368} # to process some special image size
img_h = 320
img_w = 800
center_h = 15
center_w = 386
max_lanes =  14

############## data augment ###############################
train_augments = [
     dict(name='Resize', parameters=dict(height=img_h, width=img_w, interpolation=cv2.INTER_CUBIC, p=1.0)),
     dict(name='HorizontalFlip', parameters=dict(p=0.5)),
     dict(name='RandomBrightnessContrast', parameters=dict(brightness_limit=(-0.15, 0.15), contrast_limit=(-0, 0), p=0.6)),
     dict(name='HueSaturationValue', parameters=dict(hue_shift_limit=(-10, 10), sat_shift_limit=(-10, 10), val_shift_limit=(-0, 0), p=0.7)),
     dict(name='OneOf', transforms=[dict(name='MotionBlur', parameters=dict(blur_limit=(3, 5)), p=1.0),
                                    dict(name='MedianBlur', parameters=dict(blur_limit=(3, 5)), p=1.0)], p=0.2),
     dict(name='Affine', parameters=dict(translate_percent=dict(x=(-0.1, 0.1), y=(-0.1, 0.1)), rotate=(-9, 9), scale=(0.8, 1.2), interpolation=cv2.INTER_CUBIC, p=0.7)),
     dict(name='Resize', parameters=dict(height=img_h, width=img_w, interpolation=cv2.INTER_CUBIC, p=1.0)),
]

############### lane parameter #########################
num_offsets = 72
offset_stride = 4.507

######################network parameter#################################
#####backbone#####
backbone = 'dla34'
pretrained = True

#####neck#####
neck = 'fpn'
fpn_in_channel = [128, 256, 512]
neck_dim = 64
downsample_strides = [8, 16, 32]

#####rpn head#####
rpn_head = 'local_polar_head'
rpn_inchannel = neck_dim
polar_map_size = (6, 13)
num_training_priors = polar_map_size[0]*polar_map_size[1]
num_testing_priors = 78
angle_noise_p = 0.025
rho_noise_p = 0.25

#####roi head#####
roi_head = 'global_polar_head'
num_feat_samples = 36
fc_hidden_dim = 192
prior_feat_channels = 64
num_line_groups = 6
gnn_inter_dim = 128
iou_dim = 5
o2o_angle_thres = math.pi/6
o2o_rho_thres = 50

############## train parameter ###############################
batch_size = 40
epoch_num = 32
random_seed = 3404 #3407 is all you need

######################optimizer parameter#################################
lr = 6e-4
warmup_iter = 800

######################loss parameter######################################
rpn_loss = 'polarmap_loss'
roi_loss = 'tribranch_loss'

#####cost function#####
reg_cost_weight = 6
reg_cost_weight_o2o = 6
cls_cost_weight = 1
angle_prior_thres = math.pi/5
rho_prior_thres = 80
cost_iou_width = 35.68
ota_iou_width = 8.92

#####loss function #####
g_weight = 1
iou_loss_weight = 2
cls_loss_weight = 0.33
cls_loss_alpha = 0.45
cls_loss_alpha_o2o = 0.3
rank_loss_weight = 0
end_loss_weight = 0.03
aux_loss_weight = 0.2
polarmap_loss_weight = 5
loss_iou_width = 8.92

######################postprocess parameter######################################
nms_thres = 50
conf_thres = 0.45
conf_thres_o2o = conf_thres
conf_thres_nmsfree = 0.44
is_nmsfree = True
# is_nmsfree = False