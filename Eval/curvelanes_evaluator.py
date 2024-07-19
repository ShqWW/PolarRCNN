from .base_evaluator import BaseEvaluator
from .utils.curvelanes_eval_utils import evaluate_core, write_output_curvelanes_format, load_data_curvelanes_format
from tabulate import tabulate
import numpy as np
import os
import cv2
import math
import json
from tqdm.contrib import tzip

def deresize_output(lanes, img_shape, cut_height, ocut_height, ori_img_h):
    lanes_new = []
    oimg_h = img_shape[0]
    for lane in lanes:
        lane['points'][..., 1] = ((lane['points'][..., 1]*ori_img_h-cut_height)/(ori_img_h-cut_height)*(oimg_h-ocut_height) + ocut_height)/oimg_h
        lanes_new.append(lane)
    return lanes_new

class CurveLanesEvaluator(BaseEvaluator):
    def __init__(self, cfg=None):
        super().__init__(cfg=cfg)
        self.valid_path = os.path.join(self.data_root, 'valid') 
        self.txt_path = os.path.join(self.data_root, 'valid/valid.txt')
        self.img_path = os.path.join(self.data_root, 'valid/images')
        self.label_path = os.path.join(self.data_root, 'valid/labels')
        self.cut_height = cfg.cut_height
        self.cut_height_dict = cfg.cut_height_dict
        
    def pre_process(self):
        if self.is_view:
            os.makedirs(self.view_path, exist_ok=True)
            os.makedirs(os.path.join(self.view_path, 'images'), exist_ok=True)
        else:
            os.makedirs(self.result_path, exist_ok=True)
            os.makedirs(os.path.join(self.result_path, 'labels'), exist_ok=True)

        with open(self.txt_path, 'r') as file_list:
            self.img_name_list = [line[1 if line[0] == '/' else 0:].rstrip() for line in file_list.readlines()]
            self.label_name_list = [img_name.replace('.jpg', '.lines.json').replace('images', 'labels') for img_name in self.img_name_list]

    def write_output(self, outputs, file_names_with_shape):
        lanes_list = outputs['lane_list']
        for lanes, file_name_with_shape in zip(lanes_list, file_names_with_shape):
            file_name, ori_img_shape = file_name_with_shape
            out_path = os.path.join(self.result_path, file_name.replace('.jpg', '.lines.json').replace('images', 'labels'))
            lanes = deresize_output(lanes, ori_img_shape, self.cut_height, self.cut_height_dict[ori_img_shape], self.ori_img_h)
            write_output_curvelanes_format(lanes, ori_img_shape, out_path, step_size=4)
    
    def view_output(self, outputs, file_names_with_shape, ori_imgs):
        line_paras_batch = outputs['anchor_embeddings'].copy()
        line_paras_batch[..., 0] *= math.pi
        line_paras_batch[..., 1] *= self.img_w
        lanes_list = outputs['lane_list']
        for lanes, line_paras, file_name_with_shape, ori_img in zip(lanes_list, line_paras_batch, file_names_with_shape, ori_imgs):
            file_name, ori_img_shape = file_name_with_shape
            ori_img = ori_img.numpy()
            lanes = deresize_output(lanes, ori_img_shape, self.cut_height, self.cut_height_dict[ori_img_shape], self.ori_img_h)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
            out_path_pred = os.path.join(self.view_path, file_name.replace('.jpg', '_pred.jpg'))
            out_path_anchor = os.path.join(self.view_path, file_name.replace('.jpg', '_anchor.jpg'))
            plot_img = super().view_single_img_lane(ori_img, lanes)
            cv2.imwrite(out_path_pred, plot_img)
            plot_img = self.view_single_img_line(ori_img, line_paras, self.cut_height_dict[ori_img_shape])
            cv2.imwrite(out_path_anchor, plot_img)
    
    def view_single_img_line(self, img, line_paras, cut_height):
        plot_img = img.copy()
        plot_img = self.ploter.plot_lines_oriimg_unfix(plot_img, line_paras, cut_height, color=(72, 195, 238))
        return plot_img
    
    def evaluate(self, is_sequential=False):
        self.evaluate_valset(is_sequential)

    def evaluate_valset(self, is_sequential=False):
        print('start evaluating....')
        eval_params = {'eval_width': 224, 'eval_height': 224, 'iou_thresh': 0.5, 'lane_width': 5}
        predictions = load_data_curvelanes_format(self.result_path, self.label_name_list)
        annotations = load_data_curvelanes_format(os.path.join(self.data_root, 'valid'), self.label_name_list)

        if is_sequential:
            results = []
            for prediction, annotation in zip(predictions, annotations):
                pred_lanes = prediction['Lines']
                gt_lanes = annotation['Lines']
                ori_img_shape = prediction['Size']
                result = evaluate_core(gt_lanes=gt_lanes, pr_lanes=pred_lanes, ori_img_shape=ori_img_shape, hyperp=eval_params)
                results.append(result)

        else:
            pred_lanes_list = [prediction['Lines'] for prediction in predictions] 
            gt_lanes_list = [annotation['Lines'] for annotation in annotations] 
            ori_img_shape_list = [prediction['Size'] for prediction in predictions]

            from multiprocessing import Pool, cpu_count
            from itertools import repeat
            with Pool(cpu_count()) as p:
                results = p.starmap(evaluate_core, zip(pred_lanes_list, gt_lanes_list, ori_img_shape_list,
                            repeat(eval_params)))
        
        hit_num = sum(result['hit_num'] for result in results)
        pr_num = sum(result['pr_num'] for result in results)
        gt_num = sum(result['gt_num'] for result in results)
        TP = hit_num
        FP = pr_num-hit_num
        FN = gt_num-hit_num
        P = TP / (TP + FP) if TP + FP != 0 else 0 
        R = TP / (TP + FN) if TP + FN != 0 else 0 
        F = 2 * P * R / (P + R) if P + R !=0 else 0
        overall_result_table_list = [[f"{F * 100:.3f}", f"{P * 100:.3f}", f"{R * 100:.3f}", TP, FP, FN]]
        print('Overall Result:')
        headers = ['F1@50(%)', 'P(%)', 'R(%)', 'TP', 'FP', 'FN']
        print(tabulate(overall_result_table_list, headers=headers, tablefmt="rounded_grid"))  

    def view_gt(self):
        for img_name, label_name in tzip(self.img_name_list, self.label_name_list, desc='Ploting grounding truth'):
            lanes = self.get_label(os.path.join(self.valid_path, label_name))
            ori_img = cv2.imread(os.path.join(self.valid_path, img_name))
            plot_img = super().view_single_img_lane(ori_img, lanes, is_norm=False)
            cv2.imwrite(os.path.join(self.view_path, img_name.replace('.jpg', '_gt.jpg')), plot_img)
            
    def get_label(self, label_path):
        with open(label_path, 'r') as f:
            lanes_json = json.load(f)['Lines']
        lanes = []
        for lane_json in lanes_json:
            point_list = []
            for point_json in lane_json:
                x, y = float(point_json['x']), float(point_json['y'])
                point_list.append([x, y])
            lane = np.array(point_list)[::-1]
            lanes.append({'points': lane})
        return lanes


        
        