from .base_evaluator import BaseEvaluator
from .utils.eval_utils import write_output_culane_format, culane_metric, load_data_dlrail_format, load_data_culane_format, stat_results
from tabulate import tabulate
import numpy as np
import os
import cv2
import math
from tqdm.contrib import tzip

def eval_predictions(pred_dir, anno_dir, label_name_list, width=30, ori_img_h = 224, ori_img_w = 224, iou_thresholds=[0.5], is_sequential=False):
    predictions = load_data_culane_format(pred_dir, label_name_list)
    annotations = load_data_dlrail_format(anno_dir, label_name_list)
    
    if is_sequential:
        results = []
        for prediction, annotation in zip(predictions, annotations):
            results.append(culane_metric(pred=prediction, anno=annotation, width=width, ori_img_h=ori_img_h, ori_img_w=ori_img_w))
    else:
        from multiprocessing import Pool, cpu_count
        from itertools import repeat
        with Pool(cpu_count()) as p:
            results = p.starmap(culane_metric, zip(predictions, annotations,
                        repeat(width),
                        repeat(ori_img_h),
                        repeat(ori_img_w),
                        repeat(iou_thresholds)))
                        
    results = stat_results(results, iou_thresholds)
    return results

class DLRailEvaluator(BaseEvaluator):
    def __init__(self, cfg=None):
        super().__init__(cfg=cfg)
        self.txt_path = os.path.join(self.data_root, 'list/test.txt')
        
    def pre_process(self):
        folder_list = []
        with open(self.txt_path, 'r') as f:
            for line in f.readlines():
                folder_list.append(os.path.dirname(line.strip('\n')))

        if self.is_view:
            os.makedirs(self.view_path, exist_ok=True)
            for folder in folder_list:
                os.makedirs(os.path.join(self.view_path, folder), exist_ok=True)
        else:
            os.makedirs(self.result_path, exist_ok=True)
            for folder in folder_list:
                os.makedirs(os.path.join(self.result_path, folder), exist_ok=True)

        with open(self.txt_path, 'r') as file_list:
            self.img_name_list = [line[1 if line[0] == '/' else 0:].rstrip() for line in file_list.readlines()]
            self.label_name_list = [img_name.replace('.jpg', '.lines.txt') for img_name in self.img_name_list]

    def write_output(self, outputs, file_names):
        lanes_list = outputs['lane_list']
        for lanes, file_name in zip(lanes_list, file_names):
            out_path = os.path.join(self.result_path, file_name.replace('.jpg', '.lines.txt'))
            write_output_culane_format(lanes, out_path, self.ori_img_h, self.ori_img_w, self.cut_height)

    def view_output(self, output, file_names, ori_imgs):

        line_paras_batch = output['anchor_embeddings'].copy()
        line_paras_batch[..., 0] *= math.pi
        line_paras_batch[..., 1] *= self.img_w
        lanes_list = output['lane_list']
        for lanes, line_paras, file_name, ori_img in zip(lanes_list, line_paras_batch, file_names, ori_imgs):
            ori_img = ori_img.numpy()
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
            out_path_pred = os.path.join(self.view_path, file_name.replace('.jpg', '_pred.jpg'))
            out_path_anchor = os.path.join(self.view_path, file_name.replace('.jpg', '_anchor.jpg'))
            plot_img = super().view_single_img_lane(ori_img, lanes)
            cv2.imwrite(out_path_pred, plot_img)
            plot_img = super().view_single_img_line(ori_img, line_paras)
            cv2.imwrite(out_path_anchor, plot_img)
    
    def evaluate(self, is_sequential=False):
        self.evaluate_testset(is_sequential)

    def evaluate_testset(self, is_sequential=False):
        iou_thresholds = list(np.arange(0.5, 1, 0.05))
        overall_result_dict = {}
        for iou_threshold in iou_thresholds:
            overall_result_dict[iou_threshold] = {'TP': 0, 'FP': 0, 'FN': 0} 
        
        overall_result_table_list = []

        print('start evaluating....')
        results = eval_predictions(self.result_path, self.data_root, self.label_name_list, width=30, ori_img_h=self.ori_img_h, ori_img_w=self.ori_img_w, iou_thresholds=iou_thresholds, is_sequential=is_sequential)

        for iou_threshold in iou_thresholds:
            result = results[iou_threshold]
            overall_result_dict[iou_threshold]['TP'] += result['TP']
            overall_result_dict[iou_threshold]['FP'] += result['FP']
            overall_result_dict[iou_threshold]['FN'] += result['FN']
                
        mean_F = 0
        for iou_threshold in iou_thresholds:
            TP = overall_result_dict[iou_threshold]['TP']
            FP = overall_result_dict[iou_threshold]['FP']
            FN = overall_result_dict[iou_threshold]['FN']
            P = TP / (TP + FP) if TP + FP != 0 else 0 
            R = TP / (TP + FN) if TP + FN != 0 else 0 
            F = 2 * P * R / (P + R) if P + R !=0 else 0
            mean_F += F
            overall_result_table_list.append(['@' + f"{iou_threshold * 100:.0f}", f"{F * 100:.3f}", f"{P * 100:.3f}", f"{R * 100:.3f}", TP, FP, FN])
        mean_F /= len(iou_thresholds)
        overall_result_table_list.append(['Mean', f"{mean_F * 100:.3f}", '/', '/', '/', '/', '/'] ) 
        print('Overall Result:')
        headers = ['IouThr(%)', 'F1(%)', 'P(%)', 'R(%)', 'TP', 'FP', 'FN']
        print(tabulate(overall_result_table_list, headers=headers, tablefmt="rounded_grid")) 

    def view_gt(self):
        for img_name, label_name in tzip(self.img_name_list, self.label_name_list, desc='Ploting grounding truth'):
            lanes = self.get_label(os.path.join(self.data_root, label_name))
            ori_img = cv2.imread(os.path.join(self.data_root, img_name))
            plot_img = super().view_single_img_lane(ori_img, lanes, is_norm=False)
            cv2.imwrite(os.path.join(self.view_path, img_name.replace('.jpg', '_gt.jpg')), plot_img)
            
    def get_label(self, label_path):
        with open(label_path, 'r') as f:
            lane_strs = f.readlines()
        lane_list = []
        for lane_str in lane_strs:
            lane_array = np.array(lane_str.strip(' \n').split(' ')[1:]).astype(np.float32)
            lane_array_size = int(len(lane_array)/2)
            lane_array = lane_array.reshape(lane_array_size, 2)
            ind = np.where((lane_array[:, 0] >=0)&(lane_array[:, 1] >=0))
            lane_array = lane_array[ind]
            if lane_array.shape[0]>2:
                lane_list.append({'points':lane_array})
        return lane_list

        

        
        