from .base_evaluator import BaseEvaluator
from .utils.eval_utils import write_output_culane_format, load_data_culane_format, load_data_llamas_format, llamas_metric, stat_results
from tabulate import tabulate
from utils.llamas_utils import get_horizontal_values_for_four_lanes
import numpy as np
import os
import cv2
import math
from tqdm.contrib import tzip

def eval_predictions(pred_dir, anno_dir, label_name_list, width=30, ori_img_h = 224, ori_img_w = 224, iou_thresholds=[0.5], is_sequential=False):
    annotations = load_data_llamas_format(anno_dir, label_name_list)
    predictions = load_data_culane_format(pred_dir, label_name_list)
    print('Caculating...') 
    if is_sequential:
        results = []
        for prediction, annotation in zip(predictions, annotations):
            results.append(llamas_metric(pred=prediction, anno=annotation, width=width, ori_img_h=ori_img_h, ori_img_w=ori_img_w))
    else:
        from multiprocessing import Pool, cpu_count
        from itertools import repeat
        with Pool(cpu_count()) as p:
            results = p.starmap(llamas_metric, zip(predictions, annotations,
                        repeat(width),
                        repeat(ori_img_h),
                        repeat(ori_img_w),
                        repeat(iou_thresholds)))

    results = stat_results(results, iou_thresholds)
    return results

class LLAMASEvaluator(BaseEvaluator):
    def __init__(self, cfg=None):
        super().__init__(cfg=cfg)
        self.data_root = os.path.join(self.data_root)
        if self.is_val:
            self.img_root_path = os.path.join(self.data_root, 'color_images/valid')
            self.label_root_path = os.path.join(self.data_root, 'labels/valid') 
        else:
            self.img_root_path = os.path.join(self.data_root, 'color_images/test')
        
    def pre_process(self):
        folder_list = []
        sub_img_path_name_list = os.listdir(self.img_root_path)
        self.img_name_list = []
        for sub_img_path_name in sub_img_path_name_list:
            sub_img_path = os.path.join(self.img_root_path, sub_img_path_name)
            img_name_list = os.listdir(sub_img_path)
            folder_list.append(sub_img_path_name)
            for img_name in img_name_list:
                self.img_name_list.append(os.path.join(sub_img_path_name, img_name))
        folder_list = list(set(folder_list))

        if self.is_val:
            self.label_name_list = [img_name.replace('_color_rect.png', '.lines.txt') for img_name in self.img_name_list]


        if self.is_view:
            os.makedirs(self.view_path, exist_ok=True)
            for folder in folder_list:
                os.makedirs(os.path.join(self.view_path, folder), exist_ok=True)
        else:
            os.makedirs(self.result_path, exist_ok=True)
            for folder in folder_list:
                os.makedirs(os.path.join(self.result_path, folder), exist_ok=True)

    def write_output(self, outputs, file_names):
        lanes_list = outputs['lane_list']
        for lanes, file_name in zip(lanes_list, file_names):
            out_path = os.path.join(self.result_path, file_name.replace('_color_rect.png', '.lines.txt'))
            write_output_culane_format(lanes, out_path, self.ori_img_h, self.ori_img_w, self.cut_height)

    def view_output(self, outputs, file_names, ori_imgs):
        line_paras_batch = outputs['anchor_embeddings'].copy()
        line_paras_batch[..., 0] *= math.pi
        line_paras_batch[..., 1] *= self.img_w
        lanes_list = outputs['lane_list']
        for lanes, line_paras, file_name, ori_img in zip(lanes_list, line_paras_batch, file_names, ori_imgs):
            ori_img = ori_img.numpy()
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
            out_path_pred = os.path.join(self.view_path, file_name.replace('_color_rect.png', '_pred.jpg'))
            out_path_anchor = os.path.join(self.view_path, file_name.replace('_color_rect.png', '_anchor.jpg'))
            plot_img = super().view_single_img_lane(ori_img, lanes)
            cv2.imwrite(out_path_pred, plot_img)
            plot_img = super().view_single_img_line(ori_img, line_paras)
            cv2.imwrite(out_path_anchor, plot_img)

    def evaluate(self, is_sequential=False):
        if self.is_val:
            self.evaluate_valset(is_sequential)
        else:
            print('The test label is not available, please upload the test result folder [{}] to the website \' https://unsupervised-llamas.com/llamas/\''.format(self.result_path))

    def evaluate_valset(self, is_sequential=False):
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
        if self.is_val:
            for img_name, label_name in tzip(self.img_name_list, self.label_name_list, desc='Ploting grounding truth'):
                lanes = self.get_label(os.path.join(self.label_root_path, label_name.replace('.lines.txt', '.json')))
                ori_img = cv2.imread(os.path.join(self.img_root_path, img_name))
                plot_img = super().view_single_img_lane(ori_img, lanes, is_norm=False)
                cv2.imwrite(os.path.join(self.view_path, img_name.replace('_color_rect.png', '_gt.jpg')), plot_img)
        else:
            print('The test label is not available, ignore the ground truth image')
            
    def get_label(self, label_path):
        lanes = []
        xs_list = [np.array(xs) for xs in get_horizontal_values_for_four_lanes(label_path)]
        ys = np.arange(0, self.ori_img_h, 1)
        for xs in xs_list:
            mask = (xs>0)
            lane = np.stack((xs[mask], ys[mask]), axis=-1)
            lane = lane[np.unique(lane[:, 1], return_index=True)[1]]
            if lane.shape[0]>=2:
                lane = lane[lane[:, 1].argsort()]
                lanes.append({'points':lane})
        return lanes

        
        
