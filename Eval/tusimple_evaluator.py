from .base_evaluator import BaseEvaluator
from .utils.tusimple_eval_utils import write_output_tusimple_format, tusimple_eval
from tabulate import tabulate
import numpy as np
import os
import cv2
import math
from tqdm.contrib import tzip

class TuSimpleEvaluator(BaseEvaluator):
    def __init__(self, cfg=None):
        super().__init__(cfg=cfg)
        self.label_json = 'test_tasks_0627.json'
        self.test_label_path = os.path.join(self.data_root, 'test_label.json')
        
    def pre_process(self):
        os.makedirs(self.result_path, exist_ok=True)
        self.tusimple_out_lines = []


        if self.is_view:
            file_name_list = []
            with open(os.path.join(self.data_root, 'test_set/' + self.label_json)) as f:
                line_strs = f.readlines()
                for line_str in line_strs:
                    sample_dict = eval(line_str)
                    img_path = sample_dict['raw_file']
                    file_name_list.append(img_path.replace('\\', ''))
            
            folder_list = [os.path.dirname(file_name) for file_name in file_name_list]
            os.makedirs(self.view_path, exist_ok=True)
            for folder in folder_list:
                os.makedirs(os.path.join(self.view_path, folder), exist_ok=True)

    def write_output(self, outputs, file_names):
        lanes_list = outputs['lane_list']
        for lanes, file_name in zip(lanes_list, file_names):
            json_out = write_output_tusimple_format(lanes, file_name, self.ori_img_h, self.ori_img_w)
            self.tusimple_out_lines.append(json_out) 

    def view_output(self, outputs, file_names, ori_imgs):
        line_paras_batch = outputs['anchor_embeddings'].copy()
        line_paras_batch[..., 0] *= math.pi
        line_paras_batch[..., 1] *= self.img_w
        lanes_list = outputs['lane_list']
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
        result_path = os.path.join(self.result_path, 'tusimple_predictions.json')
        with open(result_path, 'w') as output_file:
            output_file.write('\n'.join(self.tusimple_out_lines))
        self.evaluate_testset() 

    def evaluate_testset(self):
        result_path = os.path.join(self.result_path, 'tusimple_predictions.json')
        results = tusimple_eval.bench_one_submit(result_path, self.test_label_path)
        result_table_list = []
        result_table_list.append([f"{results['Accuracy'] * 100:.3f}", f"{results['F1'] * 100:.3f}", f"{results['FPR'] * 100:.3f}", f"{results['FNR'] * 100:.3f}"])
        print('Overall Result:')
        headers = ['Acc(%)', 'F1(%)', 'FPR(%)', 'FNR(%)']
        print(tabulate(result_table_list, headers=headers, tablefmt="rounded_grid"))

    def view_gt(self):
        with open(self.test_label_path, 'r') as f:
            line_strs = f.readlines()
        label_list = []
        img_path_list = []
        for line_str in line_strs:
            sample_dict = eval(line_str)
            img_path = os.path.join(sample_dict['raw_file'])
            img_path_list.append(img_path) 
            label = self.get_label(sample_dict)
            label_list.append(label)

        for img_path, label in tzip(img_path_list, label_list, desc='Ploting grounding truth'):
            ori_img = cv2.imread(os.path.join(self.data_root, 'test_set', img_path))
            plot_img = super().view_single_img_lane(ori_img, label, is_norm=False)
            cv2.imwrite(os.path.join(self.view_path, img_path.replace('.jpg', '_gt.jpg')), plot_img)

    def get_label(self, sample_dict):
        lane_xs = sample_dict['lanes']
        ys = sample_dict['h_samples']
        lanes = []
        for lane_x in lane_xs:
            lane_array = np.array([lane_x, ys]).transpose()
            ind = np.where((lane_array[:, 0] >=0))
            lane_array = lane_array[ind]
            if lane_array.shape[0]>0:
                lanes.append({'points': lane_array})
        return lanes
