from utils.ploter import Ploter, COLORS
import copy
import numpy as np
class BaseEvaluator():
    # initial some common value
    def __init__(self, cfg=None):
        self.img_h, self.img_w = cfg.img_h, cfg.img_w
        self.ori_img_h, self.ori_img_w, self.cut_height = cfg.ori_img_h, cfg.ori_img_w, cfg.cut_height
        self.data_root = cfg.data_root
        self.result_path = cfg.result_path
        
        self.is_val = cfg.is_val
        self.is_view = cfg.is_view
        if self.is_view:
            self.view_path = cfg.view_path
            self.ploter = Ploter(cfg=cfg)
    # pre process something such as make new dir
    def pre_process(self):
        pass
    
    def write_output(self, output=None, filename=None):
        pass

    def view_output(self, outputs, file_names, ori_imgs):
        pass

    def evaluate(self, is_sequential=True):
        pass

    def view_gt(self):
        pass

    def view_single_img_lane(self, img, lanes, is_norm=True):
        plot_img = img.copy()
        sort_id = self.sort_lanes(lanes)
        img_h, img_w = img.shape[0], img.shape[1]
        for lane, sort_id in zip(lanes, sort_id):
            plot_lane = lane['points']
            if is_norm:
                plot_lane[..., 0] *= img_w
                plot_lane[..., 1] *= img_h
            plot_img = self.ploter.plot_single_lane(plot_img, plot_lane, color=COLORS[sort_id], thickness=4)
        return plot_img

    def view_single_img_line(self, img, line_paras):
        plot_img = img.copy()
        plot_img = self.ploter.plot_lines_oriimg(plot_img, line_paras, color=(72, 195, 238))
        return plot_img
    
    def sort_lanes(self, lanes):
        xs = np.array([np.mean(lane['points'][:, 0]) for lane in lanes])
        sorted_indices = np.argsort(xs)
        sort_id = np.empty_like(sorted_indices)
        sort_id[sorted_indices] = np.arange(len(xs))
        return sort_id
        

        
        


    

        