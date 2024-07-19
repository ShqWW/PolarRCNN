import cv2
import numpy as np
from utils.coord_transform import CoordTrans

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]

class Ploter():
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.coord_trans = CoordTrans(cfg) 
        self.img_h, self.img_w = cfg.img_h, cfg.img_w

        self.center_h = cfg.center_h
        self.center_w = cfg.center_w
        self.max_lanes = cfg.max_lanes
        self.cut_height =  cfg.cut_height
        self.ori_img_h = cfg.ori_img_h 
        self.ori_img_w =  cfg.ori_img_w
        self.num_offsets = cfg.num_offsets
        self.sample_y_car = self.center_h - np.linspace(0, 1-1e-5, num = self.num_offsets)*self.img_h
        self.sample_y_car_reverse = self.sample_y_car[::-1]

    def plot_lanes_car(self, img, lanes, color=(0, 255, 0)):
        for lane in lanes:
            lane = self.coord_trans.cartesian2img(lane)
            for x, y in lane:
                cv2.circle(img, (int(x), int(y)), 2, color, -1)
        return img

    def plot_lanes_xs_car(self, img, lane_xs, lane_validmask, color=(0, 255, 0)):

        for lane_xs, lane_validmask in zip(lane_xs, lane_validmask):
            for x, y, m in zip(lane_xs, self.sample_y_car_reverse, lane_validmask):
                if m:
                    x = x + self.center_w
                    y = self.center_h - y
                    cv2.circle(img, (int(x), int(y)), 2, color, -1) 
        return img

    def plot_single_lane(self, img, lane, color=(0, 255, 0), thickness=4):
        lane = lane.astype(np.int32)
        for p1, p2 in zip(lane[:-1], lane[1:]):
            cv2.line(img, tuple(p1), tuple(p2), color=color, thickness=thickness)
        return img
    
    def plot_lines(self, img, line_paras, rs1=10000, rs2=10000, colors=(0, 255, 0)):
        for theta, rho in line_paras:
            ws = rho*np.cos(theta)
            hs = rho*np.sin(theta)
            w1, w2 = ws + rs1*np.sin(theta), ws - rs2*np.sin(theta)
            h1, h2 = hs - rs1*np.cos(theta), hs + rs2*np.cos(theta)
            pts1 = np.array()
            pts2 = np.array([w2, h2])
            points = np.vstack((pts1, pts2))
            points = self.coord_trans.cartesian2img(points)
            cv2.line(img, np.int_(points[0, :]), np.int_(points[-1, :]), colors, 2)
        return img


    def plot_lines_group(self, img, line_paras, num_group=1):
        sample_y_car = np.repeat(self.sample_y_car[np.newaxis, ...], line_paras.shape[0], axis=0)
        line_paras = line_paras.reshape(-1, 2)
        angles, rs = line_paras[..., 0][..., np.newaxis], line_paras[..., 1][..., np.newaxis]
        num_line, num_points = sample_y_car.shape
        sample_y_car_group = sample_y_car.reshape(num_line*num_group, -1)
        sample_x_car_group = -np.tan(angles)*sample_y_car_group + (rs/np.cos(angles))
        sample_car = np.stack((sample_x_car_group, sample_y_car_group), axis=-1).reshape(num_line,num_points, 2)
        for lane in sample_car:
            lane = self.coord_trans.cartesian2img(lane)
            for x, y in lane:
                if x>0 and x<self.img_w:
                    cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
        return img


    def plot_lines_oriimg(self, img, line_paras, rs1=10000, rs2=10000, color=(0, 255, 0), thickness=4):
        for theta, rho in line_paras:
            ws = rho*np.cos(theta)
            hs = rho*np.sin(theta)
            w1, w2 = ws + rs1*np.sin(theta), ws - rs2*np.sin(theta)
            h1, h2 = hs - rs1*np.cos(theta), hs + rs2*np.cos(theta)
            pts1 = np.array([w1, h1])
            pts2 = np.array([w2, h2])
            points = np.vstack((pts1, pts2))
            points = self.coord_trans.cartesian2img(points)
            points[..., 0] = points[..., 0]/self.img_w*self.ori_img_w
            points[..., 1] = (points[..., 1]/self.img_h*(self.ori_img_h-self.cut_height))+self.cut_height 
            cv2.line(img, np.int_(points[0, :]), np.int_(points[1, :]), color=color, thickness=thickness)
        return img


    # used for the curvelanes with multiple image size
    def plot_lines_oriimg_unfix(self, img, line_paras, cut_height, rs1=10000, rs2=10000, color=(0, 255, 0), thickness=4):
        ori_img_h, ori_img_w = img.shape[0], img.shape[1]
        for theta, rho in line_paras:
            ws = rho*np.cos(theta)
            hs = rho*np.sin(theta)
            w1, w2 = ws + rs1*np.sin(theta), ws - rs2*np.sin(theta)
            h1, h2 = hs - rs1*np.cos(theta), hs + rs2*np.cos(theta)
            pts1 = np.array([w1, h1])
            pts2 = np.array([w2, h2])
            points = np.vstack((pts1, pts2))
            points = self.coord_trans.cartesian2img(points)
            points[..., 0] = points[..., 0]/self.img_w*ori_img_w
            points[..., 1] = (points[..., 1]/self.img_h*(ori_img_h-cut_height))+cut_height 
            cv2.line(img, np.int_(points[0, :]), np.int_(points[1, :]), color=color, thickness=thickness)
        return img 