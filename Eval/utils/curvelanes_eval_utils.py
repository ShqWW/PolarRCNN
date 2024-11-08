import cv2
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import product
import json
from scipy.interpolate import interp1d


def write_output_curvelanes_format(lanes, ori_shape, out_path, step_size=1):
    ys = np.arange(0, ori_shape[0], step_size) / ori_shape[0]
    out = []
    # import pdb
    # pdb.set_trace()
    for lane in lanes:
        points = lane['points']
        function = interp1d(points[:, 1], points[:, 0], fill_value="extrapolate")
        min_y = points[:, 1].min() - 0.01
        max_y = points[:, 1].max() + 0.01
        xs = function(ys)
        xs[(ys < min_y) | (ys > max_y)] = -2
        valid_mask = (xs >= 0) & (xs < 1)
        xs = xs * ori_shape[1]
        lane_xs = xs[valid_mask]
        lane_ys = ys[valid_mask] * ori_shape[0]
        lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
        lane_coords = []
        for x, y in zip(lane_xs, lane_ys):
            lane_coords.append({"x": x, "y": y})
        out.append(lane_coords)
    with open(out_path, 'w') as f:
        json.dump({'Lines': out, 'Size': ori_shape}, f, indent=4)  # indent 参数用于格式化 JSON 文件的缩进

def load_data_curvelanes_format(data_dir, label_name_list):
    label_paths = [os.path.join(data_dir, label_name)  for label_name in label_name_list]
    labels = []
    for label_path in label_paths:
        with open(label_path, 'r') as f:
            label = json.load(f)
            labels.append(label)
    return labels

def calc_x(f, t):
    """Calc x from t.
    :param f: the param of interp
    :type f: dict
    :param t: step of interp
    :type t: int
    :return: x corrdinate
    :rtype: float
    """
    return f['a_x'] + f['b_x'] * t + f['c_x'] * t * t + f['d_x'] * t * t * t


def calc_y(f, t):
    """Calc y from t.
    :param f: the param of interp
    :type f: dict
    :param t: step of interp
    :type t: int
    :return: y corrdinate
    :rtype: float
    """
    return f['a_y'] + f['b_y'] * t + f['c_y'] * t * t + f['d_y'] * t * t * t


def spline_interp(*, lane, step_t=1):
    """Interp a line.
    :param lane: the lane to be interp
    :type lane: a list of dict
    :param step_t: the interp step
    :type step_t: int
    :return: the interp lane
    :rtype: list
    """
    interp_lane = []
    if len(lane) < 2:
        return lane
    interp_param = calc_params(lane)
    for f in interp_param:
        t = 0
        while t < f['h']:
            x = calc_x(f, t)
            y = calc_y(f, t)
            interp_lane.append({"x": x, "y": y})
            t += step_t
    interp_lane.append(lane[-1])
    return interp_lane


def calc_params(lane):
    """Calc params of a line.
    :param lane: the lane of which the param to be calculated.
    :type lane: list of dicts
    :return: param of the lane
    :rtype: list
    """
    params = []
    n_pt = len(lane)
    if n_pt < 2:
        return params
    if n_pt == 2:
        h0 = np.sqrt((lane[0]['x'] - lane[1]['x']) * (lane[0]['x'] - lane[1]['x']) +
                     (lane[0]['y'] - lane[1]['y']) * (lane[0]['y'] - lane[1]['y']))
        a_x = lane[0]['x']
        a_y = lane[0]['y']
        b_x = (lane[1]['x'] - a_x) / h0
        b_y = (lane[1]['y'] - a_y) / h0
        params.append({"a_x": a_x, "b_x": b_x, "c_x": 0, "d_x": 0, "a_y": a_y, "b_y": b_y, "c_y": 0, "d_y": 0, "h": h0})
        return params
    h = []
    for i in range(n_pt - 1):
        dx = lane[i]['x'] - lane[i + 1]['x']
        dy = lane[i]['y'] - lane[i + 1]['y']
        h.append(np.sqrt(dx * dx + dy * dy))
    A = []
    B = []
    C = []
    D_x = []
    D_y = []
    for i in range(n_pt - 2):
        A.append(h[i])
        B.append(2 * (h[i] + h[i + 1]))
        C.append(h[i + 1])
        dx1 = (lane[i + 1]['x'] - lane[i]['x']) / h[i]
        dx2 = (lane[i + 2]['x'] - lane[i + 1]['x']) / h[i + 1]
        tmpx = 6 * (dx2 - dx1)
        dy1 = (lane[i + 1]['y'] - lane[i]['y']) / h[i]
        dy2 = (lane[i + 2]['y'] - lane[i + 1]['y']) / h[i + 1]
        tmpy = 6 * (dy2 - dy1)
        if i == 0:
            C[i] /= B[i]
            D_x.append(tmpx / B[i])
            D_y.append(tmpy / B[i])
        else:
            base_v = B[i] - A[i] * C[i - 1]
            C[i] /= base_v
            D_x.append((tmpx - A[i] * D_x[i - 1]) / base_v)
            D_y.append((tmpy - A[i] * D_y[i - 1]) / base_v)

    Mx = np.zeros(n_pt)
    My = np.zeros(n_pt)
    Mx[n_pt - 2] = D_x[n_pt - 3]
    My[n_pt - 2] = D_y[n_pt - 3]
    for i in range(n_pt - 4, -1, -1):
        Mx[i + 1] = D_x[i] - C[i] * Mx[i + 2]
        My[i + 1] = D_y[i] - C[i] * My[i + 2]

    Mx[0] = 0
    Mx[-1] = 0
    My[0] = 0
    My[-1] = 0

    for i in range(n_pt - 1):
        a_x = lane[i]['x']
        b_x = (lane[i + 1]['x'] - lane[i]['x']) / h[i] - (2 * h[i] * Mx[i] + h[i] * Mx[i + 1]) / 6
        c_x = Mx[i] / 2
        d_x = (Mx[i + 1] - Mx[i]) / (6 * h[i])

        a_y = lane[i]['y']
        b_y = (lane[i + 1]['y'] - lane[i]['y']) / h[i] - (2 * h[i] * My[i] + h[i] * My[i + 1]) / 6
        c_y = My[i] / 2
        d_y = (My[i + 1] - My[i]) / (6 * h[i])

        params.append(
            {"a_x": a_x, "b_x": b_x, "c_x": c_x, "d_x": d_x, "a_y": a_y, "b_y": b_y, "c_y": c_y, "d_y": d_y, "h": h[i]})

    return params


def resize_lane(lane, x_ratio, y_ratio):
    """Resize the coordinate of a lane accroding image resize ratio.
    :param lane: the lane need to be resized
    :type lane: a list of dicts
    :param x_ratio: correspond image resize ratio in x axes.
    :type x_ratio: float
    :param y_ratio: correspond image resize ratio in y axes.
    :type y_ratio: float
    :return: resized lane
    :rtype: list
    """
    return [{"x": float(p['x']) / x_ratio, "y": float(p['y']) / y_ratio} for p in lane]


def calc_iou(lane1, lane2, hyperp):
    """Calc iou of two lane.
    :param lane1: the first lane to be calc.
    :type lane1: list of dict.
    :param lane2: the first lane to be calc.
    :type lane2: list of dict.
    :return: iou ratio.
    :rtype: float
    """
    new_height = hyperp['eval_height']
    new_width = hyperp['eval_width']
    lane_width = hyperp['lane_width']

    im1 = np.zeros((new_height, new_width), np.uint8)
    im2 = np.zeros((new_height, new_width), np.uint8)
    interp_lane1 = spline_interp(lane=lane1, step_t=1)
    interp_lane2 = spline_interp(lane=lane2, step_t=1)
    for i in range(0, len(interp_lane1) - 1):
        cv2.line(im1, (int(interp_lane1[i]['x']), int(interp_lane1[i]['y'])),
                 (int(interp_lane1[i + 1]['x']), int(interp_lane1[i + 1]['y'])), 255, lane_width)
    for i in range(0, len(interp_lane2) - 1):
        cv2.line(im2, (int(interp_lane2[i]['x']), int(interp_lane2[i]['y'])),
                 (int(interp_lane2[i + 1]['x']), int(interp_lane2[i + 1]['y'])), 255, lane_width)
    union_im = cv2.bitwise_or(im1, im2)
    union_sum = union_im.sum()
    intersection_sum = im1.sum() + im2.sum() - union_sum
    if union_sum == 0:
        return 0
    else:
        return intersection_sum / float(union_sum)


def evaluate_core(pr_lanes, gt_lanes, ori_img_shape, hyperp):
    """Core function of evaluate for every image.
    :param gt_lanes: groundtruth lanes of an image
    :type gt_lanes:  a list of lanes in an image
    :param pr_lanes: predict lanes of an image
    :type pr_lanes: a list of lanes in an image
    :return: a dict contain a series of parameters, which is:
            gt_num: groundtruth lanes number of an image
            pr_num: predict lanes number of an image
            hit_num: the matched number of groundtruth and predict lanes
            gt_curr_num: groundtruth current lanes number of an image
            pr_curr_num: predict current lanes number of an image
            hit_curr_num: the matched number of groundtruth and predict lanes in current domin
            left_error: the error of current left matched lane in x axes
            right_error: the error of current right matched lane in x axes
            census_error: the error of matched lane in x axes
    :rtype: dict
    """
    gt_num = len(gt_lanes)
    pr_num = len(pr_lanes)
    hit_num = 0
    pr_list = [False for i in range(pr_num)]
    gt_list = [False for i in range(gt_num)]

    if gt_num > 0 and pr_num > 0:
        iou_thresh = hyperp['iou_thresh']
        new_height = hyperp['eval_height']
        new_width = hyperp['eval_width']

        gt_y_ratio = np.true_divide(ori_img_shape[0], new_height)
        gt_x_ratio = np.true_divide(ori_img_shape[1], new_width)
        pr_y_ratio = np.true_divide(ori_img_shape[0], new_height)
        pr_x_ratio = np.true_divide(ori_img_shape[1], new_width)
        # resize lanes and interp lanes,
        # all the gt and pr are mapping to src img, so the scale ratio is same,
        # note that the scale ratio is not a factor but a divisor
        # print('gt_lane',gt_lanes)
        gt_lanes = list(map(lambda lane: resize_lane(lane, gt_x_ratio, gt_y_ratio), gt_lanes))
        pr_lanes = list(map(lambda lane: resize_lane(lane, pr_x_ratio, pr_y_ratio), pr_lanes))

        sorted_gt_lanes = gt_lanes
        sorted_pr_lanes = pr_lanes
        iou_mat = np.zeros((gt_num, pr_num))

        for (index_gt, gt_lane), (index_pr, pr_lane) in product(enumerate(sorted_gt_lanes), enumerate(sorted_pr_lanes)):
            iou_mat[index_gt][index_pr] = calc_iou(gt_lane, pr_lane, hyperp)

        # match_idx = Munkres().compute(make_cost_matrix(iou_mat, lambda iou: float(1.0 - iou)))
        cost_matrix = 1 - np.array(iou_mat)
        match_index_list = linear_sum_assignment(cost_matrix)

        for gt_index, pr_index in zip(*match_index_list):
            iou_val = iou_mat[gt_index][pr_index]
            if iou_val > iou_thresh:
                hit_num += 1
                pr_list[pr_index] = True
                gt_list[gt_index] = True
    return dict(gt_num=gt_num, pr_num=pr_num, hit_num=hit_num, pr_list=pr_list, gt_list=gt_list)