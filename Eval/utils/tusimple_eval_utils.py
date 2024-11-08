import json
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d


def write_output_tusimple_format(lanes, file_name, ori_img_h, ori_img_w):
    lane_list = list()
    for lane in lanes:
        points = lane["points"]
        # function = InterpolatedUnivariateSpline(points[:, 1], points[:, 0], k=min(3, len(points) - 1))
        function = interp1d(points[:, 1], points[:, 0], fill_value="extrapolate")
        min_y = points[:, 1].min() - 0.0001
        max_y = points[:, 1].max() + 0.0001
        lane_ys = np.arange(160, 720, 10) / ori_img_h
        lane_xs = function(lane_ys)
        invalid_mask = (lane_ys < min_y) | (lane_ys > max_y)
        lane_xs = lane_xs*ori_img_w
        lane_xs[invalid_mask] = -2
        lane_list.append(lane_xs.tolist())
    output = {'raw_file': file_name, 'lanes': lane_list, 'run_time': 0}
    return json.dumps(output)

class tusimple_eval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            tusimple_eval.lr.fit(ys[:, None], xs)
            k = tusimple_eval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time):
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')
        if running_time > 200 or len(gt) + 2 < len(pred):
            return 0., 0., 1.
        angles = [
            tusimple_eval.get_angle(np.array(x_gts), np.array(y_samples))
            for x_gts in gt
        ]
        threshs = [tusimple_eval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        for x_gts, thresh in zip(gt, threshs):
            accs = [
                tusimple_eval.line_accuracy(np.array(x_preds), np.array(x_gts),
                                       thresh) for x_preds in pred
            ]
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            if max_acc < tusimple_eval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt), 4.), 1.)

    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        try:
            json_pred = [
                json.loads(line) for line in open(pred_file).readlines()
            ]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception(
                'We do not get the predictions of all the test tasks')
        gts = {l['raw_file']: l for l in json_gt}
        accuracy, fp, fn = 0., 0., 0.
        for pred in json_pred:
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception(
                    'raw_file or lanes or run_time not in some predictions.')
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']
            run_time = pred['run_time']
            if raw_file not in gts:
                raise Exception(
                    'Some raw_file from your predictions do not exist in the test tasks.'
                )
            gt = gts[raw_file]
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']
            try:
                a, p, n = tusimple_eval.bench(pred_lanes, gt_lanes, y_samples,
                                         run_time)
            except BaseException as e:
                raise Exception('Format of lanes error.')
            accuracy += a
            fp += p
            fn += n
        num = len(gts)

        num = len(gts)
        # the first return parameter is the default ranking parameter

        fp = fp / num
        fn = fn / num
        tp = 1 - fp
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        accuracy /= num

        results = {}
        results = {'Accuracy': accuracy, 'F1': f1, 'FPR': fp, 'FNR': fn}
        return results