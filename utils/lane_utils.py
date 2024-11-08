import numpy as np
from shapely.geometry import Polygon, LineString

def clipline_out_of_image(line_coords, img_shape, min_length=1):
    eps = 1e-3
    height, width = img_shape
    loss_box = Polygon([[0., 0.], [width-eps, 0.], [width-eps, height-eps],
                        [0., height]])
    if line_coords.shape[0]<=2:
        return None
    line_string = LineString(line_coords)
        
    I = line_string.intersection(loss_box)
    if I.is_empty:
        return None
    if I.length < min_length:
        return None
    if isinstance(I, LineString):
        pts = np.array(I.coords)
        return pts

def points_to_lineseg(batch_points, valid_masks, num_group = 1):
    batch_size, num_points, _ = batch_points.shape
    group_batch_points = batch_points.reshape(batch_size*num_group, num_points//num_group, 2)
    group_valid_masks = valid_masks.reshape(batch_size*num_group, num_points//num_group)[..., np.newaxis]

    mu = np.sum(group_batch_points*group_valid_masks, axis=1)/(np.sum(group_valid_masks, axis=1)+1e-10)
    group_batch_points = (group_batch_points-mu[:, np.newaxis, :])*group_valid_masks
    cov_matrix = np.matmul(group_batch_points.transpose(0, 2, 1), group_batch_points)


    value_a = cov_matrix[:, 1, 1] - cov_matrix[:, 0, 0]
    value_b = 2*cov_matrix[:, 0, 1]
    alpha = np.arctan2(np.sqrt(value_a**2+value_b**2)+value_a, value_b)-np.pi/2
    r = np.sum(mu*np.stack((np.cos(alpha), np.sin(alpha)), axis=-1), axis=-1)
    line_paras_group = np.stack((alpha, r), axis=-1).reshape(batch_size, num_group, 2)
    seg_valid_masks = np.sum(group_valid_masks[..., 0], axis=-1)
    seg_valid_masks = seg_valid_masks.reshape(batch_size, num_group)
    seg_valid_masks = (seg_valid_masks>0)

    return line_paras_group, seg_valid_masks