import torch

def Lane_iou(pred, target, valid_masks, width=15, y_stride=4.507, align=True):
    '''
    The Lane_iou is the lane iou considering the lane width and lane local oritation
    align: used for calculate the iou_matrix between pred and target
    The value range of laneiou is [0, 1]
    '''
    with torch.no_grad():
        dy = y_stride*2
        _pred = pred.clone().detach()

        pred_dx = _pred[:, 2:] - _pred[:, :-2]
        pred_width = width * torch.sqrt(pred_dx.pow(2) + dy**2) / dy
        pred_width = torch.cat([pred_width[:, 0:1], pred_width, pred_width[:, -1:]], dim=1)
        
        target_dx = target[:, 2:] - target[:, :-2]
        target_width = width * torch.sqrt(target_dx.pow(2) + dy**2) / dy
        target_width = torch.cat([target_width[:, 0:1], target_width, target_width[:, -1:]], dim=1)

        valid_masks_target = valid_masks[:, 2:] & valid_masks[:, :-2]
        valid_masks_target = torch.cat([valid_masks_target[:, 0:1], valid_masks_target, valid_masks_target[:, -1:]], dim=1)
        target_width[~valid_masks_target] = width

    px1 = pred - pred_width
    px2 = pred + pred_width
    tx1 = target - target_width
    tx2 = target + target_width
    
    if align:
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
        ovr = torch.clamp(torch.min(px2, tx2) - torch.max(px1, tx1), min=0)
    else:
        valid_masks = valid_masks.unsqueeze(1).repeat(1, pred.shape[0], 1)
        union = torch.max(px2[None, ...], tx2[:, None, :]) - torch.min(px1[None, ...], tx1[:, None, :])
        ovr = torch.clamp(torch.min(px2[None, ...], tx2[:, None, :]) - torch.max(px1[None, ...], tx1[:, None, :]), min=0)
        
    union[~valid_masks] = 0.
    ovr[~valid_masks] = 0.
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9) 
    return iou

def GLane_iou(pred, target, valid_masks, width=15, y_stride=4.507, g_weight=1.):
    '''
    The GLane_iou is the lane iou considering the lane width and lane local oritation
    The value range of Glaneiou is [-para_g, 1], 
    when the para_g is 1 by default, the function calculate the stardard giou of lanes
    '''
    with torch.no_grad():
        dy = y_stride*2
        _pred = pred.clone().detach()

        pred_dx = _pred[:, 2:] - _pred[:, :-2]
        pred_width = width * torch.sqrt(pred_dx.pow(2) + dy**2) / dy
        pred_width = torch.cat([pred_width[:, 0:1], pred_width, pred_width[:, -1:]], dim=1)
        
        target_dx = target[:, 2:] - target[:, :-2]
        target_width = width * torch.sqrt(target_dx.pow(2) + dy**2) / dy
        target_width = torch.cat([target_width[:, 0:1], target_width, target_width[:, -1:]], dim=1)

        valid_masks_target = valid_masks[:, 2:] & valid_masks[:, :-2]
        valid_masks_target = torch.cat([valid_masks_target[:, 0:1], valid_masks_target, valid_masks_target[:, -1:]], dim=1)
        target_width[~valid_masks_target] = width

    px1 = pred - pred_width
    px2 = pred + pred_width
    tx1 = target - target_width
    tx2 = target + target_width
    
    union = torch.max(px2, tx2) - torch.min(px1, tx1)
    ovr = torch.clamp(torch.min(px2, tx2) - torch.max(px1, tx1), min=0)
    g_ovr = torch.clamp((union - 2*pred_width -2*target_width), min=0)

    union[~valid_masks] = 0.
    ovr[~valid_masks] = 0.
    g_ovr[~valid_masks] = 0.
    
    iou = (ovr.sum(dim=-1)- g_weight*g_ovr.sum(dim=-1)) / (union.sum(dim=-1) + 1e-9) 
    return iou

def liou_loss(pred, target, valid_masks, width=7.5, y_stride=4.507, g_weight=1.):
    return (1 - GLane_iou(pred, target, valid_masks, width, y_stride, g_weight))

