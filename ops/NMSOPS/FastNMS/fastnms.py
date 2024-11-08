import torch
import NMSCUDA

def nms(boxes, scores, overlap, top_k):
    return NMSCUDA.nms_cuda(boxes, scores, overlap, top_k)