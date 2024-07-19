import torch
from torch import nn
from .Backbone.build import build_backbone
from .Neck.build import build_neck
from .Head.roi_head.build import build_roi_head
from .Head.rpn_head.build import build_rpn_head

# from thop import profile
# from thop import clever_format

class TwoStageAnchorBasedLaneDetector(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.neck = build_neck(cfg)
        self.rpn_head = build_rpn_head(cfg) 
        self.roi_head = build_roi_head(cfg)

    def forward(self, sample_batch):
        if self.training:
            x = sample_batch['img']
        else:
            x = sample_batch
        y = self.backbone(x)[1:]
        feat = self.neck(y)
        rpn_dict = self.rpn_head(feat)

        # flops, params = profile(self.roi_head, inputs=(feat, rpn_dict))
        # flops, params = clever_format([flops, params], '%.3f')

        # print(f"flops:{flops}, paras:{params}")

        roi_dict = self.roi_head(feat, rpn_dict)
        prediction = roi_dict
        if self.training:
            prediction.update(rpn_dict)
        return prediction




