from torch import nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, cfg):
        super(FPN, self).__init__()
        assert isinstance(cfg.fpn_in_channel, list)
        self.in_channels = cfg.fpn_in_channel
        self.out_channel = cfg.neck_dim
        self.num_level = len(self.in_channels)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_level):
            l_conv = nn.Conv2d(self.in_channels[i], self.out_channel, 1, 1, 0)
            fpn_conv = nn.Conv2d(self.out_channel, self.out_channel, 3, 1, 1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        assert len(inputs) == self.num_level

        laterals = []
        for i in range(self.num_level):
           laterals.append(self.lateral_convs[i](inputs[i])) 

        for i in range(self.num_level - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode='nearest')
        
        outs = []
        for i in range(self.num_level):
            outs.append(self.fpn_convs[i](laterals[i]))

        return outs