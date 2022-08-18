

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ptsemseg.agri.agct import agri_color_transform


def get_agct_basic_net(model_dict, n_classes, n_channels, agct):
    model = agct_basic_net(
        n_classes=n_classes,
        n_channels=n_channels,
        n_maps=model_dict["n_maps"],
        agct=agct,
    )
    return model


class agct_basic_net(nn.Module):
    def __init__(self,
                 n_classes=7,
                 n_channels=5,
                 n_maps=[16],
                 agct=None):
        super(agct_basic_net, self).__init__()

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_maps = n_maps

        self.AGCT = agri_color_transform(self.n_channels, agct)
        self.n_channels += agct["n_channels"]

        self.bn = nn.BatchNorm2d(self.n_channels)
        self.conv_layer_1 = basic_conv_layer(self.n_channels, self.n_maps[0], kernel_size=3)
        self.final = nn.Conv2d(self.n_maps[0], self.n_classes, kernel_size=1)



    def forward(self, inputs):
        agct_1 = self.AGCT(inputs)
        agct_1 = self.bn(agct_1)
        middle = self.conv_layer_1(agct_1)
        final = self.final(middle)
        return final


class basic_conv_layer(nn.Module):
    def __init__(self, in_size, out_size, kernel_size):
        super(basic_conv_layer, self).__init__()
        self.k = kernel_size
        self.s = 1
        self.p = self.k // 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=self.k, stride=self.s, padding=self.p),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
            nn.Conv2d(out_size, out_size, kernel_size=self.k, stride=self.s, padding=self.p),
            nn.BatchNorm2d(out_size),
            nn.ReLU()
        )

    def forward(self, inputs):
        outputs = self.conv(inputs)
        return outputs