
import torch
import torch.nn as nn
import numpy as np
import os


class agri_correct_block(nn.Module):
    def __init__(self, n_channels = 5, n_classes=7,correct_params=None):
        super(agri_correct_block, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_maps = [16,16]
        self.atrous_kernel_size = [3,3]
        self.atrous_dilations = [1,2,4,8]

        # self.conv1 = nn.Conv2d(self.n_channels, self.n_classes, kernel_size=3)
        self.conv1 = conv_atrous(self.n_channels, self.n_maps[0], self.atrous_kernel_size[0], self.atrous_dilations)
        self.conv2 = conv_atrous(self.n_maps[0], self.n_maps[1], self.atrous_kernel_size[1], self.atrous_dilations)
        self.conv3 = conv_layer(self.n_maps[1], n_classes, 3)
        self.fusion = nn.Conv2d(2*n_classes, n_classes, kernel_size=1)
        print("Using correct_block !!! correct_block !!! correct_block !!! correct_block")

    def forward(self, input, pred):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.cat([x, pred], 1)
        x = self.fusion(x)
        print("Using correct_block !!! correct_block !!! correct_block !!! correct_block")
        return x


class conv_atrous(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, atrous_dilations):
        super(conv_atrous, self).__init__()

        stride = 1
        padding = kernel_size // 2
        self.k = np.array([kernel_size, kernel_size, kernel_size, kernel_size], dtype=int)
        self.d = atrous_dilations
        self.p = (self.d * (self.k-1)) // 2

        out_size_partial = out_size//4

        self.conv_atr0 = nn.Sequential(
            nn.Conv2d(in_size, out_size_partial, kernel_size=self.k[0], stride=stride, padding=self.p[0], dilation=self.d[0]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )
        self.conv_atr1 = nn.Sequential(
            nn.Conv2d(in_size, out_size_partial, kernel_size=self.k[1], stride=stride, padding=self.p[1], dilation=self.d[1]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )
        self.conv_atr2 = nn.Sequential(
            nn.Conv2d(in_size, out_size_partial, kernel_size=self.k[2], stride=stride, padding=self.p[2], dilation=self.d[2]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )
        self.conv_atr3 = nn.Sequential(
            nn.Conv2d(in_size, out_size_partial, kernel_size=self.k[3], stride=stride, padding=self.p[3], dilation=self.d[3]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )

    def forward(self, inputs):
        out0 = self.conv_atr0(inputs)
        out1 = self.conv_atr1(inputs)
        out2 = self.conv_atr2(inputs)
        out3 = self.conv_atr3(inputs)
        outputs = torch.cat([out0,out1,out2,out3], 1)
        return outputs

class conv_layer(nn.Module):
    def __init__(self, in_size, out_size,k_size=3):
        super(conv_layer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=k_size, stride=1, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU()
        )

    def forward(self, input):
        output = self.conv(input)
        return output