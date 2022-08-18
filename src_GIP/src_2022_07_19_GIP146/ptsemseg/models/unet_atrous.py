
"""
Adapted from:
Semantic Segmentation Algorithms Implemented in PyTorch
https://github.com/meetps/pytorch-semseg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ptsemseg.agri.agct import agri_color_transform


class unet_atrous(nn.Module):
    def __init__(
        self, n_classes=19, is_deconv=True, n_channels=3, is_batchnorm=True, smooth_layers=0, atrous_dilations = [1,2,4,8], atrous_layers = [0,0,0,0,0], kernel_size = [3,3,3,3,3], agct=None):
        super(unet_atrous, self).__init__()
        self.is_deconv = is_deconv
        self.n_channels = n_channels
        self.is_batchnorm = is_batchnorm
        self.smooth_layers = smooth_layers
        self.atrous_layers = atrous_layers
        self.kernel_size = kernel_size
        n_maps = [16, 32, 64, 128, 256]

        self.AGCT = None
        if agct is not None:
            # num_orig_channels = 5, n_channels = 0, alpha_coeffs = None, lr = None
            self.AGCT = agri_color_transform(self.n_channels, agct)
            self.n_channels += agct["n_channels"]

        # downsampling
        if self.atrous_layers[0] == 0:
            self.conv1 = unetConv2(self.n_channels, n_maps[0], self.is_batchnorm, self.kernel_size[0])
        elif self.atrous_layers[0] == 1:
            self.conv1 = unetConv2_atrous_a1(self.n_channels, n_maps[0], self.kernel_size[0], atrous_dilations)
        elif self.atrous_layers[0] == 2:
            self.conv1 = unetConv2_atrous_a2(self.n_channels, n_maps[0], self.kernel_size[0], atrous_dilations)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        if self.atrous_layers[1] == 0:
            self.conv2 = unetConv2(n_maps[0], n_maps[1], self.is_batchnorm, self.kernel_size[1])
        elif self.atrous_layers[1] == 1:
            self.conv2 = unetConv2_atrous_a1(n_maps[0], n_maps[1], self.kernel_size[1], atrous_dilations)
        elif self.atrous_layers[1] == 2:
            self.conv2 = unetConv2_atrous_a2(n_maps[0], n_maps[1], self.kernel_size[1], atrous_dilations)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        if self.atrous_layers[2] == 0:
            self.conv3 = unetConv2(n_maps[1], n_maps[2], self.is_batchnorm, self.kernel_size[2])
        elif self.atrous_layers[2] == 1:
            self.conv3 = unetConv2_atrous_a1(n_maps[1], n_maps[2], self.kernel_size[2], atrous_dilations)
        elif self.atrous_layers[2] == 2:
            self.conv3 = unetConv2_atrous_a2(n_maps[1], n_maps[2], self.kernel_size[2], atrous_dilations)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        if self.atrous_layers[3] == 0:
            self.conv4 = unetConv2(n_maps[2], n_maps[3], self.is_batchnorm, self.kernel_size[3])
        elif self.atrous_layers[3] == 1:
            self.conv4 = unetConv2_atrous_a1(n_maps[2], n_maps[3], self.kernel_size[3], atrous_dilations)
        elif self.atrous_layers[3] == 2:
            self.conv4 = unetConv2_atrous_a2(n_maps[2], n_maps[3], self.kernel_size[3], atrous_dilations)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        if self.atrous_layers[4] == 0:
            self.center = unetConv2(n_maps[3], n_maps[4], self.is_batchnorm, self.kernel_size[4])
        elif self.atrous_layers[4] == 1:
            self.center = unetConv2_atrous_a1(n_maps[3], n_maps[4], self.kernel_size[4], atrous_dilations)
        elif self.atrous_layers[4] == 2:
            self.center = unetConv2_atrous_a2(n_maps[3], n_maps[4], self.kernel_size[4], atrous_dilations)

        # upsampling
        self.up_concat4 = unetUp(n_maps[4], n_maps[3], self.is_deconv, name="up4")
        self.up_concat3 = unetUp(n_maps[3], n_maps[2], self.is_deconv, name="up3")
        self.up_concat2 = unetUp(n_maps[2], n_maps[1], self.is_deconv, name="up2")
        self.up_concat1 = unetUp(n_maps[1], n_maps[0], self.is_deconv, name="up1")

        # final conv (without any concat)
        self.final = nn.Conv2d(n_maps[0], n_classes, 1)
        if self.smooth_layers > 0:
            self.smooth1 = smooth(n_classes,n_classes)


    def forward(self, inputs):

        if self.AGCT is not None:
            inputs = self.AGCT(inputs)

        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        if self.smooth_layers == 0:
            return final
        smooth = self.smooth1(final)
        return smooth


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size):
        super(unetConv2, self).__init__()
        self.k = kernel_size
        self.s = 1
        self.p = self.k // 2

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=self.k, stride=self.s, padding=self.p),
                nn.BatchNorm2d(out_size),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, kernel_size=self.k, stride=self.s, padding=self.p),
                nn.BatchNorm2d(out_size),
                nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=self.k, stride=self.s, padding=self.p),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, kernel_size=self.k, stride=self.s, padding=self.p),
                nn.ReLU()
            )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetConv2_atrous_a1(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, atrous_dilations):
        super(unetConv2_atrous_a1, self).__init__()

        stride = 1
        padding = kernel_size // 2
        self.k = np.array([kernel_size, kernel_size, kernel_size, kernel_size], dtype=int)
        #self.d = np.array([1,2,4,8], dtype=int)
        self.d = atrous_dilations
        self.p = (self.d * (self.k-1)) // 2

        out_size_partial = out_size//4

        self.conv1_atr0 = nn.Sequential(
            nn.Conv2d(in_size, out_size_partial, kernel_size=self.k[0], stride=stride, padding=self.p[0], dilation=self.d[0]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )
        self.conv1_atr1 = nn.Sequential(
            nn.Conv2d(in_size, out_size_partial, kernel_size=self.k[1], stride=stride, padding=self.p[1], dilation=self.d[1]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )
        self.conv1_atr2 = nn.Sequential(
            nn.Conv2d(in_size, out_size_partial, kernel_size=self.k[2], stride=stride, padding=self.p[2], dilation=self.d[2]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )
        self.conv1_atr3 = nn.Sequential(
            nn.Conv2d(in_size, out_size_partial, kernel_size=self.k[3], stride=stride, padding=self.p[3], dilation=self.d[3]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_size),
            nn.ReLU()
        )

    def forward(self, inputs):
        out0 = self.conv1_atr0(inputs)
        out1 = self.conv1_atr1(inputs)
        out2 = self.conv1_atr2(inputs)
        out3 = self.conv1_atr3(inputs)
        outputs = self.conv2(torch.cat([out0,out1,out2,out3], 1))
        return outputs


class unetConv2_atrous_a2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, atrous_dilations):
        super(unetConv2_atrous_a2, self).__init__()
        self.k = np.array([kernel_size,kernel_size,kernel_size,kernel_size], dtype=int)
        # self.d = np.array([1,2,4,8], dtype=int)
        self.d = atrous_dilations
        self.p = (self.d * (self.k-1)) // 2
        self.s = 1
        out_size_partial = out_size//4

        self.conv1_atr0 = nn.Sequential(
            nn.Conv2d(in_size, out_size_partial, kernel_size=self.k[0], stride=self.s, padding=self.p[0], dilation=self.d[0]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )
        self.conv1_atr1 = nn.Sequential(
            nn.Conv2d(in_size, out_size_partial, kernel_size=self.k[1], stride=self.s, padding=self.p[1], dilation=self.d[1]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )
        self.conv1_atr2 = nn.Sequential(
            nn.Conv2d(in_size, out_size_partial, kernel_size=self.k[2], stride=self.s, padding=self.p[2], dilation=self.d[2]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )
        self.conv1_atr3 = nn.Sequential(
            nn.Conv2d(in_size, out_size_partial, kernel_size=self.k[3], stride=self.s, padding=self.p[3], dilation=self.d[3]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )
        self.conv2_atr0 = nn.Sequential(
            nn.Conv2d(out_size, out_size_partial, kernel_size=self.k[0], stride=self.s, padding=self.p[0], dilation=self.d[0]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )
        self.conv2_atr1 = nn.Sequential(
            nn.Conv2d(out_size, out_size_partial, kernel_size=self.k[1], stride=self.s, padding=self.p[1], dilation=self.d[1]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )
        self.conv2_atr2 = nn.Sequential(
            nn.Conv2d(out_size, out_size_partial, kernel_size=self.k[2], stride=self.s, padding=self.p[2], dilation=self.d[2]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )
        self.conv2_atr3 = nn.Sequential(
            nn.Conv2d(out_size, out_size_partial, kernel_size=self.k[3], stride=self.s, padding=self.p[3], dilation=self.d[3]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )

    def forward(self, inputs):
        out0 = self.conv1_atr0(inputs)
        out1 = self.conv1_atr1(inputs)
        out2 = self.conv1_atr2(inputs)
        out3 = self.conv1_atr3(inputs)
        outputs = torch.cat([out0, out1, out2, out3], 1)
        out0 = self.conv2_atr0(outputs)
        out1 = self.conv2_atr1(outputs)
        out2 = self.conv2_atr2(outputs)
        out3 = self.conv2_atr3(outputs)
        outputs = torch.cat([out0, out1, out2, out3], 1)
        return outputs


def print_shapes(name,i1,i2,o1,o2):
    print('*******************************************************')
    print("name: " + name)
    print("in1:  " + i1)
    print("in2:  " + i2)
    print("out1: " + o1)
    print("out2: " + o2)
    print('*******************************************************')


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, name="NO_NAME"):
        super(unetUp, self).__init__()
        self.name = name
        self.conv = unetConv2(in_size, out_size, is_batchnorm=False, kernel_size=3)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        return self.conv(torch.cat([inputs1, outputs2], 1))

        # AZ change
        offset = outputs2.size()[2] - inputs1.size()[2]

        # if offset == 0:
        #     return self.conv(torch.cat([inputs1, outputs2], 1))
        padding = 2 * [offset // 2, -(-offset // 2)]
        outputs1 = F.pad(inputs1, padding)

        # i1 = str(inputs1.shape)
        # i2 = str(inputs2.shape)
        # o1 = str(outputs1.shape)
        # o2 = str(outputs2.shape)
        # print_shapes(self.name,i1,i2,o1,o2)

        return self.conv(torch.cat([outputs1, outputs2], 1))


class smooth(nn.Module):
    def __init__(self, in_size, out_size):
        super(smooth, self).__init__()
        self.smooth = nn.Sequential(
            torch.nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_size),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_size),
            torch.nn.ReLU(),
        )

    def forward(self, inputs):
        outputs = self.smooth(inputs)
        return outputs
