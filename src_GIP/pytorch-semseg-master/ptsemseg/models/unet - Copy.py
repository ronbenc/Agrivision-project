
"""
Adapted from:
Semantic Segmentation Algorithms Implemented in PyTorch
https://github.com/meetps/pytorch-semseg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class unet(nn.Module):
    def __init__(
        self, feature_scale=4, n_classes=17, is_deconv=True, n_channels=3, is_batchnorm=True, smooth_layers=0):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.n_channels = n_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.smooth_layers = smooth_layers

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.n_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        if self.smooth_layers > 0:
            self.smooth1 = smooth(n_classes,n_classes)

    def forward(self, inputs):
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
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()
        kernel_size = 3
        stride = 1
        padding = 1
        # kernel_size = 5
        # stride = 1
        # padding = 2

        # AZ change: Conv2d with padding=1, NOT  padding=0
        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_size),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_size),
                nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.ReLU()
            )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        # AZ change
        # offset = outputs2.size()[2] - inputs1.size()[2]
        # padding = 2 * [offset // 2, offset // 2]

        offset1 = outputs2.size()[2] - inputs1.size()[2]
        offset2 = inputs1.size()[2] - outputs2.size()[2]
        padding = 2 * [offset1 // 2, -(offset2 // 2)]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))

        # return self.conv(torch.cat([inputs1, outputs2], 1))




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
