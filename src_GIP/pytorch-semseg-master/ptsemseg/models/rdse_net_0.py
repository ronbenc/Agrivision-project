
"""
Adapted from following sources:

Idea, network architecture:
Agrivision : Team DSSC: Residual DenseNet with Expert Network architecture.

Residual DenseNet :
Zhang, Y., Tian, Y., Kong, Y., Zhong, B. and Fu, Y., 2018. Residual dense network for image super-resolution. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2472-2481).
https://github.com/sanghyun-son/EDSR-PyTorch

Squeeze-and-excitation blocks:
Hu, J., Shen, L. and Sun, G., 2018. Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141).
https://github.com/ai-med/squeeze_and_excitation
https://github.com/moskomule/senet.pytorch

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from .unet_atrous import unetConv2, unetConv2_atrous_a1, unetConv2_atrous_a2
from ptsemseg.agri.agct import agri_color_transform


class rdse_net(nn.Module):
    def __init__(self,
                 n_classes=7,
                 n_channels=5,
                 is_batchnorm=True,
                 n_maps=[16, 32, 64, 128, 256],
                 atrous_dilations = [1,2,4,8],
                 atrous_layers = [0,0],
                 atrous_kernel_size = [3,3],
                 rd_layers = [5,5,5,5,5],
                 rd_growth=[16,16,16,16,16],
                 se_reduction=[4,4,4,4,4],
                 agct=None):
        super(rdse_net, self).__init__()

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.is_batchnorm = is_batchnorm
        self.n_maps = n_maps
        self.atrous_layers = atrous_layers
        self.atrous_kernel_size = atrous_kernel_size
        self.rd_layers = rd_layers
        self.rd_growth = rd_growth
        self.se_reduction = se_reduction
        self.n_blocks = len(self.n_maps)    # currently 4 or 5

        self.AGCT = None
        if agct is not None:
            self.AGCT = agri_color_transform(self.n_channels, agct)
            self.n_channels += agct["n_channels"]

        # down
        if self.atrous_layers[0] == 0:
            self.conv1 = unetConv(self.n_channels, n_maps[0], self.is_batchnorm, self.atrous_kernel_size[0])
        elif self.atrous_layers[0] == 1:
            self.conv1 = unetConv_atrous_a1(self.n_channels, n_maps[0], self.atrous_kernel_size[0], atrous_dilations)
        elif self.atrous_layers[0] == 2:
            self.conv1 = unetConv_atrous_a2(self.n_channels, n_maps[0], self.atrous_kernel_size[0], atrous_dilations)

        if self.atrous_layers[1] == 0:
            self.conv2 = unetConv(n_maps[0], n_maps[0], self.is_batchnorm, self.atrous_kernel_size[1])
        elif self.atrous_layers[1] == 1:
            self.conv2 = unetConv_atrous_a1(n_maps[0], n_maps[0], self.atrous_kernel_size[1], atrous_dilations)
        elif self.atrous_layers[1] == 2:
            self.conv2 = unetConv_atrous_a2(n_maps[0], n_maps[0], self.atrous_kernel_size[1], atrous_dilations)

        self.rd_down1 = RDB(growRate0 = n_maps[0], growRate = self.rd_growth[0], nConvLayers = self.rd_layers[0], do_expand=False)
        self.se_down1 = SELayer(n_channels = n_maps[0], reduction=se_reduction[0])

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.ne1 = NaiveExpand(n_maps[0], n_maps[1])
        self.rd_down2 = RDB(growRate0 = n_maps[1], growRate = self.rd_growth[1], nConvLayers = self.rd_layers[1], do_expand=False)
        self.se_down2 = SELayer(n_channels = n_maps[1], reduction=se_reduction[1])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.ne2 = NaiveExpand(n_maps[1], n_maps[2])
        self.rd_down3 = RDB(growRate0 = n_maps[2], growRate = self.rd_growth[2], nConvLayers = self.rd_layers[2], do_expand=False)
        self.se_down3 = SELayer(n_channels = n_maps[2], reduction=se_reduction[2])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.ne3 = NaiveExpand(n_maps[2], n_maps[3])
        self.rd_down4 = RDB(growRate0 = n_maps[3], growRate = self.rd_growth[3], nConvLayers = self.rd_layers[3], do_expand=False)
        self.se_down4 = SELayer(n_channels = n_maps[3], reduction=se_reduction[3])

        if self.n_blocks >= 5:
            # center
            self.maxpool4 = nn.MaxPool2d(kernel_size=2)
            self.ne4 = NaiveExpand(n_maps[3], n_maps[4])
            self.rd_center = RDB(growRate0 = n_maps[4], growRate = self.rd_growth[4], nConvLayers = self.rd_layers[4], do_expand=False)
            self.se_center = SELayer(n_channels = n_maps[4], reduction=se_reduction[4])

            # up
            self.deconv_up4 = unetUp(n_maps[4], n_maps[3])
        # end of self.n_blocks >= 5

        self.rd_up4 = RDB(growRate0 = n_maps[3], growRate = self.rd_growth[3], nConvLayers = self.rd_layers[3])
        self.se_up4 = SELayer(n_channels = n_maps[3], reduction=se_reduction[3])

        self.deconv_up3 = unetUp(n_maps[3], n_maps[2])
        self.rd_up3 = RDB(growRate0 = n_maps[2], growRate = self.rd_growth[2], nConvLayers = self.rd_layers[2])
        self.se_up3 = SELayer(n_channels = n_maps[2], reduction=se_reduction[2])

        self.deconv_up2 = unetUp(n_maps[2], n_maps[1])
        self.rd_up2 = RDB(growRate0 = n_maps[1], growRate = self.rd_growth[1], nConvLayers = self.rd_layers[1])
        self.se_up2 = SELayer(n_channels = n_maps[1], reduction=se_reduction[1])

        self.deconv_up1 = unetUp(n_maps[1], n_maps[0])
        self.se_up1 = SELayer(n_channels = n_maps[0], reduction=se_reduction[0])

        # final conv (without any concat)
        self.final = nn.Conv2d(n_maps[0], self.n_classes, 1)



    def forward(self, inputs):

        if self.AGCT is not None:
            inputs = self.AGCT(inputs)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)

        rd_down1 = self.rd_down1(conv2)
        se_down1 = self.se_down1(rd_down1)

        maxpool1 = self.maxpool1(se_down1)
        maxpool1 = self.ne1(maxpool1)

        rd_down2 = self.rd_down2(maxpool1)
        se_down2 = self.se_down2(rd_down2)

        maxpool2 = self.maxpool2(se_down2)
        maxpool2 = self.ne2(maxpool2)

        rd_down3 = self.rd_down3(maxpool2)
        se_down3 = self.se_down3(rd_down3)

        maxpool3 = self.maxpool3(se_down3)
        maxpool3 = self.ne3(maxpool3)

        rd_down4 = self.rd_down4(maxpool3)
        se_down4 = self.se_down4(rd_down4)

        if self.n_blocks >= 5:
            maxpool4 = self.maxpool4(se_down4)
            maxpool4 = self.ne4(maxpool4)

            rd_center = self.rd_center(maxpool4)
            se_center = self.se_center(rd_center)

            deconv_up4 = self.deconv_up4(se_down4, se_center)
            rd_up4 = self.rd_up4(deconv_up4)
        else:
            # self.n_blocks == 4
            rd_up4 = self.rd_up4(se_down4)

        se_up4 = self.se_up4(rd_up4)

        deconv_up3 = self.deconv_up3(se_down3, se_up4)
        rd_up3 = self.rd_up3(deconv_up3)
        se_up3 = self.se_up3(rd_up3)

        deconv_up2 = self.deconv_up2(se_down2, se_up3)
        rd_up2 = self.rd_up2(deconv_up2)
        se_up2 = self.se_up2(rd_up2)

        deconv_up1 = self.deconv_up1(se_down1, se_up2)
        se_up1 = self.se_up1(deconv_up1)

        final = self.final(se_up1)
        return final


class SELayer(nn.Module):
    def __init__(self, n_channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.BatchNorm2d(G),  # AZ insert batchnorm
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, do_expand=False, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers
        # self.do_expand = do_expand

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)
        # if self.do_expand is True:
        #     self.expand =  nn.Conv2d(G0, 2*G0, kernel_size=1, padding=0, stride=1)  # AZ insert expand
        # self.bn = nn.BatchNorm2d(G0)  # AZ insert batchnorm

    def forward(self, x):
        # return self.LFF(self.convs(x)) + x
        y = self.convs(x)
        y = self.LFF(y) + x
        # if self.do_expand is True:
        #     y = self.expand(y)
        # y = self.bn(y)
        return y


class unetConv(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size):
        super(unetConv, self).__init__()
        self.k = kernel_size
        self.s = 1
        self.p = self.k // 2

        if is_batchnorm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=self.k, stride=self.s, padding=self.p),
                nn.BatchNorm2d(out_size),
                nn.ReLU()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=self.k, stride=self.s, padding=self.p),
                nn.ReLU()
            )

    def forward(self, inputs):
        outputs = self.conv(inputs)
        return outputs


class unetConv_atrous_a1(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, atrous_dilations):
        super(unetConv_atrous_a1, self).__init__()

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


class unetConv_atrous_a2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, atrous_dilations):
        super(unetConv_atrous_a2, self).__init__()
        self.k = np.array([kernel_size,kernel_size,kernel_size,kernel_size], dtype=int)
        self.d = atrous_dilations
        self.p = (self.d * (self.k-1)) // 2
        self.s = 1
        out_size_partial = out_size//4

        self.conv_atr0 = nn.Sequential(
            nn.Conv2d(in_size, out_size_partial, kernel_size=self.k[0], stride=self.s, padding=self.p[0], dilation=self.d[0]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )
        self.conv_atr1 = nn.Sequential(
            nn.Conv2d(in_size, out_size_partial, kernel_size=self.k[1], stride=self.s, padding=self.p[1], dilation=self.d[1]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )
        self.conv_atr2 = nn.Sequential(
            nn.Conv2d(in_size, out_size_partial, kernel_size=self.k[2], stride=self.s, padding=self.p[2], dilation=self.d[2]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )
        self.conv_atr3 = nn.Sequential(
            nn.Conv2d(in_size, out_size_partial, kernel_size=self.k[3], stride=self.s, padding=self.p[3], dilation=self.d[3]),
            nn.BatchNorm2d(out_size_partial),
            nn.ReLU()
        )

    def forward(self, inputs):
        out0 = self.conv_atr0(inputs)
        out1 = self.conv_atr1(inputs)
        out2 = self.conv_atr2(inputs)
        out3 = self.conv_atr3(inputs)
        outputs = torch.cat([out0, out1, out2, out3], 1)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.up1 = nn.ConvTranspose2d(in_size // 2, out_size // 2, kernel_size=1, stride=1)
        self.up2 = nn.ConvTranspose2d(in_size, out_size // 2, kernel_size=2, stride=2)

    def forward(self, inputs1, inputs2):
        outputs1 = self.up1(inputs1)
        outputs2 = self.up2(inputs2)
        return torch.cat([outputs1, outputs2], 1)


class NaiveExpand(nn.Module):
    def __init__(self, in_size, out_size):
        super(NaiveExpand, self).__init__()
        self.expand = nn.Conv2d(in_size, out_size, kernel_size=1, padding=0, stride=1)
        # self.expand = nn.Sequential(
        #     nn.Conv2d(in_size, out_size, kernel_size=1, padding=0, stride=1),
        #     nn.BatchNorm2d(out_size)
        # )

    def forward(self, input):
        output = self.expand(input)
        return output