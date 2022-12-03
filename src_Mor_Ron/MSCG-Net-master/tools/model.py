import os
import torch.nn.functional as F
from collections import OrderedDict
from pretrainedmodels import se_resnext50_32x4d, se_resnext101_32x4d
from lib.net.scg_gcn import *

from enum import Enum
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from torchvision.utils import save_image

# assuming (N, R, G, B) order #TODO make sure RGB/BGR?
NIR = 0
RED = 1
GREEN = 2
BLUE = 3

channel_params= dict(
    NDVI = dict(alphas = torch.tensor([1, -1, 0, 0, 0, 1, 1, 0, 0, 0], dtype=torch.double), min=-1, max=1),
    gNDVI = dict(alphas = torch.tensor([1, 0, -1, 0, 0, 1, 0, 1, 0, 0], dtype=torch.double), min=-1, max=1),
    SAVI = dict(alphas = torch.tensor([1, -1, 0, 0, 0, 1.5, 1.5, 0, 0, 0.75], dtype=torch.double), min=-10000, max=10000, min_clip=-10000, max_clip=10000) # L = 0.5
)



class AppendGenericAgriculturalIndices(nn.Module):
    """GAI = (a0N + a1R + a2G + a3B + a4)/(a5N + a6R + a7G + a8B + a9)"""
    def __init__(self, alphas = None, epsilon=1e-10, learn=False, std=1.0, min=None, max=None, min_clip=None, max_clip=None)->None:
        super().__init__()
        if alphas == None:
            alphas = torch.normal(mean=0.0, std=std, size=(10, ))

        if learn:
            self.alphas = nn.Parameter(alphas)
        else:
            self.alphas = alphas
            

        self.epsilon = epsilon
        self.dim = -3
        self.min = min
        self.max = max
        self.min_clip = min_clip
        self.max_clip = max_clip
    
    def _min_max_normalize(self, x):
        return (x - self.min)/(self.max - self.min)
    
    def forward(self, x):
        if self.min_clip or self.max_clip:
            x = torch.clip(x, min=self.min_clip, max=self.max_clip)

        red_band, green_band, blue_band, nir_band = x[:, RED, :, :], x[:, GREEN, :, :], x[:, BLUE, :, :], x[:, NIR, :, :]
        nomin = self.alphas[0]*nir_band + self.alphas[1]*red_band + self.alphas[2]*green_band + self.alphas[3]*blue_band + self.alphas[4]
        denom = self.alphas[5]*nir_band + self.alphas[6]*red_band + self.alphas[7]*green_band + self.alphas[8]*blue_band + self.alphas[9]
        index = nomin/(denom + self.epsilon)

        if self.max and self.min:
            index = self._min_max_normalize(index)

        index = index.unsqueeze(self.dim)
        y = torch.cat((x, index), dim=self.dim)
        return y

class IndexTransforms(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.transforms = []

        if args.NDVI:
            self.transforms.append(AppendGenericAgriculturalIndices(**channel_params["NDVI"]))
        if args.gNDVI:
            self.transforms.append(AppendGenericAgriculturalIndices(**channel_params["gNDVI"]))
        if args.SAVI:
            self.transforms.append(AppendGenericAgriculturalIndices(**channel_params["SAVI"]))
        if args.GAI: #pass min, max, clip...
            self.transforms.append(AppendGenericAgriculturalIndices(alphas = args.GAI))
        for init_channel in args.learn:
            if init_channel == "gaussian":
                self.transforms.append(AppendGenericAgriculturalIndices(learn=True))
            else:
                self.transforms.append(AppendGenericAgriculturalIndices(**channel_params[init_channel], learn=True))

        
        self.number_of_transforms = len(self.transforms)
        self.index_transform = nn.Sequential(*self.transforms)

    def forward(self, x):
        return self.index_transform(x)



        

def load_model(args, name='MSCG-Rx50', classes=7, node_size=(32,32)):
    if name == 'MSCG-Rx50':
        net = rx50_gcn_3head_4channel(args=args, out_channels=classes)
    elif name == 'MSCG-Rx101':
        net = rx101_gcn_3head_4channel(out_channels=classes)
    else:
        print('not found the net')
        return -1

    return net


class rx50_gcn_3head_4channel(nn.Module):
    def __init__(self, args, out_channels=7, pretrained=True,
                 nodes=(32, 32), dropout=0,
                 enhance_diag=True, aux_pred=True):
        super(rx50_gcn_3head_4channel, self).__init__()  # same with  res_fdcs_v5

        self.aux_pred = aux_pred
        self.node_size = nodes
        self.num_cluster = out_channels

        resnet = se_resnext50_32x4d()

        self.index_transforms_layer = IndexTransforms(args)
        self.layer0, self.layer1, self.layer2, self.layer3, = \
            resnet.layer0, resnet.layer1, resnet.layer2, resnet.layer3

        conv_in_channels = 4 + self.index_transforms_layer.number_of_transforms

        self.conv0 = torch.nn.Conv2d(conv_in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        for child in self.layer0.children():
            for param in child.parameters():
                par = param
                break
            break

        self.conv0.parameters = torch.cat([par[:, 0, :, :].unsqueeze(1), par], 1)
        self.layer0 = torch.nn.Sequential(self.conv0, *list(self.layer0)[1:4])

        self.graph_layers1 = GCN_Layer(1024, 128, bnorm=True, activation=nn.ReLU(True), dropout=dropout)

        self.graph_layers2 = GCN_Layer(128, out_channels, bnorm=False, activation=None)

        self.scg = SCG_block(in_ch=1024,
                             hidden_ch=out_channels,
                             node_size=nodes,
                             add_diag=enhance_diag,
                             dropout=dropout)

        weight_xavier_init(self.graph_layers1, self.graph_layers2, self.scg)

    def forward(self, x):
        # add prepocess channels

        x = self.index_transforms_layer(x)
        for i, param in enumerate(self.index_transforms_layer.parameters()):
            print(f"Parameter #{i} of shape {param.shape}:\n{param.data}\n")
        # x = x[:, -1, :, :].unsqueeze(1) # intrestin experience to bottleneck the learnable channel
        

        x_size = x.size()
        print(x_size)

        # for i, param in enumerate(self.layer0.parameters()):
        #     print(f"conv Parameter #{i} of shape {param.shape}:\n{param.data}\n")

        gx = self.layer3(self.layer2(self.layer1(self.layer0(x))))
        gx90 = gx.permute(0, 1, 3, 2)
        gx180 = gx.flip(3)
        B, C, H, W = gx.size()

        A, gx, loss, z_hat = self.scg(gx)
        gx, _ = self.graph_layers2(
            self.graph_layers1((gx.reshape(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx += z_hat
        gx = gx.reshape(B, self.num_cluster, self.node_size[0], self.node_size[1])

        A, gx90, loss2, z_hat = self.scg(gx90)
        gx90, _ = self.graph_layers2(
            self.graph_layers1((gx90.reshape(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx90 += z_hat
        gx90 = gx90.reshape(B, self.num_cluster, self.node_size[1], self.node_size[0])
        gx90 = gx90.permute(0, 1, 3, 2)
        gx += gx90

        A, gx180, loss3, z_hat = self.scg(gx180)
        gx180, _ = self.graph_layers2(
            self.graph_layers1((gx180.reshape(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx180 += z_hat
        gx180 = gx180.reshape(B, self.num_cluster, self.node_size[0], self.node_size[1])
        gx180 = gx180.flip(3)
        gx += gx180

        gx = F.interpolate(gx, (H, W), mode='bilinear', align_corners=False)

        if self.training:
            return F.interpolate(gx, x_size[2:], mode='bilinear', align_corners=False), loss + loss2 + loss3
        else:
            return F.interpolate(gx, x_size[2:], mode='bilinear', align_corners=False)


class rx101_gcn_3head_4channel(nn.Module):
    def __init__(self, out_channels=7, pretrained=True,
                 nodes=(32, 32), dropout=0,
                 enhance_diag=True, aux_pred=True):
        super(rx101_gcn_3head_4channel, self).__init__()  # same with  res_fdcs_v5

        self.aux_pred = aux_pred
        self.node_size = nodes
        self.num_cluster = out_channels

        resnet = se_resnext101_32x4d()
        self.layer0, self.layer1, self.layer2, self.layer3, = \
            resnet.layer0, resnet.layer1, resnet.layer2, resnet.layer3

        self.conv0 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        for child in self.layer0.children():
            for param in child.parameters():
                par = param
                break
            break

        self.conv0.parameters = torch.cat([par[:, 0, :, :].unsqueeze(1), par], 1)
        self.layer0 = torch.nn.Sequential(self.conv0, *list(self.layer0)[1:4])

        self.graph_layers1 = GCN_Layer(1024, 128, bnorm=True, activation=nn.ReLU(True), dropout=dropout)

        self.graph_layers2 = GCN_Layer(128, out_channels, bnorm=False, activation=None)

        self.scg = SCG_block(in_ch=1024,
                             hidden_ch=out_channels,
                             node_size=nodes,
                             add_diag=enhance_diag,
                             dropout=dropout)

        weight_xavier_init(self.graph_layers1, self.graph_layers2, self.scg)

    def forward(self, x):
        x_size = x.size()

        gx = self.layer3(self.layer2(self.layer1(self.layer0(x))))
        gx90 = gx.permute(0, 1, 3, 2)
        gx180 = gx.flip(3)

        B, C, H, W = gx.size()

        A, gx, loss, z_hat = self.scg(gx)

        gx, _ = self.graph_layers2(
            self.graph_layers1((gx.view(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx += z_hat
        gx = gx.view(B, self.num_cluster, self.node_size[0], self.node_size[1])

        A, gx90, loss2, z_hat = self.scg(gx90)
        gx90, _ = self.graph_layers2(
            self.graph_layers1((gx90.view(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx90 += z_hat
        gx90 = gx90.view(B, self.num_cluster, self.node_size[1], self.node_size[0])
        gx90 = gx90.permute(0, 1, 3, 2)
        gx += gx90

        A, gx180, loss3, z_hat = self.scg(gx180)
        gx180, _ = self.graph_layers2(
            self.graph_layers1((gx180.view(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx180 += z_hat
        gx180 = gx180.view(B, self.num_cluster, self.node_size[0], self.node_size[1])
        gx180 = gx180.flip(3)
        gx += gx180

        gx = F.interpolate(gx, (H, W), mode='bilinear', align_corners=False)

        if self.training:
            return F.interpolate(gx, x_size[2:], mode='bilinear', align_corners=False), loss + loss2 + loss3
        else:
            return F.interpolate(gx, x_size[2:], mode='bilinear', align_corners=False)

