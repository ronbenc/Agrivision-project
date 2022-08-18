import torch
import torch.nn as nn
import torch.nn.functional as F
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .aspp import build_aspp
from .decoder import build_decoder
from .backbone import build_backbone
from ptsemseg.agri.agct import agri_color_transform


def convert_model(model, in_channels):
    while not isinstance(model, nn.Conv2d):
        model = next(model.children())
    model.in_channels = in_channels
    if in_channels == 4:
        model.weight = nn.Parameter(torch.cat((model.weight, model.weight[:, 0: 1, :, :]), dim=1))
    # elif in_channels == 5:
    #     model.weight = nn.Parameter(torch.cat((model.weight, model.weight[:, 0: 1, :, :], model.weight[:, 0: 1, :, :]), dim=1))
    elif in_channels > 4:
        # AZ note: original is 3 (rgb), concat until n_channel in dim=1
        for i in range(3,in_channels):
            model.weight = nn.Parameter(torch.cat((model.weight, model.weight[:, 0: 1, :, :]), dim=1))



class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 ibn_mode='none', freeze_bn=False, num_low_level_feat=1, interpolate_before_lastconv=False, pretrained=True, n_channels=5, agct=None):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, ibn_mode, BatchNorm, pretrained=pretrained)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm, num_low_level_feat, interpolate_before_lastconv)

        self.freeze_bn = freeze_bn
        self.interpolate_before_lastconv = interpolate_before_lastconv

        self.n_channels = n_channels
        self.AGCT = None
        if agct is not None:
            self.AGCT = agri_color_transform(self.n_channels, agct)
            self.n_channels += agct["n_channels"]


    def forward(self, input):
        if self.AGCT is not None:
            input = self.AGCT(input)

        x, low_level_feats = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feats)

        if not self.interpolate_before_lastconv:
            x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        # print("\tIn Model: input size", input.size(), "output size", x.size())

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


