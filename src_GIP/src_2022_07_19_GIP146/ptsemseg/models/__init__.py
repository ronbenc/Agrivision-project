
"""
Adapted from:
Semantic Segmentation Algorithms Implemented in PyTorch
https://github.com/meetps/pytorch-semseg
"""

import copy
import torchvision.models as models

import segmentation_models_pytorch as smp

from ptsemseg.models.fcn import fcn8s, fcn16s, fcn32s
from ptsemseg.models.segnet import segnet
from ptsemseg.models.unet import unet
from ptsemseg.models.unet_atrous import unet_atrous
from ptsemseg.models.rdse_net import rdse_net
from ptsemseg.models.pspnet import pspnet
from ptsemseg.models.icnet import icnet
from ptsemseg.models.linknet import linknet
from ptsemseg.models.frrn import frrn
from ptsemseg.models.unet2 import unet2
from ptsemseg.models.encoder_decoder import encoder_decoder

from ptsemseg.models_deeplab.deeplab import DeepLab, convert_model
# import apex
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP


def get_model(model_dict, n_classes, n_channels, agct=None, correct_block=None, version=None):
    name = model_dict["arch"]

    if name == "deeplab":
        model = DeepLab(num_classes=n_classes,
                        backbone=model_dict["backbone"],  # resnet101
                        output_stride=model_dict["output_stride"],  # 16
                        ibn_mode=model_dict["ibn_mode"],  # a b ab s None
                        freeze_bn=model_dict["freeze_bn"],  # False
                        num_low_level_feat=model_dict["num_low_level_feat"],    # 3
                        interpolate_before_lastconv=model_dict["interpolate_before_lastconv"], # False
                        pretrained=model_dict["pretrained"],   # False
                        n_channels=n_channels,
                        agct=agct,
                        )

        num_channels = n_channels
        if agct is not None:
            num_channels += agct["n_channels"]
        convert_model(model, num_channels)
        # model = apex.parallel.convert_syncbn_model(model)

        return model

    if name == "rdse_net":
        model = rdse_net(
            n_classes=n_classes,
            n_channels=n_channels,
            is_batchnorm=True,
            n_maps=model_dict["n_maps"],
            atrous_dilations=model_dict["atrous_dilations"],
            atrous_layers=model_dict["atrous_layers"],
            atrous_kernel_size=model_dict["atrous_kernel_size"],
            rd_layers=model_dict["rd_layers"],
            rd_growth=model_dict["rd_growth"],
            se_reduction=model_dict["se_reduction"],
            agct=agct,
            correct_block=correct_block,
        )
        return model

    if name == "smp_unet":
        model = smp.Unet(encoder_name="resnet34",
                         encoder_depth=5,
                         encoder_weights=None,
                         decoder_use_batchnorm=True,
                         decoder_channels=(256, 128, 64, 32, 16),
                         decoder_attention_type=None,
                         in_channels=n_channels,
                         classes=n_classes,
                         activation=None,
                         aux_params=None)
        return model

    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    if name in ["frrnA", "frrnB"]:
        model = model(n_classes, n_channels=n_channels, **param_dict)

    elif name in ["fcn32s", "fcn16s", "fcn8s"]:
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "segnet":
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "unet":
        model = model(n_classes=n_classes, n_channels=n_channels, **param_dict)

    elif name == "unet_atrous":
        model = model(n_classes=n_classes, n_channels=n_channels, agct=agct, **param_dict)

    elif name == "unet2":
        model = model(n_classes=n_classes, n_channels=n_channels, **param_dict)

    elif name == "encoder_decoder":
        model = model(n_classes=n_classes, n_channels=n_channels, **param_dict)

    elif name == "pspnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnetBN":
        model = model(n_classes=n_classes, **param_dict)

    else:
        model = model(n_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "fcn32s": fcn32s,
            "fcn8s": fcn8s,
            "fcn16s": fcn16s,
            "unet": unet,
            "unet_atrous": unet_atrous,
            "unet2": unet2,
            "encoder_decoder": encoder_decoder,
            "segnet": segnet,
            "pspnet": pspnet,
            "icnet": icnet,
            "icnetBN": icnet,
            "linknet": linknet,
            "frrnA": frrn,
            "frrnB": frrn,
        }[name]
    except:
        raise ("Model {} not available".format(name))
