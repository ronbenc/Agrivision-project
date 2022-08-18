import logging
import functools
import torch
import torch.nn as nn


from ptsemseg.loss.loss import (
    cross_entropy2d,
    bootstrapped_cross_entropy2d,
    multi_scale_cross_entropy2d,
    mean_iou,
    mean_iou_softmax,
    lovasz,
    focal,
)


logger = logging.getLogger("ptsemseg")


loss_names = {
    "cross_entropy": cross_entropy2d,
    "bootstrapped_cross_entropy": bootstrapped_cross_entropy2d,
    "multi_scale_cross_entropy": multi_scale_cross_entropy2d,
    "mean_iou": mean_iou,
    "mean_iou_softmax": mean_iou_softmax,
    "lovasz": lovasz,
    "focal": focal,
}


def get_loss_function(cfg):
    if cfg["training"]["loss"] is None:
        logger.info("Using default cross entropy loss")
        return cross_entropy2d

    else:
        loss_dict = cfg["training"]["loss"]
        loss_name = loss_dict["name"]
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}

        if loss_name not in loss_names:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))

        logger.info("Using {} with {} params".format(loss_name, loss_params))
        return functools.partial(loss_names[loss_name], **loss_params)


class ComposedLoss(nn.Module):

    def __init__(self, loss_info, device):
        super().__init__()

        self.names = []
        self.loss_fns = []
        self.weights = []
        # self.class_weights = loss_info["class_weights"]
        # self.ignore_index = loss_info["ignore_index"]
        # self.reduction = loss_info["reduction"]

        loss_params = {k: v for k, v in loss_info.items() if k != "names"}
        names_and_weights = loss_info["names"]
        for name, weight in names_and_weights.items():
            if name not in loss_names:
                continue
            if weight <= 0.0:
                continue
            self.names.append(name)
            self.loss_fns.append(functools.partial(loss_names[name], **loss_params))
            self.weights.append(weight)

        sum_weights = sum(self.weights)
        self.weights = [w/sum_weights for w in self.weights]
        self.weights = torch.tensor(self.weights, device=device)


    def forward(self, input, target):
        losses = torch.stack([loss_fn(input, target) for loss_fn in self.loss_fns])
        # self.weights = torch.tensor(self.weights, device=input.device)
        loss_sum = (losses * self.weights).sum()
        return loss_sum, losses
