
"""
Adapted from (cross-entropy versions):
Semantic Segmentation Algorithms Implemented in PyTorch
https://github.com/meetps/pytorch-semseg
"""

import torch
import torch.nn.functional as F
from .lovasz_loss import lovasz_softmax


def mean_iou(input, target, class_weights=None, reduction='mean', ignore_index=-1):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    input = torch.nn.functional.softmax(input, dim=1)
    input = input.data.max(1)[1]

    mask = (target != ignore_index)
    mask = mask.view(1, n * h * w)
    input_valid = input.view(1, n * h * w)
    target_valid = target.view(1, n * h * w)
    input_valid = input_valid[mask]
    target_valid = target_valid[mask]
    hist = torch.reshape(torch.bincount(c * target_valid + input_valid, minlength=c**2), (c, c))
    iou = torch.diag(hist) / (torch.sum(hist,dim=1) + torch.sum(hist,dim=0) - torch.diag(hist))
    # iou = torch.nan_to_num(iou, nan=0.0)
    # if weight is not None:
    #     iou = (iou * weight) / sum(weight)
    # miou = torch.mean(iou)
    if class_weights is not None:
        iou = (iou * class_weights)
    else:
        iou = iou / c
    miou = torch.nansum(iou)
    return (1.0 - miou)


def mean_iou_softmax(input, target, class_weights=None, reduction='mean', ignore_index=-1):
    """
    Note: currently working only if ignore_index is n_classes+1
    """
    n, c, h, w = input.size()
    nt, ht, wt = target.size()
    # EPS = 1e-6

    # Handle inconsistent size between input and target
    # if h != ht and w != wt:  # upsample labels
    #     input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    # predicted probabilities for each pixel along channel
    # activation = torch.nn.Softmax(dim=1)
    # input = activation(input)
    input = torch.nn.functional.softmax(input, dim=1)

    zero_tensor = torch.zeros(n, c+1, h, w, device=target.device)
    # zero_tensor = torch.zeros_like(input=input)
    target_one_hot = zero_tensor.scatter_(1, target.view(n, 1, h, w), 1)

    zero_tensor = torch.zeros(n, 1, h, w, device=input.device)
    input = torch.cat((input,zero_tensor), dim=1)

    # Numerator Product
    inter = input * target_one_hot
    inter = inter[:,0:c,:,:]
    ## Sum over all pixels N x C x H x W => N x C
    inter = inter.view(n, c, -1).sum(2)

    # Denominator
    union = input + target_one_hot - (input * target_one_hot)
    union = union[:, 0:c, :, :]
    ## Sum over all pixels N x C x H x W => N x C
    union = union.view(n, c, -1).sum(2)

    iou = inter / union
    # iou = torch.nan_to_num(iou, nan=0.0)
    # if weight is not None:
    #     iou = (iou * weight)/sum(weight)
    #
    # ## Return average loss over classes and batch
    # miou = torch.mean(iou)

    if class_weights is not None:
        iou = (iou * class_weights)
    else:
        iou = iou / c
    miou = torch.sum(iou)/n

    return (1.0 - miou)


def lovasz(input, target, class_weights=None, reduction='mean', ignore_index=-1):
    # res = torch.tensor(1.0, device=input.device)
    classes = 'all' # 'all' 'present'
    res = lovasz_softmax(input, target, classes=classes, per_image=False, ignore=ignore_index)
    return res


def focal(input, target, class_weights=None, reduction='mean', ignore_index=-1):
    # TODO
    res = torch.tensor(0.95, device=input.device)
    return res


def cross_entropy2d(input, target, class_weights=None, reduction='mean', ignore_index=-1):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(input, target, weight=class_weights, reduction=reduction, ignore_index=ignore_index)
    return loss


def multi_scale_cross_entropy2d(input, target, class_weights=None, reduction='mean', ignore_index=-1, scale_weight=None):
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=class_weights, reduction=reduction, ignore_index=ignore_index)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(target.device)

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(input=inp, target=target, weight=class_weights, reduction=reduction, ignore_index=ignore_index)

    return loss


def bootstrapped_cross_entropy2d(input, target, K, class_weights=None, reduction='mean', ignore_index=-1):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=class_weights, reduction='mean', ignore_index=-1):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(input, target, weight=class_weights, reduction=reduction, ignore_index=ignore_index)

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=class_weights,
            reduction=reduction,
            ignore_index=ignore_index
        )
    return loss / float(batch_size)
