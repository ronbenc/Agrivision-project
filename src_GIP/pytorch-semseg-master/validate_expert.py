
"""
Adapted from:
Semantic Segmentation Algorithms Implemented in PyTorch
https://github.com/meetps/pytorch-semseg
"""

import yaml
import torch
import argparse
import timeit
import shutil
import numpy as np
import scipy.misc as misc
import os

from torch.utils import data
import torchvision.transforms.functional as tf

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict

from agrivision_utils import remove_small_components, blackout_image_border, save_confusion_matrix, multi_rotation_prediction, override_expert_prediction
from ptsemseg.loader.agrivision_loader_utils import decode_segmap

torch.backends.cudnn.benchmark = True


def validate(cfg, cfg_exp, args):

    # output directory and log file
    out_path = cfg["validation"]["out_dir"]
    out_path_images = out_path + os.sep + 'predicted_labels' + os.sep
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if cfg["validation"]["out_pics"] > 0 and not os.path.exists(out_path_images):
        os.mkdir(out_path_images)
    shutil.copy(cfg['config_file'], out_path)
    shutil.copy(cfg['config_exp_file'], out_path)
    log_file_name = out_path + os.sep + 'validation_log.txt'
    log_file = open(log_file_name, 'w')
    is_test = cfg["validation"]["is_test"]
    if is_test > 0:
        out_path_pred = os.path.join(out_path, 'pred_labels')
        if not os.path.exists(out_path_pred):
            os.mkdir(out_path_pred)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]
    n_channels = cfg["data"]["n_channels"]

    loader = data_loader(
        data_path,
        split="val",
        img_list=cfg["validation"]["img_list"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        n_channels=n_channels,
        augmentations=None,
        debug_info=cfg["debug_info"],
    )

    n_classes = loader.n_classes
    # n_classes = loader.n_classes + 1  # allow out-of-bound class

    # valloader = data.DataLoader(loader, batch_size=cfg["training"]["batch_size"], num_workers=8)
    valloader = data.DataLoader(loader, batch_size=1, num_workers=1)
    running_metrics = runningScore(n_classes)

    # load expert network and dataloader
    data_loader_exp = get_loader(cfg_exp["data"]["dataset"])
    data_path_exp = cfg_exp["data"]["path"]
    n_channels_exp = cfg_exp["data"]["n_channels"]
    loader_exp = data_loader_exp(
        data_path_exp,
        split="val",
        img_list=cfg_exp["validation"]["img_list"],
        img_size=(cfg_exp["data"]["img_rows"], cfg_exp["data"]["img_cols"]),
        n_channels=n_channels_exp,
        augmentations=None,
        debug_info=cfg_exp["debug_info"],
    )
    n_classes_exp = loader_exp.n_classes
    valloader_exp = data.DataLoader(loader_exp, batch_size=1, num_workers=1)

    model_exp = get_model(cfg_exp["model"], n_classes_exp, n_channels_exp).to(device)
    model_path_exp = cfg_exp["validation"]["model"]
    state_exp = convert_state_dict(torch.load(model_path_exp)["model_state"])
    model_exp.load_state_dict(state_exp)
    model_exp.eval()
    model_exp.to(device)

    # collect results from expert network
    num_images = len(valloader_exp)
    pred_exp_all = np.zeros((num_images,1,cfg_exp["data"]["img_rows"], cfg_exp["data"]["img_cols"]),dtype=np.uint8)
    num_rotations_exp = cfg_exp["validation"]["num_rotations"]
    for i, (images, labels) in enumerate(valloader_exp):
        images_d = images.to(device)
        outputs_exp = model_exp(images_d)
        pred_exp = outputs_exp.data.max(1)[1].cpu().numpy()
        if num_rotations_exp > 1:
            pred_exp = multi_rotation_prediction(pred_exp, images, model_exp, device, n_rotations=num_rotations_exp, n_classes=n_classes_exp)
        pred_exp[pred_exp == 2] = 3
        pred_exp[pred_exp == 1] = 2
        pred_exp_all[i,:,:,:] = pred_exp
        # gt = labels.numpy()
        # pred_exp[gt == n_classes] = n_classes
        # decoded = loader.decode_segmap(pred_exp[0]).astype(np.uint8)
        # img_file_name = loader.get_img_prefix(index=i)
        # out_file = out_path_images + img_file_name + '_exp.png'
        # misc.imsave(out_file, decoded)
        print("done expert prediction on {0:5} of  {1:5}".format(i + 1, num_images))

    # Setup Model
    model = get_model(cfg["model"], n_classes, n_channels).to(device)
    model_path = cfg["validation"]["model"]
    state = convert_state_dict(torch.load(model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    num_rotations = cfg["validation"]["num_rotations"]

    for i, (images, labels) in enumerate(valloader):
        start_time = timeit.default_timer()
        images_d = images.to(device)
        outputs = model(images_d)
        pred = outputs.data.max(1)[1].cpu().numpy()
        # pred[pred == n_classes] = 0  # allow out-of-bound class, but consider its prediction as background
        pred_raw = np.copy(pred)

        if num_rotations > 1:
            pred = multi_rotation_prediction(pred, images, model, device, n_rotations=num_rotations, n_classes=n_classes)

        # clean prediction
        # pred_raw = np.copy(pred)
        # pred[0] = remove_small_components(pred[0], min_pixels=400)

        # start override of expert network
        pred_raw = np.copy(pred)
        pred_exp = pred_exp_all[i,:,:,:]
        pred = override_expert_prediction(pred, pred_exp)
        # end override of expert network

        gt = labels.numpy()

        if args.measure_time:
            elapsed_time = timeit.default_timer() - start_time
            min_p = pred.min()
            max_p = pred.max()
            print("Inference time (iter {0:5d}): {1:3.5f} fps | min pred {2} | max pred {3}".format(i + 1, (pred.shape[0] / elapsed_time), min_p, max_p))
        running_metrics.update(gt, pred)

        img_file_name = loader.get_img_prefix(index=i)
        if is_test > 0:
            # save pred image in challenge format
            pred_img = np.copy(pred)
            pred_img[gt == n_classes] = 0  # zero-label the oob pixels in the prediction image
            pred_img = pred_img[0].astype(np.uint8)
            out_file = os.path.join(out_path_pred, img_file_name + '.png')
            misc.imsave(out_file, pred_img)

        # save raw+predicted+gt image
        if cfg["validation"]["out_pics"] > 0:
            out_file = out_path_images + img_file_name + '.png'
            pred[gt==n_classes] = n_classes     # black-out the oob pixels in the prediction image
            border_size = 1
            decoded = decode_segmap(pred[0], loader.label_colors).astype(np.uint8)
            decoded = blackout_image_border(decoded, border_size=border_size)
            gt_color = decode_segmap(gt[0], loader.label_colors).astype(np.uint8)
            gt_color = blackout_image_border(gt_color, border_size=border_size)

            if cfg["validation"]["out_pics"] == 1:
                rgb5 = (255*images.data[0]).numpy().astype(np.uint8)
                rgb5 = rgb5.transpose(1, 2, 0)
                rgb = rgb5[:, :, 0:3]
                nir = rgb5[:, :, 3]
                nir = np.stack((nir,nir,nir),axis=2)
                ndvi = rgb5[:, :, 4]
                ndvi = np.stack((ndvi, ndvi, ndvi), axis=2)
                rgb = blackout_image_border(rgb, border_size=border_size)
                nir = blackout_image_border(nir, border_size=border_size)
                ndvi = blackout_image_border(ndvi, border_size=border_size)
                rgb_decoded = np.concatenate((rgb, nir, ndvi, decoded, gt_color),axis=1)
            elif cfg["validation"]["out_pics"] == 2:
                pred_raw[gt == n_classes] = n_classes  # black-out the oob pixels in the prediction image
                decoded_raw = decode_segmap(pred_raw[0], loader.label_colors).astype(np.uint8)
                decoded_raw = blackout_image_border(decoded_raw, border_size=border_size)

                pred_exp[gt == n_classes] = n_classes  # black-out the oob pixels in the prediction image
                decoded_exp = decode_segmap(pred_exp[0], loader.label_colors).astype(np.uint8)
                decoded_exp = blackout_image_border(decoded_exp, border_size=border_size)

                rgb_decoded = np.concatenate((decoded_raw, decoded_exp, decoded, gt_color), axis=1)

            misc.imsave(out_file, rgb_decoded)

    score, class_iou = running_metrics.get_scores()
    mean_iou = score["Mean IoU : \t"]
    log_file.write("mean iou: %.3f\n" % mean_iou)
    for k, v in class_iou.items():
        log_file.write("class %2d: %.3f\n" % (k, v))
    log_file.close()
    save_confusion_matrix(out_path, running_metrics.confusion_matrix)

    dummy = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--config_exp",
        nargs="?",
        type=str,
        default="configs/blabla.yml",
        help="Config file of expert classes to be used",
    )
    parser.add_argument(
        "--measure_time",
        dest="measure_time",
        action="store_true",
        help="Enable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.add_argument(
        "--no-measure_time",
        dest="measure_time",
        action="store_false",
        help="Disable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.set_defaults(measure_time=True)

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    with open(args.config_exp) as fp:
        cfg_exp = yaml.load(fp, Loader=yaml.FullLoader)

    cfg['config_file'] = args.config
    cfg['config_exp_file'] = args.config_exp

    validate(cfg, cfg_exp, args)
