
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

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict

from agrivision_utils import remove_small_components, \
                            dilate_components, \
                            dilate_single_category, \
                            morph_open, \
                            morph_close, \
                            blackout_image_border, \
                            save_confusion_matrix, \
                            multi_rotation_prediction
from ptsemseg.loader.agrivision_loader_utils import decode_segmap

torch.backends.cudnn.benchmark = True


def validate(cfg, args):

    # output directory and log file
    out_path = cfg["validation"]["out_dir"]
    out_path_images = out_path + os.sep + 'predicted_labels' + os.sep
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if cfg["validation"]["out_pics"] > 0 and not os.path.exists(out_path_images):
        os.mkdir(out_path_images)
    shutil.copy(cfg['config_file'], out_path)
    log_file_name = out_path + os.sep + 'validation_log.txt'
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

    if "agct" in cfg.keys():
        loader = data_loader(
            data_path,
            split="val",
            img_list=cfg["validation"]["img_list"],
            img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
            n_channels=n_channels,
            augmentations=None,
            debug_info=None,
            agct_params=cfg["agct"],
        )
    else:
        loader = data_loader(
            data_path,
            split="val",
            img_list=cfg["validation"]["img_list"],
            img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
            n_channels=n_channels,
            augmentations=None,
            debug_info=None,
        )

    n_classes = loader.n_classes
    # n_classes = loader.n_classes + 1  # allow out-of-bound class

    # valloader = data.DataLoader(loader, batch_size=cfg["training"]["batch_size"], num_workers=8)
    valloader = data.DataLoader(loader, batch_size=1, num_workers=1)
    running_metrics = runningScore(n_classes)

    # Setup Model
    if "agct" in cfg.keys():
        model = get_model(cfg["model"], n_classes, n_channels, cfg["agct"]).to(device)
    else:
        model = get_model(cfg["model"], n_classes, n_channels).to(device)
    model_path = cfg["validation"]["model"]
    state = convert_state_dict(torch.load(model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    num_rotations = cfg["validation"]["num_rotations"]

    save_img_stats = False
    if "save_img_stats" in cfg["validation"].keys() and cfg["validation"]["save_img_stats"] > 0:
        save_img_stats = True
        stats_file_name = os.path.join(out_path, 'img_stats.txt')
        stats_file = open(stats_file_name, 'w')


    for i, (images, labels) in enumerate(valloader):
        start_time = timeit.default_timer()

        images_d = images.to(device)
        outputs = model(images_d)
        pred = outputs.data.max(1)[1].cpu().numpy()
        # pred[pred == n_classes] = 0  # allow out-of-bound class, but consider its prediction as background

        del images_d
        del outputs

        if num_rotations > 1:
            pred = multi_rotation_prediction(pred, images, model, device, n_rotations=num_rotations, n_classes=n_classes)

        # clean prediction
        pred_raw = np.copy(pred)
        # pred[0] = remove_small_components(pred[0], min_pixels=1000)
        # pred[0] = dilate_components(pred[0], n_classes, radius=11)
        # pred[0] = morph_open(pred[0], n_classes, radius=21)
        # pred[0] = morph_close(pred[0], n_classes, radius=21)
        # pred[0] = dilate_single_category(pred[0], category_id=3, radius=11)

        gt = labels.numpy()

        if args.measure_time:
            elapsed_time = timeit.default_timer() - start_time
            print("Inference time (iter {0:5d}): {1:3.5f} fps ".format(i + 1, (pred.shape[0] / elapsed_time)))
            # min_p = pred.min()
            # max_p = pred.max()
            # print("Inference time (iter {0:5d}): {1:3.5f} fps | min pred {2} | max pred {3}".format(i + 1, (pred.shape[0] / elapsed_time), min_p, max_p))

        running_metrics.update(gt, pred)

        img_file_name = loader.get_img_prefix(index=i)

        if save_img_stats is True:
            curr_metrics = runningScore(n_classes)
            curr_metrics.update(gt, pred)
            score, class_iou = curr_metrics.get_scores()
            mean_iou = score["Mean IoU : \t"]

            class_score = np.zeros(shape=n_classes, dtype=np.float)
            num_gt = np.zeros(shape=n_classes,dtype=np.int32)
            for i in range(0,n_classes):
                num_gt[i] = (gt[gt==i]).shape[0]
                class_score[i] = class_iou.get(i)

            for i in range(0, n_classes):
                stats_file.write("{:6d}\t".format(num_gt[i]))
                stats_file.write("{:.3f}\t".format(class_score[i]))
            stats_file.write("{:34}".format(img_file_name))
            stats_file.write("\n")

            # stats_file.write("{:34}".format(img_file_name))
            # stats_file.write("\t{:.3f}".format(mean_iou))
            # for k, v in class_iou.items():
            #     stats_file.write("\t{:.3f}".format(v))
            # stats_file.write("\n")

            del curr_metrics

        if is_test > 0:
            # save pred image in challenge format
            pred_img = np.copy(pred)
            pred_img[gt == n_classes] = 0  # zero-label the oob pixels in the prediction image
            #pred_img[gt == n_classes] = n_classes  # assign oob value to the gt-oob pixels

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

                # rgb_file = os.path.join(data_path,'val','images','rgb', img_file_name + '.jpg')
                # rgb = misc.imread(rgb_file)
                # nir = rgb5[:, :, 0]
                # nir = np.stack((nir, nir, nir), axis=2)
                # ndvi = rgb5[:, :, 1]
                # ndvi = np.stack((ndvi, ndvi, ndvi), axis=2)

                rgb = blackout_image_border(rgb, border_size=border_size)
                nir = blackout_image_border(nir, border_size=border_size)
                ndvi = blackout_image_border(ndvi, border_size=border_size)
                rgb_decoded = np.concatenate((rgb, nir, ndvi, decoded, gt_color),axis=1)
            elif cfg["validation"]["out_pics"] == 2:
                pred_raw[gt == n_classes] = n_classes  # black-out the oob pixels in the prediction image
                decoded_raw = decode_segmap(pred_raw[0], loader.label_colors).astype(np.uint8)
                decoded_raw = blackout_image_border(decoded_raw, border_size=border_size)
                rgb_decoded = np.concatenate((decoded_raw, decoded, gt_color), axis=1)

            misc.imsave(out_file, rgb_decoded)

    score, class_iou = running_metrics.get_scores()
    mean_iou = score["Mean IoU : \t"]
    log_file = open(log_file_name, 'w')
    log_file.write("mean iou: %.3f\n" % mean_iou)
    for k, v in class_iou.items():
        log_file.write("class %2d: %.3f\n" % (k, v))
    log_file.close()
    save_confusion_matrix(out_path, running_metrics.confusion_matrix)

    if save_img_stats is True:
        stats_file.close()


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
    cfg['config_file'] = args.config

    validate(cfg, args)
