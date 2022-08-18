
import yaml
import torch
import argparse
import timeit
import shutil
import numpy as np
import scipy.misc as misc
import PIL
import os

from torch.utils import data
import torchvision.transforms.functional as tf

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict

from agrivision_utils import remove_small_components, blackout_image_border, save_confusion_matrix

torch.backends.cudnn.benchmark = True


def best_multi_pred_7_classes(P):
    # hist = np.bincount(P, minlength=7)
    # best = np.argmax(hist)
    # return best
    return np.argmax(np.bincount(P, minlength=7))


def test(cfg, args):

    # output directory and log file
    out_path = cfg["validation"]["out_dir"]
    out_path_images = os.path.join(out_path, 'pred_images')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if cfg["validation"]["out_pics"] > 0 and not os.path.exists(out_path_images):
        os.mkdir(out_path_images)
    out_path_pred = os.path.join(out_path, 'pred_labels')
    if not os.path.exists(out_path_pred):
        os.mkdir(out_path_pred)
    shutil.copy(cfg['config_file'], out_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]
    n_channels = cfg["data"]["n_channels"]

    loader = data_loader(
        data_path,
        split="test",
        img_list=cfg["data"]["test_img_list"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        n_channels=n_channels,
        augmentations=None,
        debug_info=None,
        test_mode=True
    )

    n_classes = loader.n_classes
    test_loader = data.DataLoader(loader, batch_size=1, num_workers=1)

    # Setup Model
    model = get_model(cfg["model"], n_classes, n_channels).to(device)
    model_path = cfg["validation"]["model"]
    state = convert_state_dict(torch.load(model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    for i, (images, labels) in enumerate(test_loader):
        start_time = timeit.default_timer()

        images_d = images.to(device)
        outputs = model(images_d)
        pred = outputs.data.max(1)[1].cpu().numpy()
        pred_raw = np.copy(pred)

        num_rotations = cfg["validation"]["num_rotations"]
        if num_rotations > 1:
            multi_pred = np.copy(pred)
            scale1 = 1.0
            scale2 = 1.0
            for j in range (1,num_rotations):
                angle = j*(360.0/num_rotations)
                if (angle % 90) != 0.0:
                    scale1 = 1.0 / np.sqrt(2.0)
                    scale2 = np.sqrt(2.0)
                if angle > 180:
                    angle -= 360
                rotated_images = tf.affine(images, translate=[0,0], scale=scale1, angle=angle, interpolation=tf.InterpolationMode.BILINEAR, shear=[0.0])
                rotated_images = rotated_images.to(device)
                rotated_outputs = model(rotated_images)
                # rotated_pred = rotated_outputs.data.max(1)[1].cpu().numpy()
                rotated_pred = rotated_outputs.data.max(1)[1].cpu()
                rotated_pred = tf.affine(rotated_pred, translate=[0,0], scale=scale2, angle=-angle, interpolation=tf.InterpolationMode.NEAREST, shear=[0.0])
                rotated_pred = rotated_pred.numpy()
                multi_pred = np.concatenate((multi_pred, rotated_pred), axis=0)

            multi_pred = n_classes - 1 - multi_pred # avoid advantage of '0' labeled class in argmax
            hist = np.apply_along_axis(best_multi_pred_7_classes, axis=0, arr=multi_pred)
            hist = n_classes - 1 - hist             # avoid advantage to '0' labeled class
            pred[0,:,:] = hist

        # clean prediction
        # pred_raw = np.copy(pred)
        # pred[0] = remove_small_components(pred[0], min_pixels=400)

        if args.measure_time:
            elapsed_time = timeit.default_timer() - start_time
            min_p = pred.min()
            max_p = pred.max()
            print("Inference time (iter {0:5d}): {1:3.5f} fps | min pred {2} | max pred {3}".format(i + 1, (pred.shape[0] / elapsed_time), min_p, max_p))

        img_file_name = loader.get_img_prefix(index=i)

        do_expert = False
        # pred_exp_path = 'C:/alon/seg_test_1/pytorch-semseg-master/runs/tst3d/atr_best/out23_all4/pred_labels'
        pred_exp_path = 'C:/alon/seg_test_1/pytorch-semseg-master/runs/tst3d/atr_best/zzexp1/pred_labels'
        pred_exp = np.zeros_like(pred)
        pred_raw = np.copy(pred)
        if do_expert is True:
            pred_exp_file = os.path.join(pred_exp_path, img_file_name + '.png')
            # pred_exp_img = misc.imread(pred_exp_file)
            pred_exp_img = np.array(PIL.Image.open(pred_exp_file))
            pred_exp[0,:,:] = pred_exp_img
            pred_exp[pred_exp == 2] = 3
            pred_exp[pred_exp == 1] = 2
            pred[pred == 2] = 0
            pred[pred == 3] = 0
            prev_mask = np.zeros_like(pred)
            prev_mask[pred == 1] = 1
            # prev_mask[pred == 2] = 1
            prev_mask[pred == 4] = 1
            prev_mask[pred == 5] = 1
            prev_mask[pred == 6] = 1
            m2 = np.logical_and(pred_exp == 2, prev_mask == 0)
            m3 = np.logical_and(pred_exp == 3, prev_mask == 0)
            pred[m2] = 2
            pred[m3] = 3


        lbl = labels.numpy()
        # pred[lbl == n_classes] = 0
        # save pred image in challenge format
        pred_img = pred[0].astype(np.uint8)
        out_file = os.path.join(out_path_pred, img_file_name + '.png')
        misc.imsave(out_file, pred_img)

        # save raw+predicted+gt image
        if cfg["validation"]["out_pics"] > 0:
            out_file = os.path.join(out_path_images, img_file_name + '.png')
            # pred[gt==n_classes] = n_classes     # black-out the oob pixels in the prediction image
            border_size = 1
            decoded = loader.decode_segmap(pred[0]).astype(np.uint8)
            decoded = blackout_image_border(decoded, border_size=border_size)
            decoded_raw = loader.decode_segmap(pred_raw[0]).astype(np.uint8)
            decoded_raw = blackout_image_border(decoded_raw, border_size=border_size)

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
                rgb_decoded = np.concatenate((rgb, nir, ndvi, decoded_raw, decoded),axis=1)
            elif cfg["validation"]["out_pics"] == 2:
                # pred_raw[gt == n_classes] = n_classes  # black-out the oob pixels in the prediction image
                rgb_decoded = np.concatenate((decoded_raw, decoded), axis=1)

            misc.imsave(out_file, rgb_decoded)


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

    test(cfg, args)
