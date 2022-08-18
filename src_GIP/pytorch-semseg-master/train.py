
"""
Adapted from:
Semantic Segmentation Algorithms Implemented in PyTorch
https://github.com/meetps/pytorch-semseg
"""

import os
import yaml
import datetime
import time
import shutil
import torch
import random
import argparse
import numpy as np
import scipy.misc as misc

from torch.utils import data
from tensorboardX import SummaryWriter
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function, ComposedLoss
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from agrivision_utils import save_train_loss_info, save_val_loss_info,save_best_score


def train(cfg):

    timestamp = str(datetime.datetime.now()).split(".")[0]
    timestamp = timestamp.replace(" ", "_").replace(":", "_").replace("-", "_")
    logdir = cfg["training"]["run_dir"]
    logdir = os.path.join(logdir, timestamp + "_" + os.path.basename(cfg['config_file']).split(".")[0])

    writer = SummaryWriter(log_dir=logdir)
    print("RUNDIR: {}".format(logdir))
    shutil.copy(cfg['config_file'], logdir)
    logger = get_logger(logdir)
    logger.info("Let the games begin")

    # Setup seeds
    random_seed = cfg["misc"]["random_seed"]
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Setup device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = cfg["training"].get("augmentations", None)
    # data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]
    n_channels = cfg["data"]["n_channels"]

    t_loader = data_loader(
        data_path,
        split=cfg["data"]["train_split"],
        img_list=cfg["data"]["train_img_list"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        n_channels=n_channels,
        augmentations=augmentations,
        debug_info=cfg["debug_info"],
    )

    v_loader = data_loader(
        data_path,
        split=cfg["data"]["val_split"],
        img_list=cfg["data"]["val_img_list"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        n_channels=n_channels,
        augmentations=None,
        debug_info=None,
    )

    n_classes = t_loader.n_classes
    # n_classes = t_loader.n_classes + 1  # allow out-of-bound class

    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
    )

    valloader = data.DataLoader(
        v_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
    )

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)
    # running_metrics_val = runningScore(n_classes, has_invalid_class=True)   # allow out-of-bound class

    # Setup Model
    correct_block = cfg.get("correct_block", None)
    print("Using correct_block : ", correct_block)
    model = get_model(cfg["model"], n_classes, n_channels, correct_block).to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])

    # loss_fn = get_loss_function(cfg)
    # loss_fn.keywords['weight'] = torch.FloatTensor(loss_fn.keywords.get('weight')).to(device)
    loss_fn = ComposedLoss(cfg["training"]["loss"],device)
    loss_names = ['n_iters','mean_loss'] + loss_fn.names
    for l_fn in loss_fn.loss_fns:
        # l_fn.keywords['weight'] = torch.FloatTensor(l_fn.keywords.get('weight')).to(device)
        class_weights = l_fn.keywords.get('class_weights')
        class_weights = np.array(class_weights)/sum(class_weights)
        l_fn.keywords['class_weights'] = torch.FloatTensor(class_weights).to(device)

    logger.info("Using loss {}".format(loss_fn))

    start_iter = 0
    if cfg["training"]["pretrained_model"] is not None:
        pretrained_model_path = cfg["training"]["pretrained_model"]
        if os.path.isfile(pretrained_model_path):
            logger.info("Loading model and optimizer from checkpoint '{}'".format(pretrained_model_path))
            checkpoint = torch.load(pretrained_model_path)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info("Loaded checkpoint '{}' (iter {})".format(pretrained_model_path, checkpoint["epoch"]))
            print("Loaded checkpoint '{}' (iter {})".format(pretrained_model_path, checkpoint["epoch"]))
        else:
            logger.info("No checkpoint found at '{}'".format(pretrained_model_path))
            print("No checkpoint found at: {}".format(pretrained_model_path))
            return

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -1.0
    i = start_iter
    flag = True
    train_loss_info = []
    val_loss_info = []

    while i <= cfg["training"]["train_iters"] and flag:

        for (images, labels) in trainloader:
            i += 1
            start_ts = time.time()
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss, losses_all = loss_fn(input=outputs, target=labels)

            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - start_ts)

            if (i + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss.item(),
                    time_meter.avg / cfg["training"]["batch_size"],
                )
                print(print_str)
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                time_meter.reset()

            if (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"]["train_iters"]:
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)

                        outputs = model(images_val)
                        val_loss, val_losses_all = loss_fn(input=outputs, target=labels_val)

                        pred = outputs.data.max(1)[1].cpu().numpy()
                        # pred[pred==n_classes] = 0   # allow out-of-bound class, but consider its prediction as background
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())

                writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
                logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

                for k, v in class_iou.items():
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)

                curr_loss_info = [(i+1), loss.item(), val_loss_meter.avg, score.get("Mean IoU : \t")]
                val_loss_info.append(curr_loss_info)
                save_val_loss_info(logdir, val_loss_info)
                curr_loss_info = [(i + 1), loss.item()] + losses_all.tolist()
                train_loss_info.append(curr_loss_info)
                save_train_loss_info(logdir, train_loss_info, loss_names)

                val_loss_meter.reset()
                running_metrics_val.reset()

                curr_iou = score["Mean IoU : \t"]
                # curr_iou = (class_iou[1]+class_iou[2])/2.0
                if curr_iou >= best_iou:
                    best_iou = curr_iou
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_best_score(logdir, best_iou, class_iou)
                    save_path = os.path.join(logdir, cfg["training"]["saved_model"])
                    torch.save(state, save_path)

            scheduler.step()

            if (i + 1) == cfg["training"]["train_iters"]:
                flag = False
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/blabla.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    cfg['config_file'] = args.config

    os.environ['MPLCONFIGDIR'] = os.getcwd() + "/zz_tmp_mpl_configs/"

    train(cfg)
