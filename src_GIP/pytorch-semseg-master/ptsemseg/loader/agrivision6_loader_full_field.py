import os
import numpy as np
import scipy.misc as m
import torch
from torch.utils import data

from ptsemseg.augmentations.augmentations import BasicAugmentations, CollageAugmentations
from .agrivision_loader_utils import decode_segmap, get_ndvi, get_evi, get_agct_img, read_agct_coeffs, read_val_image_list, read_train_image_list_full_field
from .agrivision6_loader import agrivision6Loader, agrivision6Loader_agct

import PIL
PIL.Image.MAX_IMAGE_PIXELS = None


class agrivision6Loader_full_field(agrivision6Loader):

    def __init__(self,
                 root,
                 split='train',
                 img_list='',
                 img_size=(512, 512),
                 n_channels=5,
                 augmentations=None,
                 debug_info=None,
                 img_norm=False,
                 test_mode=False,
                 agct_params=None):

        # no call to super.__init__() beacuse of different img_list format !!!
        # super(agrivision6Loader_full_field, self).__init__(root, split, img_list, img_size, n_channels, augmentations, debug_info, img_norm, test_mode)

        self.root = root
        self.split = split
        self.img_list = img_list
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.n_classes = len(self.label_names)
        self.n_channels = n_channels

        self.do_resize = False
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        if self.img_size[0] != 512 or self.img_size[1] != 512:
            self.do_resize = True

        self.save_training_images = 0
        self.save_training_folder = None
        if debug_info is not None:
            self.save_training_images = debug_info.get("save_training_images")
            self.save_training_folder = debug_info.get("save_training_dir")
            if self.save_training_images == 1 and not os.path.exists(self.save_training_folder):
                os.mkdir(self.save_training_folder)

        self.basic_augmentations = None
        self.collage_augmentations = None
        if augmentations is not None:
            aug_basic_params = augmentations.get("basic")
            aug_collage_params = augmentations.get("collage")
            if aug_basic_params is not None:
                self.basic_augmentations = BasicAugmentations(aug_basic_params)
            if aug_collage_params is not None:
                self.collage_augmentations = CollageAugmentations(aug_collage_params)

        self.file_prefixes = []
        self.split_prefixes = []
        self.coords = []
        self.pix_per_class = []
        if split == 'train':
            self.split_prefixes, self.file_prefixes, self.coords, self.pix_per_class = read_train_image_list_full_field(self.img_list)
        elif split == 'val' or split == 'test':
            self.split_prefixes, self.file_prefixes = read_val_image_list(self.img_list)

        self.void_classes = [self.out_of_bounds_label]
        self.valid_classes = [0, 1, 2, 3, 4, 5, 6]

        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))

        print("Found %d %s images" % (len(self.file_prefixes), split))

        if agct_params is None:
            self.num_orig_channels = self.n_classes
            self.new_channels = 0
        else:
            self.num_orig_channels = 5
            self.new_channels = agct_params["n_channels"]
            self.vegetation_labels = agct_params["vegetation_labels"]
            self.n_channels += self.new_channels
            # self.mean_rgbn = np.concatenate((self.mean_rgbn, 127.5*np.ones(shape=(1,self.new_channels),dtype=float)))
            self.mean_rgbn = np.concatenate((self.mean_rgbn, np.zeros(shape=self.new_channels, dtype=float)))
            if "alpha_trained" in agct_params.keys() and os.path.isfile(agct_params["alpha_trained"]):
                self.alpha_coeffs = read_agct_coeffs(agct_params["alpha_trained"])
                print("agct loader - reading pretrained coeffs")
            else:
                self.alpha_coeffs = agct_params["alpha_coeffs"]
                print("agct loader - using initial coeffs")


    def __getitem__(self, index):
        # read rgb + nir images
        prefix = self.file_prefixes[index]

        # split_dir = self.split_prefixes[index]
        # rgb_path = os.path.join(self.root, split_dir, 'full_field', prefix + '_rgb.png')
        # nir_path = os.path.join(self.root, split_dir, 'full_field', prefix + '_nir.png')
        # lbl_path = os.path.join(self.root, split_dir, 'full_field', prefix + '_gt_label.png')
        # u0 = self.coords[index, 0]
        # u1 = self.coords[index, 0] + 512
        # v0 = self.coords[index, 1]
        # v1 = self.coords[index, 1] + 512
        #
        # rgb_img = m.imread(rgb_path)
        # nir_img = m.imread(nir_path)
        # lbl = m.imread(lbl_path)
        # rgb_img = rgb_img[u0:u1, v0:v1,:]
        # nir_img = nir_img[u0:u1, v0:v1]
        # lbl = lbl[u0:u1, v0:v1]

        rgb_path = os.path.join(self.root, 'train_ff', 'rgb', prefix + '.png')
        nir_path = os.path.join(self.root, 'train_ff', 'nir', prefix + '.png')
        lbl_path = os.path.join(self.root, 'train_ff', 'gt_labels', prefix + '.png')
        rgb_img = m.imread(rgb_path)
        nir_img = m.imread(nir_path)
        lbl = m.imread(lbl_path)

        if self.do_resize is True:
            rgb_img = m.imresize(rgb_img, (self.img_size[0], self.img_size[1]), "bilinear")
            nir_img = m.imresize(nir_img, (self.img_size[0], self.img_size[1]), "bilinear")
            lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        rgb_img = np.array(rgb_img, dtype=np.float)
        nir_img = np.array(nir_img, dtype=np.float)
        lbl = np.array(lbl, dtype=np.uint8)

        # create ndvi image
        ndvi_img = get_ndvi(rgb_img, nir_img).astype(np.float)

        # if self.test_mode is False:
        #     lbl = self.encode_segmap(lbl)
        lbl = self.encode_segmap(lbl)
        idx_non_mask = np.zeros(shape=self.img_size, dtype=np.uint8)
        idx_non_mask[lbl == self.out_of_bounds_label] = 1

        # arrange rgb+nir+ndvi in 5-channel tensor
        # note: channels 0-4 (rgb,nir,ndvi) are in range [0..255], additional channels are float in any range !!! (hopefully but not necessarily in [-1..1] )
        img = np.zeros(shape=[self.img_size[0], self.img_size[1], self.n_channels], dtype=np.float)

        img[:, :, 0:3] = rgb_img
        img[:, :, 3] = nir_img
        img[:, :, 4] = ndvi_img
        # for i in range(self.num_orig_channels, self.n_channels):
        #     alpha = self.alpha_coeffs[i-self.num_orig_channels]
        #     img[:, :, i] = get_agct_img(img, alpha)

        img[idx_non_mask > 0] = np.zeros(shape=self.n_channels, dtype=np.float)  # [0, 0, 0, 0, 0]

        if self.save_training_images == 1:
            curr_prefix = self.save_training_folder + os.sep + prefix
            rgb_img = img[:, :, 0:3]
            nir_img = img[:, :, 3]
            ndvi_img = img[:, :, 4]
            lbl_img = lbl
            m.imsave(curr_prefix + '_0_gt_label.png', lbl_img)
            lbl_img = decode_segmap(lbl_img, self.label_colors).astype(np.uint8)
            m.imsave(curr_prefix + '_0_rgb.png', rgb_img)
            m.imsave(curr_prefix + '_0_nir.png', nir_img)
            m.imsave(curr_prefix + '_0_ndvi.png', ndvi_img)
            m.imsave(curr_prefix + '_0_gt_color.png', lbl_img)

        # to avoid wrong zero labels by affine transforms in augmentation (forward cheat)
        lbl = lbl + 1

        img[:, :, 0:self.num_orig_channels] = img[:, :, 0:self.num_orig_channels] / 255.0
        img = img.transpose(2, 0, 1)  # NHWC -> NCHW
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        if self.collage_augmentations is not None:
            img, lbl = self.collage_augmentations(img, lbl)
        if self.basic_augmentations is not None:
            img, lbl = self.basic_augmentations(img, lbl)

        if self.img_norm is True:
            for i in range(0, self.num_orig_channels):
                img[i, :, :] -= (self.mean_rgbn[i] / 255.0)
            for i in range(self.num_orig_channels, self.new_channels):
                img[i, :, :] -= self.mean_rgbn[i]

        # to avoid wrong zero labels by affine transforms in augmentation (backward cheat)
        tmp = lbl.numpy().astype(np.uint8)
        tmp[tmp == 0] = self.out_of_bounds_label + 1
        tmp = tmp - 1
        lbl = torch.from_numpy(tmp).long()

        if self.save_training_images == 1:
            curr_prefix = self.save_training_folder + os.sep + prefix
            img_to_save = img.numpy()
            img_to_save = img_to_save.transpose(1, 2, 0)
            if self.img_norm is True:
                img_to_save += (self.mean_rgbn / 255.0)
            img_to_save = 255.0 * img_to_save
            img_to_save = img_to_save.astype(np.uint8)
            rgb_img = img_to_save[:, :, 0:3]
            nir_img = img_to_save[:, :, 3]
            ndvi_img = img_to_save[:, :, 4]
            lbl_img = lbl.numpy().astype(np.uint8)
            m.imsave(curr_prefix + '_1_gt_label.png', lbl_img)
            lbl_img = decode_segmap(lbl_img, self.label_colors).astype(np.uint8)
            m.imsave(curr_prefix + '_1_rgb.png', rgb_img)
            m.imsave(curr_prefix + '_1_nir.png', nir_img)
            m.imsave(curr_prefix + '_1_ndvi.png', ndvi_img)
            m.imsave(curr_prefix + '_1_gt_color.png', lbl_img)

        return img, lbl


class agri6_ff_exp_23(agrivision6Loader_full_field):

    label_names = [
        'background',
        'double_plant',
        'planter_skip',
    ]

    label_colors = [[128, 128, 128],  # background
                    [0, 255, 0],
                    [0, 255, 255],
                    [0, 0, 0]]  # out-of-bound/mask

    out_of_bounds_label = 3
    classes_orig = [0, 1, 2, 3, 4, 5, 6, 7]
    classes_new =  [0, 0, 1, 2, 0, 0, 0, 3]

    def encode_segmap(self, mask):
        mask_new = np.zeros_like(mask)
        for i in range(0, len(self.classes_orig)):
            idx0 = self.classes_orig[i]
            idx1 = self.classes_new[i]
            mask_new[mask == idx0] = idx1
        return mask_new