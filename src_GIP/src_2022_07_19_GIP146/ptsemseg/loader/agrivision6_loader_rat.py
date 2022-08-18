import os
import numpy as np
import scipy.misc as m
import torch
from torch.utils import data

from ptsemseg.augmentations.augmentations import BasicAugmentations, CollageAugmentations
from .agrivision_loader_utils import decode_segmap, get_ndvi, get_evi, get_agct_img, read_agct_coeffs, read_val_image_list, read_train_image_list
from .agrivision6_loader import agrivision6Loader


class agrivision6Loader_rat1(agrivision6Loader):

    def __init__(self,
                root,
                split='train',
                img_list='',
                img_size=(512, 512),
                n_channels = 9,
                augmentations=None,
                debug_info = None,
                img_norm=False,
                test_mode=False):
        super(agrivision6Loader_rat1, self).__init__(root, split, img_list, img_size, n_channels, augmentations, debug_info, img_norm, test_mode)

        self.num_orig_channels = 5
        self.n_channels = n_channels
        # self.mean_rgbn = np.concatenate((self.mean_rgbn, 127.5*np.ones(shape=(1,self.new_channels),dtype=float)))
        # self.mean_rgbn = np.concatenate((self.mean_rgbn, np.zeros(shape=self.new_channels, dtype=float)))
        self.ones = 32.0 * np.ones(shape=[self.img_size[0], self.img_size[1]], dtype=np.float)


    def __getitem__(self, index):
        # read rgb + nir images
        prefix = self.file_prefixes[index]
        split_dir = self.split_prefixes[index]
        rgb_path = os.path.join(self.root, split_dir, 'images', 'rgb', prefix + '.jpg')
        nir_path = os.path.join(self.root, split_dir, 'images', 'nir', prefix + '.jpg')
        lbl_path = os.path.join(self.root, split_dir, 'gt_labels', prefix + '_gt_label.png')

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
        img = np.zeros(shape=[self.img_size[0],self.img_size[1],self.n_channels],dtype=np.float)

        img[:,:,0:3] = rgb_img
        img[:,:,3] = nir_img
        img[:,:,4] = ndvi_img

        # 1/R, 1/G, 1/B, 1/N
        img[:, :, 5] = np.divide(self.ones, img[:, :, 0] + 1.0)
        img[:, :, 6] = np.divide(self.ones, img[:, :, 1] + 1.0)
        img[:, :, 7] = np.divide(self.ones, img[:, :, 2] + 1.0)
        img[:, :, 8] = np.divide(self.ones, img[:, :, 3] + 1.0)

        img[idx_non_mask > 0] = np.zeros(shape=self.n_channels,dtype=np.float) # [0, 0, 0, 0, 0]

        if self.save_training_images == 1:
            curr_prefix = self.save_training_folder + os.sep + prefix
            rgb_img = img[:,:,0:3]
            nir_img = img[:,:,3]
            ndvi_img = img[:,:,4]
            lbl_img = lbl
            m.imsave(curr_prefix + '_0_gt_label.png', lbl_img)
            lbl_img = decode_segmap(lbl_img,self.label_colors).astype(np.uint8)
            m.imsave(curr_prefix + '_0_rgb.png', rgb_img)
            m.imsave(curr_prefix + '_0_nir.png', nir_img)
            m.imsave(curr_prefix + '_0_ndvi.png', ndvi_img)
            m.imsave(curr_prefix + '_0_gt_color.png', lbl_img)
            for i in range(self.num_orig_channels, self.n_channels):
                out_name = curr_prefix + '_0_C' + str(i).zfill(2) + '.png'
                m.imsave(out_name, img[:,:,i])


        # to avoid wrong zero labels by affine transforms in augmentation (forward cheat)
        lbl = lbl + 1

        img[:,:,0:self.num_orig_channels] = img[:,:,0:self.num_orig_channels] / 255.0
        img = img.transpose(2, 0, 1)            # NHWC -> NCHW
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        if self.collage_augmentations is not None:
            img, lbl = self.collage_augmentations(img, lbl)
        if self.basic_augmentations is not None:
            img, lbl = self.basic_augmentations(img, lbl)

        if self.img_norm is True:
            for i in range (0,self. num_orig_channels):
                img[i,:,:] -= (self.mean_rgbn[i] / 255.0)
            for i in range (self.num_orig_channels, self.new_channels):
                img[i, :, :] -= self.mean_rgbn[i]

        # to avoid wrong zero labels by affine transforms in augmentation (backward cheat)
        tmp = lbl.numpy().astype(np.uint8)
        tmp[tmp==0] = self.out_of_bounds_label + 1
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
            m.imsave(curr_prefix + '_1_gt_label.png',lbl_img)
            lbl_img = decode_segmap(lbl_img,self.label_colors).astype(np.uint8)
            m.imsave(curr_prefix + '_1_rgb.png', rgb_img)
            m.imsave(curr_prefix + '_1_nir.png', nir_img)
            m.imsave(curr_prefix + '_1_ndvi.png', ndvi_img)
            m.imsave(curr_prefix + '_1_gt_color.png', lbl_img)
            for i in range(self.num_orig_channels, self.n_channels):
                out_name = curr_prefix + '_1_C' + str(i).zfill(2) + '.png'
                m.imsave(out_name, img_to_save[:,:,i])

        return img, lbl


class agrivision6Loader_rat2(agrivision6Loader):

    def __init__(self,
                 root,
                 split='train',
                 img_list='',
                 img_size=(512, 512),
                 n_channels=17,
                 augmentations=None,
                 debug_info=None,
                 img_norm=False,
                 test_mode=False):
        super(agrivision6Loader_rat2, self).__init__(root, split, img_list, img_size, n_channels, augmentations,
                                                     debug_info, img_norm, test_mode)

        self.num_orig_channels = 5
        self.n_channels = n_channels
        # self.mean_rgbn = np.concatenate((self.mean_rgbn, 127.5*np.ones(shape=(1,self.new_channels),dtype=float)))
        # self.mean_rgbn = np.concatenate((self.mean_rgbn, np.zeros(shape=self.new_channels, dtype=float)))

    def __getitem__(self, index):
        # read rgb + nir images
        prefix = self.file_prefixes[index]
        split_dir = self.split_prefixes[index]
        rgb_path = os.path.join(self.root, split_dir, 'images', 'rgb', prefix + '.jpg')
        nir_path = os.path.join(self.root, split_dir, 'images', 'nir', prefix + '.jpg')
        lbl_path = os.path.join(self.root, split_dir, 'gt_labels', prefix + '_gt_label.png')

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

        # R/G, R/B, R/N
        img[:, :, 5] = np.divide(img[:, :, 0], img[:, :, 1] + 1.0)
        img[:, :, 6] = np.divide(img[:, :, 0], img[:, :, 2] + 1.0)
        img[:, :, 7] = np.divide(img[:, :, 0], img[:, :, 3] + 1.0)

        # G/R, G/B, G/N
        img[:, :, 8] = np.divide(img[:, :, 1], img[:, :, 0] + 1.0)
        img[:, :, 9] = np.divide(img[:, :, 1], img[:, :, 2] + 1.0)
        img[:, :, 10] = np.divide(img[:, :, 1], img[:, :, 3] + 1.0)

        # B/R, B/G, B/N
        img[:, :, 11] = np.divide(img[:, :, 2], img[:, :, 0] + 1.0)
        img[:, :, 12] = np.divide(img[:, :, 2], img[:, :, 1] + 1.0)
        img[:, :, 13] = np.divide(img[:, :, 2], img[:, :, 3] + 1.0)

        # N/R, N/G, N/B
        img[:, :, 14] = np.divide(img[:, :, 3], img[:, :, 0] + 1.0)
        img[:, :, 15] = np.divide(img[:, :, 3], img[:, :, 1] + 1.0)
        img[:, :, 16] = np.divide(img[:, :, 3], img[:, :, 2] + 1.0)

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
            for i in range(self.num_orig_channels, self.n_channels):
                out_name = curr_prefix + '_0_C' + str(i).zfill(2) + '.png'
                m.imsave(out_name, img[:,:,i])

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
            for i in range(self.num_orig_channels, self.n_channels):
                out_name = curr_prefix + '_1_C' + str(i).zfill(2) + '.png'
                m.imsave(out_name, img_to_save[:, :, i])

        return img, lbl


class agrivision6Loader_rat3(agrivision6Loader):

    def __init__(self,
                 root,
                 split='train',
                 img_list='',
                 img_size=(512, 512),
                 n_channels=21,
                 augmentations=None,
                 debug_info=None,
                 img_norm=False,
                 test_mode=False):
        super(agrivision6Loader_rat3, self).__init__(root, split, img_list, img_size, n_channels, augmentations,
                                                     debug_info, img_norm, test_mode)

        self.num_orig_channels = 5
        self.n_channels = n_channels
        # self.mean_rgbn = np.concatenate((self.mean_rgbn, 127.5*np.ones(shape=(1,self.new_channels),dtype=float)))
        # self.mean_rgbn = np.concatenate((self.mean_rgbn, np.zeros(shape=self.new_channels, dtype=float)))
        self.ones = 32.0 * np.ones(shape=[self.img_size[0], self.img_size[1]], dtype=np.float)

    def __getitem__(self, index):
        # read rgb + nir images
        prefix = self.file_prefixes[index]
        split_dir = self.split_prefixes[index]
        rgb_path = os.path.join(self.root, split_dir, 'images', 'rgb', prefix + '.jpg')
        nir_path = os.path.join(self.root, split_dir, 'images', 'nir', prefix + '.jpg')
        lbl_path = os.path.join(self.root, split_dir, 'gt_labels', prefix + '_gt_label.png')

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

        # R/G, R/B, R/N
        img[:, :, 5] = np.divide(img[:, :, 0], img[:, :, 1] + 1.0)
        img[:, :, 6] = np.divide(img[:, :, 0], img[:, :, 2] + 1.0)
        img[:, :, 7] = np.divide(img[:, :, 0], img[:, :, 3] + 1.0)

        # G/R, G/B, G/N
        img[:, :, 8] = np.divide(img[:, :, 1], img[:, :, 0] + 1.0)
        img[:, :, 9] = np.divide(img[:, :, 1], img[:, :, 2] + 1.0)
        img[:, :, 10] = np.divide(img[:, :, 1], img[:, :, 3] + 1.0)

        # B/R, B/G, B/N
        img[:, :, 11] = np.divide(img[:, :, 2], img[:, :, 0] + 1.0)
        img[:, :, 12] = np.divide(img[:, :, 2], img[:, :, 1] + 1.0)
        img[:, :, 13] = np.divide(img[:, :, 2], img[:, :, 3] + 1.0)

        # N/R, N/G, N/B
        img[:, :, 14] = np.divide(img[:, :, 3], img[:, :, 0] + 1.0)
        img[:, :, 15] = np.divide(img[:, :, 3], img[:, :, 1] + 1.0)
        img[:, :, 16] = np.divide(img[:, :, 3], img[:, :, 2] + 1.0)

        # 1/R, 1/G, 1/B, 1/N
        img[:, :, 17] = np.divide(self.ones, img[:, :, 0] + 1.0)
        img[:, :, 18] = np.divide(self.ones, img[:, :, 1] + 1.0)
        img[:, :, 19] = np.divide(self.ones, img[:, :, 2] + 1.0)
        img[:, :, 20] = np.divide(self.ones, img[:, :, 3] + 1.0)

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
            for i in range(self.num_orig_channels, self.n_channels):
                out_name = curr_prefix + '_0_C' + str(i).zfill(2) + '.png'
                m.imsave(out_name, img[:,:,i])

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
            for i in range(self.num_orig_channels, self.n_channels):
                out_name = curr_prefix + '_1_C' + str(i).zfill(2) + '.png'
                m.imsave(out_name, img_to_save[:, :, i])

        return img, lbl