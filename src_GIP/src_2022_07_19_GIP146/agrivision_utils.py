
import os
import numpy as np
import scipy.misc as misc
import skimage.measure
import skimage.morphology

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

import torchvision.transforms.functional as tf

from ptsemseg.loader.agrivision6_loader import agrivision6Loader

from ptsemseg.loader.agrivision_loader_utils import decode_segmap, \
                                                write_val_image_list, \
                                                read_val_image_list, \
                                                write_train_image_list, \
                                                read_train_image_list, \
                                                get_mean_rgbn_values_all, \
                                                get_ndvi, \
                                                get_evi


GT_LABELS = ["GT: background", "GT: cloud_shadow", "GT: double_plant", "GT: planter_skip", "GT: standing_water", "GT: waterway", "GT: weed_cluster", "GT: H_class"]
PR_LABELS = ["PR: background", "PR: cloud_shadow", "PR: double_plant", "PR: planter_skip", "PR: standing_water", "PR: waterway", "PR: weed_cluster", "PR: H_class"]


def save_val_loss_info(res_dir, loss_info):
    acc_txt_name = os.path.join(res_dir,'val_acc.txt')
    acc_fig_name = os.path.join(res_dir, 'val_acc.png')
    acc_txt = np.array(loss_info)
    # acc_txt = np.transpose(loss_info)
    iters = acc_txt[:,0]
    train_loss = acc_txt[:,1]
    val_loss = acc_txt[:,2]
    val_iou = acc_txt[:,3]
    np.savetxt(acc_txt_name, acc_txt, fmt='%.3f', delimiter='\t')

    plt.plot(iters, train_loss, 'r', label='Training loss')
    plt.plot(iters, val_loss, 'b', label='Validation loss')
    plt.plot(iters, val_iou, 'g', label='Validation IoU')

    plt.title('train/validation loss, validation iou')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(acc_fig_name)
    # plt.show()
    plt.close()


def save_train_loss_info(res_dir, loss_info, loss_names):
    acc_txt_name = os.path.join(res_dir,'train_acc.txt')
    acc_fig_name = os.path.join(res_dir, 'train_acc.png')
    acc_txt = np.array(loss_info)

    np.savetxt(acc_txt_name, acc_txt, fmt='%.3f', delimiter='\t')

    colors = ['empty', 'k', 'r', 'g', 'b', 'm', 'c']
    iters = acc_txt[:, 0]
    # mean_loss = acc_txt[:, 1]
    # cross_entropy = acc_txt[:, 2]
    # iou = acc_txt[:, 3]
    # iou_soft = acc_txt[:, 4]
    # lovasz = acc_txt[:, 5]
    # focal = acc_txt[:, 6]
    # plt.plot(iters, mean_loss, 'k', label='mean')
    # plt.plot(iters, cross_entropy, 'r', label='cross entropy')
    # plt.plot(iters, iou, 'g', label='iou')
    # plt.plot(iters, iou_soft, 'b', label='iou soft')
    # plt.plot(iters, lovasz, 'm', label='lovasz')
    # plt.plot(iters, focal, 'c', label='focal')

    for i in range(len(loss_names)-1, 0, -1):
        # reverse order to plot 'mean' on top
        plt.plot(iters, acc_txt[:,i], colors[i], label=loss_names[i])

    plt.title('train losses')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(acc_fig_name)
    # plt.show()
    plt.close()


def save_alpha_vals(res_dir, alpha_vals, alpha_grad_vals, n_channels):
    s1,s2 = alpha_grad_vals.shape
    for i in range(0,n_channels):
        alpha_name = os.path.join(res_dir, 'color_alpha_vals_' + str(i+1) + '.txt')
        alpha_grad_name = os.path.join(res_dir, 'color_alpha_grad_vals_' + str(i+1) + '.txt')
        alpha = alpha_vals[i:s1:n_channels, :]
        alpha_g = alpha_grad_vals[i:s1:n_channels, :]
        np.savetxt(alpha_name, alpha, fmt='%.6f', delimiter='\t')
        np.savetxt(alpha_grad_name, alpha_g, fmt='%.6f', delimiter='\t')

def save_best_alpha_vals(res_dir, best_alpha):
    file_name = os.path.join(res_dir, 'best_agct_vals.txt')
    np.savetxt(file_name, best_alpha, fmt='%.6f', delimiter='\t')


def save_best_score(res_dir, best_iou, class_iou):
    file_name = os.path.join(res_dir, 'best_score.txt')
    with open(file_name,'w') as f:
        f.write("best iou: %.3f\n" % best_iou)
        for k, v in class_iou.items():
            f.write("class %2d: %.3f\n" % (k,v))


def save_confusion_matrix(res_dir, confusion_matrix):
    fig_name = os.path.join(res_dir, 'confusion_matrix.png')

    num_labels = confusion_matrix.shape[0]
    gt_labels = GT_LABELS[0:num_labels]
    pr_labels = PR_LABELS[0:num_labels]
    conf_mat = np.copy(confusion_matrix)
    sum_gt = conf_mat.sum(axis=1)
    for i in range(0,num_labels):
        conf_mat[i,:] = conf_mat[i,:] / sum_gt[i]
    df_cm = pd.DataFrame(conf_mat, index=[i for i in gt_labels], columns=[i for i in pr_labels])
    sn.set(font_scale=0.5)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap='Blues', fmt='0.3f')  # font size
    plt.savefig(fig_name)
    plt.close()


def remove_small_components(map, min_pixels = 1000):
    ccl = skimage.measure.label(map, connectivity=1)
    num_ccl = ccl.max()
    for i in range(1,num_ccl+1):
        curr_region = (ccl==i)
        num_pixels = np.count_nonzero(curr_region)
        if num_pixels < min_pixels:
            map[curr_region == True] = 0
    return map

# def remove_small_components(map0, min_pixels = 1000):
#     map1 = np.copy(map0)
#     ccl = skimage.measure.label(map1, connectivity=1)
#     num_ccl = ccl.max()
#     for i in range(1,num_ccl+1):
#         curr_region = (ccl==i)
#         num_pixels = np.count_nonzero(curr_region)
#         if num_pixels < min_pixels:
#             map1[curr_region == True] = 0
#     return map1

def dilate_components(map, n_classes, radius=9):
    for i in range(1,n_classes+1):
        curr_region = (map==i)
        mask = np.zeros_like(map)
        mask[curr_region] = 1
        mask = skimage.morphology.dilation(mask, skimage.morphology.disk(radius=radius))
        map[mask==1] = i
    return map


def dilate_single_category(map, category_id=3, radius=5):
    curr_region = (map == category_id)
    mask = np.zeros_like(map)
    mask[curr_region] = 1
    mask = skimage.morphology.dilation(mask, skimage.morphology.disk(radius=radius))
    map[mask == 1] = category_id
    return map


def morph_open(map, n_classes, radius=5):
    for i in range(1,n_classes+1):
        curr_region = (map==i)
        mask = np.zeros_like(map)
        mask[curr_region] = 1
        mask = skimage.morphology.erosion(mask, skimage.morphology.disk(radius=radius))
        mask = skimage.morphology.dilation(mask, skimage.morphology.disk(radius=radius))
        map[mask==1] = i
    return map


def morph_close(map, n_classes, radius=5):
    for i in range(1,n_classes+1):
        curr_region = (map==i)
        mask = np.zeros_like(map)
        mask[curr_region] = 1
        mask = skimage.morphology.dilation(mask, skimage.morphology.disk(radius=radius))
        mask = skimage.morphology.erosion(mask, skimage.morphology.disk(radius=radius))
        map[mask==1] = i
    return map


def blackout_image_border(img, border_size=1, border_color=[0,0,0]):
    img_size = img.shape
    img[0:border_size, :, :] = border_color
    img[img_size[0]-border_size:img_size[0], :, :] = border_color
    img[:, 0:border_size, :] = border_color
    img[:, img_size[1]-border_size:img_size[1], :] = border_color
    return img


def best_multi_pred_7_classes(P):
    # hist = np.bincount(P, minlength=7)
    # best = np.argmax(hist)
    # return best
    return np.argmax(np.bincount(P, minlength=7))


def best_multi_pred_3_classes(P):
    return np.argmax(np.bincount(P, minlength=3))


def multi_rotation_prediction(single_pred, images, model, device, n_rotations=1, n_classes=7):

    if n_rotations <= 1:
        return single_pred

    multi_pred = np.copy(single_pred)
    scale1 = 1.0
    scale2 = 1.0
    for j in range(1, n_rotations):
        angle = j * (360.0 / n_rotations)
        if (angle % 90) != 0.0:
            scale1 = 1.0 / np.sqrt(2.0)
            scale2 = np.sqrt(2.0)
        if angle > 180:
            angle -= 360
        rotated_images = tf.affine(images, translate=[0, 0], scale=scale1, angle=angle, interpolation=tf.InterpolationMode.BILINEAR, shear=[0.0])
        rotated_images = rotated_images.to(device)
        rotated_outputs = model(rotated_images)
        # rotated_pred = rotated_outputs.data.max(1)[1].cpu().numpy()
        rotated_pred = rotated_outputs.data.max(1)[1].cpu()
        rotated_pred = tf.affine(rotated_pred, translate=[0, 0], scale=scale2, angle=-angle, interpolation=tf.InterpolationMode.NEAREST, shear=[0.0])
        rotated_pred = rotated_pred.numpy()
        multi_pred = np.concatenate((multi_pred, rotated_pred), axis=0)

        del rotated_images
        del rotated_outputs

    multi_pred = n_classes - 1 - multi_pred  # avoid advantage of '0' labeled class in argmax
    if n_classes == 7:
        hist = np.apply_along_axis(best_multi_pred_7_classes, axis=0, arr=multi_pred)
    elif n_classes == 3:
        hist = np.apply_along_axis(best_multi_pred_3_classes, axis=0, arr=multi_pred)
    hist = n_classes - 1 - hist  # avoid advantage to '0' labeled class in argmax

    pred = np.zeros_like(single_pred)
    pred[0, :, :] = hist
    return pred


def override_expert_prediction(pred, pred_exp):
    # override categories having expert labels 2,3

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
    return pred


def create_gt_label_images(data_dir, split_dir, img_size = (512,512), is_test = False):

    in_dir_rgb = os.path.join(data_dir, split_dir, 'images', 'rgb')
    out_dir_labels = os.path.join(data_dir, split_dir, 'gt_labels')
    if not os.path.exists(out_dir_labels):
        os.mkdir(out_dir_labels)
    if is_test is False:
        out_dir_colors = os.path.join(data_dir, split_dir, 'gt_colors')
        if not os.path.exists(out_dir_colors):
            os.mkdir(out_dir_colors)

    boundaries_dir = os.path.join(data_dir, split_dir, 'boundaries')
    masks_dir = os.path.join(data_dir, split_dir, 'masks')
    num_labels = len(agrivision6Loader.label_names)

    if is_test is False:
        label_dirs = num_labels * [None]
        label_file_names = num_labels * [None]
        all_labels = np.empty(shape=0,dtype=np.uint8)
        for i in range(0,num_labels):
            label_dirs[i] = os.path.join(data_dir, split_dir, 'labels', agrivision6Loader.label_names[i])

    for file_name in os.listdir(in_dir_rgb):
        if not file_name.endswith('.jpg'):
            continue
        # print(file_name)

        prefix = file_name.split('.')[0]
        boundary_file_name = os.path.join(boundaries_dir, prefix + '.png')
        boundary = misc.imread(boundary_file_name)
        mask_file_name = os.path.join(masks_dir, prefix + '.png')
        mask = misc.imread(mask_file_name)
        gt_label = np.zeros(shape=img_size, dtype=np.uint8)

        if is_test is False:
            for i in range(1,num_labels):
                label_file_names[i] = label_dirs[i] + prefix + '.png'
                curr_label = misc.imread(label_file_names[i])
                gt_label[curr_label == 255] = i

        #gt_label[gt_label == 0] = num_labels    # background, keep as zero label
        gt_label[boundary == 0] = num_labels    # out-of-bounds labels
        gt_label[mask == 0] = num_labels        # out-of-bounds labels

        if is_test is False:
            gt_color = decode_segmap(gt_label, agrivision6Loader.label_colors).astype(np.uint8)

        if is_test is False:
            curr_labels = np.unique(gt_label)
            all_labels = np.concatenate((all_labels,curr_labels))
            all_labels = np.unique(all_labels)
            print(file_name, curr_labels, all_labels)
        else:
            print(file_name)

        out_file = os.path.join(out_dir_labels, prefix + '_gt_label.png')
        misc.imsave(out_file, gt_label)
        if is_test is False:
            out_file = os.path.join(out_dir_colors, prefix + '_gt_color.png')
            misc.imsave(out_file, gt_color)


def create_gt_label_images_multi_prediction(data_dir, split_dir, img_size = (512,512), min_overlap_pixels=0):

    in_dir_rgb = os.path.join(data_dir, split_dir, 'images', 'rgb')
    in_dir_nir = os.path.join(data_dir, split_dir, 'images', 'nir')
    out_dir_labels = os.path.join(data_dir, split_dir, 'gt_labels')
    if not os.path.exists(out_dir_labels):
        os.mkdir(out_dir_labels)
    out_dir_colors = os.path.join(data_dir, split_dir, 'gt_colors')
    if not os.path.exists(out_dir_colors):
        os.mkdir(out_dir_colors)

    boundaries_dir = os.path.join(data_dir, split_dir, 'boundaries')
    masks_dir = os.path.join(data_dir, split_dir, 'masks')
    num_labels = len(agrivision6Loader.label_names)
    num_categories = num_labels - 1   # without background

    label_dirs = num_categories * [None]
    label_file_names = num_categories * [None]
    for i in range(0, num_categories):
        label_dirs[i] = os.path.join(data_dir, split_dir, 'labels', agrivision6Loader.label_names[i+1])

    in_dir_rgb_ext = os.path.join(data_dir, split_dir, 'images', 'rgb_ext')
    in_dir_nir_ext = os.path.join(data_dir, split_dir, 'images', 'nir_ext')
    if not os.path.exists(in_dir_rgb_ext):
        os.mkdir(in_dir_rgb_ext)
    if not os.path.exists(in_dir_nir_ext):
        os.mkdir(in_dir_nir_ext)

    for file_name in os.listdir(in_dir_rgb):
        if not file_name.endswith('.jpg'):
            continue
        print(file_name + " start")

        prefix = file_name.split('.')[0]
        boundary_file_name = os.path.join(boundaries_dir, prefix + '.png')
        boundary = misc.imread(boundary_file_name)
        mask_file_name = os.path.join(masks_dir, prefix + '.png')
        mask = misc.imread(mask_file_name)
        boundary = boundary.astype(np.bool)
        mask = mask.astype(np.bool)
        out_of_bounds = np.logical_not(np.logical_and(boundary ,mask))
        background = np.ones(shape=img_size, dtype=np.bool)
        gt_label = np.zeros(shape=img_size, dtype=np.uint8)
        img_labels = np.zeros(shape=(img_size[0],img_size[1],num_categories), dtype=np.bool)

        for i in range(0,num_categories):
            label_file_names[i] =  os.path.join(label_dirs[i], prefix + '.png')
            curr_label = misc.imread(label_file_names[i])
            img_labels[:,:,i] = curr_label.astype(np.bool)
            background = np.logical_and(background, np.logical_not(img_labels[:,:,i]))
            gt_label[curr_label == 255] = i+1

        out_of_bounds = np.logical_and(out_of_bounds, background)
        gt_label[out_of_bounds == True] = num_labels  # out-of-bounds labels
        gt_color = decode_segmap(gt_label, agrivision6Loader.label_colors).astype(np.uint8)

        out_file = os.path.join(out_dir_labels, prefix + '_gt_label.png')
        misc.imsave(out_file, gt_label)
        out_file = os.path.join(out_dir_colors, prefix + '_gt_color.png')
        misc.imsave(out_file, gt_color)

        do_overlap = False
        multi_labels = []
        for i in range(0, num_categories-1):
            for j in range(i+1, num_categories):
                overlap = np.logical_and(img_labels[:,:,i], img_labels[:,:,j])
                num_overlap = np.sum(overlap)
                if num_overlap > min_overlap_pixels:
                    multi_labels.append(i)
                    multi_labels.append(j)
                    do_overlap = True

        if do_overlap is True:
            # create additional gt-label files
            # copy rgb and nir files with separate names
            print("doing overlap")
            multi_labels = np.unique(multi_labels)
            num_multi = multi_labels.size
            idx_orig = list(range(0, num_categories))
            for n in range(1,num_multi):
                suffix = '_multi_' + str(n)
                idx = idx_orig.copy()
                multi_shuffle = np.roll(multi_labels,n)
                for i in range(0,num_multi):
                    idx[multi_labels[i]] = multi_shuffle[i]

                gt_label = np.zeros(shape=img_size, dtype=np.uint8)
                for i in range(0,num_categories):
                    # idx = num_categories - 1 - i    # backwards pass
                    curr_label = img_labels[:, :, idx[i]]
                    gt_label[curr_label == True] = idx[i] + 1

                gt_label[out_of_bounds == True] = num_labels  # out-of-bounds labels
                gt_color = decode_segmap(gt_label, agrivision6Loader.label_colors).astype(np.uint8)

                new_prefix = prefix + suffix
                out_file = os.path.join(out_dir_labels, new_prefix + '_gt_label.png')
                misc.imsave(out_file, gt_label)
                out_file = os.path.join(out_dir_colors, new_prefix + '_gt_color.png')
                misc.imsave(out_file, gt_color)

                in_file = os.path.join(in_dir_rgb, prefix + '.jpg')
                out_file = os.path.join(in_dir_rgb_ext, new_prefix + '.jpg')
                img = misc.imread(in_file)
                misc.imsave(out_file, img)
                in_file = os.path.join(in_dir_nir, prefix + '.jpg')
                out_file = os.path.join(in_dir_nir_ext, new_prefix + '.jpg')
                img = misc.imread(in_file)
                misc.imsave(out_file, img)

        print(file_name + " end")
    print("end_func")


def create_rgb_nir_ndvi_evi_gt_images(data_dir, file_list, out_dir, img_size = (512,512), is_test = False):

    split_dirs, file_prefixes = read_val_image_list(file_list)
    num_files = len(file_prefixes)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    border_size = 1
    num_labels = len(agrivision6Loader.label_names)

    for i in range (0,num_files):
        rgb_path = os.path.join(data_dir, split_dirs[i], 'images', 'rgb', file_prefixes[i] + '.jpg')
        nir_path = os.path.join(data_dir, split_dirs[i], 'images', 'nir', file_prefixes[i] + '.jpg')
        lbl_path = os.path.join(data_dir, split_dirs[i], 'gt_labels', file_prefixes[i] + '_gt_label.png')
        boundaries_path = os.path.join(data_dir, split_dirs[i], 'boundaries', file_prefixes[i] + '.png')
        masks_path = os.path.join(data_dir, split_dirs[i], 'masks', file_prefixes[i] + '.png')

        rgb = np.array(misc.imread(rgb_path), dtype=np.uint8)
        nir = np.array(misc.imread(nir_path), dtype=np.uint8)
        ndvi = get_ndvi(rgb, nir).astype(np.uint8)
        evi = get_evi(rgb, nir, min_val = -10.0, max_val = 10.0).astype(np.uint8)

        gt_label = np.zeros(shape=img_size, dtype=np.uint8)
        if is_test is False:
            gt_label = np.array(misc.imread(lbl_path), dtype=np.uint8)
        else:
            boundary = np.array(misc.imread(boundaries_path), dtype=np.uint8)
            mask = np.array(misc.imread(masks_path), dtype=np.uint8)
            gt_label[boundary == 0] = num_labels    # out-of-bounds labels
            gt_label[mask == 0] = num_labels        # out-of-bounds labels
        gt_color = decode_segmap(gt_label, agrivision6Loader.label_colors).astype(np.uint8)

        idx_non_mask = np.zeros(shape=img_size, dtype=np.uint8)
        idx_non_mask[gt_label == num_labels] = 1

        rgb[idx_non_mask > 0] = [0,0,0]
        nir[idx_non_mask > 0] = 0
        ndvi[idx_non_mask > 0] = 0
        evi[idx_non_mask > 0] = 0

        nir = np.stack((nir, nir, nir), axis=2)
        ndvi = np.stack((ndvi, ndvi, ndvi), axis=2)
        evi = np.stack((evi, evi, evi), axis=2)
        rgb = blackout_image_border(rgb, border_size=border_size)
        nir = blackout_image_border(nir, border_size=border_size)
        ndvi = blackout_image_border(ndvi, border_size=border_size)
        evi = blackout_image_border(evi, border_size=border_size)
        gt_color = blackout_image_border(gt_color, border_size=border_size)
        out_img = np.concatenate((rgb, nir, ndvi, evi, gt_color), axis=1)
        out_file = os.path.join(out_dir, file_prefixes[i] + '.png')
        misc.imsave(out_file, out_img)
        print(file_prefixes[i])
        dummy = 0





if __name__ == "__main__":

    data_dir = 'C:/alon/datasets/Agriculture-Vision/'

    # split_dir = 'train'     # 'train'   'val' 'test
    # create_gt_label_images(data_dir, split_dir, is_test=False)
    # create_gt_label_images_multi_prediction(data_dir, split_dir, min_overlap_pixels=100)


    train_img_file = 'configs/agri2020_train_all_mp.txt'
    val_img_file = 'configs/agri2020_val_all_mp.txt'
    write_val_image_list(data_dir, 'val', val_img_file)
    write_train_image_list(data_dir,'train', train_img_file)
    # splits_val, files_val = read_val_image_list(val_img_file)
    # splits_train, files_train, coords, pix_per_class = read_train_image_list(train_img_file)
    # test_img_file = 'configs/agri2020_test_small.txt'
    # write_val_image_list(data_dir, 'test', test_img_file)
    #
    # train_img_file = 'configs/agri2020_train_ep_3888.txt'
    # val_img_file = 'configs/agri2020_val_small.txt'
    # write_val_image_list(data_dir, 'val_small', val_img_file)
    # write_train_image_list(data_dir, 'train_ep_3888', train_img_file)
    # splits_val, files_val = read_val_image_list(val_img_file)
    # splits_train, files_train, coords, pix_per_class = read_train_image_list(train_img_file)
    #
    # mean_vals = get_mean_rgbn_values_all(data_dir)
    #
    # file_list = 'configs/agri2020_val_ep3_318.txt'
    # out_dir = os.path.join(data_dir, 'agri2020_planter_skip_318_visuals')
    # create_rgb_nir_ndvi_evi_gt_images(data_dir, file_list=file_list, out_dir=out_dir, is_test=False)
    # file_list = 'configs/agri2020_test_all.txt'
    # out_dir = os.path.join(data_dir, 'agri2020_test_visuals')
    # create_rgb_nir_ndvi_evi_gt_images(data_dir, file_list=file_list, out_dir=out_dir, is_test=True)

    dummy = 0