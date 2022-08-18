import os
import numpy as np
import scipy.misc as m




def decode_segmap(map, label_colors):
    n_classes = len(label_colors)
    r = map.copy()
    g = map.copy()
    b = map.copy()
    for i in range(0, n_classes):
        r[map == i] = label_colors[i][0]
        g[map == i] = label_colors[i][1]
        b[map == i] = label_colors[i][2]
    rgb = np.zeros((map.shape[0], map.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb


def get_ndvi(rgb, nir):
    # returns np.float in range [0..255]
    r = rgb[:,:,0].astype(np.float)
    nir = nir.astype(np.float)
    nomin = nir - r
    denom = nir + r
    ndvi = np.divide(nomin, denom, out=np.zeros(nomin.shape, dtype=np.float), where=denom != 0.0)
    ndvi = 127.5 * (ndvi + 1.0)
    return ndvi


def get_evi(rgb, nir, min_val = -3.0, max_val = 3.0):
    # returns np.float in range [0..255]
    g = 2.5
    c1 = 6.0
    c2 = 7.5
    l = 1.0
    r = rgb[:,:,0].astype(np.float) / 255.0
    b = rgb[:,:,2].astype(np.float) / 255.0
    nir = nir.astype(np.float) / 255.0
    nomin = g * (nir - r)
    denom = nir + c1*r - c2*b + l
    evi = np.divide(nomin, denom, out=np.zeros(nomin.shape, dtype=np.float), where=denom != 0.0)
    evi[evi < min_val] = min_val
    evi[evi > max_val] = max_val
    coeff = 255.0 / (max_val - min_val)
    evi = coeff * (evi - min_val)
    return evi


def get_agct_img(img, alpha):
    nomin = alpha[0] * img[:, :, 0] + alpha[1] * img[:, :, 1] + alpha[2] * img[:, :, 2] + alpha[3] * img[:, :, 3] + alpha[4]
    denom = alpha[5] * img[:, :, 0] + alpha[6] * img[:, :, 1] + alpha[7] * img[:, :, 2] + alpha[8] * img[:, :, 3] + alpha[9] + 1e-12
    C = np.divide(nomin, denom, out=np.zeros(nomin.shape, dtype=np.float), where=denom != 0.0)
    # C = 127.5 * (C + 1.0)
    return C


# def get_savi_img(img, alpha):
#     nomin = (1.0 + alpha[9]) * (alpha[0] * img[:, :, 0] + alpha[3] * img[:, :, 3])
#     denom = alpha[5] * img[:, :, 0] + alpha[8] * img[:, :, 3] + alpha[9] + 1e-12
#     C = np.divide(nomin, denom, out=np.zeros(nomin.shape, dtype=np.float), where=denom != 0.0)
#     # C = 127.5 * (C + 1.0)
#     return C


def read_agct_coeffs(file_name):
    coeffs = np.loadtxt(file_name, dtype=float, delimiter='\t')
    return coeffs


def get_mean_rgbn_values(data_dir):
    rgb_dir = os.path.join(data_dir, 'images', 'rgb')
    nir_dir = os.path.join(data_dir, 'images', 'nir')
    ext = 'jpg'
    prefix_list = [fn for fn in os.listdir(rgb_dir) if fn.endswith(ext)]
    num_files = len(prefix_list)
    m_vals_per_img = np.zeros(shape=(num_files, 5), dtype=np.float)
    for i in range(0, num_files):
        print(i, prefix_list[i])
        rgb = m.imread(os.path.join(rgb_dir, prefix_list[i])).astype(float)
        nir = m.imread(os.path.join(nir_dir, prefix_list[i])).astype(float)
        # nomin = nir - rgb[:, :, 0]
        # denom = nir + rgb[:, :, 0]
        # ndvi = np.divide(nomin, denom, out=np.zeros(nomin.shape, dtype=np.float), where=denom != 0.0)
        # ndvi = (127.5 * (ndvi + 1.0))
        ndvi = get_ndvi(rgb, nir)
        m_vals_per_img[i, 0] = np.mean(rgb[:, :, 0])
        m_vals_per_img[i, 1] = np.mean(rgb[:, :, 1])
        m_vals_per_img[i, 2] = np.mean(rgb[:, :, 2])
        m_vals_per_img[i, 3] = np.mean(nir)
        m_vals_per_img[i, 4] = np.mean(ndvi)
    mean_vals = np.mean(m_vals_per_img, axis=0)
    print(mean_vals)
    return num_files, mean_vals


def get_mean_rgbn_values_all(data_dir):
    split_dir = ['test', 'val', 'train']
    num_files = [0,0,0]
    mean_vals = np.zeros(shape=(len(split_dir),5), dtype=np.float)
    for i in range (0,len(split_dir)):
        num_files[i], mean_vals[i,:] = get_mean_rgbn_values(os.path.join(data_dir,split_dir[i]))
    total_files = np.sum(num_files)
    mean_vals_all = (num_files[0] * mean_vals[0,:] + num_files[1] * mean_vals[1,:] + num_files[2] * mean_vals[2,:]) / total_files
    return mean_vals_all


def write_val_image_list(data_dir, split_dir, file_name):
    rgb_dir = os.path.join(data_dir, split_dir, 'images', 'rgb')
    ext = 'jpg'
    prefix_list = [fn for fn in os.listdir(rgb_dir) if fn.endswith(ext)]
    skip = 1
    with open(file_name,'w') as f:
        for i in range(0, len(prefix_list), skip):
            # prefix = prefix_list[i]
            prefix = prefix_list[i].split(sep='.')[0]
            # f.write("%s\t%s\n" % (split_dir, prefix))
            fmt_str = "{:5}\t{:40}\n"
            print_str = fmt_str.format(split_dir, prefix)
            # print_str = fmt_str.format('val', prefix)
            f.write(print_str)


def read_val_image_list(file_name):
    num_lines = sum(1 for _ in open(file_name))
    splits = [None] * num_lines
    files = [None] * num_lines
    with open(file_name, 'r') as f:
        for i, line in enumerate(f):
            values = line.split()
            splits[i] = values[0]
            files[i] = values[1]  # values[1].split(sep='.')[0]
    return (splits, files)


def write_train_image_list(data_dir, split_dir, file_name):
    rgb_dir = os.path.join(data_dir, split_dir, 'images', 'rgb')
    ext = 'jpg'
    prefix_list = [fn for fn in os.listdir(rgb_dir) if fn.endswith(ext)]
    skip = 1
    coords = [0, 0]
    with open(file_name,'w') as f:
        for i in range(0, len(prefix_list), skip):
            # prefix = prefix_list[i]
            prefix = prefix_list[i].split(sep='.')[0]
            # f.write("%s\t%s\t%5d\t%5d\n" % (split_dir, prefix, x0 + i, y0 + i))
            fmt_str = "{:5}\t{:40}\t{:05d}\t{:05d}\n"
            # print_str = fmt_str.format(split_dir, prefix, coords[0] + i, coords[1] + i)
            print_str = fmt_str.format('train', prefix, coords[0] + i, coords[1] + i)
            f.write(print_str)


def read_train_image_list(file_name):
    num_lines = sum(1 for _ in open(file_name))
    splits = [None] * num_lines
    files = [None] * num_lines
    coords = np.zeros(shape=(num_lines,2), dtype=int)
    pix_per_class = np.zeros(shape=(num_lines,8), dtype=int)
    with open(file_name, 'r') as f:
        for i, line in enumerate(f):
            values = line.split()
            splits[i] = values[0]
            files[i] = values[1]  # values[1].split(sep='.')[0]
            # coords[i, 0] = int(values[2])
            # coords[i, 1] = int(values[3])
    return (splits, files, coords, pix_per_class)


def read_train_image_list_full_field(file_name):
    num_lines = sum(1 for _ in open(file_name))
    splits = [None] * num_lines
    files = [None] * num_lines
    coords = np.zeros(shape=(num_lines,2), dtype=int)
    pix_per_class = np.zeros(shape=(num_lines,8), dtype=int)
    with open(file_name, 'r') as f:
        for i, line in enumerate(f):
            values = line.split()
            splits[i] = values[0]
            files[i] = values[1]  # values[1].split(sep='.')[0]
            coords[i, 0] = int(values[2])
            coords[i, 1] = int(values[3])
    return (splits, files, coords, pix_per_class)