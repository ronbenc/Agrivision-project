import math
import numbers
import random
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms.functional as tf
# import torchvision.transforms
from PIL import Image, ImageOps

FILL_COLOR = 0  # 250

class BasicAugmentations(object):
    def __init__(self, basic_params):
        self.saturation = basic_params.get("saturation")
        self.hue = basic_params.get("hue")
        self.brightness = basic_params.get("brightness")
        self.gamma = basic_params.get("gamma")
        self.contrast = basic_params.get("contrast")
        self.degrees = basic_params.get("rotate")
        self.shear = basic_params.get("shear")
        self.scale = basic_params.get("scale")
        self.translate = basic_params.get("translate")
        self.hflip = basic_params.get("hflip")

    def __call__(self, img, lbl):
        img[0:3, :, :] = tf.adjust_saturation(img[0:3, :, :], random.uniform(1 - self.saturation, 1 + self.saturation))
        img[0:3, :, :] = tf.adjust_hue(img[0:3, :, :], random.uniform(-self.hue, self.hue))
        img[0:3, :, :] = tf.adjust_contrast(img[0:3, :, :], random.uniform(1 - self.contrast, 1 + self.contrast))
        img[0:3, :, :] = tf.adjust_brightness(img[0:3, :, :], random.uniform(1 - self.brightness, 1 + self.brightness))
        img[0:3, :, :] = tf.adjust_gamma(img[0:3, :, :], random.uniform(1, 1 + self.gamma))

        img[3, :, :] = tf.adjust_brightness(img[3, :, :], random.uniform(1 - self.brightness, 1 + self.brightness))
        img[3, :, :] = tf.adjust_gamma(img[3, :, :], random.uniform(1, 1 + self.gamma))
        img[4, :, :] = tf.adjust_brightness(img[4, :, :], random.uniform(1 - self.brightness, 1 + self.brightness))
        img[4, :, :] = tf.adjust_gamma(img[4, :, :], random.uniform(1, 1 + self.gamma))

        rotate_degree = 2 * random.random() * self.degrees - self.degrees
        scale_f = (self.scale[1] - self.scale[0]) * random.random() + self.scale[0]
        tu = 2 * random.random() * self.translate[0] - self.translate[0]
        tv = 2 * random.random() * self.translate[1] - self.translate[1]
        shear_x = 2 * random.random() * self.shear - self.shear
        shear_y = 2 * random.random() * self.shear - self.shear

        do_hflip = False
        if random.random() < self.hflip:
            do_hflip = True

        # do_hflip = False
        # rotate_degree = 0
        # scale_f = 1.0
        # tu = 0
        # tv = 0

        img[:, :, :] = tf.affine(img[:, :, :], translate=[tu, tv], scale=scale_f, angle=rotate_degree, interpolation=tf.InterpolationMode.BILINEAR, shear=[shear_x,shear_y])
        if do_hflip is True:
            img = tf.hflip(img)

        tmp = torch.unsqueeze(lbl, 0)
        tmp = tf.affine(tmp, translate=[tu, tv], scale=scale_f, angle=rotate_degree, interpolation=tf.InterpolationMode.NEAREST, shear=[shear_x,shear_y])
        if do_hflip is True:
            tmp = tf.hflip(tmp)
        lbl = tmp[0, :, :]

        return img, lbl

class CollageAugmentations(object):
    def __init__(self, collage_params):
        self.n_patches = collage_params.get("n_patches")

    def __call__(self, img, lbl):

        return img, lbl



class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, mask):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            mask = Image.fromarray(mask, mode="L")
            self.PIL2Numpy = True

        assert img.size == mask.size
        for a in self.augmentations:
            img, mask = a(img, mask)

        if self.PIL2Numpy:
            img, mask = np.array(img), np.array(mask, dtype=np.uint8)

        return img, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return (img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST))

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return (img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)))


class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), mask


class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, mask):
        assert img.size == mask.size
        return (
            tf.adjust_saturation(img, random.uniform(1 - self.saturation, 1 + self.saturation)),
            mask,
        )


class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_hue(img, random.uniform(-self.hue, self.hue)), mask


class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_brightness(img, random.uniform(1 - self.bf, 1 + self.bf)), mask


class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_contrast(img, random.uniform(1 - self.cf, 1 + self.cf)), mask


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        return (img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)))


class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT))
        return img, mask


class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM))
        return img, mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return (img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST))


class RandomTranslate(object):
    def __init__(self, offset):
        # tuple (delta_x, delta_y)
        self.offset = offset

    def __call__(self, img, mask):
        assert img.size == mask.size
        x_offset = int(2 * (random.random() - 0.5) * self.offset[0])
        y_offset = int(2 * (random.random() - 0.5) * self.offset[1])

        x_crop_offset = x_offset
        y_crop_offset = y_offset
        if x_offset < 0:
            x_crop_offset = 0
        if y_offset < 0:
            y_crop_offset = 0

        cropped_img = tf.crop(
            img,
            y_crop_offset,
            x_crop_offset,
            img.size[1] - abs(y_offset),
            img.size[0] - abs(x_offset),
        )

        if x_offset >= 0 and y_offset >= 0:
            padding_tuple = (0, 0, x_offset, y_offset)

        elif x_offset >= 0 and y_offset < 0:
            padding_tuple = (0, abs(y_offset), x_offset, 0)

        elif x_offset < 0 and y_offset >= 0:
            padding_tuple = (abs(x_offset), 0, 0, y_offset)

        elif x_offset < 0 and y_offset < 0:
            padding_tuple = (abs(x_offset), abs(y_offset), 0, 0)

        return (
            tf.pad(cropped_img, padding_tuple, padding_mode="reflect"),
            tf.affine(
                mask,
                translate=(-x_offset, -y_offset),
                scale=1.0,
                angle=0.0,
                shear=0.0,
                fillcolor=FILL_COLOR,
            ),
        )


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            tf.affine(
                img,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.BILINEAR,
                fillcolor=(0, 0, 0),
                shear=0.0,
            ),
            tf.affine(
                mask,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.NEAREST,
                fillcolor=FILL_COLOR,
                shear=0.0,
            ),
        )


class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size[0]) or (h >= w and h == self.size[1]):
            return img, mask
        if w > h:
            ow = self.size[0]
            oh = int(self.size[1] * h / w)
            return (img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST))
        else:
            oh = self.size[1]
            ow = int(self.size[0] * w / h)
            return (img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST))


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)

                return (
                    img.resize((self.size, self.size), Image.BILINEAR),
                    mask.resize((self.size, self.size), Image.NEAREST),
                )

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomSized(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        #self.scale = Scale(self.size)
        #self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = (img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST))

        # return self.crop(*self.scale(img, mask))
        return (img, mask)
