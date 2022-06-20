import numpy as np


def split_pol(img):
    """
    Split polarimetric images
    """
    split_h = np.split(img, 2, axis=1)
    split_v0 = np.split(split_h[0], 2, axis=0)
    split_v1 = np.split(split_h[1], 2, axis=0)

    im0 = split_v0[0]
    im45 = split_v0[1]
    im90 = split_v1[0]
    im135 = split_v1[1]

    return im0, im45, im90, im135


def stack_pol(im0, im45, im90, im135):
    """
    Stack 4 polarized grayscale images
    """
    # img_cat = np.concatenate((im0, im45, im90, im135), axis=2) # for rgb
    im_stack = np.stack((im0, im45, im90, im135), axis=-1)

    return im_stack