import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch


def split_pol(img):
    """
    Split polarimetric images
    """
    split_h = np.split(img, 2, axis=1)
    split_v0 = np.split(split_h[0], 2, axis=0)
    split_v1 = np.split(split_h[1], 2, axis=0)

    i_0 = split_v0[0]
    i_45 = split_v0[1]
    i_90 = split_v1[0]
    i_135 = split_v1[1]

    return i_0, i_45, i_90, i_135


def concatenate_pol(i_0, i_45, i_90, i_135):
    """
    Concatenate 4 polarized images
    """
    img_cat = torch.cat((i_0, i_45, i_90, i_135), dim=2)

    return img_cat


img_example = mpimg.imread("/home/witek/Documents/Dataset/pol/000000.png")
img_example = torch.from_numpy(img_example)
print(img_example.shape) # [1664, 2176, 3]
i_0, i_45, i_90, i_135 = split_pol(img_example)
print(i_0.shape) # [832, 1088, 3]
i_cat = concatenate_pol(i_0, i_45, i_90, i_135) # [832, 1088, 12]
print(i_cat.shape)

plt.imshow(i_0)
plt.show()