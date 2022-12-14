"""
Script used to split 2x2 polarised images and save the split images to files.
"""

import numpy as np
import os
import cv2


def split_pol(img):
    """
    :param img: 2x2 polarised image
    :return:
    """
    print(img.shape)
    split_h = np.split(img, 2, axis=1)
    print(split_h[0].shape)

    split_v0 = np.split(split_h[0], 2, axis=0)
    split_v1 = np.split(split_h[1], 2, axis=0)

    im00 = split_v0[0]
    im10 = split_v0[1]
    im01 = split_v1[0]
    im11 = split_v1[1]

    return im00, im10, im01, im11


def main():

    path = "/media/jungo/Research/Datasets/HAMMER/train/scene2_traj1_2/polarization"  # set a path to the polarizaton folder
    folder_read = os.path.join(path, "pol/")
    folder_write_00 = os.path.join(path, "pol00/")
    folder_write_10 = os.path.join(path, "pol10/")
    folder_write_01 = os.path.join(path, "pol01/")
    folder_write_11 = os.path.join(path, "pol11/")
    for filename in os.listdir(folder_read):
        img = cv2.imread(os.path.join(folder_read, filename))
        print(filename)
        im00, im10, im01, im11 = split_pol(img)
        cv2.imwrite(os.path.join(folder_write_00, filename), im00)
        cv2.imwrite(os.path.join(folder_write_10, filename), im10)
        cv2.imwrite(os.path.join(folder_write_01, filename), im01)
        cv2.imwrite(os.path.join(folder_write_11, filename), im11)

if __name__ == "__main__":
    main()
