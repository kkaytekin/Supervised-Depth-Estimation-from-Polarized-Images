import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def split_pol(img):
    """
    Split polarimetric images
    """
    split_h = np.split(img, 2, axis=1)
    split_v0 = np.split(split_h[0], 2, axis=0)
    split_v1 = np.split(split_h[1], 2, axis=0)

    im00 = split_v0[0]
    im10 = split_v0[1]
    im01 = split_v1[0]
    im11 = split_v1[1]

    return im00, im10, im01, im11


def stack_pol(im00, im10, im01, im11):
    """
    Stack 4 polarized grayscale images
    """
    # img_cat = np.concatenate((im0, im45, im90, im135), axis=2) # for rgb
    im_stack = np.stack((im00, im10, im01, im11), axis=-1)

    return im_stack


def main():

    sequence = "scene2_traj2_naked_2"
    path = r"/media/jungo/Research/Datasets/HAMMER/train/{0}/polarization/".format(sequence) #path to the polarization folder
    folder_read = os.path.join(path, "pol/")
    folder_write_00 = os.path.join(path, "pol2/pol00/")
    if not os.path.exists(folder_write_00):
        os.makedirs(folder_write_00)

    folder_write_10 = os.path.join(path, "pol2/pol10/")
    if not os.path.exists(folder_write_10):
        os.makedirs(folder_write_10)

    folder_write_01 = os.path.join(path, "pol2/pol01/")
    if not os.path.exists(folder_write_01):
        os.makedirs(folder_write_01)

    folder_write_11 = os.path.join(path, "pol2/pol11/")
    if not os.path.exists(folder_write_11):
        os.makedirs(folder_write_11)

    for filename in os.listdir(folder_read):
        img = cv2.imread(os.path.join(folder_read, filename))
        print(filename)
        im00, im10, im01, im11 = split_pol(img)

        im00 = Image.fromarray(im00)
        im10 = Image.fromarray(im10)
        im01 = Image.fromarray(im01)
        im11 = Image.fromarray(im11)

        # print(os.path.join(folder_write_00, filename))
        im00.save(os.path.join(folder_write_00, filename), format="png")
        im10.save(os.path.join(folder_write_10, filename), format="png")
        im01.save(os.path.join(folder_write_01, filename), format="png")
        im11.save(os.path.join(folder_write_11, filename), format="png")

        # cv2.imwrite(os.path.join(folder_write_00, filename), im00)
        # cv2.imwrite(os.path.join(folder_write_10, filename), im10)
        # cv2.imwrite(os.path.join(folder_write_01, filename), im01)
        # cv2.imwrite(os.path.join(folder_write_11, filename), im11)

if __name__ == "__main__":
    main()