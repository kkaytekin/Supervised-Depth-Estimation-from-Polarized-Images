import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def split_pol(img):
    """
    Split polarimetric images
    """
    split_h = np.split(img, 2, axis=1)
    print(split_h[0].shape)
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

    path = "/home/witek/Documents/AT3DCV/Datasets/HAMMER_mini/train/scene2_traj1_2/polarization" #path to the polarization folder
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