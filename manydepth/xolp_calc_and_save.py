import os
import numpy as np
import torch
import cv2
import scipy.interpolate
import matplotlib.pyplot as plt
from PIL import Image
import sys


def Iun_and_xolp(images, angles):
    """
    :param images: 4 concatenate images with different polarisation filters
    :param angles: angles of the polarisation filters
    :return: Iun (unpolarised image), rho (DOLP), phi (AOLP)
    """
    I = images.reshape((images.shape[0] * images.shape[1], 4))
    A = np.zeros((4, 3))
    A[:, 0] = 1
    A[:, 1] = np.cos(2 * angles)
    A[:, 2] = np.sin(2 * angles)
    # x = np.linalg.pinv(A) @ images.T
    x = np.linalg.lstsq(A, I.T, rcond=None)
    x = x[0].T
    Imax = x[:, 0] + np.sqrt(x[:, 1] ** 2 + x[:, 2] ** 2)
    Imin = x[:, 0] - np.sqrt(x[:, 1] ** 2 + x[:, 2] ** 2)
    Iun = (Imax + Imin) / 2
    # rho = np.divide(Imax - Imin, Imax + Imin)
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = np.true_divide(Imax - Imin, Imax + Imin)
        rho[rho == np.inf] = 0
        rho = np.nan_to_num(rho)
    phi = 0.5 * np.arctan2(x[:, 2], x[:, 1])
    Iun = np.reshape(Iun, (images.shape[0], images.shape[1]))
    rho = np.reshape(rho, (images.shape[0], images.shape[1]))
    phi = np.reshape(phi, (images.shape[0], images.shape[1]))
    return Iun, rho, phi


def main():
    angles = np.array([0, 45, 90, 135]) * np.pi / 180
    n = 1.5

    sequence = "scene11_traj2_naked_1"  # last used folder
    filename = "000005.png"

    path = r"/media/jungo/Research/Datasets/HAMMER/train/{0}/polarization/".format(sequence)
    path_save = r"/media/jungo/Research/Datasets/HAMMER/xolp_from_each_seq/"
    folder_read_00 = os.path.join(path, "pol00")
    folder_read_10 = os.path.join(path, "pol10")
    folder_read_01 = os.path.join(path, "pol01")
    folder_read_11 = os.path.join(path, "pol11")

    folder_write_dolp = os.path.join(path_save, "dolp")
    if not os.path.exists(folder_write_dolp):
        os.makedirs(folder_write_dolp)

    folder_write_aolp = os.path.join(path_save, "aolp")
    if not os.path.exists(folder_write_aolp):
        os.makedirs(folder_write_aolp)

    im00 = cv2.imread(os.path.join(folder_read_00, filename), cv2.IMREAD_GRAYSCALE)
    im10 = cv2.imread(os.path.join(folder_read_10, filename), cv2.IMREAD_GRAYSCALE)
    im01 = cv2.imread(os.path.join(folder_read_01, filename), cv2.IMREAD_GRAYSCALE)
    im11 = cv2.imread(os.path.join(folder_read_11, filename), cv2.IMREAD_GRAYSCALE)
    # print("grayscale shape: ", im00.shape)  # HxW
    # plt.imshow(im00)
    # plt.show()

    im_stack = np.stack((im00, im01, im10, im11), axis=-1)  # HxWx4
    # print("im_stack shape: ", im_stack.shape)

    Iun, dolp, aolp = Iun_and_xolp(im_stack, angles)

    # filename_npy = filename.split(".")[0] + ".npy"
    filename_npy = sequence + ".npy"
    print(filename_npy)

    np.save(os.path.join(folder_write_dolp, filename_npy), dolp)
    dolp_read = np.load(os.path.join(folder_write_dolp, filename_npy))
    np.save(os.path.join(folder_write_aolp, filename_npy), aolp)
    aolp_read = np.load(os.path.join(folder_write_aolp, filename_npy))

    plt.figure(1)
    plt.subplot(221)
    plt.imshow(dolp)
    plt.subplot(222)
    plt.imshow(dolp_read)
    plt.subplot(223)
    plt.imshow(aolp)
    plt.subplot(224)
    plt.imshow(aolp_read)
    plt.show()

    diff_dolp = (dolp - dolp_read).sum()
    diff_aolp = (aolp - aolp_read).sum()
    if diff_dolp != 0 or diff_aolp != 0 or dolp_read.shape != (832, 1088) or aolp_read.shape != (832, 1088) or dolp_read.sum() == 0 or aolp_read.sum() == 0:
        print("HOUSTON, WE HAVE A PROBLEM")


if __name__ == "__main__":
    main()
