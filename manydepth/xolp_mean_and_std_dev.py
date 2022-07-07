import os
import numpy as np
import torch
import cv2
import scipy.interpolate
import matplotlib.pyplot as plt
from PIL import Image
import sys

def main():

    path = r"/media/jungo/Research/Datasets/HAMMER/xolp_from_each_seq/"
    folder_read_dolp = os.path.join(path, "dolp")
    folder_read_aolp = os.path.join(path, "aolp")
    DOLP_FILES = np.zeros((46, 832, 1088))
    AOLP_FILES = np.zeros((46, 832, 1088))

    counter = 0

    for filename in os.listdir(folder_read_dolp):
        DOLP_FILES[counter] = np.load(os.path.join(folder_read_dolp, filename))
        AOLP_FILES[counter] = np.load(os.path.join(folder_read_aolp, filename))
        counter += 1

    print("DOLP MEAN: ", DOLP_FILES.mean(axis=(0, 1, 2)))
    print("DOLP STD: ", DOLP_FILES.std(axis=(0, 1, 2)))
    print("AOLP MEAN: ", AOLP_FILES.mean(axis=(0, 1, 2)))
    print("AOLP STD: ", AOLP_FILES.std(axis=(0, 1, 2)))

    print("XOLP MEAN: ", 0.5 * (DOLP_FILES.mean(axis=(0, 1, 2)) + AOLP_FILES.mean(axis=(0, 1, 2))))
    print("XOLP STD: ", 0.5 * (DOLP_FILES.std(axis=(0, 1, 2)) + AOLP_FILES.std(axis=(0, 1, 2))))


if __name__ == "__main__":
    main()
