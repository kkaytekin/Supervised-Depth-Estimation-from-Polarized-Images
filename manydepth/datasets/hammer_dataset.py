# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import skimage.transform
import numpy as np
import PIL.Image as pil
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

from manydepth.kitti_utils import generate_depth_map
from .indoor_dataset import IndoorDataset

cv2.setNumThreads(0)

class HAMMER_Dataset(IndoorDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(HAMMER_Dataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        # self.K = np.array([[0.58, 0, 0.5, 0],
        #                    [0, 0.60, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)

        # self.full_res_shape = (1088, 832)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        """
        line = self.filenames[index].split('/')
        # print(line)
        folder = '/'+os.path.join(*line[1:-2])
        frame_index = int(line[-1].split('.')[0])
        return folder, frame_index

    def get_color(self, folder, frame_index, side, do_flip, input_lookup="rgb"):
        path = self.get_image_path(folder, frame_index, side, input_lookup)
        color = self.loader(path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side=None, input_lookup="rgb"):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        if side is None:

            image_path = os.path.join(
                folder, input_lookup, f_str)
        else:

            image_path = os.path.join(
                folder, "no_proj_right", f_str)
        return image_path


    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_relative_pose(self, folder, frame_index, center_index):
        f_str = "{:06d}.txt".format(center_index)
        pose_path_center = os.path.join(
            folder,
            "_pose",
            f_str)

        f_str2 = "{:06d}.txt".format(frame_index)
        pose_path_side = os.path.join(
            folder,
            "_pose",
            f_str2)


        with open(pose_path_center, 'r') as f:
            T_c = f.read().split()
        with open(pose_path_side, 'r') as f2:
            T_s = f2.read().split()

        T_c = np.array(T_c, dtype='float').reshape(4, 4)
        T_s = np.array(T_s, dtype='float').reshape(4, 4)


        T_c_inv = np.linalg.inv(T_c)
        T = T_c_inv @ T_s

        T = np.linalg.inv(T)

        return T


    def get_depth_processed(self, folder, frame_index, side, do_flip, depth_modality):
        f_str = "{:06d}.png".format(frame_index)
        depth_path = os.path.join(
            folder,
            depth_modality,
            f_str)

        depth_gt = cv2.resize(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED), (self.width, self.height),
                              cv2.INTER_NEAREST)  # pil.open(depth_path)

        # depth_gt = depth_gt.resize([self.width, self.height], pil.NEAREST)
        depth_gt = (np.array(depth_gt).astype(np.uint16) / 1000).astype(np.float32)
        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        return depth_gt

    def get_depth_gt(self, folder, frame_index, side, do_flip):
        f_str = "{:06d}.png".format(frame_index)
        depth_path = os.path.join(
            folder,
            "_gt",
            f_str)

        depth_gt = cv2.resize(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED), (self.width, self.height),
                              cv2.INTER_NEAREST)  # pil.open(depth_path)
        # depth_gt = depth_gt.resize([self.width, self.height], pil.NEAREST)
        depth_gt = (np.array(depth_gt).astype(np.uint16) / 1000).astype(np.float32)

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
