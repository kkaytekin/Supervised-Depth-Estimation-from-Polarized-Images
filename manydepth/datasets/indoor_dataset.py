# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import random
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np
from PIL import Image  # using pillow-simd for increased speed
#import cv2
import glob

import torch
import torch.utils.data as data
from torchvision import transforms
from manydepth.utils import readlines

# cv2.setNumThreads(0)


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class IndoorDataset(data.Dataset):
    """Superclass for monocular dataloaders
    """
    def __init__(self,
                 data_path,
                 secquences,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 offset=30,
                 # modality="l515_rgbd",
                 # modality="d435",
                 modality="polarization",
                 is_train=False,
                 img_ext='.png',
                 supervised_depth=True,
                 supervised_depth_only=True,
                 depth_modality="_gt",
                 ):
        super(IndoorDataset, self).__init__()

        self.img_ext = img_ext

        self.supervised_depth = supervised_depth
        self.supervised_depth_only = supervised_depth_only
        self.depth_modality = depth_modality


        self.input_lookup = "rgb"
        if modality == "d435":
            self.input_lookup = "no_proj_left"

        self.data_path = data_path
        self.secquences = secquences
        self.modality = modality
        self.filenames = self.get_filenames(self.secquences, frame_idxs, offset, modality, depth_modality)
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.frame_offset = offset

        self.interp = Image.ANTIALIAS

        if self.supervised_depth_only:
            self.frame_idxs = [0]
        else:
            self.frame_idxs = frame_idxs

        self.is_train = is_train


        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # self.normalize = transforms.Normalize(mean=mean, std=std)

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = True#self.check_depth()

    def get_filenames(self, secquences, frame_idxs, offset, modality, depth_modality):
        filenames = []

        input_lookup = "rgb"
        if modality == "d435":
            input_lookup = "no_proj_left"
        secquences_filtered = []
        for i in range(len(secquences)):
            folder = os.path.join(self.data_path, secquences[i], modality, input_lookup)
            filenames_in_sec = sorted(glob.glob(folder + '/*.png'))
            new_secquence = []
            old_frame_index = 1
            for k in range(len(filenames_in_sec)):
                line = filenames_in_sec[k].split('/')
                frame_index = int(line[-1].split('.')[0])
                f_str = "{:06d}{}".format(frame_index, self.img_ext)
                file = os.path.join(folder, f_str)
                old_frame_index += 1
                if os.path.isfile(file) and frame_index == old_frame_index:
                    new_secquence.append(filenames_in_sec[k])
                else:
                    secquences_filtered.append(new_secquence)
                    new_secquence = []


                old_frame_index = frame_index

            secquences_filtered.append(new_secquence)

        secquences_filtered = [item for sublist in secquences_filtered for item in sublist]

        # print(secquences_filtered)
        filenames_in_sec_valid = []
        if len(secquences_filtered) > 0:
            filenames_in_sec = (secquences_filtered)
            for k in range(len(filenames_in_sec)):
                valid = True

                line = (filenames_in_sec[k]).split('/')
                folder = '/' + os.path.join(*line[1:-2])
                frame_index = int(line[-1].split('.')[0])
                for id in frame_idxs:
                    if id != "s":
                        f_str = "{:06d}{}".format(frame_index + id * offset, self.img_ext)

                        file = os.path.join(folder, input_lookup, f_str)#filenames_in_sec[k + id]
                        if not os.path.isfile(file):
                            valid = False

                        f_str_pose = "{:06d}{}".format(frame_index + id * offset, ".txt")
                        file_pose = os.path.join(folder, "_pose", f_str_pose)
                        if not os.path.isfile(file_pose):
                            valid = False

                        f_str_gt = "{:06d}{}".format(frame_index + id * offset, ".png")
                        file_gt = os.path.join(folder, "_gt", f_str_gt)
                        if not os.path.isfile(file_gt):
                            valid = False

                        f_str_gt = "{:06d}{}".format(frame_index + id * offset, ".png")
                        file_gt = os.path.join(folder, depth_modality, f_str_gt)
                        if not os.path.isfile(file_gt):
                            valid = False

                if valid:
                    filenames_in_sec_valid.append(filenames_in_sec[k])

            filenames.append(filenames_in_sec_valid)
        filenames_out = [item for sublist in filenames for item in sublist]
        for file in filenames_out:
            if not os.path.isfile(file):
                print("ERROR FOR FILE:", file)
        # print(filenames_out)
        return filenames_out

    def preprocess(self, inputs, do_color_aug, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                # check it isn't a blank frame - keep _aug as zeros so we can check for it
                if inputs[(n, im, i)].sum() == 0:
                    inputs[(n + "_aug", im, i)] = inputs[(n, im, i)]
                else:
                    if do_color_aug:

                        aug = color_aug(f)
                    else:
                        aug = f
                    inputs[(n + "_aug", im, i)] = self.to_tensor(aug)

    def __len__(self):
        return len(self.filenames)

    def load_intrinsics(self, folder):
        K = np.array([[0.58, 0, 0.5, 0],
                           [0, 0.60, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        # k = readlines(os.path.join(folder, "intrinsics.txt")).split()
        with open(os.path.join(folder, "intrinsics.txt"), 'r') as f:
            k = f.read().split()#.splitlines()
        k = np.array(k).reshape(3,3)
        K[:3,:3] = k

        K[0, :] /= self.full_res_shape[0]
        K[1, :] /= self.full_res_shape[1]



        return K.copy()

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "depth_gt"                              for ground truth depth maps

        <frame_id> is:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        try:
        # if 0 == 0:
            inputs = {}

            do_color_aug = self.is_train and random.random() > 0.5
            do_flip = False#self.is_train and random.random() > 0.5

            folder, frame_index = self.index_to_folder_and_frame_idx(index)
            side = "l"

            poses = {}
            if type(self).__name__ in ["CityscapesPreprocessedDataset", "CityscapesEvalDataset"]:
                inputs.update(self.get_colors(folder, frame_index, side, do_flip))
            else:
                for i in self.frame_idxs:
                    if i == "s":
                        other_side = {"r": "l", "l": "r"}[side]
                        inputs[("color", i, -1)] = self.get_color(
                            folder, frame_index, other_side, do_flip, self.input_lookup)


                    else:
                        try:
                            inputs[("color", i, -1)] = self.get_color(
                                folder, frame_index + i * self.frame_offset, None, do_flip, self.input_lookup)

                            if i != 0:
                                if not self.supervised_depth_only:
                                    pose = self.get_relative_pose(folder, frame_index + i * self.frame_offset, frame_index)
                                    # if do_flip:
                                    #     pose = np.linalg.inv(pose)
                                    inputs[("poses", i)] = torch.from_numpy((pose).astype(np.float32))


                        except FileNotFoundError as e:
                            if i != 0:
                                # fill with dummy values
                                inputs[("color", i, -1)] = \
                                    Image.fromarray(np.zeros((100, 100, 3)).astype(np.uint8))
                                poses[i] = None
                            else:
                                raise FileNotFoundError(f'Cannot find frame - make sure your '
                                                        f'--data_path is set correctly, or try adding'
                                                        f' the --png flag. {e}')
            #print("111111111111")
            inputs[("stereo_T")] = torch.from_numpy(np.array([[ 1, 0, 0, -0.0498921],
                                                               [0, 1, 0, 0],
                                                               [0, 0, 1, 0],
                                                               [0, 0, 0, 1]], dtype=np.float32))

            self.full_res_shape = inputs[("color", 0, -1)].size
            #print("22222222222222222222")
            # adjusting intrinsics to match each scale in the pyramid
            for scale in range(self.num_scales):
                K = self.load_intrinsics(folder)

                K[0, :] *= self.width // (2 ** scale)
                K[1, :] *= self.height // (2 ** scale)

                # print(K)

                inv_K = np.linalg.pinv(K)

                inputs[("K", scale)] = torch.from_numpy(K)
                inputs[("inv_K", scale)] = torch.from_numpy(inv_K)
            #print("3333333333333333333333333333333")
            if self.load_depth:# and False:
                #print("3.1")
                if self.supervised_depth:
                #    print("3.1.0")
                    depth = self.get_depth_processed(folder, frame_index, side, do_flip, self.depth_modality)
                #    print("3.1.1")
                    inputs["depth"] = np.expand_dims(depth, 0)
                #    print("3.1.2")
                    inputs["depth"] = torch.from_numpy(inputs["depth"].astype(np.float32))#.clamp(0.01, 2.0)
                #print("3.2")
                # print(index, folder, frame_index)

                depth_gt = self.get_depth_gt(folder, frame_index, side, do_flip)
                #print("3.3")
                inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
                inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))#.clamp(0.01, 2.0)
            #print("4444444444444444444444444")

            # if self.load_mask:
            #     inputs["mask"] = self.get_mask(folder, frame_index, side, do_flip)


            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
            #print("55555555555555555555555")
            self.preprocess(inputs, do_color_aug, color_aug)
            for i in self.frame_idxs:
                del inputs[("color", i, -1)]
                del inputs[("color_aug", i, -1)]
            #print("6666666666666666666666666666666666666")
            return inputs
        except:
            print("ERROR DURING LOADING!!!")
            print(index, folder, frame_index)


    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
