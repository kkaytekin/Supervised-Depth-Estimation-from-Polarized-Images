# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import sys
sys.path.insert(0, '..')

os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402


import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from manydepth.utils import readlines
from manydepth.layers import disp_to_depth
from manydepth import datasets, networks

from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
import argparse

class Trainer:
    def __init__(self):
        self.no_cuda = False
        self.device = torch.device("cpu" if self.no_cuda else "cuda")
        self.data_path_val = "/media/jungo/Research/Datasets/HAMMER/test_unseen/"

        self.parser = argparse.ArgumentParser(description="Pointcloud options")
        self.parser.add_argument("--use_xolp",
                                 dest="use_xolp",
                                 type=bool,
                                 help="",
                                 default=False)
        self.parser.add_argument("--use_normals",
                                 dest="use_normals",
                                 type=bool,
                                 help="",
                                 default=False)
        args = self.parser.parse_args()
        self.use_xolp = False
        self.use_normals = False
        self.run_name = "ABLATIONS_rgb"
        if args.use_xolp:
            self.use_xolp = args.use_xolp
            self.run_name = "ABLATIONS_rgb_xolp"
        elif args.use_normals:
            self.use_normals = args.use_normals
            self.run_name = "ABLATIONS_rgb_normals"
        elif args.use_normals & args.use_xolp:
            self.use_xolp = args.use_xolp
            self.use_normals = args.use_normals
            self.run_name = "ABLATIONS_rgb_xolp_normals"
            #self.run_name = "arch1++_1headatt_after_1x1_07-24_15-37-19"  # same as ABLATIONS_rgb_xolp_normals the attention layer is in joint encoder

        self.mono_weights_folder = "/media/jungo/Research/Experiments/AT3DCV/EXPERIMENTS/{0}/models/weights_47".format(self.run_name)
        self.eval_split = "HAMMER_pointcloud"
        self.path_to_save = "/home/jungo/Pictures/{0}".format(self.run_name)
        self.height = 320
        self.width = 480
        self.weights_init = "pretrained"
        self.dropout_rate = 0.1
        self.scales = [0, 1, 2, 3]
        self.batch_size = 12
        self.num_workers = 8
        self.modality = "polarization"
        self.depth_modality = "_gt"
        self.offset = 10
        self.depth_supervision = True
        self.depth_supervision_only = True
        self.min_depth = 0.1
        self.max_depth = 2.0
        self.learning_rate = 1e-4

        self.camera_matrix = torch.from_numpy(np.array([[7.067553100585937500e+02, 0.000000000000000000e+00, 5.456326819328060083e+02],
                                                        [0.000000000000000000e+00, 7.075133056640625000e+02, 3.899299663507044897e+02],
                                                        [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])).unsqueeze(0).cuda()

        if torch.cuda.is_available():
            print('Use GPU: ' + str(torch.cuda.get_device_name(torch.cuda.current_device())))
            print('With GB: ' + str(
                torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory // 1024 ** 3))

        self.models = {}
        self.parameters_to_train = []

        frames_to_load = [0, -1, 1]

        # Initialize pre-encoders
        self.models["rgb_encoder"] = \
            networks.ShallowResnetEncoder(18, self.weights_init == "pretrained")

        if self.use_normals:
            self.models["normals_encoder"] = networks.ShallowNormalsEncoder(in_channels=9,
                                                                            dropout_rate=self.dropout_rate)
            self.models["normals_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["normals_encoder"].parameters())
        if self.use_xolp:
            self.models["xolp_encoder"] = networks.ShallowEncoder(mode='XOLP',
                                                                  in_channels=2,
                                                                  dropout_rate=self.dropout_rate)
            self.models["xolp_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["xolp_encoder"].parameters())

        self.models["joint_encoder"] = networks.JointEncoder(dropout_rate=self.dropout_rate,
                                                             include_normals=self.use_normals,
                                                             include_xolp=self.use_xolp)
        self.models["rgb_encoder"].to(self.device)
        self.models["joint_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["joint_encoder"].parameters())

        self.models["mono_depth"] = \
            networks.DepthDecoder(self.models["rgb_encoder"].num_ch_enc, self.scales)
        self.models["mono_depth"].to(self.device)

        self.parameters_to_train += list(self.models["rgb_encoder"].parameters())
        self.parameters_to_train += list(self.models["mono_depth"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.learning_rate)

        # DATA
        datasets_dict = {"HAMMER": datasets.HAMMER_Dataset}
        self.dataset = datasets_dict["HAMMER"]
        fpath_test = os.path.join("../splits", self.eval_split, "{}_files.txt")
        test_filenames = readlines(fpath_test.format("test"))
        img_ext = '.png'

        test_dataset = self.dataset(
            self.data_path_val, test_filenames, self.height, self.width,
            frames_to_load, 4, is_train=False, img_ext=img_ext, offset=self.offset, modality=self.modality,
            supervised_depth=self.depth_supervision, supervised_depth_only=self.depth_supervision_only,
            depth_modality=self.depth_modality)
        self.test_loader = DataLoader(
            test_dataset, self.batch_size, False,
            num_workers=self.num_workers, pin_memory=True, drop_last=True)

        print("There are {:d} test items\n".format(len(test_dataset)))

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def test(self):
        """Validate the model on a single minibatch
        """
        print("Testing...")
        self.set_eval()
        with torch.no_grad():
            # gts = []
            # preds_mono = []
            # masks = []
            for batch_idx, inputs in enumerate(self.test_loader):

                print("Processing batch {0}".format(batch_idx))
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(self.device)

                mono_outputs = {}
                outputs = {}

                rgb_feats = self.models["rgb_encoder"](inputs["color_aug", 0, 0].float())
                feats = rgb_feats
                xolp_feats = None
                normals_feats = None
                if self.use_xolp:
                    xolp_feats = self.models["xolp_encoder"](inputs["xolp", 0, 0].float())
                if self.use_normals:
                    normals_feats = self.models["normals_encoder"](inputs["xolp", 0, 0].float())
                enc_feats = self.models["joint_encoder"](rgb_feats[-1], xolp_feats, normals_feats)
                feats.extend(enc_feats)
                mono_outputs.update(self.models['mono_depth'](feats))

                disp = mono_outputs[("disp", 0)]

                disp = F.interpolate(
                    disp, [self.height, self.width], mode="bilinear", align_corners=False)

                _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
                outputs[("mono_depth", 0, 0)] = depth

                depth_pred_mono = depth
                depth_pred_mono = torch.clamp(F.interpolate(
                    depth_pred_mono, [self.height, self.width], mode="bilinear", align_corners=False),
                    self.min_depth, self.max_depth)

                #### Prediction and Pointcloud #####
                images_path = "data/images/"
                rgb_img = inputs["color", 0, 0]
                rgb_img = rgb_img[1, :, :, :].squeeze(0).squeeze(0).cpu().numpy()
                rgb_img = np.transpose(rgb_img, (1, 2, 0))
                rgb_img = Image.fromarray(np.uint8(rgb_img * 255)).convert('RGB')
                rgb_img.save(images_path + "rgb_img.png")

                mask_img = inputs["mask", 0, 0]
                mask_img = mask_img[1, :, :, :].squeeze(0).squeeze(0).cpu().numpy()
                mask_img = Image.fromarray(np.uint8(mask_img * 255)).convert('L')
                mask_img.save(images_path + "mask_img.png")

                depth_pred_img = depth_pred_mono[1, :, :, :].squeeze(0).squeeze(0).cpu().numpy()
                depth_pred_img = depth_pred_img + 0.3  # TODO: adjust depending on the prediction if needed for better projection
                depth_pred_img = depth_pred_img / np.max(depth_pred_img) * 0.02
                depth_pred_img = Image.fromarray(np.uint16(depth_pred_img * 65535))
                depth_pred_img.save(images_path +  "depth_pred_img.png")

                depth_gt_img = inputs["depth_gt"]  # test for depth_gt
                depth_gt_img = depth_gt_img[1, :, :, :].squeeze(0).squeeze(0).cpu().numpy()
                depth_gt_img = depth_gt_img
                depth_gt_img = depth_gt_img / np.max(depth_gt_img) * 0.02
                depth_gt_img = Image.fromarray(np.uint16(depth_gt_img * 65535))
                depth_gt_img.save(images_path + "depth_gt_img.png")


                self.pointcloud(images_path + "rgb_img.png", images_path + "depth_pred_img.png", show=False)
                self.pointcloud(images_path + "rgb_img.png", images_path + "depth_gt_img.png", show=False)
                # TODO: make show=True to see the RGB and depth_pred image and their conversion to RGBD for pointscloud

                del mono_outputs

                break # TODO: erase to see other images in the batchS


    def load_mono_model(self):
        # if self.augment_normals and self.augment_xolp:
        if self.use_normals and self.use_xolp:
            model_list = ['rgb_encoder', 'mono_depth', 'normals_encoder', 'xolp_encoder', 'joint_encoder']
        elif self.use_normals:
            model_list = ['rgb_encoder', 'mono_depth', 'joint_encoder', 'normals_encoder']
        elif self.use_xolp:
            model_list = ['rgb_encoder', 'mono_depth', 'xolp_encoder', 'joint_encoder']
        else:
            model_list = ['rgb_encoder', 'mono_depth', 'joint_encoder']

        assert os.path.isdir(self.mono_weights_folder), \
            "Cannot find folder {}".format(self.mono_weights_folder)
        print("loading model from folder {}".format(self.mono_weights_folder))

        for n in model_list:
            print('loading {}'.format(n))
            path = os.path.join(self.mono_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    def pointcloud(self, color_raw, depth_raw, show=False):
        print("pointcloud")
        print(np.asarray(self.camera_matrix.squeeze(0).cpu()))
        cam_K = np.asarray(self.camera_matrix.squeeze(0).cpu())
        color_raw = o3d.io.read_image(color_raw)
        depth_raw = o3d.io.read_image(depth_raw)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

        if show:
            plt.subplot(1, 4, 1)
            plt.title('color_raw image')
            plt.imshow(color_raw)

            plt.subplot(1, 4, 2)
            plt.title('depth_pred image')
            plt.imshow(depth_raw)

            plt.subplot(1, 4, 3)
            plt.title('rgbd_rgb image')
            plt.imshow(rgbd_image.color)

            plt.subplot(1, 4, 4)
            plt.title('rgbd_depth image')
            plt.imshow(rgbd_image.depth)
            plt.show()

        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=480,
                                                      height=320,
                                                      fx=cam_K[0, 0], fy=cam_K[1, 1],
                                                      cx=cam_K[0, 2], cy=cam_K[1, 2])

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_image, intrinsic=intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # Flip it, otherwise the pointcloud will be upside down
        o3d.visualization.draw_geometries([pcd])

        return pcd
