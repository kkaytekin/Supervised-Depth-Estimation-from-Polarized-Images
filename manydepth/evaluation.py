# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os

os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import readlines
from layers import disp_to_depth, compute_depth_errors_numpy
from manydepth import datasets, networks

class Evaluation:
    def __init__(self):
        self.no_cuda = False
        self.device = torch.device("cpu" if self.no_cuda else "cuda")
        self.data_path_val = "/media/jungo/Research/Datasets/HAMMER/test_unseen/"
        self.run_name = "ABLATIONS_rgb_xolp_normals"
        self.mono_weights_folder = "/media/jungo/Research/Experiments/AT3DCV/EXPERIMENTS/{0}/models/weights_47".format(self.run_name)
        self.eval_split = "HAMMER_unseen"
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
        self.augment_normals = True
        self.augment_xolp = True

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

        # Initialise the model
        self.models["rgb_encoder"] = \
            networks.ShallowResnetEncoder(18, self.weights_init == "pretrained")
        if self.augment_normals:
            self.models["normals_encoder"] = networks.ShallowNormalsEncoder(in_channels=9,
                                                                            dropout_rate=self.dropout_rate)
            self.models["normals_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["normals_encoder"].parameters())
        if self.augment_xolp:
            self.models["xolp_encoder"] = networks.ShallowEncoder(mode='XOLP',
                                                                  in_channels=2,
                                                                  dropout_rate=self.dropout_rate)
            self.models["xolp_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["xolp_encoder"].parameters())
        self.models["joint_encoder"] = networks.JointEncoder(dropout_rate=self.dropout_rate,
                                                             include_normals=self.augment_normals,
                                                             include_xolp=self.augment_xolp)
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

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("There are {:d} test items\n".format(len(test_dataset)))

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def test(self):
        """Evaluate the model on the test data
        """
        print("Testing...")
        self.set_eval()
        with torch.no_grad():
            gts = []
            preds_mono = []
            masks = []
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
                if self.augment_xolp:
                    xolp_feats = self.models["xolp_encoder"](inputs["xolp", 0, 0].float())
                if self.augment_normals:
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
                depth_pred_mono = depth_pred_mono.detach()
                preds_mono.append(depth_pred_mono.cpu())
                del mono_outputs
                depth_gt = inputs["depth_gt"]
                gts.append(depth_gt.cpu())
                mask = inputs[("mask", 0, 0)]
                masks.append(mask.cpu())

            losses = {}
            print("MONO Depth Test:")
            self.compute_depth_losses_from_list(gts, preds_mono, losses, masks, "all")

            losses = {}
            print("\nMONO DEPTH Test - OBJECTS:")
            self.compute_depth_losses_from_list(gts, preds_mono, losses, masks, "objects")

            losses = {}
            print("\nMONO DEPTH Test - GLASS:")
            self.compute_depth_losses_from_list(gts, preds_mono, losses, masks, "glass")

            losses = {}
            print("\nMONO DEPTH Test - CUTLERY:")
            self.compute_depth_losses_from_list(gts, preds_mono, losses, masks, "cutlery")

            losses = {}
            print("\nMONO DEPTH Test - CAN:")
            self.compute_depth_losses_from_list(gts, preds_mono, losses, masks, "can")

            losses = {}
            print("\nMONO DEPTH Test - BOTTLE:")
            self.compute_depth_losses_from_list(gts, preds_mono, losses, masks, "bottle")

            losses = {}
            print("\nMONO DEPTH Test - CUP:")
            self.compute_depth_losses_from_list(gts, preds_mono, losses, masks, "cup")

            losses = {}
            print("\nMONO DEPTH Test - TEAPOT:")
            self.compute_depth_losses_from_list(gts, preds_mono, losses, masks, "teapot")

            losses = {}
            print("\nMONO DEPTH Test - REMOTE:")
            self.compute_depth_losses_from_list(gts, preds_mono, losses, masks, "remote")

            losses = {}
            print("\nMONO DEPTH Test - BOX:")
            self.compute_depth_losses_from_list(gts, preds_mono, losses, masks, "box")

            losses = {}
            print("\nMONO DEPTH Test - TABLE:")
            self.compute_depth_losses_from_list(gts, preds_mono, losses, masks, "table")

            losses = {}
            print("\nMONO DEPTH Test - WALL:")
            self.compute_depth_losses_from_list(gts, preds_mono, losses, masks, "wall")

    def compute_depth_losses_from_list(self, gts, preds, losses, masks, object="all"):
        """Compute depth metrics
        """
        errors = []
        MIN_DEPTH = self.min_depth
        MAX_DEPTH = self.max_depth
        for k in range(len(preds)):
            depth_pred_batch = preds[k]
            depth_pred_batch = torch.clamp(F.interpolate(
                depth_pred_batch, [self.height, self.width], mode="bilinear", align_corners=False),
                self.min_depth, self.max_depth)

            depth_gt_batch = gts[k]
            masks_batch = masks[k]

            for b in range(self.batch_size):
                depth_pred = depth_pred_batch.detach()[:, 0].numpy()[b]
                depth_gt = depth_gt_batch.detach()[:, 0].numpy()[b]
                mask = np.logical_and(depth_gt > MIN_DEPTH, depth_gt < MAX_DEPTH)  # self.hight x self.width

                if object == "all":
                    depth_pred = depth_pred[mask]  # 1 dim array of non-zero values
                    depth_gt = depth_gt[mask]  # 1 dim array of non-zero values

                else:
                    mask_gt = masks_batch.detach()[:, 0].numpy()[b]  # self.hight x self.width

                    if object == "box":
                        thres1 = thres2 = 20
                    elif object == "bottle":
                        thres1 = thres2 = 40
                    elif object == "can":
                        thres1 = thres2 = 60
                    elif object == "cup":
                        thres1 = thres2 = 80
                    elif object == "remote":
                        thres1 = thres2 = 100
                    elif object == "teapot":
                        thres1 = thres2 = 120
                    elif object == "cutlery":
                        thres1 = thres2 = 140
                    elif object == "glass":
                        thres1 = thres2 = 160
                    elif object == "table":
                        thres1 = thres2 = 180
                    elif object == "wall":
                        thres1 = thres2 = 200
                    elif object == "objects":
                        thres1 = 20
                        thres2 = 160

                    mask_material = np.logical_and(mask_gt >= thres1, mask_gt <= thres2)
                    mask_final = np.logical_and(mask == 1, mask_material == 1)

                    depth_pred = depth_pred[mask_final]  # 1 dim array of non-zero values
                    depth_gt = depth_gt[mask_final]  # 1 dim array of non-zero values

                depth_pred[depth_pred < MIN_DEPTH] = MIN_DEPTH
                depth_pred[depth_pred > MAX_DEPTH] = MAX_DEPTH

                try:
                    depth_errors = compute_depth_errors_numpy(depth_gt, depth_pred)
                except Exception:
                    pass

                errors.append(depth_errors)

        mean_errors = np.array(errors).mean(0)

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.5f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(mean_errors[i])

    def load_mono_model(self):
        if self.augment_normals and self.augment_xolp:
            model_list = ['rgb_encoder', 'mono_depth', 'normals_encoder', 'xolp_encoder', 'joint_encoder']
        elif self.augment_normals:
            model_list = ['rgb_encoder', 'mono_depth', 'joint_encoder', 'normals_encoder']
        elif self.augment_xolp:
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