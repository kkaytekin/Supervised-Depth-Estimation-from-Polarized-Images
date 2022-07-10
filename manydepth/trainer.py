# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os

os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import logging
from datetime import datetime

import numpy as np
import time
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from tensorboard import SummaryWriter

from torch.utils.tensorboard import SummaryWriter

import json

from .utils import readlines, sec_to_hm_str
from .layers import SSIM, BackprojectDepth, Project3D, transformation_from_parameters, \
    disp_to_depth, get_smooth_loss, compute_depth_errors, BerHuLoss, compute_depth_errors_numpy

from manydepth import datasets, networks, dpt
import matplotlib.pyplot as plt

from kornia.geometry.depth import depth_to_normals
import roma

_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting


def make_iterative_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_iterative_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type for tensor2float")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Trainer:
    def __init__(self, options):
        print(options)
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.data_path = self.opt.data_path
        self.data_path_val = self.opt.data_path_val
        self.log_dir = self.opt.log_dir

        timestamp = datetime.now()
        self.log_path = os.path.join(self.opt.log_dir,
                                     self.opt.model_name + '_' + timestamp.strftime("%m-%d_%H-%M-%S"))
        self.log_args(timestamp)

        print('Start training ...')
        if torch.cuda.is_available():
            print('Use GPU: ' + str(torch.cuda.get_device_name(torch.cuda.current_device())))
            print('With GB: ' + str(
                torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory // 1024 ** 3))

        print('Use batch size: ', self.opt.batch_size)

        print('Use training data from: ' + str(self.data_path))
        print('Use val data from: ' + str(self.data_path_val))
        print('Outputs will be saved to: ' + str(self.log_path))

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Training is using:\n  ", self.device)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        self.train_teacher_and_pose = not self.opt.freeze_teacher_and_pose
        if self.train_teacher_and_pose:
            print('using adaptive depth binning!')
            self.min_depth_tracker = self.opt.min_depth
            self.max_depth_tracker = self.opt.max_depth
        else:
            print('fixing pose network and monocular network!')

        if self.opt.train_stereo_only or self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()

        if not self.opt.train_stereo_only:
            self.matching_ids = [0]
            if self.opt.use_future_frame:
                self.matching_ids.append(1)
            for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
                self.matching_ids.append(idx)
                if idx not in frames_to_load:
                    frames_to_load.append(idx)

        print('Loading frames: {}'.format(frames_to_load))

        # MODEL SETUP
        if self.opt.train_stereo_only:
            self.models["gwc_net"] = networks.GwcNet_GC(self.opt.max_disparity)
            self.models["gwc_net"].to(self.device)
            self.parameters_to_train += list(self.models["gwc_net"].parameters())

        elif self.opt.train_dpt:
            if self.opt.midas:
                model = dpt.MidasNet_large(None, non_negative=True)
            else:
                net_w = self.opt.width
                net_h = self.opt.height
                # model = dpt.DPTDepthModel(
                #     path=None,
                #     backbone="vitl16_384",
                #     non_negative=True,
                #     enable_attention_hooks=False,
                #     invert=False
                # )

                model = dpt.DPTDepthModel(
                    path=None,
                    backbone="vitb_rn50_384",
                    non_negative=True,
                    enable_attention_hooks=False,
                    invert=False
                )

            self.models["dpt"] = model
            self.models["dpt"].to(self.device)
            self.parameters_to_train += list(self.models["dpt"].parameters())
        else:
            if not self.opt.depth_supervision_only:
                if self.opt.train_student:
                    self.models["encoder"] = networks.ResnetEncoderMatching(
                        self.opt.num_layers, self.opt.weights_init == "pretrained",
                        input_height=self.opt.height, input_width=self.opt.width,
                        # batch_size=self.opt.batch_size, scale=2, in_channels=64, out_channels=64,
                        adaptive_bins=True, min_depth_bin=self.opt.min_depth, max_depth_bin=self.opt.max_depth,
                        depth_binning=self.opt.depth_binning, num_depth_bins=self.opt.num_depth_bins,
                        batch_size=self.opt.batch_size)
                    self.models["encoder"].to(self.device)

                    if self.opt.augment_normals and self.opt.augment_xolp:
                        self.models["normals_encoder"] = networks.NormalsEncoder(dropout_rate=.1)
                        self.models["normals_encoder"].to(self.device)
                        self.parameters_to_train += list(self.models["normals_encoder"].parameters())

                        self.models["xolp_encoder"] = networks.XOLPEncoder(dropout_rate=.1)
                        self.models["xolp_encoder"].to(self.device)
                        self.parameters_to_train += list(self.models["xolp_encoder"].parameters())

                        self.models["depth"] = networks.DepthDecoder(
                            self.models["encoder"].num_ch_enc, self.opt.scales, augment_normals=True, augment_xolp=True)
                    elif self.opt.augment_normals:
                        self.models["normals_encoder"] = networks.NormalsEncoder(dropout_rate=.1)
                        self.models["normals_encoder"].to(self.device)
                        self.parameters_to_train += list(self.models["normals_encoder"].parameters())

                        self.models["depth"] = networks.DepthDecoder(
                            self.models["encoder"].num_ch_enc, self.opt.scales, augment_normals=True)
                    elif self.opt.augment_xolp:
                        self.models["xolp_encoder"] = networks.XOLPEncoder(dropout_rate=.1)
                        self.models["xolp_encoder"].to(self.device)
                        self.parameters_to_train += list(self.models["xolp_encoder"].parameters())

                        self.models["depth"] = networks.DepthDecoder(
                            self.models["encoder"].num_ch_enc, self.opt.scales, augment_xolp=True)
                    else:
                        self.models["depth"] = networks.DepthDecoder(
                            self.models["encoder"].num_ch_enc, self.opt.scales)
                    self.models["depth"].to(self.device)

                    self.parameters_to_train += list(self.models["encoder"].parameters())
                    self.parameters_to_train += list(self.models["depth"].parameters())

            self.models["mono_encoder"] = \
                networks.ResnetEncoder(18, self.opt.weights_init == "pretrained")
            self.models["mono_encoder"].to(self.device)

            if self.opt.augment_normals and self.opt.augment_xolp:
                self.models["normals_encoder"] = networks.NormalsEncoder(dropout_rate=.1)
                self.models["normals_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["normals_encoder"].parameters())

                self.models["xolp_encoder"] = networks.XOLPEncoder(dropout_rate=.1)
                self.models["xolp_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["xolp_encoder"].parameters())

                self.models["mono_depth"] = \
                    networks.DepthDecoder(self.models["mono_encoder"].num_ch_enc, self.opt.scales, augment_normals=True,
                                          augment_xolp=True)
                self.models["mono_depth"].to(self.device)
            elif self.opt.augment_normals:
                self.models["normals_encoder"] = networks.NormalsEncoder(dropout_rate=.1)
                self.models["normals_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["normals_encoder"].parameters())

                self.models["mono_depth"] = \
                    networks.DepthDecoder(self.models["mono_encoder"].num_ch_enc, self.opt.scales, augment_normals=True)
                self.models["mono_depth"].to(self.device)
            elif self.opt.augment_xolp:
                self.models["xolp_encoder"] = networks.XOLPEncoder(dropout_rate=.1)
                self.models["xolp_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["xolp_encoder"].parameters())

                self.models["mono_depth"] = \
                    networks.DepthDecoder(self.models["mono_encoder"].num_ch_enc, self.opt.scales, augment_xolp=True)
                self.models["mono_depth"].to(self.device)
            else:
                self.models["mono_depth"] = \
                    networks.DepthDecoder(self.models["mono_encoder"].num_ch_enc, self.opt.scales)
                self.models["mono_depth"].to(self.device)

            if self.train_teacher_and_pose:
                self.parameters_to_train += list(self.models["mono_encoder"].parameters())
                self.parameters_to_train += list(self.models["mono_depth"].parameters())

            if not self.opt.depth_supervision_only and not self.opt.pose_input:
                self.models["pose_encoder"] = \
                    networks.ResnetEncoder(18, self.opt.weights_init == "pretrained",
                                           num_input_images=self.num_pose_frames)
                self.models["pose_encoder"].to(self.device)

                self.models["pose"] = \
                    networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc,
                                         num_input_features=1,
                                         num_frames_to_predict_for=2)
                self.models["pose"].to(self.device)

                if self.train_teacher_and_pose:
                    self.parameters_to_train += list(self.models["pose_encoder"].parameters())
                    self.parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.mono_weights_folder is not None:
            self.load_mono_model()

        # self.berhuloss = BerHuLoss()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # DATA
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "HAMMER": datasets.HAMMER_Dataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join("splits", self.opt.split, "{}_files.txt")

        fpath_test = os.path.join("splits", self.opt.eval_split, "{}_files.txt")

        if self.opt.overfit:
            print("OVERFITTING")
            train_filenames = [self.opt.overfit_scene]
            val_filenames = [self.opt.overfit_scene]
        else:
            train_filenames = readlines(fpath.format("train"))
            val_filenames = readlines(fpath.format("val"))

        test_filenames = readlines(fpath_test.format("test"))
        img_ext = '.png'  # if self.opt.png else '.jpg'

        train_dataset = self.dataset(
            self.data_path, train_filenames, self.opt.height, self.opt.width,
            frames_to_load, 4, is_train=True, img_ext=img_ext, offset=self.opt.offset, modality=self.opt.modality,
            supervised_depth=self.opt.depth_supervision, supervised_depth_only=self.opt.depth_supervision_only,
            depth_modality=self.opt.depth_modality)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,
            worker_init_fn=seed_worker)
        val_dataset = self.dataset(
            self.data_path, val_filenames, self.opt.height, self.opt.width,
            frames_to_load, 4, is_train=False, img_ext=img_ext, offset=self.opt.offset, modality=self.opt.modality,
            supervised_depth=self.opt.depth_supervision, supervised_depth_only=self.opt.depth_supervision_only,
            depth_modality=self.opt.depth_modality)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers,
            pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        test_dataset = self.dataset(
            self.data_path_val, test_filenames, self.opt.height, self.opt.width,
            frames_to_load, 4, is_train=False, img_ext=img_ext, offset=self.opt.offset, modality=self.opt.modality,
            supervised_depth=self.opt.depth_supervision, supervised_depth_only=self.opt.depth_supervision_only,
            depth_modality=self.opt.depth_modality)
        self.test_loader = DataLoader(
            test_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        self.writers = {}
        for mode in ["train", "val", "val_mono", "test", "test_mono", "test_mono_glass"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        if not self.opt.train_stereo_only and not self.opt.depth_supervision_only:

            self.backproject_depth = {}
            self.project_3d = {}

            for scale in self.opt.scales:
                h = self.opt.height // (2 ** scale)
                w = self.opt.width // (2 ** scale)

                self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
                self.backproject_depth[scale].to(self.device)

                self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
                self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        self.depth_metric_names_mono = [
            "de_mono/abs_rel", "de_mono/sq_rel", "de_mono/rms", "de_mono/log_rms", "da_mono/a1", "da_mono/a2",
            "da_mono/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items and {:d} test items\n".format(
            len(train_dataset), len(val_dataset), len(test_dataset)))

        self.save_opts()

    def log_args(self, timestamp):
        # Dump the passed arguments into a log file
        os.makedirs(self.log_path, exist_ok=True)
        logging.basicConfig(filename=os.path.join(self.log_path, 'args.log'),
                            level=logging.INFO,
                            format='%(message)s',
                            )
        logging.info('Run started at: %s', str(timestamp))
        logging.info('=' * 8 + 'Arguments' + '=' * 9)
        for arg, value in sorted(vars(self.opt).items()):
            logging.info("%s: %r", arg, value)

    def set_train(self):
        """Convert all models to training mode
        """

        for k, m in self.models.items():
            if self.train_teacher_and_pose:
                m.train()
            else:
                # if teacher + pose is frozen, then only use training batch norm stats for
                # multi components
                if k in ['depth', 'encoder']:
                    m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        # self.test_stereo()
        # if self.opt.train_stereo_only:
        #     self.test_stereo()
        # else:
        #     self.test()
        self.test()
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):

            if self.epoch == self.opt.freeze_teacher_epoch:
                self.freeze_teacher()

            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
                if self.opt.train_stereo_only:
                    self.test_stereo()
                else:
                    self.test()

    def freeze_teacher(self):
        if self.train_teacher_and_pose:
            self.train_teacher_and_pose = False
            print('freezing teacher and pose networks!')

            # here we reinitialise our optimizer to ensure there are no updates to the
            # teacher and pose networks
            self.parameters_to_train = []
            self.parameters_to_train += list(self.models["encoder"].parameters())
            self.parameters_to_train += list(self.models["depth"].parameters())
            self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
            self.model_lr_scheduler = optim.lr_scheduler.StepLR(
                self.model_optimizer, self.opt.scheduler_step_size, 0.1)

            # set eval so that teacher + pose batch norm is running average
            self.set_eval()
            # set train so that multi batch norm is in train mode
            self.set_train()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            self.model_optimizer.zero_grad()

            if self.opt.train_stereo_only:
                outputs, losses = self.process_batch_stereo(inputs, is_train=True)
            else:
                outputs, losses, mono_outputs = self.process_batch(inputs, is_train=True)

            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                if self.opt.train_stereo_only:
                    self.val_stereo()
                else:
                    self.val()
                self.save_model()

            if self.epoch == self.opt.freeze_teacher_epoch:
                self.freeze_teacher()

            self.step += 1
        self.model_lr_scheduler.step()

    def process_batch(self, inputs, is_train=False):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        mono_outputs = {}
        outputs = {}
        losses = {}

        if not self.opt.depth_supervision_only:

            if not self.opt.pose_input:
                # predict poses for all frames
                if self.train_teacher_and_pose:
                    pose_pred = self.predict_poses(inputs, None)
                else:
                    with torch.no_grad():
                        pose_pred = self.predict_poses(inputs, None)
                outputs.update(pose_pred)
                mono_outputs.update(pose_pred)

            lookup_frames = [inputs[('color_aug', idx, 0)] for idx in self.matching_ids[1:]]
            lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

            min_depth_bin = self.min_depth_tracker
            max_depth_bin = self.max_depth_tracker

        # single frame path
        if self.train_teacher_and_pose:
            if self.opt.train_dpt:
                depth_dpt = self.models["dpt"](inputs["color_aug", 0, 0])
                mono_outputs[("depth", 0, 0)] = depth_dpt.unsqueeze(1)
            else:
                rgb_feats = self.models["mono_encoder"](inputs["color_aug", 0, 0].float())
                feats = rgb_feats
                if self.opt.augment_xolp:
                    xolp_feats = self.models["xolp_encoder"](inputs["xolp", 0, 0].float())
                    feats[-1] = torch.cat((feats[-1], xolp_feats), 1)
                if self.opt.augment_normals:
                    normal_feats = self.models["normals_encoder"](inputs["xolp", 0, 0].float())
                    feats[-1] = torch.cat((feats[-1], normal_feats), 1)

                mono_outputs.update(self.models['mono_depth'](feats))

        else:
            with torch.no_grad():
                feats = self.models["mono_encoder"](inputs["color_aug", 0, 0])
                mono_outputs.update(self.models['mono_depth'](feats))

        if not self.opt.depth_supervision_only:
            self.generate_images_pred(inputs, mono_outputs)
        else:
            # TODO
            for scale in self.opt.scales:
                if not self.opt.train_dpt:
                    disp = mono_outputs[("disp", scale)]
                    mono_outputs[("disp", 0, scale)] = disp

                    disp = F.interpolate(
                        disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    source_scale = 0

                    _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                    mono_outputs[("depth", 0, scale)] = depth
                    mono_outputs[("mono_depth", 0, scale)] = depth
                outputs = mono_outputs

        if not self.opt.depth_supervision_only:
            if self.opt.res_pose:
                pose_pred_res = self.predict_poses(inputs, None, mono_outputs, res=True)
                mono_outputs.update(pose_pred_res)

                for scale in self.opt.scales:
                    for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                        cam_points = self.backproject_depth[0](
                            mono_outputs[("depth", 0, scale)], inputs[("inv_K", 0)])
                        pix_coords = self.project_3d[0](
                            cam_points, inputs[("K", 0)], mono_outputs[("cam_T_cam_res", 0, frame_id)])

                        mono_outputs[("color_res", frame_id, scale)] = F.grid_sample(
                            mono_outputs[("color", frame_id, 0)],
                            pix_coords,
                            padding_mode="border", align_corners=True)

                        outputs[("color_res", frame_id, scale)] = mono_outputs[("color_res", frame_id, scale)]

        # if not self.opt.depth_supervision_only:
        mono_losses = self.compute_losses(inputs, mono_outputs, is_multi=False)

        if not self.opt.depth_supervision_only:
            if self.opt.train_student:
                # update multi frame outputs dictionary with single frame outputs
                for key in list(mono_outputs.keys()):
                    _key = list(key)
                    if _key[0] in ['depth', 'disp']:
                        _key[0] = 'mono_' + key[0]
                        _key = tuple(_key)
                        outputs[_key] = mono_outputs[key]

                # grab poses + frames and stack for input to the multi frame network
                # relative_poses = [mono_outputs[('cam_T_cam', 0, idx)] for idx in self.matching_ids[1:]]
                if not self.opt.pose_input:
                    relative_poses = [inputs[('relative_pose', idx)] for idx in self.matching_ids[1:]]
                else:
                    relative_poses = [inputs[('poses', idx)] for idx in self.matching_ids[1:]]
                relative_poses = torch.stack(relative_poses, 1)

                # apply static frame and zero cost volume augmentation
                batch_size = len(lookup_frames)
                augmentation_mask = torch.zeros([batch_size, 1, 1, 1]).to(self.device).float()
                if is_train and not self.opt.no_matching_augmentation:
                    for batch_idx in range(batch_size):
                        rand_num = random.random()
                        # static camera augmentation -> overwrite lookup frames with current frame
                        if rand_num < 0.25:
                            replace_frames = \
                                [inputs[('color', 0, 0)][batch_idx] for _ in self.matching_ids[1:]]
                            replace_frames = torch.stack(replace_frames, 0)
                            lookup_frames[batch_idx] = replace_frames
                            augmentation_mask[batch_idx] += 1
                        # missing cost volume augmentation -> set all poses to 0, the cost volume will
                        # skip these frames
                        elif rand_num < 0.5:
                            relative_poses[batch_idx] *= 0
                            augmentation_mask[batch_idx] += 1
                outputs['augmentation_mask'] = augmentation_mask

                # multi frame path
                features, lowest_cost, confidence_mask = self.models["encoder"](inputs["color_aug", 0, 0],
                                                                                lookup_frames,
                                                                                relative_poses,
                                                                                inputs[('K', 2)],
                                                                                inputs[('inv_K', 2)],
                                                                                min_depth_bin=min_depth_bin,
                                                                                max_depth_bin=max_depth_bin)
                outputs.update(self.models["depth"](features))

                outputs["lowest_cost"] = F.interpolate(lowest_cost.unsqueeze(1),
                                                       [self.opt.height, self.opt.width],
                                                       mode="nearest")[:, 0]

                outputs["consistency_mask"] = F.interpolate(confidence_mask.unsqueeze(1),
                                                            [self.opt.height, self.opt.width],
                                                            mode="nearest")[:, 0]

                if not self.opt.disable_motion_masking:
                    outputs["consistency_mask"] = (outputs["consistency_mask"] *
                                                   self.compute_matching_mask(outputs))

                self.generate_images_pred(inputs, outputs, is_multi=True)

                losses = self.compute_losses(inputs, outputs, is_multi=True)

                # update losses with single frame losses
                if self.train_teacher_and_pose:
                    for key, val in mono_losses.items():
                        losses[key] += val
            else:
                losses = mono_losses
                outputs = mono_outputs
        else:
            losses = mono_losses
        if not self.opt.depth_supervision_only:
            if self.opt.train_student:
                # update adaptive depth bins
                if self.train_teacher_and_pose:
                    self.update_adaptive_depth_bins(outputs)

        return outputs, losses, mono_outputs

    def update_adaptive_depth_bins(self, outputs):
        """Update the current estimates of min/max depth using exponential weighted average"""

        min_depth = outputs[('mono_depth', 0, 0)].detach().min(-1)[0].min(-1)[0]
        max_depth = outputs[('mono_depth', 0, 0)].detach().max(-1)[0].max(-1)[0]

        min_depth = min_depth.mean().cpu().item()
        max_depth = max_depth.mean().cpu().item()

        # increase range slightly
        min_depth = max(self.opt.min_depth, min_depth * 0.9)
        max_depth = max_depth * 1.1

        self.max_depth_tracker = self.max_depth_tracker * 0.99 + max_depth * 0.01
        self.min_depth_tracker = self.min_depth_tracker * 0.99 + min_depth * 0.01

        # self.opt.min_depth = self.min_depth_tracker
        # self.opt.max_depth = self.max_depth_tracker

    def predict_poses(self, inputs, features=None, outputs_past=None, res=False):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}

        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # predict poses for reprojection loss
            # select what features the pose network takes as input
            if res:
                pose_feats = {f_i: outputs_past["color_aug_warped", f_i, 0] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                    axisangle, translation = self.models["pose"](pose_inputs)
                    if not res:
                        outputs[("axisangle", 0, f_i)] = axisangle
                        outputs[("translation", 0, f_i)] = translation

                        # Invert the matrix if the frame id is negative
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

                        outputs[("cam_T_cam_inv", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i > 0))
                    else:
                        outputs[("cam_T_cam_res", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
            if not res:
                # now we need poses for matching - compute without gradients
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.matching_ids}
                with torch.no_grad():
                    # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                    for fi in self.matching_ids[1:]:
                        if fi < 0:
                            pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                            axisangle, translation = self.models["pose"](pose_inputs)
                            pose = transformation_from_parameters(
                                axisangle[:, 0], translation[:, 0], invert=True)
                            pose_inv = transformation_from_parameters(
                                axisangle[:, 0], translation[:, 0], invert=False)

                            # now find 0->fi pose
                            if fi != -1:
                                pose = torch.matmul(pose, inputs[('relative_pose', fi + 1)])

                        else:
                            pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                            axisangle, translation = self.models["pose"](pose_inputs)
                            pose = transformation_from_parameters(
                                axisangle[:, 0], translation[:, 0], invert=False)
                            pose_inv = transformation_from_parameters(
                                axisangle[:, 0], translation[:, 0], invert=True)

                            # now find 0->fi pose
                            if fi != 1:
                                pose = torch.matmul(pose, inputs[('relative_pose', fi - 1)])

                        # set missing images to 0 pose
                        for batch_idx, feat in enumerate(pose_feats[fi]):
                            if feat.sum() == 0:
                                pose[batch_idx] *= 0

                        inputs[('relative_pose', fi)] = pose
                        inputs[('relative_pose_inv', fi)] = pose_inv
        else:
            raise NotImplementedError

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses, mono_outputs = self.process_batch(inputs)

            if not self.opt.depth_supervision_only:
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("val", inputs, outputs, losses)

                del losses

            losses = {}
            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses, mono=True)

            self.log("val_mono", inputs, outputs, losses, log_images=False, log_essential_images=True)
            del inputs, mono_outputs, losses

        self.set_train()

    def test(self):
        """Validate the model on a single minibatch
        """
        print("Running full test set at Epoch: ", self.epoch)
        self.set_eval()
        losses = {}
        with torch.no_grad():
            gts = []
            preds = []
            preds_mono = []
            masks = []
            # print(self.test_loader.__len__)
            for batch_idx, inputs in enumerate(self.test_loader):
                # print(batch_idx)
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(self.device)

                mono_outputs = {}
                outputs = {}
                # mono_outputs_pre = {}

                if self.opt.train_dpt:
                    depth_dpt = self.models["dpt"](inputs["color_aug", 0, 0])
                    # print(depth_dpt.min())
                    # print(depth_dpt.max())
                    # print(depth_dpt.mean())
                    mono_outputs[("depth", 0, 0)] = depth_dpt.unsqueeze(1)
                else:

                    rgb_feats = self.models["mono_encoder"](inputs["color_aug", 0, 0].float())
                    feats = rgb_feats
                    if self.opt.augment_xolp:
                        xolp_feats = self.models["xolp_encoder"](inputs["xolp", 0, 0].float())
                        feats[-1] = torch.cat((feats[-1], xolp_feats), 1)
                    if self.opt.augment_normals:
                        normal_feats = self.models["normals_encoder"](inputs["xolp", 0, 0].float())
                        feats[-1] = torch.cat((feats[-1], normal_feats), 1)
                    mono_outputs.update(self.models['mono_depth'](feats))

                if not self.opt.depth_supervision_only:
                    if self.opt.train_student:
                        # feats_pre = self.models["mono_encoder"](inputs["color", -1, 0])
                        # mono_outputs_pre = self.models['mono_depth'](feats_pre)
                        if not self.opt.pose_input:
                            pose_pred = self.predict_poses(inputs, None)
                            outputs.update(pose_pred)
                            mono_outputs.update(pose_pred)

                            # if not self.opt.pose_input:
                            relative_poses = [inputs[('relative_pose', idx)] for idx in self.matching_ids[1:]]
                            relative_poses = torch.stack(relative_poses, 1)

                        else:

                            relative_poses = [inputs[('poses', idx)] for idx in self.matching_ids[1:]]
                            relative_poses = torch.stack(relative_poses, 1)

                        lookup_frames = [inputs[('color', idx, 0)] for idx in self.matching_ids[1:]]
                        lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

                if self.opt.train_dpt:
                    depth = mono_outputs[("depth", 0, 0)]
                else:
                    disp = mono_outputs[("disp", 0)]
                    # print(disp.shape)

                    disp = F.interpolate(
                        disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    source_scale = 0

                    _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                    outputs[("mono_depth", 0, 0)] = depth

                depth_pred_mono = depth
                depth_pred_mono = torch.clamp(F.interpolate(
                    depth_pred_mono, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False),
                    self.opt.min_depth, self.opt.max_depth)
                depth_pred_mono = depth_pred_mono.detach()
                preds_mono.append(depth_pred_mono.cpu())

                # for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                #     T = outputs[("cam_T_cam", 0, frame_id)]
                #
                #     cam_points = self.backproject_depth[source_scale](
                #         depth, inputs[("inv_K", source_scale)])
                #     pix_coords = self.project_3d[source_scale](
                #         cam_points, inputs[("K", source_scale)], T)
                #
                #     mono_outputs[("sample", frame_id, 0)] = pix_coords
                #
                # h = 48
                # w = 160
                del mono_outputs

                if not self.opt.depth_supervision_only:
                    if self.opt.train_student:
                        features = self.models["encoder"](
                            inputs["color", 0, 0],
                            lookup_frames,
                            relative_poses,
                            inputs[('K', 2)],
                            inputs[('inv_K', 2)],
                            min_depth_bin=self.min_depth_tracker,
                            max_depth_bin=self.max_depth_tracker)

                        features = features[0]

                        outputs.update(self.models["depth"](
                            features))

                        depth_pred = disp_to_depth(outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)[1]
                        depth_pred = torch.clamp(F.interpolate(
                            depth_pred, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False),
                            self.opt.min_depth, self.opt.max_depth)
                        depth_pred = depth_pred.detach()
                        preds.append(depth_pred.cpu())

                depth_gt = inputs["depth_gt"]
                gts.append(depth_gt.cpu())
                mask = inputs[("mask", 0, 0)]
                masks.append(mask.cpu())

            if not self.opt.depth_supervision_only:
                if self.opt.train_student:
                    print("Depth Test:")
                    self.compute_depth_losses_from_list(gts, preds, losses)

                    self.log("test", inputs, outputs, losses, log_images=False)

            del losses
            # print("MASKS LEN: ", len(masks))  # 10
            # print("MASKS SHAPE: ", masks[0].shape)  # 12x1x320x480

            losses = {}
            print("MONO Depth Test:")
            self.compute_depth_losses_from_list(gts, preds_mono, losses, masks, "all")
            self.log("test_mono", inputs, outputs, losses, log_images=False, log_essential_images=True, mono_depth=True)

            losses = {}
            print("\nMONO DEPTH Test - GLASS:")
            self.compute_depth_losses_from_list(gts, preds_mono, losses, masks, "glass")
            self.log("test_mono_glass", inputs, outputs, losses, log_images=False, log_essential_images=False, mono_depth=False)

    def generate_images_pred(self, inputs, outputs, is_multi=False):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            # if not is_multi and self.opt.post_process_mono_while_training:
            #     disp = outputs[("disp_pp", scale)]
            # else:
            if not self.opt.train_dpt:
                disp = outputs[("disp", scale)]

                if self.opt.v1_multiscale:
                    source_scale = scale
                else:
                    disp = F.interpolate(
                        disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    source_scale = 0

                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                outputs[("depth", 0, scale)] = depth
            else:
                depth = outputs[("depth", 0, scale)]

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                    T_GT = inputs[("stereo_T")]
                else:
                    T_GT = inputs[("poses", frame_id)]
                    if not self.opt.pose_input:
                        T = outputs[("cam_T_cam", 0, frame_id)]
                    else:
                        T = T_GT
                if is_multi:  # and self.epoch < self.opt.pose_attach_epoch:
                    # don't update posenet based on multi frame prediction
                    T = T.detach()

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                if is_multi:
                    with torch.no_grad():
                        pix_coords_with_gt_pose = self.project_3d[source_scale](
                            cam_points, inputs[("K", source_scale)], T_GT)

                        cam_points_with_gt_depth = self.backproject_depth[source_scale](
                            inputs[("depth_gt")], inputs[("inv_K", source_scale)])
                        pix_coords_gt_depth = self.project_3d[source_scale](
                            cam_points_with_gt_depth, inputs[("K", source_scale)], T)

                        pix_coords_gt_depth_gt_pose = self.project_3d[source_scale](
                            cam_points_with_gt_depth, inputs[("K", source_scale)], T_GT)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if is_multi:
                    with torch.no_grad():
                        outputs[("color_with_gt_pose", frame_id, scale)] = F.grid_sample(
                            inputs[("color", frame_id, source_scale)],
                            pix_coords_with_gt_pose,
                            padding_mode="zeros", align_corners=True)

                        outputs[("color_with_gt_depth_gt_pose", frame_id, scale)] = F.grid_sample(
                            inputs[("color", frame_id, source_scale)],
                            pix_coords_gt_depth_gt_pose,
                            padding_mode="zeros", align_corners=True)

                if not is_multi and self.opt.res_pose:
                    outputs[("color_aug_warped", frame_id, scale)] = F.grid_sample(
                        inputs[("color_aug", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border", align_corners=True)
                    outputs[("color_aug_warped", 0, scale)] = inputs[("color_aug", 0, source_scale)]

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

        else:
            # we are using automasking
            all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            reprojection_loss_mask = (idxs == 0).float()

        return reprojection_loss_mask

    @staticmethod
    def compute_motion_masks(teacher_depth, student_depth):
        """
        Generate a mask of where we cannot trust the main pathway of the network, based on the difference
        between the main pathway and the reference monodepth2
        """

        # mask where they differ by a large amount
        mask = ((student_depth - teacher_depth) / teacher_depth) < 1.0
        mask *= ((teacher_depth - student_depth) / student_depth) < 1.0
        return mask

    def compute_matching_mask(self, outputs):
        """Generate a mask of where we cannot trust the cost volume, based on the difference
        between the cost volume and the teacher, monocular network"""

        mono_output = outputs[('mono_depth', 0, 0)].detach()
        matching_depth = 1 / outputs['lowest_cost'].unsqueeze(1).to(self.device)
        # matching_depth = disp_to_depth(outputs[("disp", 0)].detach(), self.opt.min_depth, self.opt.max_depth)[1]
        # matching_depth = 1.0 / (outputs[("disp", 0)].detach())

        # mask where they differ by a large amount
        mask = ((matching_depth - mono_output) / mono_output) < 1.0
        mask *= ((mono_output - matching_depth) / matching_depth) < 1.0
        return mask[:, 0]

    def compute_losses(self, inputs, outputs, is_multi=False):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0
        sample_loss = 0
        consistency_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            reprojection_res_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            if self.opt.train_dpt:
                disp = torch.nan_to_num(
                    1. / (outputs[("depth", 0, scale)].clamp(self.opt.min_depth, self.opt.max_depth) + 1e-7))
            else:
                disp = outputs[("disp", scale)]

            color = inputs[("color", 0, scale)]
            if not self.opt.depth_supervision_only:
                color = inputs[("color", 0, scale)]
                target = inputs[("color", 0, source_scale)]

                for frame_id in self.opt.frame_ids[1:]:
                    pred = outputs[("color", frame_id, scale)]
                    reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                    if not is_multi and self.opt.res_pose:
                        reprojection_res_losses.append(torch.min(torch.cat((self.compute_reprojection_loss(
                            outputs[("color_res", frame_id, scale)], target), self.compute_reprojection_loss(
                            inputs[("color", frame_id, source_scale)], target) + torch.randn(
                            target.mean(1, True).shape).to(self.device) * 0.00001), 1), 1, True)[0])
                reprojection_losses = torch.cat(reprojection_losses, 1)
                if not is_multi and self.opt.res_pose:
                    reprojection_res_losses = torch.cat(reprojection_res_losses, 1)

                if not self.opt.disable_automasking:
                    identity_reprojection_losses = []
                    for frame_id in self.opt.frame_ids[1:]:
                        pred = inputs[("color", frame_id, source_scale)]
                        identity_reprojection_losses.append(
                            self.compute_reprojection_loss(pred, target))

                    identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                    if self.opt.avg_reprojection:
                        identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                    else:
                        # differently to Monodepth2, compute mins as we go
                        identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1,
                                                                  keepdim=True)
                else:
                    identity_reprojection_loss = None

                if self.opt.avg_reprojection:
                    reprojection_loss = reprojection_losses.mean(1, keepdim=True)
                else:
                    # differently to Monodepth2, compute mins as we go
                    reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)

                if not self.opt.disable_automasking:
                    # add random numbers to break ties
                    identity_reprojection_loss += torch.randn(
                        identity_reprojection_loss.shape).to(self.device) * 0.00001

                # find minimum losses from [reprojection, identity]
                reprojection_loss_mask = self.compute_loss_masks(reprojection_loss,
                                                                 identity_reprojection_loss)

                # find which pixels to apply reprojection loss to, and which pixels to apply
                # consistency loss to
                if is_multi:
                    reprojection_loss_mask = torch.ones_like(reprojection_loss_mask)
                    if not self.opt.disable_motion_masking:
                        reprojection_loss_mask = (reprojection_loss_mask *
                                                  outputs['consistency_mask'].unsqueeze(1))

                    if not self.opt.no_matching_augmentation:
                        reprojection_loss_mask = (reprojection_loss_mask *
                                                  (1 - outputs['augmentation_mask']))
                    consistency_mask = (1 - reprojection_loss_mask).float()

                # standard reprojection loss
                reprojection_loss = reprojection_loss * reprojection_loss_mask
                reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)

                # consistency loss:
                # encourage multi frame prediction to be like singe frame where masking is happening
                if is_multi:  # and self.opt.motion_masking_begin <= self.epoch <= self.opt.motion_masking_end:# and not self.opt.freeze_teacher_and_pose:
                    multi_depth = outputs[("depth", 0, scale)]
                    # no gradients for mono prediction!
                    if self.opt.post_process_mono_while_training:
                        mono_depth = outputs[("mono_depth_pp", 0, scale)].detach()
                    else:
                        mono_depth = outputs[("mono_depth", 0, scale)].detach()
                    consistency_loss_ = (torch.abs(multi_depth - mono_depth))
                    consistency_loss = consistency_loss_ * consistency_mask

                    consistency_loss = consistency_loss.mean()
                    losses['consistency_loss/{}'.format(scale)] = consistency_loss
                else:
                    consistency_loss = 0

                losses['reproj_loss/{}'.format(scale)] = reprojection_loss

                loss += reprojection_loss + consistency_loss

                if not is_multi and self.opt.res_pose:
                    loss += reprojection_res_losses.min(1, True)[0].mean()

            if self.opt.depth_supervision:
                mask = (inputs[("depth")] >= self.opt.min_depth).float() * (
                        inputs[("depth")] <= self.opt.max_depth).float()
                # if is_multi:
                depth = outputs[("depth", 0, scale)]
                # else:
                #     depth = outputs[("depth", 0, scale)]
                supervised_depth_loss = ((torch.abs(inputs[("depth")] - depth) * mask).sum() / mask.sum())
                losses['supervised_depth_loss/{}'.format(scale)] = supervised_depth_loss
                loss += supervised_depth_loss
            else:
                losses['supervised_depth_loss/{}'.format(scale)] = 0.

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales

        if not self.opt.depth_supervision_only and self.opt.supervise_pose:
            t_loss = 0.
            r_loss = 0.
            if not is_multi:
                for frame_id in self.opt.frame_ids[1:]:
                    if frame_id != "s":
                        T_pred = outputs[("cam_T_cam", 0, frame_id)]
                        T_GT = inputs[("poses", frame_id)]

                        R_pred = roma.rotmat_to_rotvec(T_pred[:, :3, :3])
                        R_GT = roma.rotmat_to_rotvec(T_GT[:, :3, :3])
                        t_pred = T_pred[:, :3, 3]
                        t_GT = T_GT[:, :3, 3]
                        r_loss += 0.1 * torch.pow(R_pred - R_GT, 2).mean()
                        t_loss += 1.0 * torch.pow(t_pred - t_GT, 2).mean()

                    losses['r_loss'] = r_loss
                    losses['t_loss'] = t_loss
                    total_loss += 1.0 * r_loss + 1.0 * t_loss
            else:
                losses['r_loss'] = 0
                losses['t_loss'] = 0

            if is_multi:
                losses['sample_loss/{}'.format(scale)] = sample_loss
            # total_loss += 1e-3 * sample_loss

        losses["loss"] = total_loss

        return losses

    def compute_depth_losses(self, inputs, outputs, losses, mono=False):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        min_depth = self.opt.min_depth
        max_depth = self.opt.max_depth

        if self.opt.train_student:
            if mono:
                depth_pred = outputs[("mono_depth", 0, 0)]
            else:
                depth_pred = outputs[("depth", 0, 0)]
        else:
            depth_pred = outputs[("depth", 0, 0)]

        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False), self.opt.min_depth,
            self.opt.max_depth)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = (depth_gt > min_depth) * (depth_gt < max_depth)

        # # garg/eigen crop
        # crop_mask = torch.zeros_like(mask)
        # crop_mask[:, :, 153:371, 44:1197] = 1
        # mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        if not self.opt.depth_supervision and not self.opt.train_stereo_only:
            depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=self.opt.min_depth, max=self.opt.max_depth)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        if mono:
            for i, metric in enumerate(self.depth_metric_names):
                losses[metric] = np.array(depth_errors[i].cpu())
        else:
            for i, metric in enumerate(self.depth_metric_names):
                losses[metric] = np.array(depth_errors[i].cpu())

    def compute_depth_losses_from_list(self, gts, preds, losses, masks, material="all"):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        errors = []
        MIN_DEPTH = self.opt.min_depth
        MAX_DEPTH = self.opt.max_depth
        for k in range(len(preds)):
            depth_pred_batch = preds[k]
            depth_pred_batch = torch.clamp(F.interpolate(
                depth_pred_batch, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False),
                self.opt.min_depth, self.opt.max_depth)

            depth_gt_batch = gts[k]
            masks_batch = masks[k]
            # print(masks_batch.shape)  # 12x1x320x480

            for b in range(self.opt.batch_size):

                depth_pred = depth_pred_batch.detach()[:, 0].numpy()[b]
                depth_gt = depth_gt_batch.detach()[:, 0].numpy()[b]
                mask = np.logical_and(depth_gt > MIN_DEPTH, depth_gt < MAX_DEPTH)  # 320x480

                if material == "all":
                    depth_pred = depth_pred[mask]  # 1 dim array of non-zero values
                    depth_gt = depth_gt[mask]  # 1 dim array of non-zero values

                elif material == "glass":
                    mask_gt = masks_batch.detach()[:, 0].numpy()[b]  # 320x480
                    mask_glass = np.logical_and(mask_gt >= 160, mask_gt <= 160)
                    mask_final = np.logical_and(mask == 1, mask_glass == 1)

                    depth_pred = depth_pred[mask_final]  # 1 dim array of non-zero values
                    depth_gt = depth_gt[mask_final]  # 1 dim array of non-zero values

                if not self.opt.depth_supervision and not self.opt.train_stereo_only:
                    depth_pred *= np.median(depth_gt) / np.median(depth_pred)

                depth_pred[depth_pred < MIN_DEPTH] = MIN_DEPTH
                depth_pred[depth_pred > MAX_DEPTH] = MAX_DEPTH

                depth_errors = compute_depth_errors_numpy(depth_gt, depth_pred)

                errors.append(depth_errors)

        mean_errors = np.array(errors).mean(0)

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.5f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(mean_errors[i])

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {} | timestamp: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left),
                                  time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))))

    def log(self, mode, inputs, outputs, losses, log_images=True, log_essential_images=False, mono_depth=False):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
        if log_images or log_essential_images:
            if self.opt.train_stereo_only or self.opt.depth_supervision_only or self.opt.depth_supervision:
                for j in range(min(4, self.opt.batch_size)):

                    color = inputs[("color", 0, 0)]
                    color = color[0, :3, :]
                    color = np.array(color.cpu())
                    writer.add_image(
                        "color/{}".format(j),
                        color, self.step
                    )

                    color_aug = inputs[("color_aug", 0, 0)]
                    color_aug = color_aug[0, :3, :]
                    color_aug = np.array(color_aug.cpu())
                    writer.add_image(
                        "color_aug/{}".format(j),
                        color_aug, self.step
                    )

                    if mono_depth:
                        depth = colormap(outputs[("mono_depth", 0, 0)][j, 0])
                        camera_matrix = inputs[("K", 0)][j, :3, :3].unsqueeze(0)
                        writer.add_image(
                            "depth_/{}".format(j),
                            depth, self.step)

                        normals = depth_to_normals(outputs[("mono_depth", 0, 0)][j].unsqueeze(0), camera_matrix)
                        normals = normals.squeeze(0).cpu().numpy()
                        writer.add_image(
                            "normals_/{}".format(j),
                            normals, self.step)
                    else:
                        depth = colormap(outputs[("depth", 0, 0)][j, 0])
                        writer.add_image(
                            "depth_/{}".format(j),
                            depth, self.step)

                    if self.opt.train_stereo_only:
                        disp_gt = colormap((0.0498921 * 423.164) / inputs[("depth_gt")][j, 0])

                        writer.add_image(
                            "disp_gt_/{}".format(j),
                            disp_gt, self.step)

                    depth_gt = colormap(inputs[("depth_gt")][j, 0])
                    writer.add_image(
                        "depth_gt_/{}".format(j),
                        depth_gt, self.step)

        # print(self.opt.train_stereo_only, self.opt.depth_supervision_only)
        if not self.opt.train_stereo_only and not self.opt.depth_supervision_only:
            if log_images:

                for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
                    s = 0  # log only max scale
                    for frame_id in self.opt.frame_ids:
                        writer.add_image(
                            "color_{}_{}/{}".format(frame_id, s, j),
                            inputs[("color", frame_id, s)][j].data, self.step)
                        if s == 0 and frame_id != 0:
                            writer.add_image(
                                "color_pred_{}_{}/{}".format(frame_id, s, j),
                                outputs[("color", frame_id, s)][j].data, self.step)

                            # if self.opt.train_student:
                            #     if s==0:#outputs.get("color_with_gt_pose") is not None:
                            #         writer.add_image(
                            #             "color_with_gt_pose_{}_{}/{}".format(frame_id, s, j),
                            #             outputs[("color_with_gt_pose", frame_id, s)][j].data, self.step)
                            #
                            #         writer.add_image(
                            #             "color_with_gt_depth_gt_pose_{}_{}/{}".format(frame_id, s, j),
                            #             outputs[("color_with_gt_depth_gt_pose", frame_id, s)][j].data, self.step)
                            #
                            #         writer.add_image(
                            #             "color_with_gt_pose_DIFF_{}_{}/{}".format(frame_id, s, j),
                            #             torch.abs(inputs[("color", 0, 0)] - outputs[("color_with_gt_pose", frame_id, s)]).mean(1,True)[j].data, self.step)
                            #
                            #         writer.add_image(
                            #             "color_with_gt_depth_gt_pose_DIFF_{}_{}/{}".format(frame_id, s, j),
                            #             torch.abs(inputs[("color", 0, 0)] - outputs[("color_with_gt_depth_gt_pose", frame_id, s)]).mean(1,True)[j].data, self.step)

                    if self.opt.train_student:
                        for scale in self.opt.scales:
                            disp = colormap(outputs[("disp", scale)][j, 0])
                            writer.add_image(
                                "disp_multi_{}/{}".format(scale, j),
                                disp, self.step)

                        disp = colormap(outputs[('mono_disp', s)][j, 0])
                        writer.add_image(
                            "disp_mono/{}".format(j),
                            disp, self.step)
                    else:
                        for scale in self.opt.scales:
                            disp = colormap(outputs[("disp", scale)][j, 0])
                            writer.add_image(
                                "disp_{}/{}".format(scale, j),
                                disp, self.step)

                    if inputs.get("depth_gt") is not None:
                        depth_gt = colormap(inputs[('depth_gt')][j, 0])
                        writer.add_image(
                            "depth_gt/{}".format(j),
                            depth_gt, self.step)

                    if inputs.get("depth") is not None:
                        depth = colormap(inputs[('depth')][j, 0])
                        writer.add_image(
                            "depth/{}".format(j),
                            depth, self.step)

                    if outputs.get("lowest_cost") is not None:
                        lowest_cost = outputs["lowest_cost"][j]

                        consistency_mask = \
                            outputs['consistency_mask'][j].cpu().detach().unsqueeze(0).numpy()

                        min_val = np.percentile(lowest_cost.numpy(), 10)
                        max_val = np.percentile(lowest_cost.numpy(), 90)
                        lowest_cost = torch.clamp(lowest_cost, min_val, max_val)
                        lowest_cost = colormap(lowest_cost)

                        writer.add_image(
                            "lowest_cost/{}".format(j),
                            lowest_cost, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                # save estimates of depth bins
                to_save['min_depth_bin'] = self.min_depth_tracker
                to_save['max_depth_bin'] = self.max_depth_tracker
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_mono_model(self):

        model_list = ['pose_encoder', 'pose', 'mono_encoder', 'mono_depth', 'encoder']
        for n in model_list:
            print('loading {}'.format(n))
            path = os.path.join(self.opt.mono_weights_folder, "{}.pth".format(n))
            print(path)
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if n != 'encoder':
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

            if n == 'encoder':
                min_depth_bin = pretrained_dict.get('min_depth_bin')
                max_depth_bin = pretrained_dict.get('max_depth_bin')
                print('min depth', min_depth_bin, 'max_depth', max_depth_bin)
                if min_depth_bin is not None:
                    # recompute bins
                    print('setting depth bins!')
                    self.models['encoder'].compute_depth_bins(min_depth_bin, max_depth_bin)

                    self.min_depth_tracker = min_depth_bin
                    self.max_depth_tracker = max_depth_bin
                else:
                    self.min_depth_tracker = self.opt.min_depth
                    self.max_depth_tracker = self.opt.max_depth

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            if n == 'encoder':
                min_depth_bin = pretrained_dict.get('min_depth_bin')
                max_depth_bin = pretrained_dict.get('max_depth_bin')
                print('min depth', min_depth_bin, 'max_depth', max_depth_bin)
                if min_depth_bin is not None:
                    # recompute bins
                    print('setting depth bins!')
                    self.models['encoder'].compute_depth_bins(min_depth_bin, max_depth_bin)

                    self.min_depth_tracker = min_depth_bin
                    self.max_depth_tracker = max_depth_bin

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict, strict=False)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            try:
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            except ValueError:
                print("Can't load Adam - using random")
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis


def flow2rgb(flow_map, max_value=None):
    flow_map_np = flow_map.detach().cpu().numpy().astype(float)
    _, h, w = flow_map_np.shape
    flow_map_np[:, (flow_map_np[0] == 0.0) & (flow_map_np[1] == 0.0)] = float('nan')
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0, 1)
