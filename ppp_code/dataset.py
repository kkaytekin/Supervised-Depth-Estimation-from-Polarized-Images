import os
import json
import numpy as np
import torch
import cv2
import torch.utils.data as data
from torchvision import transforms

from utils import egocentric_to_allocentric, mat_to_ortho6d_np, transform_translation_parametrization, read_image_cv2, get_2d_coord_np, crop_resize_by_warp_affine
from customized_bop_utils import load_ply, get_symmetry_transformations
from physical_normals_LS import PolarisationImage_ls, rho_spec_ls, rho_diffuse_ls, calc_normals_ls

import imgaug.augmenters as iaa
from imgaug.augmenters import (Sometimes, GaussianBlur, Add, Multiply, CoarseDropout, Invert, LinearContrast)

class PolarDataset(data.Dataset):
    def __init__(self, data_path, filename, obj_id, obj_name, obj_info, refractive_index, object_ply_path, istrain=False):
        super(PolarDataset, self).__init__()
        """
        Args:
            data_path: path to the dataset
            filenames: list containing the filenames, e.g., ['91', '194', '71', '187']
            obj_id: e.g. 1
            obj_info: diameter, size, symmetry
        """
        self.data_path = data_path
        self.filename = filename
        self.obj_id = obj_id
        self.obj_name = obj_name
        self.obj_info = obj_info
        self.istrain = istrain
        self.obj_ply_path = object_ply_path
        self.refractive_index = refractive_index

        object_plymodel_path = os.path.join(self.obj_ply_path, self.obj_name+'.ply')
        model = load_ply(object_plymodel_path)
        self.pts = model["pts"]

        gt_poses_path = os.path.join(self.data_path, self.obj_name, 'scene_gt.json')
        with open(gt_poses_path, 'r') as load_f:
            gt_poses = json.load(load_f)

        if self.istrain:
            bbox_annotation_path = os.path.join(self.data_path, self.obj_name, 'scene_gt_info.json')
            with open(bbox_annotation_path) as f:
                annotations = json.load(f)
        else:
            bbox_annotation_path = os.path.join(self.data_path, 'pred_bbox.json')
            with open(bbox_annotation_path) as f:
                annotations = json.load(f)


        self.K = np.array([351.4501953125, 0.0, 303.6462522979127, 0.0, 351.4768981933594, 251.42255782417124, 0.0, 0.0, 1.0]).reshape(
            (3, 3)).astype("float32")

        rgb_path_0 = []
        rgb_path_45 = []
        rgb_path_90 = []
        rgb_path_135 = []
        gt_pose = []

        gt_mask_visib_path = []

        gt_normal_path = []

        self.gt_pose_mat = []
        self.gt_ego_6drotation_mat = []

        self.reparametrized_gt_translation = []
        self.gt_allo_6drotation_mat = []

        self.roi_coord = []
        nocs_path = []

        self.gt_ratio_translation = []
        self.resize_ratio = []


        self.bbox_visib = []

        self.color_aug = iaa.Sequential(
            [
            Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),
            Sometimes(0.5, GaussianBlur(1.2*np.random.rand())),
            Sometimes(0.5, Add((-25, 25), per_channel=0.3)),
            Sometimes(0.3, Invert(0.2, per_channel=True)),
            Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
            Sometimes(0.5, Multiply((0.6, 1.4))),
            Sometimes(0.5, LinearContrast((0.5, 2.2), per_channel=0.3))
            ], random_order = False
        )

        count = 0
        for i in range(len(filename)):

            if self.istrain:
                annotation = annotations[filename[i]]
                bbox = annotation['bbox_visib']
                self.bbox_visib.append(bbox)
            else:
                annotation = annotations[filename[i]]
                if str(self.obj_id) in annotation:
                    bbox = annotation[str(self.obj_id)]
                    self.bbox_visib.append(bbox)
                    count += 1
                else:
                    continue

            image_path_0 = os.path.join(self.data_path, self.obj_name, 'pol', str(filename[i]).zfill(6) + '_0.png')
            image_path_45 = os.path.join(self.data_path, self.obj_name, 'pol', str(filename[i]).zfill(6) + '_45.png')
            image_path_90 = os.path.join(self.data_path, self.obj_name, 'pol', str(filename[i]).zfill(6) + '_90.png')
            image_path_135 = os.path.join(self.data_path, self.obj_name, 'pol', str(filename[i]).zfill(6) + '_135.png')
            rgb_path_0.append(image_path_0)
            rgb_path_45.append(image_path_45)
            rgb_path_90.append(image_path_90)
            rgb_path_135.append(image_path_135)

            gt_normal_p = os.path.join(self.data_path, self.obj_name, 'gt_normals_visib', '{}.png'.format(str(filename[i]).zfill(6)))
            gt_normal_path.append(gt_normal_p)

            pose_index = filename[i]  # e.g., '000001', '100002'

            obj_pose = gt_poses[pose_index]

            assert obj_pose is not None

            gt_pose.append(obj_pose) # should be a list containing dictionaries

            mask_visib_path = os.path.join(self.data_path, self.obj_name, 'mask_visib', str(filename[i]).zfill(6) + '.png')
            gt_mask_visib_path.append(mask_visib_path)

            R = np.array(obj_pose['cam_R_m2c']).reshape((3, 3)).astype("float32")  # 3x3
            t = np.array(obj_pose['cam_t_m2c']).astype("float32")  # (3, ), in meter!

            # TODO: check whether this is correct for the symmetric object!!
            # if obj_id in (3):
            #     R = symmetric_rot(R,t)

            pose = np.empty((4, 4))
            pose[:3, :3] = R  # * scale
            pose[:3, 3] = t
            pose[3] = [0, 0, 0, 1]

            self.gt_pose_mat.append(pose)

            # for allocentric continuous 6D rotation
            allo_pose = egocentric_to_allocentric(pose.astype("float32")) # 3 x 4, without the last row of [0 0 0 1], this can be done when translation is in meter or millimeter
            allo_R_6d_np = mat_to_ortho6d_np(allo_pose[:3, :3].astype("float32")) # from allocentric rotation to allocentric continuous 6d rotation
            self.gt_allo_6drotation_mat.append(allo_R_6d_np)
            #
            # for egocentric continuous 6d
            ego_R_6d_np = mat_to_ortho6d_np(R.astype("float32"))
            self.gt_ego_6drotation_mat.append(ego_R_6d_np)

            # reparametrize translation from (tx, ty, tz) to (cx, cy, tz), Note that the forward and backward transformation should be done in millimeter!
            reparametrized_translation = transform_translation_parametrization(t*1000, self.K)  # Note: t should be in millimeter
            # reparametrized_translation = reparametrized_translation * 0.001  # transform the scale from millimeter to meter
            self.reparametrized_gt_translation.append(reparametrized_translation)

            nocs_p = os.path.join(self.data_path, self.obj_name, 'gt_nocs_visib', '{}.png'.format(str(filename[i]).zfill(6)))
            nocs_path.append(nocs_p)

        print("original size of test split: {}, real size: {}".format(len(filename), count))

        self.nocs_ls = nocs_path

        self.rgb_path_0_ls = rgb_path_0
        self.rgb_path_45_ls = rgb_path_45
        self.rgb_path_90_ls = rgb_path_90
        self.rgb_path_135_ls = rgb_path_135
        self.gt_mask_path_ls = gt_mask_visib_path
        self.gt_normal_path_ls = gt_normal_path


    def __len__(self):
        return len(self.rgb_path_0_ls)

    def __getitem__(self, idx):

        self.transform = (lambda x: x)

        to_tensor = transforms.ToTensor()
        grayscale = transforms.Grayscale(num_output_channels=1)

        rgb_0 = read_image_cv2(self.rgb_path_0_ls[idx], format='RGB')
        rgb_45 = read_image_cv2(self.rgb_path_45_ls[idx], format='RGB')
        rgb_90 = read_image_cv2(self.rgb_path_90_ls[idx], format='RGB')
        rgb_135 = read_image_cv2(self.rgb_path_135_ls[idx], format='RGB')
        rgb_unpolar = 0.5 * (rgb_0 + rgb_90)
        rgb_unpolar = np.clip(rgb_unpolar, a_min=0, a_max=255).astype("float32")

        im_H_ori, im_W_ori = rgb_0.shape[:2]

        model_points, cur_extents = self._get_model_points_and_extents()

        coord_2d = get_2d_coord_np(im_W_ori, im_H_ori, low=0, high=1).transpose(1,2,0)  # CHW -> HWC, input original im_W, im_H

        bbox = self.bbox_visib[idx]

        roi_infos = {}

        roi_infos["roi_cams"] = torch.from_numpy(self.K)
        roi_infos["model_points"] = torch.as_tensor(model_points.astype("float32"))
        roi_infos["roi_extents"] = torch.tensor(cur_extents, dtype=torch.float32)

        roi_infos["adr_rgb_0"] =  self.rgb_path_0_ls[idx]

        gt_normals = read_image_cv2(self.gt_normal_path_ls[idx], format="RGB")
        mask = read_image_cv2(self.gt_mask_path_ls[idx], format="L")  # [448, 512]
        nocs = read_image_cv2(self.nocs_ls[idx], format='RGB')

        sym_info = self.get_symmetric_infos()
        if sym_info is not None:
            roi_infos["sym_info"] = sym_info

        if self.istrain is False:
            x1, y1, w, h = bbox  # this is important for coco annotation!!!
            x2 = x1 + w
            y2 = y1 + h
            bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
            bw = max(x2 - x1, 1)  # w
            bh = max(y2 - y1, 1)  # h
            scale = max(bh, bw) * 1.5
            scale = min(scale, max(512, 612)) * 1.0
            roi_coord_2d = crop_resize_by_warp_affine(
                coord_2d, bbox_center, scale, 64, 64, interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)  # HWC -> CHW

            obj_center = self.reparametrized_gt_translation[idx][:2]
            delta_c = obj_center - bbox_center
            resize_ratio = 64 / scale
            z_ratio = self.reparametrized_gt_translation[idx][2] / resize_ratio

            roi_infos["roi_coord_2d"] = torch.as_tensor(roi_coord_2d.astype("float32")).contiguous()
            roi_infos["gt_ratio_translation"] = torch.as_tensor(np.asarray([delta_c[0] / bw, delta_c[1] / bh, z_ratio]), dtype=torch.float32)
            roi_infos["resize_ratio"] = torch.tensor(resize_ratio).to(torch.float32)
            roi_infos["roi_centers"] = torch.as_tensor(bbox_center, dtype=torch.float32)
            roi_infos["roi_whs"] = torch.as_tensor(np.asarray([bw, bh], dtype=np.float32))

            roi_infos["gt_post_mat"] = torch.as_tensor(self.gt_pose_mat[idx], dtype=torch.float32)
            roi_infos["roi_scale"] = torch.as_tensor(scale, dtype=torch.float32)

            roi_img_0 = crop_resize_by_warp_affine(rgb_0, bbox_center, scale, 256, 256, interpolation=cv2.INTER_LINEAR)
            roi_img_45 = crop_resize_by_warp_affine(rgb_45, bbox_center, scale, 256, 256, interpolation=cv2.INTER_LINEAR)
            roi_img_90 = crop_resize_by_warp_affine(rgb_90, bbox_center, scale, 256, 256, interpolation=cv2.INTER_LINEAR)
            roi_img_135 = crop_resize_by_warp_affine(rgb_135, bbox_center, scale, 256, 256, interpolation=cv2.INTER_LINEAR)
            rgb_unpolar = crop_resize_by_warp_affine(rgb_unpolar, bbox_center, scale, 256, 256, interpolation=cv2.INTER_LINEAR)

            roi_infos["roi_img_0"] = to_tensor(roi_img_0).contiguous()
            roi_infos["roi_img_45"] = to_tensor(roi_img_45).contiguous()
            roi_infos["roi_img_90"] = to_tensor(roi_img_90).contiguous()
            roi_infos["roi_img_135"] = to_tensor(roi_img_135).contiguous()
            roi_infos["rgb_unpolar"] = to_tensor(rgb_unpolar).contiguous()

            i_0 = grayscale(to_tensor(roi_img_0)).squeeze(0).numpy() * 255  # [448, 512]
            i_45 = grayscale(to_tensor(roi_img_45)).squeeze(0).numpy() * 255
            i_90 = grayscale(to_tensor(roi_img_90)).squeeze(0).numpy() * 255
            i_135 = grayscale(to_tensor(roi_img_135)).squeeze(0).numpy() * 255

            roi_mask = crop_resize_by_warp_affine(mask, bbox_center, scale, 256, 256, interpolation=cv2.INTER_LINEAR)
            gt_roi_mask = crop_resize_by_warp_affine(mask, bbox_center, scale, 64, 64, interpolation=cv2.INTER_LINEAR)

            roi_infos["roi_mask"] = to_tensor(gt_roi_mask)
            mask_1 = np.array(roi_mask, dtype=bool)

            angles = np.array([0, 45, 90, 135]) * np.pi / 180
            n = self.refractive_index

            images = np.zeros((i_0.shape[0], i_0.shape[1], 4))
            images[:, :, 0][mask_1] = i_0[mask_1]
            images[:, :, 1][mask_1] = i_45[mask_1]
            images[:, :, 2][mask_1] = i_90[mask_1]
            images[:, :, 3][mask_1] = i_135[mask_1]

            rho2, phi2, Iun2, rho, phi = PolarisationImage_ls(images, angles, mask_1)
            theta_diff = rho_diffuse_ls(rho2, n)
            theta_spec1, theta_spec2 = rho_spec_ls(rho2, n)
            roi_N_diff = calc_normals_ls(phi2, theta_diff, mask_1).transpose((2, 0, 1)).astype("float32")  # [3, 512, 612], values from [-1,1]
            roi_N_spec1 = calc_normals_ls(phi2 + np.pi / 2, theta_spec1, mask_1).transpose((2, 0, 1)).astype("float32")
            roi_N_spec2 = calc_normals_ls(phi2 + np.pi / 2, theta_spec2, mask_1).transpose((2, 0, 1)).astype("float32")

            roi_infos["roi_N_diff"] = torch.from_numpy(roi_N_diff)
            roi_infos["roi_N_spec1"] = torch.from_numpy(roi_N_spec1)
            roi_infos["roi_N_spec2"] = torch.from_numpy(roi_N_spec2)

            roi_infos["roi_dolp"] = torch.as_tensor(rho2, dtype=torch.float32).unsqueeze(0)  # [1, 256, 256], from [0,1]
            roi_infos["roi_aolp"] = torch.as_tensor(phi2, dtype=torch.float32).unsqueeze(0)  # [1, 256, 256], from [-0.5pi, 0.5pi]

            roi_gt_normals = crop_resize_by_warp_affine(gt_normals, bbox_center, scale, 64, 64, interpolation=cv2.INTER_LINEAR)
            roi_gt_normals = to_tensor(roi_gt_normals)  # after loading, the value is between [0, 1], should transform to [-1, 1]
            roi_infos["roi_gt_normals"] = 2 * roi_gt_normals - 1

            roi_nocs = crop_resize_by_warp_affine(nocs, bbox_center, scale, 64, 64, interpolation=cv2.INTER_LINEAR)
            roi_infos["roi_nocs"] = to_tensor(roi_nocs)

            return roi_infos



        if np.random.rand() < 0.8:
            rgb_0 = self.color_aug(image=rgb_0)
            rgb_45 = self.color_aug(image=rgb_45)
            rgb_90 = self.color_aug(image=rgb_90)
            rgb_135 = self.color_aug(image=rgb_135)
            rgb_unpolar = self.color_aug(image=rgb_unpolar)

        x1, y1, w, h = bbox  # this is important for coco annotation!!!
        x2 = x1 + w
        y2 = y1 + h
        bbox_xyxy = np.asarray([x1, y1, x2, y2])
        bbox_center, scale = self.aug_bbox(bbox_xyxy, im_H_ori, im_W_ori) # DZI

        bw = max(bbox_xyxy[2] - bbox_xyxy[0], 1)
        bh = max(bbox_xyxy[3] - bbox_xyxy[1], 1)

        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, 64, 64, interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)  # HWC -> CHW

        obj_center = self.reparametrized_gt_translation[idx][:2]
        delta_c = obj_center - bbox_center
        resize_ratio = 64 / scale
        z_ratio = self.reparametrized_gt_translation[idx][2] / resize_ratio

        roi_infos["roi_coord_2d"] = torch.as_tensor(roi_coord_2d.astype("float32")).contiguous()
        roi_infos["gt_ratio_translation"] = torch.as_tensor(np.asarray([delta_c[0] / bw, delta_c[1] / bh, z_ratio]), dtype=torch.float32)
        roi_infos["resize_ratio"] = torch.tensor(resize_ratio).to(torch.float32)
        roi_infos["roi_centers"] = torch.as_tensor(bbox_center, dtype=torch.float32)
        roi_infos["roi_whs"] = torch.as_tensor(np.asarray([bw, bh], dtype=np.float32))

        roi_img_0 = crop_resize_by_warp_affine(rgb_0, bbox_center, scale, 256, 256, interpolation=cv2.INTER_LINEAR) # 256 256 3
        roi_img_45 = crop_resize_by_warp_affine(rgb_45, bbox_center, scale, 256, 256, interpolation=cv2.INTER_LINEAR)
        roi_img_90 = crop_resize_by_warp_affine(rgb_90, bbox_center, scale, 256, 256, interpolation=cv2.INTER_LINEAR)
        roi_img_135 = crop_resize_by_warp_affine(rgb_135, bbox_center, scale, 256, 256, interpolation=cv2.INTER_LINEAR)
        rgb_unpolar = crop_resize_by_warp_affine(rgb_unpolar, bbox_center, scale, 256, 256, interpolation=cv2.INTER_LINEAR)

        roi_infos["roi_img_0"] = to_tensor(roi_img_0).contiguous() # 3 256 256
        roi_infos["roi_img_45"] = to_tensor(roi_img_45).contiguous()
        roi_infos["roi_img_90"] = to_tensor(roi_img_90).contiguous()
        roi_infos["roi_img_135"] = to_tensor(roi_img_135).contiguous()
        roi_infos["rgb_unpolar"] = to_tensor(rgb_unpolar).contiguous()

        i_0 = grayscale(to_tensor(roi_img_0)).squeeze(0).numpy() * 255  # 256, 256
        i_45 = grayscale(to_tensor(roi_img_45)).squeeze(0).numpy() * 255
        i_90 = grayscale(to_tensor(roi_img_90)).squeeze(0).numpy() * 255
        i_135 = grayscale(to_tensor(roi_img_135)).squeeze(0).numpy() * 255

        roi_mask = crop_resize_by_warp_affine(mask, bbox_center, scale, 256, 256, interpolation=cv2.INTER_LINEAR)
        gt_roi_mask = crop_resize_by_warp_affine(mask, bbox_center, scale, 64, 64, interpolation=cv2.INTER_LINEAR)

        roi_infos["roi_mask"] = to_tensor(gt_roi_mask)
        mask_1 = np.array(roi_mask, dtype=bool)

        angles = np.array([0, 45, 90, 135]) * np.pi / 180
        n = self.refractive_index

        images = np.zeros((i_0.shape[0], i_0.shape[1], 4)) # 256 256 4
        images[:, :, 0][mask_1] = i_0[mask_1]
        images[:, :, 1][mask_1] = i_45[mask_1]
        images[:, :, 2][mask_1] = i_90[mask_1]
        images[:, :, 3][mask_1] = i_135[mask_1]

        rho2, phi2, Iun2, rho, phi = PolarisationImage_ls(images, angles, mask_1)
        theta_diff = rho_diffuse_ls(rho2, n)
        theta_spec1, theta_spec2 = rho_spec_ls(rho2, n)
        roi_N_diff = calc_normals_ls(phi2, theta_diff, mask_1).transpose((2, 0, 1)).astype("float32")  # [3, 256, 256], values from [-1,1]
        roi_N_spec1 = calc_normals_ls(phi2 + np.pi / 2, theta_spec1, mask_1).transpose((2, 0, 1)).astype("float32")
        roi_N_spec2 = calc_normals_ls(phi2 + np.pi / 2, theta_spec2, mask_1).transpose((2, 0, 1)).astype("float32")

        roi_infos["roi_N_diff"] = torch.from_numpy(roi_N_diff)
        roi_infos["roi_N_spec1"] = torch.from_numpy(roi_N_spec1)
        roi_infos["roi_N_spec2"] = torch.from_numpy(roi_N_spec2)

        roi_infos["roi_dolp"] = torch.as_tensor(rho2, dtype=torch.float32).unsqueeze(0)  # shape [1, 256, 256], value from [0,1]
        roi_infos["roi_aolp"] = torch.as_tensor(phi2, dtype=torch.float32).unsqueeze(0)  # shape [1, 256, 256],  value from [-0.5pi, 0.5pi]

        roi_gt_normals = crop_resize_by_warp_affine(gt_normals, bbox_center, scale, 64, 64,
                                                    interpolation=cv2.INTER_LINEAR)
        roi_gt_normals = to_tensor(roi_gt_normals)  # after loading, the value is between [0, 1], should transform to [-1, 1]
        roi_infos["roi_gt_normals"] = 2 * roi_gt_normals - 1

        roi_nocs = crop_resize_by_warp_affine(nocs, bbox_center, scale, 64, 64, interpolation=cv2.INTER_LINEAR)
        roi_infos["roi_nocs"] = to_tensor(roi_nocs)

        roi_infos["gt_post_mat"] = torch.as_tensor(self.gt_pose_mat[idx], dtype=torch.float32)
        roi_infos["gt_allo_6d"] =  torch.from_numpy(self.gt_allo_6drotation_mat[idx])

        return roi_infos

    def _get_model_points_and_extents(self):
        """convert to label based keys."""
        # cur_model_points[0] = model["pts"]
        pts = self.pts
        xmin, xmax = np.amin(pts[:, 0]), np.amax(pts[:, 0])
        ymin, ymax = np.amin(pts[:, 1]), np.amax(pts[:, 1])
        zmin, zmax = np.amin(pts[:, 2]), np.amax(pts[:, 2])
        size_x = xmax - xmin
        size_y = ymax - ymin
        size_z = zmax - zmin
        cur_extents = np.array([size_x, size_y, size_z], dtype="float32")

        num = 3000
        # for i in range(len(cur_model_points)):
        keep_idx = np.arange(num)
        np.random.shuffle(keep_idx)  # random sampling
        cur_model_points = self.pts[keep_idx, :]

        return cur_model_points, cur_extents

    def aug_bbox(self, bbox_xyxy, im_H, im_W):
        x1, y1, x2, y2 = bbox_xyxy.copy()
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bh = y2 - y1
        bw = x2 - x1

        scale_ratio = 1 + 0.25 * (2 * np.random.random_sample() - 1)  # [1-0.25, 1+0.25]
        shift_ratio = 0.25 * (2 * np.random.random_sample(2) - 1)  # [-0.25, 0.25]
        bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])  # (h/2, w/2)
        scale = max(y2 - y1, x2 - x1) * scale_ratio * 1.5
        scale = min(scale, max(im_H, im_W)) * 1.0

        return bbox_center, scale

    def get_symmetric_infos(self):
        """label based keys."""

        if "symmetries_discrete" in self.obj_info or "symmetries_continuous" in self.obj_info:
            sym_transforms = get_symmetry_transformations(self.obj_info, max_sym_disc_step=0.01)
            sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32) # array of shape [314, 3, 3]
        else:
            sym_info = None

        return sym_info


# class NormalDataset(data.Dataset):
#     def __init__(self, data_path, filename, obj_id, istrain=False):
#         super(NormalDataset, self).__init__()
#         """
#         This is a test dataset to test the capability of the PatchPnP module,
#         to see whether that module can recover accurate pose given GT normals (with possibly 2D UV map)
#         for resizing the 2D UV map, need to load blenderproc generated coco annotation for the bbox information
#         """
#         self.data_path = data_path
#         self.filename = filename
#         self.obj_id = obj_id
#         self.istrain = istrain
#
#         gt_poses_path = os.path.join(self.data_path, str(self.obj_id), 'output/bop_data/train_pbr/000000/scene_gt.json')
#         # gt_poses_path = os.path.join(self.data_path, str(self.obj_id), 'scene_gt.json')
#         with open(gt_poses_path, 'r') as load_f:
#             gt_poses = json.load(load_f)
#
#         coco_annotation_path = os.path.join(self.data_path, str(self.obj_id), 'output/coco_data/coco_annotations.json')
#         with open(coco_annotation_path) as f:
#             annotations = json.load(f)
#             categories = annotations['categories']
#             annotations = annotations['annotations']
#
#
#         # self.K = np.array([458.955, 0.0, 308.157, 0.0, 458.955, 262.752, 0.0, 0.0, 1.0]).reshape((3,3))
#         # TODO: now this is the WRONG intrinsics!! Change it later!!
#         self.K = np.array([850.0001945495607, 0.0, 305.5, 0.0, 850.0001945495607, 255.5, 0.0, 0.0, 1.0]).reshape(
#             (3, 3)).astype("float32")
#
#         rgb_path_0 = []
#
#         gt_pose = []
#
#
#         gt_normal_path = []
#
#         self.gt_pose_mat = []
#         self.gt_ego_6drotation_mat = []
#
#         self.reparametrized_gt_translation = []
#         # self.gt_allo_6drotation_mat = []
#
#         # self.gt_centroid = []
#         # self.gt_z = []
#         self.roi_coord = []
#
#         for i in range(len(filename)):
#             image_path_0 = os.path.join(self.data_path, str(self.obj_id), 'mitsuba_cup/output', filename[i], '0.jpg')
#
#             rgb_path_0.append(image_path_0)
#
#
#             gt_normal_p = os.path.join(self.data_path, str(self.obj_id), 'gt_normals', '{}.png'.format(filename[i]))
#             gt_normal_path.append(gt_normal_p)
#
#
#             pose_index = str(int(filename[i]))  # e.g., '91'
#             # gt_pose.extend(gt_poses[pose_index])
#             obj_pose = gt_poses[pose_index][0]
#             gt_pose.append(
#                 obj_pose)  # since the scene gt contains non-necessary information of the ground, the gt pose should only save the object pose
#
#
#
#             R = np.array(gt_pose[i]['cam_R_m2c']).reshape((3, 3)).astype("float32")  # 3x3
#             t = np.array(gt_pose[i]['cam_t_m2c']).astype("float32")  # (3, )
#
#
#             pose = np.empty((4, 4))
#             pose[:3, :3] = R  # * scale
#             pose[:3, 3] = t * 0.001  # transform the scale from millimeter to meter
#             # pose[:3, 3] = t  # TODO: only for different parameterization!
#             pose[3] = [0, 0, 0, 1]
#
#             self.gt_pose_mat.append(pose) # note that the translation in gt pose matrix is in meters!!!
#
#
#             #
#             # for egocentric continuous 6d
#             ego_R_6d_np = mat_to_ortho6d_np(R).astype("float32")
#             self.gt_ego_6drotation_mat.append(ego_R_6d_np)
#
#             # reparametrize translation from (tx, ty, tz) to (cx, cy, tz), Note that the forward and backward transformation should be done in millimeter!
#             reparametrized_translation = transform_translation_parametrization(t,
#                                                                                self.K)  # Note: t is in millimeter, this is correct for the reparametrization!
#             # reparametrized_translation = reparametrized_translation * 0.001  # transform the scale from millimeter to meter
#             self.reparametrized_gt_translation.append(reparametrized_translation)
#
#
#             # load coco annotation for the bbox information! 'image_id' = filename[i]
#             coord_2d = get_2d_coord_np(612, 512, low=0, high=1).transpose(1, 2, 0)  # CHW -> HWC, input original im_W, im_H
#             for _, annotation in enumerate(annotations):
#                 if annotation['image_id'] == int(filename[i]) and annotation['category_id'] == 1:
#                         bbox = annotation['bbox']
#                         x1, y1, w, h = bbox  # this is important for coco annotation!!!
#                         x2 = x1 + w
#                         y2 = y1 + h
#                         bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
#                         bw = max(x2 - x1, 1)  # w
#                         bh = max(y2 - y1, 1)  # h
#                         scale = max(bh, bw) * 1.0
#                         scale = min(scale, max(512, 612)) * 1.0
#                         roi_coord_2d = crop_resize_by_warp_affine(
#                             coord_2d, bbox_center, scale, 153, 128, interpolation=cv2.INTER_LINEAR).transpose(2, 0,1)  # HWC -> CHW #TODO: remove the hardcode 1/4 size of the input!!
#
#                         self.roi_coord.append(roi_coord_2d.astype("float32"))
#
#
#
#
#
#         self.rgb_path_0_ls = rgb_path_0
#
#         self.gt_pose_ls = gt_pose
#
#         self.gt_normal_path_ls = gt_normal_path
#
#     def __len__(self):
#         return len(self.rgb_path_0_ls)
#
#     def __getitem__(self, idx):
#
#         transform_prob = random.random()
#
#         if self.istrain:
#
#
#             # # TODO: find appropriate data augmentation for polarization images!
#             self.transform = (lambda x: x)
#
#         else:
#             self.transform = (lambda x: x)
#
#         to_tensor = transforms.ToTensor()
#         resize = transforms.Resize((128, 153),
#                                    interpolation=PIL.Image.NEAREST)  # 1/4 of the original input size, use this to resize the mask and the normals
#
#
#         gt_normals = pil_loader(self.gt_normal_path_ls[idx], mode="rgb")
#         gt_normals = to_tensor(gt_normals)  # after loading, the value is between [0, 1], should transform to [-1, 1]
#         gt_normals = 2 * gt_normals - 1
#         gt_normals = resize(gt_normals)
#
#         #
#         adr_rgb_0 = self.rgb_path_0_ls[idx]  # "dataset/1/rgb/000000.jpg"
#
#         gt_pose_mat = self.gt_pose_mat[idx]
#         # gt_pose_dict = self.gt_pose_ls[idx]
#
#         gt_ego_R_6d = self.gt_ego_6drotation_mat[idx]
#
#         # gt_allo_R_6d = self.gt_allo_6drotation_mat[idx]
#
#         gt_centroid_translation = self.reparametrized_gt_translation[idx]
#
#
#
#         model_points = 0
#
#         roi_cood = self.roi_coord[idx]
#
#         return  adr_rgb_0, gt_normals, gt_pose_mat, gt_ego_R_6d, gt_centroid_translation, model_points, roi_cood
#
#     def _get_model_points(self, object_path):
#         """convert to label based keys."""
#
#         model = load_ply(object_path, vertex_scale=1.0)
#         # cur_model_points[0] = model["pts"]
#
#         num = 3000
#         # for i in range(len(cur_model_points)):
#         keep_idx = np.arange(num)
#         np.random.shuffle(keep_idx)  # random sampling
#         cur_model_points = model["pts"][keep_idx, :]
#
#         return cur_model_points
#
#
# class NOCSDataset(data.Dataset):
#     def __init__(self, data_path, filename, obj_id, istrain=False):
#         super(NOCSDataset, self).__init__()
#         """
#         This is a test dataset to test the capability of the PatchPnP module,
#         to see whether that module can recover accurate pose given GT NOCS (with possibly 2D UV map)
#         for resizing the 2D UV map, need to load blenderproc generated coco annotation for the bbox information
#         """
#         self.data_path = data_path
#         self.filename = filename
#         self.obj_id = obj_id
#         self.istrain = istrain
#
#         gt_poses_path = os.path.join(self.data_path, str(self.obj_id), 'output/bop_data/train_pbr/000000/scene_gt.json')
#         # gt_poses_path = os.path.join(self.data_path, str(self.obj_id), 'scene_gt.json')
#         with open(gt_poses_path, 'r') as load_f:
#             gt_poses = json.load(load_f)
#
#         coco_annotation_path = os.path.join(self.data_path, str(self.obj_id), 'output/coco_data/coco_annotations.json')
#         with open(coco_annotation_path) as f:
#             annotations = json.load(f)
#             categories = annotations['categories']
#             annotations = annotations['annotations']
#
#
#         # self.K = np.array([458.955, 0.0, 308.157, 0.0, 458.955, 262.752, 0.0, 0.0, 1.0]).reshape((3,3))
#         # TODO: now this is the WRONG intrinsics!! Change it later!!
#         self.K = np.array([850.0001945495607, 0.0, 305.5, 0.0, 850.0001945495607, 255.5, 0.0, 0.0, 1.0]).reshape(
#             (3, 3)).astype("float32")
#
#         rgb_path_0 = []
#
#         gt_pose = []
#
#         gt_normal_path = []
#
#         self.gt_pose_mat = []
#         self.gt_ego_6drotation_mat = []
#
#         self.reparametrized_gt_translation = []
#
#         self.roi_coord = []
#
#         nocs_path = []
#
#         for i in range(len(filename)):
#             image_path_0 = os.path.join(self.data_path, str(self.obj_id), 'mitsuba_cup/output', filename[i], '0.jpg')
#
#             rgb_path_0.append(image_path_0)
#
#
#             gt_normal_p = os.path.join(self.data_path, str(self.obj_id), 'gt_normals', '{}.png'.format(filename[i]))
#             gt_normal_path.append(gt_normal_p)
#
#
#             pose_index = str(int(filename[i]))  # e.g., '91'
#             # gt_pose.extend(gt_poses[pose_index])
#             obj_pose = gt_poses[pose_index][0]
#             gt_pose.append(
#                 obj_pose)  # since the scene gt contains non-necessary information of the ground, the gt pose should only save the object pose
#
#
#
#             R = np.array(gt_pose[i]['cam_R_m2c']).reshape((3, 3)).astype("float32")  # 3x3
#             t = np.array(gt_pose[i]['cam_t_m2c']).astype("float32")  # (3, )
#
#             pose = np.empty((4, 4))
#             pose[:3, :3] = R  # * scale
#             pose[:3, 3] = t * 0.001  # transform the scale from millimeter to meter
#             # pose[:3, 3] = t  # TODO: only for different parameterization!
#             pose[3] = [0, 0, 0, 1]
#
#             self.gt_pose_mat.append(pose) # note that the translation in gt pose matrix is in meters!!!
#
#             #
#             # for egocentric continuous 6d
#             ego_R_6d_np = mat_to_ortho6d_np(R).astype("float32")
#             self.gt_ego_6drotation_mat.append(ego_R_6d_np)
#
#             reparametrized_translation = transform_translation_parametrization(t,
#                                                                                self.K)  # Note: t is in millimeter, this is correct for the reparametrization!
#             self.reparametrized_gt_translation.append(reparametrized_translation)
#
#             # load nocs path
#             nocs_p = os.path.join(self.data_path, str(self.obj_id), 'nocs', '{}.png'.format(filename[i]))
#             nocs_path.append(nocs_p)
#
#             # load coco annotation for the bbox information! 'image_id' = filename[i]
#             coord_2d = get_2d_coord_np(612, 512, low=0, high=1).transpose(1, 2, 0)  # CHW -> HWC, input original im_W, im_H
#             for _, annotation in enumerate(annotations):
#                 if annotation['image_id'] == int(filename[i]) and annotation['category_id'] == 1:
#                         bbox = annotation['bbox']
#                         x1, y1, w, h = bbox  # this is important for coco annotation!!!
#                         x2 = x1 + w
#                         y2 = y1 + h
#                         bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
#                         bw = max(x2 - x1, 1)  # w
#                         bh = max(y2 - y1, 1)  # h
#                         scale = max(bh, bw) * 1.0
#                         scale = min(scale, max(512, 612)) * 1.0
#                         roi_coord_2d = crop_resize_by_warp_affine(
#                             coord_2d, bbox_center, scale, 153, 128, interpolation=cv2.INTER_LINEAR).transpose(2, 0,1)  # HWC -> CHW #TODO: remove the hardcode 1/4 size of the input!!
#
#                         self.roi_coord.append(roi_coord_2d.astype("float32"))
#
#
#
#
#         self.nocs_ls = nocs_path
#         self.rgb_path_0_ls = rgb_path_0
#
#         self.gt_pose_ls = gt_pose
#
#         self.gt_normal_path_ls = gt_normal_path
#
#     def __len__(self):
#         return len(self.rgb_path_0_ls)
#
#     def __getitem__(self, idx):
#
#         transform_prob = random.random()
#
#         if self.istrain:
#
#             # # TODO: find appropriate data augmentation for polarization images!
#             self.transform = (lambda x: x)
#
#         else:
#             self.transform = (lambda x: x)
#
#         to_tensor = transforms.ToTensor()
#         resize = transforms.Resize((128, 153),
#                                    interpolation=PIL.Image.NEAREST)  # 1/4 of the original input size, use this to resize the mask and the normals
#
#
#
#         gt_normals = pil_loader(self.gt_normal_path_ls[idx], mode="rgb")
#         gt_normals = to_tensor(gt_normals)  # after loading, the value is between [0, 1], should transform to [-1, 1]
#         gt_normals = 2 * gt_normals - 1
#         gt_normals = resize(gt_normals)
#
#         adr_rgb_0 = self.rgb_path_0_ls[idx]  # "dataset/1/rgb/000000.jpg"
#
#         gt_pose_mat = self.gt_pose_mat[idx]
#         # gt_pose_dict = self.gt_pose_ls[idx]
#
#         gt_ego_R_6d = self.gt_ego_6drotation_mat[idx]
#
#         # gt_allo_R_6d = self.gt_allo_6drotation_mat[idx]
#
#         gt_centroid_translation = self.reparametrized_gt_translation[idx]
#
#
#         model_points = 0
#
#         roi_cood = self.roi_coord[idx]
#
#         nocs = read_image_cv2(self.nocs_ls[idx], format='RGB')
#         nocs = to_tensor(nocs)
#         nocs = resize(nocs)
#
#         return  adr_rgb_0, gt_normals, gt_pose_mat, gt_ego_R_6d, gt_centroid_translation, model_points, roi_cood, nocs
#
#     def _get_model_points(self, object_path):
#         """convert to label based keys."""
#
#         model = load_ply(object_path, vertex_scale=1.0)
#         # cur_model_points[0] = model["pts"]
#
#         num = 3000
#         # for i in range(len(cur_model_points)):
#         keep_idx = np.arange(num)
#         np.random.shuffle(keep_idx)  # random sampling
#         cur_model_points = model["pts"][keep_idx, :]
#
#         return cur_model_points
