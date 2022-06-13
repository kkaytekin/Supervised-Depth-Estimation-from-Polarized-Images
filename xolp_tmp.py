# tmp file for xolp calculation

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