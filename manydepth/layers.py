# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BerHuLoss(nn.Module):
    """Class implementing the BerHu loss."""
    def __init__(self, threshold=0.2):
        """
        Initializes the BerHuLoss class.

        Parameters
        ----------
        threshold : float
            Mask parameter
        """
        super().__init__()
        self.threshold = threshold

    def forward(self, pred, gt):
        """
        Calculates the BerHu loss.

        Parameters
        ----------
        pred : torch.Tensor [B,1,H,W]
            Predicted inverse depth map
        gt : torch.Tensor [B,1,H,W]
            Ground-truth inverse depth map

        Returns
        -------
        loss : torch.Tensor [1]
            BerHu loss
        """
        huber_c = torch.max(pred - gt)
        huber_c = self.threshold * huber_c
        diff = torch.abs(pred - gt)

        # Remove
        # mask = (gt > 0).detach()
        # diff = gt - pred
        # diff = diff[mask]
        # diff = diff.abs()

        huber_mask = (diff > huber_c).detach().float()
        diff2 = diff * huber_mask
        diff2 = diff2 ** 2
        # print(diff.shape)
        # print(diff2.shape)
        return torch.cat((diff, diff2), 1).mean(1, True)


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def warp_flow(x, flow, use_mask=False):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    Inputs:
    x: [B, C, H, W] (im2)
    flow: [B, 2, H, W] flow
    Returns:
    ouptut: [B, C, H, W]
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if grid.shape != flow.shape:
        raise ValueError('the shape of grid {0} is not equal to the shape of flow {1}.'.format(grid.shape, flow.shape))
    if x.is_cuda:
        grid = grid.to(x.get_device())
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(x, vgrid)
    if use_mask:
        mask = torch.autograd.Variable(torch.ones(x.size())).to(x.get_device())
        mask = nn.functional.grid_sample(mask, vgrid)
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        return output * mask
    else:
        return output


def transformerFwd(U,
                   flo,
                   out_size,
                   name='SpatialTransformerFwd'):
    """Forward Warping Layer described in
    'Occlusion Aware Unsupervised Learning of Optical Flow by Yang Wang et al'
    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    flo: float
        The optical flow used for forward warping
        having the shape of [num_batch, height, width, 2].
    backprop: boolean
        Indicates whether to back-propagate through forward warping layer
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    """

    def _repeat(x, n_repeats):
        rep = torch.ones(size=[n_repeats], dtype=torch.long).unsqueeze(1).transpose(1,0)
        x = x.view([-1,1]).mm(rep)
        return x.view([-1]).int()

    def _interpolate(im, x, y, out_size):
        # constants
        num_batch, height, width, channels = im.shape[0], im.shape[1], im.shape[2], im.shape[3]
        out_height = out_size[0]
        out_width = out_size[1]
        max_y = int(height - 1)
        max_x = int(width - 1)

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * (width - 1.0) / 2.0
        y = (y + 1.0) * (height - 1.0) / 2.0

        # do sampling
        x0 = (torch.floor(x)).int()
        x1 = x0 + 1
        y0 = (torch.floor(y)).int()
        y1 = y0 + 1

        x0_c = torch.clamp(x0, 0, max_x)
        x1_c = torch.clamp(x1, 0, max_x)
        y0_c = torch.clamp(y0, 0, max_y)
        y1_c = torch.clamp(y1, 0, max_y)

        dim2 = width
        dim1 = width * height
        base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width).to(im.get_device())

        base_y0 = base + y0_c * dim2
        base_y1 = base + y1_c * dim2
        idx_a = base_y0 + x0_c
        idx_b = base_y1 + x0_c
        idx_c = base_y0 + x1_c
        idx_d = base_y1 + x1_c

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = im.view([-1, channels])
        im_flat = im_flat.float()

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        wa = ((x1_f - x) * (y1_f - y)).unsqueeze(1)
        wb = ((x1_f - x) * (y - y0_f)).unsqueeze(1)
        wc = ((x - x0_f) * (y1_f - y)).unsqueeze(1)
        wd = ((x - x0_f) * (y - y0_f)).unsqueeze(1)

        zerof = torch.zeros_like(wa)
        wa = torch.where(
            (torch.eq(x0_c, x0) & torch.eq(y0_c, y0)).unsqueeze(1), wa, zerof)
        wb = torch.where(
            (torch.eq(x0_c, x0) & torch.eq(y1_c, y1)).unsqueeze(1), wb, zerof)
        wc = torch.where(
            (torch.eq(x1_c, x1) & torch.eq(y0_c, y0)).unsqueeze(1), wc, zerof)
        wd = torch.where(
            (torch.eq(x1_c, x1) & torch.eq(y1_c, y1)).unsqueeze(1), wd, zerof)

        zeros = torch.zeros(
            size=[
                int(num_batch) * int(height) *
                int(width), int(channels)
            ],
            dtype=torch.float)
        output = zeros.to(im.get_device())
        output = output.scatter_add(dim=0, index=idx_a.long().unsqueeze(1).repeat(1,channels), src=im_flat * wa)
        output = output.scatter_add(dim=0, index=idx_b.long().unsqueeze(1).repeat(1,channels), src=im_flat * wb)
        output = output.scatter_add(dim=0, index=idx_c.long().unsqueeze(1).repeat(1,channels), src=im_flat * wc)
        output = output.scatter_add(dim=0, index=idx_d.long().unsqueeze(1).repeat(1,channels), src=im_flat * wd)

        return output

    def _meshgrid(height, width):
        # This should be equivalent to:
        x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
                                 np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        return torch.from_numpy(x_t).float(), torch.from_numpy(y_t).float()

    def _transform(flo, input_dim, out_size):
        num_batch, height, width, num_channels = input_dim.shape[0:4]

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        height_f = float(height)
        width_f = float(width)
        out_height = out_size[0]
        out_width = out_size[1]
        x_s, y_s = _meshgrid(out_height, out_width)
        x_s = x_s.to(flo.get_device()).unsqueeze(0)
        x_s = x_s.repeat([num_batch, 1, 1])

        y_s = y_s.to(flo.get_device()).unsqueeze(0)
        y_s =y_s.repeat([num_batch, 1, 1])

        x_t = x_s + flo[:, :, :, 0] / ((out_width - 1.0) / 2.0)
        y_t = y_s + flo[:, :, :, 1] / ((out_height - 1.0) / 2.0)

        x_t_flat = x_t.view([-1])
        y_t_flat = y_t.view([-1])

        input_transformed = _interpolate(input_dim, x_t_flat, y_t_flat,
                                            out_size)

        output = input_transformed.view([num_batch, out_height, out_width, num_channels])
        return output

    #out_size = int(out_size)
    output = _transform(flo, U, out_size)
    return output


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv5x5(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv5x5, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(2)
        else:
            self.pad = nn.ZeroPad2d(2)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 5)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T, both=False):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        # pix_coords_ = pix_coords.clone()
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        # if both:
        #     return pix_coords, pix_coords_
        # else:
        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class DepthWarping(nn.Module):
    """
        warp a depth map from frame_idx_0 to frame_idx_1
    """

    def __init__(self, batch_size, height, width):
        super(DepthWarping, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K, T):
        points3D = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        points3D = depth.view(self.batch_size, 1, -1) * points3D
        points3D = torch.cat([points3D, self.ones], 1)
        points3D_transformed = torch.matmul(T, points3D)
        depth_warped = points3D_transformed[:, 2, :].view(self.batch_size, 1, self.height, self.width)

        return depth_warped



def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def compute_depth_errors_numpy(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
