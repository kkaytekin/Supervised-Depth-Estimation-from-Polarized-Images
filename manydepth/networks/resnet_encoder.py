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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from manydepth.layers import BackprojectDepth, Project3D
#from manydepth.softsplat import FunctionSoftsplat

from einops.einops import rearrange


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class Spatial_Attn(nn.Module):
    """Cross attention Layer using spatial relation"""

    def __init__(self, in_dim, out_dim, radii=0.3):
        super(Spatial_Attn, self).__init__()
        self.chanel_in = in_dim
        self.chanel_out = out_dim
        self.height = 24
        self.width = 80
        self.pix_num = self.height * self.width
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # Learnable parameters
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1) * 0.7)
        # TODO: decide best radii for Temporal Attn.
        self.sigma_3d = nn.Parameter(torch.ones(1) * radii * 30.0 / 36.0, requires_grad=False)

        self.context_conv = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(64))

        self.ca_conv = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(64))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(64))

    def compute_3d_attention(self, distance):
        distance_kernel = torch.exp(-(distance.detach()) / (2 * self.sigma_3d))
        attention_3d = distance_kernel
        return attention_3d

    def forward(self, context_feature, distance):
        """
            inputs :
                mask: binary mask, 0 for invalid depth, 1 for valid depth
                distance : pair-wise euclidean distance of each point (B x num_views x N x N)
                context_feature : input feature maps( B X C1 X W X H) for of context
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        batch_size, _, height, width = context_feature.size()
        pix_num = height * width

        context_feature = context_feature.view(batch_size, -1, height, width)
        distance = distance.view(batch_size, pix_num, pix_num)

        attention = self.compute_3d_attention(distance)

        proj_value = self.value_conv(context_feature).view(batch_size, -1, pix_num)
        normalizer = torch.sum(attention, dim=1, keepdim=True)
        ca_feature = torch.bmm(proj_value,
                               attention) / normalizer  # attention is a symmetric matrix now, no need for transpose
        ca_feature = ca_feature.view(batch_size,
                                     self.chanel_out,
                                     height, width).contiguous()

        out = torch.cat((self.ca_conv(ca_feature),
                         self.context_conv(context_feature)), dim=1)

        out = self.conv1(out)
        out = self.gamma * out + context_feature

        out = out.view(batch_size, -1, height, width)
        attention = attention.view(batch_size, pix_num, pix_num)#[:, 0, :, :]  # B x N x N, only central frame

        return out, attention


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(48, 160), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        # v_length = values.size(1)
        # values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z)# * v_length

        return queried_values.contiguous()


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        # self.attention = FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class ResnetEncoderMatching(nn.Module):
    """Resnet encoder adapted to include a cost volume after the 2nd block.

    Setting adaptive_bins=True will recompute the depth bins used for matching upon each
    forward pass - this is required for training from monocular video as there is an unknown scale.
    """

    def __init__(self, num_layers, pretrained, input_height, input_width,
                 min_depth_bin=0.1, max_depth_bin=20.0, num_depth_bins=96,
                 adaptive_bins=False, depth_binning='linear', batch_size=8):

        super(ResnetEncoderMatching, self).__init__()

        self.adaptive_bins = adaptive_bins
        self.depth_binning = depth_binning
        self.set_missing_to_max = True

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.num_depth_bins = num_depth_bins
        # we build the cost volume at 1/4 resolution
        self.matching_height, self.matching_width = input_height // 4, input_width // 4

        self.is_cuda = False
        self.warp_depths = None
        self.depth_bins = None

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        encoder = resnets[num_layers](pretrained)
        self.layer0 = nn.Sequential(encoder.conv1,  encoder.bn1, encoder.relu)
        self.layer1 = nn.Sequential(encoder.maxpool,  encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.backprojector = BackprojectDepth(batch_size=self.num_depth_bins,
                                              height=self.matching_height,
                                              width=self.matching_width)
        # self.backprojector_ = BackprojectDepth(batch_size=batch_size,
        #                                       height=self.matching_height,
        #                                       width=self.matching_width)
        self.projector = Project3D(batch_size=self.num_depth_bins,
                                   height=self.matching_height,
                                   width=self.matching_width)
        # self.projector_ = Project3D(batch_size=batch_size,
        #                            height=self.matching_height,
        #                            width=self.matching_width)

        self.compute_depth_bins(min_depth_bin, max_depth_bin)

        self.prematching_conv = nn.Sequential(nn.Conv2d(64, out_channels=16,
                                                        kernel_size=1, stride=1, padding=0),
                                              nn.ReLU(inplace=True)
                                              )

        self.reduce_conv = nn.Sequential(nn.Conv2d(self.num_ch_enc[1] + self.num_depth_bins,
                                                   out_channels=self.num_ch_enc[1],
                                                   kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(inplace=True)
                                         )

        # self.metric_conv = nn.Sequential(nn.Conv2d(self.num_ch_enc[1] * 2,
        #                                            out_channels=self.num_ch_enc[1],
        #                                            kernel_size=3, stride=1, padding=1),
        #                                  nn.Sigmoid()
        #                                  )

        # w = self.matching_width
        # h = self.matching_height
        # tenHor = torch.linspace(-1.0 + (1.0 / w), 1.0 - (1.0 / w), w).view(
        #     1, 1, 1, -1).expand(-1, -1, h, -1)
        # tenVer = torch.linspace(-1.0 + (1.0 / h), 1.0 - (1.0 / h), h).view(
        #     1, 1, -1, 1).expand(-1, -1, -1, w)
        #
        # self.backwarp_tenGrid = torch.cat([tenHor, tenVer], 1).cuda().repeat(batch_size, 1, 1, 1)
        #
        # self.spatial_attn = Spatial_Attn(64, 64)
        #
        # self.attention = LoFTREncoderLayer(64, 8, 'linear')
        # self.posencoding = PositionEncodingSine(64, max_shape=(48, 160), temp_bug_fix=True)
        #
        # self.f_key = nn.Sequential(
        #     nn.Conv2d(in_channels=64,
        #               out_channels=64,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(inplace=True),
        #     # nn.BatchNorm2d(64)
        # )
        # self.f_query = nn.Sequential(
        #     nn.Conv2d(in_channels=64,
        #               out_channels=64,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(inplace=True),
        #     # nn.BatchNorm2d(64)
        # )
        #
        # self.softmax = torch.nn.Softmax(dim=-1)
        #
        # self.cos = torch.nn.CosineSimilarity(dim=1)

    def compute_depth_bins(self, min_depth_bin, max_depth_bin):
        """Compute the depths bins used to build the cost volume. Bins will depend upon
        self.depth_binning, to either be linear in depth (linear) or linear in inverse depth
        (inverse)"""

        if self.depth_binning == 'inverse':
            self.depth_bins = 1 / np.linspace(1 / max_depth_bin,
                                              1 / min_depth_bin,
                                              self.num_depth_bins)[::-1]  # maintain depth order

        elif self.depth_binning == 'linear':
            self.depth_bins = np.linspace(min_depth_bin, max_depth_bin, self.num_depth_bins)
        else:
            raise NotImplementedError
        self.depth_bins = torch.from_numpy(self.depth_bins).float()

        self.warp_depths = []
        for depth in self.depth_bins:
            depth = torch.ones((1, self.matching_height, self.matching_width)) * depth
            self.warp_depths.append(depth)
        self.warp_depths = torch.stack(self.warp_depths, 0).float()
        if self.is_cuda:
            self.warp_depths = self.warp_depths.cuda()

    def match_features(self, current_feats, lookup_feats, relative_poses, K, invK):
        """Compute a cost volume based on L1 difference between current_feats and lookup_feats.

        We backwards warp the lookup_feats into the current frame using the estimated relative
        pose, known intrinsics and using hypothesised depths self.warp_depths (which are either
        linear in depth or linear in inverse depth).

        If relative_pose == 0 then this indicates that the lookup frame is missing (i.e. we are
        at the start of a sequence), and so we skip it"""

        batch_cost_volume = []  # store all cost volumes of the batch
        cost_volume_masks = []  # store locations of '0's in cost volume for confidence

        for batch_idx in range(len(current_feats)):

            volume_shape = (self.num_depth_bins, self.matching_height, self.matching_width)
            cost_volume = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)
            counts = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)

            # select an item from batch of ref feats
            _lookup_feats = lookup_feats[batch_idx:batch_idx + 1]
            _lookup_poses = relative_poses[batch_idx:batch_idx + 1]

            _K = K[batch_idx:batch_idx + 1]
            _invK = invK[batch_idx:batch_idx + 1]
            world_points = self.backprojector(self.warp_depths, _invK)

            # loop through ref images adding to the current cost volume
            for lookup_idx in range(_lookup_feats.shape[1]):
                lookup_feat = _lookup_feats[:, lookup_idx]  # 1 x C x H x W
                lookup_pose = _lookup_poses[:, lookup_idx]

                # ignore missing images
                if lookup_pose.sum() == 0:
                    continue

                lookup_feat = lookup_feat.repeat([self.num_depth_bins, 1, 1, 1])
                pix_locs = self.projector(world_points, _K, lookup_pose)
                warped = F.grid_sample(lookup_feat, pix_locs, padding_mode='zeros', mode='bilinear',
                                       align_corners=True)

                # mask values landing outside the image (and near the border)
                # we want to ignore edge pixels of the lookup images and the current image
                # because of zero padding in ResNet
                # Masking of ref image border
                x_vals = (pix_locs[..., 0].detach() / 2 + 0.5) * (
                    self.matching_width - 1)  # convert from (-1, 1) to pixel values
                y_vals = (pix_locs[..., 1].detach() / 2 + 0.5) * (self.matching_height - 1)

                edge_mask = (x_vals >= 2.0) * (x_vals <= self.matching_width - 2) * \
                            (y_vals >= 2.0) * (y_vals <= self.matching_height - 2)
                edge_mask = edge_mask.float()

                # masking of current image
                current_mask = torch.zeros_like(edge_mask)
                current_mask[:, 2:-2, 2:-2] = 1.0
                edge_mask = edge_mask * current_mask

                diffs = torch.abs(warped - current_feats[batch_idx:batch_idx + 1]).mean(1) * edge_mask
                # diffs = (1.0 - self.cos(warped, current_feats[batch_idx:batch_idx + 1])) * edge_mask



                # integrate into cost volume
                cost_volume = cost_volume + diffs
                counts = counts + (diffs > 0).float()
            # average over lookup images
            cost_volume = cost_volume / (counts + 1e-7)

            # if some missing values for a pixel location (i.e. some depths landed outside) then
            # set to max of existing values
            missing_val_mask = (cost_volume == 0).float()
            if self.set_missing_to_max:
                cost_volume = cost_volume * (1 - missing_val_mask) + \
                    cost_volume.max(0)[0].unsqueeze(0) * missing_val_mask
            batch_cost_volume.append(cost_volume)
            cost_volume_masks.append(missing_val_mask)

        batch_cost_volume = torch.stack(batch_cost_volume, 0)
        cost_volume_masks = torch.stack(cost_volume_masks, 0)

        return batch_cost_volume, cost_volume_masks

    def feature_extraction(self, image, return_all_feats=False):
        """ Run feature extraction on an image - first 2 blocks of ResNet"""

        image = (image - 0.45) / 0.225  # imagenet normalisation
        feats_0 = self.layer0(image)
        feats_1 = self.layer1(feats_0)
        # feats_2 = self.layer2(feats_1)

        if return_all_feats:
            return [feats_0, feats_1]
        else:
            return feats_1

    def indices_to_disparity(self, indices):
        """Convert cost volume indices to 1/depth for visualisation"""

        batch, height, width = indices.shape
        depth = self.depth_bins[indices.reshape(-1).cpu()]
        disp = 1 / depth.reshape((batch, height, width))
        return disp

    def compute_confidence_mask(self, cost_volume, num_bins_threshold=None):
        """ Returns a 'confidence' mask based on how many times a depth bin was observed"""

        if num_bins_threshold is None:
            num_bins_threshold = self.num_depth_bins
        confidence_mask = ((cost_volume > 0).sum(1) == num_bins_threshold).float()

        return confidence_mask

    def prepare_spatial_attn(self, depth, invK, obs_mask=None):
        """Prepare spatial distance for spatial attention
        """
        # print(depth.shape)
        batch_size, _, h, w = depth.shape
        depth = depth.view(batch_size, 1, h, w)
        # depth = (1. / disp).view(batch_size, 1, h, w)

        mask = torch.ones_like(depth) if obs_mask is None else obs_mask
        depth = depth * mask
        points = self.backprojector_(depth, invK)[:, 0:3, :]
        points_ba = points.unsqueeze(2).expand(batch_size, 3, h * w, h * w)
        points_fr = points.unsqueeze(3).expand(batch_size, 3, h * w, h * w)
        distance = torch.norm(points_fr - points_ba, p=2, dim=1)  # (Bxn) x N x N

        distance = distance.view(batch_size, h * w, h * w)  # B x N x N

        return distance

    def forward(self, current_image, lookup_images, poses, K, invK,
                min_depth_bin=None, max_depth_bin=None, prev_depth=None, prev_pose_inv=None, invK_lr=None, K_lr=None
                ):

        # feature extraction
        self.features = self.feature_extraction(current_image, return_all_feats=True)
        current_feats = self.features[-1]

        # feature extraction on lookup images - disable gradients to save memory
        with torch.no_grad():
            if self.adaptive_bins:
                self.compute_depth_bins(min_depth_bin, max_depth_bin)

            batch_size, num_frames, chns, height, width = lookup_images.shape
            lookup_images = lookup_images.reshape(batch_size * num_frames, chns, height, width)
            lookup_feats = self.feature_extraction(lookup_images,
                                                   return_all_feats=False)
            _, chns, height, width = lookup_feats.shape
            lookup_feats = lookup_feats.reshape(batch_size, num_frames, chns, height, width)

        # cam_points = self.backprojector_(prev_depth, invK)
        # pix_coords = self.projector_(cam_points, K, prev_pose_inv)
        #
        # pix_coords_ = pix_coords.permute(0, 3, 1, 2)
        # flow = pix_coords_ - self.backwarp_tenGrid
        # flow_x = flow[:, 0:1, :, :] * (((self.matching_width) - 1.0) / 2.0)
        # flow_y = flow[:, 1:2, :, :] * (((self.matching_height) - 1.0) / 2.0)
        # flow_final = torch.cat((flow_x, flow_y), 1)
        #
        # with torch.no_grad():
        #     tenMetric = torch.abs(F.grid_sample(F.interpolate(current_image, [48,160], mode='bilinear', align_corners=False).detach(),
        #         pix_coords, mode='bilinear',
        #         padding_mode="border", align_corners=True) - F.interpolate(lookup_images, [48,160], mode='bilinear', align_corners=False)).mean(1, True)
        #
        # # print(prev_depth.shape)
        # # forward warping
        # # tenMetric = self.metric_conv(torch.cat([self.features[-1].detach(), lookup_feats[:,0].detach()], 1))
        # feat_forward = FunctionSoftsplat(
        #     tenInput=lookup_feats[:,0],
        #     tenFlow=flow_final.contiguous() * 1.0, tenMetric=-20.0 * tenMetric,
        #     strType='softmax')
        #
        # # # with torch.no_grad():
        # # depth_forward = FunctionSoftsplat(
        # #     tenInput=prev_depth,
        # #     tenFlow=flow_final.contiguous() * 1.0, tenMetric=-20.0 * tenMetric,
        # #     strType='softmax')
        #
        # # print(depth_forward.shape)
        # feat_forward = feat_forward.reshape(self.features[-1].shape)
        # # depth_forward = depth_forward.reshape(prev_depth.shape)
        #
        # lookup_feats = feat_forward.unsqueeze(1)

        with torch.no_grad():
            # warp features to find cost volume
            cost_volume, missing_mask = \
                self.match_features(current_feats, lookup_feats, poses, K, invK)
            confidence_mask = self.compute_confidence_mask(cost_volume.detach() *
                                                           (1 - missing_mask.detach()))

        # for visualisation - ignore 0s in cost volume for minimum
        viz_cost_vol = cost_volume.clone().detach()
        viz_cost_vol[viz_cost_vol == 0] = 100
        mins, argmin = torch.min(viz_cost_vol, 1)
        lowest_cost = self.indices_to_disparity(argmin)

        # mask the cost volume based on the confidence
        cost_volume *= confidence_mask.unsqueeze(1)
        # if prev_depth is None:
        post_matching_feats = self.reduce_conv(torch.cat([self.features[-1], cost_volume], 1))
            # post_matching_feats = self.reduce_conv(self.features[-1])
        # else:
        #     # with torch.no_grad():
        #     #     cam_points = self.backprojector_(prev_depth.detach(), invK_lr)
        #     #     pix_coords = self.projector_(cam_points, K_lr, prev_pose_inv)
        #     #
        #     #     pix_coords_ = pix_coords.permute(0, 3, 1, 2)
        #     #     flow = pix_coords_ - self.backwarp_tenGrid
        #     #     flow_x = flow[:, 0:1, :, :] * (((self.matching_width) - 1.0) / 2.0)
        #     #     flow_y = flow[:, 1:2, :, :] * (((self.matching_height) - 1.0) / 2.0)
        #     #     flow_final = torch.cat((flow_x, flow_y), 1)
        #     #
        #     #     tenMetric = torch.abs(F.grid_sample(F.interpolate(current_image, [48,160], mode='bilinear', align_corners=False).detach(),
        #     #         pix_coords, mode='bilinear',
        #     #         padding_mode="border", align_corners=True) - F.interpolate(lookup_images, [48,160], mode='bilinear', align_corners=False)).mean(1, True)
        #     #
        #     #     # print(prev_depth.shape)
        #     #     # forward warping
        #     #     # tenMetric = self.metric_conv(torch.cat([self.features[-1].detach(), lookup_feats[:,0].detach()], 1))
        #     #     feat_forward = FunctionSoftsplat(
        #     #         tenInput=lookup_feats[:,0],
        #     #         tenFlow=flow_final.contiguous() * 1.0, tenMetric=-20.0 * tenMetric,
        #     #         strType='softmax')
        #     #
        #     #     with torch.no_grad():
        #     #         depth_forward = FunctionSoftsplat(
        #     #             tenInput=prev_depth,
        #     #             tenFlow=flow_final.contiguous() * 1.0, tenMetric=-20.0 * tenMetric,
        #     #             strType='softmax')
        #     #
        #     #     # print(depth_forward.shape)
        #     #     feat_forward = feat_forward.reshape(self.features[-1].shape)
        #     #     depth_forward = depth_forward.reshape(prev_depth.shape)
        #
        #     with torch.no_grad():
        #         distance_pre = self.prepare_spatial_attn(prev_depth, invK)
        #     #     # distance_cur = self.prepare_spatial_attn(depth_forward.detach(), invK_lr)
        #     #
        #     # # feat_pre = F.interpolate(lookup_feats[:,0], [24, 80], mode='bilinear', align_corners=False)
        #     spa_att_pre, _ = self.spatial_attn(lookup_feats[:,0], distance_pre)
        #     # # spa_att_pre = lookup_feats[:,0] + spa_att_pre#F.interpolate(spa_att_pre, [48, 160], mode='bilinear', align_corners=False)
        #     # # # feat_cur = F.interpolate(current_feats, [24, 80], mode='bilinear', align_corners=False)
        #     # # spa_att_cur, _ = self.spatial_attn(current_feats, distance_cur)
        #     # # spa_att_cur = current_feats + spa_att_cur#F.interpolate(spa_att_cur, [48, 160], mode='bilinear', align_corners=False)
        #
        #     spa_att_pre_ = rearrange(self.posencoding((spa_att_pre)), 'n c h w -> n (h w) c')
        #     spa_att_cur_ = rearrange(self.posencoding((current_feats)), 'n c h w -> n (h w) c')
        #     # with torch.no_grad():
        #     attention = self.attention(spa_att_cur_, spa_att_pre_)
        #     # attention_ = attention / attention.shape[-1] ** .5
        #     # attention = self.softmax(attention)
        #     # with torch.no_grad():
        #     # attention_back = self.attention(spa_att_pre_, spa_att_cur_)
        #     # attention_back = attention_back / attention_back.shape[-1] ** .5
        #
        #     # self_attention = self.attention(spa_att_cur_, spa_att_cur_)
        #     # self_attention_ = self_attention / self_attention.shape[-1] ** .5
        #     # with torch.no_grad():
        #     #     sim_matrix = torch.einsum("nlc,nsc->nls", self_attention_, attention_)# * 0.01
        #     #     conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        #
        #     # attention_out = torch.matmul(conf_matrix, attention).contiguous().view(current_feats.shape)
        #     # attention = attention.reshape(self.features[-1].shape)
        #     # attention_back = attention_back.reshape(self.features[-1].shape)
        #
        #     post_matching_feats = self.reduce_conv(torch.cat([current_feats, attention.reshape(current_feats.shape)], 1))
        #     # post_matching_feats = self.features[-1] + attention_out
        #     # post_matching_feats = current_feats + attention.reshape(current_feats.shape)
        # #
        self.features.append(self.layer2(post_matching_feats))
        # self.features.append(self.layer3(post_matching_feats))
        self.features.append(self.layer3(self.features[-1]))
        self.features.append(self.layer4(self.features[-1]))

        return self.features, lowest_cost, confidence_mask#, depth_forward, tenMetric

    def cuda(self):
        super().cuda()
        self.backprojector.cuda()
        self.projector.cuda()
        # self.backprojector_.cuda()
        # self.projector_.cuda()
        self.is_cuda = True
        if self.warp_depths is not None:
            self.warp_depths = self.warp_depths.cuda()

    def cpu(self):
        super().cpu()
        self.backprojector.cpu()
        self.projector.cpu()
        self.is_cuda = False
        if self.warp_depths is not None:
            self.warp_depths = self.warp_depths.cpu()

    def to(self, device):
        if str(device) == 'cpu':
            self.cpu()
        elif str(device) == 'cuda':
            self.cuda()
        else:
            raise NotImplementedError


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1, **kwargs):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)
            ### uncomment to run the 12-channel version
            # weight = self.encoder.conv1.weight.clone()
            # self.encoder.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # with torch.no_grad():
            #     self.encoder.conv1.weight[:, :3] = weight
            #     self.encoder.conv1.weight[:, 3] = self.encoder.conv1.weight[:, 0]
            ###

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):

        self.features = []
        # print("RESNET ENCODER: ", input_image) # 0-1 interval
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
