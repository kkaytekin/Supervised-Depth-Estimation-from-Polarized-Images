from ..normals_vec import rho_diffuse, rho_spec, calc_normals

import numpy as np

import torch
import torch.nn as nn

from .transunet.transunet import AttentionModule

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size, downsampling_mode, padding, dropout_p):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.nonlin = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=dropout_p)
        self.downsampling_mode = downsampling_mode
        if downsampling_mode == 'maxpool':
            self.pool = nn.MaxPool2d(2)
        elif downsampling_mode == 'avgpool':
            self.pool = nn.AvgPool2d(2)
        elif downsampling_mode == 'stride2':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.nonlin(out)
        if self.downsampling_mode not in ['stride2', 'none']:
            out = self.pool(out)
        out = self.dropout(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding, dropout):
        super(ResidualBlock, self).__init__()

        self.conv1 = ConvBlock(channels, channels, kernel_size, 'none', padding, dropout)
        self.conv2 = ConvBlock(channels, channels, kernel_size, 'none', padding, dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x

# The encoder used to extract XOLP and normals features
class ShallowEncoder(nn.Module):
    def __init__(self, mode, in_channels=2, dropout_rate=0.5):
        super(ShallowEncoder, self).__init__()
        self.in_channels = in_channels
        self.mode = mode
        self.Conv1 = ConvBlock(self.in_channels, 64, 7, 'stride2', 3, dropout_rate)
        self.ResBlock1 = ResidualBlock(64, 3, 1, dropout_rate)
        self.Conv2 = ConvBlock(64, 64, 5, 'maxpool', 2, dropout_rate)
        self.ResBlock2 = ResidualBlock(64, 3, 1, dropout_rate)
        self.Conv3 = ConvBlock(64, 64, 5, 'maxpool', 2, dropout_rate)
        self.ResBlock3 = ResidualBlock(64, 3, 1, dropout_rate)

    def forward(self, x):
        # Input:
        # xolp(C=2) or normals(C=9): BXCxHxW
        # Output:
        # extracted features: Bx64xHxW
        x = self.normalizeInput(x, self.mode)
        out = self.Conv1(x)  # 160x240
        out = self.ResBlock1(out)
        out = self.Conv2(out)  # 80x120
        out = self.ResBlock2(out)
        out = self.Conv3(out)  # 40x60
        out = self.ResBlock3(out)
        return out  # 40x60

    @staticmethod
    def normalizeInput(x, mode):
        # precomputed normalization
        if mode == 'XOLP':
            return (x - 0.08693199701957657) / 0.44430732785457433
        if mode == 'normals':
            return x
        if mode == 'RGB':
            return (x - 0.45) / 0.225

class ShallowNormalsEncoder(ShallowEncoder):
    def __init__(self, in_channels=9, dropout_rate=0.1):
        super(ShallowNormalsEncoder, self).__init__('normals', in_channels, dropout_rate)

    def forward(self, x):
        # Input:
        # DOLP and AOLP: BX2xHxW
        # Output:
        # extracted features: Bx64xWxH

        normals = self.get_normals(x).float()
        out = super(ShallowNormalsEncoder, self).forward(normals)
        return out

    @staticmethod
    def get_normals(x, n=1.5):
        # input: XOLP: Bx2xHxW
        # output: 3 concatenated normals: Bx9xHxW
        rho = x[:, 0, :, :].squeeze(dim=1)  # BxHxW
        phi = x[:, 1, :, :].squeeze(dim=1)  # BxHxW

        theta_diff = rho_diffuse(rho, n)  # BxHxW
        theta_spec1, theta_spec2 = rho_spec(rho, n)  # BxHxW
        N_diff = calc_normals(phi, theta_diff)  # Bx3xHxW
        N_spec1 = calc_normals(phi + np.pi / 2, theta_spec1)  # Bx3xHxW
        N_spec2 = calc_normals(phi + np.pi / 2, theta_spec2)  # Bx3xHxW
        N_cat = torch.cat((N_diff, N_spec1, N_spec2), dim=1)  # Bx9xHxW

        return N_cat

# Used to combine RGB, and optionally XOLP and normals features.
class JointEncoder(nn.Module):
    def __init__(self, dropout_rate=0.0, include_normals=True,
                 include_xolp=True):
        super(JointEncoder, self).__init__()
        additional_ch = 0
        if include_normals:
            additional_ch += 64
        if include_xolp:
            additional_ch += 64

        self.fc1 = ConvBlock(128+additional_ch,256,1,'none',0,dropout_rate)
        self.fc2 = ConvBlock(256,128,1,'none',0,dropout_rate)
        # After 1x1 conv, use attention to fuse encoder features
        self.AttentionBlock = AttentionModule(residual_num=1,
                                              dim=128,
                                              dropout=dropout_rate,
                                              skip_res=False)
        # self.ResBlock1 = ResidualBlock(128,3,1,dropout_rate)
        # self.ResBlock2 = ResidualBlock(128,3,1,dropout_rate)
        self.Conv1 = ConvBlock(128,256,5,'maxpool',2,dropout_rate)
        self.ResBlock3 = ResidualBlock(256,3,1,dropout_rate)
        self.ResBlock4 = ResidualBlock(256,3,1,dropout_rate)
        self.Conv2 = ConvBlock(256,512,5,'maxpool',2,dropout_rate)
        self.ResBlock5 = ResidualBlock(512,3,1,dropout_rate)
        self.ResBlock6 = ResidualBlock(512,3,1,dropout_rate)

    def forward(self, rgb_feats, xolp_feats=None, normals_feats=None):
        # Input:
        # rgb_feats: (B,128,40,60),
        # xolp_feats: (B,64,40,60),
        # normals_feats: (B,64,40,60)
        out = []
        if xolp_feats is None:
            if normals_feats is None:
                feats = rgb_feats
            else:
                feats = torch.cat((rgb_feats, normals_feats), dim=1)
        else:
            if normals_feats is None:
                feats = torch.cat((rgb_feats, xolp_feats), dim=1)
            else:
                feats = torch.cat((rgb_feats, xolp_feats, normals_feats), dim=1)
        feats = self.fc1(feats)
        feats = self.fc2(feats)
        feats = self.AttentionBlock(feats)
        # feats = self.ResBlock1(feats)
        # feats = self.ResBlock2(feats)
        feats = self.Conv1(feats)
        feats = self.ResBlock3(feats)
        feats = self.ResBlock4(feats)
        out.append(feats)  # (256,20,30)
        feats = self.Conv2(feats)
        feats = self.ResBlock5(feats)
        feats = self.ResBlock6(feats)
        out.append(feats)  # (512,10,15)
        return out
