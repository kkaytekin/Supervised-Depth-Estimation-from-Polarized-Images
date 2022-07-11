from ..normals_vec import rho_diffuse, rho_spec, calc_normals

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# A very basic & light weight encoder for proof of concept
# no skip connections.

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ReLU
    """
    def __init__(self, in_channels, out_channels,kernel_size,downsampling_mode,padding,dropout_p):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size,padding=padding)
        self.nonlin = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=dropout_p)
        self.downsampling_mode = downsampling_mode
        if downsampling_mode == 'maxpool':
            self.pool = nn.MaxPool2d(2)
        elif downsampling_mode == 'avgpool':
            self.pool = nn.AvgPool2d(2)
        elif downsampling_mode == 'stride2':
            self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size,stride=2,padding=padding)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.nonlin(out)
        if self.downsampling_mode not in ['stride2', 'none']:
            out = self.pool(out)
        out = self.dropout(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self,channels,kernel_size,padding,dropout):
        super(ResidualBlock, self).__init__()

        self.conv1 = ConvBlock(channels,channels,kernel_size,'none',padding,dropout)
        self.conv2 = ConvBlock(channels,channels,kernel_size,'none',padding,dropout)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x

class XOLPEncoder(nn.Module):
    def __init__(self, dropout_rate, in_channels = 2):
        super(XOLPEncoder, self).__init__()
        self.in_channels = in_channels
        self.Conv1 = ConvBlock(self.in_channels,64,7,'stride2',3,dropout_rate)
        self.Conv2 = ConvBlock(64,64,3,'maxpool',1,dropout_rate)
        self.Conv3 = ConvBlock(64,128,5,'stride2',2,dropout_rate)
        self.Conv4 = ConvBlock(128,128,3,'stride2',1,dropout_rate)
        self.Conv5 = ConvBlock(128,128,3,'none',1,dropout_rate)
        self.Conv6 = ConvBlock(128,128,3,'none',1,dropout_rate)
        self.Conv7 = ConvBlock(128,256,3,'maxpool',1,dropout_rate)

    def forward(self, input):
        ## Input:
        # polarized_images: 4 RGB polarized images: BX12xHxW
        ## Output:
        # extracted features: Bx128xWxH

        # fake_normals = input[:,:9,:,:].float()
        # out = self.Conv1(fake_normals)
        #xolp = self.get_debug_XOLP(input)
        input = (input - 0.08693199701957657) / 0.44430732785457433  # precomputed using 1 XOLP exammple from each sequence in train folder

        out = self.Conv1(input)
        out = self.Conv2(out)
        out = self.Conv3(out)
        out_res = self.Conv4(out)
        # one residual block
        out = self.Conv5(out_res)
        out = self.Conv6(out)
        out = self.Conv7(out+out_res)
        return out

    def get_debug_XOLP(self, input):
        # input: 4 RGB polarised images + xolp: BX14xHxW
        # output: DOLP and AOLP: Bx2xHxW
        # todo: either implement or get from dataloader
        return torch.rand(2,2,320,480)

class NormalsEncoder(nn.Module):
    def __init__(self, dropout_rate, in_channels = 9):
        super(NormalsEncoder, self).__init__()
        self.in_channels = in_channels
        self.Conv1 = ConvBlock(self.in_channels,64,7,'stride2',3,dropout_rate)
        self.Conv2 = ConvBlock(64,64,3,'maxpool',1,dropout_rate)
        self.Conv3 = ConvBlock(64,128,5,'stride2',2,dropout_rate)
        self.Conv4 = ConvBlock(128,128,3,'stride2',1,dropout_rate)
        self.Conv5 = ConvBlock(128,256,3,'maxpool',1,dropout_rate)

    def forward(self, input):
        ## Input:
        # polarized_images: 4 RGB polarized images: BX12xHxW
        ## Output:
        # extracted features: Bx128xWxH

        # fake_normals = input[:,:9,:,:].float()
        # out = self.Conv1(fake_normals)

        normals = self.get_normals(input).float()
        out = self.Conv1(normals)
        out = self.Conv2(out)
        out = self.Conv3(out)
        out = self.Conv4(out)
        out = self.Conv5(out)
        return out

    def get_normals(self, input, n=1.5):
        # input: 4 RGB polarised images + xolp: BX14xHxW
        # output: 3 concatenated normals: Bx9xHxW
        # todo: address the following issue
        # How is it possible that no error is returned?
        # Input has 12 channels for now so two lines below should not work
        # It looks like this function is not used at all
        # Is the encoder fully integrated?
        rho = input[:, 0, :, :].squeeze(dim=1)  # BxHxW
        phi = input[:, 1, :, :].squeeze(dim=1)  # BxHxW

        theta_diff = rho_diffuse(rho, n)  # BxHxW
        theta_spec1, theta_spec2 = rho_spec(rho, n)  # BxHxW
        N_diff = calc_normals(phi, theta_diff)  # Bx3xHxW
        N_spec1 = calc_normals(phi + np.pi / 2, theta_spec1)  # Bx3xHxW
        N_spec2 = calc_normals(phi + np.pi / 2, theta_spec2)  # Bx3xHxW
        N_cat = torch.cat((N_diff, N_spec1, N_spec2), dim=1)  # Bx9xHxW

        return N_cat

class ShallowEncoder(nn.Module):
    def __init__(self, mode, in_channels = 2, dropout_rate=0.5):
        super(ShallowEncoder, self).__init__()
        self.in_channels = in_channels
        self.mode = mode
        self.Conv1 = ConvBlock(self.in_channels,64,7,'stride2',3,dropout_rate)
        # self.Maxpool1 = nn.MaxPool2d(2)
        self.ResBlock1 = ResidualBlock(64,3,1,dropout_rate)
        self.Conv2 = ConvBlock(64,64,5,'stride2',2,dropout_rate)
        self.ResBlock2 = ResidualBlock(64,3,1,dropout_rate)
        self.Conv3 = ConvBlock(64,64,5,'stride2',2,dropout_rate)
        self.ResBlock3 = ResidualBlock(64,3,1,dropout_rate)

    def forward(self, input):
        ## Input:
        # xolp(C=2) or normals(C=9): BXCxHxW
        ## Output:
        # extracted features: Bx64xHxW
        input = self.normalizeInput(input, self.mode)
        out = self.Conv1(input) # 160x240
        # out = self.Maxpool1(out)
        out = self.ResBlock1(out)
        out = self.Conv2(out) #80x120
        out = self.ResBlock2(out)
        out = self.Conv3(out) #40x60
        out = self.ResBlock3(out)
        return out # 40x60

    def normalizeInput(self, input, mode):
        if mode == 'XOLP':
            # precomputed using 1 XOLP exammple from each sequence in train folder
            return (input - 0.08693199701957657) / 0.44430732785457433
        if mode == 'normals':
            #todo: implement
            return input
        if mode == 'RGB': # not used, for completeness there
            return (input - 0.45) / 0.225

class ShallowNormalsEncoder(ShallowEncoder):
    def __init__(self, in_channels = 9, dropout_rate = 0.1):
        super(ShallowNormalsEncoder, self).__init__('normals',in_channels,dropout_rate)

    def forward(self, input):
        ## Input:
        # polarized_images: 4 RGB polarized images: BX12xHxW
        ## Output:
        # extracted features: Bx128xWxH

        # fake_normals = input[:,:9,:,:].float()
        # out = self.Conv1(fake_normals)

        normals = self.get_normals(input).float()
        out = super(ShallowNormalsEncoder, self).forward(normals)

        return out

    def get_normals(self, input, n=1.5):
        # input: 4 RGB polarised images + xolp: BX14xHxW
        # output: 3 concatenated normals: Bx9xHxW
        rho = input[:, 0, :, :].squeeze(dim=1)  # BxHxW
        phi = input[:, 1, :, :].squeeze(dim=1)  # BxHxW

        theta_diff = rho_diffuse(rho, n)  # BxHxW
        theta_spec1, theta_spec2 = rho_spec(rho, n)  # BxHxW
        N_diff = calc_normals(phi, theta_diff)  # Bx3xHxW
        N_spec1 = calc_normals(phi + np.pi / 2, theta_spec1)  # Bx3xHxW
        N_spec2 = calc_normals(phi + np.pi / 2, theta_spec2)  # Bx3xHxW
        N_cat = torch.cat((N_diff, N_spec1, N_spec2), dim=1)  # Bx9xHxW

        return N_cat

class JointEncoder(nn.Module):
    def __init__(self, dropout_rate = 0.0):
        super(JointEncoder, self).__init__()
        self.fc1 = ConvBlock(256,256,1,'none',0,dropout_rate)
        self.fc2 = ConvBlock(256,128,1,'none',0,dropout_rate)
        self.ResBlock1 = ResidualBlock(128,3,1,dropout_rate)
        self.ResBlock2 = ResidualBlock(128,3,1,dropout_rate)
        self.ResBlock3 = ResidualBlock(128,3,1,dropout_rate)
        self.Conv1 = ConvBlock(128,256,5,'stride2',2,dropout_rate)
        self.ResBlock4 = ResidualBlock(256,3,1,dropout_rate)
        self.ResBlock5 = ResidualBlock(256,3,1,dropout_rate)
        self.ResBlock6 = ResidualBlock(256,3,1,dropout_rate)
        self.Conv2 = ConvBlock(256,512,5,'stride2',2,dropout_rate)
        self.ResBlock7 = ResidualBlock(512,3,1,dropout_rate)
        self.ResBlock8 = ResidualBlock(512,3,1,dropout_rate)
        self.ResBlock9 = ResidualBlock(512,3,1,dropout_rate)

    def forward(self,rgb_feats,xolp_feats,normals_feats):
        ## Input:
        # rgb_feats: (B,128,40,60),
        # xolp_feats: (B,64,40,60),
        # normals_feats: (B,64,40,60)
        out = []

        feats = torch.cat((rgb_feats,xolp_feats,normals_feats), dim = 1)
        feats = self.fc1(feats)
        feats = self.fc2(feats)
        feats = self.ResBlock1(feats)
        feats = self.ResBlock2(feats)
        feats = self.ResBlock3(feats)
        feats = self.Conv1(feats)
        feats = self.ResBlock4(feats)
        feats = self.ResBlock5(feats)
        feats = self.ResBlock6(feats)
        out.append(feats) # (256,20,30)
        feats = self.Conv2(feats)
        feats = self.ResBlock7(feats)
        feats = self.ResBlock8(feats)
        feats = self.ResBlock9(feats)
        out.append(feats) # (512,10,15)
        return out


if __name__ == '__main__':
    # rin = torch.rand(2,9,320,480)
    # enc = NormalsEncoder(0)
    # out = enc(rin)
    # print(out.shape)

    fake_imgs = torch.rand(2,3,320,480)
    enc = XOLPEncoder(0)
    out = enc(fake_imgs)
    print(out.shape)
