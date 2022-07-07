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

if __name__ == '__main__':
    # rin = torch.rand(2,9,320,480)
    # enc = NormalsEncoder(0)
    # out = enc(rin)
    # print(out.shape)

    fake_imgs = torch.rand(2,3,320,480)
    enc = XOLPEncoder(0)
    out = enc(fake_imgs)
    print(out.shape)
