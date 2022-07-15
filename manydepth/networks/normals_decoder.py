import torch.nn as nn

from manydepth.layers import upsample
from manydepth.networks.pre_encoders import ConvBlock, ResidualBlock

class NormalsDecoder(nn.Module):
    def __init__(self, dropout_rate = 0.0):
        super(NormalsDecoder, self).__init__()

        self.conv1 = ConvBlock(64,32,3,'none',1,dropout_rate)
        self.ResBlock2 = ResidualBlock(32,3,1,dropout_rate)
        self.conv2 = ConvBlock(32,16,3,'none',1,dropout_rate)
        self.ResBlock3 = ResidualBlock(16,3,1,dropout_rate)
        self.conv3 = ConvBlock(16,8,3,'none',1,dropout_rate)
        self.ResBlock4 = ResidualBlock(8,3,1,dropout_rate)
        self.conv4 = ConvBlock(8,3,3,'none',1,dropout_rate)

        self.tanh = nn.Tanh()



    def forward(self, x ):
        # Input: x: Normals features B,64,40,60
        # Output: Normals_pred: B,3,320,480

        # on the final step, normalize output
        x = self.conv1(x)
        x = upsample(x) # 80x120
        x = self.ResBlock2(x)
        x = self.conv2(x)
        x = upsample(x) # 160x240
        x = self.ResBlock3(x)
        x = self.conv3(x)
        x = upsample(x) # 320x480
        x = self.ResBlock4(x)
        x = self.conv4(x)
        x = self.tanh(x)

        return x
