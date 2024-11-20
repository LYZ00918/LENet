import torch
import torch.nn as nn
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from toolbox.backbone.mobilenet.MobileNetv2 import mobilenet_v2
from toolbox.models.lyz.paper4.DCT import MultiSpectralAttentionLayer
from toolbox.models.lyz.paper4.Linear_attention import LinearAttention
from torchvision.ops import DeformConv2d
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x1 = self.conv1(x)
        return self.sigmoid(x1)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
                                          stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class Bottleneck(nn.Module):
    expansion = 2

    def \
            __init__(self, inplanes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, inplanes*self.expansion)
        self.bn1 = nn.BatchNorm2d(inplanes*self.expansion)
        self.conv2 = conv3x3(inplanes*self.expansion, inplanes*self.expansion, stride)
        self.bn2 = nn.BatchNorm2d(inplanes*self.expansion)
        self.conv3 = conv1x1(inplanes*self.expansion, inplanes)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class DIM(nn.Module):

    def __init__(self,inchannels,outchannels,inputsolution):
        super(DIM,self).__init__()

        self.conv2 = nn.Conv2d(inchannels,outchannels,kernel_size=3,padding=1,stride=1)
        self.conv3 = nn.Conv2d(inchannels,outchannels,kernel_size=3,padding=1,stride=1)
        self.block = Bottleneck(inchannels)
        self.DCT = MultiSpectralAttentionLayer(outchannels, inputsolution[0], inputsolution[1])
        # self.LA = LinearAttention(outchannels, inputsolution, ratio)
        self.sa = SpatialAttention()
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
    def forward(self,dep,rgb):
        dep = self.conv2(dep)
        # dep = self.maxpool(dep)
        rgb = self.block(rgb)
        # print(rgb.shape)
        # rgb = self.conv3(rgb)
        dep_dct = self.DCT(dep)
        fuse = self.sa(rgb) * dep_dct

        return fuse

def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x

class direction_conv(nn.Module):
    def __init__(self,channels):
        super(direction_conv,self).__init__()
        self.h_conv1 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
        self.h_conv2 = nn.Conv2d(channels, channels, (1, 5), padding=(0, 2))
        self.h_conv3 = nn.Conv2d(channels, channels, (1, 7), padding=(0, 3))
        self.h_conv4 = nn.Conv2d(channels, channels, (1, 9), padding=(0, 4))
        self.w_conv1 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0))
        self.w_conv2 = nn.Conv2d(channels, channels, (5, 1), padding=(2, 0))
        self.w_conv3 = nn.Conv2d(channels, channels, (7, 1), padding=(3, 0))
        self.w_conv4 = nn.Conv2d(channels, channels, (9, 1), padding=(4, 0))

    def forward(self,x1,x2,x3,x4):
        x1 = self.w_conv1(self.h_conv1(x1))

        x2 = self.w_conv2(self.h_conv2(x2))

        x3 = self.w_conv3(self.h_conv3(x3))

        x4 = self.w_conv4(self.h_conv4(x4))
        # print(x4.shape)
        return x1,x2,x3,x4
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FFM(nn.Module):
    def __init__(self,in_channel,out_channel,ratio=16):
        super(FFM,self).__init__()
        self.conv1 = BasicConv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1)
        self.conv2 = BasicConv2d(out_channel*2, in_channel, kernel_size=3, padding=1, stride=1)
        self.dir_conv = direction_conv(out_channel//4)
        self.ca = ChannelAttention(out_channel*2,ratio=ratio)
        self.sa = SpatialAttention()
        # self.R_conv2 = nn.Conv2d(out_channel//4, out_channel//4, 3, padding=3, dilation=3)
        # self.R_conv3 = nn.Conv2d(out_channel//4, out_channel//4, 3, padding=5, dilation=5)
        # self.R_conv4 = nn.Conv2d(out_channel//4, out_channel//4, 3, padding=7, dilation=7)

    def forward(self,x):
        x11 = self.conv1(x)
        b, c, h, w = x11.shape
        cc = c//4
        x = channel_shuffle(x11,4)
        x1 = x[:, :cc, :, :]
        x2 = x[:, cc:cc*2, :, :]
        x3 = x[:, cc*2:cc*3, :, :]
        x4 = x[:, cc*3:, :, :]
        x1,x2,x3,x4 = self.dir_conv(x1,x2,x3,x4)

        # x4_1 = self.R_conv2(x4)
        # x4_2 = self.R_conv3(x4)
        # x4_3 = self.R_conv4(x4)
        # x4_1 = self.sa(x4_1) * x4_1
        # x4_2 = self.sa(x4_2) * x4_2
        # x4_3 = self.sa(x4_3) * x4_3
        # x4 = x4_1 + x4_2 + x4_3
        x5 = self.sa(x11) * x11
        # print(x5.shape)
        F = torch.cat((x1,x2,x3,x4,x5),dim=1)
        return self.conv2(self.ca(F)*F)




class L_W(nn.Module):
    def __init__(self):
        super(L_W, self).__init__()
        self.mobilenetv2_rgb = mobilenet_v2()
        self.mobilenetv2_dep = mobilenet_v2()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_1 = nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.conv11 = nn.Conv2d(24,32,kernel_size=3,padding=1)
        self.conv22 = nn.Conv2d(32,24,kernel_size=3,padding=1)

        self.DCT2 = MultiSpectralAttentionLayer(32, 64, 64)
        self.DCT3 = MultiSpectralAttentionLayer(32, 32, 32)
        self.DCT4 = MultiSpectralAttentionLayer(160, 16, 16)
        ##########   DECODER    ###########
        self.decoder4 = DecoderBlock(320, 160)
        self.decoder3 = DecoderBlock(160, 32)
        self.decoder2 = DecoderBlock(32, 24)
        self.decoder1 = DecoderBlock(24, 16)
        self.decoder0 = DecoderBlock(16, 10)

        self.S4 = nn.Conv2d(160, 6, 3, stride=1, padding=1)
        self.S3 = nn.Conv2d(32, 6, 3, stride=1, padding=1)
        self.S2 = nn.Conv2d(24, 6, 3, stride=1, padding=1)
        self.S1 = nn.Conv2d(16, 6, 3, stride=1, padding=1)
        self.S0 = nn.Conv2d(10, 6, 3, stride=1, padding=1)

        # self.DIM0 = DIM(16, 16, (64, 64))
        self.DIM1 = DIM(24, 32, (64, 64))
        # self.conv_A = nn.Conv2d(24, 32, kernel_size=1)
        self.conv_B = nn.Conv2d(32, 24, kernel_size=1)
        self.DIM2 = DIM(32, 32, (32, 32))
        self.DIM3 = DIM(160, 160, (16, 16))
        # self.DIM4 = DIM(320, 320, (8, 8))
        self.FFM5 = FFM(320,160,ratio=16)
        self.FFM4 = FFM(160,80,ratio=16)
        self.FFM3 = FFM(32,96,ratio=16)
        self.FFM2 = FFM(24,72,ratio=16)
        self.FFM1 = FFM(16,48,ratio=16)
    def forward(self, rgb,dep):
        x0 = self.mobilenetv2_rgb.features[0:2](rgb)
        with torch.no_grad():
            d00 = self.mobilenetv2_dep.features[0:2](dep)


        x1 = self.mobilenetv2_rgb.features[2:4](x0)
        with torch.no_grad():
            d11 = self.mobilenetv2_dep.features[2:4](d00)

        d11 = self.DIM1(d11, x1)
        d11 = self.conv_B(d11)
        y1 = x1 + d11

        x2 = self.mobilenetv2_rgb.features[4:7](y1)
        with torch.no_grad():
            d22 = self.mobilenetv2_dep.features[4:7](d11)

        d22 = self.DIM2(d22, x2)
        y2 = x2 + d22

        x3 = self.mobilenetv2_rgb.features[7:17](y2)
        with torch.no_grad():
            d33 = self.mobilenetv2_dep.features[7:17](d22)

        d33 = self.DIM3(d33,x3)
        y3 = x3 + d33

        x4 = self.mobilenetv2_rgb.features[17:18](y3)
        # with torch.no_grad():
        #     d44 = self.mobilenetv2_dep.features[17:18](d33)
        #
        # d44 = self.DIM4(d44, x4)
        # y4 = x4 + d44

        x44 = self.FFM5(x4)
        x33 = self.FFM4(x3)
        x22 = self.FFM3(x2)
        x11 = self.FFM2(x1)
        x00 = self.FFM1(x0)


        ##########   DECODER    ###########
        d4 = self.decoder4(x44)
        z4 = self.S4(d4)

        d3 = self.decoder3(d4 + x33)
        z3 = self.S3(d3)

        d2 = self.decoder2(d3 + x22)
        z2 = self.S2(d2)

        d1 = self.decoder1(d2 + x11)
        z1 = self.S1(d1)

        d0 = self.decoder0(d1 + x00)
        z0 = self.S0(d0)


        return z0, z1, z2, z3, z4
        # return z0, z1, z2, z3, z4,x0,x1,x2,x3,x4,d00,d11,d22,d33,d44




if __name__ == '__main__':
    rgb = torch.randn(10, 3, 256, 256)
    dep = torch.randn(10, 3, 256, 256)
    net = L_W()
    out = net(rgb, dep)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(out[3].shape)
    print(out[4].shape)

