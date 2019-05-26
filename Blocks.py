import torch.nn as nn
import torch
import math


class Conv2DBlockReverse(nn.Module):

    def __init__(self, num_in, num_filter,
                 kernel=(1, 1), pad=(0, 0), stride=(1, 1), g=1, bias=False):
        super(Conv2DBlockReverse, self).__init__()

        self.bn = nn.BatchNorm2d(num_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_in, num_filter, kernel_size=kernel, padding=pad,
                              stride=stride, groups=g, bias=bias)

    def forward(self, x):
        h = self.relu(self.bn(x))
        h = self.conv(h)
        return h


class Conv2DBlock(nn.Module):

    def __init__(self, num_in, num_filter,
                 kernel=(1, 1), pad=(0, 0), stride=(1, 1), g=1, bias=False):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(num_in, num_filter, kernel_size=kernel, padding=pad,
                              stride=stride, groups=g, bias=bias)
        self.bn = nn.BatchNorm2d(num_filter)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.conv(x)
        h = self.relu(self.bn(h))

        return h


class Conv3DBlock(nn.Module):

    def __init__(self, num_in, num_filter,
                 kernel=(1, 1, 1), pad=(0, 0, 0), stride=(1, 1, 1), g=1, bias=False):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(num_in, num_filter, kernel_size=kernel, padding=pad,
                              stride=stride, groups=g, bias=bias)
        self.bn = nn.BatchNorm3d(num_filter)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.conv(x)
        h = self.relu(self.bn(h))

        return h
class DeConv2DBlock(nn.Module):

    def __init__(self, num_in, num_filter,
                 kernel=(1, 1), pad=(0, 0), stride=(1, 1), g=1, bias=False):
        super(DeConv2DBlock, self).__init__()
        #print("stride:",stride,"pad:",pad,"kernel:",kernel)
        self.conv = nn.ConvTranspose2d(in_channels=num_in, out_channels=num_filter, kernel_size=kernel, stride=stride,padding=pad,groups=g)
        self.bn = nn.BatchNorm2d(num_filter)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.conv(x)
        h = self.relu(self.bn(h))

        return h


class MF_UNIT2D(nn.Module):

    def __init__(self, num_in, num_mid, num_out, g=1, stride=(1, 1), first_block=False, use_3d=True):
        super(MF_UNIT2D, self).__init__()
        num_ix = int(num_mid / 4)
        kt, pt = (3, 1) if use_3d else (1, 0)
        # prepare input
        self.conv_i1 = Conv2DBlock(num_in=num_in, num_filter=num_ix, kernel=(1, 1), pad=(0, 0))
        self.conv_i2 = Conv2DBlock(num_in=num_ix, num_filter=num_in, kernel=(1, 1), pad=(0, 0))
        # main part
        self.conv_m1 = Conv2DBlock(num_in=num_in, num_filter=num_mid, kernel=(3, 3), pad=(1, 1), stride=stride, g=g)
        if first_block:
            self.conv_m2 = Conv2DBlock(num_in=num_mid, num_filter=num_out, kernel=(1, 1), pad=(0, 0))
        else:
            self.conv_m2 = Conv2DBlock(num_in=num_mid, num_filter=num_out, kernel=(3, 3), pad=(1, 1), g=g)
        # adapter
        if first_block:
            self.conv_w1 = Conv2DBlock(num_in=num_in, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=stride)

    def forward(self, x):

        h = self.conv_i1(x)
        x_in = x + self.conv_i2(h)

        h = self.conv_m1(x_in)
        h = self.conv_m2(h)

        if hasattr(self, 'conv_w1'):
            x = self.conv_w1(x)

        return h + x


########################################################################################
########This model will have more layers, but each layer will have less compute#########
########################################################################################

class DilatedFrameBlock(nn.Module):
    def __init__(self, kernel_size, ch_in, ch_out, stride=1, groups=1):
        super(DilatedFrameBlock, self).__init__()
        padding_base = int(kernel_size / 2)

        self.conv1d0 = nn.Conv2d(ch_in, ch_out, kernel_size=(1, kernel_size), padding=(0, padding_base * 2),
                                 stride=(1, stride), dilation=2, groups=groups, bias=False)
        self.conv1d1 = nn.Conv2d(ch_in, ch_out, kernel_size=(1, kernel_size), padding=(0, padding_base * 4),
                                 stride=(1, stride), dilation=4, groups=groups, bias=False)
        # self.conv1d2 = nn.Conv2d( ch_in, ch_out, kernel_size=(1,kernel_size), padding=(0,padding_base*6), stride=(1,stride), dilation = 6, groups=groups, bias=False)
        # self.conv1d3 = nn.Conv2d( ch_in, ch_out, kernel_size=(1,kernel_size), padding=(0,padding_base*8), stride=(1,stride), dilation = 8, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.channel_mixer = nn.Conv2d(ch_out, ch_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), dilation=1,
                                       groups=1, bias=False)

        # self.channel_mixer = nn.Conv2d( ch_out*4, ch_out, kernel_size=(1,1), padding=(0,0), stride=(1,1), dilation = 1, groups=1, bias=False)

    def forward(self, x):
        # out  = self.relu(self.bn(self.conv1d0(x) + self.conv1d1(x) + self.conv1d2(x) + self.conv1d3(x)))
        out = self.relu(self.bn(self.conv1d0(x) + self.conv1d1(x)))  # + self.conv1d3(x)))
        # out1 = self.conv1d0(x)
        # out2 = self.conv1d1(x)
        # out3 = self.conv1d2(x)
        # out4 = self.conv1d3(x)
        # out  = torch.cat([out1,out2,out3,out4],1)
        out = self.channel_mixer(out)

        return out


class BasicFrameBlock(nn.Module):
    def __init__(self, kernel_size, ch_in, ch_out, stride=1, groups=1):
        super(BasicFrameBlock, self).__init__()
        padding_base = int(kernel_size / 2)

        self.conv1d0 = nn.Conv2d(ch_in, ch_out, kernel_size=(1, kernel_size), padding=(0, padding_base),
                                 stride=(1, stride), dilation=1, groups=groups, bias=False)
        self.bn0 = nn.BatchNorm2d(ch_out)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1d1 = nn.Conv2d(ch_out, ch_out, kernel_size=(1, kernel_size), padding=(0, padding_base), stride=(1, 1),
                                 dilation=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1d2 = nn.Conv2d(ch_out, ch_out, kernel_size=(1, kernel_size), padding=(0, padding_base), stride=(1, 1),
                                 dilation=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu2 = nn.ReLU(inplace=True)
        self.channel_mixer = nn.Conv2d(ch_out, ch_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), dilation=1,
                                       groups=1, bias=False)
        self.bn_mixer = nn.BatchNorm2d(ch_out)
        self.relu_mixer = nn.ReLU(inplace=True)

        def forward(self, x):
            out = self.relu0(self.bn0(self.conv1d0(x)))
            out = self.relu1(self.bn1(self.conv1d1(out)))
            out = self.relu2(self.bn2(self.conv1d2(out)))
            out = self.relu_mixer(self.bn_mixer(self.channel_mixer(out)))
            return out

