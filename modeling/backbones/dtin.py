import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.init import calculate_gain

def bulid_compress_module(in_channels, embed_channel,
                         nonlinearity='relu',
                         stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, embed_channel, 1, stride=stride))


def build_weight_branch(in_channels, embed_channel, ratio=0.2):
    assert ratio > 0
    mid_channels = int(in_channels * ratio)
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Conv2d(in_channels, mid_channels, 1),
        nn.ReLU(True),
        nn.Conv2d(mid_channels, in_channels * embed_channel, 1))

def build_bias_branch(in_channels, ratio=0.2):
    assert ratio > 0
    mid_channels = int(in_channels * ratio)
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Conv2d(in_channels, mid_channels, 1),
        nn.ReLU(True),
        nn.Conv2d(mid_channels, in_channels, 1))

class DTIN(nn.Module):
    def __init__(self, in_channels, embed_channel = 64, stride=1, dilation=1,
                 ratio=0.2):
        super(DTIN, self).__init__()
        assert embed_channel > 1
        self.embed_channel = embed_channel
        self.stride = stride
        self.dilation = dilation

        self.compress_module = bulid_compress_module(
            in_channels, embed_channel, stride)
        self.weight_branch = build_weight_branch(
            in_channels, embed_channel, ratio)
        self.bias_branch = build_bias_branch(
            in_channels, ratio)
        self.in_bn = nn.InstanceNorm2d(embed_channel)
    def forward(self, x, kernel_feats):
    
        b, c, h, w = x.shape
        e = self.embed_channel
        s = self.stride
        weights = self.weight_branch(kernel_feats).reshape(b, c, e)
        bias = self.bias_branch(kernel_feats).reshape(b, c, 1, 1).view(b, c, 1)
        compressed_feats = self.compress_module(x).reshape(b, -1, h//s, w//s).view(b, e, h, w)
        compressed_feats = self.in_bn(compressed_feats)
        compressed_feats = compressed_feats.view(b, e, h*w)
        
        
        out = torch.bmm(weights, compressed_feats).view(b, c, h, w)
        bias = bias.view(b,c,1,1)
        out = out + bias
        
        return out



