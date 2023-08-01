#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict



class FeaturePyramidNetwork(nn.Module):

    def __init__(self, in_channels_list, out_channel, extra_block=None):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()  # 存储1x1conv
        self.layer_blocks = nn.ModuleList()  # 存储3x3conv
        for in_channel in in_channels_list:
            inner_block = nn.Conv2d(in_channel, out_channel, 1)
            inner_block_gn = nn.GroupNorm(32, out_channel, 1e-5)
            layer_block = nn.Conv2d(out_channel, out_channel, 3, padding=1)
            layer_block_gn = nn.GroupNorm(32, out_channel, 1e-5)
            self.inner_blocks.append(
                nn.Sequential(inner_block, inner_block_gn)
            )
            self.layer_blocks.append(
                nn.Sequential(layer_block, layer_block_gn)
            )

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.extra_block = extra_block

    def get_result_from_inner_blocks(self, x, idx):

        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x, idx):
        
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x):
        names = list(x.keys())
        x = list(x.values())
        result = []
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        last_layer = self.get_result_from_layer_blocks(last_inner, -1)
        result.append(last_layer)
        for idx in range(len(x)-2, -1, -1):
            inner = self.get_result_from_inner_blocks(x[idx], idx)
            upsample = F.interpolate(last_inner, inner.shape[-2:], mode="nearest")
            last_inner = inner + upsample
            layer = self.get_result_from_layer_blocks(last_inner, idx)
            result.insert(0, layer)

        if self.extra_block is not None:
            names, result = self.extra_block(result, names)

        out = OrderedDict([(k, v) for k, v in zip(names, result)])
        return out


class MaxpoolOnP5(nn.Module):
    
    def forward(self, result, name):
        name.append("pool")
        p6 = F.max_pool2d(result[-1], 1, 2, 0)
        result.append(p6)
        return name, result


