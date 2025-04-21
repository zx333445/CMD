#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.jit.annotations import Dict, List
from typing import Callable, Dict, List, Optional, Tuple

from collections import OrderedDict

from .attention_layer import CSABlock


class AttFeaturePyramidNetwork(nn.Module):
    """
    Attention feature pyramid network
    params:
        in_channels_list: list, [C2,C3,C4,C5] channels
        out_channel: num channels after this module
    """

    def __init__(self, in_channels_list, out_channel, extra_block=None, extra_type=None):
        super(AttFeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()  # save 1x1 conv
        self.csablocks = nn.ModuleList()
        for in_channel in in_channels_list:
            inner_block = nn.Conv2d(in_channel, out_channel, 1)
            inner_block_gn = nn.GroupNorm(32, out_channel, 1e-5)
            self.inner_blocks.append(
                nn.Sequential(inner_block, inner_block_gn)
            )
            
            self.csablocks.append(CSABlock(out_channel))

        self.extra_block = extra_block
        self.extra_type = extra_type


    def get_result_from_inner_blocks(self, x, idx):
        num_blocks = 0
        for m in self.inner_blocks:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        out = x
        i = 0
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_csa_blocks(self, x, y, idx):
        num_blocks = 0
        for m in self.csablocks:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        out = x
        i = 0
        for module in self.csablocks:
            if i == idx:
                out = module(x, y)
            i += 1
        return out

    def forward(self, x):
        """
        forward propagate
        """
        names = list(x.keys())
        x = list(x.values())
        results = []
        # 先把最顶层的C5计算出来，因为它没有+的过程
        last_out = self.get_result_from_inner_blocks(x[-1], -1)
        results.append(last_out)
        for idx in range(len(x)-2, -1, -1):
            inner = self.get_result_from_inner_blocks(x[idx], idx)
            last_out = self.get_result_from_csa_blocks(inner, last_out, idx)
            results.insert(0, last_out)

        # faster cascade sparse
        if self.extra_block and self.extra_type == "maxpool":
           results, names  = self.extra_block(results, names)
        
        # retinanet fcos
        elif self.extra_block and self.extra_type == "last":
            results, names  = self.extra_block(results, x, names)

        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out



class FeaturePyramidNetwork(nn.Module):
    ''''''
    def __init__(self, in_channels_list, out_channel, extra_block=None, extra_type=None):
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
                nn.init.constant_(m.bias, 0) # type: ignore

        self.extra_block = extra_block
        self.extra_type = extra_type

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
               
        # faster cascade sparse
        if self.extra_block and self.extra_type == "maxpool":
           results, names  = self.extra_block(result, names)
        
        # retinanet fcos
        elif self.extra_block and self.extra_type == "last":
            results, names  = self.extra_block(result, x, names)

        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out


class MaxpoolOnP5(nn.Module):
    
    def forward(self, results, names):
        names.append("pool")
        p6 = F.max_pool2d(results[-1], 1, 2, 0)
        results.append(p6)
        return results,names 


class ExtraFPNBlock(nn.Module):
    """
    Base class for the extra block in the FPN.

    Args:
        results (List[Tensor]): the result of the FPN
        x (List[Tensor]): the original feature maps
        names (List[str]): the names for each one of the
            original feature maps

    Returns:
        results (List[Tensor]): the extended set of results
            of the FPN
        names (List[str]): the extended set of names for the results
    """

    def forward(
        self,
        results: List[Tensor],
        x: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]: # type: ignore
        pass


class LastLevelMaxPool(ExtraFPNBlock):
    """
    Applies a max_pool2d (not actual max_pool2d, we just subsample) on top of the last feature map
    """

    def forward(
        self,
        x: List[Tensor],
        y: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        # Use max pooling to simulate stride 2 subsampling
        x.append(F.max_pool2d(x[-1], kernel_size=1, stride=2, padding=0))
        return x, names
    

class LastLevelP6P7(ExtraFPNBlock):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0) # type: ignore
        self.use_P5 = in_channels == out_channels

    def forward(
        self,
        p: List[Tensor],
        c: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        p5, c5 = p[-1], c[-1]
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        p.extend([p6, p7])
        names.extend(["p6", "p7"])
        return p, names
    


if __name__ == "__main__":
    # import torch
    # from layergetter import IntermediateLayerGetter
    # from torchvision import models
    # from torchvision.models._utils import IntermediateLayerGetter as Getter
    # from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork as Feat
    # from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


    # image = torch.randn(1, 3, 224, 224)
    # model = models.resnet50(pretrained=True)

    # return_layers = {"layer1": "feat1", "layer2": "feat2",
    #                  "layer3": "feat3", "layer4": "feat4"}
    # return_layers2 = {"layer1": "feat1", "layer2": "feat2",
    #                  "layer3": "feat3", "layer4": "feat4"}
    # body = IntermediateLayerGetter(model, return_layers)
    # fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], 256, MaxpoolOnP5())
    # x = body(image)
    # out = fpn(x)
    # bottom_up = Bottom_up_path([256,256,256,256], 256, MaxpoolOnP5())
    # out = bottom_up(out)
    # import ipdb;ipdb.set_trace()
    # body2 = Getter(model, return_layers2)
    # fpn2 = Feat([256, 512, 1024, 2048], 256, LastLevelMaxPool())
    # x2 = body2(image)
    # out2 = fpn2(x2)
    # body3 = Getter(model, return_layers2)
    # fpn3 = Feat([256, 512, 1024, 2048], 256, LastLevelMaxPool())
    # x3 = body3(image)
    # out3 = fpn3(x2)
    # import ipdb;ipdb.set_trace()
    C2 = torch.randn(1, 256, 224, 224)
    C3 = torch.randn(1, 512, 112, 112)
    C4 = torch.randn(1, 1024, 56, 56)
    C5 = torch.randn(1, 2048, 28, 28)
    x = {"C2": C2, "C3": C3, "C4": C4, "C5": C5}
    import ipdb;ipdb.set_trace()



