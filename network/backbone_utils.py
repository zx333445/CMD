#!/usr/bin/env python
# coding=utf-8
import torch.nn as nn
from torchvision.models import resnet
from torchvision.models import densenet
from torchvision import models
from .layergetter import IntermediateLayerGetter, DenseNetLayerGetter, SwinLayerGetter, ConvNextLayerGetter
from .fpn import FeaturePyramidNetwork, AttFeaturePyramidNetwork, MaxpoolOnP5, LastLevelMaxPool, LastLevelP6P7
from .misc import FrozenBatchNorm2d


class BackboneWithFPN(nn.Module):

    def __init__(self, backbone, return_layers,
                 in_channels_list, out_channel, extra_type=None):
        super(BackboneWithFPN, self).__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers)
        # faster,cascade,sparse
        if extra_type == 'maxpool':
            self.fpn = FeaturePyramidNetwork(in_channels_list,
                                            out_channel,
                                            extra_block=MaxpoolOnP5(),
                                            extra_type=extra_type)
        # fcos, retinanet
        elif extra_type == 'last':
            self.fpn = FeaturePyramidNetwork(in_channels_list,
                                             out_channel,
                                             extra_block=LastLevelP6P7(2048,256),
                                             extra_type=extra_type)     
        self.out_channels = out_channel

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


class BackboneWithFPNForDensenet(nn.Module):

    def __init__(self, backbone, in_channels_list, out_channel, extra_type=None):
        super(BackboneWithFPNForDensenet, self).__init__()
        self.body = DenseNetLayerGetter(backbone)
        # faster,cascade,sparse
        if extra_type == 'maxpool':
            self.fpn = AttFeaturePyramidNetwork(in_channels_list,
                                            out_channel,
                                            extra_block=MaxpoolOnP5(),
                                            extra_type=extra_type)
        # fcos, retinanet
        elif extra_type == 'last':
            self.fpn = AttFeaturePyramidNetwork(in_channels_list,
                                             out_channel,
                                             extra_block=LastLevelP6P7(2048,256),
                                             extra_type=extra_type)     
        self.out_channels = out_channel

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


class FPNForSwin(nn.Module):
    ''''''
    def __init__(self, backbone, in_channels_list, out_channel, extra_type=None):
        super().__init__()
        self.body = SwinLayerGetter(backbone)
        # faster,cascade,sparse
        if extra_type == 'maxpool':
            self.fpn = FeaturePyramidNetwork(in_channels_list,
                                             out_channel,
                                             extra_block=MaxpoolOnP5(),
                                             extra_type=extra_type)
        # fcos, retinanet
        elif extra_type == 'last':
            self.fpn = FeaturePyramidNetwork(in_channels_list,
                                             out_channel,
                                             extra_block=LastLevelP6P7(in_channels_list[-1],256),
                                             extra_type=extra_type)     
        self.out_channels = out_channel

    def forward(self,x):
        x = self.body(x)
        x = self.fpn(x)
        return x
    

class AttFPNForSwin(nn.Module):
    ''''''
    def __init__(self, backbone, in_channels_list, out_channel, extra_type=None):
        super().__init__()
        self.body = SwinLayerGetter(backbone)
        # faster,cascade,sparse
        if extra_type == 'maxpool':
            self.fpn = AttFeaturePyramidNetwork(in_channels_list,
                                            out_channel,
                                            extra_block=MaxpoolOnP5(),
                                            extra_type=extra_type)
        # fcos, retinanet
        elif extra_type == 'last':
            self.fpn = AttFeaturePyramidNetwork(in_channels_list,
                                             out_channel,
                                             extra_block=LastLevelP6P7(in_channels_list[-1],256),
                                             extra_type=extra_type)     
        self.out_channels = out_channel

    def forward(self,x):
        x = self.body(x)
        x = self.fpn(x)
        return x


class BackboneWithFPNForConvNext(nn.Module):
    ''''''
    def __init__(self, backbone, in_channels_list, out_channel):
        super().__init__()
        self.body = ConvNextLayerGetter(backbone)
        self.fpn = FeaturePyramidNetwork(in_channels_list,
                                         out_channel,
                                         extra_block=MaxpoolOnP5())
        self.out_channels = out_channel

    def forward(self,x):
        x = self.body(x)
        x = self.fpn(x)
        return x


def resnet_fpn_backbone(backbone_name, extra_type='maxpool'):
 
    backbone = resnet.__dict__[backbone_name](
        weights= models.ResNet50_Weights.DEFAULT
        # weights = None
        # pretrained = pretrained  
    )
    for name, param in backbone.named_parameters():
        if "layer2" not in name and "layer3" not in name and "layer4" not in name:
            param.requires_grad_(False)

    return_layers = {"layer1": "0", "layer2": "1",
                     "layer3": "2", "layer4": "3"}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channel = 256
    return BackboneWithFPN(backbone, return_layers,
                          in_channels_list, out_channel, extra_type=extra_type)


def densenet_fpn_backbone(extra_type='maxpool'):
    backbone = densenet.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
    for name, param in backbone.features.named_parameters():
        if "denseblock" not in name and "transition" not in name:
            param.requires_grad_(False)

    # in_channels_list = [128, 256, 512, 1024]  # densenet121
    # in_channels_list = [192, 384, 1056, 2208] # densenet161
    in_channels_list = [128, 256, 640, 1664]  # densenet169
    # in_channels_list = [128, 256, 896, 1920]
    out_channel = 256
    return BackboneWithFPNForDensenet(backbone,
                                      in_channels_list,
                                      out_channel,
                                      extra_type=extra_type)


def swin_fpn_backbone(extra_type='maxpool'):
    backbone = models.swin_s(weights = models.Swin_S_Weights.DEFAULT)

    # in_channels_list = [128,256,512,1024]  # swin-b
    in_channels_list = [96,192,384,768]  # swin-s
    out_channel = 256
    return FPNForSwin(backbone, in_channels_list, out_channel, extra_type=extra_type)


def swin_attfpn_backbone(extra_type='maxpool'):
    backbone = models.swin_s(weights = models.Swin_S_Weights.DEFAULT)

    # in_channels_list = [128,256,512,1024]  # swin-b
    in_channels_list = [96,192,384,768]  # swin-s
    out_channel = 256
    return AttFPNForSwin(backbone, in_channels_list, out_channel, extra_type=extra_type)


def convnext_fpn_backbone():
    backbone = models.convnext_base(weights = models.ConvNeXt_Base_Weights.DEFAULT)

    in_channels_list = [128,256,512,1024]
    out_channel = 256
    return BackboneWithFPNForConvNext(backbone,
                                  in_channels_list,
                                  out_channel)



if __name__ == "__main__":
    import torch

    x = torch.randn(6, 3, 256, 256)
    # net = resnet_fpn_backbone("resnet50", True)
    # net = densenet_fpn_backbone("densenet161", True)
    net = swin_fpn_backbone()
    out = net(x)
    import ipdb;ipdb.set_trace()

