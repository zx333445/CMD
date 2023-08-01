#!/usr/bin/env python
# coding=utf-8
import torch.nn as nn
from torchvision.models import resnet
from torchvision.models import densenet
from torchvision import models
from .layergetter import IntermediateLayerGetter, DenseNetLayerGetter, SwinLayerGetter, ConvNextLayerGetter
from .fpn import FeaturePyramidNetwork, MaxpoolOnP5
from .misc import FrozenBatchNorm2d


class BackboneWithFPN(nn.Module):

    def __init__(self, backbone, return_layers,
                 in_channels_list, out_channel):
        super(BackboneWithFPN, self).__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers)
        self.fpn = FeaturePyramidNetwork(in_channels_list,
                                         out_channel,
                                         extra_block=MaxpoolOnP5())
        self.out_channels = out_channel

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


class BackboneWithFPNForDensenet(nn.Module):

    def __init__(self, backbone, in_channels_list, out_channel):
        super(BackboneWithFPNForDensenet, self).__init__()
        self.body = DenseNetLayerGetter(backbone)
        self.fpn = FeaturePyramidNetwork(in_channels_list,
                                         out_channel,
                                         extra_block=MaxpoolOnP5())
        self.out_channels = out_channel

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


class BackboneWithFPNForSwin(nn.Module):
    
    def __init__(self, backbone, in_channels_list, out_channel):
        super().__init__()
        self.body = SwinLayerGetter(backbone)
        self.fpn = FeaturePyramidNetwork(in_channels_list,
                                         out_channel,
                                         extra_block=MaxpoolOnP5())
        self.out_channels = out_channel

    def forward(self,x):
        x = self.body(x)
        x = self.fpn(x)
        return x


class BackboneWithFPNForConvNext(nn.Module):
    
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



def resnet_fpn_backbone(backbone_name, pretrained):
 
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
                          in_channels_list, out_channel)


def densenet_fpn_backbone(backbone_name, pretrained):
    backbone = densenet.__dict__[backbone_name](
        weights=models.DenseNet169_Weights.DEFAULT   
        # pretrained = pretrained    
    )
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
                                      out_channel)


def swin_fpn_backbone():
    backbone = models.swin_b(weights = models.Swin_B_Weights.DEFAULT)

    in_channels_list = [128,256,512,1024]  # swin-b
    # in_channels_list = [96,192,384,768]  # swin-s
    out_channel = 256
    return BackboneWithFPNForSwin(backbone,
                                  in_channels_list,
                                  out_channel)


def convnext_fpn_backbone():
    backbone = models.convnext_base(weights = models.ConvNeXt_Base_Weights.DEFAULT)

    in_channels_list = [128,256,512,1024]
    out_channel = 256
    return BackboneWithFPNForConvNext(backbone,
                                  in_channels_list,
                                  out_channel)


