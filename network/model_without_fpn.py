#!/usr/bin/env python
# coding=utf-8
from torchvision import models
from torchvision.models import resnet
from torchvision.models import densenet
from torchvision.ops import MultiScaleRoIAlign

from .faster_rcnn_framework import CascadeMiningDet
from .rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork



def create_dense_model(num_classes, backbone_name, pretrained):

    backbone = densenet.__dict__[backbone_name](
        # weights=models.DenseNet169_Weights.DEFAULT   
        pretrained = pretrained    
    ).features   
    
    for name, param in backbone.named_parameters():
        if "denseblock" not in name and "transition" not in name:
            param.requires_grad_(False)

    backbone.out_channels = 1664  

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],  
                                    output_size=[7, 7],   
                                    sampling_ratio=2)  

    model = CascadeMiningDet(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


# if __name__ == "__main__":

#     model = create_dense_model(4,'densenet169',True)
#     print(model)
