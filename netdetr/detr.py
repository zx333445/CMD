# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn

from .transform import DETRTransform
from .detr_util import DETRbody, SetCriterion


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self,
            num_classes=2,
            min_size = 800, 
            max_size = 1333,
            image_mean = [0.485, 0.456, 0.406], 
            image_std = [0.229, 0.224, 0.225],
            num_queries=100):
        super().__init__()
        self.num_classes = num_classes
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.num_queries = num_queries
        #  transform
        self.transform = DETRTransform(self.min_size, self.max_size, self.image_mean, self.image_std)
        #  model body
        self.model = DETRbody(num_classes=self.num_classes, num_queries=self.num_queries)
        param_dict = torch.load("/home/stat-zx/CTCdet/netdetr/detr-r50-e632da11.pth")['model']
        self.model.load_state_dict({k: v for k, v in param_dict.items() if 'class_embed' not in k},strict=False)
        #  losses
        self.losses = SetCriterion(num_classes=self.num_classes)

    def forward(self, images, targets=None):
        samples, gt = self.transform(images, targets)
        out = self.model(samples)
        
        if self.training:
            loss_dict = self.losses(out, gt)
            return loss_dict
        else:
            img_sizes = torch.stack([torch.tensor(img.shape[1:]) for img in images],dim=0)
            return self.transform.postprocess(out, img_sizes)
