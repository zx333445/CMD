#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append("..")

from torchvision.ops import MultiScaleRoIAlign

from network.transform import GeneralizedRCNNTransform
from network.rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork
from network.faster_rcnn_framework import FasterRCNNBase, TwoMLPHead, FastRCNNPredictor
from .component import RoiAtt, CmdRoIHeads


class CascadeMiningDet(FasterRCNNBase):
    ''''''
    def __init__(self, backbone, num_classes=None,
                 # transform parameter
                 min_size=800, max_size=1333,      
                 image_mean=None, image_std=None,  
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, 
                 rpn_pre_nms_top_n_test=1000,    
                 rpn_post_nms_top_n_train=2000, 
                 rpn_post_nms_top_n_test=1000,  
                 rpn_nms_thresh=0.7,  
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,  
                 rpn_batch_size_per_image=256, 
                 rpn_positive_fraction=0.5,  
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None,   
                 roiatt = None,    # RoiAtt
                 box_head=None,    # TwoMLPHead
                 sec_box_head=None,
                 thr_box_head=None,
                 box_predictor=None,   # FastRCNNPredictor
                 sec_box_predictor=None,
                 thr_box_predictor=None,
                 # post process parameters
                 box_score_thresh=0.05, 
                 box_nms_thresh=0.5, 
                 box_detections_per_img=100,
                 box_batch_size_per_image=512, 
                 box_positive_fraction=0.25,  
                 bbox_reg_weights=None):
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels"
                "specifying the number of output channels  (assumed to be the"
                "same for all the levels"
            )

        assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor "
                                 "is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (96,), (128,), (256,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorsGenerator(
                anchor_sizes, aspect_ratios
            )

        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # RPN
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  # 在哪些特征层进行roi pooling
                output_size=[7, 7],
                sampling_ratio=2)
            

        # roiatt
        if roiatt is None:
            roiatt = RoiAtt(feat_channel=256, hidden_channel=128)

        resolution = box_roi_pool.output_size[0]  
        representation_size = 1024

        # box_head
        if box_head is None:
            box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)

        if sec_box_head is None:
            sec_box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)

        if thr_box_head is None:
            thr_box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)
        
        # box_predictor
        if box_predictor is None:
            box_predictor = FastRCNNPredictor(representation_size, num_classes)
            
        if sec_box_predictor is None:
            sec_box_predictor = FastRCNNPredictor(representation_size, num_classes)
        
        if thr_box_predictor is None:
            thr_box_predictor = FastRCNNPredictor(representation_size, num_classes)

        # roi_heads
        roi_heads = CmdRoIHeads( 
            box_roi_pool,
            roiatt,
            box_head, sec_box_head, thr_box_head,
            box_predictor, sec_box_predictor, thr_box_predictor,
            box_batch_size_per_image, box_positive_fraction,  # 512  0.25
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)  # 0.05  0.5  100

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(CascadeMiningDet, self).__init__(backbone, rpn, roi_heads, transform)