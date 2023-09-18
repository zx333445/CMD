#!/usr/bin/env python
# coding=utf-8
import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign

from .roi_head import CascadeRoIHeads,RoIHeads
from .transform import GeneralizedRCNNTransform
from .rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork
from .roiatt import RoiAtt

class FasterRCNNBase(nn.Module):
    """
    Main class for Generalized R-CNN.
    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:         
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                          boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2  
            original_image_sizes.append((val[0], val[1]))
        # original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)  

        # print(images.tensors.shape)
        features = self.backbone(images.tensors) 
        if isinstance(features, torch.Tensor):  
            features = OrderedDict([('0', features)])  

        # proposals: List[Tensor], Tensor_shape: [num_proposals, 4],
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class AttConvHead(nn.Module):
    '''
    inchannels 256
    resolution 7
    representation 1024
    '''
    def __init__(self, in_channels, resolution, representation_size):
        super().__init__()

        self.attconv = nn.Conv2d(in_channels,in_channels,resolution)
        self.spaconv = nn.Conv2d(1,1,3,padding=1)
        self.conv = nn.Conv2d(in_channels,representation_size,resolution)

    def forward(self,x):
        att = torch.sigmoid(self.attconv(x))
        x = att*x
        channel_pool = torch.max(x,1)[0].unsqueeze(1)
        spa = torch.sigmoid(self.spaconv(channel_pool))
        x = spa*x
        out = F.relu(self.conv(x)).view(x.shape[0],-1)
        return out


class TwoConvHead(nn.Module):
    '''
    inchannels 256
    resolution 7
    representation 1024
    '''
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels,3,padding=1)
        self.conv2 = nn.Conv2d(in_channels,in_channels,6)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        out = x.view(x.shape[0],-1)
        return out


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas



class FasterRCNN(FasterRCNNBase):
    """
    Implements Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
          between 0 and H and 0 and W
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between
          0 and H and 0 and W
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes

    """

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
                 box_head=None, 
                 box_predictor=None,
                 box_score_thresh=0.05, 
                 box_nms_thresh=0.5, 
                 box_detections_per_img=100,
                 # box_detections_per_img=20,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,   
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

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  
                output_size=[7, 7],
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]  
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2, 
                representation_size
            )

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        roi_heads = RoIHeads(
            box_roi_pool,box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,  # 0.5  0.5
            box_batch_size_per_image, box_positive_fraction,  # 512  0.25
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)  # 0.05  0.5  100

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)





class CascadeMiningDet(FasterRCNNBase):
    """
    Implements Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
          between 0 and H and 0 and W
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between
          0 and H and 0 and W
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes

    """

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
                 sec_roiatt = None,
                 thr_roiatt = None,
                 box_head=None,    # TwoMLPHead
                 sec_box_head=None,
                 thr_box_head=None,
                 box_predictor=None,   # FastRCNNPredictor
                 sec_box_predictor=None,
                 thr_box_predictor=None,
                 box_score_thresh=0.05, 
                 box_nms_thresh=0.5, 
                 box_detections_per_img=100,
                 # box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,  
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

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  
                output_size=[7, 7],
                sampling_ratio=2)
            
        if roiatt is None:
            roiatt = RoiAtt(feat_channel=256, hidden_channel=128)

        if sec_roiatt is None:
            sec_roiatt = RoiAtt(feat_channel=256, hidden_channel=128)

        if thr_roiatt is None:
            thr_roiatt = RoiAtt(feat_channel=256, hidden_channel=128)
   
        resolution = box_roi_pool.output_size[0] 
        representation_size = 1024
        if box_head is None:
            box_head = TwoMLPHead(
                out_channels * resolution ** 2, 
                representation_size
            )

        if sec_box_head is None:
            sec_box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)

        if thr_box_head is None:
            thr_box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)
        

        if box_predictor is None:
            box_predictor = FastRCNNPredictor(representation_size, num_classes)
            
        if sec_box_predictor is None:
            sec_box_predictor = FastRCNNPredictor(representation_size, num_classes)
        
        if thr_box_predictor is None:
            thr_box_predictor = FastRCNNPredictor(representation_size, num_classes)

        roi_heads = CascadeRoIHeads( 
            box_roi_pool,
            roiatt,sec_roiatt,thr_roiatt,
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
