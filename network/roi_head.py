#!/usr/bin/env python
# coding=utf-8
from typing import Optional, List, Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .focalloss import CEFocalLoss,FocalLoss
from . import det_utils
from . import boxes as box_ops


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits : 预测类别概率信息[num_anchors, num_classes]
        box_regression : 预测边目标界框回归信息
        labels : 真实类别信息
        regression_targets : 真实目标边界框信息

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0) 
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)

    # focal = FocalLoss(3) 
    # import ipdb;ipdb.set_trace()
    # classification_loss = focal(class_logits, labels)
    
    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    # sampled_pos_inds_subset = torch.nonzero(torch.gt(labels, 0)).squeeze(1)
    sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]

    labels_pos = labels[sampled_pos_inds_subset]

    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)
    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum"
    )
    box_loss = box_loss / labels.numel()
    return classification_loss, box_loss



class RoIHeads(torch.nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,   # Multi-scale RoIAlign pooling
                 box_head,       # TwoMLPHead
                 box_predictor,  # FastRCNNPredictor
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,  # default: 0.5, 0.5
                 batch_size_per_image, positive_fraction,  # default: 512, 0.25
                 bbox_reg_weights,  # None
                 # Faster R-CNN inference
                 score_thresh,        # default: 0.05
                 nms_thresh,          # default: 0.5
                 detection_per_img):  # default: 100
        super(RoIHeads, self).__init__()

        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # default: 0.5
            bg_iou_thresh,  # default: 0.5
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,  # default: 512
            positive_fraction)     # default: 0.25

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool    # Multi-scale RoIAlign pooling
        self.box_head = box_head            # TwoMLPHead
        self.box_predictor = box_predictor  # FastRCNNPredictor

        self.score_thresh = score_thresh  # default: 0.05
        self.nms_thresh = nms_thresh      # default: 0.5
        self.detection_per_img = detection_per_img  # default: 100

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        Args:
            proposals:
            gt_boxes:
            gt_labels:

        Returns:

        """
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            device = proposals_in_image.device
            if gt_boxes_in_image.numel() == 0: 
                # background image
                
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = torch.tensor(0,device=device)

                # label ignore proposals (between low and high threshold)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_in_image[ignore_inds] = torch.tensor(-1,device=device)  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        # BalancedPositiveNegativeSampler
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            # img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        Args:
            proposals: 一个batch中每张图像rpn预测的boxes
            gt_boxes:  一个batch中每张图像对应的真实目标边界框

        Returns:

        """
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]
        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])

    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets     # type: Optional[List[Dict[str, Tensor]]]
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """  
        Args:
            proposals: rpn预测的boxes
            targets:
        Returns:
        """
        self.check_targets(targets)
        assert targets is not None

        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposal
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)

        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, labels, regression_targets

    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
        对网络的预测数据进行后处理，包括
        (1)根据proposal以及预测的回归参数计算出最终bbox坐标
        (2)对预测类别结果进行softmax处理
        (3)裁剪预测的boxes信息，将越界的坐标调整到图片边界上
        (4)移除所有背景信息
        (5)移除低概率目标
        (6)移除小尺寸目标
        (7)执行nms处理，并按scores进行排序
        (8)根据scores排序返回前topk个目标
        Args:
            class_logits: 网络预测类别概率信息
            box_regression: 网络预测的边界框回归参数
            proposals: rpn输出的proposal   List[tensor(1000,4),]长度为1
            image_shapes: 打包成batch前每张图像的宽高

        Returns:

        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]

        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove prediction with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            # gt: Computes input > other element-wise.
            # inds = torch.nonzero(torch.gt(scores, self.score_thresh)).squeeze(1)
            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximun suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self,
                features,       # type: Dict[str, Tensor]
                proposals,      # type: List[Tensor]
                image_shapes,   # type: List[Tuple[int, int]]
                targets=None    # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"

        if self.training:
            proposals, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)

        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return result, losses




class CascadeRoIHeads(torch.nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,   # Multi-scale RoIAlign pooling
                 roiatt,         # Roiatt
                 sec_roiatt,
                 thr_roiatt,
                 box_head,       # TwoMLPHead
                 sec_box_head,
                 thr_box_head,
                 box_predictor,  # FastRCNNPredictor
                 sec_box_predictor,
                 thr_box_predictor,
                 # Faster R-CNN training
                 batch_size_per_image, positive_fraction,  # default: 512, 0.25
                 bbox_reg_weights,  # None
                 # Faster R-CNN inference
                 score_thresh,        # default: 0.05
                 nms_thresh,          # default: 0.5
                 detection_per_img):  # default: 100
        super(CascadeRoIHeads, self).__init__()

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,  # default: 512
            positive_fraction)     # default: 0.25
        
        self.hard_sampler = det_utils.HardSampleMiningSampler(
            batch_size_per_image,
            positive_fraction
        )

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool    # Multi-scale RoIAlign pooling

        self.roiatt = roiatt      # Roiatt
        # self.sec_roiatt = sec_roiatt
        # self.thr_roiatt = thr_roiatt
        self.box_head = box_head            # TwoMLPHead
        self.sec_box_head = sec_box_head
        self.thr_box_head = thr_box_head
        
        self.box_predictor = box_predictor  # FastRCNNPredictor
        self.sec_box_predictor = sec_box_predictor
        self.thr_box_predictor = thr_box_predictor

        self.score_thresh = score_thresh  # default: 0.05
        self.nms_thresh = nms_thresh      # default: 0.5
        self.detection_per_img = detection_per_img  # default: 100


    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels, iou_thresh):
        # type: (List[Tensor], List[Tensor], List[Tensor], float) -> Tuple[List[Tensor], List[Tensor]]
        """
        Args:
            proposals:
            gt_boxes:
            gt_labels:
            iou_thresh: 
        Returns:

        """
        matched_idxs = []
        labels = []
        # assign ground-truth boxes for each proposal
        proposal_matcher = det_utils.Matcher(
            iou_thresh,  # default: 0.5
            iou_thresh,  # default: 0.5
            allow_low_quality_matches=False)

        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            device = proposals_in_image.device
            if gt_boxes_in_image.numel() == 0: 
                # background image
                
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # label background (below the low threshold)
                bg_inds = matched_idxs_in_image == proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = torch.tensor(0,device=device)

                # label ignore proposals (between low and high threshold)
                ignore_inds = matched_idxs_in_image == proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_in_image[ignore_inds] = torch.tensor(-1,device=device)  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels


    def subsample(self, labels, pred_labels):
        # type: (List[Tensor],List[Tensor]) -> List[Tensor]
        # BalancedPositiveNegativeSampler
        # HardSampleMiningSampler
        if pred_labels:
            sampled_pos_inds, sampled_neg_inds = self.hard_sampler(labels,pred_labels)
        else:    
            sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            # img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds


    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        Args:
            proposals: 一个batch中每张图像rpn预测的boxes
            gt_boxes:  一个batch中每张图像对应的真实目标边界框
        Returns:
        """
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]
        return proposals


    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])


    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets,    # type: Optional[List[Dict[str, Tensor]]]
                                iou_thresh,  # type: float
                                pred_labels = None  # type: List[Tensor]
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
        Args:
            proposals: rpn预测的boxes
            targets:
        Returns:
        """
        self.check_targets(targets)
        assert targets is not None

        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposal
        if iou_thresh == 0.5:
            proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, iou_thresh)

        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels,pred_labels)
        matched_gt_boxes = []
        num_images = len(proposals)

        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, labels, regression_targets


    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
        对网络的预测数据进行后处理，包括
        (1)根据proposal以及预测的回归参数计算出最终bbox坐标
        (2)对预测类别结果进行softmax处理
        (3)裁剪预测的boxes信息,将越界的坐标调整到图片边界上
        (4)移除所有背景信息
        (5)移除低概率目标
        (6)移除小尺寸目标
        (7)执行nms处理,并按scores进行排序
        (8)根据scores排序返回前topk个目标
        Args:
            class_logits: 网络预测类别概率信息
            box_regression: 网络预测的边界框回归参数
            proposals: rpn输出的proposal   List[tensor(1000,4),]长度为1
            image_shapes: 打包成batch前每张图像的宽高
        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove prediction with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            # gt: Computes input > other element-wise.
            # inds = torch.nonzero(torch.gt(scores, self.score_thresh)).squeeze(1)
            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximun suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels


    def forward(self,
                features,       # type: Dict[str, Tensor]
                proposals,      # type: List[Tensor]
                image_shapes,   # type: List[Tuple[int, int]]
                targets=None    # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """

        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"

        
        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {} 
        if self.training:
            proposals, labels, regression_targets = self.select_training_samples(proposals, targets, iou_thresh=0.5)
            
            # box_features_shape: [num_proposals, channel, height, width]
            box_features = self.box_roi_pool(features, proposals, image_shapes)
            box_features = self.roiatt(box_features,features['0'])
            box_features = self.box_head(box_features)

            class_logits, box_regression = self.box_predictor(box_features)
            
            pred_boxes = self.box_coder.decode(box_regression, proposals)
            pred_boxes = pred_boxes[:,1:]
            
            proposals_per_image = [len(b) for b in proposals]
            sec_proposals = pred_boxes.split(proposals_per_image,dim=0)
            sec_proposals = [sec.contiguous().view(-1,4) for sec in sec_proposals]

            pred_labels = torch.arange(1,pred_boxes.shape[1]+1,device=proposals[0].device).expand_as(torch.randn(pred_boxes.shape[0],pred_boxes.shape[1]))
            pred_labels = pred_labels.split(proposals_per_image,dim=0)
            pred_labels = [lab.contiguous().view(-1) for lab in pred_labels]


            sec_proposals, sec_labels, sec_regression_targets = self.select_training_samples(sec_proposals, targets, iou_thresh=0.6, pred_labels=pred_labels)            
            # sec_proposals, sec_labels, sec_regression_targets = self.select_training_samples(sec_proposals, targets, iou_thresh=0.5, pred_labels=pred_labels)
            # sec_proposals, sec_labels, sec_regression_targets = self.select_training_samples(sec_proposals, targets, iou_thresh=0.6)
            sec_boxfeatures = self.box_roi_pool(features,sec_proposals,image_shapes)
            # sec_boxfeatures = self.sec_roiatt(sec_boxfeatures,features['0'])
            sec_boxfeatures = self.sec_box_head(sec_boxfeatures)
            sec_class_logits, sec_box_regression = self.sec_box_predictor(sec_boxfeatures)
            sec_predboxes = self.box_coder.decode(sec_box_regression, sec_proposals)
            sec_predboxes = sec_predboxes[:,1:]

            sec_proposals_per_image = [len(b) for b in sec_proposals]
            thr_proposals = sec_predboxes.split(sec_proposals_per_image,dim=0)
            thr_proposals = [thr.contiguous().view(-1,4) for thr in thr_proposals]

            sec_pred_labels = torch.arange(1,sec_predboxes.shape[1]+1,device=proposals[0].device).expand_as(torch.randn(sec_predboxes.shape[0],sec_predboxes.shape[1]))
            sec_pred_labels = sec_pred_labels.split(sec_proposals_per_image,dim=0)
            sec_pred_labels = [lab.contiguous().view(-1) for lab in sec_pred_labels]

            thr_proposals, thr_labels, thr_regression_targets = self.select_training_samples(thr_proposals, targets, iou_thresh=0.7, pred_labels=sec_pred_labels)
            # thr_proposals, thr_labels, thr_regression_targets = self.select_training_samples(thr_proposals, targets, iou_thresh=0.5, pred_labels=sec_pred_labels)
            # thr_proposals, thr_labels, thr_regression_targets = self.select_training_samples(thr_proposals, targets, iou_thresh=0.7)
            thr_boxfeatures = self.box_roi_pool(features,thr_proposals,image_shapes)
            # thr_boxfeatures = self.thr_roiatt(thr_boxfeatures,features['0'])
            thr_boxfeatures = self.thr_box_head(thr_boxfeatures)
            thr_class_logits, thr_box_regression = self.thr_box_predictor(thr_boxfeatures)

            assert labels is not None and regression_targets is not None

            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            
            sec_loss_classifier,sec_loss_box_reg = fastrcnn_loss(
                sec_class_logits, sec_box_regression, sec_labels, sec_regression_targets)
            
            thr_loss_classifier,thr_loss_box_reg = fastrcnn_loss(
                thr_class_logits, thr_box_regression, thr_labels, thr_regression_targets)
            
            losses = {
                "loss_classifier": loss_classifier + sec_loss_classifier + thr_loss_classifier,
                "loss_box_reg": loss_box_reg + sec_loss_box_reg + thr_loss_box_reg
            }


        else:
            box_features = self.box_roi_pool(features, proposals, image_shapes)
            box_features = self.roiatt(box_features,features['0'])
            box_features = self.box_head(box_features)
            class_logits, box_regression = self.box_predictor(box_features)
            pred_boxes = self.box_coder.decode(box_regression, proposals)
            pred_boxes = pred_boxes[:,1:]

            proposals_per_image = [len(b) for b in proposals]
            sec_proposals = pred_boxes.split(proposals_per_image,dim=0)
            sec_proposals = [sec.contiguous().view(-1,4) for sec in sec_proposals]

            sec_boxfeatures = self.box_roi_pool(features,sec_proposals,image_shapes)
            # sec_boxfeatures = self.sec_roiatt(sec_boxfeatures,features['0'])
            sec_boxfeatures = self.sec_box_head(sec_boxfeatures)
            sec_class_logits, sec_box_regression = self.sec_box_predictor(sec_boxfeatures)
            sec_predboxes = self.box_coder.decode(sec_box_regression, sec_proposals)
            sec_predboxes = sec_predboxes[:,1:]

            sec_proposals_per_image = [len(b) for b in sec_proposals]
            thr_proposals = sec_predboxes.split(sec_proposals_per_image,dim=0)
            thr_proposals = [thr.contiguous().view(-1,4) for thr in thr_proposals]

            thr_boxfeatures = self.box_roi_pool(features,thr_proposals,image_shapes)
            # thr_boxfeatures = self.thr_roiatt(thr_boxfeatures,features['0'])
            thr_boxfeatures = self.thr_box_head(thr_boxfeatures)
            thr_class_logits, thr_box_regression = self.thr_box_predictor(thr_boxfeatures)
            
            boxes, scores, labels = self.postprocess_detections(thr_class_logits, thr_box_regression, thr_proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return result, losses


