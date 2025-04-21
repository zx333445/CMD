#!/usr/bin/env python
# coding=utf-8
from typing import Optional, List, Dict, Tuple

import sys
sys.path.append("..")
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from network import det_utils
from network import boxes as box_ops
from network.roi_head import fastrcnn_loss


class RoiAtt(nn.Module):
    def __init__(self, feat_channel, hidden_channel):
        super().__init__()

        self.hidden_channel = hidden_channel

        self.conv_q = nn.Conv2d(feat_channel,hidden_channel,1)
        self.conv_k = nn.Conv2d(feat_channel,hidden_channel,1)
        self.conv_v = nn.Conv2d(feat_channel,hidden_channel,1)
        self.conv_re = nn.Conv2d(hidden_channel,feat_channel,1)
        
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0) # type: ignore


    def forward(self,rois,feature):
        BS_roinum,C,h,w = rois.shape
        BS = feature.shape[0]
        roinum = BS_roinum//BS        

        q = self.conv_q(rois)
        q = q.contiguous().view(BS,roinum,-1)

        k = self.conv_k(rois)
        k = k.contiguous().view(BS,roinum,-1)
        k = k.permute(0,2,1)

        v = self.conv_v(rois)
        v = v.contiguous().view(BS,roinum,-1)

        score = torch.bmm(q,k)
        score = torch.softmax(score, dim=2)

        y = torch.bmm(score,v)
        y = y.contiguous().view(BS_roinum,self.hidden_channel,h,w)
        y = self.conv_re(y)

        out = rois + y

        return out


class HardSampleMiningSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """
    def __init__(self, batch_size_per_image, positive_fraction):
        # type: (int, float) -> None
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image # 256
        self.positive_fraction = positive_fraction # 0.5

    def __call__(self, matched_idxs, pred_labels):
        # type: (List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]

        pos_idx = []
        neg_idx = []

        for matched_idxs_per_image, pred_labels_per_image in zip(matched_idxs,pred_labels):

            positive_idx = torch.where(torch.ge(matched_idxs_per_image, 1))[0]
            wrong_idx = matched_idxs_per_image != pred_labels_per_image
            inter_idx = wrong_idx[positive_idx]
            hard_positive = positive_idx[inter_idx]

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive_idx.numel(), num_pos)

            # randomly select positive and negative examples
            # Returns a random permutation of integers from 0 to n - 1.
            num_hard = hard_positive.numel()

            if num_hard >= num_pos:
                perm1 = torch.randperm(hard_positive.numel(), device=positive_idx.device)[:num_pos]
                pos_idx_per_image = hard_positive[perm1]
            else:
                perm1 = torch.randperm(positive_idx[~inter_idx].numel(), device=positive_idx.device)[:(num_pos-num_hard)]
                pos_idx_per_image = torch.cat([hard_positive,positive_idx[perm1]])
            

            negative_idx = torch.where(torch.eq(matched_idxs_per_image, 0))[0]

            num_neg = self.batch_size_per_image - num_hard
            num_neg = min(negative_idx.numel(), num_neg)

            perm2 = torch.randperm(negative_idx.numel(), device=negative_idx.device)[:num_neg]
            neg_idx_per_image = negative_idx[perm2]
            
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


class CmdRoIHeads(torch.nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,   # Multi-scale RoIAlign pooling
                 roiatt,   # RoiAtt
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
        super().__init__()
        
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,  # default: 512
            positive_fraction)     # default: 0.25
        
        self.hard_sampler = HardSampleMiningSampler(
            batch_size_per_image,
            positive_fraction
        )

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool    # Multi-scale RoIAlign pooling

        self.roiatt = roiatt                # RoiAtt
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
        # if pred_label HardSampleMiningSampler or BalancedPositiveNegativeSampler
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
        将gt_boxes拼接到proposal后面
        Args:
            proposals: 一个batch中每张图像rpn预测的boxes
            gt_boxes:  一个batch中每张图像对应的真实目标边界框

        Returns:

        """
        # cat默认dim = 0
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
                                pred_labels = None  # type: List[Tensor] # type: ignore
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
        划分正负样本,统计对应gt的标签以及边界框回归信息,
        list元素个数为batch_size
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

        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, iou_thresh)
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
            # first stage
            proposals, labels, regression_targets = self.select_training_samples(proposals, targets, iou_thresh=0.5)
            # proposals, labels, regression_targets = self.select_training_samples(proposals, targets, iou_thresh=0.6)
            
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

            # second stage
            # sec_proposals, sec_labels, sec_regression_targets = self.select_training_samples(sec_proposals, targets, iou_thresh=0.6, pred_labels=pred_labels)            
            sec_proposals, sec_labels, sec_regression_targets = self.select_training_samples(sec_proposals, targets, iou_thresh=0.5, pred_labels=pred_labels)
            # sec_proposals, sec_labels, sec_regression_targets = self.select_training_samples(sec_proposals, targets, iou_thresh=0.7, pred_labels=pred_labels)
            # sec_proposals, sec_labels, sec_regression_targets = self.select_training_samples(sec_proposals, targets, iou_thresh=0.5)
            sec_boxfeatures = self.box_roi_pool(features,sec_proposals,image_shapes)
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

            # third stage
            # thr_proposals, thr_labels, thr_regression_targets = self.select_training_samples(thr_proposals, targets, iou_thresh=0.7, pred_labels=sec_pred_labels)
            thr_proposals, thr_labels, thr_regression_targets = self.select_training_samples(thr_proposals, targets, iou_thresh=0.5, pred_labels=sec_pred_labels)
            # thr_proposals, thr_labels, thr_regression_targets = self.select_training_samples(thr_proposals, targets, iou_thresh=0.8, pred_labels=sec_pred_labels)
            # thr_proposals, thr_labels, thr_regression_targets = self.select_training_samples(thr_proposals, targets, iou_thresh=0.5)
            thr_boxfeatures = self.box_roi_pool(features,thr_proposals,image_shapes)
            thr_boxfeatures = self.thr_box_head(thr_boxfeatures)
            thr_class_logits, thr_box_regression = self.thr_box_predictor(thr_boxfeatures)

            # compute loss
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
            # first stage
            box_features = self.box_roi_pool(features, proposals, image_shapes)
            box_features = self.roiatt(box_features,features['0'])
            box_features = self.box_head(box_features)
            class_logits, box_regression = self.box_predictor(box_features)
            pred_boxes = self.box_coder.decode(box_regression, proposals)
            pred_boxes = pred_boxes[:,1:]

            proposals_per_image = [len(b) for b in proposals]
            sec_proposals = pred_boxes.split(proposals_per_image,dim=0)
            sec_proposals = [sec.contiguous().view(-1,4) for sec in sec_proposals]

            # second stage
            sec_boxfeatures = self.box_roi_pool(features,sec_proposals,image_shapes)
            sec_boxfeatures = self.sec_box_head(sec_boxfeatures)
            sec_class_logits, sec_box_regression = self.sec_box_predictor(sec_boxfeatures)
            sec_predboxes = self.box_coder.decode(sec_box_regression, sec_proposals)
            sec_predboxes = sec_predboxes[:,1:]

            sec_proposals_per_image = [len(b) for b in sec_proposals]
            thr_proposals = sec_predboxes.split(sec_proposals_per_image,dim=0)
            thr_proposals = [thr.contiguous().view(-1,4) for thr in thr_proposals]

            # third stage
            thr_boxfeatures = self.box_roi_pool(features,thr_proposals,image_shapes)
            thr_boxfeatures = self.thr_box_head(thr_boxfeatures)
            thr_class_logits, thr_box_regression = self.thr_box_predictor(thr_boxfeatures)
            
            # post process
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