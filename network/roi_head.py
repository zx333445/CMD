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
        class_logits : 预测类别概率信息,shape=[num_anchors, num_classes]
        box_regression : 预测边目标界框回归信息
        labels : 真实类别信息
        regression_targets : 真实目标边界框信息

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    # List[tensor(512),...]长度为2  ->  tensor(1024)
    # List[tensor(512,4),...]长度为2  ->  tensor(1024,4)
    labels = torch.cat(labels, dim=0) 
    regression_targets = torch.cat(regression_targets, dim=0)

    # 计算类别损失信息
    # class_logits shape tensor(1024,5)
    classification_loss = F.cross_entropy(class_logits, labels)
    # 测试Focalloss ?
    # focal = FocalLoss(3) 
    # import ipdb;ipdb.set_trace()
    # classification_loss = focal(class_logits, labels)
    
    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    # 返回标签类别大于0的索引
    # sampled_pos_inds_subset = torch.nonzero(torch.gt(labels, 0)).squeeze(1)
    sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]

    # 返回标签类别大于0位置的类别信息
    labels_pos = labels[sampled_pos_inds_subset]

    # shape=[num_proposal, num_classes]
    # tensor(1024,84) -> tensor(1024,21,4)
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)
        # 获取指定索引proposal的指定类别box信息
        # 此处切片若有5个正样本,例如索引为[[200,209,511,1023],[2,5,5,8]],
        # 则分别从第一个维度与第二个维度选取对应索引的样本与类别,最终shape为tensor(5,4)
    # 计算边界框损失信息
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
        为每个proposal匹配对应的gt_box,并划分到正负样本中
        Args:
            proposals:
            gt_boxes:
            gt_labels:

        Returns:

        """
        matched_idxs = []
        labels = []
        # 遍历每张图像的proposals, gt_boxes, gt_labels信息,长度为2的列表逐个返回元素
        # tensor(2000,4) tensor(num_gt,4) tensor(num_gt)
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            device = proposals_in_image.device
            if gt_boxes_in_image.numel() == 0:  # 该张图像中没有gt框，为背景
                # background image
                
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                # 计算proposal与每个gt_box的iou重合度 tensor(num_gt,2000)
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)

                # 计算proposal与每个gt_box匹配的iou最大值，并记录索引，
                # iou < low_threshold索引值为 -1， low_threshold <= iou < high_threshold索引值为 -2
                # 此处没有中间值,没有-2的索引
                # shape 为tensor(2000)分别匹配的gtbox的索引
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                # 限制最小值，防止匹配标签时出现越界的情况
                # 注意-1, -2对应的gt索引会调整到0,获取的标签类别为第0个gt的类别（实际上并不是）,后续会进一步处理
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                # 获取proposal匹配到的gt对应label值 tensor(2000)
                # 此处用索引匹配gt框,故上一步需要将-1与-2变为0以防匹配出现错误
                # gt_labels没有等于0的值(配置文件中获取),故不会取到0,而负样本在之后会赋值为0,即为背景类对应的值
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # label background (below the low threshold)
                # 将gt索引为-1的类别设置为0，即背景，负样本
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = torch.tensor(0,device=device)

                # label ignore proposals (between low and high threshold)
                # 将gt索引为-2的类别设置为-1, 即废弃样本
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
        # 遍历每张图片的正负样本索引
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            # 记录所有采集样本索引(包括正样本和负样本),将所有不为零的位置的索引均返回
            # 只记录采样的位置(原先正负样本的索引值为1,其余值为0,而此处为对应proposal序列的位置索引)
            # img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        # List[tensor(512),...]长度为2
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
                                targets     # type: Optional[List[Dict[str, Tensor]]]
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

        # 检查target数据是否为空
        self.check_targets(targets)
        # 如果不加这句，jit.script会不通过(看不懂)
        assert targets is not None

        dtype = proposals[0].dtype
        device = proposals[0].device

        # 获取标注好的boxes以及labels信息
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposal
        # 将gt_boxes拼接到proposal后面
        # List[tensor(2000,4),...]
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        # 为每个proposal匹配对应的gt_box，并划分到正负样本中
        # matched_idxs  tensor(2000)每个proposal对应的gt框的索引
        # labels  tensor(2000)每个proposal对应的label值(负样本为0,即背景类)
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        
        # sample a fixed proportion of positive-negative proposals
        # 按给定数量和比例采样正负样本
        # List[tensor(512),...]长度为2
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)

        # 遍历每张图像 2
        for img_id in range(num_images):
            # 获取每张图像的正负样本索引 tensor(512)即样本在proposal中的位置
            img_sampled_inds = sampled_inds[img_id]
            # 获取对应正负样本的proposals信息 tensor(512,4)
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            # 获取对应正负样本的真实类别信息 tensor(512)
            labels[img_id] = labels[img_id][img_sampled_inds]
            # 获取对应正负样本的gt索引信息 tensor(512)
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            # 获取对应正负样本的gt box信息 tensor(512,4) ?负样本是否匹配到第一个gt框,负样本不参与回归框损失计算
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        # 根据gt和proposal计算边框回归参数（针对gt的）
        # List[tensor(512,4),...]长度为2
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)

        # List[tensor(512,4),...]  List[tensor(512),...]  List[tensor(512,4),...]长度为2
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
        # 预测目标类别数
        num_classes = class_logits.shape[-1]

        # 获取每张图像的预测bbox数量
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        # 根据proposal以及预测的回归参数计算出最终bbox坐标
        # 此处pred_boxes维度为 tensor(1024,5,4) 用于CTC网络时
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # 对预测类别结果进行softmax处理
        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        # 根据每张图像的预测bbox数量分割结果
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        # 遍历每张图像预测信息
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            # 裁剪预测的boxes信息，将越界的坐标调整到图片边界上
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove prediction with the background label
            # 移除索引为0的所有信息（0代表背景）
            # 注意此处boxes第三个维度为坐标,此处对第二个维度(类别维度)切片
            # (512,5,4) -> (512,4,4)
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            # 对每个框都给出每个类别的预测信息,此处操作相当于把每类的预测都算作一个预测实例
            # (512,4,4) -> (2048,4)即proposal提供512个框,而根据类别数(去掉背景类),预测出的框有2048个
            # (512,4) -> (2048)
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            # 移除低概率目标，self.scores_thresh=0.05
            # gt: Computes input > other element-wise.
            # inds = torch.nonzero(torch.gt(scores, self.score_thresh)).squeeze(1)
            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            # 移除小目标
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximun suppression, independently done per class
            # 执行nms处理，执行后的结果会按照scores从大到小进行排序返回
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

            # keep only topk scoring predictions
            # 获取scores排在前topk个预测目标
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

        # 检查targets的数据类型是否正确
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"

        if self.training:
            # 划分正负样本，统计对应gt的标签以及边界框回归信息
            # List[tensor(512,4),...]  List[tensor(512),...]  List[tensor(512,4),...]长度为2
            # 即抽取的正负样本坐标信息,对应的gt框的类别信息,与对应gt框的回归参数
            proposals, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        # 将采集样本通过Multi-scale RoIAlign pooling层
        # box_features_shape: [num_proposals, channel, height, width]
        # 训练时为 tensor(1024,256,7,7) 此处1024充当batch的功能,即同时1024个样本进入网络(2*512)
        box_features = self.box_roi_pool(features, proposals, image_shapes)

        # 通过roi_pooling后的两层全连接层
        # box_features_shape: [num_proposals, representation_size]
        # 训练时为 tensor(1024,1024) (两个全连接层结点个数为1024)
        box_features = self.box_head(box_features)

        # 接着分别预测目标类别和边界框回归参数
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
            # 预测模式时传入proposal List[tensor(1000,4),]长度为1 (预测时一次只进入一张图片)
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

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool    # Multi-scale RoIAlign pooling

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
        为每个proposal匹配对应的gt_box,并划分到正负样本中
        Args:
            proposals:
            gt_boxes:
            gt_labels:
            iou_thresh: 每次进入新的检测头增加iou阈值 [0.5,0.6,0.7]
        Returns:

        """
        matched_idxs = []
        labels = []
        # 因需要依次改变iou阈值,将matcher放到方法中实例化
        # assign ground-truth boxes for each proposal
        proposal_matcher = det_utils.Matcher(
            iou_thresh,  # default: 0.5
            iou_thresh,  # default: 0.5
            allow_low_quality_matches=False)

        # 遍历每张图像的proposals, gt_boxes, gt_labels信息,长度为2的列表逐个返回元素
        # tensor(2000,4) tensor(num_gt,4) tensor(num_gt)
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            device = proposals_in_image.device
            if gt_boxes_in_image.numel() == 0:  # 该张图像中没有gt框，为背景
                # background image
                
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                # 计算proposal与每个gt_box的iou重合度 tensor(num_gt,2000)
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)

                # 计算proposal与每个gt_box匹配的iou最大值，并记录索引，
                # iou < low_threshold索引值为 -1， low_threshold <= iou < high_threshold索引值为 -2
                # 此处没有中间值,没有-2的索引
                # shape 为tensor(2000)分别匹配的gtbox的索引
                matched_idxs_in_image = proposal_matcher(match_quality_matrix)

                # 限制最小值，防止匹配标签时出现越界的情况
                # 注意-1, -2对应的gt索引会调整到0,获取的标签类别为第0个gt的类别（实际上并不是）,后续会进一步处理
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                # 获取proposal匹配到的gt对应label值 tensor(2000)
                # 此处用索引匹配gt框,故上一步需要将-1与-2变为0以防匹配出现错误
                # gt_labels没有等于0的值(配置文件中获取),故不会取到0,而负样本在之后会赋值为0,即为背景类对应的值
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # label background (below the low threshold)
                # 将gt索引为-1的类别设置为0，即背景，负样本
                bg_inds = matched_idxs_in_image == proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = torch.tensor(0,device=device)

                # label ignore proposals (between low and high threshold)
                # 将gt索引为-2的类别设置为-1, 即废弃样本
                ignore_inds = matched_idxs_in_image == proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_in_image[ignore_inds] = torch.tensor(-1,device=device)  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels


    def subsample(self, labels):
        # type: (List[Tensor],List[Tensor]) -> List[Tensor]
        # BalancedPositiveNegativeSampler  
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        # 遍历每张图片的正负样本索引
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            # 记录所有采集样本索引(包括正样本和负样本),将所有不为零的位置的索引均返回
            # 只记录采样的位置(原先正负样本的索引值为1,其余值为0,而此处为对应proposal序列的位置索引)
            # img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        # List[tensor(512),...]长度为2
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
                                iou_thresh  # type: float
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
        # 检查target数据是否为空
        self.check_targets(targets)
        # 如果不加这句，jit.script会不通过(看不懂)
        assert targets is not None

        dtype = proposals[0].dtype
        device = proposals[0].device

        # 获取标注好的boxes以及labels信息
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposal
        # 将gt_boxes拼接到proposal后面
        # 只在第一轮检测时添加
        # List[tensor(2000,4),...]
        # if iou_thresh == 0.5:
        #     proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        # 为每个proposal匹配对应的gt_box，并划分到正负样本中
        # matched_idxs  tensor(2000)每个proposal对应的gt框的索引
        # labels  tensor(2000)每个proposal对应的label值(负样本为0,即背景类)
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, iou_thresh)

        # sample a fixed proportion of positive-negative proposals
        # 按给定数量和比例采样正负样本
        # List[tensor(512),...]长度为2
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)

        # 遍历每张图像
        for img_id in range(num_images):
            # 获取每张图像的正负样本索引 tensor(512)即样本在proposal中的位置
            img_sampled_inds = sampled_inds[img_id]
            # 获取对应正负样本的proposals信息 tensor(512,4)
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            # 获取对应正负样本的真实类别信息 tensor(512)
            labels[img_id] = labels[img_id][img_sampled_inds]
            # 获取对应正负样本的gt索引信息 tensor(512)
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            # 获取对应正负样本的gt box信息 tensor(512,4) ?负样本是否匹配到第一个gt框,负样本不参与回归框损失计算
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        # 根据gt和proposal计算边框回归参数（针对gt的）
        # List[tensor(512,4),...]长度为2
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)

        # List[tensor(512,4),...]  List[tensor(512),...]  List[tensor(512,4),...]长度为2
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

        Returns:

        """
        device = class_logits.device
        # 预测目标类别数
        num_classes = class_logits.shape[-1]

        # 获取每张图像的预测bbox数量
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        # 根据proposal以及预测的回归参数计算出最终bbox坐标
        # 此处pred_boxes维度为 tensor(1024,3,4) 用于CTC网络时
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # 对预测类别结果进行softmax处理
        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        # 根据每张图像的预测bbox数量分割结果
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        # 遍历每张图像预测信息
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            # 裁剪预测的boxes信息，将越界的坐标调整到图片边界上
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove prediction with the background label
            # 移除索引为0的所有信息（0代表背景）
            # 注意此处boxes第三个维度为坐标,此处对第二个维度(类别维度)切片
            # (512,3,4) -> (512,2,4)
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            # 对每个框都给出每个类别的预测信息,此处操作相当于把每类的预测都算作一个预测实例
            # (512,2,4) -> (1024,4)即proposal提供512个框,而根据类别数(去掉背景类),预测出的框有1024个
            # (512,2) -> (1024)
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            # 移除低概率目标，self.scores_thresh=0.05
            # gt: Computes input > other element-wise.
            # inds = torch.nonzero(torch.gt(scores, self.score_thresh)).squeeze(1)
            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            # 移除小目标
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximun suppression, independently done per class
            # 执行nms处理，执行后的结果会按照scores从大到小进行排序返回
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

            # keep only topk scoring predictions
            # 获取scores排在前topk个预测目标
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

        # 检查targets的数据类型是否正确
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"

        
        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {} 
        if self.training:
            # 划分正负样本，统计对应gt的标签以及边界框回归信息
            # List[tensor(512,4),...]  List[tensor(512),...]  List[tensor(512,4),...]长度为2
            # 即抽取的正负样本坐标信息,对应的gt框的类别信息,与对应gt框的回归参数
            proposals, labels, regression_targets = self.select_training_samples(proposals, targets, iou_thresh=0.5)
            
            # 将采集样本通过Multi-scale RoIAlign pooling层
            # box_features_shape: [num_proposals, channel, height, width]
            # 训练时为 tensor(1024,256,7,7) 此处1024充当batch的功能,即同时1024个样本进入网络(2*512)
            box_features = self.box_roi_pool(features, proposals, image_shapes)

            # 通过roi_pooling后的两层全连接层
            # box_features_shape: [num_proposals, representation_size]
            # 训练时为 tensor(1024,1024) (两个全连接层结点个数为1024)
            box_features = self.box_head(box_features)

            # 接着分别预测目标类别和边界框回归参数
            class_logits, box_regression = self.box_predictor(box_features)
            
            # 将regression作用至proposal,后再从中进行选择正负样本训练
            # 注意需要去掉预测为背景类的框(试试不去掉的,后续label生成时需改为从0开始)
            pred_boxes = self.box_coder.decode(box_regression, proposals)
            pred_boxes = pred_boxes[:,1:]
            
            # import ipdb;ipdb.set_trace()
            # 将预测后的结果再返回为传入时的格式List[tensor(512,4),...]长度为2
            proposals_per_image = [len(b) for b in proposals]
            sec_proposals = pred_boxes.split(proposals_per_image,dim=0)
            sec_proposals = [sec.contiguous().view(-1,4) for sec in sec_proposals]

            # 生成pred_labels,因预测时对每个类别都生成一个实例,因此只需按照boxes维度生成类别数的重复即可
            # 在挑选错分样本前已经匹配过正负样本,因此挑选训练样本时,相当于优先选择定位精准但分类错误的样本
            pred_labels = torch.arange(1,pred_boxes.shape[1]+1,device=proposals[0].device).expand_as(torch.randn(pred_boxes.shape[0],pred_boxes.shape[1]))
            pred_labels = pred_labels.split(proposals_per_image,dim=0)
            pred_labels = [lab.contiguous().view(-1) for lab in pred_labels]

            # 第二次循环筛选过程与检测过程(此处将iou阈值设为0.6)
            # 注意box_head和box_predictor需要重新实例化,和第一轮检测头不共享参数
            # 最终返回用于第三轮循环筛选的thr_proposals
            sec_proposals, sec_labels, sec_regression_targets = self.select_training_samples(sec_proposals, targets, iou_thresh=0.6)
            sec_boxfeatures = self.box_roi_pool(features,sec_proposals,image_shapes)
            sec_boxfeatures = self.sec_box_head(sec_boxfeatures)
            sec_class_logits, sec_box_regression = self.sec_box_predictor(sec_boxfeatures)
            sec_predboxes = self.box_coder.decode(sec_box_regression, sec_proposals)
            sec_predboxes = sec_predboxes[:,1:]

            # 注意这里proposals可能出现不够的情况,重新计算一下分割数量即可(样本采样器出现了重复,导致样本不够的,已解决)
            sec_proposals_per_image = [len(b) for b in sec_proposals]
            thr_proposals = sec_predboxes.split(sec_proposals_per_image,dim=0)
            thr_proposals = [thr.contiguous().view(-1,4) for thr in thr_proposals]

            # 因proposal不够,所以预测boxes的维度也会发生变化,因此重新给出labels的维度
            sec_pred_labels = torch.arange(1,sec_predboxes.shape[1]+1,device=proposals[0].device).expand_as(torch.randn(sec_predboxes.shape[0],sec_predboxes.shape[1]))
            sec_pred_labels = sec_pred_labels.split(sec_proposals_per_image,dim=0)
            sec_pred_labels = [lab.contiguous().view(-1) for lab in sec_pred_labels]

            # 第三次循环(iou阈值0.7)
            thr_proposals, thr_labels, thr_regression_targets = self.select_training_samples(thr_proposals, targets, iou_thresh=0.7)
            thr_boxfeatures = self.box_roi_pool(features,thr_proposals,image_shapes)
            thr_boxfeatures = self.thr_box_head(thr_boxfeatures)
            thr_class_logits, thr_box_regression = self.thr_box_predictor(thr_boxfeatures)


            # 分别计算三次检测头的损失按类加和
            assert labels is not None and regression_targets is not None

            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            
            sec_loss_classifier,sec_loss_box_reg = fastrcnn_loss(
                sec_class_logits, sec_box_regression, sec_labels, sec_regression_targets)
            
            thr_loss_classifier,thr_loss_box_reg = fastrcnn_loss(
                thr_class_logits, thr_box_regression, thr_labels, thr_regression_targets)
            
            # 试一下将后两阶段的回归损失减小(考虑已经有了比较好的定位效果)
            losses = {
                "loss_classifier": loss_classifier + sec_loss_classifier + thr_loss_classifier,
                "loss_box_reg": loss_box_reg + sec_loss_box_reg + thr_loss_box_reg
            }


        else:
            # 第一轮检测返回sec_proposals
            box_features = self.box_roi_pool(features, proposals, image_shapes)
            box_features = self.box_head(box_features)
            class_logits, box_regression = self.box_predictor(box_features)
            pred_boxes = self.box_coder.decode(box_regression, proposals)
            pred_boxes = pred_boxes[:,1:]
            # 预测模式时传入proposal List[tensor(1000,4),]长度为1 (预测时一次只进入一张图片)
            proposals_per_image = [len(b) for b in proposals]
            sec_proposals = pred_boxes.split(proposals_per_image,dim=0)
            sec_proposals = [sec.contiguous().view(-1,4) for sec in sec_proposals]

            # 第二轮检测返回thr_proposals
            sec_boxfeatures = self.box_roi_pool(features,sec_proposals,image_shapes)
            sec_boxfeatures = self.sec_box_head(sec_boxfeatures)
            sec_class_logits, sec_box_regression = self.sec_box_predictor(sec_boxfeatures)
            sec_predboxes = self.box_coder.decode(sec_box_regression, sec_proposals)
            sec_predboxes = sec_predboxes[:,1:]
            # 预测时没有选择样本过程,因此需要重新计算proposal的数量
            sec_proposals_per_image = [len(b) for b in sec_proposals]
            thr_proposals = sec_predboxes.split(sec_proposals_per_image,dim=0)
            thr_proposals = [thr.contiguous().view(-1,4) for thr in thr_proposals]

            # 第三轮检测返回类别预测与检测回归参数
            thr_boxfeatures = self.box_roi_pool(features,thr_proposals,image_shapes)
            thr_boxfeatures = self.thr_box_head(thr_boxfeatures)
            thr_class_logits, thr_box_regression = self.thr_box_predictor(thr_boxfeatures)
            
            # 将结果作用至proposal并后处理为最终检测框
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


class RRAMRoIHeads(torch.nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }
    def __init__(self,
                 box_roi_pool,   # Multi-scale RoIAlign pooling
                 box_ramhead,    # RAMHead
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
        super().__init__()

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
        self.box_ramhead = box_ramhead      # RamHead
        self.box_head = box_head            # TwoMLPHead
        self.box_predictor = box_predictor  # FastRCNNPredictor

        self.score_thresh = score_thresh  # default: 0.05
        self.nms_thresh = nms_thresh      # default: 0.5
        self.detection_per_img = detection_per_img  # default: 100

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []

        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            device = proposals_in_image.device
            if gt_boxes_in_image.numel() == 0:  # 该张图像中没有gt框，为背景
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

                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = torch.tensor(0,device=device)

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
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)

        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]

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

        self.check_targets(targets)
        assert targets is not None

        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        proposals = self.add_gt_proposals(proposals, gt_boxes)
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        
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
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

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

            # (512,5,4) -> (512,4,4)
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

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
        box_features = self.box_ramhead(box_features,features['1']) 
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