#!/usr/bin/env python
# coding=utf-8
import torch
import math
from typing import List, Tuple
from torch import Tensor


class BalancedPositiveNegativeSampler(object):
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

    def __call__(self, matched_idxs):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        # 遍历每张图像的matched_idxs 
        # tensor(217413)
        for matched_idxs_per_image in matched_idxs:
            # >= 1的为正样本, nonzero返回非零元素索引
            # positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            # tensor(N) N个正样本的索引
            positive = torch.where(torch.ge(matched_idxs_per_image, 1))[0]
            # = 0的为负样本
            # negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)
            negative = torch.where(torch.eq(matched_idxs_per_image, 0))[0]

            # 指定正样本的数量 128
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            # 如果正样本数量不够就直接采用所有正样本
            num_pos = min(positive.numel(), num_pos)
            # 指定负样本数量
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            # 如果负样本数量不够就直接采用所有负样本
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            # Returns a random permutation of integers from 0 to n - 1.
            # 随机选择指定数量的正负样本 
            # 生成正样本个数个打乱排列顺序的整数序列,切片前num_pos个作为抽取的样本索引
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            # tensor(128) tenosr(128)
            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            # tensor(217413)  均为0的tensor mask
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



class HardSampleMiningSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    正样本中优先选择上轮检测器错分的样本,进行二次训练以优化分类结果,需要传入匹配到的实际label与预测结果的label
    添加负样本错分匹配后效果不佳
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
        """
        Arguments:
            matched_idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.
                即为各proposal对应的gt标签,为0是负样本背景类(iou小于阈值),其余>=1的是匹配到的gt框的真实类别标签
            pred_labels: 各proposal对应的预测标签,即上一轮检测器给出的分类预测结果,注意要传入list形式长度为batchsize

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        # 遍历每张图像的matched_idxs 
        # tensor(217413)
        for matched_idxs_per_image, pred_labels_per_image in zip(matched_idxs,pred_labels):
            # >= 1的为正样本, nonzero返回非零元素索引
            # positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            # tensor(N) N个正样本的索引
            # wrong_idx即所有匹配的label不等于预测label的mask,后提取其中的正样本索引对应的相等情况inter_idx
            # 再用inter_idx提取正样本索引中不等的索引
            # 注意positive_idx长度与inter_idx长度相同,但前者为位置索引,后者为True/False的mask
            # hard_positive即所有正样本中错分类的样本
            positive_idx = torch.where(torch.ge(matched_idxs_per_image, 1))[0]
            # import ipdb;ipdb.set_trace()
            wrong_idx = matched_idxs_per_image != pred_labels_per_image
            inter_idx = wrong_idx[positive_idx]
            hard_positive = positive_idx[inter_idx]

            # 指定正样本的数量 128
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            # 如果正样本数量不够指定数量就直接采用所有正样本
            num_pos = min(positive_idx.numel(), num_pos)

            # randomly select positive and negative examples
            # Returns a random permutation of integers from 0 to n - 1.
            # 随机选择指定数量的正负样本 
            # 生成正样本个数个打乱排列顺序的整数序列,切片前num_pos个作为抽取的样本索引
            num_hard = hard_positive.numel()

            # 试一下只用错误分类的正样本
            # perm1 = torch.randperm(hard_positive.numel(), device=positive_idx.device)[:num_hard]
            # pos_idx_per_image = hard_positive[perm1]

            # 如果难样本数量超出指定正样本数量,则直接从中抽取训练用正样本
            # 如果不够数量则从剩余正样本中随机抽取(num_pos-num_hard)个正样本,并和hard样本拼接在一起
            if num_hard >= num_pos:
                perm1 = torch.randperm(hard_positive.numel(), device=positive_idx.device)[:num_pos]
                pos_idx_per_image = hard_positive[perm1]
            else:
                perm1 = torch.randperm(positive_idx[~inter_idx].numel(), device=positive_idx.device)[:(num_pos-num_hard)]
                pos_idx_per_image = torch.cat([hard_positive,positive_idx[~inter_idx][perm1]])
            
            
            # = 0的为负样本
            # negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)
            negative_idx = torch.where(torch.eq(matched_idxs_per_image, 0))[0]
            # 指定负样本数量
            # num_neg = self.batch_size_per_image - num_pos
            num_neg = self.batch_size_per_image - num_hard
            # protect against not enough negative examples
            # 如果负样本数量不够就直接采用所有负样本
            num_neg = min(negative_idx.numel(), num_neg)

            perm2 = torch.randperm(negative_idx.numel(), device=negative_idx.device)[:num_neg]
            neg_idx_per_image = negative_idx[perm2]
            
            # int_idx = wrong_idx[negative_idx]
            # hard_negative = negative_idx[int_idx]
            
            # # 负样本错分数量
            # num_neghard = hard_negative.numel()

            # # 执行同正样本一样的选择操作
            # if num_neghard >= num_neg:
            #     perm2 = torch.randperm(hard_negative.numel(),device=negative_idx.device)[:num_neg]
            #     neg_idx_per_image = hard_negative[perm2]
            # else:
            #     perm2 = torch.randperm(negative_idx[~int_idx].numel(), device=negative_idx.device)[:(num_neg-num_neghard)]
            #     neg_idx_per_image = torch.cat([hard_negative,negative_idx[~int_idx][perm2]])   

            # create binary mask from indices
            # tensor(217413)  均为0的tensor mask
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



@torch.jit._script_if_tracing
def encode_boxes(reference_boxes, proposals, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Encode a set of proposals with respect to some
    reference boxes

    Arguments:
        reference_boxes (Tensor): reference boxes(gt)
        proposals (Tensor): boxes to be encoded(anchors)
        weights:
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    # unsqueeze()
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    # tensor(434826,1)
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    # tensor(434826,1)
    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    # parse widths and heights
    # tensor(434826,1)
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    # parse coordinate of center point
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    # tensor(434826,4)
    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        # type: (Tuple[float, float, float, float], float) -> None
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        结合anchors和与之对应的gt计算regression参数
        Args:
            reference_boxes: List[Tensor] 每个proposal/anchor对应的gt_boxes
            proposals: List[Tensor] anchors/proposals

        Returns: regression parameters

        """
        # 统计每张图像的anchors个数，方便后面拼接在一起处理后在分开
        # reference_boxes和proposal数据结构相同
        # List[217413,217413]
        boxes_per_image = [len(b) for b in reference_boxes]
        # tensor(434826,4)
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)

        # targets_dx, targets_dy, targets_dw, targets_dh
        targets = self.encode_single(reference_boxes, proposals)

        # 返回计算的参数时由tensor(434826,4)-> tuple(tensor(217413,4),...)长度为2
        return targets.split(boxes_per_image, dim=0)

    def encode_single(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    def decode(self, rel_codes, boxes):
        # type: (Tensor, List[Tensor]) -> Tensor
        """

        Args:
            rel_codes: bbox regression parameters   shape:tensor(434826,4)  tensor(1024,3*4)
            boxes: anchors/proposals传入anchors列表(批次中每张图像为一个列表元素)

        Returns:

        """
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        # [217413,217413]
        boxes_per_image = [b.size(0) for b in boxes]
        # tensor(434826,4)
        concat_boxes = torch.cat(boxes, dim=0) # 此处拼接后将一个batch中所有anchor坐标拼接在一起

        # 蜜汁操作,即得到anchors共有多少个
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val

        # 将预测的bbox回归参数应用到对应anchors上得到预测bbox的坐标
        # tensor(434826,4)
        pred_boxes = self.decode_single(
            rel_codes, concat_boxes
        )

        # 防止pred_boxes为空时导致reshape报错
        # tensor(434826,1,4) 
        # 注意此处第二个维度,在ROIHead预测多分类时为类别数(此处CTC网络为3)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)

        return pred_boxes

    def decode_single(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes (bbox regression parameters)
            boxes (Tensor): reference boxes (anchors/proposals)
            此处两个参数,在第一个维度需要一致,第二个维度codes根据类别数变化为(num_classes*4),boxes就为4
        """
        boxes = boxes.to(rel_codes.dtype)

        # xmin, ymin, xmax, ymax
        widths = boxes[:, 2] - boxes[:, 0]   # anchor/proposal宽度
        heights = boxes[:, 3] - boxes[:, 1]  # anchor/proposal高度
        ctr_x = boxes[:, 0] + 0.5 * widths   # anchor/proposal中心x坐标
        ctr_y = boxes[:, 1] + 0.5 * heights  # anchor/proposal中心y坐标

        wx, wy, ww, wh = self.weights  # RPN中为[1,1,1,1], fastrcnn中为[10,10,5,5]
        # 此处切片方式0::4是为了保留第二个维度的存在   [434826,1]
        dx = rel_codes[:, 0::4] / wx   # 预测anchors/proposals的中心坐标x回归参数
        dy = rel_codes[:, 1::4] / wy   # 预测anchors/proposals的中心坐标y回归参数
        dw = rel_codes[:, 2::4] / ww   # 预测anchors/proposals的宽度回归参数
        dh = rel_codes[:, 3::4] / wh   # 预测anchors/proposals的高度回归参数

        # limit max value, prevent sending too large values into torch.exp()
        # self.bbox_xform_clip=math.log(1000. / 16)   4.135
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # xmin
        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymin
        pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        # xmax
        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymax
        pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h

        # 此处使用cat拼接dim=1即可以实现tensor(434826,4)   
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes   # tensor(434826,4)


class Matcher(object):
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLDS': int,
    }

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        # type: (float, float, bool) -> None
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold  # 0.7
        self.low_threshold = low_threshold    # 0.3
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        计算anchors与每个gtboxes匹配的iou最大值,并记录索引,
        iou<low_threshold索引值为-1(负样本), low_threshold<=iou<high_threshold索引值为-2(抛弃样本)
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        # M x N 的每一列代表一个anchor/proposal与所有gt的匹配iou值
        # dim=0即按行数变化为一组,即返回每列的最大值
        # matched_vals代表每列的最大值，即每个anchor与所有gt匹配的最大iou值 tensor(217413)
        # matches对应最大值所在的索引 tensor(217413)
        matched_vals, matches = match_quality_matrix.max(dim=0)  # the dimension to reduce.

        # 如果使用保留最大iou框(小于筛选阈值)的方法,则将索引值copy一份备用
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative (unassigned) values
        # 计算iou小于low_threshold的索引
        # tensor(217413) 均为logical值
        below_low_threshold = matched_vals < self.low_threshold
        # 计算iou在low_threshold与high_threshold之间的索引值
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        # iou小于low_threshold的matches索引置为-1
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD  # -1

        # iou在[low_threshold, high_threshold]之间的matches索引置为-2
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS    # -2

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        # 对于每个gt boxes寻找与其iou最大的anchor，
        # highest_quality_foreach_gt为匹配到的最大iou值
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)  # the dimension to reduce.

        # Find highest quality match available, even if it is low, including ties
        # 寻找每个gt boxes与其iou最大的anchor索引，一个gt匹配到的最大iou可能有多个anchor
        # gt_pred_pairs_of_highest_quality = torch.nonzero(
        #     match_quality_matrix == highest_quality_foreach_gt[:, None]
        # )

        # torch.eq返回与两个传入tensor相同shape的logical值,表示两tensor该位置元素是否相同
        # torch.where返回为Ture的位置的索引值 
        gt_pred_pairs_of_highest_quality = torch.where(
            torch.eq(match_quality_matrix, highest_quality_foreach_gt[:, None])
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        # gt_pred_pairs_of_highest_quality[:, 0]代表是对应的gt index(不需要)
        # pre_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        pre_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        # 保留该anchor匹配gt最大iou的索引，即使iou低于设定的阈值
        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]


def smooth_l1_loss(input, target, beta: float = 1. / 9, size_average: bool = True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    # cond = n < beta
    cond = torch.lt(n, beta)
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
