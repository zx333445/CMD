#!/usr/bin/env python
# coding=utf-8
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import complete_box_iou_loss, distance_box_iou_loss, FrozenBatchNorm2d, generalized_box_iou_loss


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
        for matched_idxs_per_image in matched_idxs:
            positive = torch.where(torch.ge(matched_idxs_per_image, 1))[0]
            # negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)
            negative = torch.where(torch.eq(matched_idxs_per_image, 0))[0]

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            # Returns a random permutation of integers from 0 to n - 1.
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
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
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)

        # targets_dx, targets_dy, targets_dw, targets_dh
        targets = self.encode_single(reference_boxes, proposals)

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
            rel_codes: bbox regression parameters  
            boxes: anchors/proposals

        Returns:

        """
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0) 

        box_sum = 0
        for val in boxes_per_image:
            box_sum += val

        pred_boxes = self.decode_single(
            rel_codes, concat_boxes
        )

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
        """
        boxes = boxes.to(rel_codes.dtype)

        # xmin, ymin, xmax, ymax
        widths = boxes[:, 2] - boxes[:, 0]   
        heights = boxes[:, 3] - boxes[:, 1]  
        ctr_x = boxes[:, 0] + 0.5 * widths   
        ctr_y = boxes[:, 1] + 0.5 * heights  

        wx, wy, ww, wh = self.weights  
        dx = rel_codes[:, 0::4] / wx   
        dy = rel_codes[:, 1::4] / wy   
        dw = rel_codes[:, 2::4] / ww   
        dh = rel_codes[:, 3::4] / wh   

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
 
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes  


class BoxLinearCoder:
    """
    The linear box-to-box transform defined in FCOS. The transformation is parameterized
    by the distance from the center of (square) src box to 4 edges of the target box.
    """

    def __init__(self, normalize_by_size: bool = True) -> None:
        """
        Args:
            normalize_by_size (bool): normalize deltas by the size of src (anchor) boxes.
        """
        self.normalize_by_size = normalize_by_size

    def encode(self, reference_boxes: Tensor, proposals: Tensor) -> Tensor:
        """
        Encode a set of proposals with respect to some reference boxes

        Args:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded

        Returns:
            Tensor: the encoded relative box offsets that can be used to
            decode the boxes.

        """

        # get the center of reference_boxes
        reference_boxes_ctr_x = 0.5 * (reference_boxes[..., 0] + reference_boxes[..., 2])
        reference_boxes_ctr_y = 0.5 * (reference_boxes[..., 1] + reference_boxes[..., 3])

        # get box regression transformation deltas
        target_l = reference_boxes_ctr_x - proposals[..., 0]
        target_t = reference_boxes_ctr_y - proposals[..., 1]
        target_r = proposals[..., 2] - reference_boxes_ctr_x
        target_b = proposals[..., 3] - reference_boxes_ctr_y

        targets = torch.stack((target_l, target_t, target_r, target_b), dim=-1)

        if self.normalize_by_size:
            reference_boxes_w = reference_boxes[..., 2] - reference_boxes[..., 0]
            reference_boxes_h = reference_boxes[..., 3] - reference_boxes[..., 1]
            reference_boxes_size = torch.stack(
                (reference_boxes_w, reference_boxes_h, reference_boxes_w, reference_boxes_h), dim=-1
            )
            targets = targets / reference_boxes_size
        return targets

    def decode(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:

        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Args:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.

        Returns:
            Tensor: the predicted boxes with the encoded relative box offsets.

        .. note::
            This method assumes that ``rel_codes`` and ``boxes`` have same size for 0th dimension. i.e. ``len(rel_codes) == len(boxes)``.

        """

        boxes = boxes.to(dtype=rel_codes.dtype)

        ctr_x = 0.5 * (boxes[..., 0] + boxes[..., 2])
        ctr_y = 0.5 * (boxes[..., 1] + boxes[..., 3])

        if self.normalize_by_size:
            boxes_w = boxes[..., 2] - boxes[..., 0]
            boxes_h = boxes[..., 3] - boxes[..., 1]

            list_box_size = torch.stack((boxes_w, boxes_h, boxes_w, boxes_h), dim=-1)
            rel_codes = rel_codes * list_box_size

        pred_boxes1 = ctr_x - rel_codes[..., 0]
        pred_boxes2 = ctr_y - rel_codes[..., 1]
        pred_boxes3 = ctr_x + rel_codes[..., 2]
        pred_boxes4 = ctr_y + rel_codes[..., 3]

        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=-1)
        return pred_boxes


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
        matched_vals, matches = match_quality_matrix.max(dim=0)  # the dimension to reduce.

        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD  # -1
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
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)  # the dimension to reduce.
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

        # gt_pred_pairs_of_highest_quality
        # pre_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        pre_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]


@torch.jit.unused
def _fake_cast_onnx(v: Tensor) -> int:
    return v  # type: ignore[return-value]


def _topk_min(input: Tensor, orig_kval: int, axis: int) -> int:
    """
    ONNX spec requires the k-value to be less than or equal to the number of inputs along
    provided dim. Certain models use the number of elements along a particular axis instead of K
    if K exceeds the number of elements along that axis. Previously, python's min() function was
    used to determine whether to use the provided k-value or the specified dim axis value.

    However, in cases where the model is being exported in tracing mode, python min() is
    static causing the model to be traced incorrectly and eventually fail at the topk node.
    In order to avoid this situation, in tracing mode, torch.min() is used instead.

    Args:
        input (Tensor): The original input tensor.
        orig_kval (int): The provided k-value.
        axis(int): Axis along which we retrieve the input size.

    Returns:
        min_kval (int): Appropriately selected k-value.
    """
    if not torch.jit.is_tracing():
        return min(orig_kval, input.size(axis))
    axis_dim_val = torch._shape_as_tensor(input)[axis].unsqueeze(0)
    min_kval = torch.min(torch.cat((torch.tensor([orig_kval], dtype=axis_dim_val.dtype), axis_dim_val), 0))
    return _fake_cast_onnx(min_kval)


def overwrite_eps(model: nn.Module, eps: float) -> None:
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.

    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps


def _box_loss(
    type: str,
    box_coder: BoxCoder,
    anchors_per_image: Tensor,
    matched_gt_boxes_per_image: Tensor,
    bbox_regression_per_image: Tensor,
    cnf: Optional[Dict[str, float]] = None,
) -> Tensor:
    torch._assert(type in ["l1", "smooth_l1", "ciou", "diou", "giou"], f"Unsupported loss: {type}")

    if type == "l1":
        target_regression = box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
        return F.l1_loss(bbox_regression_per_image, target_regression, reduction="sum")
    elif type == "smooth_l1":
        target_regression = box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
        beta = cnf["beta"] if cnf is not None and "beta" in cnf else 1.0
        return F.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction="sum", beta=beta)
    else:
        bbox_per_image = box_coder.decode_single(bbox_regression_per_image, anchors_per_image)
        eps = cnf["eps"] if cnf is not None and "eps" in cnf else 1e-7
        if type == "ciou":
            return complete_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction="sum", eps=eps)
        if type == "diou":
            return distance_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction="sum", eps=eps)
        # otherwise giou
        return generalized_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction="sum", eps=eps)


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


class SSDMatcher(Matcher):
    def __init__(self, threshold: float) -> None:
        super().__init__(threshold, threshold, allow_low_quality_matches=False)

    def __call__(self, match_quality_matrix: Tensor) -> Tensor:
        matches = super().__call__(match_quality_matrix)

        # For each gt, find the prediction with which it has the highest quality
        _, highest_quality_pred_foreach_gt = match_quality_matrix.max(dim=1)
        matches[highest_quality_pred_foreach_gt] = torch.arange(
            highest_quality_pred_foreach_gt.size(0), dtype=torch.int64, device=highest_quality_pred_foreach_gt.device
        )

        return matches
    

def retrieve_out_channels(model: nn.Module, size: Tuple[int, int]) -> List[int]:
    """
    This method retrieves the number of output channels of a specific model.

    Args:
        model (nn.Module): The model for which we estimate the out_channels.
            It should return a single Tensor or an OrderedDict[Tensor].
        size (Tuple[int, int]): The size (wxh) of the input.

    Returns:
        out_channels (List[int]): A list of the output channels of the model.
    """
    in_training = model.training
    model.eval()

    with torch.no_grad():
        # Use dummy data to retrieve the feature map sizes to avoid hard-coding their values
        device = next(model.parameters()).device
        tmp_img = torch.zeros((1, 3, size[1], size[0]), device=device)
        features = model(tmp_img)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        out_channels = [x.size(1) for x in features.values()]

    if in_training:
        model.train()

    return out_channels