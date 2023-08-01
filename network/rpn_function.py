#!/usr/bin/env python
# coding=utf-8
from typing import List, Optional, Dict, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision

from . import det_utils
from . import boxes as box_ops
from .image_list import ImageList


@torch.jit.unused
def _onnx_get_num_anchors_and_pre_nms_top_n(ob, orig_pre_nms_top_n):
    # type: (Tensor, int) -> Tuple[int, int]
    from torch.onnx import operators
    num_anchors = operators.shape_as_tensor(ob)[1].unsqueeze(0)
    pre_nms_top_n = torch.min(torch.cat(
        (torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype),
         num_anchors), 0))

    return num_anchors, pre_nms_top_n


class AnchorsGenerator(nn.Module):
    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }

    """
    anchors生成器
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.(fpn生成五种尺度特征图,size与ratios元组长度为5)

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Arguments:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """

    def __init__(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorsGenerator, self).__init__()
        '''此处判断若size第一个索引不是元组则将size每个元素单独变为元组,并最终组装为大元组.
        即由(32,64,128,256,512)  -> ((32,),(64),(128,),(256,),(512,))
        而ratios则第一个索引不为元组时,将整体包裹进元组并循环size长度次数,
        即由(0.5,1.0,2.0)  ->  ((0.5,1.0,2.0),(0.5,1.0,2.0),(0.5,1.0,2.0),(0.5,1.0,2.0),(0.5,1.0,2.0))'''
        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        # 每个anchor面积参数对应三个比例,要求元组长度相同
        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device=torch.device("cpu")):
        # type: (List[int], List[float], torch.dtype, torch.device) -> Tensor
        """
        compute anchor sizes
        Arguments:
            scales: sqrt(anchor_area)
            aspect_ratios: h/w ratios
            dtype: float32
            device: cpu/gpu
        """
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1.0 / h_ratios

        # [r1, r2, r3]' * [s1, s2, s3]
        # number of elements is len(ratios)*len(scales)
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        # left-top, right-bottom coordinate relative to anchor center(0, 0)
        # 生成的anchors模板都是以（0, 0）为中心的, shape [len(ratios)*len(scales), 4]
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

        return base_anchors.round()  # round 四舍五入

    def set_cell_anchors(self, dtype, device):
        '''生成anchors模板的方法,此处若为FPN五种尺度,则每种尺度生成三种比例模板anchors,
        cell_anchors为五个元素的列表,每个元素为shape为(3,4)的tensor,
        即三种框和分别的左上角与右下角四个坐标信息'''
        # type: (torch.dtype, torch.device) -> None
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            # suppose that all anchors have the same device
            # which is a valid assumption in the current state of the codebase
            if cell_anchors[0].device == device:
                return

        # 根据提供的sizes和aspect_ratios生成anchors模板
        # anchors模板都是以(0, 0)为中心的anchor
        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        # 计算每个预测特征层上每个滑动窗口的预测目标数
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        """
        anchors position in grid coordinate axis map into origin image
        计算预测特征图对应原始图像上的所有anchors的坐标
        Args:
            grid_sizes: 预测特征矩阵的height和width
            strides: 预测特征矩阵上一步对应原始图像上的步距
        """
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        # 遍历每个预测特征层的grid_size，strides和cell_anchors(每个迭代对象应为长度为尺度个数,即长度为5的列表)
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            # shape: List[长度为grid_width] 对应原图上的x坐标(列)
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            # shape: List[长度为grid_height] 对应原图上的y坐标(行)
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height

            # 计算预测特征矩阵上每个点对应原图上的坐标(anchors模板的坐标偏移量)
            # torch.meshgrid函数分别传入行坐标和列坐标，生成网格行坐标矩阵和网格列坐标矩阵
            # shift1_y和shift_x都生成维度为[grid_height, grid_width]的矩阵,分别对应各点的x.y坐标相对于0.0点的偏移量
            # 即共有grid_height*grid_width个anchor中心点
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            # 计算anchors坐标(xmin, ymin, xmax, ymax)在原图上的坐标偏移量
            # shape: tensor[grid_width*grid_height, 4]特征图每个像素点都生成一组anchors,即偏移量的个数为width*height
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            # 将anchors模板与原图上的坐标偏移量相加得到原图上所有anchors的坐标信息(shape不同时会使用广播机制)
            # reshape到(-1，4)之前的shape为(该尺度特征图偏移量个数,该尺度特征图base_anchor数量3,两点的xy坐标4)
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchor.reshape(-1, 4))

        return anchors  # List[Tensor(all_num_anchors, 4),...]此处list长度为特征图尺度个数为5

    def cached_grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        """将计算得到的所有anchors信息进行缓存"""
        key = str(grid_sizes) + str(strides)
        # self._cache是字典类型
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list, feature_maps):
        '''前向计算过程'''
        # type: (ImageList, List[Tensor]) -> List[Tensor]
        # 获取每个预测特征层的尺寸List[torch.size(height, width),...]长度为5
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])

        # 获取输入图像的height和width
        image_size = image_list.tensors.shape[-2:]

        # 获取变量类型和设备类型
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        # one step in feature map equate n pixel stride in origin image
        # 计算特征层上的一步等于原始图像上的步长,g的格式为torch.size(height,weight)
        # [[hs,ws],...]长度为5
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]

        # 根据提供的sizes和aspect_ratios生成anchors模板
        # cell_anchors维度为List[tensor( 各层len(ratios)*len(scales) = 3, 坐标4),...]长度为5
        self.set_cell_anchors(dtype, device)

        # 计算/读取所有anchors的坐标信息（这里的anchors信息是映射到原图上的所有anchors信息，不是anchors模板）
        # 得到的是一个list列表，对应每张预测特征图映射回原图的anchors坐标信息
        # List[Tensor(num_anchors, 4),...]此处list长度为特征图尺度个数为5
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)

        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])
        # 遍历一个batch中的每张图像
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            # 遍历每张预测特征图映射回原图的anchors坐标信息
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        # anchors_inimage为长度为5的列表,每个元素为一个预测层生成的anchors坐标
        # anchors是个list，每个元素为一张图像的所有anchors信息,长度即为batchsize(图像个数)2
        # 此处cat用于将每一张图像的所有预测特征层的anchors坐标信息拼接在一起,
        # 拼接后长度为2,每个元素为一张图像上所有生成的anchors坐标List[tensor(all_num_anchors,4),...]
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        # Clear the cache in case that memory leaks.
        self._cache.clear()
        return anchors


class RPNHead(nn.Module):
    """
    add a RPN head with classification and regression
    通过滑动窗口计算预测目标概率与bbox regression参数

    Arguments:
        in_channels: number of channels of the input feature
        num_anchors: number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        # 3x3 滑动窗口
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 计算预测的目标分数（这里的目标只是指前景或者背景）
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        nn.init.normal_(self.cls_logits.weight, mean=0., std=0.01)
        # nn.init.constant_(self.cls_logits.bias, -2.0)
        nn.init.constant_(self.cls_logits.bias, 0.) 
        
        # 计算预测的目标bbox regression参数
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
        nn.init.normal_(self.bbox_pred.weight, std=0.01)
        nn.init.constant_(self.bbox_pred.bias, 0.)
        # for layer in self.children():
        #     if isinstance(layer, nn.Conv2d):
        #         torch.nn.init.normal_(layer.weight, std=0.01)
        #         torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        logits = []
        bbox_reg = []
        # List[(B,C,H,W),...]长度为5,conv2d要求进入的tensor维度有4个
        # logits维度List[tensor(B,num_anchor = 3,H,W),...],bbox_reg维度List[tensor(B,num_anchor*4 = 12,H,W),...]长度均为5
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def permute_and_flatten(layer, N, A, C, H, W):
    # type: (Tensor, int, int, int, int, int) -> Tensor
    """
    调整tensor顺序，并进行reshape
    Args:
        layer: 预测特征层上预测的目标概率或bboxes regression参数
        N: batch_size
        A: anchors_num_per_position
        C: classes_num or 4(bbox coordinate)
        H: height
        W: width

    Returns:
        layer: 调整tensor顺序，并reshape后的结果[N, -1, C]
    """
    # view和reshape功能是一样的，先展平所有元素在按照给定shape排列
    # view函数只能用于内存中连续存储的tensor，permute等操作会使tensor在内存中变得不再连续，此时就不能再调用view函数
    # reshape则不需要依赖目标tensor是否在内存中是连续的
    # [batch_size, anchors_num_per_position * (C or 4), height, width]
    layer = layer.view(N, A, C, H, W)
    # 调换tensor维度
    layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, -1, C]
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    # type: (List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    对box_cls和box_regression两个list中的每个预测特征层的预测信息
    的tensor排列顺序以及shape进行调整 -> [N, -1, C]
    Args:
        box_cls: 每个预测特征层上的预测目标概率
        box_regression: 每个预测特征层上的预测目标bboxes regression参数

    Returns:

    """
    box_cls_flattened = []
    box_regression_flattened = []

    # 遍历每个预测特征层
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        # [batch_size, anchors_num_per_position * classes_num, height, width] 
        # 每个点生成k个anchor,每个anchor产生一个概率分数
        # 注意，当计算RPN中的proposal时，classes_num=1,只区分目标和背景
        # [N = 2, A*C = 3*1, h, w]
        N, AxC, H, W = box_cls_per_level.shape
        # # [batch_size, anchors_num_per_position * 4, height, width]
        Ax4 = box_regression_per_level.shape[1]
        # anchors_num_per_position
        A = Ax4 // 4
        # classes_num
        C = AxC // A

        # [N, -1, C],  -1处应为H*W*A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        # [N, -1, C]
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    # 将所有特征层的anchor预测信息在dim=1维度,即5个预测特征层的维度拼接,
    # 并将前两个维度展平(flatten与reshape实现的功能一致)
    # 即将N与-1处推理的维度结合,最终维度为tensor( N*anchor_num = 2*217413, 1或4)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)  # start_dim, end_dim
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


class RegionProposalNetwork(torch.nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Arguments:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[str]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[str]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
        'pre_nms_top_n': Dict[str, int],
        'post_nms_top_n': Dict[str, int],
    }

    def __init__(self, anchor_generator, head,
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # 当iou大于fg_iou_thresh(0.7)时视为正样本
            bg_iou_thresh,  # 当iou小于bg_iou_thresh(0.3)时视为负样本
            allow_low_quality_matches=True
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction  # 256, 0.5
        )

        # use during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        """
        计算每个anchors最匹配的gt,并划分为正样本,背景以及废弃的样本
        Args:
            anchors: (List[Tensor])
            targets: (List[Dict[Tensor])
        Returns:
            labels: 标记anchors归属类别(1, 0, -1分别对应正样本,背景,废弃的样本)
                    注意,在RPN中只有前景和背景,所有正样本的类别都是1,0代表背景
            matched_gt_boxes:与anchors匹配的gt
        """
        labels = []
        matched_gt_boxes = []
        # 遍历每张图像的anchors和targets
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            device = anchors_per_image.device
            if gt_boxes.numel() == 0:
                
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                # 计算anchors与真实bbox的iou信息
                # set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                # tensor(num_gt 3, num_anchor 217413)
                match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)

                # 计算每个anchors与gt匹配iou最大的索引(如果iou<0.3索引置为-1, 0.3<iou<0.7索引为-2)
                # tensor(217413)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                # 这里使用clamp设置下限0是为了方便取每个anchors对应的gt_boxes信息
                # 负样本和舍弃的样本都是负值，所以为了防止越界直接置为0
                # 因为后面是通过labels_per_image变量来记录正样本位置的，
                # 所以负样本和舍弃的样本对应的gt_boxes信息并没有什么意义，
                # 反正计算目标边界框回归损失时只会用到正样本。

                # tensor(217413,4)即每个anchor匹配到的gtbox坐标,
                # 只有原先为正样本的索引取到的坐标信息是有效的,其余坐标都取到clamp限制的0索引处,
                # 只为方便代码运行的操作
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                # 记录所有anchors匹配后的标签(正样本处标记为1，负样本处标记为0，丢弃样本处标记为-1)
                # tensor(217413) 
                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_per_image[bg_indices] = torch.tensor(0.0,device=device)

                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_per_image[inds_to_discard] = torch.tensor(-1.0,device=device)

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        # type: (Tensor, List[int]) -> Tensor
        """
        获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值
        Args:
            objectness: Tensor(每张图像的预测目标概率信息 )
            num_anchors_per_level: List(每个预测特征层上的预测的anchors个数)
        Returns:

        """
        r = []  # 记录每个预测特征层上预测目标概率前pre_nms_top_n的索引信息
        offset = 0
        # 遍历每个预测特征层上的预测目标概率信息
        # tensor.split方法按照num_per_level的列表,在axis=1上分割,每次取出一个预测层上的anchors的概率信息
        # ob tensor(2,163200) tensor(2,40800) ...
        for ob in objectness.split(num_anchors_per_level, 1):
            if torchvision._is_tracing():
                num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(ob, self.pre_nms_top_n())
            else:
                # 预测特征层上的预测的anchors个数
                num_anchors = ob.shape[1]  
                # 选择阈值 2000 与该层anchors个数两个值中较小的一个值作为使用阈值
                pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors) 

            # Returns the k largest elements of the given input tensor along a given dimension
            # tensor.topk方法返回指定个数个tensor排序的内容与索引(返回两个对象),此处只要前topn 2000 个概率值的索引
            # offset为偏移量,因为每次使用topk方法只用于各层的proposal,但索引返回时需要加上上一层所有proposal数量
            # 返回值 tensor(2,8663)因最后一层只有663个anchor,而其他四层则均为2000个
            # (形成可迭代对象添加至列表,然后在拼接)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]
        """
        筛除小boxes框,nms处理,根据预测概率获取前post_nms_top_n个目标
        Args:
            proposals: 预测的bbox坐标 tensor(2,217413,4)
            objectness: 预测的目标概率 tensor(434826,1)
            image_shapes: batch中每张图片的size信息 List[(h1,w1),(h2,w2)]
            num_anchors_per_level: 每个预测特征层上预测anchors的数目 List[163200,40800,10200,2550,663]

        Returns:

        """
        num_images = proposals.shape[0]
        device = proposals.device

        # do not backprop throught objectness
        # reshape后为 tensor(2,217413)
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        # Returns a tensor of size size filled with fill_value
        # levels负责记录分隔不同预测特征层上的anchors索引信息,即(0,1,2,3,4)分别指示五层特征层的anchor
        # tensor (217413) (0-163200,1-40800,2-10200,3-2550,4-663)
        levels = [torch.full((n, ), idx, dtype=torch.int64, device=device)
                  for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, 0)

        # Expand this tensor to the same size as objectness
        # 将第一个维度复制为Batchsize的数值,即图片个数,使维度和objectness的维度一致
        # tensor(2,217413)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        # 获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值
        # tensor(2,8663)
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        # tensor(2) ([0,1])
        image_range = torch.arange(num_images, device=device)
        # tensor(2,1)
        batch_idx = image_range[:, None]  

        # 根据每个预测特征层预测概率排前pre_nms_top_n的anchors索引值获取相应概率信息
        # 两个idx的维度均为 tensor(2,x),索引时分别返回两个图像中排前pre_nms_topn 2000的值
        # 两者维度均为tensor(2,8663)
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        # 预测概率排前pre_nms_top_n的anchors索引值获取相应bbox坐标信息
        # tensor(2,8663,4)
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        # 遍历每张图像的相关预测信息
        # 此处除最后image_shapes为长度为2的列表,其余均为tensor,且第一个维度值为2(batch_size)
        # zip遍历时tensor按第一个维度拆开循环返回,list则逐元素返回
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            # 调整预测的boxes信息，将越界的坐标调整到图片边界上
            # 传入的boxes tensor(8663,4), img_shape为tuple(h,w)
            # 传出的boxes tensor(8663,4)不会减少boxes的数量
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # 返回boxes满足宽，高都大于min_size的索引 tensor(N)
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            # boxes tensor(N,4)   scores tensor(N)   lvl tensor(N)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # 移除小概率boxes，参考下面这个链接
            # https://github.com/pytorch/vision/pull/3205
            keep = torch.where(torch.ge(scores, self.score_thresh))[0]  # ge: >=
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions
            # 上步nms操作后已将概率分数进行过排序,此处直接切片前top_n个keep索引即可
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            # List[tensor(2000,4),...] 
            # List[tensor(2000),...]
            # 长度均为2(batch size)
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        计算RPN损失,包括类别损失(前景与背景),bbox regression损失
        Arguments:
            objectness (Tensor)：预测的前景概率
            pred_bbox_deltas (Tensor):预测的bbox regression
            labels (List[Tensor])：真实的标签 1, 0, -1(batch中每一张图片的labels对应List的一个元素中)
            regression_targets (List[Tensor]):真实的bbox regression

        Returns:
            objectness_loss (Tensor) : 类别损失
            box_loss (Tensor)：边界框回归损失
        """
        # 按照给定的batch_size_per_image, positive_fraction选择正负样本
        # List[tensor(217413),...]长度为2 为索引值,除抽样的样本外均为0
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        
        # 将一个batch中的所有正负样本List(Tensor)分别拼接在一起，并获取非零位置的索引
        # sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        # 从模板0 mask(维度同anchors数量一致)中将抽样位置的索引变为1，后用where找出不为0的位置
        # 这样的操作将抽样后打乱的序列索引,按前后顺序排列好
        # shape为tensor(B 2*128) tensor(B 2*128)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        # sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        # 将所有正负样本索引拼接在一起
        # sampled_inds tensor(512)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        # tensor(434826,1) -> tensor(434826)
        objectness = objectness.flatten()

        # tensor(434826)
        labels = torch.cat(labels, dim=0)
        # tensor(434826,4)
        regression_targets = torch.cat(regression_targets, dim=0)

        # 计算边界框回归损失
        box_loss = det_utils.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            size_average=False
        ) / (sampled_inds.numel())
        # box_loss = F.l1_loss(
        #     pred_bbox_deltas[sampled_pos_inds],
        #     regression_targets[sampled_pos_inds],
        #     reduction="sum"
        # ) / (sampled_inds.numel())
        # 计算目标预测概率损失
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def forward(self,
                images,        # type: ImageList
                features,      # type: Dict[str, Tensor]
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        # features是所有预测特征层组成的OrderedDict
        # 此处取出字典中所有值并变为列表
        features = list(features.values())

        # 计算每个预测特征层上的预测目标概率和bboxes regression参数
        # objectness和pred_bbox_deltas都是list
        # objectness维度List[tensor(B = 2,num_anchor = 3,H,W),...],
        # pred_box维度List[tensor(B = 2,num_anchor*4 = 12,H,W),...]长度均为5
        objectness, pred_bbox_deltas = self.head(features)

        # 生成一个batch图像的所有anchors信息,list(tensor)元素个数等于batch_size
        # 长度为2,每个元素为一张图像上5层feature所有生成的anchors坐标
        # List[tensor(all_num_anchors=217413,4),...]
        anchors = self.anchor_generator(images, features)

        # batch_size=2
        num_images = len(anchors)

        # numel() Returns the total number of elements in the input tensor.
        # 计算每个预测特征层上的对应的anchors数量
        # 长度为特征层数量的List[163200,40800,10200,2550,663]
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        # 调整内部tensor格式以及shape
        # tensor(B*anchor_num = 2*217413, 1)   tensor(B*anchor_num = 2*217413, 4)
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness,
                                                                    pred_bbox_deltas)

        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        # proposal作为faster R-CNN的参数项进入网络,不进行反向传播,此处detach后消除梯度信息
        # 将预测的bbox regression参数应用到anchors上得到最终预测bbox坐标
        # proposals  shape: tensor(Batch 2, Num_anchor 217413, 4)
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4) 

        # 筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
        # List[tensor(2000,4),...] 
        # List[tensor(2000),...]
        # 长度均为2(batch size)        
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None
            # 计算每个anchors最匹配的gt，并将anchors进行分类，前景，背景以及废弃的anchors
            # 传入的anchors List[tensor(217413,4),...]长度为2
            # 返回labels List[tensor(217413),...]长度为2  (正样本处标记为1，负样本处标记为0，丢弃样本处标记为-1)
            # matched_gt_boxes List[tensor(217413,4),...]长度为2  即每个anchor匹配到的gtbox坐标
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            # 结合anchors以及对应的gt，计算regression参数
            # tuple(tensor(217413,4),...)长度为2
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)

            # 输入的参数objectness tensor(434826,1)  pred_deltas tensor(434826,4)
            # labels  List[tensor(217413),...]长度为2
            # regression  tuple(tensor(217413,4),...)长度为2
            # 回归的损失使用对anchor预测的回归参数t*与该anchor匹配的gtbox得出的实际偏移量t计算
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }
        # boxes为list[(2000,4),...]长度为2
        # losses为字典
        return boxes, losses
