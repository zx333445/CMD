import torch
import math
from torch import nn
from typing import Tuple, List, Dict, Optional, Union
from torch import nn, Tensor
from .pooling import MultiScaleRoIAlign
import sys
sys.path.append('..')
from .sparse_rcnn_loss import BoxCoder, SparseRCNNLoss
from network.transform import GeneralizedRCNNTransform
from .common import FrozenBatchNorm2d


class DynamicConv(nn.Module):
    def __init__(self, in_channel, inner_channel, pooling_resolution, activation=nn.ReLU, **kwargs):
        super(DynamicConv, self).__init__()
        self.in_channel = in_channel
        self.inner_channel = inner_channel
        self.param_num = in_channel * inner_channel
        self.dynamic_layer = nn.Linear(in_channel, 2 * in_channel * inner_channel)

        self.norm1 = nn.LayerNorm(inner_channel)
        self.norm2 = nn.LayerNorm(in_channel)
        self.activation = activation(**kwargs)
        flatten_dim = in_channel * pooling_resolution ** 2
        self.out_layer = nn.Linear(flatten_dim, in_channel)
        self.norm3 = nn.LayerNorm(in_channel)

    def forward(self, x: torch.Tensor, param_x: torch.Tensor):
        """
        :param x:  [pooling_resolution**2,N * nr_boxes,in_channel]
        :param param_x: [N * nr_boxes, in_channel]
        :return:
        """
        # [N*nr_boxes,49,in_channel]
        x = x.permute(1, 0, 2)
        # [N*nr_boxes, 2*in_channel * inner_channel]
        params = self.dynamic_layer(param_x)
        # [N*nr_boxes,in_channel,inner_channel]
        param1 = params[:, :self.param_num].view(-1, self.in_channel, self.inner_channel)

        x = torch.bmm(x, param1)
        x = self.norm1(x)
        x = self.activation(x)

        # [N*nr_boxes,inner_channel,in_channel]
        param2 = params[:, self.param_num:].view(-1, self.inner_channel, self.in_channel)

        x = torch.bmm(x, param2)
        x = self.norm2(x)
        x = self.activation(x)

        x = x.flatten(1)
        x = self.out_layer(x)
        x = self.norm3(x)
        x = self.activation(x)
        return x


class RCNNHead(nn.Module):
    def __init__(self,
                 in_channel=256,
                 inner_channel=64,
                 num_cls=3,
                 dim_feedforward=2048,
                 nhead=8,
                 dropout=0.1,
                 pooling_resolution=7,
                 activation=nn.ReLU,
                 cls_tower_num=1,
                 reg_tower_num=3, **kwargs):
        super(RCNNHead, self).__init__()
        self.self_attn = nn.MultiheadAttention(in_channel, nhead, dropout=dropout)

        self.inst_interact = DynamicConv(in_channel,
                                         inner_channel,
                                         pooling_resolution=pooling_resolution,
                                         activation=activation,
                                         **kwargs)

        self.feed_forward = nn.Sequential(
            nn.Linear(in_channel, dim_feedforward),
            activation(**kwargs),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, in_channel)
        )

        self.norm1 = nn.LayerNorm(in_channel)
        self.norm2 = nn.LayerNorm(in_channel)
        self.norm3 = nn.LayerNorm(in_channel)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.cls_tower = list()
        for _ in range(cls_tower_num):
            self.cls_tower.append(nn.Linear(in_channel, in_channel, False))
            self.cls_tower.append(nn.LayerNorm(in_channel))
            self.cls_tower.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*self.cls_tower)

        self.reg_tower = list()
        for _ in range(reg_tower_num):
            self.reg_tower.append(nn.Linear(in_channel, in_channel, False))
            self.reg_tower.append(nn.LayerNorm(in_channel))
            self.reg_tower.append(nn.ReLU(inplace=True))
        self.reg_tower = nn.Sequential(*self.reg_tower)

        self.class_logits = nn.Linear(in_channel, num_cls)
        self.bboxes_delta = nn.Linear(in_channel, 4)
        self.box_coder = BoxCoder()

    def forward(self, x: torch.Tensor, params_x: torch.Tensor, boxes: torch.Tensor):
        """
        :param x: [N * nr_boxes,in_channel,pooling_resolution,pooling_resolution]
        :param params_x: [N, nr_boxes, in_channel]
        :param boxes:[N * nr_boxes,4]
        :return:
        """
        nxp, c, _, _ = x.shape
        n, p, _ = params_x.shape
        # [res**2,N * nr_boxes,in_channel]
        x = x.view(nxp, c, -1).permute(2, 0, 1)
        # [nr_boxes, N, in_channel]
        params_x = params_x.permute(1, 0, 2)
        params_attn = self.self_attn(params_x, params_x, value=params_x)[0]
        params_x = self.norm1(params_x + self.dropout1(params_attn))

        params_x = params_x.permute(1, 0, 2).contiguous().view(-1, params_x.size(2))
        # [N*nr_boxes,in_channel]
        param_intersect = self.inst_interact(x, params_x)
        params_x = self.norm2(params_x + self.dropout2(param_intersect))

        param_feedforward = self.feed_forward(params_x)
        # [N*nr_boxes,in_channel]
        out = self.norm3(params_x + self.dropout3(param_feedforward))
        cls_tower = self.cls_tower(out)
        reg_tower = self.reg_tower(out)
        cls_out = self.class_logits(cls_tower)
        reg_delta = self.bboxes_delta(reg_tower)
        pred_bboxes = self.box_coder.decoder(reg_delta, boxes)
        return cls_out.view(n, p, -1), pred_bboxes.view(n, p, -1), out.view(n, p, -1)


class DynamicHead(nn.Module):
    def __init__(self,
                 in_channel=256,
                 inner_channel=64,
                 num_cls=2,
                 dim_feedforward=2048,
                 nhead=8,
                 dropout=0.1,
                 pooling_resolution=7,
                 activation=nn.ReLU,
                 cls_tower_num=1,
                 reg_tower_num=3,
                 num_heads=6,
                 return_intermediate=True,
                 **kwargs):
        super(DynamicHead, self).__init__()
        self.pooling_layer = MultiScaleRoIAlign(
            ['p2', 'p3', 'p4', 'p5'],
            output_size=pooling_resolution,
            sampling_ratio=2
        )
        self.heads = list()
        for _ in range(num_heads):
            self.heads.append(RCNNHead(in_channel,
                                       inner_channel,
                                       num_cls,
                                       dim_feedforward,
                                       nhead,
                                       dropout,
                                       pooling_resolution,
                                       activation,
                                       cls_tower_num,
                                       reg_tower_num,
                                       **kwargs))
        self.heads = nn.ModuleList(self.heads)
        self.return_intermediate = return_intermediate

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            if p.shape[-1] == num_cls:
                nn.init.constant_(p, -math.log((1 - 0.01) / 0.01))

    def forward(self, x, init_boxes, init_params, shapes):
        """
        :param x: dict("p2":(bs,in_channel,h,w),...)
        :param init_boxes:[num_proposal,4]
        :param init_params:[num_proposal,in_channel]
        :param shapes: [(640,640),...]
        :return:
        """
        inter_class_logits = list()
        inter_pred_bboxes = list()
        bs = x['p2'].shape[0]
        bboxes = init_boxes[None, :, :].repeat(bs, 1, 1)
        proposal_features = init_params[None, :, :].repeat(bs, 1, 1)
        class_logits = None
        pred_bboxes = None
        for rcnn_head in self.heads:
            roi_features = self.pooling_layer(x, [box for box in bboxes], shapes)
            class_logits, pred_bboxes, proposal_features = rcnn_head(roi_features, proposal_features,
                                                                     bboxes.view(-1, 4))
            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()
        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)
        return class_logits[None], pred_bboxes[None]


default_cfg = {
    "in_channel": 256,
    "inner_channel": 64,
    "num_cls": 2,
    "dim_feedforward": 2048,
    "nhead": 8,
    "dropout": 0,
    "pooling_resolution": 7,
    "activation": nn.ReLU,
    "cls_tower_num": 1,
    "reg_tower_num": 3,
    "num_heads": 6,
    "return_intermediate": True,
    "num_proposals": 100,
    "backbone": "densenet169",
    "pretrained": True,
    "norm_layer": FrozenBatchNorm2d,
    # loss cfg
    "iou_type": "giou",
    "iou_weights": 2.0,
    "iou_cost": 1.0,
    "cls_weights": 2.0,
    "cls_cost": 1.0,
    "l1_weights": 5.0,
    "l1_cost": 1.0
}


def resize_boxes(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    """
    将boxes参数根据图像的缩放情况进行相应缩放

    Arguments:
        original_size: 图像缩放前的尺寸
        new_size: 图像缩放后的尺寸
    """
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratios_height, ratios_width = ratios
    # Removes a tensor dimension, boxes [minibatch, 4]
    # Returns a tuple of all slices along a given dimension, already without it.
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


class SparseRCNN(nn.Module):
    def __init__(self,
                 backbone,
                 num_cls,
                 min_size = 800,
                 max_size = 1333,
                 image_mean = [0.485, 0.456, 0.406],
                 image_std = [0.229, 0.224, 0.225],
                 num_proposals = 100,
                 in_channel = 256):
        super(SparseRCNN, self).__init__()
        # 对数据进行标准化，缩放，打包成batch等处理部分
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        self.backbones = backbone
        self.num_cls = num_cls
        self.num_proposals = num_proposals
        self.init_proposal_features = nn.Embedding(num_proposals, in_channel)
        self.init_proposal_boxes = nn.Embedding(num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)

        ## 注意这里dynamichead里面也有numcls参数
        self.head = DynamicHead(num_cls=num_cls,inplace=True)
        self.shape_weights = None
        self.loss = SparseRCNNLoss()

    def forward(self, images, targets=None):

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2  # 防止输入的是个一维向量
            original_image_sizes.append((val[0], val[1]))
        
        # resize图片与标记,并返回为tensor格式
        images,targets = self.transform(images,targets)
        features = self.backbones(images.tensors)
        p2, p3, p4, p5, _ = [f for f in features.values()]
        h, w = images.tensors.shape[2:]
        shapes = [(h, w)] * images.tensors.size(0)
        if self.shape_weights is None:
            self.shape_weights = torch.tensor([w, h, w, h], device=images.tensors.device)[None, :]

        init_boxes = self.init_proposal_boxes.weight * self.shape_weights
        init_boxes_x1y1 = init_boxes[:, :2] - init_boxes[:, 2:] / 2.0
        init_boxes_x2y2 = init_boxes_x1y1 + init_boxes[:, 2:]
        init_boxes_xyxy = torch.cat([init_boxes_x1y1, init_boxes_x2y2], dim=-1)
        cls_predicts, box_predicts = self.head({"p2": p2, "p3": p3, "p4": p4, "p5": p5},
                                               init_boxes_xyxy,
                                               self.init_proposal_features.weight,
                                               shapes)
        ret = dict()
        if self.training:
            assert targets is not None
            cls_losses, iou_losses, l1_losses, pos_num = self.loss(cls_predicts, box_predicts, targets, shapes[0])
            ret['cls_loss'] = cls_losses
            ret['iou_loss'] = iou_losses
            ret['l1_loss'] = l1_losses
            ret['match_num'] = pos_num
            return ret
        else:
            cls_predict = cls_predicts[-1]
            box_predict = box_predicts[-1]
            predicts = self.post_process(cls_predict, box_predict, shapes)
            ret['boxes'] = resize_boxes(predicts[0][:,0:4], shapes[0], original_image_sizes[0])  # 将bboxes缩放回原图像尺度上
            ret['scores'] = predicts[0][:,4]
            ret['labels'] = predicts[0][:,5].int()
            return [ret]

    def post_process(self, cls_predict, box_predict, shapes):
        assert len(cls_predict) == len(box_predict)
        scores = cls_predict.sigmoid()
        result = list()
        labels = torch.arange(self.num_cls, device=cls_predict.device). \
            unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)
        for score, box, shape in zip(scores, box_predict, shapes):
            scores_per_image, topk_indices = score.flatten(0, 1).topk(self.num_proposals, sorted=False)
            labels_per_image = labels[topk_indices]
            box = box.view(-1, 1, 4).repeat(1, self.num_cls, 1).view(-1, 4)
            x1y1x2y2 = box[topk_indices]
            x1y1x2y2[:, [0, 2]] = x1y1x2y2[:, [0, 2]].clamp(min=0, max=shape[1])
            x1y1x2y2[:, [1, 3]] = x1y1x2y2[:, [1, 3]].clamp(min=0, max=shape[0])
            result.append(torch.cat([x1y1x2y2, scores_per_image[:, None], labels_per_image[:, None]], dim=-1))
        return result


def roi_test():
    m = MultiScaleRoIAlign(['feat1', 'feat3'], 7, 2)
    i = dict()
    i['feat1'] = torch.rand(2, 5, 64, 64)
    i['feat2'] = torch.rand(2, 5, 32, 32)
    i['feat3'] = torch.rand(2, 5, 16, 16)
    boxes1 = torch.rand(6, 4) * 256
    boxes1[..., 2:] += boxes1[..., :2]
    boxes2 = boxes1[:2, ...]
    input_boxes = [boxes1, boxes2]
    image_sizes = [(512, 512), (512, 512)]
    output = m(i, input_boxes, image_sizes)
    print(output.shape)


if __name__ == '__main__':
    input_x = torch.rand((4, 3, 640, 640))
    net = SparseRCNN()
    net.eval()
    out = net(input_x)
    import ipdb;ipdb.set_trace()
