#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEFocalLoss(nn.Module):
    '''二分类focalloss'''
    def __init__(self, alpha=0.25, gamma=2, from_logits=True, reduce=True):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.from_logits:
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduce=False
            )
        else:
            # 此处需要传入的预测需要为概率值,不是预测分数,即需要经过sigmoid映射
            bce_loss = F.binary_cross_entropy(inputs, targets,
                                              reduce=False)
        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduce:
            return torch.mean(f_loss)
        else:
            return f_loss


class CEFocalLoss(nn.Module):
    '''多分类focalloss'''
    def __init__(self, class_nums, alpha=[0.75, 0.25], gamma=2, size_avg=True):
        super(CEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_nums, 1)
        else:
            self.alpha = torch.as_tensor(alpha)

        self.gamma = gamma
        self.class_nums = class_nums
        self.size_avg = size_avg

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = torch.zeros(N, C, dtype=inputs.dtype,
                                 device=inputs.device)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids, 1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -alpha * (torch.pow((1-probs), self.gamma))*log_p

        if self.size_avg:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=0.75, gamma=2, use_alpha=True, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha)
            # self.alpha = torch.tensor(alpha)

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):

        prob = self.softmax(pred.view(-1,self.class_num))
        prob = prob.clamp(min=0.0001,max=1.0)

        target_ = torch.zeros(target.size(0),self.class_num,device=pred.device)
        # target_ = torch.zeros(target.size(0),self.class_num)
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.use_alpha:
            self.alpha.to(device=pred.device) # type: ignore
            batch_loss = - self.alpha.double() * torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double() # type: ignore
        else:
            batch_loss = - torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss