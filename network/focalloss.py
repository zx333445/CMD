#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEFocalLoss(nn.Module):
    ''''''
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
            bce_loss = F.binary_cross_entropy(inputs, targets,
                                              reduce=False)
        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduce:
            return torch.mean(f_loss)
        else:
            return f_loss


class CEFocalLoss(nn.Module):
    ''''''
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
            self.alpha.to(device=pred.device)
            batch_loss = - self.alpha.double() * torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss
