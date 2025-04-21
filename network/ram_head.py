#!/usr/bin/env python
# coding=utf-8
'''
ROI注意力和全局注意力模块,
添加到ROIpooling后,TwoMLPHead之前,
放入ROIHead中
'''

import torch
import torch.nn as nn


class RAM_Head(nn.Module):
    def __init__(self, feat_channel, hidden_channel, globalatt = False):
        super().__init__()
        self.hidden_channel = hidden_channel
        self.globalatt = globalatt

        self.conv_q = nn.Conv2d(feat_channel,hidden_channel,1)
        self.conv_k = nn.Conv2d(feat_channel,hidden_channel,1)
        self.conv_v = nn.Conv2d(feat_channel,hidden_channel,1)
        self.conv_re = nn.Conv2d(hidden_channel,feat_channel,1)
        self.down = nn.MaxPool2d(kernel_size=2)

        if self.globalatt:
            self.convq_g = nn.Conv2d(feat_channel,hidden_channel,1)
            self.convk_g = nn.Conv2d(feat_channel,hidden_channel,1)
            self.convv_g = nn.Conv2d(feat_channel,hidden_channel,1)
            self.convre_g = nn.Conv2d(hidden_channel,feat_channel,1)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def global_attention(self,feature,rois):
        BS,C,H,W = feature.shape
        BS_roinum,_,h,w = rois.shape
        roinum = BS_roinum//BS
        _H = H//2
        _W = W//2

        q = self.convq_g(rois)
        q = q.permute(0,2,3,1)
        q = q.reshape(BS, roinum, h, w, self.hidden_channel)
        q = q.reshape(BS, roinum*h*w, self.hidden_channel)

        ds = self.down(feature)
        k = self.convk_g(ds)
        k = k.reshape(BS, self.hidden_channel, _H*_W)

        v = self.convv_g(ds)
        v = v.permute(0,2,3,1)
        v = v.reshape(BS, _H*_W, self.hidden_channel)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        score = torch.bmm(q,k)
        score = torch.softmax(score, dim=2)

        y = torch.bmm(score,v)
        y = y.reshape(BS, roinum, h, w, self.hidden_channel)
        y = y.reshape(BS*roinum, h, w, self.hidden_channel)
        y=  y.permute(0,3,1,2)
        y = y.contiguous()

        y = self.convre_g(y)
        y = y.contiguous()

        out = rois + y
        return out


    def forward(self,rois,feature):
        BS_roinum,C,h,w = rois.shape
        BS = feature.shape[0]
        roinum = BS_roinum//BS
        _h = h//2
        _w = w//2

        q = self.conv_q(rois)
        q = q.permute(0,2,3,1)
        q = q.reshape(BS, roinum, h, w, self.hidden_channel)
        q = q.reshape(BS, roinum*h*w, self.hidden_channel)

        ds = self.down(rois)
        k = self.conv_k(ds)
        k = k.permute(0,2,3,1)
        k = k.reshape(BS, roinum, _h, _w, self.hidden_channel)
        k = k.reshape(BS, roinum*_h*_w, self.hidden_channel)
        # 此处是为了矩阵相乘的转置操作
        k = k.permute(0,2,1)

        v = self.conv_v(ds)
        v = v.permute(0,2,3,1)
        v = v.reshape(BS, roinum, _h, _w, self.hidden_channel)
        v = v.reshape(BS, roinum*_h*_w, self.hidden_channel)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        score = torch.bmm(q,k)
        score = torch.softmax(score, dim=2)

        y = torch.bmm(score,v)
        y = y.reshape(BS, roinum, h, w, self.hidden_channel)
        y = y.reshape(BS*roinum, h, w, self.hidden_channel)
        y = y.permute(0,3,1,2)
        y = y.contiguous()
        
        y = self.conv_re(y)
        y = y.contiguous()

        if self.globalatt:
            return self.global_attention(feature, rois + y)

        out = rois + y
        return out