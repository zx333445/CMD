#!/usr/bin/env python
# coding=utf-8
'''
ROI注意力模块,
添加到ROIpooling后,TwoMLPHead之前,
放入ROIHead中
'''

import torch
import torch.nn as nn


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
                nn.init.constant_(m.bias, 0)


    def forward(self,rois,feature):
        # 传入参数时维度为
        # rois tensor(B*numroi,C,h,w)  (1024,256,7,7)
        # feature tensor(B,C,H,W)  (2,256,200,200)
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

