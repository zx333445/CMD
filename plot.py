#!/usr/bin/env python
# coding=utf-8

import torch
import os
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from network.backbone_utils import resnet_fpn_backbone, densenet_fpn_backbone, swin_fpn_backbone, convnext_fpn_backbone
from network import CascadeMiningDet,FasterRCNN
from torchvision import transforms


def creat_model():

    # get devices
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    CLASSES = {"__background__", "CTC", "CTC样"}
    # backbone = densenet_fpn_backbone('densenet169', True)
    backbone = swin_fpn_backbone()
    # backbone = convnext_fpn_backbone()
    # model = FasterRCNN(backbone,num_classes=len(CLASSES))
    model = CascadeMiningDet(backbone, num_classes=len(CLASSES))
    model.to(device)

    # pth_path = '/home/stat-zx/CTC_cascade/results/models/cmd_swinb.pth'
    pth_path = '/home/stat-zx/CTC_cascade/results/models/cas_swinb.pth'
    statdic = torch.load(pth_path)
    model.load_state_dict(statdic)
    model.eval()

    return model


def iou(box1, box2):
    """
    计算两个定位框之间的IoU(交并比)
    box1: 第一个定位框，格式为 [x1, y1, x2, y2]
    box2: 第二个定位框，格式为 [x1, y1, x2, y2]
    """
    # 计算交集的坐标范围
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集的面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算并集的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    # 计算IoU
    iou = intersection / union

    return iou


def filter_low_prob_boxes(boxes, scores, iou_threshold):
    """
    计算所有定位框两两之间的IoU,并去掉大于iou阈值的两个框中预测概率较小的预测框(即去掉同一目标的重叠但不同类框)
    boxes: 所有定位框，格式为 [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    scores: 所有预测框的预测概率,格式为 [score1, score2, ...]
    iou_threshold: IoU(交并比)阈值,大于该阈值的框将被认为是重叠的
    返回的是需要排除的定位框索引
    """
    num_boxes = len(boxes)
    high_iou_boxes = []

    for i in range(num_boxes):
        for j in range(num_boxes):
            if i != j and iou(boxes[i], boxes[j]) > iou_threshold:
                if scores[i] > scores[j]:
                    high_iou_boxes.append(j)
                else:
                    high_iou_boxes.append(i)

    return list(set(high_iou_boxes))


def predict(model,image_path,save_path):

    # load image
    original_img = Image.open(image_path)
    coimg = original_img.copy()
    
    # from pil image to tensor, do not normalize image
    transform=transforms.Compose([transforms.ToTensor()])
    img = transform(coimg)
    
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0) 
    outputs = model(img.to('cuda:1'))
    
    if len(outputs[-1]["boxes"]) == 0:
        print('没有检测到CTC/CTC样细胞')
    else:
        new_output_index = torch.where(outputs[-1]["scores"] > 0.6)
        new_boxes = outputs[-1]["boxes"][new_output_index]
        new_scores = outputs[-1]["scores"][new_output_index]
        new_labels = outputs[-1]["labels"][new_output_index]

        coords = [] 
        for i in range(len(new_boxes)):
            new_box = new_boxes[i].tolist()
            coords.append([new_box[0], new_box[1],
                            new_box[2], new_box[3]])
        
        if len(coords) == 0:
            print('没有检测到CTC/CTC样细胞')
        else:
            coords_score = new_scores.tolist()
            coords_labels = new_labels.tolist()

            # exclude_index = filter_low_prob_boxes(coords,coords_score,0.6)
            # for id in exclude_index:
            #     del coords[id]
            #     del coords_score[id]
            #     del coords_labels[id]

            
            draw = ImageDraw.Draw(original_img)
            
            # 根据图片大小调整绘制定位框的线条宽度和书写概率文字的大小
            tl = round(0.002*(original_img.size[0]+original_img.size[1]) + 1)
            font = ImageFont.truetype(font="/usr/share/fonts/dejavu/DejaVuSans.ttf",size=5*tl)
            for box,score,label in zip(coords,coords_score,coords_labels):
                if label == 1:
                    draw.rectangle(box, outline=(255,0,0), width=tl)
                    draw.text((box[0] + tl,box[1] + tl), f'{score:.2f}',(255,0,0),font)
                else:
                    # draw.rectangle(box, outline=(0,0,255), width=tl)
                    draw.ellipse(box,outline=(255,0,0),width=tl)
                    draw.text((box[0] + tl,box[1] + tl), f'{score:.2f}',(255,0,0),font)

        # 保存预测的图片结果
        original_img.save(save_path)


def draw_gt(csv_root,save_path):
    '''绘制出真实标注的gt框,用于对比预测框准确度'''
    data = pd.read_csv(csv_root)
    img_list = list(data.image_path)
    gt_list = list(data.annotation)
    for path,anno in tqdm(zip(img_list,gt_list)):
        if isinstance(anno,str):
            img = Image.open(path)
            draw = ImageDraw.Draw(img)
            object_list = anno.split(';')
            for object in object_list:
                fields = object.split(' ')
                label = fields[0]
                xmin = float(fields[1])
                xmax = float(fields[3])
                ymin = float(fields[2])
                ymax = float(fields[4])
                box = [xmin,ymin,xmax,ymax]

                tl = round(0.002*(img.size[0]+img.size[1]) + 1)
                if label == '1':
                    draw.rectangle(box, outline=(255,0,0), width=tl)
                else:
                    # draw.rectangle(box, outline=(0,0,255), width=tl)
                    draw.ellipse(box,outline=(255,0,0),width=tl)

            img.save(os.path.join(save_path,path.split('/')[-1]))




if __name__=="__main__":

    model = creat_model()

    path_list = list(pd.read_csv('/home/stat-zx/CTC_data/test_pos.csv')['image_path'])

    for path in tqdm(path_list):
        predict(model,image_path=path, save_path=path.replace('CTC_data/JPEGImages','caspred'))


    # draw_gt('/home/stat-zx/CTC_data/test_pos.csv','/home/stat-zx/testgt')