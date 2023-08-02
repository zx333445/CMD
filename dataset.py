#!/usr/bin/env python
# coding=utf-8
import os
import torch

from PIL import Image
import pandas as pd
from torch.utils.data import Dataset



class CTCDataset(Dataset):
    ''''''
    def __init__(self, ctc_root, transforms, train = True, csv_name = "train.csv"):
        super().__init__()
        self.root = ctc_root
        annotation = pd.read_csv(os.path.join(self.root,csv_name))

        self.image_list = list(annotation['image_path'])

        for path in self.image_list:
            assert os.path.exists(path),  "not found {} file.".format(path)

        self.annotations = list(annotation['annotation'])
        # 传入预先定义好的transforms
        # 此处传入的transforms包括将图片转为tensor(同时归一化)与水平翻转等的组合
        # 而图像大小改变与打包成batch的操作在generalizetransform中实现
        self.transforms = transforms
        # 用于判断是否传入target
        self.train = train


    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, idx):
        
        img_path = self.image_list[idx]
        image = Image.open(img_path).convert('RGB')

        annotation = self.annotations[idx]
        
        if type(annotation) != str:
            annotation = str(annotation)

        # 不是训练模式时transform不对target进行,使用None/空{}即可
        if self.train:        
            boxes = []
            labels = []           
            annotation_list = annotation.split(";")       
            for anno in annotation_list:
                label = int(anno[0])
                x = []
                y = []
                anno = anno[2:]  # one box coord str
                anno = anno.split(" ")
                for i in range(len(anno)):
                    if i % 2 == 0:
                        x.append(float(anno[i]))
                    else:
                        y.append(float(anno[i]))

                xmin = min(x)
                xmax = max(x)
                ymin = min(y)
                ymax = max(y)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
            
            # convert anything to torch tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
        else:
            target = {}

        # 对target不进行totensor操作(已经是tensor),但翻转操作需要对bbox进行
        image, target = self.transforms(image, target)

        return image, target

