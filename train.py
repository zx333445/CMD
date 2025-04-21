#!/usr/bin/env python
# coding=utf-8
"""
train model
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import models

import argparse
import os
import datetime
import time

from network.backbone_utils import resnet_fpn_backbone, densenet_fpn_backbone, swin_fpn_backbone, swin_attfpn_backbone, convnext_fpn_backbone
from network import ssdresbackbone, SSD, FCOS, RetinaNet, FasterRCNN, CascadeRCNN, RRAMandGRAM
from netsparse import SparseRCNN
from netyolo import YOLOv3, YOLOv7
from netdetr import DETR
from netcmd import CascadeMiningDet
from tool import transforms as T
import _utils
from trainer import main_process
from datasets import CTCDataset


parser = argparse.ArgumentParser(description="CTC object detection")
#
parser.add_argument("--pretrained", help="whether use pretrained weight", type=bool, default=True)
parser.add_argument("--train_csv_path", type=str, default="./train.csv", help="train case path csv, default ./train.csv")
parser.add_argument("--val_csv_path", type=str, default="./val.csv", help="val case path csv, default ./val.csv")
parser.add_argument("--test_csv_path", type=str, default="./test.csv", help="test case path csv, default ./test.csv")
parser.add_argument("--logdir", help="tensorboard log dir", type=str, default="./logs")
parser.add_argument("--fold", help="train fold", type=int, default=1)
# train param
parser.add_argument("--train_batch_size", help="train batch size", type=int, default=2)
parser.add_argument("--val_batch_size", help="val batch size", type=int, default=1)
parser.add_argument("--test_batch_size", help="test batch size", type=int, default=1)
parser.add_argument("--num_workers", help="number of workers", type=int, default=16)
parser.add_argument("--num_epochs", help="number of Epoch", type=int, default=50)
parser.add_argument("--save_model_path", help="model saving dir", type=str, default="/home/stat-zx/CTCdet/results/densenet169.pth")
# optimizer
subparsers = parser.add_subparsers(help="optimizer type", dest="optimizer_type")
subparsers.required = True
# SGD
sgd_parser = subparsers.add_parser("SGD")
sgd_parser.add_argument("--sgd_lr", help="SGD learning rate", type=float, default=0.01)
sgd_parser.add_argument("--momentum", help="SGD momentum", type=float, default=0.9)
sgd_parser.add_argument("--weight_decay", help="SGD weight decay", type=float, default=5e-4)
# AdamW
adam_parser = subparsers.add_parser("AdamW")
adam_parser.add_argument("--adamW_lr", help="Adam learning rate", type=float, default=2e-5)
adam_parser.add_argument("--adamW_decay", help="Adam learning rate decay", type=float, default=1e-4)

args = parser.parse_args()


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(args)
    CLASSES = {"__background__", "CTC","CTC-like"}
    
    print("===============Loading data===============")
    data_transform = {
        "train": T.Compose([
            T.ImgAugTransform(),
            T.ToTensor()]),
        "val": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
    }

    dataset = CTCDataset(args.train_csv_path, transforms=data_transform["train"], train=True, stain=False)
    dataset_val = CTCDataset(args.val_csv_path, transforms=data_transform["val"], train=False, stain=False)
    dataset_test = CTCDataset(args.test_csv_path, transforms=data_transform["test"], train=False, stain=False)

    print("====Creating dataloader====")
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=_utils.collate_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=_utils.collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False,num_workers=args.num_workers, collate_fn=_utils.collate_fn)
    dataloaders = {"train": dataloader, "val": dataloader_val, "test": dataloader_test,}

    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    
    print("===============Loading model===============")
    # SSD
    # model = models.detection.ssd300_vgg16(weights_backbone = models.VGG16_Weights.DEFAULT, num_classes=len(CLASSES))

    # FCOS
    # backbone = swin_fpn_backbone(extra_type='last')
    # model = FCOS(backbone, num_classes=len(CLASSES))

    # Retinanet
    # backbone = swin_fpn_backbone(extra_type='last')
    # model = RetinaNet(backbone, num_classes=len(CLASSES))

    # Faster RCNN
    # backbone = swin_fpn_backbone(extra_type='maxpool')
    # model = FasterRCNN(backbone, num_classes=len(CLASSES))

    # Cascade RCNN
    # backbone = swin_fpn_backbone(extra_type='maxpool')
    # model = CascadeRCNN(backbone, num_classes=len(CLASSES))
    
    # Sparse RCNN
    # backbone = swin_fpn_backbone(extra_type='maxpool')
    # model = SparseRCNN(backbone=backbone,num_cls=len(CLASSES))

    # YOLOv3
    # model = YOLOv3(num_classes=len(CLASSES))

    # YOLOv7
    # model = YOLOv7(num_classes=len(CLASSES))

    # DETR
    # model = DETR(num_classes=len(CLASSES))

    # AttFPN
    # backbone = swin_attfpn_backbone(extra_type='maxpool')
    # model = FasterRCNN(backbone, num_classes=len(CLASSES))

    # RRAM & GRAM
    # backbone = swin_fpn_backbone(extra_type='maxpool')
    # model = RRAMandGRAM(backbone, num_classes=len(CLASSES))

    # CascadeMiningDet
    backbone = swin_fpn_backbone(extra_type='maxpool')
    model = CascadeMiningDet(backbone, num_classes=len(CLASSES))
    model.to(device)
    print(model)
    print("===============Setting optimizer===============")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(params, lr=args.sgd_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

    elif args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=args.adamW_lr, weight_decay=args.adamW_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

    print("===============Start training===============")
    start_time = time.time()
    main_process(model=model, optimizer=optimizer, lr_sche=lr_scheduler,
                 dataloaders=dataloaders, num_epochs=args.num_epochs, use_tensorboard=True,
                 device=device, save_model_path=args.save_model_path,
                 fold=args.fold, writer=writer)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main(args)