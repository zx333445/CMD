#!/usr/bin/env python
# coding=utf-8
"""
train model
"""
import torch
from torch.utils.tensorboard import SummaryWriter
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import argparse
import sys
import os
import datetime
import time

from network.backbone_utils import resnet_fpn_backbone, densenet_fpn_backbone, swin_fpn_backbone, convnext_fpn_backbone
from network import CascadeMiningDet,FasterRCNN,FastRCNNPredictor
from network.model_without_fpn import create_dense_model
from tool import transforms as T
import _utils
from dataset import CTCDataset
import trainer



def parse_args(args):
    parser = argparse.ArgumentParser(description="CTC object detection")
    subparsers = parser.add_subparsers(
        help="optimizer type",
        dest="optimizer_type"
    )
    subparsers.required = True

    parser.add_argument("--model_name",
                        help="backbone",
                        type=str,
                        default="densenet169")
    parser.add_argument("--pretrained",
                        help="whether use pretrained weight",
                        type=bool,
                        default=True)
    parser.add_argument("--device",
                        help="cuda or cpu",
                        type=str,
                        default="cuda:0")
    parser.add_argument("--seed",
                        help="seed",
                        type=int,
                        default=32)
    parser.add_argument("--root",
                        help="image root dir",
                        type=str,
                        default="/home/stat-zx/CTC_data")
    parser.add_argument("--train_batch_size",
                        help="train batch size",
                        type=int,
                        default=2)
    parser.add_argument("--val_batch_size",
                        help="val batch_size",
                        type=int,
                        default=1)
    parser.add_argument("--test_batch_size",
                        help="test batch_size",
                        type=int,
                        default=1)
    parser.add_argument("--num_workers",
                        help="number of workers",
                        type=int,
                        default=12)
    parser.add_argument("--log_dir",
                        help="tensorboard log dir",
                        type=str,
                        default="./logs")
    sgd_parser = subparsers.add_parser("SGD")
    sgd_parser.add_argument("--sgd_lr",
                            help="SGD learning rate",
                            type=float,
                            default=0.01)
    sgd_parser.add_argument("--momentum",
                            help="SGD momentum",
                            type=float,
                            default=0.9)
    sgd_parser.add_argument("--weight_decay",
                            help="SGD weight decay",
                            type=float,
                            default=5e-4)
    adam_parser = subparsers.add_parser("Adam")
    adam_parser.add_argument("--adam_lr",
                             help="Adam learning rate",
                             type=float,
                             default=0.01)
    parser.add_argument("--step_size",
                        help="StepLR",
                        type=int,
                        default=8)
    parser.add_argument("--gamma",
                        help="StepLR gamma",
                        type=float,
                        default=0.1)
    parser.add_argument("--num_epochs",
                        help="number of Epoch",
                        type=int,
                        default=50)
    parser.add_argument("--save_model_path",
                       help="model saving dir",
                       type=str,
                       default="/home/stat-zx/CTC_myfastrcnn/results/models/densenet169.pth")
    parser.add_argument("--record_iter",
                       help="record step",
                       type=int,
                       default=10)
    parser.add_argument("--voc_results_dir",
                        help="pred boxes dir",
                        type=str,
                        default="/home/stat-zx/detection_results/")
    parser.add_argument("--pretrained_resnet50_coco",
                        help="whether use resnet50 coco pretrained weight",
                        type=bool,
                        default=False)
    parser.add_argument("--fpn",
                        help="whether use fpn",
                        type=bool,
                        default=True)
    parser.add_argument("--ReduceLROnPlateau",
                        help="lr decay method",
                        type=bool,
                        default=False)
    parser.add_argument("--Cosine",
                        help="lr decay method",
                        type=bool,
                        default=True)
    
    return parser.parse_args(args)



def main(args=None):
    model_urls = {
        "fasterrcnn_resnet50_fpn_coco":
        "http://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    }
    
    # 从终端传入参数进入parse_args进行解析
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    print(args)
    

    CLASSES = {"__background__", "CTC","CTC样"}
    # CLASSES = {"__background__", "CTC"}

    # if args.pretrained_resnet50_coco:
    #     state_dict = load_state_dict_from_url(
    #         model_urls["fasterrcnn_resnet50_fpn_coco"],
    #         progress=True
    #     )
    #     new_dict = {}
    #     for k,v in state_dict.items():
    #         if 'backbone' in k:
    #             new_dict[k.replace('backbone.','')] = v

    #     backbone = resnet_fpn_backbone("resnet50", False)
    #     backbone.load_state_dict(new_dict)
    #     model = CascadeMiningDet(backbone, num_classes=len(CLASSES))

        
    if args.fpn:
        # backbone = resnet_fpn_backbone(args.model_name, args.pretrained)
        # backbone = densenet_fpn_backbone(args.model_name, args.pretrained)
        backbone = swin_fpn_backbone()
        # backbone = convnext_fpn_backbone()
        model = CascadeMiningDet(backbone, num_classes=len(CLASSES))
    else:
        model = create_dense_model(num_classes=len(CLASSES),backbone_name=args.model_name,pretrained=args.pretrained)
    
    print(model)


    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 实例化dataset载入数据
    # 注意normalize在训练集使用的话,则验证集与测试集也需要使用
    # 注意在网络结构中的Generalizedtransform中包含了对数据的标准化,此处不应该包含normalize ?   
    print("===============Loading data===============")
    data_transform = {
        "train": T.Compose([T.ImgAugTransform(),
                            T.ToTensor()]),
        "val": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
    }

    dataset = CTCDataset(
        args.root, 
        transforms=data_transform["train"],
        train=True, 
        csv_name="train_pos.csv")

    dataset_val = CTCDataset(
        args.root, 
        transforms=data_transform["val"],
        train=False,
        csv_name="val_pos.csv")

    dataset_test = CTCDataset(
        args.root, 
        transforms=data_transform["test"], 
        train=False,
        csv_name="test_pos.csv")

    
    # datalodaer包装dataset,注意collate_fn函数为自己指定,返回image与target的列表
    print("====Creating dataloader====")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_utils.collate_fn
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_utils.collate_fn
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_utils.collate_fn
    )
    dataloaders = {
        "train": dataloader,
        "val": dataloader_val,
        "test": dataloader_test,
    }
    # 创建文件夹为tensorboard记录地址
    logdir = args.log_dir
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    
    print("===============Loading model===============")
    model.to(device)

    # 打印出模型参数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(params, lr=args.sgd_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)


        if args.ReduceLROnPlateau:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.1, patience=10,
                verbose=False, threshold=0.0001, threshold_mode="rel",
                cooldown=0, min_lr=0, eps=1e-8
            )
        elif args.Cosine:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, args.num_epochs
            )
        else:
            # lr_scheduler = torch.optim.lr_scheduler.StepLR(
            #     optimizer, step_size=args.step_size, gamma=args.gamma
            # )
            # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            #     optimizer, milestones=[8, 24], gamma=args.gamma
            # )
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[25, 45], gamma=args.gamma
            )
            # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            #     optimizer, milestones=[50, 60], gamma=args.gamma
            # )

    elif args.optimizer_type == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.adam_lr)
        lr_scheduler = None

    print("===============Start training===============")
    start_time = time.time()
    trainer.train_process(model=model, optimizer=optimizer,
                         lr_sche=lr_scheduler,
                         dataloaders=dataloaders,
                         num_epochs=args.num_epochs,
                         use_tensorboard=True,
                         device=device,
                         save_model_path=args.save_model_path,
                         record_iter=args.record_iter,
                         writer=writer,
                         ReduceLROnPlateau=args.ReduceLROnPlateau)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()