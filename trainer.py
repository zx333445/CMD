import math
import sys
import torch
import pandas as pd

import sys
sys.path.append(".")
sys.path.append("..")
import tool.utils as utils
from tool.voc_eval_new import custom_voc_eval
# 使用tensorboard可视化损失
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import copy


def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False


def train_process(model, optimizer, lr_sche,
                  dataloaders, num_epochs,
                  use_tensorboard,
                  device,
                  # model save params
                  save_model_path,
                  record_iter,
                  # tensorboard
                  writer=None,
                  ReduceLROnPlateau=False):
    savefig_flag = True
    model.train()
    model.apply(freeze_bn)
    
    running_loss = 0.0
    running_loss_classifier = 0.0
    running_loss_box_reg = 0.0
    running_loss_objectness = 0.0
    running_loss_rpn_box_reg = 0.0
    best_score = 0.0
    best_stat_dict = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        lr_scheduler = None
        print("====Epoch {0}====".format(epoch))
        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(dataloaders['train']) - 1)
            lr_scheduler = utils.warmup_lr_scheduler(optimizer,
                                                    warmup_iters,
                                                    warmup_factor)
        for i, (images, targets) in enumerate(tqdm(dataloaders['train']), 0):

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()}
                      for t in targets]
            optimizer.zero_grad()
            # 得到损失值字典
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            losses_total = losses.item()
            # roi分类损失
            loss_classifier = loss_dict['loss_classifier'].item()
            # roi回归损失
            loss_box_reg = loss_dict['loss_box_reg'].item()
            # rpn分类损失
            loss_objectness = loss_dict['loss_objectness'].item()
            # rpn回归损失
            loss_rpn_box_reg = loss_dict['loss_rpn_box_reg'].item()
            # 学习率
            lr = optimizer.param_groups[0]['lr']
            # lr_small = optimizer.param_groups[0]["lr"]
            # lr_large = optimizer.param_groups[1]["lr"]
            running_loss += losses_total
            running_loss_classifier += loss_classifier
            running_loss_box_reg += loss_box_reg
            running_loss_objectness += loss_objectness
            running_loss_rpn_box_reg += loss_rpn_box_reg

            if (i+1) % record_iter == 0:
                print('''Epoch{0} loss:{1:.4f}
                         loss_classifier:{2:.4f} loss_box_reg:{3:.4f}
                         loss_objectness:{4:.4f} loss_rpn_box_reg:{5:.4f}\n'''.format(
                          epoch,
                          losses_total, loss_classifier,
                          loss_box_reg, loss_objectness,
                          loss_rpn_box_reg
                      ))
                if use_tensorboard:
                    # 写入tensorboard
                    writer.add_scalar("Total loss",
                                     running_loss / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("RoI classification loss",
                                     running_loss_classifier / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("RoI reg loss",
                                     running_loss_box_reg / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("RPN classification loss",
                                     running_loss_objectness / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("RPN reg loss",
                                     running_loss_rpn_box_reg / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("lr",
                                     lr,
                                     epoch * len(dataloaders['train']) + i)
                    running_loss = 0.0
                    running_loss_classifier = 0.0
                    running_loss_box_reg = 0.0
                    running_loss_objectness = 0.0
                    running_loss_rpn_box_reg = 0.0


        valap_dict,val_mAP = custom_voc_evaluate(
            model, dataloaders["val"], device=device,
            gt_csv_path="/home/stat-zx/CTC_data/val_pos.csv",
            cls_csv_path="/home/stat-zx/CTC_cascade/results/val_cls.csv",
            loc_csv_path="/home/stat-zx/CTC_cascade/results/val_loc.csv"
        )

        # 打印该epoch后验证集的各类别AP值
        print(f"Epoch: {epoch}, | val CTC AP :{valap_dict['1']:.4f}")
        print(f"Epoch: {epoch}, | val CTC样 AP :{valap_dict['2']:.4f}")
        print(f"Epoch: {epoch}, | val mAP: {val_mAP:.4f}")
        
        if not ReduceLROnPlateau:
            lr_sche.step()
        else:
            lr_sche.step(val_mAP)
        
        if val_mAP > best_score:
            best_score = val_mAP
            best_stat_dict = copy.deepcopy(model.state_dict())
            savefig_flag = True
        else:
            savefig_flag = False
        
        if use_tensorboard:
            # writer.add_figure(
            #     "Validation PR-curve",
            #     val_fig,
            #     global_step=epoch
            # )
            writer.add_scalar(
                'Validation mAP',
                val_mAP,
                global_step=epoch
            )
            # writer.add_scalar(
            #     "Validation acc",
            #     acc,
            #     global_step=epoch
            # )
            # writer.add_scalar(
            #     "Validation auc",
            #     roc_auc,
            #     global_step=epoch
            # )
        
        # 因为evaluate函数中将模型转为eval模式,此处需要再次转为train进行下一次epoch
        model.train()
        model.apply(freeze_bn)


    print("===============训练完成===============")
    print(f"Best Valid mAP: {best_score:.4f}")
    torch.save(best_stat_dict, save_model_path)

    print("===============开始测试===============")
    model.load_state_dict(best_stat_dict)

    testap_dict,test_mAP = custom_voc_evaluate(
        model, dataloaders["test"], device=device,
            gt_csv_path="/home/stat-zx/CTC_data/test_pos.csv",
            cls_csv_path="/home/stat-zx/CTC_cascade/results/test_cls.csv",
            loc_csv_path="/home/stat-zx/CTC_cascade/results/test_loc.csv"
    )


    print(f"Test CTC AP :{testap_dict['1']:.4f}")
    print(f"Test CTC样 AP :{testap_dict['2']:.4f}")
    print(f"Test mAP: {test_mAP:.4f}")
    if use_tensorboard:
        writer.close()


@torch.no_grad()
def custom_voc_evaluate(model, data_loader, device,
                        gt_csv_path,
                        cls_csv_path,
                        loc_csv_path,
                        savefig_flag=False):
    '''计算voc指标,不使用coco'''                    
    cpu_device = torch.device('cpu')
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test'
    
    # perd存储图片的中CTC预测框的最高置信度score
    # locs存储各个预测框的预测标签,置信度与坐标信息
    preds = [] 
    locs = []
    for image, _ in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()}
                  for t in outputs]
        # 没有预测框时,即没有可检测目标,pred与locs都存储为空
        if len(outputs[-1]["boxes"]) == 0:
            # if no pred boxes, means that the image is negative
            preds.append(0)
            locs.append("")

        else:
            # preds.append(torch.max(outputs[-1]["scores"]).tolist())
            # we keep those pred boxes whose score is more than 0.6
            new_output_index = torch.where(outputs[-1]["scores"] > 0.6)
            new_boxes = outputs[-1]["boxes"][new_output_index]
            new_scores = outputs[-1]["scores"][new_output_index]
            new_labels = outputs[-1]["labels"][new_output_index]
            
            # label为1即预测为CTC细胞,label为2则为CTC样细胞
            # CTC_index = torch.where(new_labels == 1)
            CTC_index = torch.where(new_labels != 3)
            # 注意此处判断时,torch.where返回嵌套元组没有符合条件时长度也为1,
            # 需要取出代表index的对象index[0]
            if len(new_boxes) != 0 and len(CTC_index[0]) != 0:
                # 只储存一张图中CTC预测框置信度最高的score                
                preds.append(torch.max(new_scores[CTC_index]).tolist())
            else:
                preds.append(0)
            
            # used to save pred coords x1 y1 x2 y2
            # used to save pred box scores
            # 存储该张图片所有预测框的坐标与标签与置信度,最后写入line中,按图片存储为loc信息
            coords = [] 
            for i in range(len(new_boxes)):
                new_box = new_boxes[i].tolist()
                coords.append([new_box[0], new_box[1],
                               new_box[2], new_box[3]])
            coords_score = new_scores.tolist()
            coords_labels = new_labels.tolist()
            line = ""
            for i in range(len(new_boxes)):
                if i == len(new_boxes) - 1:
                    line += str(coords_labels[i]) + ' ' + str(coords_score[i]) + ' ' +  \
                            str(coords[i][0]) + ' ' + str(coords[i][1]) + ' ' + \
                            str(coords[i][2]) + ' ' + str(coords[i][3])
                else:
                    line += str(coords_labels[i]) + ' ' + str(coords_score[i]) + ' ' + \
                            str(coords[i][0]) + ' ' + str(coords[i][1]) + ' ' + \
                            str(coords[i][2]) + ' ' + str(coords[i][3]) + ';'

            locs.append(line)
    print("====write cls pred results to csv====")
    cls_res = pd.DataFrame(
        {"image_path": data_loader.dataset.image_list,
         "prediction": preds}
    )
    cls_res.to_csv(cls_csv_path, columns=["image_path", "prediction"],
                   sep=',', index=None)
    
    print("====write loc pred results to csv====") 
    loc_res = pd.DataFrame(
        {"image_path": data_loader.dataset.image_list,
         "prediction": locs}
    )
    loc_res.to_csv(loc_csv_path, columns=["image_path", "prediction"],
                   sep=',', index=None)

    AP_dict, mAP = custom_voc_eval(gt_csv_path, loc_csv_path)
    return AP_dict,mAP



