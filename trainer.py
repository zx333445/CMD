import math
import sys
import os
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc

import sys
sys.path.append(".")
sys.path.append("..")
import tool.utils as utils
from tool.voc_eval import write_custom_voc_results_file, do_python_eval
from tool.voc_eval_new import custom_voc_eval
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import copy


def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False


def train_one_epoch(epoch, model, loader, optimizer, lr_scheduler, device, writer=None):
    model.train()
    model.apply(freeze_bn)
    train_loss = 0.0

    for i, (images, targets) in enumerate(tqdm(loader), 0):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item() # type: ignore
        train_loss += loss_value
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        if (i+1) % 100 == 0:
            print(f'batch{i+1} loss:{loss_value:.4f}')
        losses.backward() # type: ignore
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print(f'epoch{epoch} loss:{train_loss/len(loader):.4f}')
    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)  


def validate(epoch, model, loader, device, save_model_path, gt_csv):                   
    model.eval()
    locs = []
    for i, (image, _) in enumerate(tqdm(loader)):
        image = list(img.to(device) for img in image)
        with torch.no_grad():
            outputs = model(image)
        outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

        if len(outputs[-1]["boxes"]) == 0:
            # if no pred boxes, means that the image is negative
            locs.append("")
        else:
            # we keep those pred boxes whose score is more than 0.5
            new_output_index = torch.where(outputs[-1]["scores"] > 0.5)
            new_boxes = outputs[-1]["boxes"][new_output_index]
            new_scores = outputs[-1]["scores"][new_output_index]
            new_labels = outputs[-1]["labels"][new_output_index]
            # used to save pred coords x1 y1 x2 y2
            # used to save pred box scores
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
    loc_res = pd.DataFrame({"image_path": loader.dataset.image_list,
                            "prediction": locs})
    valpath = os.path.join(os.path.dirname(save_model_path),f'{loader.dataset.image_list[0].split("/")[3]}_val.csv')
    loc_res.to_csv(valpath, index=False)
    valap_dict,val_mAP,valmf1 = custom_voc_eval(gt_csv=gt_csv, pred_csv=valpath, label_list=['1','2']) 
    print(f"Epoch: {epoch}, | val CTC AP :{valap_dict['1']:.4f}")
    print(f"Epoch: {epoch}, | val CTC-like AP: {valap_dict['2']:.4f}")
    print(f"Epoch: {epoch}, | val mAP: {val_mAP:.4f}")    

    return val_mAP


def summary(model, loader, device, save_model_path, gt_csv):
    model.eval()
    locs = []
    for i, (image, _) in enumerate(tqdm(loader)):
        image = list(img.to(device) for img in image)
        with torch.no_grad():
            outputs = model(image)
        outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

        if len(outputs[-1]["boxes"]) == 0:
            # if no pred boxes, means that the image is negative
            locs.append("")
        else:
            # we keep those pred boxes whose score is more than 0.5
            new_output_index = torch.where(outputs[-1]["scores"] > 0.5)
            new_boxes = outputs[-1]["boxes"][new_output_index]
            new_scores = outputs[-1]["scores"][new_output_index]
            new_labels = outputs[-1]["labels"][new_output_index]
            # used to save pred coords x1 y1 x2 y2
            # used to save pred box scores
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
    loc_res = pd.DataFrame({"image_path": loader.dataset.image_list,
                            "prediction": locs})
    testpath = os.path.join(os.path.dirname(save_model_path),f'{loader.dataset.image_list[0].split("/")[3]}_test.csv')
    loc_res.to_csv(testpath, index=False)
    testap_dict,test_mAP,testmf1 = custom_voc_eval(gt_csv=gt_csv, pred_csv=testpath, label_list=['1','2']) 
    print(f"Test CTC AP :{testap_dict['1']:.4f}")
    print(f"Test CTC-like AP: {testap_dict['2']:.4f}")
    print(f"Test mAP: {test_mAP:.4f}")
    print(f"Test F1-score: {testmf1:.4f}")

    return test_mAP


def main_process(model, optimizer, lr_sche,
                  dataloaders, num_epochs,
                  use_tensorboard,
                  device,
                  save_model_path,
                  fold,
                  writer=None):

    best_score = 0.0

    for epoch in range(num_epochs):
        train_one_epoch(epoch, model, dataloaders['train'], optimizer, lr_sche, device, writer)
        val_mAP = validate(epoch, model, dataloaders['val'], device, save_model_path, gt_csv=f'/home/stat-zx/CTCdet/csvfiles/fold{fold}/val.csv')
        
        if val_mAP > best_score:
            best_score = val_mAP
            best_epoch = epoch
            best_stat_dict = copy.deepcopy(model.state_dict())
        
        if use_tensorboard:
            writer.add_scalar('Validation mAP',val_mAP,global_step=epoch) # type: ignore

    print("Training Done!")
    print(f"Best Valid mAP: {best_score:.4f} at epoch {best_epoch}")
    torch.save(best_stat_dict, save_model_path)

    print("===============Start Testing===============")
    model.load_state_dict(best_stat_dict)
    test_mAP = summary(model, dataloaders['test'], device, save_model_path, gt_csv="/home/stat-zx/CTCdet/csvfiles/test.csv")
    if use_tensorboard:
        writer.close() # type: ignore
