#!/usr/bin/env python
# coding=utf-8
import numpy as np

from collections import namedtuple


Object = namedtuple("Object",
                    ["image_path", "object_id", "object_type", "coordinates"])
                    
Prediction = namedtuple("Prediction",
                        ["image_path", "pred_type", "probability", "coordinates"])


def voc_ap(recall, precision, use_07_metric=False):
    """
    Calculate the AP value using recall and precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap = ap + p/11.

    else:
        mrec = np.concatenate(([0.], recall, [1.]))
        mprec = np.concatenate(([0.], precision, [0.]))
        for i in range(mprec.size - 1, 0, -1):
            mprec[i-1] = np.maximum(mprec[i-1], mprec[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i+1] - mrec[i]) * mprec[i+1])

    return ap


def custom_voc_eval(gt_csv, pred_csv, label_list = ['1','2'], ovthresh=0.5, use_07_metric=False):
    """
    Do custom eval, include mAP and FROC,
    此函数已改为多类别分别计算AP值,并返回mAP,需传入label_list循环各类别,
    计算froc则在froc.py中进行
    
    gt_csv: path/to/ground_truth_csv
    pred_csv: path/to/pred_csv
    label_list: 用于循环计算ap值的label列表
    ovthresh: iou threshold
    
    """
    # parse ground truth csv, by parsing the ground truth csv,
    # we get ground box info
    # 按类别为键分别存储每张图片的gt框,即第一层键为类别,第二层为图片路径
    # 每个路径下存储该图片该类别所有gt框
    object_dict = dict([(k,{}) for k in label_list])
    obj_id = 0
    with open(gt_csv) as f:
        # skip header
        next(f)
        for line in f:
            image_path, annotation = line.strip("\n").split(",")
            if annotation == "":
                continue
            
            object_annos = annotation.split(";")
            for object_anno in object_annos:
                fields = object_anno.split(" ")  # one box
                object_type = fields[0]
                coords = np.array(list(map(float, fields[1:])))
                
                obj = Object(image_path, obj_id, object_type, coords)
                # 此处判断该框路径是否在该类别的字典中已存在
                if image_path in object_dict[obj.object_type]:
                    object_dict[obj.object_type][image_path].append(obj)
                else:
                    object_dict[obj.object_type][image_path] = [obj]
                obj_id += 1
    
    # parse prediction csv, by parsing pred csv, we get the pre box info
    preds = []
    with open(pred_csv) as f:
        # skip header
        next(f)
        for line in f:
            image_path, prediction = line.strip("\n").split(",")
            
            if prediction == "":
                continue

            coord_predictions = prediction.split(";")
            for coord_prediction in coord_predictions:
                fields = coord_prediction.split(" ")
                pred_type = fields[0]
                probability, x1, y1, x2, y2 = list(map(float, fields[1:]))
                pred = Prediction(image_path, pred_type, probability,
                                  np.array([x1, y1, x2, y2]))
                preds.append(pred)


    # 对每个类别都计算一个ap值分别返回
    AP_dict = {}
    for label in label_list:
        label_preds = [pred for pred in preds if pred.pred_type == label]

        # sort prediction by probability, decrease order
        # key参数指定排序的规则,此处即使用preds的probability属性进行排序,reverse为T为降序
        label_preds = sorted(label_preds, key=lambda x: x.probability, reverse=True)
        nd = len(label_preds)  # total number of pred boxes
        
        # set()创建一个无序不重复元素集,此处用于存储
        object_hitted = set()
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        # 取出每个种类的每个图片path的gt框个数,求和后即为该类别所有gt框数量
        num_object = sum([len(obj) for obj in object_dict[label].values()])

        # loop over each pred box to see if it matches one ground box
        # 循环每个预测框,匹配gt框
        for d in range(nd):
            # 先判断预测的图像文件是否在gt解析对象字典中
            if label_preds[d].image_path in object_dict[label]:
                # one pred box coords
                bb = label_preds[d].coordinates.astype(float)
                image_path = label_preds[d].image_path
                # set the initial max overlap iou
                ovmax = -np.inf
                # ground box on the image
                R = [i.coordinates for i in object_dict[label][image_path]]
                try:
                    BBGT = np.stack(R, axis=0)
                except ValueError:
                    import ipdb;ipdb.set_trace()
                R_img_id = [i.object_id for i in object_dict[label][image_path]]
                BBGT_hitted_flag = np.stack(R_img_id, axis=0)

                if BBGT.size > 0:
                    # cal the iou between pred box and all the gt boxes on
                    # the image
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])

                    # cal inter area width
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih  # inter area

                    # cal iou为什么都加1.?
                    # 因为像素点起点为(0,0),终点为(weight-1,height-1)计算值时需要加1.
                    union = (
                        (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + \
                        (BBGT[:, 2] - BBGT[:, 0] + 1.) * (BBGT[:, 3] - BBGT[:, 1] + 1.) - \
                        inters
                    )
                    overlaps = inters / union
                    # find the max iou
                    ovmax = np.max(overlaps)
                    # find the index of the max iou
                    jmax = np.argmax(overlaps)
                
                
                if ovmax > ovthresh:
                    # 如果最大的iou大于iou阈值0.5,则该预测框判断为TP,同时将该gt框设为已标记
                    # 如果匹配到的gt框大于阈值但已被标记,则将预测框判断为FP,  # 这步考虑一下,并无判断失误的框应该算作fp框吗
                    # 因预测框按照置信度排序,证明有更高置信度的预测框匹配该gt                
                    if BBGT_hitted_flag[jmax] not in object_hitted:
                        tp[d] = 1.
                        object_hitted.add(BBGT_hitted_flag[jmax])
                    # else:
                    #     fp[d] = 1.

                else:
                    fp[d] = 1.
            
            # 此处应该加上这个判断,如果测试集存在没有gt框的图片,
            # 那么预测的框图片路径不在obj字典中,说明也是fp假阳性,
            # 不加这步会导致漏掉假阳性的框
            else:
                fp[d] = 1.

        # 检测一下数量是否有遗漏
        print(f'预测框数量: {len(label_preds)}')
        print(f'TP框数量: {sum(tp)}')
        print(f'FP框数量: {sum(fp)}')

        # 每循环一个预测框就对该框做一个判断,只能为fp或tp,
        # 而累加就代表着在该预测框对应置信度阈值时所有有效框计算的fp与tp值           
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        # 此处计算的rec与prec为各阈值时累加值计算的一串数
        # cal recall
        rec = tp / float(num_object)
        # cal precision
        prec = tp / np.maximum(tp+fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        # 存储该类别的ap值
        AP_dict[label] = ap

    # 计算平均map
    mAP = sum([ap for ap in AP_dict.values()])/len(label_list)
    return AP_dict,mAP


if __name__ == "__main__":
    # gt_csv = "../statistic_description/tmp/test.csv"
    # pred_csv = "../tmp/detection_results/loc.csv"
    gt_csv = "/home/stat-caolei/code/TCT_V3/doctor_gt.csv"
    pred_csv = "/home/stat-caolei/code/TCT_V3/doctor_pred.csv"

    recall, precision, ap = custom_voc_eval(gt_csv, pred_csv)
    import ipdb;ipdb.set_trace()

