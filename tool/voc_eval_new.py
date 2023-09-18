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
    gt_csv: path/to/ground_truth_csv
    pred_csv: path/to/pred_csv
    ovthresh: iou threshold    
    """
    # parse ground truth csv, by parsing the ground truth csv,
    # we get ground box info
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

    AP_dict = {}
    for label in label_list:
        label_preds = [pred for pred in preds if pred.pred_type == label]

        # sort prediction by probability, decrease order
        label_preds = sorted(label_preds, key=lambda x: x.probability, reverse=True)
        nd = len(label_preds)  # total number of pred boxes
        
        object_hitted = set()
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        num_object = sum([len(obj) for obj in object_dict[label].values()])

        # loop over each pred box to see if it matches one ground box
        for d in range(nd):
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
                    if BBGT_hitted_flag[jmax] not in object_hitted:
                        tp[d] = 1.
                        object_hitted.add(BBGT_hitted_flag[jmax])
                    else:
                        fp[d] = 1.
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

        print(f'预测框数量: {len(label_preds)}')
        print(f'TP框数量: {sum(tp)}')
        print(f'FP框数量: {sum(fp)}')
         
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        # cal recall
        rec = tp / float(num_object)
        # cal precision
        prec = tp / np.maximum(tp+fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        AP_dict[label] = ap

    mAP = sum([ap for ap in AP_dict.values()])/len(label_list)
    return AP_dict,mAP


if __name__ == "__main__":
    # gt_csv = "../statistic_description/tmp/test.csv"
    # pred_csv = "../tmp/detection_results/loc.csv"

    recall, precision, ap = custom_voc_eval(gt_csv, pred_csv)
    import ipdb;ipdb.set_trace()

