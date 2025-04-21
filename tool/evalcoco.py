"""
该脚本用于调用训练好的模型权重去计算验证集/测试集的COCO指标
以及每个类别的mAP(IoU=0.5)
COCO数据集的评价标准中,把IoU的值从50%到95%每隔5%进行了一次划分
0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 ,每次测试的时候都是在IoU=这个数上测试
在这10组precision-recall对中,10个PR曲线下得到的AP值,然后对这10个AP进行平均,得到了一个AP@[0.5:0.95]

竟然算出来比voc标准的要高(原因是忘了筛选结果)
"""
import torch
from tqdm import tqdm
import numpy as np
import _utils

from network.backbone_utils import resnet_fpn_backbone, densenet_fpn_backbone, swin_fpn_backbone, convnext_fpn_backbone
from network import ssdresbackbone, SSD, FCOS, RetinaNet, FasterRCNN, CascadeRCNN
from netsparse import SparseRCNN
from tool import transforms as T

from datasets import TCTDataset
from tool.coco_utils import get_coco_api_from_dataset
from tool.coco_eval import CocoEvaluator


def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


def main():
    device = torch.device('cuda:0')   
    print("Using {} device training.".format(device.type))

    CLASSES = {"__background__", "abnormal"}
    class_dict = {'abnormal':1}
    category_index = {v: k for k, v in class_dict.items()}

    print("=========loading data===========")
    dataset = TCTDataset(
        "/home/stat-zx/TCT_FIFTH", 
        transforms=T.Compose([T.ToTensor()]), 
        train=True,
        csv_name='test.csv')

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=12,
        collate_fn=_utils.collate_fn
    )

    # pth_path = '/home/stat-zx/TCTdet/results/ssd.pth'
    # pth_path = '/home/stat-zx/TCTdet/results/fcos.pth'
    # pth_path = '/home/stat-zx/TCTdet/results/retina.pth'
    # pth_path = '/home/stat-zx/TCTdet/results/faster.pth'
    # pth_path = '/home/stat-zx/TCTdet/results/cascade.pth'
    pth_path = '/home/stat-zx/TCTdet/results/sparse.pth'

    # SSD
    # model = SSD(ssdresbackbone(), num_classes=len(CLASSES))

    # FCOS
    # backbone = resnet_fpn_backbone('resnet50', True)
    # model = FCOS(backbone, num_classes=len(CLASSES))

    # Retinanet
    # backbone = resnet_fpn_backbone('resnet50', True)
    # model = RetinaNet(backbone, num_classes=len(CLASSES))

    # Faster RCNN
    # backbone = resnet_fpn_backbone('resnet50', True)
    # model = FasterRCNN(backbone, num_classes=len(CLASSES))

    # Cascade RCNN
    # backbone = resnet_fpn_backbone('resnet50', True)
    # model = CascadeRCNN(backbone, num_classes=len(CLASSES))
    
    # Sparse RCNN
    backbone = resnet_fpn_backbone('resnet50', True)
    model = SparseRCNN(backbone=backbone,num_cls=len(CLASSES))
    print(model)
    
    model.to(device)
    statdic = torch.load(pth_path)
    model.load_state_dict(statdic)
    model.eval()

    # evaluate on the test dataset
    coco = get_coco_api_from_dataset(dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    cpu_device = torch.device("cpu")

    model.eval()
    with torch.no_grad():
        for image, targets in tqdm(dataloader, desc="validation..."):
            # 将图片传入指定设备device
            image = list(img.to(device) for img in image)
            outputs = model(image)
            
            # 别忘了筛选保留结果其他模型
            # new_output_index = torch.where(outputs[-1]["scores"] > 0.5)
            # new_boxes = outputs[-1]["boxes"][new_output_index]
            # new_scores = outputs[-1]["scores"][new_output_index]
            # new_labels = outputs[-1]["labels"][new_output_index]
            # Saprse RCNN的结果保留方式与其他不同
            new_output_index = torch.where(outputs["scores"] > 0.5)
            new_boxes = outputs["boxes"][new_output_index]
            new_scores = outputs["scores"][new_output_index]
            new_labels = outputs["labels"][new_output_index]

            new_outputs = [{'boxes':new_boxes,'labels':new_labels,'scores':new_scores}]
            new_outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in new_outputs]
            res = {target["image_id"].item(): output for target, output in zip(targets, new_outputs)}
            coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_eval = coco_evaluator.coco_eval["bbox"]
    # calculate COCO info for all classes
    coco_stats, print_coco = summarize(coco_eval)

    # calculate voc info for every classes(IoU=0.5)
    voc_map_info_list = []
    for i in range(len(category_index)):
        stats, _ = summarize(coco_eval, catId=i)
        voc_map_info_list.append(" {:15}: {}".format(category_index[i + 1], stats[1]))

    print_voc = "\n".join(voc_map_info_list)
    print(print_voc)

    # 将验证结果保存至txt文件中
    # with open("recordssd_mAP.txt", "w") as f:
    # with open("recordfcos_mAP.txt", "w") as f:
    # with open("recordretina_mAP.txt", "w") as f:
    # with open("recordfaster_mAP.txt", "w") as f:
    # with open("recordcascade_mAP.txt", "w") as f:
    with open("recordsparse_mAP.txt", "w") as f:
        record_lines = ["COCO results:",
                        print_coco,
                        "",
                        "mAP(IoU=0.5) for each category:",
                        print_voc]
        f.write("\n".join(record_lines))


if __name__ == "__main__":

    main()
