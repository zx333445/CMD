import argparse
from collections import namedtuple
import numpy as np
from skimage.measure import points_in_poly
import matplotlib.pyplot as plt


Object = namedtuple('Object',
                    ['image_path', 'object_id', 'object_type', 'coordinates',
                     'hit_flag'])
Prediction = namedtuple('Prediction',
                        ['image_path','pred_type', 'probability', 'coordinates'])


parser = argparse.ArgumentParser(description='Compute FROC')
parser.add_argument('gt_csv', default=None, metavar='GT_CSV',
                    type=str, help="Path to the ground truch csv file")
# parser.add_argument('pred_csv', default=None, metavar='PRED_PATH',
#                     type=str, help="Path to the predicted csv file")
# parser.add_argument('--fps', default='0.125,0.25,0.5,1,2,4,8', type=str,
#                     help='False positives per image to compute FROC, comma '
#                     'seperated, default "0.125,0.25,0.5,1,2,4,8"')
parser.add_argument('--type',default='1',type=str,
                    help="the object type of the froc curve {'1':'CTC','2':'CTC样'}")

def inside_object(pred, obj):
    # bounding box
    if obj.object_type == '0':
        x1, y1, x2, y2 = obj.coordinates
        x, y = pred.coordinates
        return x1 <= x <= x2 and y1 <= y <= y2
    # bounding ellipse
    if obj.object_type == '1':
        x1, y1, x2, y2 = obj.coordinates
        x, y = pred.coordinates
        x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
        x_axis, y_axis = (x2 - x1) / 2, (y2 - y1) / 2
        return ((x - x_center)/x_axis)**2 + ((y - y_center)/y_axis)**2 <= 1
    # mask/polygon
    if obj.object_type == '2':
        num_points = len(obj.coordinates) // 2
        poly_points = obj.coordinates.reshape(num_points, 2, order='C')
        return points_in_poly(pred.coordinates.reshape(1, 2), poly_points)[0]


def froc_curve(gt_csv,pred_csv,type,fps_list):
    '''多类别时是否应该每个类别绘制一个froc曲线,
    修改该函数为返回fps和froc曲线值(灵敏度)的列表,
    以实现多个模型froc曲线画在一张图上'''

    
    # iou overlap threshold, we set 0.5
    ovthresh = 0.5
    # parse ground truth csv
    num_image = 0
    num_object = 0
    object_dict = {}
    with open(gt_csv) as f:
        # header
        next(f)
        for line in f:
            image_path, annotation, _ = line.strip('\n').split(',')

            if annotation == '':
                num_image += 1
                continue

            object_annos = annotation.split(';')
            for object_anno in object_annos:
                fields = object_anno.split(' ')
                object_type = fields[0]
                # object_type = '1'
                # 判断类别是否是命令行指定的绘制froc曲线的类别
                if object_type == type:                    
                    # 判断完类别符合后先将gt数加1
                    num_object += 1
                    coords = np.array(list(map(float, fields[1:])))
                    hit_flag = False
                    obj = Object(image_path, num_object, object_type, coords,
                                hit_flag)
                    if image_path in object_dict:
                        object_dict[image_path].append(obj)
                    else:
                        object_dict[image_path] = [obj]
                    
            num_image += 1

    # parse prediction truth csv
    preds = []
    with open(pred_csv) as f:
        # header
        next(f)
        for line in f:
            image_path, prediction = line.strip('\n').split(',')

            if prediction == '':
                continue

            coord_predictions = prediction.split(';')
            for coord_prediction in coord_predictions:
                fields = coord_prediction.split(' ')
                pred_type = fields[0]
                # 同样根据指定的类别,筛选该类别预测框
                if pred_type == type:
                    probability, x1, y1, x2, y2 = list(map(float, fields[1:]))
                    pred = Prediction(image_path, pred_type, probability,
                                    np.array([x1, y1, x2, y2]))
                    preds.append(pred)
    # sort prediction by probabiliyt
    # 排序后之后循环时按照预测置信度高低依次匹配gt框,确保每次匹配的预测框是最为合适的
    preds = sorted(preds, key=lambda x: x.probability, reverse=True)

    # compute hits and false positives
    hits = 0
    false_positives = 0
    fps_idx = 0
    fps = fps_list
    fps_flag = [str(i) for i in fps]
    froc = []
    for i in range(len(preds)):
        pred = preds[i]

        # 此处有一个问题,如果预测框对应图片不在obj里面,那这个框也应该被判断是fp才对吧
        if pred.image_path in object_dict:
            objs = object_dict[pred.image_path]
            # gt boxes coords in this image
            BBGT = np.array([k.coordinates for k in objs]).astype(float)
            # gt boxes hit flag
            BBGT_HIT_FLAG = [k.hit_flag for k in objs]
            # this pred box coord
            bb = pred.coordinates.astype(float)
            # set the initial max iou
            ovmax = -np.inf

            if BBGT.size > 0:
                # calculate ious between this box and every gt boxes
                # on this image
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])

                # calculate overlap area width
                iw = np.maximum(ixmax-ixmin+1., 0.)
                # overlap area height
                ih = np.maximum(iymax-iymin+1., 0.)
                # overlap areas
                inters = iw * ih
                ## calculate ious
                # union
                union = ((bb[2]-bb[0]+1.)*(bb[3]-bb[1]+1.)+\
                        (BBGT[:, 2]-BBGT[:, 0]+1.)*(BBGT[:, 3]-BBGT[:, 1]+1.)-\
                        inters)
                # ious
                overlaps = inters / union
                # find the maximum iou
                ovmax = np.max(overlaps)
                # find the maximum iou index
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                # 如果最大的iou大于iou阈值0.5,则该预测框判断为TP,同时将该gt框设为已标记
                # 如果匹配到的gt框大于阈值但已被标记,则将预测框判断为FP,
                # 因预测框按照置信度排序,证明有更高置信度的预测框匹配该gt
                if not BBGT_HIT_FLAG[jmax]:
                    BBGT_HIT_FLAG[jmax] = True
                    hits += 1
                else:
                    false_positives += 1
            else:
                false_positives += 1

            # 在进行到该预测框置信度阈值时计算的fps超过给出的节点时,
            # 记录该阈值时的灵敏度(召回率)为froc曲线值
            # 若记录的值已经达到fps节点数,则停止循环,不进行下一个预测框
            if false_positives / num_image >= fps[fps_idx]:
                sensitivity = hits / float(num_object)
                froc.append(sensitivity)
                fps_idx += 1

                if len(fps) == len(froc):
                    break

        else:
            false_positives += 1

    # 检测一下数量是否有遗漏
    print(f'gt框数量: {num_object}')
    print(f'预测框数量: {len(preds)}')
    print(f'TP框数量: {hits}')
    print(f'FP框数量: {false_positives}')

    # 当预测框不够计算到节点值时,添加froc记录值最后的数值
    # 直至长度相同            
    while len(froc) < len(fps):
        froc.append(froc[-1])

    # 打印出各fps节点的灵敏度值
    print("False positives per image:")
    print("\t".join(fps_flag))
    print("Sensitivity:")
    print("\t".join(map(lambda x: "{:.3f}".format(x), froc)))
    print("FROC:")
    print(np.mean(froc))

    return froc
    


if __name__ == '__main__':
    
    args = parser.parse_args()   
    pred_csv = ['/home/stat-zx/TCTdet/results/faster_loc.csv',
                '/home/stat-zx/TCTdet/results/cascade_loc.csv',
                '/home/stat-zx/TCTdet/results/sparse_loc.csv',] 
    # pred_csv = ['/home/stat-zx/TCTdet/results/ssd_loc.csv',
    #             '/home/stat-zx/TCTdet/results/retina_loc.csv',
    #             '/home/stat-zx/TCTdet/results/fcos_loc.csv',]
    
    
    # 注意若出现灵敏度达不到最高值时,说明点分的不够细,有的值未记录上
    fps_list = list(np.arange(0, 1.05, 0.05))
    # fps_list = [0.0,0.005,0.01,0.02,0.04,0.08,0.1,0.5]
    # fps_list = [0.125,0.25,0.5,1,2,4,8,16,32]
    print("========plot curve========")
    palette = plt.get_cmap('tab10')
    

    # 设置rcParams参数(绘图theme)
    # 坐标刻度文字大小、坐标刻度线长度宽度、坐标轴宽度
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['axes.linewidth'] = 1.5
    
    
    fig, ax = plt.subplots(figsize=(7, 6))
    plt.xlim(fps_list[0],fps_list[-1])
    plt.ylim(0,1.)
    ax.set_xlabel("False Positives Per Image",fontsize = 16)
    ax.set_ylabel("True Positive Rate",fontsize = 16)
    plt.grid(linestyle="dashed")
    labels = ['Faster R-CNN','Cascade R-CNN','Sparse R-CNN']
    # labels = ['SSD','Retina Net','FCOS']
    for i in range(len(pred_csv)):  
        froc = froc_curve(args.gt_csv,pred_csv[i],args.type,fps_list)
        ax.plot(fps_list, froc, 
                linewidth=2,
                color = palette(i+3),
                label=labels[i])
    # plt.subplots_adjust(right=0.6)
    plt.legend(loc = 'lower right', fontsize = 16)
    # plt.title("FROC of one-stage models",y=1.05,fontsize = 16)
    plt.title("FROC of two-stage models",y=1.05,fontsize = 16)
    # plt.tight_layout()
    # plt.savefig("TCT_froc1.png", dpi=300)
    plt.savefig("TCT_froc2.png", dpi=300)
    plt.close()
