import random
import numpy as np
import torch
from torchvision.transforms import functional as F
import PIL
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    '''组合多个transforms函数'''
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1) # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(1)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    '''将PIL图像转为Tensor'''
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize(object):
    """
    Modified Normalize
    """

    def __call__(self, image, target):
        image = F.normalize(image,
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        return image, target


class ImgAugTransform(object):
    """
    Use imgaug package to do data augmentation
    """
    def __init__(self):
        # someof即每次从后续序列中选择部分项目,此处为0~3个项目
        self.aug = iaa.SomeOf((0, 3), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.Affine(rotate=90),
                    iaa.Affine(rotate=180),
                    iaa.Affine(rotate=270)]),  # 旋转one of 角度
            iaa.Multiply((0.8, 1.5),per_channel=0.5),  # 亮度变化
            iaa.Grayscale(0.6),   # 灰度图                
            iaa.GaussianBlur(sigma=(0.0, 5.0))  # 高斯扰动(变模糊)
        ])

    def __call__(self, image, target):
        image = np.array(image)
        bbs_list = []
        boxes = target["boxes"].numpy()
        for box in boxes:
            bbs_ele = BoundingBox(x1=box[0], y1=box[1],
                                  x2=box[2], y2=box[3])
            bbs_list.append(bbs_ele)
        bbs = BoundingBoxesOnImage(bbs_list, shape=image.shape)
        seq_det = self.aug.to_deterministic()
        image, boxes = seq_det(image=image, bounding_boxes=bbs)
        boxes = np.array([[i.x1, i.y1, i.x2, i.y2]
                          for i in boxes.bounding_boxes])
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        return PIL.Image.fromarray(image), target