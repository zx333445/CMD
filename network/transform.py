#!/usr/bin/env python
# coding=utf-8
import math
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn, Tensor

from .image_list import ImageList


def _resize_image(image, self_min_size, self_max_size):
    # type: (Tensor, float, float) -> Tensor
    im_shape = torch.tensor(image.shape[-2:])
    min_size = float(torch.min(im_shape))    
    max_size = float(torch.max(im_shape))    
    scale_factor = self_min_size / min_size  

    if max_size * scale_factor > self_max_size:
        scale_factor = self_max_size / max_size   

    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True,
        align_corners=False)[0]

    return image


def resize_boxes(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    """
    Arguments:
        original_size: 图像缩放前的尺寸
        new_size: 图像缩放后的尺寸
    """
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratios_height, ratios_width = ratios
    # Removes a tensor dimension, boxes [minibatch, 4]
    # Returns a tuple of all slices along a given dimension, already without it.
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)



class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size      
        self.max_size = max_size     
        self.image_mean = image_mean  
        self.image_std = image_std   

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # [:, None, None]: shape [3] -> [3, 1, 1]
        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k):
        # type: (List[int]) -> int
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    def resize(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
        """
        Args:
            image: 输入的图片
            target: 输入图片的相关信息(包括bboxes信息)
        Returns:
            image: 缩放后的图片
            target: 缩放bboxes后的图片相关信息
        """
        # image shape is [channel, height, width]
        h, w = image.shape[-2:]

        if self.training:
            size = float(self.torch_choice(self.min_size))  
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])    

        image = _resize_image(image, size, float(self.max_size))

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, [h, w], image.shape[-2:])
        target["boxes"] = bbox

        return image, target


    def max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        """
        Args:
            images: 输入的一批图片
            size_divisible: 将图像高和宽调整到该数的整数倍
        Returns:
            batched_imgs: 打包成一个batch后的tensor数据
        """

        max_size = self.max_by_axis([list(img.shape) for img in images])

        stride = float(size_divisible)
        # max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        # [batch, channel, height, width]
        batch_shape = [len(images)] + max_size

        batched_imgs = images[0].new_full(size=batch_shape, fill_value=0)
        for img, pad_img in zip(images, batched_imgs):
            # copy_: Copies the elements from src into self tensor and returns self
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self,
                    result,                # type: List[Dict[str, Tensor]]
                    image_shapes,          # type: List[Tuple[int, int]]
                    original_image_sizes   # type: List[Tuple[int, int]]
                    ):
        # type: (...) -> List[Dict[str, Tensor]]
        """
        Args:
            result: list(dict), 网络的预测结果, len(result) == batch_size
            image_shapes: list(torch.Size), 图像预处理缩放后的尺寸, len(image_shapes) == batch_size
            original_image_sizes: list(torch.Size), 图像的原始尺寸, len(original_image_sizes) == batch_size
        Returns:
        """
        if self.training:
            return result

        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)  
            result[i]["boxes"] = boxes
        return result

    def __repr__(self):
        """"""
        format_string = self.__class__.__name__ + '('
        _indent = '\n    '
        format_string += "{0}Normalize(mean={1}, std={2})".format(_indent, self.image_mean, self.image_std)
        format_string += "{0}Resize(min_size={1}, max_size={2}, mode='bilinear')".format(_indent, self.min_size,
                                                                                         self.max_size)
        format_string += '\n)'
        return format_string

    def forward(self,
                images,       # type: List[Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image = self.normalize(image)               
            image, target_index = self.resize(image, target_index)   
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)  # 将images打包成一个batch
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])

        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets











