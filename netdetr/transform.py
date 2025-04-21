from typing import Optional, List

import torch
from torch import nn, Tensor
import torchvision.transforms.functional as F

from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class DETRTransform(nn.Module):
    def __init__(self, 
                 min_size = 800, 
                 max_size = 1333,
                 image_mean = [0.485, 0.456, 0.406], 
                 image_std = [0.229, 0.224, 0.225]):
        super(DETRTransform, self).__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def forward(self, images, targets):
        images = [img for img in images]
        if targets is not None:
            targets = [{k: v for k,v in t.items()} for t in targets]
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError(f"images is expected to be a list of 3d tensors of shape [C, H, W], got {image.shape}")
            image, target_index = self.resize(image, target_index, self.min_size, self.max_size)
            # note that here detr normalize boxes to cxcywh and within [0, 1] by /wh
            image, target_index = self.normalize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        out_images = self.nested_tensor_from_tensor_list(images)
        return out_images, targets
            
    def normalize(self, image, target):
        image = F.normalize(image, mean=self.image_mean, std=self.image_std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32, device=boxes.device)
            target["boxes"] = boxes
        return image, target
    
    def nested_tensor_from_tensor_list(self, tensor_list: List[Tensor]):
        # TODO make this more general
        if tensor_list[0].ndim == 3:

            # TODO make it support different-sized images
            max_size = _max_by_axis([list(img.shape) for img in tensor_list])
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = [len(tensor_list)] + max_size
            b, c, h, w = batch_shape
            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], :img.shape[2]] = False
        else:
            raise ValueError('not supported')
        return NestedTensor(tensor, mask)

    def resize(self, image, target, size, max_size=None):
        # size can be min_size (scalar) or (w, h) tuple

        def get_size_with_aspect_ratio(image_size, size, max_size=None):
            w, h = image_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)

        def get_size(image_size, size, max_size=None):
            if isinstance(size, (list, tuple)):
                return size[::-1]
            else:
                return get_size_with_aspect_ratio(image_size, size, max_size)

        size = get_size(image.shape[1:], size, max_size)
        rescaled_image = F.resize(image, size)

        if target is None:
            return rescaled_image, None

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.shape[1:], image.shape[1:]))
        ratio_width, ratio_height = ratios

        target = target.copy()
        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height],device=boxes.device)
            target["boxes"] = scaled_boxes

        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        h, w = size
        target["size"] = torch.tensor([h, w])

        if "masks" in target:
            target['masks'] = nn.functional.interpolate(
                target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

        return rescaled_image, target 

    def postprocess(self, outputs, img_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(img_sizes)
        assert img_sizes.shape[1] == 2

        prob = torch.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = img_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results