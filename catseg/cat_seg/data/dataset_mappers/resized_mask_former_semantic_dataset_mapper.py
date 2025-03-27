# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

# The following code is a modified version of detectron2's MaskFormerSemanticDatasetMapper
# The original code can be found at 
# https://github.com/facebookresearch/Mask2Former/blob/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/mask2former/data/dataset_mappers/mask_former_semantic_dataset_mapper.py

import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances
from .mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper
import cv2

__all__ = ["ResizedMaskFormerSemanticDatasetMapper"]


class ResizedMaskFormerSemanticDatasetMapper(MaskFormerSemanticDatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image ∏d annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        
        new_width = 1024
        new_height = 512
        
        if "sem_seg_file_name" in dataset_dict:
        # Original loading of segmentation mask
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")

            # Resize segmentation mask to match the training images (half size in this context)
            sem_seg_gt_resized = cv2.resize(sem_seg_gt, (new_width, new_height), interpolation=cv2.INTER_NEAREST_EXACT)

            # Update sem_seg_gt to carry forward
            sem_seg_gt = sem_seg_gt_resized
        else:
            sem_seg_gt = None

        # resize image
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST_EXACT)
        
        # Now, update the dataset_dict with the new dimensions
        dataset_dict["width"] = new_width
        dataset_dict["height"] = new_height
        
        utils.check_image_size(dataset_dict, image)
        
        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        # import ipdb; ipdb.set_trace()
        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            # The ori_size is not the real original size, but size before padding
            dataset_dict['ori_size'] = image_size
            padding_size = [
                0,
                self.size_divisibility - image_size[1], # w: (left, right)
                0,
                self.size_divisibility - image_size[0], # h: 0,(top, bottom)
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict
