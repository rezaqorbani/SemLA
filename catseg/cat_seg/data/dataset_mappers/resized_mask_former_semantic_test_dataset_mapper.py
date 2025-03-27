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

from detectron2.data.dataset_mapper import DatasetMapper


import cv2

__all__ = ["ResizedMaskFormerSemanticTestDatasetMapper"]

class ResizedMaskFormerSemanticTestDatasetMapper(DatasetMapper):
    """
    A version of ResizedMaskFormerSemanticDatasetMapper for evaluation purposes.
    This mapper resizes the masks to match the resolution of input images but skips
    any training-specific augmentations.
    """
        
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)

        new_width = 1024
        new_height = 512
        
        if "sem_seg_file_name" in dataset_dict:
            # Resizing segmentation mask to match image resolution
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")

            sem_seg_gt_resized = cv2.resize(sem_seg_gt, (new_width, new_height), interpolation=cv2.INTER_NEAREST_EXACT)
            sem_seg_gt = sem_seg_gt_resized
            
        else:
            sem_seg_gt = None
        
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST_EXACT)
        
        #Update the dataset_dict with the new dimensions
        dataset_dict["width"] = new_width
        dataset_dict["height"] = new_height
        
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict