# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_cat_seg_config, add_lora_config

# dataset loading
from .data.dataset_mappers.detr_panoptic_dataset_mapper import DETRPanopticDatasetMapper
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)
from .data.dataset_mappers.resized_mask_former_semantic_dataset_mapper import (
    ResizedMaskFormerSemanticDatasetMapper,
)
from .data.dataset_mappers.resized_mask_former_semantic_test_dataset_mapper import(
    ResizedMaskFormerSemanticTestDatasetMapper
)
# models
from .cat_seg_model import CATSeg
from .test_time_augmentation import SemanticSegmentorWithTTA