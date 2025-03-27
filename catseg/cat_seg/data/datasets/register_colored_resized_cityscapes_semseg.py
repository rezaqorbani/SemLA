# Copyright (c) Meta Platforms, Inc. All Rights Reserved

# The following code is a modified version of detectron2's Cityscapes Panoptic Segmentation Dataset Mapper
# The original code can be found at
# https://github.com/facebookresearch/detectron2/blob/9604f5995cc628619f0e4fd913453b4d7d61db3f/detectron2/data/datasets/cityscapes.py

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.cityscapes import load_cityscapes_semantic
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata, CITYSCAPES_CATEGORIES
        
_RAW_CITYSCAPES_COLORED_RESIZED_SPLITS = {
    "cs-normal_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cs-normal_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
}

def register_colored_resized_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_COLORED_RESIZED_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="resized_cityscapes_sem_seg",
            ignore_label=255,
            thing_colors=[k["color"] for k in CITYSCAPES_CATEGORIES if k["name"] in meta["thing_classes"]],
            stuff_colors=[k["color"] for k in CITYSCAPES_CATEGORIES if k["name"] in meta["stuff_classes"]],
            **meta,
        )
        
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_colored_resized_cityscapes(_root)