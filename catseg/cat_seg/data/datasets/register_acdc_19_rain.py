# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

# The following code is a modified version of detectron2's Cityscapes Panoptic Segmentation Dataset Mapper
# The original code can be found at
# https://github.com/facebookresearch/detectron2/blob/9604f5995cc628619f0e4fd913453b4d7d61db3f/detectron2/data/datasets/cityscapes.py

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from .acdc import load_acdc_semantic
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata, CITYSCAPES_CATEGORIES

ACDC_RAIN_SPLITS = {
    "acdc-rain_{task}_train": ("acdc/rgb_anon/rain/train/",
                                    "acdc/gt/rain/train/"),
    "acdc-rain_{task}_val": ("acdc/rgb_anon/rain/val/",
                                    "acdc/gt/rain/val/"),
    "acdc-rain_{task}_test": ("acdc/rgb_anon/rain/test/",
                                    "acdc/gt/rain/test/"),
}

def register_acdc_rain(root):
    for key, (image_dir, gt_dir) in ACDC_RAIN_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        sem_key = key.format(task="sem_seg")

        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_acdc_semantic(x, y)
        )

        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="acdc_sem_seg",
            ignore_label=255,
            thing_colors=[k["color"] for k in CITYSCAPES_CATEGORIES if k["name"] in meta["thing_classes"]],
            stuff_colors=[k["color"] for k in CITYSCAPES_CATEGORIES if k["name"] in meta["stuff_classes"]],
            **meta,
        )
        
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_acdc_rain(_root)