# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

# The following code is a modified version of detectron2's Cityscapes Panoptic Segmentation Dataset Mapper
# The original code can be found at
# https://github.com/facebookresearch/detectron2/blob/9604f5995cc628619f0e4fd913453b4d7d61db3f/detectron2/data/datasets/cityscapes.py

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

BDD100K_SEM_SEG_CATEGORIES = [
    {"name": "road", "id": 0, "trainId": 0},
    {"name": "sidewalk", "id": 1, "trainId": 1},
    {"name": "building", "id": 2, "trainId": 2},
    {"name": "wall", "id": 3, "trainId": 3},
    {"name": "fence", "id": 4, "trainId": 4},
    {"name": "pole", "id": 5, "trainId": 5},
    {"name": "traffic light", "id": 6, "trainId": 6},
    {"name": "traffic sign", "id": 7, "trainId": 7},
    {"name": "vegetation", "id": 8, "trainId": 8},
    {"name": "terrain", "id": 9, "trainId": 9},
    {"name": "sky", "id": 10, "trainId": 10},
    {"name": "person", "id": 11, "trainId": 11},
    {"name": "rider", "id": 12, "trainId": 12},
    {"name": "car", "id": 13, "trainId": 13},
    {"name": "truck", "id": 14, "trainId": 14},
    {"name": "bus", "id": 15, "trainId": 15},
    {"name": "train", "id": 16, "trainId": 16},
    {"name": "motorcycle", "id": 17, "trainId": 17},
    {"name": "bicycle", "id": 18, "trainId": 18},
]

def _get_bdd100k_sem_seg_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in BDD100K_SEM_SEG_CATEGORIES]
    assert len(stuff_ids) == 19, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in BDD100K_SEM_SEG_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }

    return ret

def register_bdd100k_sem_seg(root):
    root = os.path.join(root, "bdd100k")
    meta = _get_bdd100k_sem_seg_meta()
    for name, dirname in [("train", "train"), ("val", "val")]:
        image_dir = os.path.join(root, "images", "10k" ,dirname)
        gt_dir = os.path.join(root, "labels", "sem_seg", "masks", dirname)
        name = f"bdd_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_bdd100k_sem_seg(_root)