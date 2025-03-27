# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

# The following code is a modified version of detectron2's Cityscapes Panoptic Segmentation Dataset Mapper
# The original code can be found at
# https://github.com/facebookresearch/detectron2/blob/9604f5995cc628619f0e4fd913453b4d7d61db3f/detectron2/data/datasets/cityscapes.py

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

import logging

logger = logging.getLogger(__name__)

IDD_COLORED_SPLITS = {
    "idd_{task}_train": ("IDD_Segmentation/leftImg8bit/train", "IDD_Segmentation/gtFine/train"),
    "idd_{task}_val": ("IDD_Segmentation/leftImg8bit/val", "IDD_Segmentation/gtFine/val"),
}

# Only one category is kept for each class with more than one category. Special categories are removed.
IDD_SEM_SEG_CATEGORIES = {
    "road": 0,
    "parking": 1,
    "sidewalk": 2,
    "rail track": 3,
    # "person": 4,
    "animal": 4,
    "rider": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "auto rickshaw": 8,
    "car": 9,
    "truck": 10,
    "bus": 11,
    "caravan": 12,
    # "trailer": 12,
    # "train": 12,  
    "curb": 13,
    "wall": 14,
    "fence": 15,
    "guard rail": 16,
    "billboard": 17,
    "traffic sign": 18,
    "traffic light": 19,
    "pole": 20,
    # "polegroup": 20,
    # "obs-str-bar-fallback": 21, # This was the original
    "bar": 21, # This was the original
    "building": 22,
    # "tunnel": 23,
    "bridge": 23,
    "vegetation": 24,
    "sky": 25,
    # "drivable fallback": 1,
    # "non-drivable fallback": 3,
    # "vehicle fallback": 12,
    # "fallback background": 25,
    # "out of roi": 255,
    # "rectification border": 255,
    # "license plate": 255,
    # "unlabeled": 255,
    # "ego vehicle": 255,
}


def _get_idd_sem_seg_meta():

    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k for k in IDD_SEM_SEG_CATEGORIES.values()]
    # assert len(stuff_ids) == 19, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: k for k, k in enumerate(stuff_ids)}
    stuff_classes = [k for k in IDD_SEM_SEG_CATEGORIES.keys()]

    ret = {
        # "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }

    return ret

def _get_idd_files(image_dir, gt_dir):
    files = []
    # scan through the directory
    folders = PathManager.ls(image_dir)
    logger.info(f"{len(folders)} folder found in '{image_dir}'.")
    for folder in folders:
        folder_img_dir = os.path.join(image_dir, folder)
        folder_gt_dir = os.path.join(gt_dir, folder)
        
        for basename in PathManager.ls(folder_img_dir):
            image_file = os.path.join(folder_img_dir, basename)

            suffix = "_leftImg8bit"
            suffix_with_extension = "_leftImg8bit.png"
            assert suffix in basename, basename
            basename = basename[: -len(suffix_with_extension)]

            label_file = os.path.join(folder_gt_dir, basename + "_gtFine_labellevel3Ids.png")

            files.append((image_file, label_file))

    assert len(files), "No images found in {}".format(image_dir)

    for f in files[0]:
        assert PathManager.isfile(f), f

    return files

def load_idd_semantic(image_dir, gt_dir):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".

    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    # gt_dir is small and contain many small files. make sense to fetch to local first
    gt_dir = PathManager.get_local_path(gt_dir)
    for image_file, label_file in _get_idd_files(image_dir, gt_dir):
        # label_file = label_file.replace("labelIds", "labelTrainIds")

        ret.append(
            {
                "file_name": image_file,
                "sem_seg_file_name": label_file,
            }
        )

    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Ground truth not found!"  # noqa
    return ret


def register_idd_dataset(root):
    for key, (image_dir, gt_dir) in IDD_COLORED_SPLITS.items():
        
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        meta = _get_idd_sem_seg_meta()
        sem_key = key.format(task="sem_seg")

        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_idd_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta
        )
        
_root = os.getenv("DETECTRON2_DATASETS", "datasets")

register_idd_dataset(_root)