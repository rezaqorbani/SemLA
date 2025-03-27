# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

# The following code is a modified version of detectron2's Cityscapes Panoptic Segmentation Dataset Mapper
# The original code can be found at
# https://github.com/facebookresearch/detectron2/blob/9604f5995cc628619f0e4fd913453b4d7d61db3f/detectron2/data/datasets/cityscapes.py

import functools
import json
import logging
import multiprocessing as mp
import os
from itertools import chain

import numpy as np
import pycocotools.mask as mask_util

from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from PIL import Image

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

logger = logging.getLogger(__name__)


def _get_muses_files(image_dir, gt_dir):
    files = []
    # scan through the directory
    # cities = PathManager.ls(image_dir)
    # logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    # for city in cities:
        # city_img_dir = os.path.join(image_dir, city)
        # city_gt_dir = os.path.join(gt_dir, city)
    for basename in PathManager.ls(image_dir):
        image_file = os.path.join(image_dir, basename)

        suffix = "frame_camera.png"

        assert basename.endswith(suffix), basename
        if basename.endswith(suffix):
            basename = basename[: -len(suffix)]
            
            instance_file = os.path.join(
                gt_dir, basename + "gtFine_instanceIds.png"
            )
            label_file = os.path.join(gt_dir, basename + "gt_labelIds.png")
            json_file = os.path.join(gt_dir, basename + "gtFine_polygons.json")

            files.append((image_file, label_file))

    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def load_muses_semantic(image_dir, gt_dir):
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
    for image_file, label_file in _get_muses_files(
        image_dir, gt_dir
    ):
        label_file = label_file.replace("labelIds", "labelTrainIds")

        ret.append(
            {
                "file_name": image_file,
                "sem_seg_file_name": label_file,
                "height": 1080,
                "width": 1920,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
    return ret