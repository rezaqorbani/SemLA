import os
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog


classes = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves",
    "curtain", "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling",
    "books", "refrigerator", "television", "paper", "towel", "shower curtain",
    "box", "whiteboard", "person", "night stand", "toilet", "sink", "lamp",
    "bathtub", "bag", "structure", "furniture", "prop"
]


def get_nyudv2_dicts(path: str, mode: str):
    """
    Get a mapping for rgb, depth and labels
    args:
        path: path to the dataset root.
        mode: train or eval
    """
    dataset_dicts = []
    file_path = f"{path}/{mode}.txt"
    
    # Read the first image to get height and width
    with open(file_path, "r") as file:
        first_line = file.readline().strip()
        rgb, label, depth = first_line.split(",")
        height, width = cv2.imread(f"{path}/{rgb}").shape[:2]
    
    with open(file_path, "r") as file:
        for idx, line in enumerate(file):
            rgb, label, depth = line.strip().split(",")
            record = dict()
            record["file_name"] = f"{path}/{rgb}"
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
            # corresponding depth
            record["depth_name"] = f"{path}/{depth}"
            # corresponding label
            record["sem_seg_file_name"] = f"{path}/{label}"
            dataset_dicts.append(record)
    return dataset_dicts

def register_nyudv2_dataset(base_path):
    for d in ["train", "test"]:
        split = 'val' if d == 'test' else 'train'
        DatasetCatalog.register(f"nyu_sem_seg_{split}", lambda d=d: get_nyudv2_dicts(base_path, d))
        MetadataCatalog.get(f"nyu_sem_seg_{split}").set(
            evaluator_type="sem_seg",
            ignore_label=255,
            num_labels=40,
            stuff_classes=classes,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_nyudv2_dataset(_root + "nyudv2")