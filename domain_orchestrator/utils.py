from argparse import Namespace

import logging
import os
from typing import Literal

import torch

DETECTRON2_DATASET_PATH = os.getenv("DETECTRON2_DATASETS")

def get_domain_args(
    domain_name: str,
    split: Literal['train', 'val'],
    mode: str = "lora",
    base_model_path: str = "models/model_final.pth",
    num_gpus: int = 1,
    get_cofing_only: bool = False,
):
    logger_names = ["detectron2", "d2", "fvcore"]
    for name in logger_names:
        logger = logging.getLogger(name)
        if logger.hasHandlers():
            logger.handlers.clear()

    parts = domain_name.split("-")

    split_list = ["", "", ""]

    for i, part in enumerate(parts):
        split_list[i] = part

    dataset, domain, sub_domain = split_list

    # Supported configurations
    MODE_CHECK = {"lora"}

    DATASET_CHECK = {
        "cs",
        "acdc",
        "muses",
        "bdd",
        "mv",
        "a150",
        "idd",
        "pc59",
        "nyu",
        "coconutL"
    }

    CS_DOMAIN_CHECK = {"normal", "rain"}
    CS_SUB_DOMAIN_CHECK = ["25mm", "50mm", "75mm", "100mm", "200mm"]

    ACDC_DOMAIN_CHECK = {"fog", "night", "snow", "rain"}

    MUSES_DOMAIN_CHECK = {"clear", "rain", "fog", "snow"}
    MUSES_SUB_DOMAIN_CHECK = {"day", "night"}

    # Configurations assertions
    assert dataset in DATASET_CHECK

    if dataset == "cs":
        assert (
            domain in CS_DOMAIN_CHECK
        ), "Domain '{domain}' not supported for Cityscapes"
        if domain == "rain":
            assert (
                sub_domain in CS_SUB_DOMAIN_CHECK
            ), "Given volume '{volume}' is not supported for Cityscapes"
        elif domain == "normal":
            assert (
                sub_domain == ""
            ), f"Volume '{sub_domain}' is not supported for this domain in Cityscapes"
    elif dataset == "muses":
        assert (
            domain in MUSES_DOMAIN_CHECK
        ), f"Domain '{domain}' not supported for MUSES"
        assert (
            sub_domain in MUSES_SUB_DOMAIN_CHECK
        ), f"Given illumination '{sub_domain}' is not supported for MUSES"
    elif dataset == "acdc":
        assert (
            domain in ACDC_DOMAIN_CHECK
        ), f"Domain '{domain}' is not supported for ACDC"
        assert sub_domain == "", "Volume is not supported in ACDC"

    assert mode in MODE_CHECK, "Mode '{mode}' not supported"

    configs = {
        "cs": {
            "rain": f"configs/cityscapes/rain/{sub_domain}/{mode}-{domain}-{sub_domain}.yaml",
            "normal": f"configs/cityscapes/normal/{mode}-{domain}.yaml",
        },
        "acdc": {f"{domain}": f"configs/acdc/{domain}/{mode}-{domain}-acdc.yaml"},
        "muses": {
            f"{domain}": f"configs/muses/{domain}/muses-{domain}-{sub_domain}.yaml"
        },
        "bdd": "configs/bdd/bdd.yaml",
        "mv": "configs/mv/mv.yaml",
        "nyu": "configs/nyu/nyu.yaml",
        "a150": "configs/a150/a150.yaml",
        "idd": "configs/idd/idd.yaml",
        'pc59': 'configs/pc59/pc59.yaml',
        'nyu': 'configs/nyu/nyu.yaml',
        'coconutL': 'configs/coconutL/coconutL.yaml'
    }

    datasets = {
        "cs": {
            "normal": {
                "train": f"{DETECTRON2_DATASET_PATH}cityscapes/leftImg8bit/train/",
                "val": f"{DETECTRON2_DATASET_PATH}cityscapes/leftImg8bit/val/",
            },
        },
        "acdc": {
            "train": f"{DETECTRON2_DATASET_PATH}acdc/rgb_anon/{domain}/train/",
            "val": f"{DETECTRON2_DATASET_PATH}acdc/rgb_anon/{domain}/val/",
        },
        "muses": {
            "train": f"{DETECTRON2_DATASET_PATH}muses/frame_camera/train/{domain}/{sub_domain}/",
            "val": f"{DETECTRON2_DATASET_PATH}muses/frame_camera/val/{domain}/{sub_domain}/",
        },
        "bdd": {
            "train": f"{DETECTRON2_DATASET_PATH}bdd100k/images/10k/train/",
            "val": f"{DETECTRON2_DATASET_PATH}bdd100k/images/10k/val/",
        },
        "mv": {
            "train": f"{DETECTRON2_DATASET_PATH}mapillary_vistas/train/images/",
            "val": f"{DETECTRON2_DATASET_PATH}mapillary_vistas/val/images/",
        },
        "a150": {
            "train": f"{DETECTRON2_DATASET_PATH}ADE20k/images/training/",
            "val": f"{DETECTRON2_DATASET_PATH}ADE20k/images/validation/",
        },
        "idd": {
            "train": f"{DETECTRON2_DATASET_PATH}IDD_Segmentation/leftImg8bit/train/",
            "val": f"{DETECTRON2_DATASET_PATH}IDD_Segmentation/leftImg8bit/val/",
        },
        "pc59": {
            "train": f"{DETECTRON2_DATASET_PATH}pascal_ctx_d2/images/training",
            "val": f"{DETECTRON2_DATASET_PATH}pascal_ctx_d2/images/validation",
        },
        "nyu": {
            "train": f"{DETECTRON2_DATASET_PATH}nyudv2_splitted/train/rgb",
            "val": f"{DETECTRON2_DATASET_PATH}nyudv2_splitted/test/rgb",
        },
        "coconutL": {
            "train": f"{DETECTRON2_DATASET_PATH}coconut-l/train2017/",
            "val": f"{DETECTRON2_DATASET_PATH}coconut-l/val2017",
        },
    }

    # Output path configuration
    output_path = (
        f"output/{dataset}/{mode}-{dataset}"
        + (f"-{domain}" if  domain != "" else "")
        + (f"-{sub_domain}" if sub_domain != "" else "")
        + "/eval/"
    )

    # Constructing the return values
    if domain == "" and sub_domain == "":
        config_file = configs[dataset]
    else:
        config_file = configs[dataset][domain]

    train_dataset_path = (
        datasets[dataset]["train"]
        if dataset != "cs"
        else datasets[dataset][domain]["train"]
    )

    val_dataset_path = (
        datasets[dataset]["val"]
        if dataset != "cs"
        else datasets[dataset][domain]["val"]
    )

    args = Namespace(
        config_file=config_file,
        eval_only=True,
        num_gpus=num_gpus,
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        opts=[
            "OUTPUT_DIR",
            output_path,
            "TEST.SLIDING_WINDOW",
            "True",
            "MODEL.SEM_SEG_HEAD.POOLING_SIZES",
            "[1,1]",
            "MODEL.WEIGHTS",
            base_model_path,
        ],
        resume=True,
    )

    if get_cofing_only:
        return args
    else:
        from catseg.train_net import Trainer, setup

        data_loader = Trainer.build_test_loader(
            setup(args), f"{domain_name}_sem_seg_{split}"
        )

        evaluator = Trainer.build_evaluator(setup(args), f"{domain_name}_sem_seg_{split}")

        return args, evaluator, data_loader


def custom_domain_args(
    config_file,
    output_path,
    num_gpus=1,
    model_path="models/model_final.pth",
    dataset_path: str = None,
    seed=None,
):

    args = Namespace(
        config_file=config_file,
        eval_only=True,
        num_gpus=num_gpus,
        dataset_path=dataset_path,
        opts=[
            "OUTPUT_DIR",
            output_path,
            "TEST.SLIDING_WINDOW",
            "True",
            "MODEL.SEM_SEG_HEAD.POOLING_SIZES",
            "[1,1]",
            "MODEL.WEIGHTS",
            model_path,
        ],
        resume=True,
        model_path=model_path,
    )

    if seed != None:
        args.opts.extend(["SEED", seed])

    return args


def benchmark_catseg(model, args):

    import detectron2.utils.comm as comm
    from detectron2.evaluation import verify_results

    from catseg.train_net import Trainer, set_random_seed, setup

    cfg = setup(args)
    set_random_seed(cfg.SEED)
    res = Trainer.test(cfg, model)
    if cfg.TEST.AUG.ENABLED:
        res.update(Trainer.test_with_TTA(cfg, model))
    if comm.is_main_process():
        verify_results(cfg, res)
    return res

def get_device() -> str:

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    return device

def load_catseg_model(args, model_path: str = None):
    from catseg.train_net import Trainer, setup
    from detectron2.checkpoint import DetectionCheckpointer

    print("Loading base model ...")
    
    try:
        cfg = setup(args)
        cfg.defrost()
        cfg.MODEL.DEVICE = get_device() # Device is cuda by default so we need to overwrite
        cfg.freeze()
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS if model_path is None else model_path, resume=args.resume
        )
        print("Base model loaded.\n")
        return model
    except AttributeError as e:
        print(f"Error: Invalid model configuration: {e}")
        raise
    except FileNotFoundError:
        print(f"Error: Model weights not found at the specified path.")
        raise
    except Exception as e:
        print(f"Unexpected error while loading the base model: {e}")
        raise