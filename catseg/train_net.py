# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import copy
import itertools
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Set
import glob
from PIL import Image
import weakref
import sys

import numpy as np
import torch
import random

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.utils.comm import all_gather, is_main_process, synchronize

from detectron2.config import get_cfg

from detectron2.data import (
    MetadataCatalog,
    build_detection_train_loader,
    build_detection_test_loader,
)
import detectron2.data.transforms as T

from detectron2.checkpoint import DetectionCheckpointer

from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results,
)
from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator

from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer

from detectron2.solver.build import maybe_add_gradient_clipping

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.utils.file_io import PathManager

# Add the root of the project to the python path to find cat_seg
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))  # Adds current directory

# MaskFormer
from cat_seg import (
    DETRPanopticDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    ResizedMaskFormerSemanticDatasetMapper,
    ResizedMaskFormerSemanticTestDatasetMapper,
    add_cat_seg_config,
    add_lora_config,
)

import cv2
import peft
import json
import wandb 


def set_random_seed(seed=0, deterministic=True):
    # This function is from https://github.com/MarcBotet/hamlet/blob/4c9b8db6712b2e4680f40ac4d8b3f653cd77d641/mmseg/apis/train.py#L24
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """  # noqa
    if comm.get_world_size() == 1:
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
    ddp = torch.nn.DistributedDataParallel(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp

class SemSegEvaluator(SemSegEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            b_conf_matrix_list = all_gather(self._b_conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

            self._b_conf_matrix = np.zeros_like(self._b_conf_matrix)
            for b_conf_matrix in b_conf_matrix_list:
                self._b_conf_matrix += b_conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=float)
        iou = np.full(self._num_classes, np.nan, dtype=float)
        tp = self._conf_matrix.diagonal()[:-1].astype(float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        if self._compute_boundary_iou:
            b_iou = np.full(self._num_classes, np.nan, dtype=float)
            b_tp = self._b_conf_matrix.diagonal()[:-1].astype(float)
            b_pos_gt = np.sum(self._b_conf_matrix[:-1, :-1], axis=0).astype(float)
            b_pos_pred = np.sum(self._b_conf_matrix[:-1, :-1], axis=1).astype(float)
            b_union = b_pos_gt + b_pos_pred - b_tp
            b_iou_valid = b_union > 0
            b_iou[b_iou_valid] = b_tp[b_iou_valid] / b_union[b_iou_valid]

        res = {}
        res["mIoU"] = 100 * miou
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results

class VOCbEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int)
            pred[pred >= 20] = 20
            with PathManager.open(
                self.input_file_to_gt_file[input["file_name"]], "rb"
            ) as f:
                gt = np.array(Image.open(f), dtype=np.int)

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

class CityscapesSemSegEvaluator(CityscapesSemSegEvaluator):

        def evaluate_image(self):
            comm.synchronize()
            if comm.get_rank() > 0:
                return
            # Load the Cityscapes eval script *after* setting the required env var,
            # since the script reads CITYSCAPES_DATASET into global variables at load time.
            import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval

            self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

            # set some global states in cityscapes evaluation API, before evaluating
            cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
            cityscapes_eval.args.predictionWalk = None
            cityscapes_eval.args.JSONOutput = False
            cityscapes_eval.args.colorized = False

            # These lines are adopted from
            # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py # noqa
            gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
            groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_labelIds.png"))
            assert len(
                groundTruthImgList
            ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
                cityscapes_eval.args.groundTruthSearch
            )
            predictionImgList = []
            for gt in groundTruthImgList:
                predictionImgList.append(cityscapes_eval.getPrediction(cityscapes_eval.args, gt))
            results = cityscapes_eval.evaluateImgLists(
                predictionImgList, groundTruthImgList, cityscapes_eval.args
            )
            ret = OrderedDict()
            for pred_path, gt_path in zip(predictionImgList, groundTruthImgList):
                results = cityscapes_eval.evaluateImgLists(
                    [pred_path], [gt_path], cityscapes_eval.args
                )
                ret.update({str(pred_path): {
                    "IoU": 100.0 * results["averageScoreClasses"],
                    "iIoU": 100.0 * results["averageScoreInstClasses"],
                    "IoU_sup": 100.0 * results["averageScoreCategories"],
                    "iIoU_sup": 100.0 * results["averageScoreInstCategories"],
                }})

            self._working_dir.cleanup()
            return ret

class ACDCSemSegEvaluator(CityscapesEvaluator):
    """
    Evaluate semantic segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """

    def process(self, inputs, outputs):
        from cityscapesscripts.helpers.labels import trainId2label

        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_filename = os.path.join(self._temp_dir, basename + "_pred.png")

            output_pred = output["sem_seg"].argmax(dim=0).to(self._cpu_device).numpy()
            # output_pred = cv2.resize(
            #     output_pred, (1920, 1080), interpolation=cv2.INTER_NEAREST_EXACT
            # )
            pred = 255 * np.ones(output_pred.shape, dtype=np.uint8)
            for train_id, label in trainId2label.items():
                if label.ignoreInEval:
                    continue
                pred[output_pred == train_id] = label.id
            Image.fromarray(pred).save(pred_filename)
    
    def process_image(self, inputs, outputs):
        from cityscapesscripts.helpers.labels import trainId2label

        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_filename = os.path.join(self._temp_dir, basename + "_pred.png")

            output_pred = output["sem_seg"].argmax(dim=0).to(self._cpu_device).numpy()
            # output_pred = cv2.resize(
            #     output_pred, (1920, 1080), interpolation=cv2.INTER_NEAREST_EXACT
            # )
            pred = 255 * np.ones(output_pred.shape, dtype=np.uint8)
            for train_id, label in trainId2label.items():
                if label.ignoreInEval:
                    continue
                pred[output_pred == train_id] = label.id
            Image.fromarray(pred).save(pred_filename)
        
        return pred_filename

    def evaluate(self):
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.
        import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False
        # ACDC does not provide instance-level annotations
        cityscapes_eval.args.evalInstLevelScore = False

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
        groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gt_labelIds.png"))
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(
                cityscapes_eval.getPrediction(cityscapes_eval.args, gt)
            )
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )
        ret = OrderedDict()
        ret["sem_seg"] = {
            "IoU": 100.0 * results["averageScoreClasses"],
            "IoU_sup": 100.0 * results["averageScoreCategories"],
        }
        self._working_dir.cleanup()
        return ret
    
    def evaluate_image(self):
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.
        import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))
        print("Evaluating results under {} ...".format(self._temp_dir))
        
        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False
        cityscapes_eval.args.quiet = True
        cityscapes_eval.args.evalInstLevelScore = False

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py # noqa
        
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
        groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gt_labelIds.png"))
        
        print("Ground truth: ", groundTruthImgList)
    
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(
                cityscapes_eval.getPrediction(cityscapes_eval.args, gt)
            )
            
        ret = OrderedDict()
        for pred_path, gt_path in zip(predictionImgList, groundTruthImgList):
            results = cityscapes_eval.evaluateImgLists(
                [pred_path], [gt_path], cityscapes_eval.args
            )
            ret.update({str(pred_path): {
                "IoU": 100.0 * results["averageScoreClasses"],
                "iIoU": 100.0 * results["averageScoreInstClasses"],
                "IoU_sup": 100.0 * results["averageScoreCategories"],
                "iIoU_sup": 100.0 * results["averageScoreInstCategories"],
            }})

        self._working_dir.cleanup()
        return ret


class MUSESSemSegEvaluator(ACDCSemSegEvaluator):
    def evaluate(self):
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.
        import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False
        # ACDC does not provide instance-level annotations
        cityscapes_eval.args.evalInstLevelScore = False

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
        groundTruthImgList = glob.glob(os.path.join(gt_dir, "*_gt_labelIds.png"))
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(
                cityscapes_eval.getPrediction(cityscapes_eval.args, gt)
            )
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )
        ret = OrderedDict()
        ret["sem_seg"] = {
            "IoU": 100.0 * results["averageScoreClasses"],
            "IoU_sup": 100.0 * results["averageScoreCategories"],
        }
        self._working_dir.cleanup()
        return ret
    
class ResizedCityscapesSemSegEvaluator(CityscapesSemSegEvaluator):
    def process(self, inputs, outputs):
        from cityscapesscripts.helpers.labels import trainId2label

        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_filename = os.path.join(self._temp_dir, basename + "_pred.png")

            output_pred = output["sem_seg"].argmax(dim=0).cpu().numpy()
            # Resize prediction to the desired resolution using cv2
            output_pred = cv2.resize(
                output_pred, (2048, 1024), interpolation=cv2.INTER_NEAREST_EXACT
            )
            pred = 255 * np.ones(output_pred.shape, dtype=np.uint8)
            for train_id, label in trainId2label.items():
                if label.ignoreInEval:
                    continue
                pred[output_pred == train_id] = label.id

            # Convert to PIL Image to save
            pred = Image.fromarray(pred)
            # Save the resized prediction
            pred.save(pred_filename)

    def process_image(self, inputs, outputs):
        from cityscapesscripts.helpers.labels import trainId2label

        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_filename = os.path.join(self._temp_dir, basename + "_pred.png")

            output_pred = output["sem_seg"].argmax(dim=0).cpu().numpy()
            # Resize prediction to the desired resolution using cv2
            output_pred = cv2.resize(
                output_pred, (2048, 1024), interpolation=cv2.INTER_NEAREST_EXACT
            )
            pred = 255 * np.ones(output_pred.shape, dtype=np.uint8)
            for train_id, label in trainId2label.items():
                if label.ignoreInEval:
                    continue
                pred[output_pred == train_id] = label.id

            # Convert to PIL Image to save
            pred = Image.fromarray(pred)
            # Save the resized prediction
            pred.save(pred_filename)
            
        return pred_filename
            
    def evaluate(self):
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.
        import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False
        cityscapes_eval.args.quiet = False

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
        groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_labelIds.png"))
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(cityscapes_eval.getPrediction(cityscapes_eval.args, gt))
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )
        ret = OrderedDict()
        ret["sem_seg"] = {
            "IoU": 100.0 * results["averageScoreClasses"],
            "iIoU": 100.0 * results["averageScoreInstClasses"],
            "IoU_sup": 100.0 * results["averageScoreCategories"],
            "iIoU_sup": 100.0 * results["averageScoreInstCategories"],
        }
        self._working_dir.cleanup()
        return ret

    def evaluate_image(self):
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.
        import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
        groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_labelIds.png"))
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(cityscapes_eval.getPrediction(cityscapes_eval.args, gt))
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )
        ret = OrderedDict()
        for pred_path, gt_path in zip(predictionImgList, groundTruthImgList):
            results = cityscapes_eval.evaluateImgLists(
                [pred_path], [gt_path], cityscapes_eval.args
            )
            ret.update({str(pred_path): {
                "IoU": 100.0 * results["averageScoreClasses"],
                "iIoU": 100.0 * results["averageScoreInstClasses"],
                "IoU_sup": 100.0 * results["averageScoreCategories"],
                "iIoU_sup": 100.0 * results["averageScoreInstCategories"],
            }})

        self._working_dir.cleanup()
        return ret


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    def reset_trainer(self, cfg, model):
        """
        Resets the trainer attributes after assigning new model
        """
        self.model = model
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.data_loader = self.build_train_loader(cfg)

        self.model = create_ddp_model(
            self.model, broadcast_buffers=False, find_unused_parameters=True
        )
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            self.model, self.data_loader, self.optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)

        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )

        self.cfg = cfg

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type == "sem_seg_background":
            evaluator_list.append(
                VOCbEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
        ]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "resized_cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return ResizedCityscapesSemSegEvaluator(dataset_name)  # Rainy Cityscapes
        if evaluator_type == "cityscapes_panoptic_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
        if evaluator_type == "acdc_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "ACDCEvaluator currently do not work with multiple machines."
            return ACDCSemSegEvaluator(dataset_name)
        if evaluator_type == "muses_sem_seg": 
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "ACDCEvaluator currently do not work with multiple machines."
            return MUSESSemSegEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(
                cfg,
                True,
            )
        # Resized semantic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "resized_mask_former_semantic":
            mapper = ResizedMaskFormerSemanticDatasetMapper(
                cfg,
                True,
            )
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
        # DETR-style dataset mapper for COCO panoptic segmentation
        elif cfg.INPUT.DATASET_MAPPER_NAME == "detr_panoptic":
            mapper = DETRPanopticDatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # Reszied semantic segmentation test dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "resized_mask_former_semantic":
            mapper = ResizedMaskFormerSemanticTestDatasetMapper(cfg, False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        else:
            return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        # import ipdb;
        # ipdb.set_trace()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = (
                        hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    )
                if "clip_model" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.CLIP_MULTIPLIER
                # for deformable detr

                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def add_lora(cfg, model):
    logger = logging.getLogger("detectron2.trainer")
    config = peft.LoraConfig(
        r=cfg.MODEL.LORA.RANK,
        lora_alpha=cfg.MODEL.LORA.ALPHA,
        lora_dropout=cfg.MODEL.LORA.DROPOUT,
        target_modules=cfg.MODEL.LORA.MODUELS,
        bias=cfg.MODEL.LORA.BIAS,
        use_rslora=cfg.MODEL.LORA.USE_RSLORA,
        use_dora=cfg.MODEL.LORA.USE_DORA,
    )
    peft_model = peft.get_peft_model(model, config, adapter_name=cfg.MODEL.LORA.NAME)
    
    logger.info("LoRAs injected for training.")

    return peft_model


def load_lora(cfg, model): # Loads the lora
    logger = logging.getLogger("detectron2.trainer")
    model = peft.PeftModel.from_pretrained(
        model, cfg.MODEL.LORA.DB_PATH + cfg.MODEL.LORA.NAME, cfg.MODEL.LORA.NAME
    ) # DB_PATH where loras are saved (parameters)
    # model wrapped -> trained Lora
    logger.info("Pre-trained LoRA loaded.")
    return


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg() # Detectron
    # for poly lr schedule
    add_deeplab_config(cfg) # 
    add_cat_seg_config(cfg) #
    add_lora_config(cfg)    # 
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former"
    )

    return cfg


def main(args):

    cfg = setup(args)
    set_random_seed(cfg.SEED)
    wandb.init(mode="offline", sync_tensorboard=True, name= cfg.MODEL.LORA.NAME)
    torch.set_float32_matmul_precision("high")

    if args.eval_only:
        model = Trainer.build_model(cfg)

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        # Load lora from lora db and attach to base model
        if cfg.MODEL.LORA.ENABLED:
            load_lora(cfg, model)

        res = Trainer.test(cfg, model)
        
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        print("And the results is: ...")
        print(res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    # Add lora to the base model, reset the necessary model confgis/parameters to train lora
    if cfg.MODEL.LORA.ENABLED:
        peft_model = add_lora(cfg, trainer.model)
        trainer.reset_trainer(cfg, peft_model)
        # Attaching LoRAs changes the modules to which hooks are set, we need to reset
        trainer.model.base_model.model.reset_forward_hooks()
        trainer.model.print_trainable_parameters()

    output = trainer.train()

    # Save only the LoRA weights to LoRA DB
    if cfg.MODEL.LORA.ENABLED == True:
        trainer.model.save_pretrained(cfg.MODEL.LORA.DB_PATH)

    return output

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )