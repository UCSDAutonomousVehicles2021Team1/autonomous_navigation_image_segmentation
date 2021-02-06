import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import torch
import subprocess


# import some common detectron2 utilities
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
%matplotlib inline

class COCOFormatTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        return COCOEvaluator(dataset_name, distributed = False, output_dir = output_folder)


def train_model(train_path, val_path, test_path):
	register_coco_instances("train_detector", {}, "./f1tenth_testdata/train/train.json", "./f1tenth_testdata/train/")
	register_coco_instances("val_detector", {}, "./f1tenth_testdata/validation/validation.json", "./f1tenth_testdata/validation/")
	register_coco_instances("test_detector", {}, "./f1tenth_testdata/test/test.json", "./f1tenth_testdata/test/")