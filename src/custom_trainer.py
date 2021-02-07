from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
import os

#Custom Trainer to implement COCO Evaluator
class COCOFormatTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        return COCOEvaluator(dataset_name, distributed = False, \
            output_dir = output_folder)