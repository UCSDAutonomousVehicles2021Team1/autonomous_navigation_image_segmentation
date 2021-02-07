import detectron2

# import some common libraries
import os
import torch
import subprocess
import json

# import some common detectron2 utilities
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, \
    build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

#Import custom evaluator
from custom_trainer import COCOFormatTrainer

YAML_EXTENSION_SIZE = 5

def train_models(imagedir_name, train_path, train_annotations, val_path, \
    val_annotations, output_dir, config_files, metrics_dir):
    #Generate require metadata and load dataset into dictionary
    train_dataset_metadata, train_dataset_dicts, val_dataset_metadata, \
        val_dataset_dicts = setup_data(imagedir_name, train_path, \
            train_annotations, val_path, val_annotations)
    #Iterate through the model configs and store model names and metrics
    model_names = []
    for config in config_files:
        #Store model name
        model_name = config.split("/")[-1][:-YAML_EXTENSION_SIZE]
        #Setup the model config with the dataset names and device to train on
        #and output dir
        cfg = setup_config(config, output_dir, model_name)
        #Create trainer and start training
        trainer = COCOFormatTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        #Clean the metrics and dump into another file
        cleaned_metrics(cfg.OUTPUT_DIR, metrics_dir, model_name)
        model_names.append(model_name)
    #Returning model names to use when evaluating
    return model_names

def setup_data(imagedir_name, train_path, train_annotations, val_path, \
    val_annotations):
    #Register the train and validation datasets into a dictionary
    register_coco_instances("train_detector", {}, os.path.join(train_path, \
        train_annotations), os.path.join(train_path, imagedir_name))
    register_coco_instances("val_detector", {}, os.path.join(val_path, \
        val_annotations), os.path.join(val_path, imagedir_name))
    #Store train and validation metadata and dictionaries
    train_dataset_metadata = MetadataCatalog.get("train_detector")
    train_dataset_dicts = DatasetCatalog.get("train_detector")
    val_dataset_metadata = MetadataCatalog.get("val_detector")
    val_dataset_dicts = DatasetCatalog.get("val_detector")

    return train_dataset_metadata, train_dataset_dicts, val_dataset_metadata, \
        val_dataset_dicts

def setup_config(config, output_dir, model_name):
    cfg = get_cfg()
    #Load model parameters
    cfg.merge_from_file(config)
    #Set datasets
    cfg.DATASETS.TRAIN = ("train_detector",)
    cfg.DATASETS.TEST = ("val_detector",)
    #Create output directory to store results
    cfg.OUTPUT_DIR = os.path.join(output_dir, model_name)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)
    #Set config device
    if torch.cuda.is_available():
        print("Using CUDA")
        cfg.MODEL.DEVICE = 'cuda'
    else:
        print("Using CPU")
        cfg.MODEL.DEVICE = 'cpu'
    return cfg

def cleaned_metrics(output_dir, metrics_dir, model_name):
    #Open the unclean metrics file
    with open(os.path.join(output_dir, 'metrics.json'), 'r') as metric:
        #Read the lines
        metrics_file = metric.readlines()
        #Clean the metric file
        cleaned_metrics = [{metric.split(": ")[0]:float(metric.split(": ")[1] \
            ) for metric in metrics} for metrics in \
            [all_metrics.strip("\n").replace("{", "").replace("}","" \
            ).replace("\"", "").split(", ") for all_metrics in metrics_file]]
        #Make a new directory to store it
        os.makedirs(metrics_dir, exist_ok = True)
        #Dump into dictionary
        cleaned_metrics = {"metrics":cleaned_metrics}
        #Dump dictionary into json format
        with open(os.path.join(metrics_dir, model_name + \
            "_train_results.json"), "w") as output_file:
            json.dump(cleaned_metrics, output_file)
        #Remove uncleaned metrics so that we can generate anew
        subprocess.call(["rm", "-r", "-f", os.path.join(output_dir, \
            'metrics.json')])
        subprocess.call(["rm", "-r", "-f", os.path.join(output_dir, \
            'inference/')])