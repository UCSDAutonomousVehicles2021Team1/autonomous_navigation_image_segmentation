import os
import cv2
import subprocess
import torch
import json
import numpy as np
import ffmpeg

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, \
    build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from custom_trainer import COCOFormatTrainer

def best_inference(best_model_name, config_dir, model_dir, test_path_images, \
    test_annotations, confidence_threshold, result_metrics_dir, \
    video_img_dir, video, framerate):
    #Storing test data
    test_dataset_metadata, test_dataset_dicts = setup_data(test_path_images, \
        test_annotations)
    #Creating config file
    cfg = setup_config(config_dir, model_dir, best_model_name, \
        confidence_threshold)
    #Creating predictor
    predictor = DefaultPredictor(cfg)
    #Creating evaluator
    evaluator = COCOEvaluator("test_detector", \
        distributed = False, output_dir=os.path.join(model_dir, "final_test"))
    #Building test loader
    test_loader = build_detection_test_loader(cfg, "test_detector")
    #Loading in train
    trainer = COCOFormatTrainer(cfg)
    #Getting inference results on trainer
    test_results = inference_on_dataset(trainer.model, test_loader, evaluator)
    #Dumping into json file
    with open(os.path.join(result_metrics_dir, 'test_results.json'), 'w') as \
        outfile:
        json.dump(dict(test_results), outfile)
    #Generating video from predictions
    create_video(video_img_dir, video, test_loader, \
        test_dataset_metadata, test_dataset_dicts, predictor, framerate)

def setup_data(test_path_images, test_annotations):
    #Register the test datasets into a dictionary
    register_coco_instances("test_detector", {}, test_annotations, \
        test_path_images)
    #Store test meta data and dictionaries
    test_dataset_metadata = MetadataCatalog.get("test_detector")
    test_dataset_dicts = DatasetCatalog.get("test_detector")
    return test_dataset_metadata, test_dataset_dicts

def setup_config(config_dir, model_dir, model_name, confidence_threshold):
    cfg = get_cfg()
    #Load model parameters
    cfg.merge_from_file(os.path.join(config_dir, model_name + '.yaml'))
    #Set datasets
    cfg.DATASETS.TRAIN = ("test_detector",)
    cfg.DATASETS.TEST = ()
    #Create output directory to store results
    cfg.OUTPUT_DIR = os.path.join(model_dir, "final_test")
    #Generating metric saving directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)
    #Loading weights
    cfg.MODEL.WEIGHTS = os.path.join(model_dir + model_name + "/", \
    "model_final.pth")
    #Setting confidence threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    #Set config device
    if torch.cuda.is_available():
        print("Using CUDA")
        cfg.MODEL.DEVICE = 'cuda'
    else:
        print("Using CPU")
        cfg.MODEL.DEVICE = 'cpu'
    return cfg

def create_video(video_img_dir, video, test_loader, \
    test_dataset_metadata, test_dataset_dicts, predictor, framerate):
    os.makedirs(video_img_dir, exist_ok = True)
    #Iterating through images in dictionary
    for i, d in enumerate(test_dataset_dicts):    
        im = cv2.imread(d["file_name"])
        #Predicting bbox and segmentation
        outputs = predictor(im)
        #Generating visualizer for image
        v = Visualizer(im[:, :, ::-1],
                       metadata=test_dataset_metadata, 
                       scale=1, 
                       instance_mode=ColorMode.IMAGE_BW
        )
        #Drawing over the visualizer image
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #Saving image to directory to generate video later
        cv2.imwrite(os.path.join(video_img_dir, "image{}.jpg".format(i + 1)), \
            v.get_image()[:, :, ::-1].astype(np.float))
    #Generate video via ffmpeg
    os.makedirs("/".join(video.split("/")[:-1]), exist_ok = True)
    ffmpeg.input('{}*.jpg'.format(video_img_dir), pattern_type='glob', \
        framerate=framerate).output(video).run()