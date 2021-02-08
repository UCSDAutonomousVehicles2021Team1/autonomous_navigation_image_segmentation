# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import torch


# import some common detectron2 utilities
import detectron2
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


def main_eda(files, outdir, **kwargs):
    os.makedirs(outdir, exist_ok = True)
    register_coco_instances("train_lane_cone_detector", {}, files[0], files[1])
    train_dataset_metadata = MetadataCatalog.get("train_lane_cone_detector")
    train_dataset_dicts = DatasetCatalog.get("train_lane_cone_detector")
    plot_raw_image(train_dataset_metadata, train_dataset_dicts, outdir)
    plot_labelled_image(train_dataset_metadata, train_dataset_dicts, outdir)

def plot_raw_image(train_dataset_metadata, train_dataset_dicts, outdir):
    img = cv2.imread(train_dataset_dicts[0]["file_name"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float)
    fig = plt.figure(figsize = (20,5))
    ax = fig.add_subplot(111)
    ax.imshow(img/255, interpolation='none')
    ax.set_title('Training Image Raw')
    ax.axis("off")
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, 'raw_image.png'))
    plt.close()

def plot_labelled_image(train_dataset_metadata, train_dataset_dicts, outdir):
    img = cv2.imread(train_dataset_dicts[0]["file_name"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float)
    visualizer = Visualizer(img[:, :, ::-1], metadata=train_dataset_metadata, scale=1)
    vis = visualizer.draw_dataset_dict(train_dataset_dicts[0])
    fig = plt.figure(figsize = (20,5))
    ax = fig.add_subplot(111)
    ax.imshow(vis.get_image()[:, :, ::-1]/255, interpolation='none')
    ax.set_title('Training Image with Labels')
    ax.axis("off")
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, 'labelled_image.png'))
    plt.close()
    
