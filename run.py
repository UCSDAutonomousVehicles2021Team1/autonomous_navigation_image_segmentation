import sys
import json
import os
import subprocess

sys.path.insert(0, 'src')
from etl import move_data
from eda import main_eda
from utils import convert_notebook
from train import train_models
from evaluate import find_best_model
from inference import best_inference



def main(targets):

    data_config = json.load(open('config/data-params.json'))
    eda_config = json.load(open('config/eda-params.json'))
    training_config = json.load(open('config/training-params.json'))
    evaluate_config = json.load(open('config/evaluate-params.json'))
    inference_config = json.load(open('config/inference-params.json'))
    test_config = json.load(open('config/test-params.json'))

    repo_files = ['notebooks', '.git', 'src', '.gitignore', 'test', 'run.py', \
    'config', 'Dockerfile']

    if 'data' in targets:
        move_data(**data_config)

    if 'eda' in targets:
        main_eda(**eda_config)
#         execute notebook / convert to html
        convert_notebook(**eda_config)

    if 'train' in targets:
        model_names = train_models(**training_config)

    if 'evaluate' in targets:
        best_model_name = find_best_model(model_names, **evaluate_config)

    if 'inference' in targets:
        best_inference(best_model_name, **inference_config)
 
    if 'test' in targets:
        move_data(**test_config)
        main_eda(**eda_config)
        convert_notebook(**eda_config)
        model_names = train_models(**training_config)
        best_model_name = find_best_model(model_names, **evaluate_config)
        print("Found best model: {}".format(best_model_name))
        best_inference(best_model_name, **inference_config)

    if 'clean' in targets:
        for file in os.listdir():
            if not file in repo_files:
                subprocess.call(["rm", "-r", "-f", file])

    if 'all' in targets:
        move_data(**data_config)
        main_eda(**eda_config)
        main_eda(**eda_config)
        convert_notebook(**eda_config)
        model_names = train_models(**training_config)
        best_model_name = find_best_model(model_names, **evaluate_config)
        print("Found best model: {}".format(best_model_name))
        best_inference(best_model_name, **inference_config)

if __name__ == '__main__':

    targets = sys.argv[1:]
    main(targets)
