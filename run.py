import sys
import json
import os

sys.path.insert(0, 'src')
from etl import move_data
# from eda import main_eda
# from utils import convert_notebook
# from tuning import find_metrics
# from generate import create_launch_files



def main(targets):

    # data_config = json.load(open('config/data-params.json'))
    # eda_config = json.load(open('config/eda-params.json'))
    # tuning_config = json.load(open('config/tuning-params.json'))
    # generate_config = json.load(open('config/generate-params.json'))
    test_config = json.load(open('config/test-params.json'))

    # if 'data' in targets:
    #     move_data(**data_config)

#     if 'eda' in targets:
#         main_eda(**eda_config)
        
# #         execute notebook / convert to html
#         convert_notebook(**eda_config)
    
#     if 'tune' in targets:
#         find_metrics(**tuning_config)
        
#     if 'generate' in targets:
#         create_launch_files(**generate_config)

    if 'test' in targets:
        move_data(**test_config)
#         main_eda(**eda_config)
#         convert_notebook(**eda_config)
#         find_metrics(**tuning_config)
#         create_launch_files(**generate_config)
        
#     if 'all' in targets:
#         move_data(**data_config)
#         main_eda(**eda_config)
#         convert_notebook(**eda_config)
#         find_metrics(**tuning_config)
#         create_launch_files(**generate_config)

if __name__ == '__main__':

    targets = sys.argv[1:]
    main(targets)
