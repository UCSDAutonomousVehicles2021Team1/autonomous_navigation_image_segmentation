# autonomous_navigation_image_segmentation

Image segmentation pipeline for Autonomous Navigation. Allows you to train several models on the Detectron 2 Network, retrieve the model that performed best in certain metrics (specified by user) and use that for inference

**Table of contents**
- [What does it do](#what-does-it-do)
- [Targets](#targets)
- [Usage Instructions](#usage-instructions)
- [References](#references)

## What does it do

It allows for the use of training multiple models (under the Detectron2 network) and evaluating them with the help of user-specified metrics. 

## Targets

1. data:

  This target allows you to move data into the main repository. For formatting instructions go to test/testdata and see how the data inputted should be arranged
  
2. eda:

  This target runs an eda on the data. Currently it's setup to run on test data
  
3. train:

  This target trains all the models provided in the configs/model_configs folder. The train and validation metrics are outputted into a folder of your choice. These metrics have been cleaned up and dumped in .json format for easy loading
  
4. evaluate:

  This target evaluates all the validation metrics based off an order you specify in evaluate-params.json under config. The model that performs best is then returned under a folder of your choice in the form of a .txt
  
5. inference:

  This target uses the best model found earlier to run an inference on labelled test data. It provides metrics once again as well as a video output of all the predictions made.
  
6. test:

  This target runs all previous targets on test data and is mainly used to ensure the repository is still working as intended as well as give a demo of what the current targets look like
  
7. all:

  This target works similarly to the test data except it runs on the data inputs you specified in data-params.json under config
  
8. clean:

  This target cleans the repository to bring it back to the original state. WARNING: It will delete model weights as well as the model configuration that performed best so please save results appropriately

Currently we don't support running train, evaluate or inference individually. Please use them all together

Example call: ```python run.py data eda train evaluate inference```


## Usage Instructions

1. Clone this repository
   ```
   git clone https://github.com/sisaha9/autonomous_navigation_image_segmentation.git
   ```
   Once cloned, switch directories to inside this repository

2. Build the Docker image
   ```
   docker build -t ai_nav
   docker run --rm -it ai_nav /bin/bash.
   ```

3. Modify target parameters by going to config/. Add your own model configurations in .yaml format to config/model_configs/. Refer to the API to see how to change parameters there: https://detectron2.readthedocs.io/en/latest/tutorials/configs.html

4. Once you have made all the changes to the configs (you really only need to change the model configurations and data inputs) run the following command
   ```: https://github.com/facebookresearch/detectron2
   python run.py all
   ```
   If you want to see a test run
    ```
    python run.py test
    ```
5. Once you are done. Copy the model weights of the best model and any results you want to save outside the container. Refer to this StackOverflow thread if you are unsure: https://stackoverflow.com/questions/22049212/docker-copying-files-from-docker-container-to-host?rq=1. Once done you can exit and use the model weights for your own inference

## References

- This work heavily relies on FAIR's Detectron2 Network: https://github.com/facebookresearch/detectron2

- The test data is labelled using MakeSense in the COCO JSON format: https://github.com/SkalskiP/make-sense 

- utils.py and creation of notebook for EDA taken from Aaron Fraenkel: https://github.com/afraenkel
