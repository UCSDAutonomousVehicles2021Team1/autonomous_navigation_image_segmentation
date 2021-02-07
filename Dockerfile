FROM ucsdets/datascience-notebook:2020.2-stable

USER root


# Install python3, pip3
RUN apt-get update --fix-missing && \
    apt-get install -y git \
                       build-essential \
                       cmake \
                       vim \
                       wget \
                       unzip \
                       ffmpeg

RUN conda install cudatoolkit=10.1 \
				  cudnn \
				  nccl \
				  -y

# Upgrade pip
RUN pip install --upgrade pip

RUN pip install --no-cache-dir numpy \
                               scipy \
                               pandas \
                               pyyaml \
                               notebook \
                               matplotlib \
                               seaborn \
                               tensorboard cmake \
                               opencv-python \
                               ffmpeg-python
			       
RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

USER jovyan

RUN ln -s /usr/local/nvidia/bin/nvidia-smi /opt/conda/bin/nvidia-smi

USER $NB_UID:$NB_GID
ENV PATH=${PATH}:/usr/local/nvidia/bin

RUN pip install 'git+https://github.com/facebookresearch/fvcore'
RUN git clone https://github.com/sisaha9/detectron2 detectron2_repo
RUN pip install -e detectron2_repo
