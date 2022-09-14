FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt update && \
	apt install -y git \
	wget \
	unzip \
	libgl1-mesa-glx \
	libxrender1

RUN mkdir /app
RUN mkdir /app/data
RUN mkdir /app/models

WORKDIR /app

COPY requirements.txt /app/data/requirements.txt

RUN git clone https://github.com/DCBIA-OrthoLab/fly-by-cnn.git

RUN pip install -r /app/data/requirements.txt
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html

RUN wget https://github.com/DCBIA-OrthoLab/fly-by-cnn/releases/download/3.0/07-21-22_val-loss0.169.pth -O /app/models/model.pth
RUN wget https://github.com/DCBIA-OrthoLab/fly-by-cnn/releases/download/3.0/dental_models.zip -O /app/data/dental_models.zip
RUN unzip /app/data/dental_models.zip -d /app/data/

RUN chmod +x /app/fly-by-cnn/src/py/dental_model_seg.py

ENV PATH="${PATH}:/app/fly-by-cnn/src/py/"