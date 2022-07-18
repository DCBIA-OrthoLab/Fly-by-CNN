FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt update && \
	apt install -y git \
	wget \
	unzip \
	libgl1-mesa-glx \
	libxrender1


RUN mkdir /app
RUN mkdir data
WORKDIR /app/data

ARG RELEASE_VERSION="3.6"

RUN wget https://github.com/MathieuLeclercq/fly-by-cnn/releases/download/$RELEASE_VERSION/requirements.txt

WORKDIR /app
RUN wget https://github.com/MathieuLeclercq/fly-by-cnn/archive/refs/tags/$RELEASE_VERSION.zip
RUN unzip $RELEASE_VERSION.zip

RUN mv fly-by-cnn-$RELEASE_VERSION/src/py/ py
RUN rm -rf $RELEASE_VERSION.zip \
	rm -rf $RELEASE_VERSION

RUN rm -rf  fly-by-cnn-$RELEASE_VERSION

RUN pip install -r /app/data/requirements.txt
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html

WORKDIR /app/py/FiboSeg/