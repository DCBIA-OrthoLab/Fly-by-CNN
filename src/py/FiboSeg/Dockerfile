#FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
FROM python:3.9-bullseye

RUN apt-get update && \
	apt-get install -y git \
	# apt-get install -y nvidia-container-toolkit \
	# libx11-6 \
	# libgl1 \
	# libopengl0 \
	# libegl1 \
	wget\
	-y unzip

RUN apt install -y libgl1-mesa-glx

#ARG DEBIAN_FRONTEND=noninteractive
#RUN apt install -y python3.9
#RUN export alias python=python3.9
#RUN apt install -y pip

RUN mkdir /app
RUN mkdir data
WORKDIR /app/data


#RUN wget https://github.com/MathieuLeclercq/fly-by-cnn/releases/download/3.2/examples_scans.zip
#RUN unzip examples_scans.zip
#RUN rm -rf examples_scans.zip

RUN wget https://github.com/MathieuLeclercq/fly-by-cnn/releases/download/3.2/requirements.txt

ARG RELEASE_VERSION="3.2"
WORKDIR /app
RUN wget https://github.com/MathieuLeclercq/fly-by-cnn/archive/refs/tags/$RELEASE_VERSION.zip
RUN unzip $RELEASE_VERSION.zip

RUN mv fly-by-cnn-$RELEASE_VERSION/src/py/ py
RUN rm -rf $RELEASE_VERSION.zip \
	rm -rf $RELEASE_VERSION

RUN rm -rf  fly-by-cnn-$RELEASE_VERSION

RUN pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r /app/data/requirements.txt
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu111_pyt1101/download.html

#RUN echo 'cd py/FiboSeg' >> ~/.bashrc
WORKDIR /app/py/FiboSeg/