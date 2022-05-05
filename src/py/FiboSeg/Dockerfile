
FROM python:3.9-bullseye

RUN apt-get update && \
	apt-get install -y git \
	wget\
	-y unzip

RUN apt install -y libgl1-mesa-glx


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

RUN pip install --no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install -r /app/data/requirements.txt
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html

WORKDIR /app/py/FiboSeg/