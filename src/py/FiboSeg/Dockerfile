FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

RUN apt-get update && \
	apt-get install -y git \
	# apt-get install -y nvidia-container-toolkit \
	# libx11-6 \
	# libgl1 \
	# libopengl0 \
	# libegl1 \
	wget\
	-y unzip

RUN mkdir /app
RUN mkdir data
WORKDIR /app/data


RUN wget https://github.com/MathieuLeclercq/fly-by-cnn/releases/download/3.2/dataset_Fiboseg.zip
RUN unzip dataset_Fiboseg.zip
RUN rm -rf dataset_Fiboseg.zip


ARG RELEASE_VERSION="3.2"
WORKDIR /app
RUN wget https://github.com/MathieuLeclercq/fly-by-cnn/archive/refs/tags/$RELEASE_VERSION.zip
RUN unzip $RELEASE_VERSION.zip

RUN mv fly-by-cnn-$RELEASE_VERSION/src/py/FiboSeg FiboSeg
RUN rm -rf $RELEASE_VERSION.zip \
	rm -rf $RELEASE_VERSION

#RUN pip install -r MULTI_SEG/requirements.txt


#RUN mkdir scans


