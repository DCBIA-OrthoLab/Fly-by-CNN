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


ARG 
WORKDIR /app
RUN wget https://github.com/MathieuLeclercq/fly-by-cnn/archive/refs/tags/3.2.zip
RUN unzip 3.2.zip
# RUN unzip AMASSS_CBCT-$RELEASE.zip
# RUN unzip AMASSS_CBCT-1.0.0-alpha.zip
RUN mv AMASSS_CBCT-$RELEASE_VERSION/MULTI_SEG MULTI_SEG
# RUN mv /app/3.2/src/py/FiboSeg FiboSeg
RUN rm -rf 3.2.zip \
	rm -rf 3.2

RUN pip install -r MULTI_SEG/requirements.txt


RUN mkdir scans


