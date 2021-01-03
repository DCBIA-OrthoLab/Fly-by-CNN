FROM tensorflow/tensorflow:2.4.0
WORKDIR /app

RUN apt-get update && apt-get install -y \
	git \
	libx11-6 \
	libgl1

RUN pip install itk vtk pandas
RUN git clone https://github.com/DCBIA-OrthoLab/fly-by-cnn.git