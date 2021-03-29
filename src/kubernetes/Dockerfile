FROM tensorflow/tensorflow:2.4.1-gpu
WORKDIR /app

RUN apt-get update && apt-get install -y \
	git \
	libx11-6 \
	libgl1 \
	libopengl0 \
	libegl1

COPY vtk8.2.0.zip /usr/local/lib/python3.6/dist-packages/

WORKDIR /usr/local/lib/python3.6/dist-packages/
RUN unzip vtk8.2.0.zip

RUN pip install itk sklearn pandas matplotlib

WORKDIR /app
git clone --depth 1 --branch 2.1  https://github.com/DCBIA-OrthoLab/fly-by-cnn.git

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.6/dist-packages/vtkmodules/
ENV MESA_GL_VERSION_OVERRIDE=3.2