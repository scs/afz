FROM nvcr.io/nvidia/l4t-ml:r32.4.3-py3

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install libgfortran3 libgtk2.0-dev pkg-config libcanberra-gtk-module libcanberra-gtk3-module -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install scikit-build
RUN pip3 install cmake opencv-python Keras==2.3.1 scikit-learn==0.22.2.post1

WORKDIR /root/afz
COPY . /root/afz

RUN mkdir output

ENTRYPOINT ["python3", "main_camera.py"]
