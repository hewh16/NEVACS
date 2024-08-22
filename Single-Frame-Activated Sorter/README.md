# Single-Frame-Activated Sorter

The project aims to provide image-activated sorting benchmark for NEVACS. The event stream from the event camera is received and preprocessed, and the processed event image is then inferred by the detection model and the multi-object tracking. The processing result based on a single frame is selected to output the particle classification result, which would be sent to the sorter using the serial port.

## System requirements

### Hardware requirements

The single-frame-activated sorter requires a standard personal computer with a GPU (Nvidia RTX3080), a Prophesee EVK4 event camera mounted via USB3.0, and a USB3.0/TTL converter.

### OS requirements

The single-frame-activated sorter is supported for Linux and has been tested on the following system:
Linux: Ubuntu 20.04

### Software dependencies

The single-frame-activated sorter requires and has been tested on OpenCV 3.3.0, OpenEB 3.1.0, Python 3.8, PyTorch 1.9.0, TensorRT 8.2.3

## Build and install

Download the source_code.zip, and unzip the file anywhere on your computer. A folder 'source_code' would generate. Enter the 'source_code' folder and then open a terminal at the folder path. Then start to build the project, which would take typically 5-15 minutes through the following steps.

### generate .wts from pytorch with .pt

```
python gen_wts.py -w detection_model.pt -o detection_model.wts

```

### build

```
mkdir build
cd build
cp ../detection_model.wts . 
cmake ..
make
./yolov5_pro -s ../detection_model.wts ../detection_model.engine n 

```

## Run a demo

It typically takes less than 1 minute to run the demo.

```
./yolov5_pro -d ../detection_model.engine ../data  

```

Expected output: event images with detected bounding boxes under the 'output' folder.

## Run on your own data

Use your own event image data saved in the format of '.png' to replace all the files in the 'data' folder then follow the operation in 'Run a demo'.

## Special Note

If don't have an event camera, to run a demo, the only difference is to use the 'CMakeList_offline.txt' instead of 'CMakeList.txt' in the build process.


