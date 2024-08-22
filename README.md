# NEVACS

The project aims to perform NEVACS for video-activated sorting. The event stream from the event camera is received and preprocessed, and the preprocessed event image is then inferred by the spiking neural network (SNN) classification model, which is deployed on the neuromorphic chip.  The classification results of the SNN classification are then used for sort control.

## System requirements

#### Hardware requirements

NEVACS requires a standard personal computer with a neuromorphic chip (HP201, Lynxi). To run without the neuromorphic chip, you can use the code in Single-Frame-Activated Sorter folder, and a standard personal computer with a GPU (RTX 3080, Nvidia) is required. 

#### OS requirements

NEVACS is supported for Linux and has been tested on the following system:
Linux: Ubuntu 20.04

#### Software dependencies

NEVACS requires and has been tested on OpenCV 3.3.0, OpenEB 3.1.0, Python 3.6, PyTorch 1.9.0, lynbidl-mmlab-v1.3.0, mmcv 1.5.0

Note: lynbidl-mmlab-v1.3.0 is not being able to open source due to commercial interests, please purchase the software from Lynxi Technologies Co., Ltd., and install it according to the included guide from Lynxi.

## How to run a demo

Download the zip file or git clone the whole repo anywhere on your computer. A folder 'NEVACS' would generate. Enter the 'NEVACS' folder and then open a terminal at the folder path. Then choose one of the following modes to run a demo, which would take typically less than 30 seconds to automatically build and start to inference. The spatiotemporal classification results of all particles would be displayed one by one on the terminal console, which would probably need less than 5 minutes. 

#### Run a demo with neuromorphic chip

```
cd tools  
python3 test.py --config clif2fc1ce_itout-b16x1-celldataset --checkpoint latest.pth --use_lyngor 1 --use_legacy 0

```

#### Run a demo with GPU

```
cd tools  
python3 test.py --config clif2fc1ce_itout-b16x1-celldataset --checkpoint latest.pth --use_lyngor 0 --use_legacy 0

```

Expected output: 1. The trained and deployed SNN classification under the 'config/' folder; 2. The spatiotemporal classification results of Hela cells and beads outputed in the terminal.

## Run on your own data

Use your own event image data saved in the format of '.png' or '.jpg' to replace all the files in the 'data/cell-dataset/' folder. The training data should be put under the 'data/cell-dataset/train/' folder, while the test data should be put under the 'data/cell-dataset/test/' folder. Then follow the operation below. 

#### Training the SNN classification model with GPU

```
cd tools  
python3 train.py --config clif2fc1ce_itout-b16x1-celldataset

```

After training the SNN classification model, choose to run one of the running modes in "How to run a demo" in line with your own hardware configuration. The newly trained model would be automatically built and applied to your test data. 


