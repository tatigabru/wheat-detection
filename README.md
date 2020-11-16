# Wheat Detection Challenge

This is the code for ["Wheat Detection Challenge"](https://www.kaggle.com/c/global-wheat-detection).

The solution is based on EfficientDet detectors, augmentation, [weighted box fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) and pseudolabling. Below you will find description of full pipeline and instructions how to run training, inference on competitions data or inference on your own data.

The solution have been packed using Docker or package to simplify environment preparation.

## Table of content

- [Requirements](#requirements)
    - [Software](#software)
    - [Hardware](#hardware)
- [Models](#models)
- [Training](#training)
    -[Prepare environment](#prepare-environment)
    -[Dataset](#dataset)
    -[Make folds](#make-folds)
- [Inference](#inference)    

## Requirements

#### Software

- Ubuntu 18.04
- Docker (19.03.6, build 369ce74a3c)
- Docker-compose (version 1.27.4, build 40524192)
- Nvidia-Docker (Nvidia Driver Version: 396.44, nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04 for GPU)

 Packages and software specified in `Dockerfile` and `requirements.txt`

#### Hardware

Recommended minimal configuration:

  - Nvidia GPU with at least 11GB Memory *
  - Disk space 20+ GB (free)
  - 16 GB RAM

\* It is possible to calculate predictions on CPU, but training requires GPU.

## Models
EfficientDet models from [Ross Wightman efficientdet-pytorch](https://github.com/rwightman/efficientdet-pytorch). All base models used were pre-trained on MS Coco dataset. 

Pretrained models could be loaded [here]() (kaggle json required).
 
## Training

#### Prepare environment 

#### Install docker

Install [Docker Engine](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) and 
[Docker Compose](https://docs.docker.com/compose/install/)

#### Git clone 
Clone current repository

**Starting service**

Build docker image, start docker-compose service in daemon mode and install requirements inside container.

```bash
$ make build && make start && make install
```

#### Dataset 

The dataset is available on [kaggle platform](https://www.kaggle.com/c/global-wheat-detection/data).

The script for dowloading is in `scripts/download_dataset.sh`. 

You need to have kaggle account, create directory .kaggle and copy kaggle.json in it to access the data.

```bash
$ make data
```

#### Make folds
python -m src.folds.make_folds

<<<<<<< HEAD
#### Images preprocessing and augmentations

The original tiles were scaled to 512x512 px and split location-wise, as in notebooks/. 

The notebook for combining tiles is in notebooks/. 

The data were normalised as in ImageNet.

The images were augmented using [albumentations libruary](https://albumentations.readthedocs.io/en/latest/index.html).

#### Run training 
```bash
$ make train
```

## Inference


#### Get models weigths 

 - Unpack the model weights to `models/` directory



**Start inference** (`models/` directory should be provided with pretrained models)
```bash
$ make inference
```

After pipeline execution final prediction will appear in `data/preds/` directory.

Change path and file names for your folders in `scripts/stage4`. 
Alternatevely, run

```bash
$ docker exec open-cities-dev \
    python -m src.predict_tif \
      --configs configs/stage3-srx50-2-f0.yaml \
      --src_path <path/to/your/tif/file.tif> \
      --dst_path <path/for/result.tif> \
      --batch_size 4 \ 
      --gpu '0' \     
```

**Stop service**

After everything is done stop docker container
```bash
$ make stop
```

Bring everything down, removing the container entirely
=======
## Run train 

## Inference

>>>>>>> 303ab158c9d0c48e8e46a4fb68b6a83cf73cffb2

```bash
$ make clean
```
