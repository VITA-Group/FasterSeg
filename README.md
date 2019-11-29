<!-- 
TODO
BiSeNetHead
delete: common, lr_scheduler
latency/seg_ops
engine/lr_policy
delete all 8s
Prepare searched architect and weight in folder
Latency -->
# FasterSeg: Searching for Faster Real-time Semantic Segmentation

<!-- [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/chenwydj/ultra_high_resolution_segmentation.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/chenwydj/ultra_high_resolution_segmentation/context:python) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

<!-- <a href="https://arxiv.org/abs/1905.06368">Collaborative Global-Local Networks for Memory-Efﬁcient Segmentation of Ultra-High Resolution Images</a> -->

Wuyang Chen, Xinyu Gong, Xianming Liu, Qian Zhang, Yuan Li, Zhangyang Wang

In ICLR 2020.
<!-- [[Youtube](https://www.youtube.com/watch?v=am1GiItQI88)] -->

## Overview

We present FasterSeg, an automatically designed semantic segmentation network with not only state-of-the-art performance but also faster speed than current methods. 

Highlights:
* **Novel search space**: support multi-resolution branches.
* **Fine-grained latency regularization**: alleviate the ``architecture collapse'' problem.
* **Teacher-student co-searching**: distill the teacher to the student for further accuracy boost.
* **SOTA**: FasterSeg achieves extremely fast speed (over 30\% faster than the closest competitor on CityScapes) and maintains competitive accuracy.

<p align="center">
<img src="images/table4.png" alt="Cityscapes" width="550"/></br>
</p>

## Methods

<p align="center">
<img src="images/figure1.png" alt="supernet" width="800"/></br>
</p>

<p align="center">
<img src="images/figure6.png" alt="fasterseg" width="500"/></br>
</p>

## Prerequisites
- Ubuntu
- Python 3
- NVIDIA GPU + CUDA CuDNN

## Installation
* Clone this repo:
```bash
git clone https://github.com/chenwydj/FasterSeg.git
cd FasterSeg
```
* Install dependencies:
```bash
pip install requirements.txt
```
* Install [TensorRT](https://github.com/NVIDIA/TensorRT) (v5.1.5.0): a library for high performance inference on NVIDIA GPUs with [Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/index.html#python).

## Usage
* **whole pipeline: pretrain the supernet &rarr; search the archtecture &rarr; train the teacher &rarr; train the student.**
* you can monitor the whole process in the Tensorboard.

### 1. Search
```bash
cd search
```
#### 1.1 Pretrain the supernet
We first pretrain the supernet without updating the architecture parameter for 20 epochs.
* set `C.pretrain = True` in `config_search_8s.py`.
* start the pretrain process:
```bash
CUDA_VISIBLE_DEVICES=0 python train_search.py
```
* The pretrained weight will be saved in a folder like ```FasterSeg/search/search-pretrain-256x512_F12.L16_batch3-20200101-012345```.

#### 1.2 Search the architecture
We start architecture searching for 30 epochs.
* set the name of your pretrained folder (see above) `C.pretrain = "search-pretrain-256x512_F12.L16_batch3-20200101-012345"` in `config_search_8s.py`.
* start the search process:
```bash
CUDA_VISIBLE_DEVICES=0 python train_search.py
```
* The searched architecture will be saved in a folder like ```FasterSeg/search/search-224x448_F12.L16_batch2-20200102-123456```.

### 2. Train from scratch
* copy the folder which contains the searched architecture into `FasterSeg/train/` or create a symlink via `ln -s FasterSeg/search/search-224x448_F12.L16_batch2-20200102-123456 FasterSeg/train/search-224x448_F12.L16_batch2-20200102-123456`
#### 2.1 Train the teacher network
* uncomment the `## train teacher model only ##` section in `config_train.py` and comment the `## train student with KL distillation from teacher ##` section.
* set the name of your searched folder (see above) `C.load_path = search-224x448_F12.L16_batch2-20200102-123456`
* start the teacher's training process:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```
* The trained teacher will be saved in a folder like `train-512x1024_teacher_batch12-20200103-234501`
#### 2.2 Train the student network (FasterSeg)
* uncomment the `## train student with KL distillation from teacher ##` section in `config_train.py` and comment the `## train teacher model only ##` section.
* set the name of your searched folder (see above) `C.load_path = search-224x448_F12.L16_batch2-20200102-123456`.
* set the name of your teacher's folder (see above) `C.teacher_path = train-512x1024_teacher_batch12-20200103-234501`.
* start the student's training process:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```


## Citation
```
@inproceedings{chen2020fasterseg,
  title={FasterSeg: Searching for Faster Real-time Semantic Segmentation},
  authors={Chen, Wuyang and Gong, Xinyu and Liu, Xianming and Zhang, Qian and Li, Yuan and Wang, Zhangyang},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```

## Acknowledgement
* Segmentation training and evaluation code from [BiSeNet](https://github.com/ycszen/TorchSeg).
* Search method from the [DARTS](https://github.com/quark0/darts).
* slimmable_ops from the [Slimmable Networks](https://github.com/JiahuiYu/slimmable_networks).
* Segmentation metrics code from [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/utils/metrics.py).