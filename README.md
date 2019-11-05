## 1. Introduction
This repo. is the PyTorch implementation of multi-object tracker OTCD.
The paper is [real-time online multi-object tracking in compressed domain](https://ieeexplore.ieee.org/abstract/document/8734056).
There maybe a slight gap between the performance obtained by this script and the performance reported in the paper.


## Requirements
```
PyTorch = 0.3
python >= 3.5
```

## Usage
1) down load this script
```
git clone https://github.com/liuqk3/OTCD.git
cd OTCD
```
2) download the pretrained model from [BaiduYunPan](https://pan.baidu.com/s/1faVx3KvolH_uXgvwSXYhxg), the extraction code
 is ```e0kq```. Then put the two models to ```./save```

3) run the tracker

```
python tracking_on_mot.py --mot_dir path/to/MOT-dataset
```

## code for training
The training scripts are also publishde in ```useful_scripts```