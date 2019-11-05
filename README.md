## 1. Introduction
This repo. is the PyTorch implementation of multi-object tracker OTCD.
The paper is [real-time online multi-object tracking in compressed domain](https://ieeexplore.ieee.org/abstract/document/8734056).
There maybe a slight gap between the performance obtained by this script and the performance reported in the paper.

## 2. Citation
```
@article{liu2019real,
  title={Real-Time Online Multi-Object Tracking in Compressed Domain},
  author={Liu, Qiankun and Liu, Bin and Wu, Yue and Li, Weihai and Yu, Nenghai},
  journal={IEEE Access},
  volume={7},
  pages={76489--76499},
  year={2019},
  publisher={IEEE}
}
```

## 3. Requirements
```
PyTorch = 0.3
python >= 3.5
```

## 4. Usage
1) down load this script
```
git clone https://github.com/liuqk3/OTCD.git
cd OTCD
```
2) download the pretrained model from [BaiduYunPan](https://pan.baidu.com/s/1faVx3KvolH_uXgvwSXYhxg), the extraction code
 is ```e0kq```. Then put all models to ```./save```. If you have any problems with the download process, please email me.

3) run the tracker

```
python tracking_on_mot.py --mot_dir path/to/MOT-dataset
```

## 5. Code for training
The training scripts are also published in ```useful_scripts```. You can train all the models by the given scripts.