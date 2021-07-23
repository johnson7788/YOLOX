<div align="center"><img src="assets/logo.png" width="350"></div>
<img src="assets/demo.png" >

## Introduction
YOLOX is an anchor-free version of YOLO, with a simpler design but better performance! It aims to bridge the gap between research and industrial communities.
For more details, please refer to our [report on Arxiv](https://arxiv.org/abs/2107.08430).

<img src="assets/git_fig.png" width="1000" >

## Updates!!
* 【2021/07/20】 We have released our technical report on [Arxiv](https://arxiv.org/abs/2107.08430).

## Comming soon
- [ ] YOLOX-P6 and larger model.
- [ ] Objects365 pretrain.
- [ ] Transformer modules.
- [ ] More features in need.

## Benchmark

#### Standard Models.
|Model |size |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---: | :---:       |:---:     |:---:  | :---: | :----: |
|[YOLOX-s](./exps/default/yolox_s.py)    |640  |39.6      |9.8     |9.0 | 26.8 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EW62gmO2vnNNs5npxjzunVwB9p307qqygaCkXdTO88BLUg?e=NMTQYw)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.pth) |
|[YOLOX-m](./exps/default/yolox_m.py)    |640  |46.4      |12.3     |25.3 |73.8| [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/ERMTP7VFqrVBrXKMU7Vl4TcBQs0SUeCT7kvc-JdIbej4tQ?e=1MDo9y)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_m.pth) |
|[YOLOX-l](./exps/default/yolox_l.py)    |640  |50.0  |14.5 |54.2| 155.6 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EWA8w_IEOzBKvuueBqfaZh0BeoG5sVzR-XYbOJO4YlOkRw?e=wHWOBE)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_l.pth) |
|[YOLOX-x](./exps/default/yolox_x.py)   |640  |**51.2**      | 17.3 |99.1 |281.9 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EdgVPHBziOVBtGAXHfeHI5kBza0q9yyueMGdT0wXZfI1rQ?e=tABO5u)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_x.pth) |
|[YOLOX-Darknet53](./exps/default/yolov3.py)   |640  | 47.4      | 11.1 |63.7 | 185.3 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EZ-MV1r_fMFPkPrNjvbJEMoBLOLAnXH-XKEB77w8LhXL6Q?e=mf6wOc)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_darknet53.pth) |

#### Light Models.
|Model |size |mAP<sup>val<br>0.5:0.95 | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---:  |  :---:       |:---:     |:---:  | :---: |
|[YOLOX-Nano](./exps/default/nano.py) |416  |25.3  | 0.91 |1.08 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EdcREey-krhLtdtSnxolxiUBjWMy6EFdiaO9bdOwZ5ygCQ?e=yQpdds)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_nano.pth) |
|[YOLOX-Tiny](./exps/default/yolox_tiny.py) |416  |31.7 | 5.06 |6.45 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EYtjNFPqvZBBrQ-VowLcSr4B6Z5TdTflUsr_gO2CwhC3bQ?e=SBTwXj)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_tiny.pth) |

## Quick Start

<details>
<summary>Installation</summary>

Step1. 安装 YOLOX.
```shell
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -U pip && pip3 install -r requirements.txt
pip3 install -v -e .  # or  python3 setup.py develop
```
Step2. 安装，当训练的时候 [apex](https://github.com/NVIDIA/apex).

```shell
# skip this step if you don't want to train model.
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
Step3. 安装 [pycocotools](https://github.com/cocodataset/cocoapi).
```angular2html
pip install pycocotools
```

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

</details>

<details>
<summary>示例</summary>

Step1. 下载预训练模型，从上面提到的链接

Step2. 使用-n或者-f指定要使用的detector的检测模型的配置

```shell
使用-n指定模型的配置, -c指定预训练的模型参数，必须要要和-n指定的配置对应, -p指定图片，--conf是置信度
python tools/demo.py image -n yolox-s -c /path/to/your/yolox_s.pth.tar --path assets/dog.jpg --conf 0.3 --nms 0.65 --tsize 640 --save_result --device [cpu/gpu]
python tools/demo.py image -n yolox-s -c model/yolox_s.pth --path assets/dog.jpg --conf 0.3 --nms 0.65 --tsize 640 --save_result --device cpu
```
or
```shell
或者使用-f指定模型的配置
python tools/demo.py image -f exps/default/yolox_s.py -c /path/to/your/yolox_s.pth.tar --path assets/dog.jpg --conf 0.3 --nms 0.65 --tsize 640 --save_result --device [cpu/gpu]
```
对于视频的目标检测:
```shell
python tools/demo.py video -n yolox-s -c /path/to/your/yolox_s.pth.tar --path /path/to/your/video --conf 0.3 --nms 0.65 --tsize 640 --save_result --device [cpu/gpu]
```


</details>

<details>
<summary>重现COCO训练</summary>

Step1. 准备 COCO数据集
```shell
cd <YOLOX_HOME>
ln -s /path/to/your/COCO ./datasets/COCO
```

Step2. Reproduce our results on COCO by specifying -n:

```shell
python tools/train.py -n yolox-s -d 8 -b 64 --fp16 -o
                         yolox-m
                         yolox-l
                         yolox-x
```
* -d: 使用的GPU的数量
* -b: 使用的总的 batch size, 你的数量应该是，如果你希望每个GPU是8的batch-size, 那么-b 64应该设置，计算方法： num-gpu * 8
* --fp16:是否使用fp16计算

上面的命令等价于下面的-f

```shell
python tools/train.py -f exps/default/yolox-s.py -d 8 -b 64 --fp16 -o
                         exps/default/yolox-m.py
                         exps/default/yolox-l.py
                         exps/default/yolox-x.py
```

</details>


<details>
<summary>评估</summary>

支持快速评估

```shell
python tools/eval.py -n  yolox-s -c yolox_s.pth.tar -b 64 -d 8 --conf 0.001 [--fp16] [--fuse]
                         yolox-m
                         yolox-l
                         yolox-x
```
* --fuse: fuse conv and bn
* -d: 使用GPU数量进行评估，默认所有GPU
* -b: 等同于上面提到的batch-size

To reproduce speed test, we use the following command:
```shell
python tools/eval.py -n  yolox-s -c yolox_s.pth.tar -b 1 -d 1 --conf 0.001 --fp16 --fuse
                         yolox-m
                         yolox-l
                         yolox-x
```

</details>


<details open>
<summary>其它文档</summary>

*  [Training on custom data](docs/train_custom_data.md).

</details>

# 部署


1.  [ONNX export and an ONNXRuntime](./demo/ONNXRuntime)
2.  [TensorRT in C++ and Python](./demo/TensorRT)
3.  [ncnn in C++ and Java](./demo/ncnn)
4.  [OpenVINO in C++ and Python](./demo/OpenVINO)

## Cite YOLOX
If you use YOLOX in your research, please cite our work by using the following BibTeX entry:

```latex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
