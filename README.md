# ESPT: A Self-Supervised Episodic Spatial Pretext Task for Improving Few-Shot Learning

This repository contains the reference Pytorch source code for the following paper:

[ESPT: A Self-Supervised Episodic Spatial Pretext Task for Improving Few-Shot Learning](https://arxiv.org/abs/2304.13287), which has been accepted by AAAI 2023

Yi Rong, Xiongbo Lu, Zhaoyang Sun, Yaxiong Chen, Shengwu Xiong 

If you find our code or paper useful to your research work, please consider citing our work using the following bibtex:
```
@InProceedings{Rong_ESPT_2023,
    title     = {ESPT: A Self-Supervised Episodic Spatial Pretext Task for Improving Few-Shot Learning},
    volume    = {37},
    url       = {https://ojs.aaai.org/index.php/AAAI/article/view/26148},
    DOI       = {10.1609/aaai.v37i8.26148},
    number    = {8},
    booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
    author    = {Rong, Yi and Lu, Xiongbo and Sun, Zhaoyang and Chen, Yaxiong and Xiong, Shengwu},
    year      = {2023},
    month     = {Jun.},
    pages     = {9596-9605}
}
```
## Environment
**Pytorch 1.7.0** and **torchvision 0.8.0** or higher version with cuda support are used for our implementation.

All the experiments are conducted on **Intel(R)Xeon(R) Gold 5117 @2.00GHz CPU**, **NVIDIA A100 Tensor Core GPU**, and **Ubuntu 18.04.6** LTS operation system.

## Data Preparing
1. At first, you need to set the value of `data_path` in `config.yml`, it should be the **absolute path** of the folder that stores all the data.

2. The following datasets are used in our paper: 
    - CUB_200_2011 \[[Dataset Page](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [Download Link](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view)\]
    - mini-ImageNet \[[Dataset Page](https://github.com/twitter/meta-learning-lstm), [Download Link](https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view)\] 
    - tiered-ImageNet (derived from [DeepEMD](https://arxiv.org/abs/2003.06777)'s [implementation](https://github.com/icoz69/DeepEMD)) \[[Dataset Page](https://github.com/icoz69/DeepEMD), [Download Link](https://drive.google.com/file/d/1ANczVwnI1BDHIF65TgulaGALFnXBvRfs/view)\]
    
    You can download individual datasets using the download links provided above, and extract them into your `data_path` folder. 

3. Then, you need to pre-process all the datasets one-by-one into their corresponding few-shot versions by using the following commands:
    ```
    cd data
    python init_CUB_fewshot.py
    python init_mini-ImageNet.py
    python init_tiered-ImageNet_DeepEMD.py
    ```
4. After that, the following folders will exist in your `data_path`:
    - `CUB_fewshot_raw`: 100/50/50 classes for train/validation/test, using raw uncropped images as input
    - `mini-ImageNet`: 64/16/20 classes for train/validation/test
    - `tiered-ImageNet_DeepEMD`: 351/91/160 classes for train/validation/test, images have size of 224x224
    
    Under each folder, images are organized into `train`, `val`, and `test` folders. In addition, you can also find folders named `val_pre` and `test_pre`, which contain validation and testing images pre-resized to the size of 84x84 for the sake of speed.

## Model Training and Testing




