# 对话式图像搜索机器人
基于原作者的文章进行了代码调优以及用户体验优化
1. 支持语音输入关系描述
2. 历史信息权重可调，加入清空历史记录选项
3. 支持多用户多线程操作
4. 返回多张图片，并接收用户反馈作为下轮对话输入

# Dialog-based interactive image retrieval 

## About this repository
This repository contains an implementation of the models introduced in the paper [Dialog-based Interactive Image Retrieval](https://papers.nips.cc/paper/7348-dialog-based-interactive-image-retrieval.pdf). The network  is implemented using [PyTorch](https://pytorch.org/) and the rest of the framework is in Python. The user model is built directly on top of [ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch). 

## Citing this work
If you find this work useful in your research, please consider citing:
```
@incollection{NIPS2018_7348,
title = {Dialog-based Interactive Image Retrieval},
author = {Guo, Xiaoxiao and Wu, Hui and Cheng, Yu and Rennie, Steven and Tesauro, Gerald and Feris, Rogerio},
booktitle = {Advances in Neural Information Processing Systems 31},
editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
pages = {676--686},
year = {2018},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/7348-dialog-based-interactive-image-retrieval.pdf}
}
```
## Project page
The project page is available at [https://www.spacewu.com/posts/fashion-retrieval/](https://www.spacewu.com/posts/fashion-retrieval/).

## Dependencies
To get started with the framework, install the following dependencies:
- Python 3.6
- [PyTorch 0.4.1](https://pytorch.org/get-started/previous-versions/)

## Dataset
The  dataset used in the paper is built on the [Attribute Discovery Dataset](http://tamaraberg.com/attributesDataset/index.html). Please refer to the [dataset README](dataset/) for our dataset details. The pre-computed image features and user captioning model can be downloaded from [here](https://ibm.box.com/s/a1zml3pyx4v8yblvy48oyjt1vsbjbkrk). 

## License
MIT License
