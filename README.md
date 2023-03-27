# AM<sup>3</sup>Net

We have relased the code of Adaptive Mutual-learning-based Multimodal Data Fusion Network (AM<sup>3</sup>Net) algorithm. And the paper has been published in [*IEEE TCSVT 2022*](https://ieeexplore.ieee.org/abstract/document/9698196). In this paper, we conducted the experiments on the hyperspectral and lidar dataset(Houston and Trento) and multispectral and synthetic aperture radar data (grss-dfc-2007 datasets).

If you have any queries, please do not hesitate to contact me (jinping_wang@foxmail.com).

## Requirements：
- We have tested our algorithm in the following on Windows with CUDA=11.0.

      torch==1.7.0+cu110
      visdom==0.1.8
      numpy==1.19.5
      scipy==1.5.4
      sklearn=0.24.2
      random
      mmcv==1.3.0
      cupy-cuda110==8.5.0

- mmcn is provided by open-mmlab [https://github.com/open-mmlab/mmcv]: python setup.py install

## Train：
- If you want to run on other dataset, conduct the *data.mat*. 

- Trento Data (Hyperspectral and LiDAR Data): Trento dataset is provided by Professor Prof. L. Bruzzone from the University of Trento.

      data.mat  [C1>>C2]
      ----> ground (H*W)
      ----> HSI_data (H*W*C1)
      ----> Lidar_data (H*W*C2)
      
- Start a Visdom server: python -m visdom.server and go to http://localhost:8097 to see the visualizations.

## Test：
- You can use the pre-trained models to obtain the classification results

## Citation：
- Please cite us if our project is helpful to you!

> J. Wang, J. Li, Y. Shi, J. Lai and X. Tan, "AM3Net: Adaptive Mutual-learning-based Multimodal Data Fusion Network," in IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 8, pp. 5411-5426, Aug. 2022, doi: 10.1109/TCSVT.2022.3148257.

Bibtex format :

> @ARTICLE{9698196,
author={Wang, Jinping and Li, Jun and Shi, Yanli and Lai, Jianhuang and Tan, Xiaojun},
journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
title={AM$3$Net: Adaptive Mutual-learning-based Multimodal Data Fusion Network}, 
year={2022},
volume={32},
number={8},
pages={5411-5426},
doi={10.1109/TCSVT.2022.3148257}}
     
     
