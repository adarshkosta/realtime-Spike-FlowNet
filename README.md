# Spike-FlowNet: Event-based Optical Flow Estimation with Energy-Efficient Hybrid Neural Networks


![Predicted flow.](pic.png)  

This repository contains the code associated with [Spike-FlowNet: Event-based Optical Flow Estimation with Energy-Efficient Hybrid Neural Networks](https://arxiv.org/abs/2003.06696).


## Introduction
Event-based cameras display great potential for a variety of tasks such as high-speed motion detection and navigation in low-light environments where conventional frame-based cameras suffer critically. This is attributed to their high temporal resolution, high dynamic range, and low-power consumption. However, conventional computer vision methods as well as deep Analog Neural Networks (ANNs) are not suited to work well with the asynchronous and discrete nature of event camera outputs. Spiking Neural Networks (SNNs) serve as ideal paradigms to handle event camera outputs, but deep SNNs suffer in terms of performance due to spike vanishing phenomenon. To overcome these issues, we present Spike-FlowNet, a deep hybrid neural network architecture integrating SNNs and ANNs for efficiently estimating optical flow from sparse event camera outputs without sacrificing the performance. The network is end-to-end trained with self-supervised learning on Multi-Vehicle Stereo Event Camera (MVSEC) dataset. Spike-FlowNet outperforms its corresponding ANN-based method in terms of the optical flow prediction capability while providing significant computational efficiency.


## Installation
Clone this repository using: 
```git clone https://github.com/chan8972/Spike-FlowNet.git```

Create a conda environment using the [environment.yml](environment.yml) file: 
```conda env create -f environment.yml```

Activate the conda environment: 
```conda activate spikeflownet```


## Dataset

The data for the ```outdoor_day``` and ```indoor_flying``` sequences can be found [__**here**__](https://daniilidis-group.github.io/mvsec/download/#hdf5-files). 

Ground truth flow computed from the paper can also be downloaded [__**here**__](https://drive.google.com/drive/folders/1XS0AQTuCwUaWOmtjyJWRHkbXjj_igJLp?usp=sharing).

Download the ```*_data.hdf5``` and ```*_gt.hdf5``` files from the above link in their respective folders inside the [/datasets](datasets) directory.

Example: Download ```indoor_flying1_data.hdf5``` and ```indoor_flying1_gt.hdf5``` files into [/datasets/indoor_flying1](datasets/indoor_flying1) directory.

Convert the hdf5 files into encoded format using [/encoding/split_coding.py](encoding/split_coding.py).


## Training

The basic syntax is:

For ```dt=1``` : ```python3 main_snn_flow_dt1.py```

For ```dt=4``` :  ```python3 main_snn_flow_dt4.py```


## Pre-trained Model

The pretrained models for ```dt=1``` and ```dt=4``` can be found in [/pretrain](/pretrain) directory.


## Testing with flow visualization

The basic syntax is:

For ```dt=1``` : ```python3 main_snn_flow_dt1.py --evaluate --pretrained='checkpoint_path' --render``` 

For ```dt=4``` : ```python3 main_snn_flow_dt4.py --evaluate --pretrained='checkpoint_path' --render```


## Some useful Command-line arguments 

```--data``` : specifies the dataset folder [/datasets](datasets/)

```--savedir``` : directory for saving training results

```--workers``` : number of workers to use

```--render``` : render flow outputs while evaluating

```--evaluate-interval``` : how many epochs to evaluate after

```--pretrained``` : path to pretrained model

Other available command line arguments for hyperparameter tuning can be found in the ```main_snn_flow_dt*.py``` files.


## Citations

If you find this code useful in your research, please consider citing:

[Chankyu Lee, Adarsh Kumar Kosta, Alex Zihao Zhu, Kenneth Chaney, Kostas Daniilidis and Kaushik Roy, "Spike-FlowNet: Event-based Optical Flow Estimation with Energy-Efficient Hybrid Neural Networks", European Conference on Computer Vision (ECCV), 2020](https://arxiv.org/abs/2003.06696)

```
@InProceedings{lee2020spike,
author = {Lee, Chankyu and Kosta, Adarsh Kumar and Zhu, Alex Zihao and Chaney, Kenneth and Daniilidis, Kostas and Roy, Kaushik},
title = {Spike-FlowNet: Event-based Optical Flow Estimation with Energy-Efficient Hybrid Neural Networks},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2020}
}
```


## Authors

Chankyu Lee, Adarsh Kumar Kosta, Alex Zihao Zhu, Kenneth Chaney, Kostas Daniilidis and Kaushik Roy.

-A collaboration of the [C-BRIC](https://engineering.purdue.edu/C-BRIC) teams from Purdue University and University of Pennsylvania.


## Acknowledgments

Parts of this code were derived from [daniilidis-group/EV-FlowNet](https://github.com/daniilidis-group/EV-FlowNet) and [ClementPinard/FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch).
