```markdown
# MLAD_SimpleNet
Code Extensions for Advanced Machine Learning in Anomaly Detection course on SimpleNet

### Changes

- Added support for ResNet backbones.
- Extended compatibility for MVTec LOCO AD and VisA datasets.

# SimpleNet

**SimpleNet: A Simple Network for Image Anomaly Detection and Localization**

*Zhikang Liu, Yiming Zhou, Yuansheng Xu, Zilei Wang*

[Paper link](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)

## Introduction

SimpleNet is a simple defect detection and localization network built with a feature encoder, feature generator, and defect discriminator. It is designed to be conceptually simple without complex network designs, training schemes, or external data sources.

## Get Started 

### Environment 

**Python 3.8**

**Packages**:
- torch==1.12.1
- torchvision==0.13.1
- numpy==1.22.4
- opencv-python==4.5.1

(Above environment setups are not the minimum requirements, other versions might work too.)

### Data

Edit `run.sh` to set the dataset class and dataset path.

#### MVTec AD

Download the dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad/).

#### MVTec LOCO AD

Download the dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-loco-ad/).

#### VisA

Download the dataset from [here]([https://github.com/zhangzjn/ADer/blob/main/data/README.md/###VisA].

The dataset folders/files follow their original structure.

### Run

#### Demo train

Specify the dataset path (line 1) and log folder (line 10) in `run.sh` before running.

`run.sh` provides the configuration to train models on MVTec AD dataset.

```bash
bash run.sh
```

## Citation
```
@inproceedings{liu2023simplenet,
  title={SimpleNet: A Simple Network for Image Anomaly Detection and Localization},
  author={Liu, Zhikang and Zhou, Yiming and Xu, Yuansheng and Wang, Zilei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20402--20411},
  year={2023}
}
```

**Original Repository Link:** [[https://github.com/DonaldRR/SimpleNet]

```
