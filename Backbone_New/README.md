# SimpleNet: An Efficient Network for Image Anomaly Detection and Localization

## Overview

SimpleNet is designed for efficient image anomaly detection and localization. The default dataset used in this implementation is the `metal_nut` subset from the `MVTec` dataset.

## Installation

To set up the environment for this project, follow these steps:

```sh
# Create and activate a new conda environment
conda create -n simpleNetEnv python=3.8
conda activate simpleNetEnv

# Install PyTorch and other dependencies
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install opencv-python==4.5.5.64 scikit-learn scipy tqdm


## Training

1. **Setup**: Ensure your dataset path is correctly configured in `main.py`.
2. **Run Training**: Use the following command to start the training process:

```sh
python main.py --train
```

## Testing

1. **Setup**: Ensure your dataset path is correctly configured in `main.py`.
2. **Run Testing**: Use the following command to start the testing process:

```sh
python main.py --test
```

## Results

The table below shows the performance metrics for different ResNet backbones after 150 epochs of training.

| Backbone  | Epochs | F1 Score (%) | Accuracy (%) | AUROC (%) |
|:---------:|:------:|-------------:|-------------:|----------:|
| ResNet18  |  150   |         97.4 |         95.7 |      99.1 |
| ResNet34  |  150   |         97.4 |         95.7 |      99.3 |
| ResNet50  |  150   |        100.0 |        100.0 |     100.0 |

## Sample Results

Here are some visual results of the anomaly detection and localization:

![Sample Result 1](./demo/1.jpg)
![Sample Result 2](./demo/2.jpg)
![Sample Result 3](./demo/3.jpg)
![Sample Result 4](./demo/4.jpg)

## Acknowledgements

This implementation is inspired by the work available in the [DonaldRR/SimpleNet](https://github.com/DonaldRR/SimpleNet) repository.
```
