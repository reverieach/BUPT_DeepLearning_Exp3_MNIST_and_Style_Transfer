# Deep Learning Experiment 3: Autoencoder & Style Transfer
# 深度学习实验三：自编码器与风格迁移

This repository contains the implementation and reports for Experiment 3, focusing on Unsupervised Learning and Generative Models.

## 📂 Project Structure

- **`MNIST_Autoencoder/`**: Part 1 - Variational Autoencoders (VAE) & Denoising Autoencoders (DAE) on MNIST.
    - `Experiment_Report_MNIST.md`: Detailed report for Part 1.
    - `train.py`: Training script for Autoencoders.
- **`style_transfer/`**: Part 2 - Arbitrary Style Transfer using AdaIN.
    - `Experiment_Report_StyleTransfer.md`: Detailed report for Part 2 (includes BUPT scenic spot transfer).
    - `README_StyleTransfer.md`: Setup guide and operational manual.
    - `run_lab.py`: Interactive script to run training and style transfer tasks.
    - `custom_tasks.py`: Implementation of BUPT photo transfer and Alpha blending.

## 🚀 Quick Start

### 1. MNIST Autoencoder
Classification and reconstruction on handwritten digits.
```bash
cd MNIST_Autoencoder
python train.py
```

### 2. Style Transfer (AdaIN)
Turning photos into artistic masterpieces.
> **Note**: This requires MS-COCO and WikiArt datasets (not included in repo).

**Run the interactive lab runner:**
```bash
cd style_transfer
python run_lab.py
```
This script will guide you through:
1.  Training the AdaIN model.
2.  Generating training progress visualization.
3.  Task 3: BUPT Scenic Spot Transfer.
4.  Task 4: Alpha Blending Experiment.

## 📝 Experiment Reports
- [MNIST Report](MNIST_Autoencoder/Experiment_Report_MNIST.md)
- [Style Transfer Report](style_transfer/Experiment_Report_StyleTransfer.md)
