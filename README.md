# ResNet-50 on CIFAR-10 with FastAPI + Docker

This project trains a ResNet-50 model on the CIFAR-10 dataset and deploys it as a web API using FastAPI and Docker.

<!-- --- -->

## Project Overview

- **Task**: Image classification (10 classes)
- **Model**: ResNet-50 (PyTorch, pretrained on ImageNet, fine-tuned on CIFAR-10)
- **Deployment**: FastAPI + Docker container

<!-- --- -->

## How to Run

### 1. Train the Model

```bash
cd training
python train.py

