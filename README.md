# CIFAR10-Transfer-Learning
CIFAR-10 image classification using CNNs and MobileNetV2 transfer learning
# CIFAR-10 Image Classification using Transfer Learning

A deep learning project comparing a custom CNN trained from scratch with ImageNet-pretrained MobileNetV2 for CIFAR-10 image classification. The project explores data augmentation, transfer learning, feature extraction, selective fine-tuning, model evaluation, and error analysis.

## Overview

The objective of this project is to classify CIFAR-10 images into 10 object categories using convolutional neural networks and transfer learning.

Three different approaches were implemented and compared:

1. CNN trained from scratch
2. MobileNetV2 with frozen ImageNet-pretrained features
3. MobileNetV2 with selective fine-tuning

The final fine-tuned MobileNetV2 achieved **88.51% test accuracy**, compared with **70.81%** for the CNN baseline.

### Performance Improvement

CNN Baseline
70.81%

        ↓ +15.00 percentage points

MobileNetV2 Feature Extraction
85.91%

        ↓ +2.60 percentage points

MobileNetV2 Fine-Tuning
88.51%

Overall, transfer learning and fine-tuning improved test accuracy by:
+17.70 percentage points
