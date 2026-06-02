# Autonomous Driving Scene Understanding using Attention U-Net

## Overview

This project presents a deep learning-based framework for **pixel-level semantic segmentation** and **depth estimation** in autonomous driving environments. The system leverages **U-Net** and **Attention U-Net** architectures to accurately identify road scenes and estimate object distances, enabling enhanced perception for self-driving vehicles.

The solution is trained and evaluated on the **Cityscapes Dataset**, achieving strong performance in both segmentation and depth estimation tasks.

---

## Features

* Semantic segmentation of urban driving scenes
* Monocular depth estimation
* Attention-enhanced U-Net architecture
* Multi-class scene understanding
* Real-time perception pipeline
* GPU acceleration using CUDA
* Comprehensive model evaluation

---

## Problem Statement

Autonomous vehicles require a detailed understanding of their surroundings to make safe navigation decisions. This project addresses two critical perception tasks:

### Semantic Segmentation

Classifies every pixel in an image into semantic categories such as roads, vehicles, pedestrians, buildings, sky, and vegetation.

### Depth Estimation

Predicts the distance of objects from the vehicle using image data, enabling obstacle avoidance, scene understanding, and path planning.

---

## Output

The following animation demonstrates the model performing semantic segmentation and depth-aware scene understanding on real-world driving footage.

<img src="https://github.com/cjaitej/Self-Driving-Cars/raw/main/test5.gif" width="100%" alt="Self Driving Car Segmentation Output">


## System Architecture

### Segmentation Model

* U-Net
* Attention U-Net
* Encoder-Decoder Architecture
* Skip Connections
* Attention Gates

### Depth Estimation Model

* U-Net based Encoder-Decoder Network
* Multi-scale Feature Extraction
* Custom Depth Estimation Loss Function

The architecture combines feature extraction and spatial localization capabilities to generate accurate segmentation masks and depth maps.

---

## Dataset

### Cityscapes Dataset

The Cityscapes dataset is designed for semantic urban scene understanding and contains diverse driving scenarios collected across multiple cities.

#### Dataset Characteristics

* 5,000 finely annotated images
* 20,000 coarsely annotated images
* 30 semantic classes
* Data collected from 50 cities
* Stereo image pairs
* GPS coordinates
* Vehicle odometry information
* Diverse urban driving environments

---

## Technologies Used

| Technology       | Purpose                   |
| ---------------- | ------------------------- |
| Python           | Core Development          |
| PyTorch          | Deep Learning Framework   |
| CUDA             | GPU Acceleration          |
| NumPy            | Numerical Computing       |
| Matplotlib       | Visualization             |
| Plotly           | Interactive Visualization |
| TorchMetrics     | Performance Evaluation    |
| Jupyter Notebook | Experimentation           |
| VS Code          | Development Environment   |
| Git              | Version Control           |

---

## Training Configuration

### Hyperparameters

| Parameter     | Value |
| ------------- | ----- |
| Learning Rate | 1e-5  |
| Batch Size    | 2     |
| Weight Decay  | 0.01  |
| Epochs        | 3000  |
| Optimizer     | AdamW |

---

## Loss Functions

### Segmentation Loss

The segmentation model utilizes a combination of:

* Cross Entropy Loss
* Intersection over Union (IoU) Loss

```python
Segmentation Loss = Cross Entropy Loss + IoU Loss
```

### Depth Estimation Loss

The depth estimation model combines:

* Mean Squared Error (MSE)
* Smooth L1 Loss
* Structural Similarity Index (SSIM)

This combination helps preserve both numerical accuracy and structural consistency of depth maps.

---

## Evaluation Metrics

### Semantic Segmentation

* Accuracy
* F1 Score
* Confusion Matrix

### Depth Estimation

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)

---

## Results

### Training Performance

#### Depth Estimation

| Metric | Value   |
| ------ | ------- |
| MAE    | 0.01068 |
| MSE    | 0.00034 |

#### Semantic Segmentation

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 95.31% |
| F1 Score | 95.22% |

---

### Testing Performance

#### Depth Estimation

| Metric | Value   |
| ------ | ------- |
| MAE    | 0.01823 |
| MSE    | 0.00229 |

#### Semantic Segmentation

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 91.64% |
| F1 Score | 91.38% |

---

## Segmentation Classes

| Class      | Color Representation |
| ---------- | -------------------- |
| Background | Black                |
| Sky        | Blue                 |
| Trees      | Green                |
| People     | Red                  |
| Roads      | Purple               |
| Vehicles   | Dark Blue            |
| Buildings  | Gray                 |

---

## Applications

* Autonomous Driving
* Lane Detection
* Obstacle Detection
* Pedestrian Detection and Tracking
* Vehicle Detection and Tracking
* Traffic Sign Recognition
* Environment Understanding
* Path Planning
* Driver Assistance Systems

---

## Future Improvements

* Transformer-based segmentation architectures
* Real-time deployment on embedded systems
* Sensor fusion using LiDAR and cameras
* Multi-task learning frameworks
* Enhanced robustness under adverse weather conditions
* Lightweight model optimization for edge deployment

---

## References

1. Attention Is All You Need
   https://arxiv.org/abs/1706.03762

2. U-Net: Convolutional Networks for Biomedical Image Segmentation
   https://arxiv.org/abs/1505.04597

3. Attention U-Net
   https://arxiv.org/abs/1804.03999

4. Cityscapes Dataset
   https://www.cityscapes-dataset.com/

---

## Conclusion

This project demonstrates the effectiveness of combining semantic segmentation and depth estimation for autonomous driving perception. By leveraging U-Net and Attention U-Net architectures, the system achieves high segmentation accuracy and reliable depth prediction, contributing toward safer and more intelligent autonomous vehicle systems.

The results highlight the potential of deep learning-based perception frameworks in enabling robust scene understanding for next-generation self-driving technologies.
