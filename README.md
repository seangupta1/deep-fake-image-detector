# Deep Fake Detector

A deep learning project that detects whether a facial image is **real or AI-generated (deep fake)** using PyTorch, OpenCV, and Grad-CAM for model explainability.

> **My Role:** I was primarily responsible for designing the neural network architecture, implementing and training the model, tuning hyperparameters, and developing Grad-CAM visualizations to explain the model's predictions.

## Project Overview

Deep fake technology can generate highly realistic synthetic images, making it increasingly difficult to distinguish manipulated media from authentic photographs. This project applies computer vision and deep learning techniques to classify facial images as either **Real** or **Fake**.

The system uses a custom image classification model built with a ResNet50 architecture and trained on the OpenForensics dataset.

### Final Performance

- **Test Accuracy:** 95.8%
- **ROC-AUC:** 0.9716
- **Test Loss:** 0.12
- **Precision:** 93.89%
- **Recall:** 87.67%
- **Specificity:** 93.96%

---

## My Contributions

### Model Development
- Selected ResNet50 as the backbone architecture
- Removed the default classification layer
- Designed a custom classification head with fully connected layers, ReLU activation, and dropout
- Configured the network for binary classification of real vs fake images

### Training Pipeline
- Implemented the full training and validation loops in PyTorch
- Tuned hyperparameters including learning rate, batch size, and scheduler settings
- Added early stopping and automatic checkpoint saving
- Evaluated the final model on a held-out test set

### Explainable AI
- Implemented Grad-CAM (Gradient-weighted Class Activation Mapping)
- Generated heatmaps to visualize which image regions influenced predictions
- Analyzed whether the model focused on meaningful facial features or background artifacts

---

## Dataset

**Dataset:** OpenForensics Dataset

The dataset contains labeled images of both real and manipulated faces. Since some images contain multiple people, the preprocessing pipeline detects and crops individual faces so that each sample represents a single face.

Kaggle Mirror: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

---

## Data Preprocessing

Images were prepared using OpenCV and custom Python scripts.

### Preprocessing Steps

1. Load images directly from ZIP archives
2. Detect faces using Haar Cascade classifiers
3. Crop individual faces
4. Resize images while preserving aspect ratio
5. Pad images to 150 × 150 pixels
6. Convert images to tensors
7. Normalize pixel values

### Data Augmentation

Applied only to the training set:

- Random horizontal flips
- Brightness, contrast, and saturation jitter

---

## Model Architecture

The classifier is built using a ResNet50-based convolutional neural network.

```text
Input Image (150x150 RGB)
        ↓
ResNet50 Feature Extractor
        ↓
Fully Connected Layer
        ↓
ReLU Activation
        ↓
Dropout (0.2)
        ↓
Final Linear Layer
        ↓
Single Logit Output
        ↓
Sigmoid → Real or Fake (during inference)
