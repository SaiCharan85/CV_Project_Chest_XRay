# Pneumonia Detection in Chest X-Ray Images Using Deep Learning

A complete end-to-end system for detecting pneumonia in pediatric chest X-ray images using a ResNet50 convolutional neural network. The model is trained in Python, exported to ONNX format, and deployed in a modular C++ inference pipeline using OpenCV and ONNX Runtime.

**Course**: CS 5330 Computer Vision, Northeastern University  
**Date**: April 2026

## Team Members

| Name | Email |
|------|-------|
| Sri Sai Charan Yarlagadda | yarlagadda.sr@northeastern.edu |
| Dina Barua | barua.d@northeastern.edu |
| Sushma Ramesh | ramesh.sus@northeastern.edu |

## Demo Video

https://drive.google.com/file/d/1zTbUbuaylyGgzlvAqJuNVuGPNbDXRsZQ/view?usp=sharing

## Table of Contents

- [Overview](#overview)
- [Results at a Glance](#results-at-a-glance)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Dataset Setup](#dataset-setup)
- [Step 1: Model Training (Python)](#step-1-model-training-python)
- [Step 2: ONNX Export](#step-2-onnx-export)
- [Step 3: C++ Build and Compilation](#step-3-c-build-and-compilation)
- [Step 4: Running Inference](#step-4-running-inference)
- [Step 5: Understanding the Output](#step-5-understanding-the-output)
- [System Architecture](#system-architecture)
- [C++ Class Reference](#c-class-reference)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualization](#visualization)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Acknowledgements](#acknowledgements)

## Overview

Pneumonia is a leading cause of death among children under five worldwide. Diagnosis relies on radiologist interpretation of chest X-rays, which is subjective, slow, and unavailable in many regions. This project automates pneumonia detection using deep learning.

The system works in two phases:

1. **Training phase** (Python): Fine-tune a ResNet50 model pre-trained on ImageNet using the ChestXRay2017 pediatric chest X-ray dataset. Export the trained model to ONNX format.

2. **Inference phase** (C++): Load the ONNX model using ONNX Runtime, preprocess test images with OpenCV, run inference, compute evaluation metrics, and generate annotated visualizations showing actual vs. predicted labels.

The C++ deployment eliminates the Python dependency at inference time, enabling real-time processing suitable for clinical environments.

## Results at a Glance

| Metric | Value |
|--------|-------|
| Test Accuracy | 84.46% |
| Precision | 80.33% |
| Recall | **99.49%** |
| F1 Score | 88.89% |
| Inference Speed | 0.199 sec/image |
| Total Test Images | 624 |
| True Positives | 388 |
| False Positives | 95 |
| True Negatives | 139 |
| False Negatives | 2 |

The system catches 388 out of 390 pneumonia cases (99.49% recall), missing only 2. In medical screening, this high recall is critical because a missed pneumonia case is far more dangerous than a false alarm.

## Project Structure

```
CV_FINAL/
│
├── train.py                          # Python: ResNet50 training script
├── export_onnx.py                    # Python: Export trained model to ONNX
├── best_model.pth                    # Saved PyTorch weights (~90 MB, not in repo)
├── pneumonia_detector.onnx           # Exported ONNX model (~90 MB, not in repo)
│
├── chest_xray/chest_xray/            # Dataset (not in repo, download separately)
│   ├── train/
│   │   ├── NORMAL/                   # 1,341 training images
│   │   └── PNEUMONIA/                # 3,875 training images
│   ├── val/
│   │   ├── NORMAL/                   # 8 validation images
│   │   └── PNEUMONIA/                # 8 validation images
│   └── test/
│       ├── NORMAL/                   # 234 test images
│       └── PNEUMONIA/                # 390 test images
│
├── output/                           # Generated annotated X-ray images
│
└── src/
    ├── main.cpp                      # Pipeline orchestrator
    ├── ImageLoader.h                 # Header: image loading
    ├── ImageLoader.cpp               # Loads JPEGs from NORMAL/PNEUMONIA folders
    ├── ImagePreprocessor.h           # Header: preprocessing
    ├── ImagePreprocessor.cpp         # Resize, normalize, CHW tensor conversion
    ├── Model_Inference.h             # Header: ONNX inference
    ├── Model_Inference.cpp           # ONNX Runtime session, forward pass, softmax
    ├── MetricsComputer.h             # Header: evaluation metrics
    ├── MetricsComputer.cpp           # Accuracy, precision, recall, F1, confusion matrix
    ├── Visualizer.h                  # Header: image annotation
    └── Visualizer.cpp                # Annotate X-rays with actual/predicted labels
```

## Prerequisites

### For Training (Python)

- Python 3.8+
- PyTorch 2.x with torchvision
- CUDA or MPS (Apple GPU) for accelerated training (optional, CPU works too)

### For Inference (C++)

- C++17 compatible compiler
- Visual Studio 2022 (Windows) or g++ with C++17 support (macOS/Linux)
- OpenCV 4.x
- ONNX Runtime

### Package Managers

**Windows (vcpkg):**
```bash
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install opencv:x64-windows onnx-runtime:x64-windows
.\vcpkg integrate install
```

**macOS (Homebrew):**
```bash
brew install opencv onnxruntime
```

## Dataset Setup

1. Download the ChestXRay2017 dataset from Kaggle:
   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

2. Extract and place it so the folder structure matches:
   ```
   chest_xray/chest_xray/train/NORMAL/
   chest_xray/chest_xray/train/PNEUMONIA/
   chest_xray/chest_xray/test/NORMAL/
   chest_xray/chest_xray/test/PNEUMONIA/
   ```

3. Verify image counts:
   - Training: 1,341 normal + 3,875 pneumonia = 5,216
   - Test: 234 normal + 390 pneumonia = 624

The dataset has a roughly 3:1 class imbalance (pneumonia to normal) in the training set. This is handled during training with a weighted random sampler.

## Step 1: Model Training (Python)

```bash
cd CV_FINAL
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows
pip install torch torchvision
```

Run the training script:
```bash
python train.py
```

**What `train.py` does:**

1. Loads the ChestXRay2017 training set with data augmentation (random resized crop, horizontal flip)
2. Applies ImageNet normalization: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
3. Creates a weighted random sampler to handle the 3:1 class imbalance
4. Loads ResNet50 pre-trained on ImageNet and replaces the final fully connected layer (2048 to 2 classes)
5. Trains for 10 epochs using cross-entropy loss with Adam optimizer
6. Saves the best model checkpoint to `best_model.pth`

**Training output:**
- The best checkpoint is saved at the epoch with highest validation accuracy
- Current best: epoch 2, achieving 81.89% test accuracy after initial training
- Note: the validation set is only 16 images, so validation accuracy is not a reliable signal

## Step 2: ONNX Export

After training, export the model to ONNX format for C++ deployment:

```bash
python export_onnx.py
```

**What `export_onnx.py` does:**

1. Loads the trained weights from `best_model.pth`
2. Creates a ResNet50 model with the modified final layer (2048 to 2)
3. Exports to `pneumonia_detector.onnx` using `torch.onnx.export`
4. Configuration: opset version 12, dynamic batch size, constant folding enabled

**ONNX model specifications:**
- Input name: `"input"`, shape: `[batch_size, 3, 224, 224]`
- Output name: `"output"`, shape: `[batch_size, 2]` (logits for [NORMAL, PNEUMONIA])
- File size: approximately 90 MB

## Step 3: C++ Build and Compilation

### Option A: Visual Studio 2022 (Windows, recommended)

1. Open Visual Studio and create a new C++ Console Application
2. Add all `.cpp` files under **Source Files** in Solution Explorer
3. Add all `.h` files under **Header Files**
4. If using vcpkg with `integrate install`, includes and libs are configured automatically
5. Set the build target to **x64**
6. Build the solution (Ctrl+Shift+B)

**Important:** Make sure all 6 source files are added to the project:
- `main.cpp`, `ImageLoader.cpp`, `ImagePreprocessor.cpp`
- `Model_Inference.cpp`, `MetricsComputer.cpp`, `Visualizer.cpp`

Missing source files will cause LNK2019 (unresolved external symbol) linker errors.

### Option B: Command line (macOS/Linux)

```bash
g++ -std=c++17 \
    src/main.cpp \
    src/ImageLoader.cpp \
    src/ImagePreprocessor.cpp \
    src/Model_Inference.cpp \
    src/MetricsComputer.cpp \
    src/Visualizer.cpp \
    $(pkg-config --cflags --libs opencv4) \
    -I/opt/homebrew/include/onnxruntime \
    -L/opt/homebrew/lib -lonnxruntime \
    -o pneumonia_detector
```

## Step 4: Running Inference

Make sure the following files are in your working directory (or adjust paths in `main.cpp`):
- `pneumonia_detector.onnx` (the exported model)
- `chest_xray/chest_xray/` (the dataset folder)

Run the executable:
```bash
./pneumonia_detector
```

The program will:
1. Load all 624 test images (234 normal + 390 pneumonia)
2. Preprocess each image (resize, normalize, CHW conversion)
3. Run ONNX Runtime inference on each image
4. Print running accuracy every 50 images
5. Print the full evaluation report (accuracy, precision, recall, F1)
6. Print the confusion matrix
7. Save annotated X-ray images to the `output/` folder

## Step 5: Understanding the Output

### Terminal Output

```
Loaded 234 images from chest_xray/chest_xray/test/NORMAL
Loaded 390 images from chest_xray/chest_xray/test/PNEUMONIA
Total test images: 624
Loading model from: pneumonia_detector.onnx
Model loaded successfully!
Processed 50/624  | Running accuracy: 62%
Processed 100/624 | Running accuracy: 70%
...
Processed 624/624 | Running accuracy: 84.4551%

===== Inference Complete =====
Total images:  624
Correct:       527
Accuracy:      84.4551%
Total time:    124.306 sec
Per image:     0.199208 sec

========== Evaluation Report ==========
  Accuracy  : 0.8446
  Precision : 0.8033
  Recall    : 0.9949
  F1 Score  : 0.8889
========================================

========= Confusion Matrix =============
                Predicted
              NORMAL  PNEUMONIA
Actual NORMAL  [ 139      95 ]
Actual PNEUM   [   2     388 ]
========================================
```

### Annotated Images

The `output/` folder contains annotated X-ray images sampled from all four quadrants of the confusion matrix:

- **True Positives**: actual PNEUMONIA, correctly predicted as PNEUMONIA (green banner)
- **True Negatives**: actual NORMAL, correctly predicted as NORMAL (green banner)
- **False Positives**: actual NORMAL, incorrectly predicted as PNEUMONIA (red banner)
- **False Negatives**: actual PNEUMONIA, incorrectly predicted as NORMAL (red banner)

Each image displays:
- Top banner: ground truth actual label (dark blue)
- Bottom banner: predicted label + confidence score (green if correct, red if wrong)
- Right side of bottom banner: "CORRECT" or "WRONG" status text

## System Architecture

```
Chest X-ray (JPEG)
       │
       ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│ ImageLoader  │────▶│ Preprocessor │────▶│ Model Inference │────▶│  Evaluation   │
│              │     │              │     │                 │     │              │
│ Read JPEGs   │     │ Resize 224²  │     │ ONNX Runtime    │     │ Metrics +    │
│ Assign labels│     │ BGR→RGB      │     │ ResNet50 fwd    │     │ Visualizer   │
│              │     │ Normalize    │     │ Softmax         │     │              │
│              │     │ CHW layout   │     │ Threshold 0.5   │     │              │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘
                                                                        │
                                                              ┌─────────┴─────────┐
                                                              ▼                   ▼
                                                     Evaluation Report    Annotated Images
                                                     (terminal)          (output/ folder)
```

## C++ Class Reference

### ImageLoader

```cpp
static std::vector<ImageData> loadFromFolder(const std::string& folder_path, int label);
static std::vector<ImageData> loadDataset(const std::string& dataset_root, const std::string& split);
```

- Reads all `.jpg`, `.jpeg`, and `.png` files from a directory
- Each `ImageData` contains: `cv::Mat image`, `std::string filename`, `int label`
- Labels: 0 = NORMAL, 1 = PNEUMONIA
- `loadDataset` loads both NORMAL and PNEUMONIA folders for a given split (e.g., "test")

### ImagePreprocessor

```cpp
static std::vector<float> preprocess(const cv::Mat& image);
```

- Input: raw BGR image of any size
- Output: flat `std::vector<float>` of size 150,528 (3 x 224 x 224) in CHW format
- Pipeline: resize to 224x224, BGR to RGB, scale to [0,1], normalize with ImageNet mean/std

### ModelInference

```cpp
ModelInference(const std::string& model_path);
std::vector<float> infer(const std::vector<float>& input_tensor);
std::string predict(const std::vector<float>& input_tensor, float threshold = 0.5f);
```

- Constructor loads the ONNX model into an ONNX Runtime session
- `infer()` returns raw logits: [score_normal, score_pneumonia]
- `predict()` applies numerically stable softmax, compares P(PNEUMONIA) to threshold, returns "NORMAL" or "PNEUMONIA"

### MetricsComputer

```cpp
static Metrics compute(const std::vector<int>& predicted, const std::vector<int>& true_labels);
static void printReport(const Metrics& m);
static void printConfusionMatrix(const Metrics& m);
```

- Computes TP, TN, FP, FN from predicted and true label vectors
- Calculates accuracy, precision, recall, and F1 score
- Prints formatted evaluation report and 2x2 confusion matrix to terminal

### Visualizer

```cpp
Visualizer(const std::string& output_dir);
void annotateAndSave(const cv::Mat& image, const std::string& filename,
                     const std::string& actual_label, const std::string& prediction,
                     float confidence);
```

- Creates the output directory if it does not exist
- Draws a dark blue banner at the top with the actual (ground truth) label
- Draws a green (correct) or red (wrong) banner at the bottom with the predicted label, confidence score, and CORRECT/WRONG status
- Saves the annotated image to the output directory

## Preprocessing Pipeline

Every image goes through these 5 steps identically in both Python (training) and C++ (inference):

| Step | Operation | Details |
|------|-----------|---------|
| 1 | Resize | Bilinear interpolation to 224 x 224 pixels |
| 2 | Color convert | BGR (OpenCV default) to RGB |
| 3 | Scale | Pixel values from [0, 255] to [0.0, 1.0] |
| 4 | Normalize | Channel-wise: (pixel - mean) / std using ImageNet statistics |
| 5 | Layout | HWC to CHW, flatten to 150,528 float vector |

**ImageNet normalization values:**
- Mean: [0.485, 0.456, 0.406] (R, G, B)
- Std: [0.229, 0.224, 0.225] (R, G, B)

Matching the preprocessing exactly between training and inference is critical. Even small differences in normalization will cause significant divergence in model outputs.

## Evaluation Metrics

| Metric | Formula | Our Value | Meaning |
|--------|---------|-----------|---------|
| Accuracy | (TP+TN) / Total | 84.46% | Overall correctness |
| Precision | TP / (TP+FP) | 80.33% | When model says pneumonia, how often is it right? |
| Recall | TP / (TP+FN) | 99.49% | What % of actual pneumonia cases do we catch? |
| F1 Score | 2 x (P x R) / (P + R) | 88.89% | Harmonic mean of precision and recall |

**Why recall matters most:** In medical screening, a false negative (missing a sick patient) is far more dangerous than a false positive (flagging a healthy patient for review). Our 99.49% recall means only 2 out of 390 pneumonia cases are missed.

## Visualization

The Visualizer generates annotated images for qualitative analysis. Images are sampled from all four confusion matrix categories to provide a balanced view:

- **3 True Positives**: pneumonia correctly detected (green banner)
- **3 True Negatives**: normal correctly identified (green banner)
- **3 False Positives**: normal incorrectly flagged as pneumonia (red banner)
- **2 False Negatives**: pneumonia missed (red banner, only 2 exist in entire test set)

## Troubleshooting

### Common Build Errors

**LNK2019: unresolved external symbol**
- One or more `.cpp` files are not added to the Visual Studio project
- Right-click Source Files in Solution Explorer, Add Existing Item, select all `.cpp` files

**C1014: too many include files: depth = 1024**
- A header file is including itself (e.g., `MetricsComputer.h` contains `.cpp` code instead of declarations)
- Make sure the `.h` file has `#pragma once` and contains only the struct/class declarations

**Model file not found**
- Ensure `pneumonia_detector.onnx` is in the working directory (same folder as the executable)
- In Visual Studio, the working directory defaults to the project folder, not the build output folder

**No images loaded**
- Check the dataset path in `main.cpp` matches your folder structure
- Default path is `chest_xray/chest_xray` relative to the working directory

### Common Runtime Issues

**Low accuracy or different results than expected**
- Verify preprocessing matches between Python and C++: same resize method, same normalization values, same channel order (RGB not BGR)
- Ensure you are using the correct ONNX model file (exported from the best checkpoint)

**Slow inference**
- Build in Release mode, not Debug (Debug is 5-10x slower)
- ONNX Runtime is configured for single-threaded CPU execution by default

## References

1. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in Proc. IEEE CVPR, 2016, pp. 770-778.
2. P. Rajpurkar, J. Irvin, K. Zhu, et al., "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning," arXiv:1711.05225, 2017.
3. D. S. Kermany, M. Goldbaum, W. Cai, et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning," Cell, vol. 172, no. 5, pp. 1122-1131, 2018.
4. G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely Connected Convolutional Networks," in Proc. IEEE CVPR, 2017, pp. 2261-2269.
5. J. Deng, W. Dong, R. Socher, et al., "ImageNet: A Large-Scale Hierarchical Image Database," in Proc. IEEE CVPR, 2009, pp. 248-255.
6. ONNX Runtime C++ API Documentation: https://onnxruntime.ai/docs/api/cpp/
7. OpenCV Documentation: https://docs.opencv.org/
8. ChestXRay2017 Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Acknowledgements

We would like to thank **Prof. Bruce Maxwell** for his guidance, instruction, and support throughout the CS 5330 Computer Vision course at Northeastern University. We also thank the Guangzhou Women and Children's Medical Center for making the ChestXRay2017 dataset publicly available through Kaggle.

---

*CS 5330 Computer Vision Final Project, Northeastern University, April 2026*
