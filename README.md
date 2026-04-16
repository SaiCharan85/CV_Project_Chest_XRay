# Pneumonia Detection — Project Handoff README

**Course**: CS 5330 Computer Vision Final Project  
**Due**: April 23, 2026  
**Dataset**: ChestXRay2017 (chest_xray/chest_xray/)

---

## Team Split

| Person | Task | Status |
|--------|------|--------|
| Sushma | Python training + ONNX export + C++ ImageLoader + ImagePreprocessor |  Done |
| Charan (Person 2) | `ModelInference.h/.cpp` + batch inference loop in `main.cpp` | 🔲 Up next |
| Dina (Person 3) | `MetricsComputer.h/.cpp` + `Visualizer.h/.cpp` + CLI + report | 🔲 After Charan |

---

## Project Structure

```
CV_FINAL/
├── train.py                    # Python training script (ResNet50)
├── export_onnx.py              # Exports best_model.pth → .onnx
├── best_model.pth              # Saved PyTorch weights (~90MB)
├── pneumonia_detector.onnx     # ONNX model ready for C++ inference (~90MB)
├── chest_xray/chest_xray/      # Dataset
│   ├── train/NORMAL/           # 1341 images
│   ├── train/PNEUMONIA/        # 3875 images
│   ├── test/NORMAL/            # 234 images
│   └── test/PNEUMONIA/         # 390 images
├── venv/                       # Python virtual environment
└── src/
    ├── ImageLoader.h/.cpp          # ✅ Done — loads JPEGs from folders
    ├── ImagePreprocessor.h/.cpp    # ✅ Done — resize, normalize, CHW tensor
    ├── ModelInference.h/.cpp       # 🔲 Charan's task
    ├── MetricsComputer.h/.cpp      # 🔲 Dina's task
    ├── Visualizer.h/.cpp           # 🔲 Dina's task
    └── main.cpp                    # Skeleton — add inference + metrics calls
```

---

## Environment Setup (Mac)

```bash
# 1. install dependencies
brew install opencv onnxruntime

# 2. activate python venv (for any python work)
cd ~/Documents/CV_FINAL
source venv/bin/activate
```

---

## What's Already Done (Sushma)

### Python Training (`train.py`)
- Fine-tuned ResNet50 on ChestXRay2017 using PyTorch
- Handles class imbalance (1341 normal vs 3875 pneumonia) with weighted sampler
- ImageNet normalization: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- Achieved **81.89% test accuracy** after 10 epochs on MPS (Apple GPU)
- Best model saved to `best_model.pth`

### ONNX Export (`export_onnx.py`)
- Exported `best_model.pth` → `pneumonia_detector.onnx` (~90MB)
- Input shape: `[batch_size, 3, 224, 224]`
- Output shape: `[batch_size, 2]` → `[P(NORMAL), P(PNEUMONIA)]`
- Opset version: 12

### `ImageLoader` (C++)
- `ImageLoader::loadFromFolder(path, label)` — loads all JPEGs from a folder
- `ImageLoader::loadDataset(root, split)` — loads NORMAL + PNEUMONIA for a split
- Labels: `0 = NORMAL`, `1 = PNEUMONIA`
- Returns `vector<ImageData>` where each entry has `.image`, `.filename`, `.label`

### `ImagePreprocessor` (C++)
- `ImagePreprocessor::preprocess(cv::Mat)` — returns `vector<float>` of size 150528
- Pipeline: resize to 224×224 → BGR→RGB → normalize with ImageNet mean/std → CHW layout
- Verified: tensor size = `3 × 224 × 224 = 150528` ✅

### Compile Command
```bash
g++ -std=c++17 src/main.cpp src/ImageLoader.cpp src/ImagePreprocessor.cpp \
    $(pkg-config --cflags --libs opencv4) \
    -o pneumonia_detector
```

---

## Charan's Task — `ModelInference.h/.cpp`

### What you need to implement

**`ModelInference.h`**:
```cpp
#pragma once
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <vector>
#include <string>

class ModelInference {
public:
    ModelInference(const std::string& model_path);
    std::vector<float> infer(const std::vector<float>& input_tensor);
    std::string predict(const std::vector<float>& input_tensor, float threshold = 0.5f);
private:
    Ort::Env env;
    Ort::Session session;
    Ort::SessionOptions session_options;
};
```

**Key details**:
- Model path: `"pneumonia_detector.onnx"`
- Input name: `"input"`, shape: `{1, 3, 224, 224}`
- Output name: `"output"`, shape: `{1, 2}`
- `infer()` returns raw logits `[score_normal, score_pneumonia]`
- `predict()` applies softmax, compares `P(PNEUMONIA)` to threshold, returns `"NORMAL"` or `"PNEUMONIA"`

**ONNX Runtime includes** (add to compile command):
```bash
-I/opt/homebrew/include/onnxruntime \
-L/opt/homebrew/lib \
-lonnxruntime
```

**Full compile command once ModelInference is added**:
```bash
g++ -std=c++17 src/main.cpp src/ImageLoader.cpp src/ImagePreprocessor.cpp src/ModelInference.cpp \
    $(pkg-config --cflags --libs opencv4) \
    -I/opt/homebrew/include/onnxruntime \
    -L/opt/homebrew/lib -lonnxruntime \
    -o pneumonia_detector
```

### What to add in `main.cpp`
After Sushma's loader/preprocessor block, add a loop like:
```cpp
ModelInference model("pneumonia_detector.onnx");
for (auto& img : images) {
    auto tensor = ImagePreprocessor::preprocess(img.image);
    std::string prediction = model.predict(tensor);
    // store prediction + img.label for MetricsComputer
}
```

---

## Dina's Task — `MetricsComputer` + `Visualizer` + CLI

### `MetricsComputer.h/.cpp`
Compute from vectors of `predicted_labels` and `true_labels`:
- Accuracy = (TP + TN) / total
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)  ← most important, don't miss pneumonia cases
- F1 = 2 × (precision × recall) / (precision + recall)
- Confusion matrix (2×2)

### `Visualizer.h/.cpp`
- Takes a `cv::Mat` image + prediction string + confidence score
- Draws prediction label and confidence on the image (use `cv::putText`)
- Saves annotated images to an `output/` folder
- Generate at least 10 sample annotated X-rays for the report

### CLI flags to add in `main.cpp`
```
--model   path to .onnx file     (default: pneumonia_detector.onnx)
--data    path to dataset root    (default: chest_xray/chest_xray)
--split   which split to run on   (default: test)
--threshold  decision threshold   (default: 0.5)
--output  folder for visualizations
```

---

## Target Metrics (from proposal)
- Test Accuracy ≥ 85%
- Precision ≥ 0.85
- Recall ≥ 0.80 (critical)
- F1 ≥ 0.82
- Inference speed < 1 sec/image

Current test accuracy after 10 epochs: **81.89%** — more training or fine-tuning the threshold can push this over 85%.

---

## Notes
- Val set is only 16 images — val accuracy (100%) is not a reliable signal, test accuracy is what matters
- The model tends to over-predict PNEUMONIA (safer for medical use, but watch precision)
- All Python work runs inside `venv` — always `source venv/bin/activate` first
- `best_model.pth` is the checkpoint from epoch 2 (first time val acc hit 100%)