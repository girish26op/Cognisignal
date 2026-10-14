# CogniSignal: Adaptive Multi-Model Machine Learning Pipeline for Real-Time PCU-Weighted Dynamic Traffic Signal Optimization

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent transportation machine learning project developed for **CB5305 - Machine Learning Project-Based Learning (PBL)** at **Chennai Institute of Technology**.

---

## 🚦 System Architecture & Overview

Traditional traffic lights operate on rigid, static timers that fail to reflect live vehicular load, resulting in severe urban gridlock, excess fuel waste, and blocked emergency vehicles.

**CogniSignal** resolves this through a multi-model pipeline:
1. **Multi-Class Vehicle Detection**: Evaluates a **Custom PyTorch CNN (from scratch)**, **Faster R-CNN**, and **YOLOv8 Small**.
2. **OpenCV Feature Extraction**: 6-stage classical computer vision filter pipeline (Grayscale $\to$ Gaussian Blur $\to$ Canny Edges $\to$ Dilation $\to$ Contours $\to$ Bounding Boxes).
3. **Transportation Engineering PCU Load Weighting**:
   - $\text{Car} = 1.0\text{ PCU}$
   - $\text{Bus} = 2.5\text{ PCU}$
   - $\text{Truck} = 3.0\text{ PCU}$
   - $\text{Bike / Motorcycle} = 0.5\text{ PCU}$
   - $\text{Auto-Rickshaw} = 1.2\text{ PCU}$
4. **Dynamic Signal Duration Optimization**: Computes traffic density per lane and dynamically allocates green-light duration between **15s and 90s**.
5. **Tabular Machine Learning**: Random Forest Classifier (**94.2% Accuracy**) and Regressor (**1.82s MAE**).
6. **Interactive Streamlit Web Dashboard**: 7 interactive tabs for detection, OpenCV inspection, model benchmarking, custom training, tabular ML, signal simulation, and analytics.

---

## 🏆 Multi-Model Benchmark Results

| Model Name | Precision | Recall | F1-Score | mAP@0.5 | FPS | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN (From Scratch)** | 0.742 | 0.698 | 0.719 | 0.712 | 42.5 | Baseline |
| **Faster R-CNN (MobileNetV3 FPN)** | 0.881 | 0.854 | 0.867 | 0.862 | 18.2 | Evaluated |
| **YOLOv8 Small** | **0.915** | **0.892** | **0.903** | **0.908** | **65.0** | 🏆 **SELECTED BEST** |

Winning model is dynamically selected and saved into `models/selected_best_model.json`.

---

## 📁 Repository Directory Structure

```text
ml pbl/
├── src/
│   ├── __init__.py                 # Package initializer
│   ├── custom_cnn.py               # Custom dual-head PyTorch CNN from scratch
│   ├── faster_rcnn_detector.py     # Faster R-CNN detector with MobileNetV3 FPN
│   ├── yolo_detector.py            # Ultralytics YOLOv8 real-time detector
│   ├── vehicle_detector.py         # Unified detector factory with auto-selection
│   ├── signal_optimizer.py         # PCU load calculation & dynamic green time logic
│   ├── feature_extractor.py        # 6-stage OpenCV feature extraction pipeline
│   ├── model_evaluator.py          # Unified comparative benchmarking engine
│   ├── tabular_model.py            # Random Forest classification & regression models
│   └── visualization.py            # Publication-grade plotting suite
├── data/
│   ├── traffic_dataset.csv         # Generated traffic features dataset
│   ├── generate_sample_assets.py   # Dataset & asset generation utility
│   └── sample_images/              # Diverse test road images
├── models/
│   ├── selected_best_model.json    # Winning model configuration
│   ├── custom_cnn_weights.pth      # Custom PyTorch CNN weights
│   └── tabular_models.pkl          # Trained Random Forest model binary
├── outputs/
│   ├── annotated_output.jpg        # CLI inference output
│   └── plots/                      # Confusion matrix, ROC curve, comparisons
├── evaluate_all.py                 # Benchmark execution CLI script
├── main.py                         # Single-image end-to-end CLI driver
├── app.py                          # 7-Tab Interactive Streamlit Dashboard
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## ⚡ Quickstart Guide

### 1. Run Automated Benchmark
```bash
py evaluate_all.py
```

### 2. Run CLI Detection on a Sample Frame
```bash
py main.py --image data/sample_images/traffic_highway.jpg
```

### 3. Launch the Streamlit Web Application
```bash
py -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 👥 Contributors

- **Mithun Prasath Logesh** (Reg No: 210425244033)
- **Girish R** (Reg No: 210425244045)
- **Department**: Computer Science and Engineering (Computer Science and Business Systems)
- **Institution**: Chennai Institute of Technology (Autonomous), Affiliated to Anna University
