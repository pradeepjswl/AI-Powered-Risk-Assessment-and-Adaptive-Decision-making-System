## 🚀 Real-Time Object Detection with YOLOv8

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![YOLOv8](https://img.shields.io/badge/Framework-YOLOv8-00FFFF)
![OpenCV](https://img.shields.io/badge/Tools-OpenCV-%23FF0000)

A high-performance real-time object detection system powered by YOLOv8, supporting both webcam and video inputs with GPU acceleration.


## 📋 Table of Contents
- [Features](#✨-features)
- [Installation](#⚙️-installation)
- [Usage](#🚦-usage)
- [Training](#🎓-training-a-custom-model)
- [Controls](#🎮-controls)
- [Requirements](#📦-requirements)
- [Notes](#📝-notes)

---

## ✨ Features
- **Real-Time Detection** 📹: Webcam or video file input with live bounding boxes
- **GPU Acceleration** 🚄: CUDA support for faster inference (if available)
- **Custom Training** 🧠: Train models on your own dataset
- **High Accuracy** 🎯: Powered by state-of-the-art YOLOv8 architecture

---

## ⚙️ Installation
1. **Create virtual environment**:
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   
# Install dependencies:
pip install -r requirements.txt
PyTorch Ultralytics

# 🚦 Usage
Real-Time Detection
python real_time_detection.py
Custom Model Inference

from detection import RealTimeObjectDetection

detector = RealTimeObjectDetection(
    model_path='path/to/custom_model.pt',
    confidence=0.45  # Adjust detection threshold
)
detector.run_realtime()
# 🎓 Training a Custom Model
Prepare dataset in YOLO format:

dataset/

  ├── images/
  
  └── labels/
  
# Create data.yaml:
yaml
train: ../dataset/images/train

val: ../dataset/images/val

names: ['class1', 'class2', ...]

Start training:

detector = RealTimeObjectDetection()
detector.train_model(
    data_yaml_path='configs/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
# 🎮 Controls
Action	Key

Quit Detection	q

Toggle Confidence	c

Save Frame	s

# 📦 Requirements

Python 3.8+ Python
NVIDIA GPU (recommended) CUDA
Webcam or video source
See requirements.txt for full package list

# 📝 Notes
⚡ Auto GPU Detection: Uses CUDA if available, falls back to CPU

🔧 Configurable Threshold: Adjust confidence in code (default: 0.25)

🌐 Pretrained Models: Automatically downloads COCO weights on first run

📈 Performance: Achieves 65 FPS on RTX 3080 (640x640 resolution)
