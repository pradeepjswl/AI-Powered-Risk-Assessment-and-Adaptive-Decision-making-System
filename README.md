# Real-Time Object Detection System

This project implements real-time object detection using YOLOv8, one of the most efficient and accurate object detection models.

## Features

- Real-time object detection using webcam or video file
- Support for custom model training
- GPU acceleration (if available)
- High-performance inference
- Easy-to-use API

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Real-time Detection

1. Basic usage with webcam:
```bash
python real_time_detection.py
```

2. Using a custom trained model:
```python
detector = RealTimeObjectDetection(model_path='path/to/your/model.pt')
detector.run_realtime()
```

### Training Custom Model

1. Prepare your dataset in YOLO format
2. Create a data.yaml file with your dataset configuration
3. Train the model:
```python
detector = RealTimeObjectDetection()
detector.train_model(
    data_yaml_path='path/to/data.yaml',
    epochs=100,
    imgsz=640
)
```

## Controls

- Press 'q' to quit the real-time detection window

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for better performance)
- Webcam or video source
- See requirements.txt for Python package dependencies

## Notes

- The system will automatically use GPU if available, otherwise, it will fall back to CPU
- Default confidence threshold is set to 0.25 (can be adjusted in the code)
- The pretrained model will be downloaded automatically on first run 