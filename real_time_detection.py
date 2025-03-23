import cv2
import torch
from ultralytics import YOLO
import numpy as np
import time
from datetime import datetime
from playsound import playsound
import os
import threading
import sys
from pathlib import Path

class RealTimeObjectDetection:
    def __init__(self, model_path=None):
        """
        Initialize the real-time object detection system
        """
        if model_path:
            self.model = YOLO(model_path)
        else:
            # Use YOLOv8x for better accuracy
            self.model = YOLO('yolov8x.pt')
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Define different categories of objects with confidence thresholds
        self.object_categories = {
            'vehicles': {
                'classes': ['car', 'truck', 'bus', 'motorcycle', 'bicycle'],
                'threshold': 0.3,
                'sound': 'vehicle_alert.mp3',
                'color': (0, 0, 255)  # Red
            },
            'persons': {
                'classes': ['person'],
                'threshold': 0.3,
                'sound': 'person_alert.mp3',
                'color': (255, 0, 0)  # Blue
            },
            'dangers': {
                'classes': ['knife', 'gun', 'bottle', 'fire'],
                'threshold': 0.2,
                'sound': 'danger_alert.mp3',
                'color': (0, 0, 255)  # Red
            },
            'traffic': {
                'classes': ['traffic light', 'stop sign'],
                'threshold': 0.3,
                'sound': 'behavior_alert.mp3',
                'color': (255, 255, 0)  # Yellow
            }
        }
        
        # Initialize warning log
        self.warning_log = []
        
        # Initialize behavior tracking
        self.behavior_history = {}
        self.last_alert_time = {}
        
        # Initialize camera
        self.cap = None
        
        # Ensure sound files exist
        self.ensure_sound_files()

    def ensure_sound_files(self):
        """Ensure all required sound files exist"""
        sounds_dir = Path('sounds')
        if not sounds_dir.exists():
            print("Creating sounds directory...")
            sounds_dir.mkdir()
            
        # Check each sound file
        for category in self.object_categories.values():
            sound_file = sounds_dir / category['sound']
            if not sound_file.exists():
                print(f"Warning: Sound file {sound_file} not found!")
                print("Please run setup_sounds.py to download or generate sound files")
                return False
        return True

    def initialize_camera(self, source=0):
        """
        Initialize camera with error handling
        """
        try:
            # Try different camera backends
            backends = [
                cv2.CAP_DSHOW,  # DirectShow (Windows)
                cv2.CAP_ANY,    # Auto-detect
                cv2.CAP_MSMF    # Media Foundation (Windows)
            ]
            
            for backend in backends:
                self.cap = cv2.VideoCapture(source + backend)
                if self.cap.isOpened():
                    # Set camera properties for better performance
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    print(f"Successfully opened camera with backend: {backend}")
                    return True
                
            print("Failed to open camera with all backends")
            return False
            
        except Exception as e:
            print(f"Error initializing camera: {str(e)}")
            return False

    def play_alert(self, alert_type):
        """Play alert sound in a separate thread"""
        try:
            sound_file = os.path.join('sounds', alert_type)
            if os.path.exists(sound_file):
                current_time = time.time()
                if alert_type not in self.last_alert_time or \
                   current_time - self.last_alert_time[alert_type] > 2.0:  # Minimum 2 seconds between alerts
                    threading.Thread(target=playsound, args=(sound_file,), daemon=True).start()
                    self.last_alert_time[alert_type] = current_time
        except Exception as e:
            print(f"Error playing alert sound: {str(e)}")

    def detect_objects(self, frame, conf_threshold=0.25):
        """
        Detect objects in a single frame with enhanced warning system
        """
        try:
            # Perform detection
            results = self.model(frame, conf=conf_threshold)[0]
            
            # Track detections for each category
            detections = {category: [] for category in self.object_categories.keys()}
            
            # Process each detection
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = results.names[cls]
                
                # Calculate distance
                distance = self.calculate_distance(box)
                
                # Categorize detection
                for category, info in self.object_categories.items():
                    if class_name in info['classes']:
                        detections[category].append((distance, conf))
                        
                        # Check if alert should be triggered
                        if distance > info['threshold']:
                            self.play_alert(info['sound'])
                            warning_text = f"WARNING: {class_name} detected!"
                            cv2.putText(frame, warning_text, (10, 30 + len(detections[category]) * 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, info['color'], 2)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), info['color'], 2)
                        
                        # Draw label with confidence and distance
                        label = f"{class_name} {conf:.2f} ({distance:.2f})"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, info['color'], 2)
            
            # Display detection counts
            y_offset = 30
            for category, count in detections.items():
                cv2.putText(frame, f"{category.title()}: {len(count)}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30
            
            return frame, results.boxes
            
        except Exception as e:
            print(f"Error in object detection: {str(e)}")
            return frame, []

    def calculate_distance(self, box):
        """
        Calculate approximate distance based on bounding box size
        Returns normalized distance (0 to 1)
        """
        try:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            box_height = y2 - y1
            return box_height / 720  # Normalize by frame height
        except Exception as e:
            print(f"Error calculating distance: {str(e)}")
            return 0.0

    def run_realtime(self, source=0):
        """
        Run real-time object detection on video stream
        """
        if not self.initialize_camera(source):
            print("Failed to initialize camera. Please check your camera connection.")
            return
        
        print("Starting enhanced real-time object detection...")
        print("Press 'q' to quit")
        print("Press 's' to save warning log")
        
        frame_count = 0
        start_time = time.time()
        consecutive_failures = 0
        max_failures = 5
        
        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print("Too many consecutive frame capture failures. Restarting camera...")
                        self.cap.release()
                        if not self.initialize_camera(source):
                            print("Failed to reinitialize camera. Exiting...")
                            break
                        consecutive_failures = 0
                    continue
                
                consecutive_failures = 0  # Reset failure counter on successful capture
                
                # Perform detection
                frame, _ = self.detect_objects(frame)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - start_time)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30 + len(self.object_categories) * 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    start_time = time.time()
                
                # Display frame
                cv2.imshow('Real-time Object Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_warning_log()
                
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                time.sleep(1)  # Wait before retrying
                continue
        
        # Cleanup
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def save_warning_log(self):
        """
        Save warning log to file
        """
        try:
            with open('warning_log.txt', 'w') as f:
                for warning in self.warning_log:
                    f.write(warning + '\n')
            print("Warning log saved to warning_log.txt")
        except Exception as e:
            print(f"Error saving warning log: {str(e)}")

def main():
    try:
        # Initialize detector
        detector = RealTimeObjectDetection()
        
        # Run real-time detection
        detector.run_realtime()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 