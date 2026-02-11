# 🚗 Car Detection Using YOLOv8

## 📌 Project Overview
This project implements a real-time Car Detection System using YOLOv8 (Deep Learning – Computer Vision).  
The system processes traffic videos frame by frame and detects moving vehicles using a pretrained object detection model.  
Detected cars are highlighted with bounding boxes and the output video is saved automatically.

---

## 🎯 Features
- Real-time vehicle detection
- Bounding box visualization
- Class-based filtering (Car, Bus, Truck)
- CPU-optimized inference
- Output video generation

---

## 🛠️ Tech Stack
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Deep Learning
- Computer Vision

---

## 📂 Project Structure

Car-Detection-YOLOv8/
│── car_detection_video.py
│── input_video.mp4
│── output_detected.mp4
│── README.md

---

## ⚙️ Installation

1️⃣ Clone Repository

git clone https://github.com/your-username/Car-Detection-YOLOv8.git
cd Car-Detection-YOLOv8

2️⃣ Install Dependencies

pip install ultralytics opencv-python

---

## ▶️ How to Run

python car_detection_video.py

Press "Q" to stop detection.

Output video will be saved as:
output_detected.mp4

---

