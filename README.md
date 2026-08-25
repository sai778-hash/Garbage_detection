# 🗑️ AI-Based Garbage Detection and Classification Using YOLOv8

An AI-powered computer vision system for detecting and classifying garbage in real time using **YOLOv8, Python, OpenCV, and PyTorch**.

## 📌 Project Overview

Improper waste disposal is a major environmental problem. Manual identification and segregation of waste is time-consuming and difficult to scale.

This project uses **YOLOv8-based deep learning and computer vision** to identify garbage from images and real-time webcam input. The system can locate objects using a YOLOv8 detection model and classify the detected regions using a custom-trained garbage classification model.

The project can be used as a foundation for **smart waste management, automated garbage monitoring, and waste segregation systems**.

## 🎯 Objectives

* Detect garbage objects using computer vision.
* Classify detected garbage into predefined categories.
* Perform real-time garbage detection using a webcam.
* Reduce the need for manual waste identification.
* Provide a foundation for automated waste segregation systems.
* Demonstrate the application of deep learning in smart waste management.

## 🧠 Technologies Used

* **Python**
* **YOLOv8**
* **PyTorch**
* **OpenCV**
* **Ultralytics**
* **NumPy**
* **Computer Vision**
* **Deep Learning**

## 📂 Project Structure

```text
Garbage_detection/
│
├── dataset/
│   ├── train/
│   └── val/
│
├── runs/
│   └── classify/
│
├── test.jpg
│
├── webcam_classify.py
├── webcam_detect.py
│
├── yolov8n.pt
├── yolov8n-cls.pt
│
└── README.md
```

## 🔍 System Architecture

```text
                Input Image / Webcam
                         │
                         ▼
                  Image Acquisition
                         │
                         ▼
                   YOLOv8 Detection
                         │
                         ▼
                  Object / Region
                     Detection
                         │
                         ▼
                 Image Cropping
                         │
                         ▼
              Custom YOLOv8 Classifier
                         │
                         ▼
                Garbage Classification
                         │
                         ▼
                 Detection Result
                         │
                         ▼
              Bounding Box + Class
```

## ⚙️ Working Principle

The system follows a two-stage approach:

### 1. Object Detection

A YOLOv8 detection model identifies regions of interest in the input image or webcam frame.

### 2. Object Classification

The detected region is cropped and passed to a custom-trained YOLOv8 classification model.

The classifier predicts the garbage category and confidence score.

### 3. Result Visualization

The predicted class, confidence score, and bounding box are displayed on the webcam frame.

## 📊 Dataset

The dataset is organized into training and validation directories:

```text
dataset/
├── train/
└── val/
```

The training data is used to train the garbage classification model, while the validation data is used to evaluate the model during training.

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/sai-1186/Garbage_detection.git
cd Garbage_detection
```

Install the required Python packages:

```bash
pip install ultralytics opencv-python numpy
```

## 🏋️ Model Training

The classification model can be trained using Ultralytics YOLOv8.

Example:

```python
from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="dataset",
    epochs=50,
    imgsz=224
)
```

After training, the best model is generally stored under:

```text
runs/classify/train/weights/best.pt
```

## 📷 Real-Time Classification

To perform real-time classification using the webcam:

```bash
python webcam_classify.py
```

The webcam captures frames and the trained model predicts the garbage category.

## 🎥 Real-Time Detection + Classification

The `webcam_detect.py` program combines detection and classification.

Run:

```bash
python webcam_detect.py
```

The program uses:

* `yolov8n.pt` for object detection.
* `runs/classify/train/weights/best.pt` for custom garbage classification.
* OpenCV for webcam input and result visualization.

The detection program also supports configurable parameters such as camera index, detection confidence, classification confidence, image size, and CPU/GPU device.

Example:

```bash
python webcam_detect.py --camera 0 --device cpu
```

For a system with a compatible CUDA GPU:

```bash
python webcam_detect.py --camera 0 --device 0
```

## 📈 Output

The system provides:

* Real-time webcam detection.
* Bounding boxes around detected regions.
* Predicted garbage class.
* Classification confidence.
* Real-time visual feedback.

## 🌱 Applications

* Smart garbage bins
* Automated waste segregation
* Recycling plants
* Smart city waste monitoring
* Environmental monitoring
* Garbage collection systems
* Educational AI and computer vision projects

## 🔮 Future Improvements

* Add more garbage categories.
* Increase the size and diversity of the dataset.
* Improve model accuracy.
* Deploy the model on Raspberry Pi or edge devices.
* Integrate ESP32-CAM for image acquisition.
* Add IoT-based monitoring.
* Connect the system to a mobile/web dashboard.
* Automatically separate recyclable and non-recyclable waste.
* Deploy the model using TensorFlow Lite or another edge-optimized format.

## 👨‍💻 Author

**Sai Kiran**

GitHub:
https://github.com/sai-1186

## 📜 License

This project is intended for educational and research purposes.
