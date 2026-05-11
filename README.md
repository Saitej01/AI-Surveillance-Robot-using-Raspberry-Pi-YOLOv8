# 🤖 Autonomous Surveillance Robot

An intelligent AI-powered surveillance robot built using Raspberry Pi, YOLOv8, OpenCV, and OCR for real-time person detection, number plate recognition, obstacle avoidance, and live video streaming.

---

## 📌 Overview

The **Autonomous Surveillance Robot** is a smart mobile monitoring system capable of:

- Detecting humans in real time
- Detecting and recognizing vehicle number plates
- Extracting text using OCR
- Avoiding obstacles autonomously
- Streaming live video over a web interface
- Saving detected images for evidence and analysis

The project combines **Embedded Systems**, **Computer Vision**, **Deep Learning**, and **IoT** technologies into a low-cost autonomous security platform.

---

# 🚀 Features

✅ Real-Time Person Detection  
✅ Vehicle Number Plate Detection  
✅ OCR-Based Text Extraction  
✅ Flask Web Streaming  
✅ Autonomous Obstacle Avoidance  
✅ Automatic Evidence Image Saving  
✅ Remote Monitoring through Browser  
✅ Multithreaded Real-Time Processing  

---

# 🛠️ Hardware Components

| Component | Description |
|---|---|
| Raspberry Pi 4 | Main processing unit |
| Raspberry Pi Camera | Captures live video |
| HC-SR04 Ultrasonic Sensor | Obstacle detection |
| L298N Motor Driver | Controls DC motors |
| DC Motors | Robot movement |
| SG90 Servo Motor | Camera/sensor positioning |
| Robot Chassis | Mechanical structure |
| Battery Pack | Power supply |

---

# 💻 Software & Technologies Used

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- Flask
- Tesseract OCR
- NumPy
- Threading
- Raspberry Pi OS

---

# 🧠 AI Models

## 👤 Person Detection Model
- Model File: `person.pt`
- Architecture: YOLOv8m
- Detects humans in real time
- Trained on custom datasets with different lighting conditions and backgrounds

## 🚗 Number Plate Detection Model
- Model File: `plate.pt`
- Architecture: YOLOv8m
- Detects Indian vehicle number plates
- Integrated with OCR for text extraction

---

# ⚙️ Working Principle

1. Camera captures live video frames
2. YOLOv8 detects:
   - Persons
   - Vehicle number plates
3. OCR extracts number plate text
4. Images are saved automatically
5. Ultrasonic sensor detects nearby obstacles
6. Flask streams processed video to browser
7. L298N controls robot movement

---

# 🔌 GPIO Connections

| Function | BOARD Pin | BCM Pin |
|---|---|---|
| Left Motor EN | 12 | GPIO18 |
| Left Motor IN1 | 16 | GPIO23 |
| Left Motor IN2 | 38 | GPIO20 |
| Right Motor EN | 35 | GPIO19 |
| Right Motor IN3 | 13 | GPIO27 |
| Right Motor IN4 | 37 | GPIO26 |
| Front Ultrasonic TRIG | 33 | GPIO13 |
| Front Ultrasonic ECHO | 29 | GPIO5 |

---

# 📸 Functionalities

## 👤 Person Detection
- Detects multiple persons simultaneously
- Works in indoor and outdoor environments
- Saves detected images automatically

## 🚗 Number Plate Recognition
- Detects Indian number plates
- Extracts alphanumeric text using Tesseract OCR
- Saves cropped plate images and logs

## 📡 Live Video Streaming
- Flask-based MJPEG streaming
- Accessible from laptop/mobile browser
- Real-time bounding boxes and labels

## 🚧 Obstacle Avoidance
- Ultrasonic sensor measures object distance
- Robot stops or changes direction automatically

---

# 📊 Performance

| Parameter | Performance |
|---|---|
| FPS | 15–22 FPS |
| YOLO Inference Time | 25–40 ms |
| OCR Accuracy | 80–95% |
| Person Detection Range | 6–8 meters |
| Obstacle Detection Range | Up to 4 meters |

---

# 📥 Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Autonomous-Surveillance-Robot.git
cd Autonomous-Surveillance-Robot
```


## Install Tesseract OCR

```bash
sudo apt update
sudo apt install tesseract-ocr
```

## 4️⃣ Run the Project in raspberry_pi

```bash
python3 Raspberry_pi_code.py
```

---

# 🌐 Access Live Stream

Open browser and visit:

```bash
http://<RaspberryPi-IP>:5000
```

Example:

```bash
http://192.168.1.5:5000
```

---

# 🎯 Applications

- Smart Surveillance
- Campus Security
- Parking Monitoring
- Traffic Monitoring
- Industrial Monitoring
- Restricted Area Security
- Home Security

---

# 🔮 Future Improvements

- Cloud Storage Integration
- GPS Tracking
- Night Vision Support
- SLAM-Based Navigation
- TensorRT / ONNX Optimization
- Advanced YOLO Models (YOLOv9 / YOLOv10)
