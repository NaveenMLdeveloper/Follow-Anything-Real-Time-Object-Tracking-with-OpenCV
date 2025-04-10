# 🚁 Follow Anything: Real-Time Object Tracking with OpenCV and YOLO
![image](https://github.com/user-attachments/assets/49fc5767-02c1-405f-a9f6-a7e630f3f9e4)

## 📘 Project Overview
This project demonstrates an interactive real-time object tracking system, where the user can **mark an object** via webcam and the system will continuously **track and follow it** using **OpenCV**, **YOLOv8**, and **PyQt5**. This concept is designed to simulate how **drones can follow selected objects**, similar to “Follow Anything” applications.

---

## 🎯 Objective
To build a desktop-based system that:
- Captures webcam input
- Detects all visible objects using YOLOv8
- Allows user to **select an object** to track
- Tracks the selected object in real-time
- Simulates "follow mode" similar to drone-based tracking systems

---

## 🎥 Features
- 📸 Live webcam feed
- 🎯 Object detection using YOLOv8
- 🧠 Object selection through dropdown menu
- 🛰️ Real-time tracking with OpenCV's CSRT tracker
- 🎨 PyQt5 GUI with snapshot and live view panels


---

## 🛠️ Technologies Used
- 🐍 Python
- 🎯 OpenCV
- 🤖 YOLOv8 (via Ultralytics)
- 🪟 PyQt5 (for UI)
- 🔤 NumPy

---

## 🚀 How It Works

1. **Start the Application**: Launch the PyQt5 interface.
2. **Capture Frame**: Click “Step 1” to take a snapshot and detect objects.
3. **Select Object**: Choose the object from the dropdown list (e.g., person, dog).
4. **Start Tracking**: The tracker locks onto the selected object and follows it in real-time.


