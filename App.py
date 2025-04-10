import sys
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QComboBox, QVBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

# Load YOLO model
model = YOLO("yolov8n.pt")

target_id = None  # Selected object ID
tracked_objects = {}  # Stores detected objects
object_names = {}  # Stores object names
tracking_initialized = False  # Track status
snapshot_image = None  # Captured image


class VideoThread(QThread):
    update_frame = pyqtSignal(QImage)
    update_dropdown = pyqtSignal(dict)
    update_snapshot = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        global tracking_initialized, target_id
        tracker = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # If tracking is enabled, follow selected object
            if target_id and target_id in tracked_objects:
                x1, y1, x2, y2 = tracked_objects[target_id]

                # Initialize tracker only once
                if not tracking_initialized:
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                    tracking_initialized = True

                # Track object
                if tracking_initialized and tracker:
                    success, box = tracker.update(frame)
                    if success:
                        x, y, w, h = map(int, box)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        cv2.putText(frame, f"Tracking: {object_names.get(target_id, 'Unknown')}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            q_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.update_frame.emit(q_img)

        cap.release()

    def detect_objects(self, frame):
        """Detect objects from snapshot and update dropdown."""
        global tracked_objects, object_names, snapshot_image
        results = model(frame, stream=True)
        tracked_objects.clear()
        object_names.clear()

        for detection in results:
            boxes = detection.boxes.xyxy.cpu().numpy()
            class_ids = detection.boxes.cls.cpu().numpy().astype(int)
            names = detection.names

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                obj_id = f"{class_ids[i]}_{i}"
                tracked_objects[obj_id] = (x1, y1, x2, y2)
                object_names[obj_id] = names[class_ids[i]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(frame, f"{names[class_ids[i]]} ({obj_id})", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        snapshot_image = frame
        rgb_snapshot = cv2.cvtColor(snapshot_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_snapshot.shape
        q_img = QImage(rgb_snapshot.data, w, h, ch * w, QImage.Format_RGB888)
        self.update_snapshot.emit(q_img)

        self.update_dropdown.emit(object_names)


class ObjectTrackerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.video_thread = VideoThread()
        self.video_thread.update_frame.connect(self.update_image)
        self.video_thread.update_dropdown.connect(self.update_dropdown)
        self.video_thread.update_snapshot.connect(self.update_snapshot_image)

    def initUI(self):
        self.setWindowTitle("Object Tracking System")
        self.setGeometry(100, 100, 700, 600)

        self.layout = QVBoxLayout()

        self.capture_button = QPushButton("Step 1: Capture Image & Detect Objects")
        self.capture_button.clicked.connect(self.capture_snapshot)
        self.capture_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #f39c12; color: white; border-radius: 5px;")
        self.layout.addWidget(self.capture_button)

        self.snapshot_label = QLabel("Captured Image & Detection:")
        self.snapshot_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.snapshot_label)

        self.snapshot_display = QLabel()
        self.snapshot_display.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.snapshot_display)

        self.label = QLabel("Step 2: Select Object to Track")
        self.layout.addWidget(self.label)

        self.comboBox = QComboBox()
        self.comboBox.activated[str].connect(self.set_target)
        self.layout.addWidget(self.comboBox)

        self.start_button = QPushButton("Step 3: Start Tracking")
        self.start_button.clicked.connect(self.start_tracking)
        self.start_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #3498db; color: white; border-radius: 5px;")
        self.layout.addWidget(self.start_button)

        self.video_label = QLabel("Live Video Tracking:")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_display)

        self.setLayout(self.layout)

    def capture_snapshot(self):
        """Step 1: Capture image and detect objects."""
        global snapshot_image
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.video_thread.detect_objects(frame)

    def set_target(self, selected_text):
        """Step 2: Fix the selected target when clicked."""
        global target_id, tracking_initialized
        if "(" in selected_text and ")" in selected_text:
            target_id = selected_text.split("(")[1].split(")")[0]
            tracking_initialized = False  # Reset tracking
            print(f"Target locked: {target_id}")
        else:
            target_id = None
            print("No target selected")

    def update_image(self, image):
        """Step 3: Update the live tracking feed."""
        self.video_display.setPixmap(QPixmap.fromImage(image))

    def update_snapshot_image(self, image):
        """Update the captured snapshot with object detection."""
        self.snapshot_display.setPixmap(QPixmap.fromImage(image))

    def update_dropdown(self, objects):
        """Update dropdown list with detected objects."""
        self.comboBox.clear()
        self.comboBox.addItem("None")
        for obj_id, obj_name in objects.items():
            self.comboBox.addItem(f"{obj_name} ({obj_id})")

    def start_tracking(self):
        """Start tracking after object selection."""
        self.video_thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectTrackerApp()
    window.show()
    sys.exit(app.exec_())
