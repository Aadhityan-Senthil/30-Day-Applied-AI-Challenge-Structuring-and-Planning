import cv2
import numpy as np
from ultralytics import YOLO

# Load Haar Cascade for face detection
haar_cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Load YOLOv8 model (Pretrained on COCO dataset)
yolo_model = YOLO("yolov8n.pt")

# Start webcam capture
cap = cv2.VideoCapture(0)  # 0 for default webcam

if not cap.isOpened():
    print("Error: Unable to access the webcam!")
    exit()

# Choose the detection method (Haar Cascade or YOLOv8)
use_haar = True  # Change to False if you want YOLOv8 detection

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame!")
        break

    if use_haar:
        # Convert frame to grayscale (Haar works better in grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw bounding boxes around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(frame, "Haar Cascade: Face Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    else:
        # Perform YOLOv8 object detection
        results = yolo_model(frame)

        # Get annotated frame
        frame = results[0].plot()

        cv2.putText(frame, "YOLOv8: Object Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Real-Time Object Detection", frame)

    # Press 'q' to exit, 'h' for Haar, 'y' for YOLO
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("h"):
        use_haar = True
    elif key == ord("y"):
        use_haar = False

# Release resources
cap.release()
cv2.destroyAllWindows()
