# 🚀 Day 8: Real-Time Object Detection using OpenCV & YOLOv8  

## 📌 Overview  
In this project, we implement **Real-Time Object Detection** using **OpenCV’s Haar Cascade** and **YOLOv8**. The goal is to compare traditional and deep learning-based object detection techniques in a live webcam feed.  

## 🚀 What You'll Learn  
✅ Implement **Haar Cascade** for face detection (traditional approach)  
✅ Use **YOLOv8** for object detection (deep learning-based approach)  
✅ Switch dynamically between Haar and YOLOv8 using hotkeys  
✅ Optimize real-time detection for better FPS  

## 📂 Project Structure  
```
Day-8-Object-Detection/
│── README.md  # Documentation  
│── Object detection.py  # Main Python script  
│── haarcascade_frontalface_default.xml  # Haar Cascade model file  
│── yolov8n.pt  # YOLOv8 pre-trained model  
```

## 📌 Why Haar Cascade & YOLOv8?  
📒 **Haar Cascade:** Works well for **face detection** but is less accurate for general objects.  
💻 **YOLOv8:** More powerful and can detect **multiple objects** in real-time.  

## 📊 Dataset & Models  
We use **pre-trained models** for both approaches:  
🔹 **Haar Cascade Model**: Download from OpenCV’s GitHub  
🔹 **YOLOv8 Model**: Use **ultralytics** to load the model  

## 🔧 Technologies Used  
🔹 Python  
🔹 OpenCV (for image processing)  
🔹 YOLOv8 (ultralytics package)  
🔹 NumPy  

## 📜 How to Run the Project?  

#### 1️⃣ Clone the repository  
```bash
git clone https://github.com/Aadhityan-Senthil/30-Day-Applied-AI-Challenge.git  
cd 30-Day-Applied-AI-Challenge/Day-8-Object-Detection-using-OpenCV
```

#### 2️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

#### 3️⃣ Download Models  
**Haar Cascade:**  
```bash
wget https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml
```
  
**YOLOv8:**  
```bash
pip install ultralytics  
```

#### 4️⃣ Run the script  
```bash
python Object detection.py
```

## 📈 Results & Analysis  
✅ **Best Detection Model:** YOLOv8 🏆  
✅ **Performance Metrics:**  
| Model | Accuracy | Speed | Robustness |  
|--------|------------|---------|-------------|  
| Haar Cascade | ⭐⭐ | ✅ Fast | ❌ Limited to faces |  
| YOLOv8 | ⭐⭐⭐⭐⭐ | ⏳ Slightly laggy | ✅ Detects multiple objects |  

##💡 **Key Takeaways:**  
🔹 **Haar Cascade** is lightweight but **less accurate**.  
🔹 **YOLOv8** is powerful but **computationally expensive**.  
🔹 **Optimizations** like **frame skipping & multi-threading** can improve performance.  

## 🔜 Next Steps  
🔹 Optimize **YOLOv8 performance** using GPU acceleration  
🔹 Try **frame skipping** and **multi-threading** for real-time improvements  
🔹 Experiment with **custom-trained YOLO models**  

## ⭐ Contribute & Connect!  
📢 Follow my **30-day AI challenge** & share your feedback! 🚀🔥  
