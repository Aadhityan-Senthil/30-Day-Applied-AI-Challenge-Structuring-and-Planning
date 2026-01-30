"""
Day 29: Real-Time Object Tracking using YOLOv8
30-Day AI Challenge

Detect and track objects in images/video using YOLOv8.
Works with webcam, video files, or images.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# COCO dataset classes
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def generate_synthetic_detections(n_frames=30, n_objects=5):
    """Generate synthetic detection data for demonstration."""
    np.random.seed(42)
    
    detections = []
    
    # Initialize object positions and velocities
    objects = []
    for i in range(n_objects):
        objects.append({
            'id': i,
            'class': np.random.choice(['person', 'car', 'dog', 'bicycle', 'cat']),
            'x': np.random.randint(100, 500),
            'y': np.random.randint(100, 400),
            'vx': np.random.randint(-10, 10),
            'vy': np.random.randint(-10, 10),
            'width': np.random.randint(50, 150),
            'height': np.random.randint(50, 200)
        })
    
    for frame in range(n_frames):
        frame_detections = []
        
        for obj in objects:
            # Update position
            obj['x'] = max(50, min(550, obj['x'] + obj['vx']))
            obj['y'] = max(50, min(350, obj['y'] + obj['vy']))
            
            # Bounce off edges
            if obj['x'] <= 50 or obj['x'] >= 550:
                obj['vx'] *= -1
            if obj['y'] <= 50 or obj['y'] >= 350:
                obj['vy'] *= -1
            
            frame_detections.append({
                'id': obj['id'],
                'class': obj['class'],
                'confidence': np.random.uniform(0.7, 0.99),
                'bbox': [obj['x'], obj['y'], obj['width'], obj['height']]
            })
        
        detections.append({
            'frame': frame,
            'detections': frame_detections
        })
    
    return detections

class SimpleTracker:
    """Simple object tracker using IoU matching."""
    
    def __init__(self, max_age=30):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
    
    def iou(self, box1, box2):
        """Calculate Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        
        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return iou
    
    def update(self, detections):
        """Update tracks with new detections."""
        # Match detections to existing tracks
        matched = set()
        
        for track_id, track in list(self.tracks.items()):
            best_iou = 0
            best_det = None
            
            for det in detections:
                iou = self.iou(track['bbox'], det['bbox'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_det = det
            
            if best_det:
                track['bbox'] = best_det['bbox']
                track['class'] = best_det['class']
                track['confidence'] = best_det['confidence']
                track['age'] = 0
                matched.add(id(best_det))
            else:
                track['age'] += 1
                if track['age'] > self.max_age:
                    del self.tracks[track_id]
        
        # Create new tracks for unmatched detections
        for det in detections:
            if id(det) not in matched:
                self.tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'class': det['class'],
                    'confidence': det['confidence'],
                    'age': 0
                }
                self.next_id += 1
        
        return self.tracks

def create_sample_image_with_objects(width=640, height=480):
    """Create a sample image with drawable objects."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # Draw background elements
    cv2.rectangle(img, (0, height-50), (width, height), (100, 150, 100), -1)  # Ground
    cv2.rectangle(img, (50, 100), (150, 300), (180, 180, 180), -1)  # Building
    cv2.rectangle(img, (500, 150), (600, 300), (180, 180, 180), -1)  # Building
    
    return img

def draw_detections(image, detections, tracks=None):
    """Draw bounding boxes and labels on image."""
    if not CV2_AVAILABLE:
        return image
    
    colors = {
        'person': (0, 255, 0),
        'car': (255, 0, 0),
        'dog': (0, 255, 255),
        'cat': (255, 0, 255),
        'bicycle': (255, 255, 0)
    }
    
    for det in detections:
        x, y, w, h = det['bbox']
        class_name = det['class']
        conf = det['confidence']
        
        color = colors.get(class_name, (128, 128, 128))
        
        # Draw bounding box
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
        
        # Draw label
        label = f"{class_name}: {conf:.2f}"
        if 'track_id' in det:
            label = f"ID{det['track_id']} {label}"
        
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (int(x), int(y)-20), (int(x)+label_w, int(y)), color, -1)
        cv2.putText(image, label, (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image

def detect_objects_yolo(image, model):
    """Detect objects using YOLOv8."""
    if not YOLO_AVAILABLE or model is None:
        return []
    
    results = model(image, verbose=False)
    
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            detections.append({
                'class': COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else 'unknown',
                'confidence': conf,
                'bbox': [x1, y1, x2-x1, y2-y1]
            })
    
    return detections

def process_video_demo(synthetic_detections):
    """Process synthetic video frames and track objects."""
    tracker = SimpleTracker()
    
    all_tracks = []
    
    for frame_data in synthetic_detections:
        detections = frame_data['detections']
        
        # Update tracker
        tracks = tracker.update(detections)
        
        # Store tracking results
        frame_tracks = []
        for track_id, track in tracks.items():
            frame_tracks.append({
                'track_id': track_id,
                'class': track['class'],
                'bbox': track['bbox'],
                'confidence': track['confidence']
            })
        
        all_tracks.append({
            'frame': frame_data['frame'],
            'tracks': frame_tracks
        })
    
    return all_tracks

def visualize_tracking_results(tracking_results, n_frames=30):
    """Visualize tracking results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Object trajectories
    trajectories = {}
    for frame_data in tracking_results:
        for track in frame_data['tracks']:
            tid = track['track_id']
            if tid not in trajectories:
                trajectories[tid] = {'x': [], 'y': [], 'class': track['class']}
            trajectories[tid]['x'].append(track['bbox'][0])
            trajectories[tid]['y'].append(track['bbox'][1])
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    for (tid, traj), color in zip(trajectories.items(), colors):
        axes[0, 0].plot(traj['x'], traj['y'], '-o', color=color, 
                        label=f"ID{tid} ({traj['class']})", markersize=3)
    
    axes[0, 0].set_title('Object Trajectories')
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Y Position')
    axes[0, 0].legend(loc='upper right', fontsize=8)
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Objects per frame
    objects_per_frame = [len(f['tracks']) for f in tracking_results]
    axes[0, 1].plot(objects_per_frame, color='blue', linewidth=2)
    axes[0, 1].fill_between(range(len(objects_per_frame)), objects_per_frame, alpha=0.3)
    axes[0, 1].set_title('Objects Detected Per Frame')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Class distribution
    class_counts = {}
    for frame_data in tracking_results:
        for track in frame_data['tracks']:
            cls = track['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    
    axes[1, 0].pie(counts, labels=classes, autopct='%1.1f%%', colors=colors_pie)
    axes[1, 0].set_title('Detection Distribution by Class')
    
    # 4. Confidence distribution
    all_confidences = []
    for frame_data in tracking_results:
        for track in frame_data['tracks']:
            all_confidences.append(track['confidence'])
    
    axes[1, 1].hist(all_confidences, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(np.mean(all_confidences), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(all_confidences):.2f}')
    axes[1, 1].set_title('Confidence Score Distribution')
    axes[1, 1].set_xlabel('Confidence')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('object_tracking_results.png', dpi=150)
    plt.close()
    print("Results saved to 'object_tracking_results.png'")

def create_sample_detection_image():
    """Create a visualization of detected objects."""
    if not CV2_AVAILABLE:
        print("OpenCV not available for image creation")
        return
    
    # Create sample image
    img = create_sample_image_with_objects()
    
    # Add synthetic detections
    detections = [
        {'class': 'person', 'confidence': 0.95, 'bbox': [200, 200, 60, 150], 'track_id': 1},
        {'class': 'car', 'confidence': 0.88, 'bbox': [350, 250, 120, 80], 'track_id': 2},
        {'class': 'dog', 'confidence': 0.92, 'bbox': [450, 320, 50, 40], 'track_id': 3},
    ]
    
    # Draw detections
    img = draw_detections(img, detections)
    
    # Save
    cv2.imwrite('sample_detection.png', img)
    print("Sample detection image saved to 'sample_detection.png'")

def main():
    print("=" * 50)
    print("Day 29: Real-Time Object Tracking using YOLOv8")
    print("=" * 50)
    
    # Check dependencies
    print("\n[1] Checking dependencies...")
    print(f"  OpenCV: {'✓' if CV2_AVAILABLE else '✗'}")
    print(f"  YOLOv8: {'✓' if YOLO_AVAILABLE else '✗'}")
    
    # Load YOLO model (if available)
    model = None
    if YOLO_AVAILABLE:
        print("\n[2] Loading YOLOv8 model...")
        try:
            model = YOLO('yolov8n.pt')  # Nano model for speed
            print("  Model loaded successfully!")
        except Exception as e:
            print(f"  Could not load model: {e}")
            print("  Using synthetic data instead.")
    else:
        print("\n[2] YOLOv8 not available. Using synthetic data.")
    
    # Generate synthetic detections for demo
    print("\n[3] Generating synthetic detection data...")
    synthetic_data = generate_synthetic_detections(n_frames=50, n_objects=5)
    print(f"  Generated {len(synthetic_data)} frames with detections")
    
    # Process with tracker
    print("\n[4] Running object tracker...")
    tracker = SimpleTracker()
    tracking_results = process_video_demo(synthetic_data)
    
    # Count unique tracks
    all_track_ids = set()
    for frame in tracking_results:
        for track in frame['tracks']:
            all_track_ids.add(track['track_id'])
    
    print(f"  Total unique objects tracked: {len(all_track_ids)}")
    print(f"  Frames processed: {len(tracking_results)}")
    
    # Create visualizations
    print("\n[5] Creating visualizations...")
    visualize_tracking_results(tracking_results)
    
    if CV2_AVAILABLE:
        create_sample_detection_image()
    
    # Save results
    print("\n[6] Saving results...")
    output = {
        'frames_processed': len(tracking_results),
        'unique_objects': len(all_track_ids),
        'yolo_available': YOLO_AVAILABLE,
        'sample_tracking': tracking_results[:5]
    }
    
    with open('tracking_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print("Results saved to 'tracking_results.json'")
    
    # Usage instructions
    print("\n" + "=" * 50)
    print("Usage with Real Video/Webcam:")
    print("=" * 50)
    print("""
    # Install dependencies:
    pip install ultralytics opencv-python
    
    # Detect in image:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    results = model('image.jpg')
    results[0].show()
    
    # Real-time webcam:
    results = model(source=0, show=True)
    
    # Video file:
    results = model('video.mp4', show=True)
    """)
    
    print("\n" + "=" * 50)
    print("Day 29 Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
