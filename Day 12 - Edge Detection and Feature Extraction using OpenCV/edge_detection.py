"""
Day 12: Edge Detection and Feature Extraction using OpenCV
30-Day AI Challenge

Demonstrates various edge detection and feature extraction techniques
including Canny, Sobel, Laplacian, and corner detection.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
import os

def download_sample_image():
    """Download a sample image for processing."""
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"
    filename = "sample_image.png"
    
    if not os.path.exists(filename):
        # Create a sample image with shapes instead
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Draw various shapes
        cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), -1)
        cv2.circle(img, (300, 100), 50, (0, 255, 0), -1)
        cv2.ellipse(img, (100, 300), (60, 40), 0, 0, 360, (255, 0, 0), -1)
        cv2.polygon(img, np.array([[250, 250], [350, 250], [300, 350]]), True, (255, 255, 0), -1)
        
        # Add some lines
        cv2.line(img, (200, 50), (200, 350), (128, 128, 128), 3)
        cv2.line(img, (50, 200), (350, 200), (128, 128, 128), 3)
        
        cv2.imwrite(filename, img)
        print(f"Created sample image: {filename}")
    
    return filename

def apply_edge_detection(image):
    """Apply various edge detection techniques."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1. Canny Edge Detection
    canny = cv2.Canny(blurred, 50, 150)
    
    # 2. Sobel Edge Detection
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)
    
    # 3. Laplacian Edge Detection
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = np.uint8(np.abs(laplacian) / np.abs(laplacian).max() * 255)
    
    # 4. Prewitt Edge Detection (manual)
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    prewitt_x = cv2.filter2D(blurred, cv2.CV_64F, kernel_x)
    prewitt_y = cv2.filter2D(blurred, cv2.CV_64F, kernel_y)
    prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)
    prewitt = np.uint8(prewitt / prewitt.max() * 255)
    
    return {
        'original': image,
        'grayscale': gray,
        'canny': canny,
        'sobel': sobel_combined,
        'laplacian': laplacian,
        'prewitt': prewitt
    }

def detect_corners(image):
    """Detect corners using Harris and Shi-Tomasi methods."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    # Harris Corner Detection
    harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    harris_img = image.copy()
    harris_img[harris > 0.01 * harris.max()] = [0, 0, 255]
    
    # Shi-Tomasi Corner Detection
    gray_uint8 = np.uint8(gray)
    corners = cv2.goodFeaturesToTrack(gray_uint8, maxCorners=100, qualityLevel=0.01, minDistance=10)
    shi_tomasi_img = image.copy()
    
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(shi_tomasi_img, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    return harris_img, shi_tomasi_img

def detect_contours(image):
    """Detect and draw contours."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    
    print(f"Found {len(contours)} contours")
    return contour_img, contours

def extract_hog_features(image):
    """Extract HOG (Histogram of Oriented Gradients) features."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    
    # HOG parameters
    win_size = (128, 128)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    features = hog.compute(gray)
    
    print(f"HOG feature vector length: {len(features)}")
    return features

def plot_results(edge_results, harris_img, shi_tomasi_img, contour_img):
    """Plot all results in a grid."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Row 1: Original, Grayscale, Canny
    axes[0, 0].imshow(cv2.cvtColor(edge_results['original'], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(edge_results['grayscale'], cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(edge_results['canny'], cmap='gray')
    axes[0, 2].set_title('Canny Edge Detection')
    axes[0, 2].axis('off')
    
    # Row 2: Sobel, Laplacian, Prewitt
    axes[1, 0].imshow(edge_results['sobel'], cmap='gray')
    axes[1, 0].set_title('Sobel Edge Detection')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(edge_results['laplacian'], cmap='gray')
    axes[1, 1].set_title('Laplacian Edge Detection')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(edge_results['prewitt'], cmap='gray')
    axes[1, 2].set_title('Prewitt Edge Detection')
    axes[1, 2].axis('off')
    
    # Row 3: Harris, Shi-Tomasi, Contours
    axes[2, 0].imshow(cv2.cvtColor(harris_img, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title('Harris Corner Detection')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(cv2.cvtColor(shi_tomasi_img, cv2.COLOR_BGR2RGB))
    axes[2, 1].set_title('Shi-Tomasi Corner Detection')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    axes[2, 2].set_title('Contour Detection')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('feature_extraction_results.png', dpi=150)
    plt.close()
    print("Results saved as 'feature_extraction_results.png'")

def main():
    print("=" * 50)
    print("Day 12: Edge Detection & Feature Extraction")
    print("=" * 50)
    
    # Load or create sample image
    print("\n[1] Loading sample image...")
    image_path = download_sample_image()
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error loading image!")
        return
    
    # Edge Detection
    print("\n[2] Applying edge detection techniques...")
    edge_results = apply_edge_detection(image)
    
    # Corner Detection
    print("\n[3] Detecting corners...")
    harris_img, shi_tomasi_img = detect_corners(image)
    
    # Contour Detection
    print("\n[4] Detecting contours...")
    contour_img, contours = detect_contours(image)
    
    # HOG Features
    print("\n[5] Extracting HOG features...")
    hog_features = extract_hog_features(image)
    
    # Plot all results
    print("\n[6] Generating visualization...")
    plot_results(edge_results, harris_img, shi_tomasi_img, contour_img)
    
    print("\n" + "=" * 50)
    print("Day 12 Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
