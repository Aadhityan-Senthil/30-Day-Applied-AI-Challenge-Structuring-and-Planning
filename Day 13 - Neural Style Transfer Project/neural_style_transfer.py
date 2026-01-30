"""
Day 13: Neural Style Transfer Project
30-Day AI Challenge

Apply the artistic style of one image to the content of another
using deep learning and VGG19 network.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Disable GPU if not available
tf.config.set_visible_devices([], 'GPU')

def load_and_process_image(image_path, max_dim=512):
    """Load and preprocess image for VGG19."""
    img = Image.open(image_path)
    
    # Resize while maintaining aspect ratio
    scale = max_dim / max(img.size)
    new_size = tuple(int(dim * scale) for dim in img.size)
    img = img.resize(new_size, Image.LANCZOS)
    
    img = np.array(img)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Preprocess for VGG19
    img = tf.keras.applications.vgg19.preprocess_input(img)
    
    return img

def deprocess_image(processed_img):
    """Convert processed image back to displayable format."""
    img = processed_img.copy()
    
    if len(img.shape) == 4:
        img = np.squeeze(img, 0)
    
    # Reverse VGG preprocessing
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]  # BGR to RGB
    
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def create_sample_images():
    """Create sample content and style images."""
    # Content image: Simple scene
    content = np.ones((256, 256, 3), dtype=np.uint8) * 200
    
    # Draw a house
    content[100:200, 80:180] = [150, 100, 50]  # House body
    content[60:110, 80:180] = [100, 50, 50]    # Roof area
    
    # Draw triangle roof
    for i in range(50):
        content[60+i, 80+i:180-i] = [139, 69, 19]
    
    # Door
    content[140:200, 115:145] = [80, 40, 20]
    
    # Windows
    content[120:145, 90:110] = [135, 206, 235]
    content[120:145, 150:170] = [135, 206, 235]
    
    # Sun
    for y in range(256):
        for x in range(256):
            if (x - 220)**2 + (y - 40)**2 < 30**2:
                content[y, x] = [255, 255, 0]
    
    # Sky gradient
    for y in range(60):
        content[y, :] = [135 + y, 206, 235]
    
    # Grass
    content[200:, :] = [34, 139, 34]
    
    Image.fromarray(content).save('content_image.jpg')
    
    # Style image: Abstract colorful pattern
    style = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Create swirling pattern
    for y in range(256):
        for x in range(256):
            r = int(127 + 127 * np.sin(x / 20 + y / 30))
            g = int(127 + 127 * np.sin(y / 25 + x / 15))
            b = int(127 + 127 * np.cos((x + y) / 35))
            style[y, x] = [r, g, b]
    
    Image.fromarray(style).save('style_image.jpg')
    
    print("Created sample images: content_image.jpg, style_image.jpg")
    return 'content_image.jpg', 'style_image.jpg'

def get_model():
    """Create a model that returns style and content features."""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    # Content layer
    content_layers = ['block5_conv2']
    
    # Style layers
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    
    outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
    
    model = tf.keras.Model([vgg.input], outputs)
    return model, style_layers, content_layers

def gram_matrix(tensor):
    """Calculate Gram matrix for style representation."""
    channels = int(tensor.shape[-1])
    a = tf.reshape(tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def compute_loss(model, style_layers, content_layers, 
                 style_features, content_features, 
                 generated_image, style_weight=1e-2, content_weight=1e4):
    """Compute total loss for style transfer."""
    outputs = model(generated_image)
    
    style_outputs = outputs[:len(style_layers)]
    content_outputs = outputs[len(style_layers):]
    
    # Style loss
    style_loss = 0
    for target, output in zip(style_features, style_outputs):
        style_loss += tf.reduce_mean((gram_matrix(output) - gram_matrix(target))**2)
    style_loss *= style_weight / len(style_layers)
    
    # Content loss
    content_loss = 0
    for target, output in zip(content_features, content_outputs):
        content_loss += tf.reduce_mean((output - target)**2)
    content_loss *= content_weight / len(content_layers)
    
    total_loss = style_loss + content_loss
    return total_loss, style_loss, content_loss

@tf.function
def train_step(model, style_layers, content_layers, 
               style_features, content_features, 
               generated_image, optimizer):
    """Single optimization step."""
    with tf.GradientTape() as tape:
        loss, style_loss, content_loss = compute_loss(
            model, style_layers, content_layers,
            style_features, content_features, generated_image
        )
    
    grad = tape.gradient(loss, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    
    # Clip values
    generated_image.assign(tf.clip_by_value(generated_image, -150, 150))
    
    return loss, style_loss, content_loss

def style_transfer(content_path, style_path, iterations=100):
    """Perform neural style transfer."""
    print("Loading images...")
    content_image = load_and_process_image(content_path)
    style_image = load_and_process_image(style_path)
    
    print("Building model...")
    model, style_layers, content_layers = get_model()
    
    # Extract features
    style_outputs = model(style_image)[:len(style_layers)]
    content_outputs = model(content_image)[len(style_layers):]
    
    # Initialize generated image with content
    generated_image = tf.Variable(content_image, dtype=tf.float32)
    
    # Optimizer
    optimizer = tf.optimizers.Adam(learning_rate=5.0)
    
    print(f"Starting style transfer ({iterations} iterations)...")
    
    results = []
    for i in range(iterations):
        loss, style_loss, content_loss = train_step(
            model, style_layers, content_layers,
            style_outputs, content_outputs,
            generated_image, optimizer
        )
        
        if (i + 1) % 20 == 0:
            print(f"Iteration {i+1}/{iterations} - Loss: {loss:.2f}")
            results.append(deprocess_image(generated_image.numpy()))
    
    final_image = deprocess_image(generated_image.numpy())
    return final_image, results

def plot_results(content_path, style_path, result_image, intermediate_results):
    """Plot comparison of images."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: Content, Style, Result
    content = Image.open(content_path)
    style = Image.open(style_path)
    
    axes[0, 0].imshow(content)
    axes[0, 0].set_title('Content Image', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(style)
    axes[0, 1].set_title('Style Image', fontsize=14)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(result_image)
    axes[0, 2].set_title('Stylized Result', fontsize=14)
    axes[0, 2].axis('off')
    
    # Bottom row: Intermediate results
    for i, (ax, img) in enumerate(zip(axes[1], intermediate_results[:3])):
        ax.imshow(img)
        ax.set_title(f'Iteration {(i+1)*20}', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('style_transfer_results.png', dpi=150)
    plt.close()
    print("Results saved as 'style_transfer_results.png'")

def main():
    print("=" * 50)
    print("Day 13: Neural Style Transfer")
    print("=" * 50)
    
    # Create sample images
    print("\n[1] Creating sample images...")
    content_path, style_path = create_sample_images()
    
    # Perform style transfer
    print("\n[2] Performing neural style transfer...")
    result_image, intermediate = style_transfer(content_path, style_path, iterations=100)
    
    # Save final result
    Image.fromarray(result_image).save('stylized_output.jpg')
    print("\nStylized image saved as 'stylized_output.jpg'")
    
    # Plot comparison
    print("\n[3] Generating comparison visualization...")
    plot_results(content_path, style_path, result_image, intermediate)
    
    print("\n" + "=" * 50)
    print("Day 13 Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
