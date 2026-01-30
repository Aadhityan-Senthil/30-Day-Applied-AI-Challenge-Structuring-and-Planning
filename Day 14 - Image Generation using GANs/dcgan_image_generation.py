"""
Day 14: Image Generation using GANs
30-Day AI Challenge

Implement a Deep Convolutional GAN (DCGAN) to generate images.
Uses MNIST for demonstration purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import os

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Hyperparameters
LATENT_DIM = 100
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.0002
BETA_1 = 0.5

def load_and_preprocess_data():
    """Load MNIST and preprocess for GAN training."""
    (x_train, _), (_, _) = mnist.load_data()
    
    # Normalize to [-1, 1]
    x_train = x_train.astype('float32')
    x_train = (x_train - 127.5) / 127.5
    
    # Add channel dimension
    x_train = np.expand_dims(x_train, axis=-1)
    
    print(f"Training data shape: {x_train.shape}")
    return x_train

def build_generator(latent_dim=LATENT_DIM):
    """Build the generator network."""
    model = keras.Sequential([
        # Foundation for 7x7 feature map
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(latent_dim,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Reshape((7, 7, 256)),
        
        # Upsample to 14x14
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        # Upsample to 14x14
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        # Upsample to 28x28
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ], name='generator')
    
    return model

def build_discriminator():
    """Build the discriminator network."""
    model = keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1)),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(1)
    ], name='discriminator')
    
    return model

class DCGAN(keras.Model):
    """Deep Convolutional GAN model."""
    
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = build_generator(latent_dim)
        self.discriminator = build_discriminator()
        
    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
        self.g_loss_metric = keras.metrics.Mean(name='g_loss')
        self.d_loss_metric = keras.metrics.Mean(name='d_loss')
    
    @property
    def metrics(self):
        return [self.g_loss_metric, self.d_loss_metric]
    
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        
        # Sample random noise
        random_noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        # Train Discriminator
        with tf.GradientTape() as tape:
            # Generate fake images
            fake_images = self.generator(random_noise, training=True)
            
            # Get discriminator predictions
            real_predictions = self.discriminator(real_images, training=True)
            fake_predictions = self.discriminator(fake_images, training=True)
            
            # Calculate discriminator loss
            real_loss = self.loss_fn(tf.ones_like(real_predictions), real_predictions)
            fake_loss = self.loss_fn(tf.zeros_like(fake_predictions), fake_predictions)
            d_loss = (real_loss + fake_loss) / 2
        
        # Update discriminator weights
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))
        
        # Train Generator
        random_noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_noise, training=True)
            fake_predictions = self.discriminator(fake_images, training=True)
            g_loss = self.loss_fn(tf.ones_like(fake_predictions), fake_predictions)
        
        # Update generator weights
        g_grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))
        
        # Update metrics
        self.g_loss_metric.update_state(g_loss)
        self.d_loss_metric.update_state(d_loss)
        
        return {'g_loss': self.g_loss_metric.result(), 'd_loss': self.d_loss_metric.result()}

class GANMonitor(keras.callbacks.Callback):
    """Callback to save generated images during training."""
    
    def __init__(self, num_images=16, latent_dim=LATENT_DIM):
        self.num_images = num_images
        self.latent_dim = latent_dim
        self.seed = tf.random.normal([num_images, latent_dim])
        self.generated_images = []
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            generated = self.model.generator(self.seed, training=False)
            self.generated_images.append((epoch + 1, generated.numpy()))
            print(f"  Saved sample images at epoch {epoch + 1}")

def generate_and_save_images(generator, epoch, seed, filename):
    """Generate and save a grid of images."""
    predictions = generator(seed, training=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    
    for i, ax in enumerate(axes.flat):
        img = predictions[i, :, :, 0] * 0.5 + 0.5  # Denormalize
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    plt.suptitle(f'Generated Images - Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_training_progress(history, monitor):
    """Plot training losses and generated samples."""
    fig = plt.figure(figsize=(15, 10))
    
    # Loss plot
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(history['g_loss'], label='Generator Loss')
    ax1.plot(history['d_loss'], label='Discriminator Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Generated images at different epochs
    for i, (epoch, images) in enumerate(monitor.generated_images[:3]):
        ax = fig.add_subplot(2, 3, i + 4)
        # Show first 9 images in a grid
        grid = np.zeros((28*3, 28*3))
        for j in range(9):
            row, col = j // 3, j % 3
            grid[row*28:(row+1)*28, col*28:(col+1)*28] = images[j, :, :, 0] * 0.5 + 0.5
        ax.imshow(grid, cmap='gray')
        ax.set_title(f'Epoch {epoch}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150)
    plt.close()
    print("Training progress saved as 'training_progress.png'")

def main():
    print("=" * 50)
    print("Day 14: Image Generation using DCGAN")
    print("=" * 50)
    
    # Load data
    print("\n[1] Loading MNIST data...")
    x_train = load_and_preprocess_data()
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.shuffle(60000).batch(BATCH_SIZE)
    
    # Build GAN
    print("\n[2] Building DCGAN model...")
    gan = DCGAN(LATENT_DIM)
    
    # Compile
    gan.compile(
        g_optimizer=keras.optimizers.Adam(LEARNING_RATE, BETA_1),
        d_optimizer=keras.optimizers.Adam(LEARNING_RATE, BETA_1),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
    )
    
    # Print model summaries
    print("\nGenerator Architecture:")
    gan.generator.summary()
    print("\nDiscriminator Architecture:")
    gan.discriminator.summary()
    
    # Callbacks
    monitor = GANMonitor()
    
    # Train
    print(f"\n[3] Training for {EPOCHS} epochs...")
    history = {'g_loss': [], 'd_loss': []}
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        epoch_g_loss = []
        epoch_d_loss = []
        
        for batch in dataset:
            losses = gan.train_step(batch)
            epoch_g_loss.append(float(losses['g_loss']))
            epoch_d_loss.append(float(losses['d_loss']))
        
        avg_g_loss = np.mean(epoch_g_loss)
        avg_d_loss = np.mean(epoch_d_loss)
        history['g_loss'].append(avg_g_loss)
        history['d_loss'].append(avg_d_loss)
        
        print(f"  G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
        
        # Save samples periodically
        monitor.model = gan
        monitor.on_epoch_end(epoch)
    
    # Generate final images
    print("\n[4] Generating final images...")
    seed = tf.random.normal([16, LATENT_DIM])
    generate_and_save_images(gan.generator, EPOCHS, seed, 'final_generated_images.png')
    print("Final images saved as 'final_generated_images.png'")
    
    # Plot training progress
    print("\n[5] Plotting training progress...")
    plot_training_progress(history, monitor)
    
    # Save generator model
    gan.generator.save('generator_model.h5')
    print("\nGenerator model saved as 'generator_model.h5'")
    
    print("\n" + "=" * 50)
    print("Day 14 Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
