#!/usr/bin/env python3
"""
Test Script for External Image Segmentation

This script demonstrates how to use external images with the segmentation model.
It includes a simple test with a sample image or creates a test image.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# Add the parent directory to the path to import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_test_image(save_path='test_image.jpg', size=(256, 256)):
    """
    Create a simple test image for segmentation.
    
    Args:
        save_path: Path to save the test image
        size: Size of the test image (width, height)
    """
    # Create a simple test image with a circle (simulating a pet)
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # Create a white background
    img.fill(255)
    
    # Draw a circle (simulating a pet)
    center_x, center_y = size[0] // 2, size[1] // 2
    radius = min(size) // 4
    
    # Create a grid of coordinates
    y, x = np.ogrid[:size[1], :size[0]]
    
    # Create the circle mask
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    # Color the circle (brown for a pet)
    img[mask] = [139, 69, 19]  # Brown color
    
    # Save the image
    test_image = Image.fromarray(img)
    test_image.save(save_path)
    print(f"Test image created and saved to: {save_path}")
    
    return save_path

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """
    Load and preprocess an image for segmentation.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the model input
    
    Returns:
        tuple: (preprocessed_image, original_image)
    """
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image using PIL
    original_image = Image.open(image_path)
    
    # Convert to RGB if necessary
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')
    
    # Convert to numpy array
    image_array = np.array(original_image)
    
    # Resize the image
    resized_image = tf.image.resize(image_array, target_size)
    
    # Normalize to [0, 1] range
    normalized_image = tf.cast(resized_image, tf.float32) / 255.0
    
    # Add batch dimension for model input
    batched_image = tf.expand_dims(normalized_image, axis=0)
    
    return batched_image, image_array

def create_simple_model():
    """
    Create a simple U-Net model for testing.
    This is a simplified version of the model architecture.
    """
    # Simple encoder
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    
    # Simple convolutional layers
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    # Simple decoder
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same')(x)
    
    # Output layer
    outputs = tf.keras.layers.Conv2D(3, 1, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def segment_image(image_path, model):
    """
    Perform segmentation on an image.
    
    Args:
        image_path: Path to the input image
        model: Segmentation model
    
    Returns:
        tuple: (original_image, predicted_mask)
    """
    print(f"Processing image: {image_path}")
    
    # Load and preprocess the image
    preprocessed_image, original_image = load_and_preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(preprocessed_image)
    
    # Convert to mask
    predicted_mask = np.argmax(prediction[0], axis=-1)
    
    return original_image, predicted_mask

def display_results(original_image, predicted_mask):
    """
    Display the segmentation results.
    
    Args:
        original_image: Original input image
        predicted_mask: Predicted segmentation mask
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Predicted mask
    axes[1].imshow(predicted_mask, cmap='viridis')
    axes[1].set_title('Predicted Segmentation Mask')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to test external image segmentation.
    """
    print("External Image Segmentation Test")
    print("=" * 40)
    
    # Check if a specific image was provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return 1
    else:
        # Create a test image
        print("No image provided. Creating a test image...")
        image_path = create_test_image()
    
    try:
        # Create a simple model for testing
        print("Creating a simple test model...")
        model = create_simple_model()
        
        # Perform segmentation
        original_image, predicted_mask = segment_image(image_path, model)
        
        # Display results
        print("Displaying results...")
        display_results(original_image, predicted_mask)
        
        # Print statistics
        print("\nSegmentation Statistics:")
        print(f"Original image shape: {original_image.shape}")
        print(f"Predicted mask shape: {predicted_mask.shape}")
        print(f"Unique mask values: {np.unique(predicted_mask)}")
        
        # Count pixels in each class
        for class_id in np.unique(predicted_mask):
            count = np.sum(predicted_mask == class_id)
            percentage = (count / predicted_mask.size) * 100
            print(f"Class {class_id}: {count} pixels ({percentage:.1f}%)")
        
        print("\nTest completed successfully!")
        print("\nNote: This is a test model with random weights.")
        print("For real segmentation, use a trained model from the main notebook.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 