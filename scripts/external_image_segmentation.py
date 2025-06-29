#!/usr/bin/env python3
"""
External Image Segmentation Script

This script demonstrates how to use the trained U-Net segmentation model
with external images (not from the training dataset).

Usage:
    python external_image_segmentation.py --image path/to/your/image.jpg
    python external_image_segmentation.py --image path/to/your/image.jpg --model path/to/saved/model
"""

import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Import the model components from the original script
from tensorflow_examples.models.pix2pix import pix2pix

def load_and_preprocess_external_image(image_path, target_size=(128, 128)):
    """
    Load and preprocess an external image for segmentation.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for the image (width, height)
    
    Returns:
        tuple: (preprocessed_image, original_image)
    """
    # Load the image
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
    
    # Add batch dimension
    batched_image = tf.expand_dims(normalized_image, axis=0)
    
    return batched_image, image_array

def create_unet_model(output_channels=3):
    """
    Create the U-Net model architecture (same as in the original script).
    
    Args:
        output_channels (int): Number of output classes
    
    Returns:
        tf.keras.Model: The U-Net model
    """
    # Encoder (MobileNetV2)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=[128, 128, 3], 
        include_top=False
    )
    
    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    
    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False
    
    # Decoder/upsampler
    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]
    
    def unet_model(output_channels):
        inputs = tf.keras.layers.Input(shape=[128, 128, 3])
        
        # Downsampling through the model
        skips = down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])
        
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])
        
        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            filters=output_channels, 
            kernel_size=3, 
            strides=2,
            padding='same'
        )  # 64x64 -> 128x128
        
        x = last(x)
        return tf.keras.Model(inputs=inputs, outputs=x)
    
    return unet_model(output_channels)

def create_mask(pred_mask):
    """
    Convert prediction logits to mask.
    
    Args:
        pred_mask: Model prediction logits
    
    Returns:
        numpy.ndarray: Segmentation mask
    """
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0].numpy()

def visualize_segmentation(original_image, predicted_mask, save_path=None):
    """
    Visualize the segmentation results.
    
    Args:
        original_image: Original input image
        predicted_mask: Predicted segmentation mask
        save_path (str, optional): Path to save the visualization
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Predicted mask
    axes[1].imshow(predicted_mask, cmap='viridis')
    axes[1].set_title('Predicted Segmentation Mask')
    axes[1].axis('off')
    
    # Overlay mask on original image
    # Create a colored mask overlay
    colored_mask = np.zeros_like(original_image)
    colored_mask[predicted_mask == 0] = [255, 0, 0]    # Red for background
    colored_mask[predicted_mask == 1] = [0, 255, 0]    # Green for object
    colored_mask[predicted_mask == 2] = [0, 0, 255]    # Blue for border
    
    # Blend with original image
    alpha = 0.6
    overlay = cv2.addWeighted(original_image, 1-alpha, colored_mask, alpha, 0)
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def segment_external_image(image_path, model_path=None, save_output=True):
    """
    Perform segmentation on an external image.
    
    Args:
        image_path (str): Path to the input image
        model_path (str, optional): Path to saved model weights
        save_output (bool): Whether to save the output visualization
    
    Returns:
        tuple: (original_image, predicted_mask)
    """
    print(f"Loading image: {image_path}")
    
    # Load and preprocess the image
    preprocessed_image, original_image = load_and_preprocess_external_image(image_path)
    
    # Create or load the model
    if model_path and os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Creating new model (untrained)")
        model = create_unet_model(output_channels=3)
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
    
    # Make prediction
    print("Making prediction...")
    prediction = model.predict(preprocessed_image)
    predicted_mask = create_mask(prediction)
    
    # Visualize results
    print("Creating visualization...")
    output_path = None
    if save_output:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"segmentation_result_{base_name}.png"
    
    visualize_segmentation(original_image, predicted_mask, output_path)
    
    return original_image, predicted_mask

def main():
    parser = argparse.ArgumentParser(description='Segment external images using U-Net')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', help='Path to saved model (optional)')
    parser.add_argument('--no-save', action='store_true', help='Do not save output visualization')
    
    args = parser.parse_args()
    
    try:
        # Perform segmentation
        original_image, predicted_mask = segment_external_image(
            args.image, 
            args.model, 
            save_output=not args.no_save
        )
        
        print("Segmentation completed successfully!")
        print(f"Mask shape: {predicted_mask.shape}")
        print(f"Unique mask values: {np.unique(predicted_mask)}")
        
    except Exception as e:
        print(f"Error during segmentation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 