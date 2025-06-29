"""
Simple External Image Segmentation

This script shows how to use external images with the trained segmentation model.
It's a simplified version that works with the existing notebook setup.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

from config import settings


def load_external_image(image_path, target_size=(128, 128)):
    """
    Load and preprocess an external image for segmentation.
    
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
    
    # Normalize to [0, 1] range (same as training data)
    normalized_image = tf.cast(resized_image, tf.float32) / 255.0
    
    # Add batch dimension for model input
    batched_image = tf.expand_dims(normalized_image, axis=0)
    
    return batched_image, image_array


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


def display_segmentation(original_image, predicted_mask):
    """
    Display the segmentation results.
    
    Args:
        original_image: Original input image
        predicted_mask: Predicted segmentation mask
    """
    # Create a figure with subplots
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


def segment_external_image(image_path, model):
    """
    Perform segmentation on an external image using the provided model.
    
    Args:
        image_path: Path to the input image
        model: Trained segmentation model
    
    Returns:
        tuple: (original_image, predicted_mask)
    """
    print(f"Loading image: {image_path}")
    
    # Load and preprocess the image
    preprocessed_image, original_image = load_external_image(image_path)
    
    # Make prediction
    print("Making prediction...")
    prediction = model.predict(preprocessed_image)
    predicted_mask = create_mask(prediction)
    
    # Display results
    print("Displaying results...")
    display_segmentation(original_image, predicted_mask)
    
    return original_image, predicted_mask


# Example usage function
def example_usage():
    """
    Example of how to use this script with a trained model.
    """
    print("Example: How to use external images for segmentation")
    print("=" * 50)
    
    # This would be your trained model from the notebook
    # model = your_trained_model_here
    
    # Example image path (replace with your image)
    image_path = settings.IMAGE_PATH
    model = tf.keras.models.load_model(settings.MODEL_SAVE_PATH)
    segment_external_image(image_path, model)
    
    # print(f"To use this script:")
    # print(f"1. Train your model using the notebook")
    # print(f"2. Save the model: model.save('my_segmentation_model')")
    # print(f"3. Load the model: model = tf.keras.models.load_model('my_segmentation_model')")
    # print(f"4. Use this script with your model and image")
    # print(f"5. Call: segment_external_image('{image_path}', model)")


if __name__ == "__main__":
    example_usage()