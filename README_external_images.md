# Using External Images for Segmentation

This guide explains how to use external images (not from the training dataset) with the trained U-Net segmentation model.

## Overview

The segmentation model is trained on the Oxford-IIIT Pet Dataset, which classifies each pixel into 3 categories:
- **Class 0**: Background (surrounding pixel)
- **Class 1**: Pet object (main subject)
- **Class 2**: Pet border (boundary pixels)

## Prerequisites

1. **Trained Model**: You need a trained segmentation model. You can either:
   - Train the model using the notebook (`notebook/segmentation.ipynb`)
   - Use a pre-trained model if available

2. **Dependencies**: Make sure you have all required packages installed:
   ```bash
   pip install -r requirements.txt
   # or for GPU support
   pip install -r requirements_gpu.txt
   ```

## Method 1: Using the Notebook (Recommended)

### Step 1: Train and Save the Model

1. Open the notebook: `notebook/segmentation.ipynb`
2. Run all cells to train the model
3. After training, save the model:
   ```python
   # Save the trained model
   model.save('my_segmentation_model')
   ```

### Step 2: Add External Image Processing

Add this code cell to your notebook after the model training:

```python
# Function to load and preprocess external images
def load_external_image(image_path, target_size=(128, 128)):
    """
    Load and preprocess an external image for segmentation.
    """
    import numpy as np
    from PIL import Image
    
    # Load image
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

# Function to segment external images
def segment_external_image(image_path, model):
    """
    Perform segmentation on an external image.
    """
    # Load and preprocess the image
    preprocessed_image, original_image = load_external_image(image_path)
    
    # Make prediction
    prediction = model.predict(preprocessed_image)
    predicted_mask = create_mask(prediction)
    
    # Display results
    display([original_image, predicted_mask])
    
    return original_image, predicted_mask

# Example usage
# Replace 'path/to/your/image.jpg' with your actual image path
image_path = 'path/to/your/image.jpg'
original_image, predicted_mask = segment_external_image(image_path, model)
```

### Step 3: Use Your External Image

```python
# Example with your image
image_path = 'your_image.jpg'  # Replace with your image path
original_image, predicted_mask = segment_external_image(image_path, model)
```

## Method 2: Using the Python Script

### Step 1: Save Your Trained Model

In your notebook, save the model:
```python
model.save('my_segmentation_model')
```

### Step 2: Use the External Image Script

1. **Simple Script**: Use `scripts/simple_external_segmentation.py`
   ```python
   # Load your saved model
   model = tf.keras.models.load_model('my_segmentation_model')
   
   # Use the segmentation function
   from simple_external_segmentation import segment_external_image
   
   image_path = 'your_image.jpg'
   original_image, predicted_mask = segment_external_image(image_path, model)
   ```

2. **Advanced Script**: Use `scripts/external_image_segmentation.py`
   ```bash
   python external_image_segmentation.py --image your_image.jpg --model my_segmentation_model
   ```

## Image Requirements

### Supported Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

### Image Processing
- Images are automatically resized to 128x128 pixels (model input size)
- Images are converted to RGB format if needed
- Pixel values are normalized to [0, 1] range

### Best Practices
1. **Image Quality**: Use clear, well-lit images
2. **Subject**: The model works best with pet images (cats, dogs, etc.)
3. **Background**: Simple backgrounds work better than complex ones
4. **Size**: Original image size doesn't matter (will be resized)

## Understanding the Output

The segmentation model produces a mask where each pixel is classified:

- **Red (Class 0)**: Background pixels
- **Green (Class 1)**: Main object/subject pixels  
- **Blue (Class 2)**: Border/edge pixels

### Interpreting Results

```python
# Check the unique values in the mask
unique_values = np.unique(predicted_mask)
print(f"Mask contains classes: {unique_values}")

# Count pixels in each class
for class_id in unique_values:
    count = np.sum(predicted_mask == class_id)
    percentage = (count / predicted_mask.size) * 100
    print(f"Class {class_id}: {count} pixels ({percentage:.1f}%)")
```

## Troubleshooting

### Common Issues

1. **Model Not Found**
   ```
   Error: No such file or directory: 'my_segmentation_model'
   ```
   **Solution**: Make sure you've saved the model after training

2. **Image Not Found**
   ```
   Error: Image not found: path/to/image.jpg
   ```
   **Solution**: Check the image path is correct

3. **Poor Segmentation Results**
   - Try different images
   - Ensure the image contains a pet/animal
   - Use images with simple backgrounds
   - Make sure the image is clear and well-lit

4. **Memory Issues**
   - Reduce image size before processing
   - Close other applications to free memory
   - Use smaller batch sizes

### Performance Tips

1. **GPU Usage**: If available, the model will automatically use GPU
2. **Batch Processing**: Process multiple images at once for efficiency
3. **Model Optimization**: Consider model quantization for faster inference

## Example Workflow

Here's a complete example workflow:

```python
# 1. Train the model (in notebook)
# ... (run all training cells)

# 2. Save the model
model.save('my_segmentation_model')

# 3. Load the model for external use
model = tf.keras.models.load_model('my_segmentation_model')

# 4. Process external images
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']

for image_path in image_paths:
    try:
        original_image, predicted_mask = segment_external_image(image_path, model)
        print(f"Successfully processed: {image_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
```

## Advanced Usage

### Custom Preprocessing

You can modify the preprocessing function for specific needs:

```python
def custom_preprocess(image_path, target_size=(128, 128)):
    # Load image
    image = Image.open(image_path)
    
    # Apply custom preprocessing
    # e.g., adjust brightness, contrast, etc.
    
    # Continue with standard preprocessing
    image_array = np.array(image)
    resized_image = tf.image.resize(image_array, target_size)
    normalized_image = tf.cast(resized_image, tf.float32) / 255.0
    batched_image = tf.expand_dims(normalized_image, axis=0)
    
    return batched_image, image_array
```

### Batch Processing

Process multiple images efficiently:

```python
def batch_segment_images(image_paths, model):
    results = []
    
    for image_path in image_paths:
        try:
            original_image, predicted_mask = segment_external_image(image_path, model)
            results.append({
                'path': image_path,
                'original': original_image,
                'mask': predicted_mask,
                'success': True
            })
        except Exception as e:
            results.append({
                'path': image_path,
                'error': str(e),
                'success': False
            })
    
    return results
```

This guide should help you successfully use external images with your trained segmentation model! 