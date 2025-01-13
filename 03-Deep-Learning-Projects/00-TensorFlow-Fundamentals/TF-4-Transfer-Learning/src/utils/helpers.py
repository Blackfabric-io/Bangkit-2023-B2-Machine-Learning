"""Utility functions for data loading and preprocessing."""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

def create_data_generator(data_dir, batch_size=10, target_size=(28, 28), class_mode='binary'):
    """Create a data generator for training.
    
    Args:
        data_dir (str): Directory containing class subdirectories
        batch_size (int): Number of images per batch
        target_size (tuple): Target size for images (height, width)
        class_mode (str): Type of label arrays to return
        
    Returns:
        DirectoryIterator: Generator yielding batches of (x, y)
        
    Raises:
        ValueError: If data_dir doesn't exist or has invalid structure
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory not found: {data_dir}")
        
    # Verify directory structure
    required_subdirs = {'happy', 'sad'}
    existing_subdirs = set(os.listdir(data_dir))
    if not required_subdirs.issubset(existing_subdirs):
        raise ValueError(
            f"Data directory must contain 'happy' and 'sad' subdirectories. "
            f"Found: {existing_subdirs}"
        )
    
    # Create generator with preprocessing
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    return datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode='grayscale'
    )

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image for prediction.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        numpy.ndarray: Preprocessed image array of shape (28, 28, 1)
        
    Raises:
        ValueError: If image file doesn't exist or can't be processed
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image not found: {image_path}")
        
    try:
        # Load and convert to grayscale
        img = load_img(image_path, color_mode='grayscale', target_size=(28, 28))
        
        # Convert to array and normalize
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        
        return img_array
        
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def validate_image_directory(directory):
    """Validate structure and contents of image directory.
    
    Args:
        directory (str): Path to image directory
        
    Returns:
        dict: Statistics about the directory contents
        
    Raises:
        ValueError: If directory structure is invalid
    """
    if not os.path.exists(directory):
        raise ValueError(f"Directory not found: {directory}")
        
    stats = {'happy': 0, 'sad': 0}
    
    for class_name in stats:
        class_dir = os.path.join(directory, class_name)
        if not os.path.exists(class_dir):
            raise ValueError(f"Missing class directory: {class_dir}")
            
        # Count valid images
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        for filename in os.listdir(class_dir):
            if os.path.splitext(filename)[1].lower() in valid_extensions:
                stats[class_name] += 1
                
    return stats 