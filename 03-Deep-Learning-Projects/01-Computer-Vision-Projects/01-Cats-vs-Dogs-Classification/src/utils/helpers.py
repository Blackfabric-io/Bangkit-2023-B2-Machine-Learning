"""Helper functions for data loading and preprocessing."""

import os
from typing import Tuple, Dict
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

def create_data_generators(
    data_dir: str,
    target_size: Tuple[int, int] = (150, 150),
    batch_size: int = 32,
    validation_split: float = 0.2
) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator, tf.keras.preprocessing.image.DirectoryIterator]:
    """Create train and validation data generators with augmentation.
    
    Args:
        data_dir: Directory containing 'cats' and 'dogs' subdirectories
        target_size: Size to resize images to
        batch_size: Number of images per batch
        validation_split: Fraction of data to use for validation
        
    Returns:
        Tuple of (train_generator, validation_generator)
        
    Raises:
        ValueError: If data directory structure is invalid
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
        
    # Verify directory structure
    required_subdirs = {'cats', 'dogs'}
    existing_subdirs = set(os.listdir(data_dir))
    if not required_subdirs.issubset(existing_subdirs):
        raise ValueError(
            f"Data directory must contain 'cats' and 'dogs' subdirectories. "
            f"Found: {existing_subdirs}"
        )
    
    # Create training generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Create training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    
    # Create validation generator
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    return train_generator, validation_generator

def load_and_preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (150, 150)
) -> np.ndarray:
    """Load and preprocess a single image for prediction.
    
    Args:
        image_path: Path to image file
        target_size: Size to resize image to
        
    Returns:
        Preprocessed image array
        
    Raises:
        ValueError: If image file doesn't exist or can't be processed
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
        
    try:
        # Load and preprocess the image
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        return img_array
        
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")

def validate_data_directory(directory: str) -> Dict[str, int]:
    """Validate structure and count images in data directory.
    
    Args:
        directory: Path to data directory
        
    Returns:
        Dictionary with image counts per class
        
    Raises:
        ValueError: If directory structure is invalid
    """
    if not os.path.exists(directory):
        raise ValueError(f"Directory not found: {directory}")
        
    stats = {'cats': 0, 'dogs': 0}
    
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