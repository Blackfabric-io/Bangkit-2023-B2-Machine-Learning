"""Helper functions for image loading and visualization."""

import os
from typing import List, Tuple, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load and preprocess an image.
    
    Args:
        image_path: Path to image file
        target_size: Optional size to resize image to (height, width)
        
    Returns:
        Image array in RGB format
        
    Raises:
        ValueError: If image file doesn't exist or can't be processed
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
        
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image")
            
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if target_size is not None:
            image = cv2.resize(image, (target_size[1], target_size[0]))
        
        return image
        
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")

def save_image(image: np.ndarray, output_path: str) -> None:
    """Save an image to file.
    
    Args:
        image: Image array in RGB format
        output_path: Path to save image to
        
    Raises:
        ValueError: If image format is invalid or save fails
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array")
        
    try:
        # Convert to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save image
        cv2.imwrite(output_path, image)
        
    except Exception as e:
        raise ValueError(f"Failed to save image: {str(e)}")

def visualize_augmentations(
    original_image: np.ndarray,
    augmented_images: List[np.ndarray],
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """Visualize original image alongside its augmented versions.
    
    Args:
        original_image: Original image array
        augmented_images: List of augmented image arrays
        figsize: Figure size for the plot
        
    Raises:
        ValueError: If image formats are invalid
    """
    if not isinstance(original_image, np.ndarray):
        raise ValueError("Original image must be a numpy array")
        
    if not isinstance(augmented_images, list) or not all(isinstance(img, np.ndarray) for img in augmented_images):
        raise ValueError("augmented_images must be a list of numpy arrays")
        
    # Create figure
    n_images = len(augmented_images) + 1
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    # Plot original image
    axes[0].imshow(original_image)
    axes[0].axis('off')
    axes[0].set_title('Original')
    
    # Plot augmented images
    for i, aug_img in enumerate(augmented_images, 1):
        axes[i].imshow(aug_img)
        axes[i].axis('off')
        axes[i].set_title(f'Augmented {i}')
    
    plt.tight_layout()
    plt.show() 