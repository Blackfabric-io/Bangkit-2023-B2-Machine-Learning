"""Base module containing core image augmentation implementations."""

from typing import Tuple, Optional, Dict, Any, List, Union
import tensorflow as tf
import numpy as np
import cv2
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AugmentationConfig:
    """Configuration for image augmentation parameters."""
    
    # Geometric transformations
    rotation_range: float = 30.0
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    shear_range: float = 0.2
    zoom_range: float = 0.2
    horizontal_flip: bool = True
    vertical_flip: bool = False
    
    # Color transformations
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.8, 1.2)
    hue_range: float = 0.1
    
    # Noise and filtering
    noise_stddev: float = 0.01
    gaussian_blur_range: Tuple[float, float] = (0.0, 1.0)
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.rotation_range < 0:
            raise ValueError("rotation_range must be non-negative")
            
        if not 0 <= self.width_shift_range <= 1:
            raise ValueError("width_shift_range must be between 0 and 1")
            
        if not 0 <= self.height_shift_range <= 1:
            raise ValueError("height_shift_range must be between 0 and 1")
            
        if self.shear_range < 0:
            raise ValueError("shear_range must be non-negative")
            
        if not 0 <= self.zoom_range <= 1:
            raise ValueError("zoom_range must be between 0 and 1")
            
        if not 0 <= self.brightness_range[0] <= self.brightness_range[1]:
            raise ValueError("Invalid brightness_range")
            
        if not 0 <= self.contrast_range[0] <= self.contrast_range[1]:
            raise ValueError("Invalid contrast_range")
            
        if not 0 <= self.saturation_range[0] <= self.saturation_range[1]:
            raise ValueError("Invalid saturation_range")
            
        if not 0 <= self.hue_range <= 0.5:
            raise ValueError("hue_range must be between 0 and 0.5")
            
        if self.noise_stddev < 0:
            raise ValueError("noise_stddev must be non-negative")
            
        if not 0 <= self.gaussian_blur_range[0] <= self.gaussian_blur_range[1]:
            raise ValueError("Invalid gaussian_blur_range")

class ImageAugmenter:
    """Class for applying various image augmentation techniques."""
    
    def __init__(self, config: Optional[AugmentationConfig] = None) -> None:
        """Initialize augmenter with configuration.
        
        Args:
            config: Augmentation parameters configuration
        """
        self.config = config or AugmentationConfig()
        self.config.validate()
        
    def augment(self, image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Apply configured augmentations to an image.
        
        Args:
            image: Input image array (shape: [height, width, channels])
            seed: Random seed for reproducibility
            
        Returns:
            Augmented image array
            
        Raises:
            ValueError: If image format is invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array")
            
        if len(image.shape) != 3:
            raise ValueError("Input must have shape [height, width, channels]")
            
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        
        # Apply geometric transformations
        image = self._apply_geometric_transforms(image)
        
        # Apply color transformations
        image = self._apply_color_transforms(image)
        
        # Apply noise and filtering
        image = self._apply_noise_and_filtering(image)
        
        return image
    
    def augment_batch(self, 
                     images: np.ndarray, 
                     seed: Optional[int] = None) -> np.ndarray:
        """Apply augmentations to a batch of images.
        
        Args:
            images: Batch of images (shape: [batch_size, height, width, channels])
            seed: Random seed for reproducibility
            
        Returns:
            Batch of augmented images
            
        Raises:
            ValueError: If image format is invalid
        """
        if not isinstance(images, np.ndarray):
            raise ValueError("Input must be a numpy array")
            
        if len(images.shape) != 4:
            raise ValueError("Input must have shape [batch_size, height, width, channels]")
            
        return np.array([self.augment(img, seed) for img in images])
    
    def _apply_geometric_transforms(self, image: np.ndarray) -> np.ndarray:
        """Apply geometric transformations to image."""
        # Create affine transformation matrix
        height, width = image.shape[:2]
        matrix = self._get_transform_matrix(width, height)
        
        # Apply transformation
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            borderMode=cv2.BORDER_REFLECT_101
        )
    
    def _apply_color_transforms(self, image: np.ndarray) -> np.ndarray:
        """Apply color transformations to image."""
        # Convert to HSV for color transformations
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Apply transformations
        hsv[..., 0] *= np.random.uniform(1 - self.config.hue_range, 1 + self.config.hue_range)  # Hue
        hsv[..., 1] *= np.random.uniform(*self.config.saturation_range)  # Saturation
        hsv[..., 2] *= np.random.uniform(*self.config.brightness_range)  # Value
        
        # Clip values and convert back
        hsv = np.clip(hsv, 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Apply contrast
        image = image.astype(np.float32)
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        contrast_factor = np.random.uniform(*self.config.contrast_range)
        image = (image - mean) * contrast_factor + mean
        
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def _apply_noise_and_filtering(self, image: np.ndarray) -> np.ndarray:
        """Apply noise and filtering to image."""
        # Add Gaussian noise
        if self.config.noise_stddev > 0:
            noise = np.random.normal(0, self.config.noise_stddev * 255, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Apply Gaussian blur
        blur_sigma = np.random.uniform(*self.config.gaussian_blur_range)
        if blur_sigma > 0:
            kernel_size = int(blur_sigma * 3) * 2 + 1  # Ensure odd kernel size
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), blur_sigma)
        
        return image
    
    def _get_transform_matrix(self, width: int, height: int) -> np.ndarray:
        """Generate affine transformation matrix for geometric transforms."""
        # Center point
        center_x = width / 2
        center_y = height / 2
        
        # Random transformation parameters
        angle = np.random.uniform(-self.config.rotation_range, self.config.rotation_range)
        tx = np.random.uniform(-self.config.width_shift_range, self.config.width_shift_range) * width
        ty = np.random.uniform(-self.config.height_shift_range, self.config.height_shift_range) * height
        shear = np.random.uniform(-self.config.shear_range, self.config.shear_range)
        zoom = 1 + np.random.uniform(-self.config.zoom_range, self.config.zoom_range)
        
        # Flip flags
        flip_h = np.random.random() < 0.5 if self.config.horizontal_flip else False
        flip_v = np.random.random() < 0.5 if self.config.vertical_flip else False
        
        # Create transformation matrix
        matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, zoom)
        
        # Add translation
        matrix[:, 2] += [tx, ty]
        
        # Add shear
        shear_matrix = np.array([[1, shear, 0], [0, 1, 0]], dtype=np.float32)
        matrix = np.dot(shear_matrix, matrix)
        
        # Add flips
        if flip_h:
            matrix[0, :] *= -1
            matrix[0, 2] += width
        if flip_v:
            matrix[1, :] *= -1
            matrix[1, 2] += height
        
        return matrix 