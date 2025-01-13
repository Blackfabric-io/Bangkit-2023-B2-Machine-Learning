"""Tests for utility functions."""

import os
import pytest
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.utils import load_image, save_image, visualize_augmentations

@pytest.fixture
def sample_image_file(tmp_path):
    """Create a temporary test image file."""
    image_path = tmp_path / "test_image.jpg"
    
    # Create random RGB image
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    img.save(image_path)
    
    return str(image_path)

def test_load_image(sample_image_file):
    """Test image loading functionality."""
    # Test basic loading
    image = load_image(sample_image_file)
    assert isinstance(image, np.ndarray)
    assert image.shape == (100, 100, 3)
    assert image.dtype == np.uint8
    
    # Test loading with resize
    image = load_image(sample_image_file, target_size=(50, 50))
    assert image.shape == (50, 50, 3)

def test_load_image_invalid_path():
    """Test loading with invalid image path."""
    with pytest.raises(ValueError):
        load_image("nonexistent_image.jpg")

def test_save_image(tmp_path, sample_image_file):
    """Test image saving functionality."""
    # Load test image
    image = load_image(sample_image_file)
    
    # Save to new path
    output_path = str(tmp_path / "output.jpg")
    save_image(image, output_path)
    
    # Verify file exists
    assert os.path.exists(output_path)
    
    # Verify can be loaded back
    loaded = load_image(output_path)
    assert loaded.shape == image.shape

def test_save_image_invalid_input(tmp_path):
    """Test saving with invalid input."""
    output_path = str(tmp_path / "output.jpg")
    
    with pytest.raises(ValueError):
        save_image([1, 2, 3], output_path)  # Wrong input type

def test_visualize_augmentations(sample_image_file):
    """Test augmentation visualization."""
    # Load test image
    original = load_image(sample_image_file)
    
    # Create dummy augmented images
    augmented = [
        np.random.randint(0, 255, original.shape, dtype=np.uint8)
        for _ in range(2)
    ]
    
    # Test visualization (should not raise)
    visualize_augmentations(original, augmented)
    plt.close()  # Clean up

def test_visualize_augmentations_invalid_input():
    """Test visualization with invalid inputs."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Wrong original image type
    with pytest.raises(ValueError):
        visualize_augmentations([1, 2, 3], [image])
    
    # Wrong augmented images type
    with pytest.raises(ValueError):
        visualize_augmentations(image, image)  # Should be list
    
    # Wrong augmented image type
    with pytest.raises(ValueError):
        visualize_augmentations(image, [[1, 2, 3]]) 