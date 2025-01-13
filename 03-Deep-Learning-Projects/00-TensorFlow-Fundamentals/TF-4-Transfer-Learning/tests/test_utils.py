"""Tests for utility functions."""

import os
import pytest
import numpy as np
from PIL import Image
from src.utils.helpers import (
    create_data_generator,
    load_and_preprocess_image,
    validate_image_directory
)

@pytest.fixture
def sample_image_directory(tmp_path):
    """Create a temporary directory with sample images."""
    # Create class directories
    happy_dir = tmp_path / "happy"
    sad_dir = tmp_path / "sad"
    happy_dir.mkdir()
    sad_dir.mkdir()
    
    # Create sample images
    for i in range(3):
        # Happy images
        img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8))
        img.save(happy_dir / f"happy_{i}.jpg")
        
        # Sad images
        img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8))
        img.save(sad_dir / f"sad_{i}.jpg")
    
    return tmp_path

def test_create_data_generator(sample_image_directory):
    """Test data generator creation with valid directory."""
    generator = create_data_generator(
        str(sample_image_directory),
        batch_size=2,
        target_size=(28, 28)
    )
    
    # Check generator properties
    assert generator.batch_size == 2
    assert generator.target_size == (28, 28)
    assert generator.class_mode == 'binary'
    assert generator.color_mode == 'grayscale'
    
    # Check first batch
    batch_x, batch_y = next(generator)
    assert batch_x.shape == (2, 28, 28, 1)
    assert batch_y.shape == (2,)
    assert np.all(batch_x >= 0) and np.all(batch_x <= 1)
    assert np.all(np.isin(batch_y, [0, 1]))

def test_create_data_generator_invalid_dir():
    """Test data generator creation with invalid directory."""
    with pytest.raises(ValueError):
        create_data_generator("nonexistent_dir")

def test_create_data_generator_missing_class(tmp_path):
    """Test data generator creation with missing class directory."""
    # Create only one class directory
    (tmp_path / "happy").mkdir()
    
    with pytest.raises(ValueError):
        create_data_generator(str(tmp_path))

def test_load_and_preprocess_image(sample_image_directory):
    """Test image loading and preprocessing."""
    image_path = str(sample_image_directory / "happy" / "happy_0.jpg")
    
    # Load and preprocess
    img_array = load_and_preprocess_image(image_path)
    
    # Check properties
    assert img_array.shape == (28, 28, 1)
    assert img_array.dtype == np.float32
    assert np.all(img_array >= 0) and np.all(img_array <= 1)

def test_load_and_preprocess_invalid_image():
    """Test image loading with invalid file."""
    with pytest.raises(ValueError):
        load_and_preprocess_image("nonexistent_image.jpg")

def test_validate_image_directory(sample_image_directory):
    """Test directory validation with valid structure."""
    stats = validate_image_directory(str(sample_image_directory))
    
    assert isinstance(stats, dict)
    assert set(stats.keys()) == {'happy', 'sad'}
    assert stats['happy'] == 3
    assert stats['sad'] == 3

def test_validate_invalid_directory():
    """Test directory validation with invalid directory."""
    with pytest.raises(ValueError):
        validate_image_directory("nonexistent_dir")

def test_validate_directory_missing_class(tmp_path):
    """Test directory validation with missing class."""
    # Create only one class directory
    (tmp_path / "happy").mkdir()
    
    with pytest.raises(ValueError):
        validate_image_directory(str(tmp_path))

def test_validate_directory_empty_class(tmp_path):
    """Test directory validation with empty class directory."""
    # Create empty class directories
    (tmp_path / "happy").mkdir()
    (tmp_path / "sad").mkdir()
    
    stats = validate_image_directory(str(tmp_path))
    assert stats['happy'] == 0
    assert stats['sad'] == 0 