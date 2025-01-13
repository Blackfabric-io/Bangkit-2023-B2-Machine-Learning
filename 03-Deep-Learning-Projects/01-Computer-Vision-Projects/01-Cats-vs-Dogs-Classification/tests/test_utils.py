"""Tests for utility functions."""

import os
import pytest
import numpy as np
from PIL import Image
from src.utils import create_data_generators, load_and_preprocess_image
from src.utils.helpers import validate_data_directory

@pytest.fixture
def sample_data_directory(tmp_path):
    """Create a temporary directory with sample images."""
    # Create class directories
    cats_dir = tmp_path / "cats"
    dogs_dir = tmp_path / "dogs"
    cats_dir.mkdir()
    dogs_dir.mkdir()
    
    # Create sample images
    for i in range(3):
        # Cat images
        img = Image.fromarray(np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8))
        img.save(cats_dir / f"cat_{i}.jpg")
        
        # Dog images
        img = Image.fromarray(np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8))
        img.save(dogs_dir / f"dog_{i}.jpg")
    
    return tmp_path

def test_create_data_generators(sample_data_directory):
    """Test data generator creation with valid directory."""
    train_gen, val_gen = create_data_generators(
        str(sample_data_directory),
        batch_size=2,
        validation_split=0.5
    )
    
    # Check generator properties
    assert train_gen.batch_size == 2
    assert val_gen.batch_size == 2
    assert train_gen.target_size == (150, 150)
    assert val_gen.target_size == (150, 150)
    
    # Check first batch
    batch_x, batch_y = next(train_gen)
    assert batch_x.shape == (2, 150, 150, 3)
    assert batch_y.shape == (2,)
    assert np.all(batch_x >= 0) and np.all(batch_x <= 1)
    assert np.all(np.isin(batch_y, [0, 1]))

def test_create_data_generators_invalid_dir():
    """Test data generator creation with invalid directory."""
    with pytest.raises(ValueError):
        create_data_generators("nonexistent_dir")

def test_create_data_generators_missing_class(tmp_path):
    """Test data generator creation with missing class directory."""
    # Create only one class directory
    (tmp_path / "cats").mkdir()
    
    with pytest.raises(ValueError):
        create_data_generators(str(tmp_path))

def test_load_and_preprocess_image(sample_data_directory):
    """Test image loading and preprocessing."""
    image_path = str(sample_data_directory / "cats" / "cat_0.jpg")
    
    # Load and preprocess
    img_array = load_and_preprocess_image(image_path)
    
    # Check properties
    assert img_array.shape == (1, 150, 150, 3)
    assert img_array.dtype == np.float32
    assert np.all(img_array >= 0) and np.all(img_array <= 1)

def test_load_and_preprocess_invalid_image():
    """Test image loading with invalid file."""
    with pytest.raises(ValueError):
        load_and_preprocess_image("nonexistent_image.jpg")

def test_validate_data_directory(sample_data_directory):
    """Test directory validation with valid structure."""
    stats = validate_data_directory(str(sample_data_directory))
    
    assert isinstance(stats, dict)
    assert set(stats.keys()) == {'cats', 'dogs'}
    assert stats['cats'] == 3
    assert stats['dogs'] == 3

def test_validate_invalid_directory():
    """Test directory validation with invalid directory."""
    with pytest.raises(ValueError):
        validate_data_directory("nonexistent_dir")

def test_validate_directory_missing_class(tmp_path):
    """Test directory validation with missing class."""
    # Create only one class directory
    (tmp_path / "cats").mkdir()
    
    with pytest.raises(ValueError):
        validate_data_directory(str(tmp_path))

def test_validate_directory_empty_class(tmp_path):
    """Test directory validation with empty class directory."""
    # Create empty class directories
    (tmp_path / "cats").mkdir()
    (tmp_path / "dogs").mkdir()
    
    stats = validate_data_directory(str(tmp_path))
    assert stats['cats'] == 0
    assert stats['dogs'] == 0 