"""Tests for core augmentation functionality."""

import pytest
import numpy as np
from src.core import ImageAugmenter, AugmentationConfig

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

@pytest.fixture
def default_augmenter():
    """Create an augmenter with default configuration."""
    return ImageAugmenter()

def test_config_validation():
    """Test configuration parameter validation."""
    # Valid configuration
    config = AugmentationConfig()
    config.validate()  # Should not raise
    
    # Invalid rotation range
    with pytest.raises(ValueError):
        config = AugmentationConfig(rotation_range=-30)
        config.validate()
    
    # Invalid shift range
    with pytest.raises(ValueError):
        config = AugmentationConfig(width_shift_range=1.5)
        config.validate()
    
    # Invalid brightness range
    with pytest.raises(ValueError):
        config = AugmentationConfig(brightness_range=(1.2, 0.8))
        config.validate()

def test_augmenter_initialization():
    """Test augmenter initialization with different configs."""
    # Default config
    augmenter = ImageAugmenter()
    assert augmenter.config is not None
    
    # Custom config
    config = AugmentationConfig(rotation_range=45)
    augmenter = ImageAugmenter(config)
    assert augmenter.config.rotation_range == 45

def test_single_image_augmentation(sample_image, default_augmenter):
    """Test augmentation of a single image."""
    augmented = default_augmenter.augment(sample_image)
    
    # Check output format
    assert isinstance(augmented, np.ndarray)
    assert augmented.shape == sample_image.shape
    assert augmented.dtype == np.uint8
    
    # Check value range
    assert np.all(augmented >= 0)
    assert np.all(augmented <= 255)
    
    # Check that augmentation made changes
    assert not np.array_equal(augmented, sample_image)

def test_batch_augmentation(sample_image, default_augmenter):
    """Test augmentation of image batches."""
    # Create batch
    batch = np.stack([sample_image] * 3)
    
    augmented = default_augmenter.augment_batch(batch)
    
    # Check output format
    assert isinstance(augmented, np.ndarray)
    assert augmented.shape == batch.shape
    assert augmented.dtype == np.uint8
    
    # Check that each image was augmented differently
    assert not np.array_equal(augmented[0], augmented[1])
    assert not np.array_equal(augmented[1], augmented[2])

def test_invalid_input_handling(default_augmenter):
    """Test handling of invalid inputs."""
    # Wrong input type
    with pytest.raises(ValueError):
        default_augmenter.augment([1, 2, 3])
    
    # Wrong number of dimensions
    with pytest.raises(ValueError):
        default_augmenter.augment(np.random.rand(100, 100))
    
    # Wrong batch dimensions
    with pytest.raises(ValueError):
        default_augmenter.augment_batch(np.random.rand(100, 100, 3))

def test_reproducibility(sample_image):
    """Test that augmentations are reproducible with same seed."""
    augmenter = ImageAugmenter()
    
    # Generate two augmentations with same seed
    aug1 = augmenter.augment(sample_image, seed=42)
    aug2 = augmenter.augment(sample_image, seed=42)
    
    # Should be identical
    assert np.array_equal(aug1, aug2)
    
    # Different seed should give different result
    aug3 = augmenter.augment(sample_image, seed=43)
    assert not np.array_equal(aug1, aug3)

def test_geometric_transforms(sample_image):
    """Test geometric transformation components."""
    # Test rotation only
    config = AugmentationConfig(
        rotation_range=90,
        width_shift_range=0,
        height_shift_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=False,
        vertical_flip=False
    )
    augmenter = ImageAugmenter(config)
    augmented = augmenter.augment(sample_image, seed=42)
    assert not np.array_equal(augmented, sample_image)
    
    # Test flips only
    config = AugmentationConfig(
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=True,
        vertical_flip=True
    )
    augmenter = ImageAugmenter(config)
    augmented = augmenter.augment(sample_image, seed=42)
    assert not np.array_equal(augmented, sample_image)

def test_color_transforms(sample_image):
    """Test color transformation components."""
    # Test brightness only
    config = AugmentationConfig(
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=False,
        vertical_flip=False,
        brightness_range=(0.5, 1.5),
        contrast_range=(1.0, 1.0),
        saturation_range=(1.0, 1.0),
        hue_range=0
    )
    augmenter = ImageAugmenter(config)
    augmented = augmenter.augment(sample_image, seed=42)
    assert not np.array_equal(augmented, sample_image)
    
    # Test contrast only
    config = AugmentationConfig(
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=False,
        vertical_flip=False,
        brightness_range=(1.0, 1.0),
        contrast_range=(0.5, 1.5),
        saturation_range=(1.0, 1.0),
        hue_range=0
    )
    augmenter = ImageAugmenter(config)
    augmented = augmenter.augment(sample_image, seed=42)
    assert not np.array_equal(augmented, sample_image) 