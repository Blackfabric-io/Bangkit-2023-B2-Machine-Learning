import pytest
from src.utils import get_dataset_info

def test_get_dataset_info(tmp_path):
    # Get dataset info
    info = get_dataset_info(str(tmp_path))
    
    # Check info structure
    assert 'image_shape' in info
    assert 'num_classes' in info
    assert 'splits' in info
    assert 'description' in info
    
    # Check values
    assert len(info['image_shape']) == 3  # Height, width, channels
    assert info['num_classes'] == 2  # Cats and dogs
    assert isinstance(info['splits'], dict)
    assert 'train' in info['splits']
    assert isinstance(info['description'], str)
    assert len(info['description']) > 0 