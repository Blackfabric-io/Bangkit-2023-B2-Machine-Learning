import pytest
import tensorflow as tf
from src.utils import get_dataset_info

def test_get_dataset_info(tmp_path):
    # Get dataset info
    info = get_dataset_info(str(tmp_path))
    
    # Check info structure
    assert 'image_shape' in info
    assert 'num_classes' in info
    assert 'splits' in info
    
    # Check values
    assert info['image_shape'] == (300, 300, 3)
    assert info['num_classes'] == 3
    assert isinstance(info['splits'], dict)
    assert 'train' in info['splits']
    assert 'test' in info['splits']
    
    # Check split sizes
    assert info['splits']['train'] == 2520
    assert info['splits']['test'] == 372 