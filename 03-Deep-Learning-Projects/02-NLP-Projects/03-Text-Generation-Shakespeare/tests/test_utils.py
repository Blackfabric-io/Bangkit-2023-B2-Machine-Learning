"""Tests for utility functions."""

import os
import pytest
import numpy as np
from src.utils.helpers import (
    clean_text,
    split_input_target,
    create_training_examples,
    save_model_artifacts,
    load_model_artifacts
)

def test_clean_text():
    """Test text cleaning functionality."""
    text = "Hello\nWorld!\rThis\tis a   test"
    cleaned = clean_text(text)
    
    assert cleaned == "hello world! this is a test"
    assert cleaned.islower()
    assert "\n" not in cleaned
    assert "\r" not in cleaned
    assert "\t" not in cleaned
    assert "   " not in cleaned

def test_split_input_target():
    """Test sequence splitting functionality."""
    sequence = [1, 2, 3, 4, 5]
    input_seq, target_seq = split_input_target(sequence)
    
    assert len(input_seq) == len(target_seq)
    assert input_seq == [1, 2, 3, 4]
    assert target_seq == [2, 3, 4, 5]

def test_create_training_examples():
    """Test creation of training examples."""
    text = "hello world"
    seq_length = 3
    stride = 1
    
    X, y = create_training_examples(text, seq_length, stride)
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(X) == len(y)
    assert X.shape[1] == seq_length
    assert y.shape[1] == seq_length

def test_model_artifacts_saving_loading(tmp_path):
    """Test saving and loading of model artifacts."""
    # Create mock artifacts
    tokenizer = type('MockTokenizer', (), {
        'get_config': lambda: {'config': 'test'},
        'word_index': {'test': 1},
        'index_word': {1: 'test'}
    })
    config = {'test_param': 'value'}
    
    # Save artifacts
    save_model_artifacts(str(tmp_path), tokenizer, config)
    
    # Check files exist
    assert os.path.exists(os.path.join(tmp_path, 'tokenizer.npy'))
    assert os.path.exists(os.path.join(tmp_path, 'config.npy'))
    
    # Load and verify artifacts
    loaded_tokenizer, loaded_config = load_model_artifacts(str(tmp_path))
    assert loaded_config['test_param'] == 'value' 