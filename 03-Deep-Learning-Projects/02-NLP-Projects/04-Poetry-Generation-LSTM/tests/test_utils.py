"""Tests for utility functions."""

import os
import pytest
import numpy as np
from src.utils.helpers import (
    clean_text,
    preprocess_corpus,
    format_poetry,
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

def test_preprocess_corpus():
    """Test corpus preprocessing."""
    corpus = [
        "Line 1\n",
        "",  # Empty line
        "  Line 2  ",
        "LINE 3!"
    ]
    processed = preprocess_corpus(corpus)
    
    assert len(processed) == 3  # Empty line removed
    assert all(line.islower() for line in processed)
    assert all("\n" not in line for line in processed)
    assert processed[0] == "line 1"
    assert processed[1] == "line 2"
    assert processed[2] == "line 3!"

def test_format_poetry():
    """Test poetry formatting."""
    text = "this is a long line that should be split into multiple lines"
    formatted = format_poetry(text, words_per_line=4)
    
    lines = formatted.split("\n")
    assert len(lines) == 3
    assert lines[0] == "this is a long"
    assert lines[1] == "line that should be"
    assert lines[2] == "split into multiple lines"

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