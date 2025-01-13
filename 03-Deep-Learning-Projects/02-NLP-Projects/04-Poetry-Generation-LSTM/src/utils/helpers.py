"""Helper functions for text preprocessing and data loading."""

import os
from typing import List, Tuple, Optional
import numpy as np

def load_poetry(filename: str) -> List[str]:
    """Load poetry data from a file.
    
    Args:
        filename: Path to the poetry file
        
    Returns:
        List of poetry lines
    """
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.read().lower().split('\n')
    return [line.strip() for line in lines if line.strip()]

def clean_text(text: str) -> str:
    """Clean and preprocess text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and extra whitespace
    special_chars = ['\n', '\r', '\t']
    for char in special_chars:
        text = text.replace(char, ' ')
    
    # Remove multiple spaces
    text = ' '.join(text.split())
    
    return text

def preprocess_corpus(corpus: List[str]) -> List[str]:
    """Preprocess the entire corpus.
    
    Args:
        corpus: List of text lines
        
    Returns:
        Preprocessed lines
    """
    processed = []
    for line in corpus:
        # Clean text
        line = clean_text(line)
        if line:  # Only keep non-empty lines
            processed.append(line)
    return processed

def save_model_artifacts(model_dir: str,
                        tokenizer: object,
                        config: dict,
                        create_dir: bool = True) -> None:
    """Save model artifacts.
    
    Args:
        model_dir: Directory to save artifacts
        tokenizer: Trained tokenizer
        config: Model configuration
        create_dir: Whether to create directory if it doesn't exist
    """
    if create_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    # Save tokenizer
    tokenizer_config = {
        'config': tokenizer.get_config(),
        'word_index': tokenizer.word_index,
        'index_word': tokenizer.index_word,
    }
    np.save(os.path.join(model_dir, 'tokenizer.npy'), tokenizer_config)
    
    # Save model config
    np.save(os.path.join(model_dir, 'config.npy'), config)

def load_model_artifacts(model_dir: str) -> Tuple[object, dict]:
    """Load model artifacts.
    
    Args:
        model_dir: Directory containing artifacts
        
    Returns:
        Tuple of (tokenizer, config)
    """
    # Load tokenizer
    tokenizer_config = np.load(os.path.join(model_dir, 'tokenizer.npy'),
                             allow_pickle=True).item()
    tokenizer = tf.keras.preprocessing.text.Tokenizer.from_config(
        tokenizer_config['config']
    )
    tokenizer.word_index = tokenizer_config['word_index']
    tokenizer.index_word = tokenizer_config['index_word']
    
    # Load config
    config = np.load(os.path.join(model_dir, 'config.npy'),
                    allow_pickle=True).item()
    
    return tokenizer, config

def format_poetry(text: str, words_per_line: int = 8) -> str:
    """Format text as poetry with line breaks.
    
    Args:
        text: Input text
        words_per_line: Number of words per line
        
    Returns:
        Formatted poetry
    """
    words = text.split()
    lines = []
    for i in range(0, len(words), words_per_line):
        line = ' '.join(words[i:i + words_per_line])
        lines.append(line)
    return '\n'.join(lines) 