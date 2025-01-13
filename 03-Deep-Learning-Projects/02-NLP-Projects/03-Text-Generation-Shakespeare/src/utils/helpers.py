"""Helper functions for text preprocessing and data loading."""

import os
from typing import List, Tuple, Optional
import numpy as np

def load_text(filename: str) -> str:
    """Load text data from a file.
    
    Args:
        filename: Path to the text file
        
    Returns:
        Text content as string
    """
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

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

def split_input_target(sequence: List[int]) -> Tuple[List[int], List[int]]:
    """Split sequence into input and target.
    
    Args:
        sequence: Input sequence
        
    Returns:
        Tuple of (input sequence, target sequence)
    """
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

def create_training_examples(text: str,
                           seq_length: int,
                           stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Create training examples from text.
    
    Args:
        text: Input text
        seq_length: Length of each sequence
        stride: Step size between sequences
        
    Returns:
        Tuple of (input sequences, target sequences)
    """
    # Create sequences
    sequences = []
    for i in range(0, len(text) - seq_length, stride):
        sequences.append(text[i:i + seq_length + 1])
    
    # Split into input and target
    input_sequences = []
    target_sequences = []
    for seq in sequences:
        input_seq, target_seq = split_input_target(seq)
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
    
    return np.array(input_sequences), np.array(target_sequences)

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