"""Tests for core text preprocessing and tokenization functionality."""

import pytest
from src.core import TextPreprocessor, TokenizerWrapper

@pytest.fixture
def text_processor():
    """Create text processor instance."""
    return TextPreprocessor()

@pytest.fixture
def tokenizer():
    """Create tokenizer instance."""
    return TokenizerWrapper()

@pytest.fixture
def sample_texts():
    """Create sample texts."""
    return [
        "This is a test sentence",
        "Another example of text",
        "More sample data here"
    ]

@pytest.fixture
def sample_labels():
    """Create sample labels."""
    return ["tech", "business", "sport"]

def test_remove_stopwords(text_processor):
    """Test stopwords removal."""
    text = "This is a test sentence with stopwords"
    processed = text_processor.remove_stopwords(text)
    
    # Check that stopwords are removed
    assert "is" not in processed
    assert "a" not in processed
    assert "with" not in processed
    
    # Check that content words remain
    assert "test" in processed
    assert "sentence" in processed
    assert "stopwords" in processed

def test_process_text_file(text_processor, tmp_path):
    """Test text file processing."""
    # Create test file
    test_file = tmp_path / "test.csv"
    test_file.write_text(
        "category,text\n"
        "tech,This is a test article\n"
        "sport,Another test article here"
    )
    
    # Process file
    texts, labels = text_processor.process_text_file(str(test_file))
    
    # Check results
    assert len(texts) == 2
    assert len(labels) == 2
    assert "test article" in texts[0]
    assert labels[0] == "tech"

def test_tokenizer_initialization(tokenizer):
    """Test tokenizer initialization."""
    assert tokenizer.tokenizer.oov_token == "<OOV>"
    assert not tokenizer.is_fitted
    assert not tokenizer.is_label_fitted

def test_fit_text_tokenizer(tokenizer, sample_texts):
    """Test text tokenizer fitting."""
    tokenizer.fit_text_tokenizer(sample_texts)
    
    assert tokenizer.is_fitted
    assert len(tokenizer.word_index) > 0
    assert "<OOV>" in tokenizer.word_index

def test_fit_label_tokenizer(tokenizer, sample_labels):
    """Test label tokenizer fitting."""
    tokenizer.fit_label_tokenizer(sample_labels)
    
    assert tokenizer.is_label_fitted
    assert len(tokenizer.label_word_index) == 3
    assert all(label in tokenizer.label_word_index for label in sample_labels)

def test_get_padded_sequences(tokenizer, sample_texts):
    """Test sequence padding."""
    # Fit tokenizer first
    tokenizer.fit_text_tokenizer(sample_texts)
    
    # Get sequences
    sequences = tokenizer.get_padded_sequences(sample_texts)
    
    assert len(sequences) == len(sample_texts)
    assert all(len(seq) == len(sequences[0]) for seq in sequences)

def test_get_label_sequences(tokenizer, sample_labels):
    """Test label sequence generation."""
    # Fit tokenizer first
    tokenizer.fit_label_tokenizer(sample_labels)
    
    # Get sequences
    sequences = tokenizer.get_label_sequences(sample_labels)
    
    assert len(sequences) == len(sample_labels)
    assert all(isinstance(seq[0], int) for seq in sequences)

def test_tokenizer_errors(tokenizer, sample_texts):
    """Test tokenizer error handling."""
    # Test sequence generation before fitting
    with pytest.raises(ValueError):
        tokenizer.get_padded_sequences(sample_texts)
    
    # Test label sequence generation before fitting
    with pytest.raises(ValueError):
        tokenizer.get_label_sequences(["tech"])

def test_text_processor_errors(text_processor):
    """Test text processor error handling."""
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        text_processor.process_text_file("nonexistent.csv")
    
    # Test with invalid file format
    with pytest.raises(ValueError):
        text_processor.process_text_file(__file__)  # Use this test file as invalid input 