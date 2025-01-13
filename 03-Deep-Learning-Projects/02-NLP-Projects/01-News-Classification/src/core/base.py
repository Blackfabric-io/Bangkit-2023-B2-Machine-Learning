"""Core classes for text preprocessing and tokenization."""

from typing import List, Dict, Optional, Tuple, Union
import logging
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Class for text preprocessing operations."""
    
    def __init__(self) -> None:
        """Initialize with list of stopwords."""
        self.stopwords = [
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
            "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", 
            "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", 
            "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", 
            "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", 
            "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", 
            "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", 
            "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", 
            "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", 
            "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", 
            "them", "themselves", "then", "there", "there's", "these", "they", "they'd", 
            "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", 
            "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", 
            "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", 
            "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", 
            "you're", "you've", "your", "yours", "yourself", "yourselves"
        ]
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text with stopwords removed
            
        Example:
            >>> processor = TextPreprocessor()
            >>> processor.remove_stopwords("I am going to the store")
            "going store"
        """
        # Convert to lowercase
        text = text.lower()
        
        # Split into words and filter out stopwords
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        
        return " ".join(filtered_words)
    
    def process_text_file(self, 
                         filepath: str, 
                         delimiter: str = ',', 
                         text_column: int = 1, 
                         label_column: int = 0) -> Tuple[List[str], List[str]]:
        """Process a text file containing articles and labels.
        
        Args:
            filepath: Path to the CSV file
            delimiter: Column delimiter in the file
            text_column: Index of the text column
            label_column: Index of the label column
            
        Returns:
            Tuple of (processed_texts, labels)
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        try:
            import csv
            processed_texts = []
            labels = []
            
            with open(filepath, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter=delimiter)
                next(reader)  # Skip header
                
                for row in reader:
                    if len(row) <= max(text_column, label_column):
                        raise ValueError(f"Invalid row format: {row}")
                    
                    text = self.remove_stopwords(row[text_column])
                    label = row[label_column]
                    
                    processed_texts.append(text)
                    labels.append(label)
            
            logger.info(f"Processed {len(processed_texts)} articles")
            return processed_texts, labels
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise

class TokenizerWrapper:
    """Wrapper class for text tokenization operations."""
    
    def __init__(self, oov_token: str = "<OOV>") -> None:
        """Initialize tokenizer.
        
        Args:
            oov_token: Token to use for out-of-vocabulary words
        """
        self.tokenizer = Tokenizer(oov_token=oov_token)
        self.label_tokenizer = Tokenizer()
        self.is_fitted = False
        self.is_label_fitted = False
    
    def fit_text_tokenizer(self, texts: List[str]) -> None:
        """Fit tokenizer on texts.
        
        Args:
            texts: List of texts to fit on
        """
        self.tokenizer.fit_on_texts(texts)
        self.is_fitted = True
        logger.info(f"Vocabulary size: {len(self.tokenizer.word_index)}")
    
    def fit_label_tokenizer(self, labels: List[str]) -> None:
        """Fit tokenizer on labels.
        
        Args:
            labels: List of labels to fit on
        """
        self.label_tokenizer.fit_on_texts(labels)
        self.is_label_fitted = True
        logger.info(f"Number of classes: {len(self.label_tokenizer.word_index)}")
    
    def get_padded_sequences(self, 
                           texts: List[str], 
                           padding: str = 'post',
                           maxlen: Optional[int] = None) -> np.ndarray:
        """Convert texts to padded sequences.
        
        Args:
            texts: List of texts to convert
            padding: Padding type ('pre' or 'post')
            maxlen: Maximum sequence length
            
        Returns:
            Numpy array of padded sequences
            
        Raises:
            ValueError: If tokenizer is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before generating sequences")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, padding=padding, maxlen=maxlen)
        
        return padded_sequences
    
    def get_label_sequences(self, labels: List[str]) -> List[int]:
        """Convert labels to sequences.
        
        Args:
            labels: List of labels to convert
            
        Returns:
            List of label sequences
            
        Raises:
            ValueError: If label tokenizer is not fitted
        """
        if not self.is_label_fitted:
            raise ValueError("Label tokenizer must be fitted before generating sequences")
        
        return self.label_tokenizer.texts_to_sequences(labels)
    
    @property
    def word_index(self) -> Dict[str, int]:
        """Get word index dictionary.
        
        Returns:
            Dictionary mapping words to indices
        """
        return self.tokenizer.word_index
    
    @property
    def label_word_index(self) -> Dict[str, int]:
        """Get label word index dictionary.
        
        Returns:
            Dictionary mapping labels to indices
        """
        return self.label_tokenizer.word_index 