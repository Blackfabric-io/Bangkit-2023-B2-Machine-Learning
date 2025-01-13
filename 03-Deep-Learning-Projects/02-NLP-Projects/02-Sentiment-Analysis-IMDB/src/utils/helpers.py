"""Helper functions for text preprocessing and data loading."""

import csv
from typing import List, Tuple

def remove_stopwords(sentence: str) -> str:
    """Remove stopwords from a sentence.
    
    Args:
        sentence: Input text
        
    Returns:
        Text with stopwords removed
    """
    stopwords = [
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
    
    sentence = sentence.lower()
    words = sentence.split()
    no_words = [w for w in words if w not in stopwords]
    
    return " ".join(no_words)

def load_data(filename: str) -> Tuple[List[str], List[str]]:
    """Load data from a CSV file.
    
    Args:
        filename: Path to the CSV file
        
    Returns:
        Tuple of (sentences, labels)
    """
    sentences = []
    labels = []
    
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip header
        for row in reader:
            labels.append(row[0])
            sentence = row[1]
            sentence = remove_stopwords(sentence)
            sentences.append(sentence)
            
    return sentences, labels

def split_data(sentences: List[str], 
               labels: List[str], 
               train_split: float = 0.8) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split data into training and validation sets.
    
    Args:
        sentences: List of preprocessed sentences
        labels: List of corresponding labels
        train_split: Proportion of data to use for training
        
    Returns:
        Tuple of (train_sentences, train_labels, val_sentences, val_labels)
    """
    size = len(sentences)
    train_size = int(size * train_split)
    
    train_sentences = sentences[:train_size]
    train_labels = labels[:train_size]
    val_sentences = sentences[train_size:]
    val_labels = labels[train_size:]
    
    return train_sentences, train_labels, val_sentences, val_labels 