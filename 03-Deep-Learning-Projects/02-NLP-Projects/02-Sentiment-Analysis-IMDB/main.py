"""Main script for sentiment analysis."""

import argparse
from src.core.base import SentimentAnalyzer
from src.utils.helpers import load_data, split_data

def main():
    """Run sentiment analysis on IMDB dataset."""
    parser = argparse.ArgumentParser(description='Sentiment Analysis on IMDB Dataset')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the CSV data file')
    parser.add_argument('--num_words', type=int, default=1000,
                       help='Maximum number of words to keep')
    parser.add_argument('--embedding_dim', type=int, default=16,
                       help='Dimensionality of the dense embedding')
    parser.add_argument('--maxlen', type=int, default=120,
                       help='Maximum length of all sequences')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    args = parser.parse_args()
    
    # Load and preprocess data
    print("Loading data...")
    sentences, labels = load_data(args.data_path)
    
    # Split data
    train_sentences, train_labels, val_sentences, val_labels = split_data(
        sentences, labels
    )
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(
        num_words=args.num_words,
        embedding_dim=args.embedding_dim,
        maxlen=args.maxlen
    )
    
    # Prepare data
    print("Preparing data...")
    X_train, y_train = analyzer.prepare_data(train_sentences, train_labels)
    X_val, y_val = analyzer.prepare_data(val_sentences, val_labels)
    
    # Create and train model
    print("Creating model...")
    num_classes = len(set(labels))
    analyzer.create_model(num_classes)
    
    print("Training model...")
    history = analyzer.train(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Example prediction
    print("\nExample predictions:")
    example_texts = [
        "This movie was fantastic! I really enjoyed it",
        "What a terrible waste of time. I hated every minute"
    ]
    predictions = analyzer.predict(example_texts)
    for text, pred in zip(example_texts, predictions):
        print(f"Text: {text}")
        print(f"Prediction: {pred.argmax()}\n")

if __name__ == "__main__":
    main() 