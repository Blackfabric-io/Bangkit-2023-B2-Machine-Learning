"""Main script for Shakespeare text generation."""

import argparse
from src.core.base import TextGenerator
from src.utils.helpers import load_text, clean_text

def main():
    """Run Shakespeare text generation."""
    parser = argparse.ArgumentParser(description='Shakespeare Text Generation')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input text file')
    parser.add_argument('--vocab_size', type=int, default=1000,
                       help='Maximum vocabulary size')
    parser.add_argument('--embedding_dim', type=int, default=100,
                       help='Embedding dimension')
    parser.add_argument('--rnn_units', type=int, default=256,
                       help='Number of RNN units')
    parser.add_argument('--maxlen', type=int, default=40,
                       help='Maximum sequence length')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Training batch size')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--seed_text', type=str, default="To be or not to be",
                       help='Seed text for generation')
    parser.add_argument('--num_words', type=int, default=100,
                       help='Number of words to generate')
    args = parser.parse_args()
    
    # Load and preprocess data
    print("Loading data...")
    text = load_text(args.input_file)
    text = clean_text(text)
    
    # Initialize generator
    generator = TextGenerator(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        rnn_units=args.rnn_units,
        maxlen=args.maxlen
    )
    
    # Prepare data
    print("Preparing data...")
    X_train, y_train = generator.prepare_data(text)
    
    # Create and train model
    print("Creating model...")
    generator.create_model(temperature=args.temperature)
    
    print("Training model...")
    history = generator.train(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Generate text
    print("\nGenerating text...")
    generated_text = generator.generate_text(
        args.seed_text,
        next_words=args.num_words,
        temperature=args.temperature
    )
    
    print(f"\nSeed text: {args.seed_text}")
    print(f"Generated text:\n{generated_text}")

if __name__ == "__main__":
    main() 