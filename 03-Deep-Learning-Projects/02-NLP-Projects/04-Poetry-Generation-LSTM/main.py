"""Main script for poetry generation."""

import argparse
from src.core.base import PoetryGenerator
from src.utils.helpers import load_poetry, preprocess_corpus, format_poetry

def main():
    """Run poetry generation."""
    parser = argparse.ArgumentParser(description='Poetry Generation with LSTM')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input poetry file')
    parser.add_argument('--vocab_size', type=int, default=5000,
                       help='Maximum vocabulary size')
    parser.add_argument('--embedding_dim', type=int, default=100,
                       help='Embedding dimension')
    parser.add_argument('--rnn_units', type=int, default=150,
                       help='Number of RNN units')
    parser.add_argument('--maxlen', type=int, default=40,
                       help='Maximum sequence length')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Training batch size')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--seed_text', type=str, default="The moonlight shines",
                       help='Seed text for generation')
    parser.add_argument('--num_words', type=int, default=50,
                       help='Number of words to generate')
    parser.add_argument('--words_per_line', type=int, default=8,
                       help='Number of words per line in output')
    args = parser.parse_args()
    
    # Load and preprocess data
    print("Loading data...")
    corpus = load_poetry(args.input_file)
    corpus = preprocess_corpus(corpus)
    
    # Initialize generator
    generator = PoetryGenerator(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        rnn_units=args.rnn_units,
        maxlen=args.maxlen
    )
    
    # Prepare sequences
    print("Preparing sequences...")
    X_train, y_train = generator.prepare_sequences(corpus)
    
    # Create and train model
    print("Creating model...")
    generator.create_model()
    
    print("Training model...")
    history = generator.train(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Generate poetry
    print("\nGenerating poetry...")
    generated_text = generator.generate_poetry(
        args.seed_text,
        next_words=args.num_words,
        temperature=args.temperature
    )
    
    # Format the output
    formatted_poetry = format_poetry(generated_text, args.words_per_line)
    
    print(f"\nSeed text: {args.seed_text}")
    print(f"\nGenerated poetry:\n\n{formatted_poetry}")

if __name__ == "__main__":
    main() 