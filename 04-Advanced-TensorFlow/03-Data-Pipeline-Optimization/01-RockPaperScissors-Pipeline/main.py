import argparse
import logging
from src.core import RockPaperScissorsModel
from src.utils import get_dataset_info
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train Rock Paper Scissors classifier')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory to store the dataset')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Get dataset info
    dataset_info = get_dataset_info(args.data_dir)
    logger.info(f"Dataset info: {dataset_info}")
    
    # Initialize model
    model = RockPaperScissorsModel(batch_size=args.batch_size)
    
    # Load and preprocess data
    train_batches, validation_batches = model.load_data(args.data_dir)
    
    # Build and train model
    model.build_model()
    model.get_model_info()
    
    # Train model
    history = model.train(train_batches, validation_batches, epochs=args.epochs)
    
    # Print final metrics
    logger.info("Training completed!")
    logger.info(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    logger.info(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

if __name__ == "__main__":
    main() 