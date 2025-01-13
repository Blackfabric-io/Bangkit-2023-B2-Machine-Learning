import argparse
import logging
import os
from src.core import CatDogClassifier
from src.utils import get_dataset_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train Cats vs Dogs classifier using transfer learning')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory to store the dataset')
    parser.add_argument('--model_path', type=str, default='data/mobilenet_v2_feature_vector',
                       help='Path to MobileNet model')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--num_examples', type=int, default=500,
                       help='Number of examples to use for training')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Get dataset info
    dataset_info = get_dataset_info(args.data_dir)
    logger.info(f"Dataset info: {dataset_info}")
    
    # Initialize model
    model = CatDogClassifier(batch_size=args.batch_size,
                            num_examples=args.num_examples)
    
    # Load feature extractor
    model.load_feature_extractor(args.model_path)
    
    # Load and preprocess data
    train_batches, validation_batches, test_batches = model.load_data(args.data_dir)
    
    # Build and train model
    model.build_model()
    model.get_model_info()
    
    # Train model
    history = model.train(train_batches, validation_batches, epochs=args.epochs)
    
    # Evaluate model
    metrics = model.evaluate(test_batches)
    logger.info("Training completed!")
    logger.info(f"Test loss: {metrics[0]:.4f}")
    logger.info(f"Test accuracy: {metrics[1]:.4f}")

if __name__ == "__main__":
    main() 