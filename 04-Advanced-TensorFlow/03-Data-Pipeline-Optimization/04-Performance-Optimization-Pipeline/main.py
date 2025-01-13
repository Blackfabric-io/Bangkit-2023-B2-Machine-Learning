import argparse
import logging
import os
from pathlib import Path

from src.core import ParallelDataPipeline, create_model
from src.utils.helpers import load_dataset, get_tfrecord_pattern, validate_image_shape

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description='Train a model using an optimized parallel data pipeline'
    )
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing the dataset')
    parser.add_argument('--dataset-name', type=str, default='cats_vs_dogs',
                       help='Name of the dataset to use')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--shuffle-buffer', type=int, default=1024,
                       help='Size of shuffle buffer')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs to train')
    
    args = parser.parse_args()
    
    try:
        # Load dataset info
        logger.info("Loading dataset...")
        _, info = load_dataset(
            dataset_name=args.dataset_name,
            data_dir=args.data_dir
        )
        
        # Get TFRecord file pattern
        file_pattern = get_tfrecord_pattern(
            data_dir=args.data_dir,
            dataset_name=args.dataset_name,
            version=str(info.version)
        )
        
        # Create data pipeline
        logger.info("Creating data pipeline...")
        pipeline = ParallelDataPipeline(
            file_pattern=file_pattern,
            batch_size=args.batch_size,
            shuffle_buffer=args.shuffle_buffer
        )
        
        # Create dataset
        dataset = pipeline.create_dataset()
        
        # Validate dataset
        logger.info("Validating dataset...")
        validate_image_shape(dataset)
        
        # Log dataset info
        dataset_info = pipeline.get_dataset_info()
        logger.info(f"Dataset info: {dataset_info}")
        
        # Create and train model
        logger.info("Creating model...")
        model = create_model()
        
        logger.info("Training model...")
        history = model.fit(
            dataset,
            epochs=args.epochs
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 