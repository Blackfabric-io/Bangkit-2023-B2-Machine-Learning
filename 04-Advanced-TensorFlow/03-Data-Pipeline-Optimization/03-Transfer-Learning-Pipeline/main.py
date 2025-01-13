import argparse
import logging
from pathlib import Path

from src.core import StructuredDataModel, df_to_dataset
from src.utils.helpers import load_and_preprocess_data, get_feature_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train a structured data classifier')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to the CSV data file')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train')
    parser.add_argument('--hidden-units', type=int, nargs='+', default=[128, 128],
                       help='Hidden layer sizes')
    parser.add_argument('--embedding-dim', type=int, default=8,
                       help='Dimension for embedding categorical features')
    
    args = parser.parse_args()
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    train_df, val_df, test_df = load_and_preprocess_data(args.data_path)
    
    # Get feature information
    numeric_features, categorical_features = get_feature_info(train_df)
    logger.info(f"Found {len(numeric_features)} numeric features and "
                f"{len(categorical_features)} categorical features")
    
    # Create embedding dimensions dict for categorical features
    embedding_dims = {col: args.embedding_dim for col in categorical_features.keys()}
    
    # Create datasets
    train_ds = df_to_dataset(train_df, batch_size=args.batch_size)
    val_ds = df_to_dataset(val_df, shuffle=False, batch_size=args.batch_size)
    test_ds = df_to_dataset(test_df, shuffle=False, batch_size=args.batch_size)
    
    # Create and train model
    logger.info("Creating model...")
    model = StructuredDataModel(numeric_features=numeric_features,
                              categorical_features=categorical_features,
                              embedding_dims=embedding_dims)
    
    model.build_model(hidden_units=args.hidden_units)
    model.compile_model()
    
    logger.info("Training model...")
    history = model.train(train_ds, validation_ds=val_ds, epochs=args.epochs)
    
    # Evaluate model
    logger.info("Evaluating model...")
    loss, accuracy = model.evaluate(test_ds)
    logger.info(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 