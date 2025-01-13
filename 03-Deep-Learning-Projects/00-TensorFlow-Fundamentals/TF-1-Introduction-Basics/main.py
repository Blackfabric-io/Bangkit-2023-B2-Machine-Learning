#!/usr/bin/env python3
"""Main script for house price prediction."""

import argparse
import numpy as np
from src import HousePriceModel, scale_price, unscale_price

def main():
    """Run house price prediction model."""
    parser = argparse.ArgumentParser(description='Predict house prices based on number of bedrooms')
    parser.add_argument('--bedrooms', type=float, help='Number of bedrooms to predict price for')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    args = parser.parse_args()
    
    model = HousePriceModel()
    
    if args.train:
        print("Training model...")
        bedrooms, prices = model.prepare_data()
        history = model.train(bedrooms, prices, epochs=args.epochs)
        print(f"Final loss: {history.history['loss'][-1]:.4f}")
    
    if args.bedrooms is not None:
        prediction = model.predict(np.array([args.bedrooms]))
        price = unscale_price(prediction[0, 0])
        print(f"\nPredicted price for {args.bedrooms} bedroom(s):")
        print(f"${price[0]:.1f}k")

if __name__ == "__main__":
    main() 