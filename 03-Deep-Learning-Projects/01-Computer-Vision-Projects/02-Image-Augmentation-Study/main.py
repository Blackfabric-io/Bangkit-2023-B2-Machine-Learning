#!/usr/bin/env python3
"""Command line interface for image augmentation."""

import os
import argparse
import logging
from datetime import datetime

from src.core import ImageAugmenter, AugmentationConfig
from src.utils import load_image, save_image, visualize_augmentations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Apply image augmentation techniques'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input image file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory to save augmented images'
    )
    
    parser.add_argument(
        '--num-augmentations',
        type=int,
        default=5,
        help='Number of augmented versions to generate'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize augmentations'
    )
    
    # Geometric transformation parameters
    parser.add_argument('--rotation-range', type=float, default=30.0)
    parser.add_argument('--width-shift-range', type=float, default=0.2)
    parser.add_argument('--height-shift-range', type=float, default=0.2)
    parser.add_argument('--shear-range', type=float, default=0.2)
    parser.add_argument('--zoom-range', type=float, default=0.2)
    parser.add_argument('--horizontal-flip', action='store_true')
    parser.add_argument('--vertical-flip', action='store_true')
    
    # Color transformation parameters
    parser.add_argument('--brightness-min', type=float, default=0.8)
    parser.add_argument('--brightness-max', type=float, default=1.2)
    parser.add_argument('--contrast-min', type=float, default=0.8)
    parser.add_argument('--contrast-max', type=float, default=1.2)
    parser.add_argument('--saturation-min', type=float, default=0.8)
    parser.add_argument('--saturation-max', type=float, default=1.2)
    parser.add_argument('--hue-range', type=float, default=0.1)
    
    # Noise parameters
    parser.add_argument('--noise-stddev', type=float, default=0.01)
    parser.add_argument('--blur-min', type=float, default=0.0)
    parser.add_argument('--blur-max', type=float, default=1.0)
    
    return parser.parse_args()

def main():
    """Main augmentation function."""
    args = parse_args()
    
    # Load input image
    logger.info("Loading input image...")
    try:
        image = load_image(args.input)
    except ValueError as e:
        logger.error("Failed to load image: %s", str(e))
        return 1
    
    # Create augmentation config
    config = AugmentationConfig(
        rotation_range=args.rotation_range,
        width_shift_range=args.width_shift_range,
        height_shift_range=args.height_shift_range,
        shear_range=args.shear_range,
        zoom_range=args.zoom_range,
        horizontal_flip=args.horizontal_flip,
        vertical_flip=args.vertical_flip,
        brightness_range=(args.brightness_min, args.brightness_max),
        contrast_range=(args.contrast_min, args.contrast_max),
        saturation_range=(args.saturation_min, args.saturation_max),
        hue_range=args.hue_range,
        noise_stddev=args.noise_stddev,
        gaussian_blur_range=(args.blur_min, args.blur_max)
    )
    
    # Create augmenter
    logger.info("Creating augmenter...")
    try:
        augmenter = ImageAugmenter(config)
    except ValueError as e:
        logger.error("Invalid augmentation configuration: %s", str(e))
        return 1
    
    # Generate augmentations
    logger.info("Generating %d augmented images...", args.num_augmentations)
    augmented_images = []
    for i in range(args.num_augmentations):
        try:
            augmented = augmenter.augment(image, seed=i)
            augmented_images.append(augmented)
        except Exception as e:
            logger.error("Failed to generate augmentation %d: %s", i+1, str(e))
            continue
    
    # Save augmented images
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    logger.info("Saving augmented images...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, aug_img in enumerate(augmented_images):
        output_path = os.path.join(
            args.output_dir,
            f"augmented_{timestamp}_{i+1}.jpg"
        )
        try:
            save_image(aug_img, output_path)
        except ValueError as e:
            logger.error("Failed to save augmentation %d: %s", i+1, str(e))
            continue
    
    # Visualize if requested
    if args.visualize:
        logger.info("Visualizing augmentations...")
        visualize_augmentations(image, augmented_images)
    
    logger.info("Done!")
    return 0

if __name__ == '__main__':
    exit(main()) 