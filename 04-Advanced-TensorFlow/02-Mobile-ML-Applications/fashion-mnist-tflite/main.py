import argparse
import logging
import pathlib
from src.core import FashionMNISTModel
from src.utils import save_test_images, create_interpreter
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train and convert Fashion MNIST model to TFLite')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory to store the dataset')
    parser.add_argument('--export_dir', type=str, default='saved_model/1',
                       help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--optimization', type=str, choices=['LATENCY', 'SIZE', 'NONE'],
                       default='LATENCY', help='TFLite optimization strategy')
    args = parser.parse_args()
    
    # Initialize model
    model = FashionMNISTModel(batch_size=args.batch_size)
    
    # Load and preprocess data
    train_batches, validation_batches, test_batches = model.load_data(args.data_dir)
    
    # Build and train model
    model.build_model()
    history = model.train(train_batches, validation_batches, epochs=args.epochs)
    
    # Convert to TFLite
    tflite_model = model.convert_to_tflite(args.export_dir, args.optimization)
    
    # Save TFLite model
    tflite_model_file = pathlib.Path('./model.tflite')
    tflite_model_file.write_bytes(tflite_model)
    logger.info(f"TFLite model saved to {tflite_model_file}")
    
    # Create interpreter and test
    interpreter, input_index, output_index = create_interpreter(tflite_model)
    
    # Test the model
    predictions = []
    test_labels = []
    test_images = []
    
    for img, label in test_batches.take(50):
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        predictions.append(interpreter.get_tensor(output_index))
        test_labels.append(label[0])
        test_images.append(np.array(img))
    
    # Save test images
    save_test_images(test_batches, model.class_names)
    
    # Plot a test prediction
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    from src.core.visualization import plot_image, plot_value_array
    plot_image(0, predictions, test_labels, test_images, model.class_names)
    plt.subplot(1,2,2)
    plot_value_array(0, predictions, test_labels)
    plt.show()

if __name__ == "__main__":
    main() 