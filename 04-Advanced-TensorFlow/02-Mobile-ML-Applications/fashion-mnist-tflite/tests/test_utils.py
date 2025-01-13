import pytest
import tensorflow as tf
import os
from src.utils import save_test_images, create_interpreter

def test_save_test_images(tmp_path):
    # Create dummy test dataset
    images = tf.ones((5, 28, 28, 1), dtype=tf.float32)
    labels = tf.zeros((5,), dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(1)
    
    # Create dummy class names
    class_names = ['class_0', 'class_1']
    
    # Save images
    output_dir = str(tmp_path / 'test_images')
    save_test_images(dataset, class_names, output_dir, num_images=5)
    
    # Check if images were saved
    assert os.path.exists(output_dir)
    saved_files = os.listdir(output_dir)
    assert len(saved_files) == 5
    assert all(f.endswith('.jpg') for f in saved_files)

def test_create_interpreter():
    # Create a dummy model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Create interpreter
    interpreter, input_index, output_index = create_interpreter(tflite_model)
    
    # Check interpreter
    assert isinstance(interpreter, tf.lite.Interpreter)
    assert isinstance(input_index, int)
    assert isinstance(output_index, int)
    
    # Check tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    assert len(input_details) == 1
    assert len(output_details) == 1
    assert input_details[0]['index'] == input_index
    assert output_details[0]['index'] == output_index 