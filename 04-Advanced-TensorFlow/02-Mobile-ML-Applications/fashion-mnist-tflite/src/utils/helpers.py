import tensorflow as tf
from PIL import Image
import os
from typing import Tuple, List, Any
import logging

logger = logging.getLogger(__name__)

def save_test_images(test_batches: tf.data.Dataset, class_names: List[str], 
                    output_dir: str = 'test_images', num_images: int = 50) -> None:
    """Save test images to disk.
    
    Args:
        test_batches: Test dataset
        class_names: List of class names
        output_dir: Directory to save images
        num_images: Number of images to save
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving {num_images} test images to {output_dir}...")
    
    for index, (image, label) in enumerate(test_batches.take(num_images)):
        image = tf.cast(image * 255.0, tf.uint8)
        image = tf.squeeze(image).numpy()
        pil_image = Image.fromarray(image)
        pil_image.save(os.path.join(output_dir, 
                                   f"{class_names[label[0]].lower()}_{index}.jpg"))

def create_interpreter(tflite_model: bytes) -> Tuple[tf.lite.Interpreter, int, int]:
    """Create a TFLite interpreter.
    
    Args:
        tflite_model: TFLite model as bytes
        
    Returns:
        Tuple of (interpreter, input_index, output_index)
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    return interpreter, input_index, output_index 