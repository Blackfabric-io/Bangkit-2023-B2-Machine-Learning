from typing import Tuple, Dict, Any, Optional
import logging
import tensorflow as tf
import multiprocessing

logger = logging.getLogger(__name__)

def create_model() -> tf.keras.Model:
    """Creates and compiles a MobileNetV2 model for binary classification.
    
    Returns:
        A compiled Keras model
    """
    try:
        input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
        base_model = tf.keras.applications.MobileNetV2(
            input_tensor=input_layer,
            weights='imagenet',
            include_top=False
        )
        base_model.trainable = False
        
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(2, activation='softmax')(x)
        
        model = tf.keras.models.Model(inputs=input_layer, outputs=x)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['acc']
        )
        return model
        
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise

class ParallelDataPipeline:
    """A class for creating optimized parallel data pipelines."""
    
    def __init__(self, 
                 file_pattern: str,
                 batch_size: int = 32,
                 shuffle_buffer: int = 1024):
        """Initialize the pipeline.
        
        Args:
            file_pattern: Pattern to match TFRecord files
            batch_size: Size of batches to create
            shuffle_buffer: Size of the shuffle buffer
        """
        self.file_pattern = file_pattern
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.feature_description = {
            'image': tf.io.FixedLenFeature((), tf.string, ""),
            'label': tf.io.FixedLenFeature((), tf.int64, -1),
        }
        
    def read_tfrecord(self, serialized_example: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Parse and decode a TFRecord example.
        
        Args:
            serialized_example: A serialized tf.train.Example
            
        Returns:
            Tuple of (image, label) tensors
        """
        try:
            # Parse the example
            example = tf.io.parse_single_example(
                serialized_example, 
                self.feature_description
            )
            
            # Decode and process the image
            image = tf.io.decode_jpeg(example['image'], channels=3)
            image = tf.cast(image, tf.float32)
            image = image / 255.0
            image = tf.image.resize(image, [224, 224])
            
            return image, example['label']
            
        except Exception as e:
            logger.error(f"Error processing TFRecord: {str(e)}")
            raise
            
    def create_dataset(self) -> tf.data.Dataset:
        """Create an optimized dataset from TFRecord files.
        
        Returns:
            A tf.data.Dataset ready for training
        """
        try:
            # List the TFRecord files
            files = tf.data.Dataset.list_files(self.file_pattern)
            
            # Parallelize extraction
            dataset = files.interleave(
                tf.data.TFRecordDataset,
                cycle_length=4,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            
            # Parallelize transformation
            cores = multiprocessing.cpu_count()
            dataset = dataset.map(
                self.read_tfrecord,
                num_parallel_calls=cores
            )
            
            # Cache, shuffle, batch and prefetch
            dataset = dataset.cache()
            dataset = dataset.shuffle(self.shuffle_buffer)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            raise
            
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        try:
            files = tf.data.Dataset.list_files(self.file_pattern)
            num_files = sum(1 for _ in files)
            
            return {
                'num_tfrecord_files': num_files,
                'batch_size': self.batch_size,
                'shuffle_buffer': self.shuffle_buffer,
                'num_parallel_cores': multiprocessing.cpu_count()
            }
            
        except Exception as e:
            logger.error(f"Error getting dataset info: {str(e)}")
            raise 