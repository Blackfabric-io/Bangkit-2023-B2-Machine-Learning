from typing import Optional, Tuple
import os
import logging
import tensorflow as tf
import tensorflow_datasets as tfds

logger = logging.getLogger(__name__)

def load_dataset(dataset_name: str,
                data_dir: str,
                split: str = tfds.Split.TRAIN,
                with_info: bool = True) -> Tuple[tf.data.Dataset, Optional[tfds.core.DatasetInfo]]:
    """Load a dataset using TensorFlow Datasets.
    
    Args:
        dataset_name: Name of the dataset to load
        data_dir: Directory to store/load the data
        split: Which split of the data to load
        with_info: Whether to return the dataset info
        
    Returns:
        Tuple of (dataset, info) if with_info=True, else just dataset
    """
    try:
        logger.info(f"Loading dataset {dataset_name} from {data_dir}")
        return tfds.load(
            name=dataset_name,
            split=split,
            with_info=with_info,
            data_dir=data_dir
        )
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def get_tfrecord_pattern(data_dir: str,
                        dataset_name: str,
                        version: str,
                        split: str = 'train') -> str:
    """Generate the file pattern for TFRecord files.
    
    Args:
        data_dir: Base directory containing the data
        dataset_name: Name of the dataset
        version: Version of the dataset
        split: Which split to use (train/test)
        
    Returns:
        File pattern string for matching TFRecord files
    """
    try:
        pattern = os.path.join(
            data_dir,
            dataset_name,
            version,
            f"{dataset_name}-{split}.tfrecord*"
        )
        
        if not tf.io.gfile.glob(pattern):
            raise FileNotFoundError(f"No TFRecord files found matching pattern: {pattern}")
            
        return pattern
        
    except Exception as e:
        logger.error(f"Error generating TFRecord pattern: {str(e)}")
        raise

def validate_image_shape(dataset: tf.data.Dataset,
                        expected_shape: Tuple[int, int, int] = (224, 224, 3)) -> bool:
    """Validate that images in the dataset have the expected shape.
    
    Args:
        dataset: Dataset to validate
        expected_shape: Expected shape of the images
        
    Returns:
        True if validation passes, raises error otherwise
    """
    try:
        for images, _ in dataset.take(1):
            if images.shape[1:] != expected_shape:
                raise ValueError(
                    f"Invalid image shape. Expected {expected_shape}, "
                    f"got {images.shape[1:]}"
                )
        return True
        
    except Exception as e:
        logger.error(f"Error validating dataset: {str(e)}")
        raise 