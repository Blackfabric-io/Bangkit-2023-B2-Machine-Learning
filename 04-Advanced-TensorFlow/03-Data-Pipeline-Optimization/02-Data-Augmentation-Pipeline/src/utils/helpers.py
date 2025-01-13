import tensorflow_datasets as tfds
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def get_dataset_info(data_dir: str) -> Dict[str, Any]:
    """Get information about the Cats vs Dogs dataset.
    
    Args:
        data_dir: Directory where the dataset is stored
        
    Returns:
        Dictionary containing dataset information
    """
    logger.info("Retrieving dataset information...")
    _, info = tfds.load('cats_vs_dogs',
                       data_dir=data_dir,
                       with_info=True)
    
    return {
        'image_shape': info.features['image'].shape,
        'num_classes': info.features['label'].num_classes,
        'splits': {
            'train': info.splits['train'].num_examples,
            'test': info.splits['test'].num_examples if 'test' in info.splits else None
        },
        'description': info.description
    } 