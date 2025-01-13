import pytest
import tensorflow as tf
import os

from src.core import ParallelDataPipeline, create_model

@pytest.fixture
def mock_tfrecord(tmp_path):
    """Create a mock TFRecord file for testing."""
    file_path = os.path.join(tmp_path, "test.tfrecord")
    
    # Create a mock image
    image = tf.random.uniform((64, 64, 3), dtype=tf.uint8)
    image_bytes = tf.io.encode_jpeg(image).numpy()
    
    # Create example
    feature = {
        'image': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_bytes])
        ),
        'label': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[1])
        )
    }
    example = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    
    # Write TFRecord
    with tf.io.TFRecordWriter(file_path) as writer:
        writer.write(example.SerializeToString())
    
    return file_path

def test_create_model():
    """Test model creation."""
    model = create_model()
    
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) > 0
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 2)

def test_parallel_pipeline_initialization():
    """Test pipeline initialization."""
    pipeline = ParallelDataPipeline(
        file_pattern="test*.tfrecord",
        batch_size=16,
        shuffle_buffer=512
    )
    
    assert pipeline.file_pattern == "test*.tfrecord"
    assert pipeline.batch_size == 16
    assert pipeline.shuffle_buffer == 512
    assert isinstance(pipeline.feature_description, dict)

def test_read_tfrecord(mock_tfrecord):
    """Test TFRecord reading and parsing."""
    pipeline = ParallelDataPipeline(file_pattern=mock_tfrecord)
    
    # Create a dataset with one example
    dataset = tf.data.TFRecordDataset([mock_tfrecord])
    example = next(iter(dataset))
    
    # Process the example
    image, label = pipeline.read_tfrecord(example)
    
    assert isinstance(image, tf.Tensor)
    assert isinstance(label, tf.Tensor)
    assert image.shape == (224, 224, 3)
    assert label.dtype == tf.int64

def test_create_dataset(mock_tfrecord):
    """Test dataset creation with all optimizations."""
    pipeline = ParallelDataPipeline(
        file_pattern=mock_tfrecord,
        batch_size=4
    )
    
    dataset = pipeline.create_dataset()
    
    assert isinstance(dataset, tf.data.Dataset)
    
    # Check the first batch
    images, labels = next(iter(dataset))
    assert images.shape[0] <= 4  # Batch size
    assert images.shape[1:] == (224, 224, 3)  # Image dimensions
    assert labels.shape[0] == images.shape[0]  # Matching batch sizes

def test_get_dataset_info(mock_tfrecord):
    """Test dataset info retrieval."""
    pipeline = ParallelDataPipeline(file_pattern=mock_tfrecord)
    info = pipeline.get_dataset_info()
    
    assert isinstance(info, dict)
    assert 'num_tfrecord_files' in info
    assert 'batch_size' in info
    assert 'shuffle_buffer' in info
    assert 'num_parallel_cores' in info 