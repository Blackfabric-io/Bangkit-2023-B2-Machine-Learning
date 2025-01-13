import pytest
import pandas as pd
import tensorflow as tf
from src.core import StructuredDataModel, df_to_dataset

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        'numeric_1': [1, 2, 3, 4, 5],
        'numeric_2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'category_1': ['A', 'B', 'A', 'C', 'B'],
        'target': [0, 1, 0, 1, 1]
    }
    return pd.DataFrame(data)

def test_df_to_dataset(sample_data):
    """Test conversion of DataFrame to tf.data.Dataset."""
    ds = df_to_dataset(sample_data, batch_size=2)
    
    # Check that it returns a Dataset
    assert isinstance(ds, tf.data.Dataset)
    
    # Check the structure of the dataset
    for features, labels in ds.take(1):
        assert isinstance(features, dict)
        assert all(key in features for key in ['numeric_1', 'numeric_2', 'category_1'])
        assert isinstance(labels, tf.Tensor)

def test_structured_data_model():
    """Test StructuredDataModel initialization and methods."""
    numeric_features = ['numeric_1', 'numeric_2']
    categorical_features = {'category_1': ['A', 'B', 'C']}
    embedding_dims = {'category_1': 4}
    
    model = StructuredDataModel(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        embedding_dims=embedding_dims
    )
    
    # Check feature columns were created
    assert len(model.feature_columns) == 3  # 2 numeric + 1 embedding
    
    # Test model building
    keras_model = model.build_model(hidden_units=[64, 32])
    assert isinstance(keras_model, tf.keras.Model)
    assert len(keras_model.layers) == 4  # feature layer + 2 hidden + output
    
    # Test model compilation
    model.compile_model()
    assert model.model.optimizer is not None
    assert model.model.loss is not None

def test_model_training(sample_data):
    """Test model training workflow."""
    model = StructuredDataModel(
        numeric_features=['numeric_1', 'numeric_2'],
        categorical_features={'category_1': ['A', 'B', 'C']}
    )
    
    train_ds = df_to_dataset(sample_data, batch_size=2)
    val_ds = df_to_dataset(sample_data, shuffle=False, batch_size=2)
    
    model.build_model(hidden_units=[64])
    model.compile_model()
    
    history = model.train(train_ds, validation_ds=val_ds, epochs=2)
    assert isinstance(history, tf.keras.callbacks.History)
    assert len(history.history['loss']) == 2
    
    # Test evaluation
    loss, accuracy = model.evaluate(val_ds)
    assert isinstance(loss, float)
    assert isinstance(accuracy, float) 