"""Core implementation of real-time forecasting model."""

import tensorflow as tf
from typing import List, Optional, Tuple, Union

class RealTimeModel:
    """Real-time forecasting model based on LSTM architecture."""
    
    def __init__(
        self,
        input_width: int = 24,
        label_width: int = 1,
        shift: int = 1,
        learning_rate: float = 0.001,
        lstm_units: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        use_residual: bool = True,
        use_attention: bool = True
    ):
        """Initialize model.
        
        Args:
            input_width: Number of input time steps
            label_width: Number of output time steps
            shift: Number of time steps to shift the target
            learning_rate: Learning rate for optimization
            lstm_units: List of LSTM layer sizes
            dropout_rate: Dropout rate for regularization
            use_residual: Whether to use residual connections
            use_attention: Whether to use attention mechanism
        """
        # Validate parameters
        if input_width <= 0:
            raise ValueError("input_width must be positive")
        if label_width <= 0:
            raise ValueError("label_width must be positive")
        if shift <= 0:
            raise ValueError("shift must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.learning_rate = learning_rate
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.use_attention = use_attention
        self.model = None
    
    def build_model(self, num_features: int) -> None:
        """Build model architecture.
        
        Args:
            num_features: Number of input features
        """
        # Input layer
        inputs = tf.keras.layers.Input(shape=(self.input_width, num_features))
        x = inputs
        
        # LSTM layers with residual connections
        lstm_outputs = []
        for i, units in enumerate(self.lstm_units):
            lstm = tf.keras.layers.LSTM(
                units,
                return_sequences=True,
                name=f'lstm_{i+1}'
            )(x)
            
            # Add dropout
            lstm = tf.keras.layers.Dropout(
                self.dropout_rate,
                name=f'dropout_{i+1}'
            )(lstm)
            
            # Add residual connection if requested
            if self.use_residual and i > 0:
                # Project previous output to match current dimensions
                residual = tf.keras.layers.Dense(
                    units,
                    name=f'residual_proj_{i+1}'
                )(x)
                lstm = tf.keras.layers.Add(
                    name=f'residual_add_{i+1}'
                )([lstm, residual])
            
            x = lstm
            lstm_outputs.append(lstm)
        
        # Attention mechanism
        if self.use_attention and len(lstm_outputs) > 1:
            # Multi-head attention
            attention = tf.keras.layers.MultiHeadAttention(
                num_heads=4,
                key_dim=lstm_outputs[-1].shape[-1],
                name='multi_head_attention'
            )(lstm_outputs[-1], lstm_outputs[-1])
            
            # Add & normalize
            x = tf.keras.layers.Add(
                name='attention_add'
            )([x, attention])
            x = tf.keras.layers.LayerNormalization(
                name='attention_norm'
            )(x)
        
        # Output layers
        x = tf.keras.layers.Dense(
            32,
            activation='relu',
            name='dense_1'
        )(x)
        x = tf.keras.layers.Dropout(
            self.dropout_rate,
            name='dropout_final'
        )(x)
        
        # Final output
        outputs = tf.keras.layers.Dense(
            num_features,
            name='output'
        )(x)
        
        # Create model
        self.model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name='real_time_forecaster'
        )
    
    def compile_model(self) -> None:
        """Compile model with optimizer and loss function."""
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.RootMeanSquaredError()
            ]
        )
    
    def summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model must be built before getting summary")
        self.model.summary()
    
    def save(self, path: str) -> None:
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("Model must be built before saving")
        self.model.save(path)
    
    @classmethod
    def load(cls, path: str) -> 'RealTimeModel':
        """Load model from disk.
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded model instance
        """
        model = cls()
        model.model = tf.keras.models.load_model(path)
        return model
    
    def get_config(self) -> dict:
        """Get model configuration.
        
        Returns:
            Model configuration dictionary
        """
        return {
            'input_width': self.input_width,
            'label_width': self.label_width,
            'shift': self.shift,
            'learning_rate': self.learning_rate,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'use_residual': self.use_residual,
            'use_attention': self.use_attention
        }
    
    @classmethod
    def from_config(cls, config: dict) -> 'RealTimeModel':
        """Create model from configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Model instance
        """
        return cls(**config) 