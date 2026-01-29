"""
LSTM Model for RUL Prediction
Deep learning model for remaining useful life estimation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from typing import Dict, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AttentionLayer(layers.Layer):
    """
    Attention mechanism for LSTM
    Allows model to focus on important time steps
    """

    def __init__(self, units: int, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W = None
        self.b = None
        self.u = None

    def build(self, input_shape):
        # Weight matrices
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            name="attention_context",
            shape=(self.units,),
            initializer="glorot_uniform",
            trainable=True,
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # x shape: (batch_size, time_steps, features)

        # Score calculation
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        # score shape: (batch_size, time_steps, units)

        # Attention weights
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        # attention_weights shape: (batch_size, time_steps)

        attention_weights = tf.expand_dims(attention_weights, -1)
        # attention_weights shape: (batch_size, time_steps, 1)

        # Context vector
        context_vector = tf.reduce_sum(x * attention_weights, axis=1)
        # context_vector shape: (batch_size, features)

        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class LSTMRULPredictor:
    """
    LSTM-based model for RUL prediction
    """

    def __init__(self, config: Dict):
        """
        Initialize LSTM model

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.lstm_config = config.get("lstm", {})
        self.model = None
        self.history = None

        logger.info("LSTM RUL Predictor initialized")

    def build_model(
        self, input_shape: Tuple[int, int], custom_architecture: Optional[Dict] = None
    ) -> Model:
        """
        Build LSTM model architecture

        Args:
            input_shape: (sequence_length, n_features)
            custom_architecture: Optional custom architecture config

        Returns:
            Compiled Keras model
        """
        arch_config = custom_architecture or self.lstm_config.get("architecture", {})

        # Input layer
        inputs = keras.Input(shape=input_shape, name="sequence_input")
        x = inputs

        # LSTM layers
        lstm_layers_config = arch_config.get("lstm_layers", [])

        for i, lstm_config in enumerate(lstm_layers_config):
            x = layers.LSTM(
                units=lstm_config["units"],
                return_sequences=lstm_config.get("return_sequences", False),
                dropout=lstm_config.get("dropout", 0.2),
                recurrent_dropout=lstm_config.get("recurrent_dropout", 0.2),
                name=f"lstm_{i + 1}",
            )(x)

        # Attention mechanism
        attention_config = arch_config.get("attention", {})
        if attention_config.get("enabled", False):
            # Need sequences for attention - rebuild last LSTM with return_sequences=True
            if len(lstm_layers_config) > 0:
                x = inputs
                for i, lstm_config in enumerate(lstm_layers_config[:-1]):
                    x = layers.LSTM(
                        units=lstm_config["units"],
                        return_sequences=True,
                        dropout=lstm_config.get("dropout", 0.2),
                        recurrent_dropout=lstm_config.get("recurrent_dropout", 0.2),
                        name=f"lstm_{i + 1}",
                    )(x)

                # Last LSTM with return_sequences=True for attention
                last_lstm = lstm_layers_config[-1]
                x = layers.LSTM(
                    units=last_lstm["units"],
                    return_sequences=True,
                    dropout=last_lstm.get("dropout", 0.2),
                    recurrent_dropout=last_lstm.get("recurrent_dropout", 0.2),
                    name=f"lstm_{len(lstm_layers_config)}",
                )(x)

                # Apply attention
                x = AttentionLayer(
                    units=attention_config.get("units", 64), name="attention"
                )(x)

        # Dense layers
        dense_layers_config = arch_config.get("dense_layers", [])
        for i, dense_config in enumerate(dense_layers_config):
            x = layers.Dense(
                units=dense_config["units"],
                activation=dense_config.get("activation", "relu"),
                name=f"dense_{i + 1}",
            )(x)

            dropout = dense_config.get("dropout", 0.0)
            if dropout > 0:
                x = layers.Dropout(dropout, name=f"dropout_{i + 1}")(x)

        # Output layer
        output_config = arch_config.get("output", {"units": 1, "activation": "linear"})
        outputs = layers.Dense(
            units=output_config["units"],
            activation=output_config["activation"],
            name="rul_output",
        )(x)

        # Create model
        model = Model(inputs=inputs, outputs=outputs, name="lstm_rul_predictor")

        # Compile model
        training_config = self.lstm_config.get("training", {})
        optimizer = keras.optimizers.Adam(
            learning_rate=training_config.get("learning_rate", 0.001)
        )

        model.compile(
            optimizer=optimizer,
            loss=training_config.get("loss", "mse"),
            metrics=training_config.get("metrics", ["mae", "mse"]),
        )

        self.model = model
        logger.info(f"LSTM model built. Total parameters: {model.count_params()}")

        return model

    def get_callbacks(self, checkpoint_path: Optional[str] = None) -> list:
        """
        Get training callbacks

        Args:
            checkpoint_path: Path to save model checkpoints

        Returns:
            List of Keras callbacks
        """
        training_config = self.lstm_config.get("training", {})
        callbacks = []

        # Early stopping
        early_stop_config = training_config.get("early_stopping", {})
        if early_stop_config.get("enabled", True):
            callbacks.append(
                EarlyStopping(
                    monitor=early_stop_config.get("monitor", "val_loss"),
                    patience=early_stop_config.get("patience", 20),
                    restore_best_weights=early_stop_config.get(
                        "restore_best_weights", True
                    ),
                    min_delta=early_stop_config.get("min_delta", 0.001),
                    verbose=1,
                )
            )

        # Learning rate schedule
        lr_schedule_config = training_config.get("lr_schedule", {})
        if lr_schedule_config.get("enabled", True):
            callbacks.append(
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=lr_schedule_config.get("factor", 0.5),
                    patience=lr_schedule_config.get("patience", 10),
                    min_lr=lr_schedule_config.get("min_lr", 0.00001),
                    verbose=1,
                )
            )

        # Model checkpoint
        if checkpoint_path:
            callbacks.append(
                ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1,
                )
            )

        return callbacks

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        checkpoint_path: Optional[str] = None,
    ) -> keras.callbacks.History:
        """
        Train LSTM model

        Args:
            X_train: Training sequences (n_samples, sequence_length, n_features)
            y_train: Training targets (n_samples,)
            X_val: Validation sequences
            y_val: Validation targets
            checkpoint_path: Path to save checkpoints

        Returns:
            Training history
        """
        if self.model is None:
            # Build model if not already built
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)

        training_config = self.lstm_config.get("training", {})

        # Validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Training
        logger.info(
            f"Training LSTM model. Train samples: {len(X_train)}, Val samples: {len(X_val) if X_val is not None else 0}"
        )

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=training_config.get("epochs", 200),
            batch_size=training_config.get("batch_size", 32),
            callbacks=self.get_callbacks(checkpoint_path),
            verbose=1,
        )

        logger.info("Training completed")

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict RUL

        Args:
            X: Input sequences (n_samples, sequence_length, n_features)

        Returns:
            Predicted RUL values
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")

        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test data

        Args:
            X_test: Test sequences
            y_test: Test targets

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")

        # Get predictions
        y_pred = self.predict(X_test)

        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))

        # R-squared
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # MAPE (avoid division by zero)
        mask = y_test != 0
        mape = (
            np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
            if np.any(mask)
            else 0
        )

        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
        }

        logger.info(f"Evaluation metrics: {metrics}")

        return metrics

    def save_model(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")

        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from file"""
        self.model = keras.models.load_model(
            filepath, custom_objects={"AttentionLayer": AttentionLayer}
        )
        logger.info(f"Model loaded from {filepath}")

    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if self.model is None:
            return "Model not built"

        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return "\n".join(summary_list)
