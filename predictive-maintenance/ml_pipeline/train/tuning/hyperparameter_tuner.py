"""
Hyperparameter Tuning for ML Models
Supports grid search and random search for LSTM and Random Forest
"""

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import itertools

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Hyperparameter tuning framework for both LSTM and Random Forest
    """

    def __init__(self, config: Dict):
        """
        Initialize tuner

        Args:
            config: Configuration dictionary
        """
        self.config = config
        logger.info("Hyperparameter Tuner initialized")

    def tune_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        search_space: Optional[Dict] = None,
        method: str = "random_search",
        n_trials: int = 50,
        cv_folds: int = 5,
    ) -> Dict:
        """
        Tune Random Forest hyperparameters

        Args:
            X_train: Training features
            y_train: Training labels
            search_space: Parameter search space
            method: 'random_search' or 'grid_search'
            n_trials: Number of random search trials
            cv_folds: Cross-validation folds

        Returns:
            Dictionary with best parameters and scores
        """
        rf_config = self.config.get("random_forest", {})
        tuning_config = rf_config.get("hyperparameter_tuning", {})

        # Default search space
        if search_space is None:
            search_space = tuning_config.get(
                "search_space",
                {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [10, 20, 30, 40, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None],
                    "bootstrap": [True, False],
                },
            )

        logger.info(f"Tuning Random Forest with {method}")
        logger.info(f"Search space: {search_space}")

        # Base model
        base_model = RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight="balanced"
        )

        # Search
        if method == "random_search":
            search = RandomizedSearchCV(
                base_model,
                param_distributions=search_space,
                n_iter=n_trials,
                cv=cv_folds,
                scoring="f1_weighted",
                n_jobs=-1,
                verbose=2,
                random_state=42,
            )
        else:  # grid_search
            search = GridSearchCV(
                base_model,
                param_grid=search_space,
                cv=cv_folds,
                scoring="f1_weighted",
                n_jobs=-1,
                verbose=2,
            )

        # Fit search
        search.fit(X_train, y_train)

        results = {
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "cv_results": {
                "mean_test_score": search.cv_results_["mean_test_score"].tolist(),
                "std_test_score": search.cv_results_["std_test_score"].tolist(),
                "params": search.cv_results_["params"],
            },
        }

        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best CV score: {results['best_score']:.4f}")

        return results

    def tune_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        search_space: Optional[Dict] = None,
        n_trials: int = 20,
    ) -> Dict:
        """
        Tune LSTM hyperparameters using random search

        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            search_space: Parameter search space
            n_trials: Number of trials

        Returns:
            Dictionary with best parameters and scores
        """
        from models.lstm_model import LSTMRULPredictor

        lstm_config = self.config.get("lstm", {})
        tuning_config = lstm_config.get("hyperparameter_tuning", {})

        # Default search space
        if search_space is None:
            search_space = tuning_config.get(
                "search_space",
                {
                    "lstm_units_1": [64, 128, 256],
                    "lstm_units_2": [32, 64, 128],
                    "dropout_rate": [0.1, 0.2, 0.3, 0.4],
                    "learning_rate": [0.0001, 0.001, 0.01],
                    "batch_size": [16, 32, 64],
                },
            )

        logger.info(f"Tuning LSTM with random search ({n_trials} trials)")
        logger.info(f"Search space: {search_space}")

        input_shape = (X_train.shape[1], X_train.shape[2])
        best_score = float("inf")
        best_params = None
        trial_results = []

        # Random search
        for trial in range(n_trials):
            # Sample parameters
            params = {
                "lstm_units_1": np.random.choice(search_space["lstm_units_1"]),
                "lstm_units_2": np.random.choice(search_space["lstm_units_2"]),
                "dropout_rate": np.random.choice(search_space["dropout_rate"]),
                "learning_rate": np.random.choice(search_space["learning_rate"]),
                "batch_size": np.random.choice(search_space["batch_size"]),
            }

            logger.info(f"Trial {trial + 1}/{n_trials}: {params}")

            # Build custom architecture
            custom_arch = {
                "lstm_layers": [
                    {
                        "units": params["lstm_units_1"],
                        "return_sequences": True,
                        "dropout": params["dropout_rate"],
                        "recurrent_dropout": params["dropout_rate"],
                    },
                    {
                        "units": params["lstm_units_2"],
                        "return_sequences": False,
                        "dropout": params["dropout_rate"],
                        "recurrent_dropout": params["dropout_rate"],
                    },
                ],
                "attention": {"enabled": False},
                "dense_layers": [
                    {
                        "units": 64,
                        "activation": "relu",
                        "dropout": params["dropout_rate"],
                    },
                    {"units": 32, "activation": "relu", "dropout": 0.2},
                ],
                "output": {"units": 1, "activation": "linear"},
            }

            # Train model
            try:
                model = LSTMRULPredictor(self.config)
                model.build_model(input_shape, custom_arch)

                # Update learning rate
                import tensorflow as tf

                model.model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=params["learning_rate"]
                    ),
                    loss="mse",
                    metrics=["mae"],
                )

                # Train with reduced epochs for tuning
                history = model.train(
                    X_train, y_train, X_val, y_val, checkpoint_path=None
                )

                # Get validation loss
                val_loss = min(history.history["val_loss"])

                trial_results.append({"params": params, "val_loss": float(val_loss)})

                # Update best
                if val_loss < best_score:
                    best_score = val_loss
                    best_params = params
                    logger.info(f"New best score: {best_score:.4f}")

            except Exception as e:
                logger.error(f"Trial {trial + 1} failed: {e}")
                continue

        results = {
            "best_params": best_params,
            "best_score": float(best_score),
            "trial_results": trial_results,
        }

        logger.info(f"Best LSTM parameters: {best_params}")
        logger.info(f"Best validation loss: {best_score:.4f}")

        return results

    def grid_search_lstm_manual(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_grid: Dict[str, List],
    ) -> Dict:
        """
        Manual grid search for LSTM (exhaustive)

        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            param_grid: Grid of parameters to search

        Returns:
            Dictionary with best parameters and scores
        """
        from models.lstm_model import LSTMRULPredictor

        logger.info("Manual grid search for LSTM")
        logger.info(f"Parameter grid: {param_grid}")

        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))

        logger.info(f"Total combinations: {len(combinations)}")

        input_shape = (X_train.shape[1], X_train.shape[2])
        best_score = float("inf")
        best_params = None
        all_results = []

        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            logger.info(f"Combination {i + 1}/{len(combinations)}: {params}")

            # Build and train model
            try:
                custom_arch = {
                    "lstm_layers": [
                        {
                            "units": params.get("lstm_units_1", 128),
                            "return_sequences": True,
                            "dropout": params.get("dropout_rate", 0.2),
                        },
                        {
                            "units": params.get("lstm_units_2", 64),
                            "return_sequences": False,
                            "dropout": params.get("dropout_rate", 0.2),
                        },
                    ],
                    "attention": {"enabled": False},
                    "dense_layers": [{"units": 32, "activation": "relu"}],
                    "output": {"units": 1, "activation": "linear"},
                }

                model = LSTMRULPredictor(self.config)
                model.build_model(input_shape, custom_arch)

                # Train
                history = model.train(X_train, y_train, X_val, y_val)
                val_loss = min(history.history["val_loss"])

                all_results.append({"params": params, "val_loss": float(val_loss)})

                if val_loss < best_score:
                    best_score = val_loss
                    best_params = params
                    logger.info(f"New best: {best_score:.4f}")

            except Exception as e:
                logger.error(f"Combination failed: {e}")
                continue

        results = {
            "best_params": best_params,
            "best_score": float(best_score),
            "all_results": all_results,
        }

        return results
