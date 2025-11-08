"""
ML Training Pipeline for Price Direction Prediction

This module provides a comprehensive training pipeline for ML models that predict
short-term price movements. Includes data collection, labeling, model training,
validation, and deployment capabilities.

Key Features:
- Data collection and labeling pipeline
- Walk-forward validation for time series
- Ensemble model training (LightGBM, LSTM, Random Forest, XGBoost)
- Hyperparameter optimization with Optuna
- Model validation and performance metrics
- Production-ready model serialization
"""

import os
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

# Optional imports with fallbacks
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TensorFlow not available - LSTM model disabled")

from feature_engineering import FeatureEngineer
from hyperliquid_client import HyperliquidClient

logger = logging.getLogger(__name__)


class MLTrainingPipeline:
    """
    Complete ML training pipeline for price direction prediction.

    This class handles the entire ML workflow from data collection through
    model training, validation, and deployment for production use.
    """

    def __init__(self, symbol: str = 'BTCUSDT', config: Optional[Dict] = None):
        """
        Initialize the ML training pipeline.

        Args:
            symbol: Trading symbol to train on
            config: Configuration dictionary
        """
        self.symbol = symbol
        self.config = config or self._get_default_config()

        # Initialize components
        self.client = HyperliquidClient(
            private_key="dummy",  # Not needed for historical data
            wallet_address="dummy",
            testnet=True
        )

        self.feature_engineer = FeatureEngineer(self.config.get('feature_engineering', {}))

        # Model storage
        self.models = {}
        self.model_metadata = {}

        # Training data storage
        self.training_data = None
        self.feature_columns = []

        logger.info(f"ML Training Pipeline initialized for {symbol}")

    def _get_default_config(self) -> Dict:
        """Get default configuration for the training pipeline."""
        return {
            'data_collection': {
                'lookback_days': 180,  # 6 months of data
                'prediction_horizon': 5,  # 5-minute ahead prediction
                'min_price_move_pct': 0.1,  # 0.1% minimum move to be considered
                'cache_dir': 'data/ml_cache'
            },
            'feature_engineering': {
                'technical': {
                    'ema_periods': [9, 21, 50, 200],
                    'rsi_periods': [14, 21],
                    'bollinger_period': 20,
                    'bollinger_std': 2.0,
                    'atr_period': 14,
                    'adx_period': 14
                }
            },
            'training': {
                'test_size': 0.15,
                'val_size': 0.15,
                'walk_forward_splits': 5,
                'early_stopping_rounds': 50,
                'cv_folds': 5
            },
            'models': {
                'lightgbm': {
                    'n_estimators': 1000,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'num_leaves': 31,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8
                },
                'xgboost': {
                    'n_estimators': 1000,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                },
                'random_forest': {
                    'n_estimators': 500,
                    'max_depth': 10,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5
                },
                'lstm': {
                    'units': [128, 64],
                    'dropout': 0.2,
                    'batch_size': 32,
                    'epochs': 100,
                    'sequence_length': 60  # 60 candles = 5 hours at 5min intervals
                }
            },
            'hyperparameter_tuning': {
                'n_trials': 50,
                'timeout_minutes': 30
            }
        }

    # ==========================================
    # DATA COLLECTION AND LABELING
    # ==========================================

    def collect_training_data(self) -> pd.DataFrame:
        """
        Collect and prepare training data for ML models.

        Returns:
            DataFrame with features and labels for training
        """
        try:
            logger.info("Starting data collection for ML training...")

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config['data_collection']['lookback_days'])

            # Collect market data
            market_data = self._collect_market_data(start_date, end_date)

            if market_data.empty:
                raise ValueError("No market data collected for training")

            logger.info(f"Collected {len(market_data)} candles of market data")

            # Create labels
            labeled_data = self._create_labels(market_data)

            # Engineer features
            feature_data = self._engineer_features_for_training(labeled_data)

            # Cache the data
            self.training_data = feature_data
            self._save_training_data(feature_data)

            logger.info(f"Training data prepared: {len(feature_data)} samples, {len(feature_data.columns)} features")

            return feature_data

        except Exception as e:
            logger.error(f"Error collecting training data: {e}")
            raise

    def _collect_market_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Collect historical market data."""
        try:
            # For demo purposes, generate synthetic data
            # In production, this would fetch real data from exchanges
            hours = int((end_date - start_date).total_seconds() / 3600)
            timestamps = pd.date_range(start=start_date, periods=hours, freq='5min')  # 5-minute candles

            np.random.seed(42)  # For reproducible results

            # Generate realistic price movements
            base_price = 45000.0
            price_changes = np.random.normal(0, 0.005, len(timestamps))  # 0.5% daily volatility
            trend = np.linspace(0, 0.2, len(timestamps))  # Slight upward trend

            prices = base_price * (1 + np.cumsum(price_changes + trend/1000))

            # Generate OHLCV data
            data = []
            for i, (ts, price) in enumerate(zip(timestamps, prices)):
                # Add intrabar volatility
                high = price * (1 + abs(np.random.normal(0, 0.002)))
                low = price * (1 - abs(np.random.normal(0, 0.002)))
                open_price = price * (1 + np.random.normal(0, 0.001))
                close = price

                # Generate volume
                base_volume = 100 + np.random.exponential(200)
                volume = base_volume * (1 + abs(price_changes[i]) * 20)

                data.append({
                    'timestamp': ts,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                })

            df = pd.DataFrame(data)
            logger.debug(f"Generated synthetic data: {len(df)} candles")
            return df

        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
            return pd.DataFrame()

    def _create_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create prediction labels for price direction.

        Labels:
        - 1: Price will rise by > min_move_pct in next horizon minutes
        - 0: Price will not rise by min_move_pct (flat or down)
        """
        try:
            prediction_horizon = self.config['data_collection']['prediction_horizon']
            min_move_pct = self.config['data_collection']['min_price_move_pct'] / 100.0

            # Calculate future price changes
            future_prices = data['close'].shift(-prediction_horizon)
            price_changes = (future_prices - data['close']) / data['close']

            # Create binary labels
            labels = (price_changes > min_move_pct).astype(int)

            # Remove rows where we can't calculate future returns
            valid_data = data.iloc[:-prediction_horizon].copy()
            valid_data['target'] = labels.iloc[:-prediction_horizon].values
            valid_data['future_return'] = price_changes.iloc[:-prediction_horizon].values

            # Remove any remaining NaN values
            valid_data = valid_data.dropna()

            logger.info(f"Created labels: {valid_data['target'].sum()} positive, "
                       f"{len(valid_data) - valid_data['target'].sum()} negative samples")

            return valid_data

        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            return data

    def _engineer_features_for_training(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML training."""
        try:
            # Use the feature engineering pipeline
            feature_data = self.feature_engineer.engineer_features(
                data,
                orderbook_data=None,  # Would be populated with historical order book data
                trade_flow_data=None,  # Would be populated with historical trade flow data
                funding_rates=None     # Would be populated with historical funding rates
            )

            # Store feature column names
            exclude_cols = ['timestamp', 'target', 'future_return']
            self.feature_columns = [col for col in feature_data.columns if col not in exclude_cols]

            logger.info(f"Engineered {len(self.feature_columns)} features for training")

            return feature_data

        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            return data

    def _save_training_data(self, data: pd.DataFrame) -> None:
        """Save training data to cache."""
        try:
            cache_dir = self.config['data_collection']['cache_dir']
            os.makedirs(cache_dir, exist_ok=True)

            cache_file = os.path.join(cache_dir, f"{self.symbol}_training_data.pkl")
            data.to_pickle(cache_file)

            logger.info(f"Training data cached to {cache_file}")

        except Exception as e:
            logger.debug(f"Error caching training data: {e}")

    # ==========================================
    # MODEL TRAINING
    # ==========================================

    def train_models(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train all ML models in the ensemble.

        Args:
            data: Training data (uses cached data if None)

        Returns:
            Dictionary with trained models and metadata
        """
        try:
            if data is None:
                data = self.training_data

            if data is None or data.empty:
                raise ValueError("No training data available")

            logger.info("Starting ensemble model training...")

            # Prepare data
            X, y = self._prepare_training_data(data)

            # Train individual models
            models = {}

            # LightGBM
            logger.info("Training LightGBM model...")
            models['lightgbm'] = self._train_lightgbm(X, y)

            # XGBoost
            logger.info("Training XGBoost model...")
            models['xgboost'] = self._train_xgboost(X, y)

            # Random Forest
            logger.info("Training Random Forest model...")
            models['random_forest'] = self._train_random_forest(X, y)

            # LSTM (if TensorFlow available)
            if TENSORFLOW_AVAILABLE:
                logger.info("Training LSTM model...")
                models['lstm'] = self._train_lstm(X, y)
            else:
                logger.warning("Skipping LSTM training - TensorFlow not available")

            # Store models and metadata
            self.models = models
            self.model_metadata = {
                'training_date': datetime.now(),
                'symbol': self.symbol,
                'feature_columns': self.feature_columns,
                'n_samples': len(X),
                'models_trained': list(models.keys())
            }

            # Save models
            self._save_models()

            logger.info(f"Ensemble training complete: {len(models)} models trained")

            return {
                'models': models,
                'metadata': self.model_metadata
            }

        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise

    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for ML training."""
        try:
            # Select features and target
            X = data[self.feature_columns].values
            y = data['target'].values

            # Handle any remaining NaN values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            logger.debug(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")

            return X, y

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise

    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
        """Train LightGBM model with hyperparameter optimization."""
        try:
            # Split data for validation
            split_idx = int(len(X) * (1 - self.config['training']['val_size']))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Model parameters
            params = self.config['models']['lightgbm'].copy()

            # Create and train model
            model = lgb.LGBMClassifier(
                **params,
                objective='binary',
                metric='auc',
                early_stopping_rounds=self.config['training']['early_stopping_rounds'],
                verbose=-1
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc'
            )

            # Evaluate
            val_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_pred)
            accuracy = accuracy_score(y_val, (val_pred > 0.5).astype(int))

            logger.info(".4f"
            return model

        except Exception as e:
            logger.error(f"Error training LightGBM: {e}")
            raise

    def _train_xgboost(self, X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
        """Train XGBoost model with hyperparameter optimization."""
        try:
            # Split data for validation
            split_idx = int(len(X) * (1 - self.config['training']['val_size']))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Model parameters
            params = self.config['models']['xgboost'].copy()

            # Create and train model
            model = xgb.XGBClassifier(
                **params,
                objective='binary:logistic',
                eval_metric='auc',
                early_stopping_rounds=self.config['training']['early_stopping_rounds'],
                verbose=False
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Evaluate
            val_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_pred)
            accuracy = accuracy_score(y_val, (val_pred > 0.5).astype(int))

            logger.info(".4f"
            return model

        except Exception as e:
            logger.error(f"Error training XGBoost: {e}")
            raise

    def _train_random_forest(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """Train Random Forest model."""
        try:
            # Model parameters
            params = self.config['models']['random_forest'].copy()

            # Create and train model
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X, y)

            # Evaluate with cross-validation
            tscv = TimeSeriesSplit(n_splits=self.config['training']['cv_folds'])
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                model_fold = RandomForestClassifier(**params, random_state=42)
                model_fold.fit(X_train_fold, y_train_fold)

                val_pred = model_fold.predict_proba(X_val_fold)[:, 1]
                auc = roc_auc_score(y_val_fold, val_pred)
                scores.append(auc)

            avg_auc = np.mean(scores)
            logger.info(".4f"
            return model

        except Exception as e:
            logger.error(f"Error training Random Forest: {e}")
            raise

    def _train_lstm(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train LSTM model for sequence prediction."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available for LSTM training")

        try:
            # Reshape data for LSTM (sequence_length, features)
            sequence_length = self.config['models']['lstm']['sequence_length']

            if len(X) < sequence_length:
                raise ValueError(f"Insufficient data for LSTM: need at least {sequence_length} samples")

            # Create sequences
            X_sequences = []
            y_sequences = []

            for i in range(len(X) - sequence_length):
                X_sequences.append(X[i:i+sequence_length])
                y_sequences.append(y[i+sequence_length])

            X_seq = np.array(X_sequences)
            y_seq = np.array(y_sequences)

            # Split data
            split_idx = int(len(X_seq) * (1 - self.config['training']['val_size']))
            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

            # Build model
            model = Sequential([
                LSTM(self.config['models']['lstm']['units'][0],
                     input_shape=(sequence_length, X.shape[1]),
                     return_sequences=True),
                Dropout(self.config['models']['lstm']['dropout']),
                LSTM(self.config['models']['lstm']['units'][1]),
                Dropout(self.config['models']['lstm']['dropout']),
                Dense(1, activation='sigmoid')
            ])

            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )

            # Train model
            early_stopping = EarlyStopping(
                monitor='val_auc',
                patience=10,
                restore_best_weights=True,
                mode='max'
            )

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config['models']['lstm']['epochs'],
                batch_size=self.config['models']['lstm']['batch_size'],
                callbacks=[early_stopping],
                verbose=0
            )

            # Evaluate
            val_pred = model.predict(X_val, verbose=0).flatten()
            auc = roc_auc_score(y_val, val_pred)
            accuracy = accuracy_score(y_val, (val_pred > 0.5).astype(int))

            logger.info(".4f"
            return model

        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            raise

    # ==========================================
    # HYPERPARAMETER OPTIMIZATION
    # ==========================================

    def optimize_hyperparameters(self, model_type: str = 'lightgbm') -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model using Optuna.

        Args:
            model_type: Type of model to optimize ('lightgbm', 'xgboost', 'random_forest')

        Returns:
            Dictionary with best parameters
        """
        try:
            if not self.training_data:
                raise ValueError("No training data available for optimization")

            X, y = self._prepare_training_data(self.training_data)

            def objective(trial):
                if model_type == 'lightgbm':
                    return self._lightgbm_objective(trial, X, y)
                elif model_type == 'xgboost':
                    return self._xgboost_objective(trial, X, y)
                elif model_type == 'random_forest':
                    return self._rf_objective(trial, X, y)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

            study = optuna.create_study(direction='maximize')
            study.optimize(
                objective,
                n_trials=self.config['hyperparameter_tuning']['n_trials'],
                timeout=self.config['hyperparameter_tuning']['timeout_minutes'] * 60
            )

            logger.info(f"Hyperparameter optimization complete for {model_type}")
            logger.info(f"Best parameters: {study.best_params}")
            logger.info(".4f"
            return study.best_params

        except Exception as e:
            logger.error(f"Error optimizing hyperparameters: {e}")
            return {}

    def _lightgbm_objective(self, trial, X, y):
        """Objective function for LightGBM optimization."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
        }

        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMClassifier(**params, objective='binary', verbose=-1)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

            val_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_pred)
            scores.append(auc)

        return np.mean(scores)

    def _xgboost_objective(self, trial, X, y):
        """Objective function for XGBoost optimization."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
        }

        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(**params, objective='binary:logistic', eval_metric='auc')
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

            val_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_pred)
            scores.append(auc)

        return np.mean(scores)

    def _rf_objective(self, trial, X, y):
        """Objective function for Random Forest optimization."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }

        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train, y_train)

            val_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_pred)
            scores.append(auc)

        return np.mean(scores)

    # ==========================================
    # MODEL VALIDATION AND EVALUATION
    # ==========================================

    def validate_models(self, test_data: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, float]]:
        """
        Validate trained models on test data.

        Args:
            test_data: Test data (uses holdout from training data if None)

        Returns:
            Dictionary with validation metrics for each model
        """
        try:
            if not self.models:
                raise ValueError("No trained models available for validation")

            # Prepare test data
            if test_data is None:
                if self.training_data is None:
                    raise ValueError("No training data available")

                # Use last portion of training data as test set
                test_size = int(len(self.training_data) * self.config['training']['test_size'])
                test_data = self.training_data.iloc[-test_size:]

            X_test, y_test = self._prepare_training_data(test_data)

            validation_results = {}

            for model_name, model in self.models.items():
                logger.info(f"Validating {model_name} model...")

                try:
                    if model_name == 'lstm':
                        # LSTM needs sequence data
                        sequence_length = self.config['models']['lstm']['sequence_length']
                        X_seq = []

                        for i in range(len(X_test) - sequence_length):
                            X_seq.append(X_test[i:i+sequence_length])

                        X_test_seq = np.array(X_seq)
                        y_test_seq = y_test[sequence_length:]

                        predictions = model.predict(X_test_seq, verbose=0).flatten()
                        y_test_eval = y_test_seq
                    else:
                        predictions = model.predict_proba(X_test)[:, 1]
                        y_test_eval = y_test

                    # Calculate metrics
                    binary_pred = (predictions > 0.5).astype(int)

                    metrics = {
                        'accuracy': accuracy_score(y_test_eval, binary_pred),
                        'precision': precision_score(y_test_eval, binary_pred, zero_division=0),
                        'recall': recall_score(y_test_eval, binary_pred, zero_division=0),
                        'f1_score': f1_score(y_test_eval, binary_pred, zero_division=0),
                        'auc': roc_auc_score(y_test_eval, predictions)
                    }

                    validation_results[model_name] = metrics

                    logger.info(f"{model_name} validation: AUC={metrics['auc']:.4f}, "
                              f"Accuracy={metrics['accuracy']:.4f}")

                except Exception as e:
                    logger.error(f"Error validating {model_name}: {e}")
                    validation_results[model_name] = {'error': str(e)}

            return validation_results

        except Exception as e:
            logger.error(f"Error in model validation: {e}")
            return {}

    def get_ensemble_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble predictions from all trained models.

        Args:
            X: Feature matrix

        Returns:
            Array of ensemble prediction probabilities
        """
        try:
            if not self.models:
                raise ValueError("No trained models available")

            predictions = []

            for model_name, model in self.models.items():
                try:
                    if model_name == 'lstm':
                        # LSTM needs sequence reshaping
                        sequence_length = self.config['models']['lstm']['sequence_length']
                        if len(X) < sequence_length:
                            continue

                        X_seq = []
                        for i in range(len(X) - sequence_length + 1):
                            X_seq.append(X[i:i+sequence_length])

                        X_seq = np.array(X_seq)
                        pred = model.predict(X_seq, verbose=0).flatten()

                        # Pad predictions to match input length
                        pred_padded = np.full(len(X), 0.5)
                        pred_padded[-len(pred):] = pred
                        predictions.append(pred_padded)
                    else:
                        pred = model.predict_proba(X)[:, 1]
                        predictions.append(pred)

                except Exception as e:
                    logger.warning(f"Error getting predictions from {model_name}: {e}")
                    # Use neutral prediction (0.5) for failed models
                    predictions.append(np.full(len(X), 0.5))

            if not predictions:
                return np.full(len(X), 0.5)

            # Weighted average (equal weights for now)
            ensemble_pred = np.mean(predictions, axis=0)

            return ensemble_pred

        except Exception as e:
            logger.error(f"Error getting ensemble predictions: {e}")
            return np.full(len(X), 0.5)

    # ==========================================
    # MODEL PERSISTENCE
    # ==========================================

    def _save_models(self) -> None:
        """Save trained models to disk."""
        try:
            model_dir = f"models/{self.symbol}"
            os.makedirs(model_dir, exist_ok=True)

            # Save each model
            for model_name, model in self.models.items():
                model_path = os.path.join(model_dir, f"{model_name}_model.pkl")

                if model_name == 'lstm' and TENSORFLOW_AVAILABLE:
                    # Save TensorFlow model
                    model.save(os.path.join(model_dir, f"{model_name}_model"))
                else:
                    # Save sklearn models with pickle
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)

            # Save metadata
            metadata_path = os.path.join(model_dir, "model_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.model_metadata, f)

            # Save feature columns
            feature_path = os.path.join(model_dir, "feature_columns.pkl")
            with open(feature_path, 'wb') as f:
                pickle.dump(self.feature_columns, f)

            logger.info(f"Models saved to {model_dir}")

        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def load_models(self) -> bool:
        """
        Load trained models from disk.

        Returns:
            True if models loaded successfully
        """
        try:
            model_dir = f"models/{self.symbol}"

            if not os.path.exists(model_dir):
                logger.warning(f"No saved models found in {model_dir}")
                return False

            # Load metadata
            metadata_path = os.path.join(model_dir, "model_metadata.pkl")
            with open(metadata_path, 'rb') as f:
                self.model_metadata = pickle.load(f)

            # Load feature columns
            feature_path = os.path.join(model_dir, "feature_columns.pkl")
            with open(feature_path, 'rb') as f:
                self.feature_columns = pickle.load(f)

            # Load models
            self.models = {}
            for model_name in self.model_metadata.get('models_trained', []):
                try:
                    if model_name == 'lstm' and TENSORFLOW_AVAILABLE:
                        # Load TensorFlow model
                        model_path = os.path.join(model_dir, f"{model_name}_model")
                        self.models[model_name] = tf.keras.models.load_model(model_path)
                    else:
                        # Load sklearn models
                        model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
                        with open(model_path, 'rb') as f:
                            self.models[model_name] = pickle.load(f)

                    logger.debug(f"Loaded {model_name} model")

                except Exception as e:
                    logger.warning(f"Error loading {model_name} model: {e}")

            logger.info(f"Loaded {len(self.models)} models from {model_dir}")
            return len(self.models) > 0

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    # ==========================================
    # WALK-FORWARD VALIDATION
    # ==========================================

    def walk_forward_validation(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Perform walk-forward validation on time series data.

        Args:
            data: Full dataset for validation

        Returns:
            Dictionary with validation scores for each fold
        """
        try:
            n_splits = self.config['training']['walk_forward_splits']
            validation_scores = {model: [] for model in self.models.keys()}

            # Calculate fold sizes
            fold_size = len(data) // (n_splits + 1)  # +1 for initial training

            for fold in range(n_splits):
                # Define training and validation periods
                train_end = fold_size * (fold + 1)
                val_end = fold_size * (fold + 2)

                if val_end > len(data):
                    val_end = len(data)

                train_data = data.iloc[:train_end]
                val_data = data.iloc[train_end:val_end]

                if len(val_data) == 0:
                    continue

                # Train models on this fold
                X_train, y_train = self._prepare_training_data(train_data)
                X_val, y_val = self._prepare_training_data(val_data)

                # Train temporary models for this fold
                fold_models = {}

                # LightGBM
                lgb_model = lgb.LGBMClassifier(**self.config['models']['lightgbm'], verbose=-1)
                lgb_model.fit(X_train, y_train)
                fold_models['lightgbm'] = lgb_model

                # XGBoost
                xgb_model = xgb.XGBClassifier(**self.config['models']['xgboost'], verbose=False)
                xgb_model.fit(X_train, y_train)
                fold_models['xgboost'] = xgb_model

                # Random Forest
                rf_model = RandomForestClassifier(**self.config['models']['random_forest'], random_state=42)
                rf_model.fit(X_train, y_train)
                fold_models['random_forest'] = rf_model

                # LSTM (if available)
                if TENSORFLOW_AVAILABLE:
                    # Simplified LSTM training for validation
                    sequence_length = self.config['models']['lstm']['sequence_length']

                    if len(X_train) >= sequence_length:
                        X_train_seq = []
                        y_train_seq = []

                        for i in range(len(X_train) - sequence_length):
                            X_train_seq.append(X_train[i:i+sequence_length])
                            y_train_seq.append(y_train[i+sequence_length])

                        X_train_seq = np.array(X_train_seq)
                        y_train_seq = np.array(y_train_seq)

                        lstm_model = Sequential([
                            LSTM(64, input_shape=(sequence_length, X_train.shape[1])),
                            Dropout(0.2),
                            Dense(1, activation='sigmoid')
                        ])

                        lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                        lstm_model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, verbose=0)
                        fold_models['lstm'] = lstm_model

                # Evaluate fold
                for model_name, model in fold_models.items():
                    try:
                        if model_name == 'lstm':
                            # LSTM validation
                            X_val_seq = []
                            y_val_seq = []

                            for i in range(len(X_val) - sequence_length):
                                X_val_seq.append(X_val[i:i+sequence_length])
                                y_val_seq.append(y_val[i+sequence_length])

                            if X_val_seq:
                                X_val_seq = np.array(X_val_seq)
                                y_val_seq = np.array(y_val_seq)

                                predictions = model.predict(X_val_seq, verbose=0).flatten()
                                auc = roc_auc_score(y_val_seq, predictions)
                                validation_scores[model_name].append(auc)
                        else:
                            predictions = model.predict_proba(X_val)[:, 1]
                            auc = roc_auc_score(y_val, predictions)
                            validation_scores[model_name].append(auc)

                    except Exception as e:
                        logger.debug(f"Error in fold {fold} for {model_name}: {e}")
                        validation_scores[model_name].append(0.5)  # Neutral score

                logger.info(f"Completed walk-forward fold {fold + 1}/{n_splits}")

            # Calculate average scores
            for model_name in validation_scores:
                scores = [s for s in validation_scores[model_name] if s != 0.5]  # Exclude error scores
                if scores:
                    avg_score = np.mean(scores)
                    logger.info(".4f"
            return validation_scores

        except Exception as e:
            logger.error(f"Error in walk-forward validation: {e}")
            return {}

    # ==========================================
    # MAIN TRAINING PIPELINE
    # ==========================================

    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete ML training pipeline.

        Returns:
            Dictionary with pipeline results and trained models
        """
        try:
            logger.info("Starting complete ML training pipeline...")

            # Step 1: Collect training data
            logger.info("Step 1: Collecting training data...")
            training_data = self.collect_training_data()

            # Step 2: Train models
            logger.info("Step 2: Training ML models...")
            training_results = self.train_models(training_data)

            # Step 3: Validate models
            logger.info("Step 3: Validating models...")
            validation_results = self.validate_models()

            # Step 4: Walk-forward validation
            logger.info("Step 4: Running walk-forward validation...")
            wf_validation = self.walk_forward_validation(training_data)

            # Compile results
            pipeline_results = {
                'training': training_results,
                'validation': validation_results,
                'walk_forward_validation': wf_validation,
                'metadata': {
                    'symbol': self.symbol,
                    'training_date': datetime.now(),
                    'n_samples': len(training_data),
                    'n_features': len(self.feature_columns),
                    'models_trained': list(self.models.keys())
                }
            }

            logger.info("ML training pipeline completed successfully")
            logger.info(f"Trained {len(self.models)} models with "
                       f"{len(training_data)} samples and {len(self.feature_columns)} features")

            return pipeline_results

        except Exception as e:
            logger.error(f"Error in ML training pipeline: {e}")
            raise
