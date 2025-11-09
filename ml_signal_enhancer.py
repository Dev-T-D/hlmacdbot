"""
ML Signal Enhancer for Real-Time Trading

This module integrates trained ML models with the trading bot to enhance signal quality
and provide confidence-based position sizing. Includes real-time inference, model
monitoring, and fallback mechanisms.

Key Features:
- Real-time ML inference with low latency
- Confidence-based position sizing with Kelly Criterion
- Model monitoring and performance tracking
- Fallback to MACD-only trading
- Explainability with SHAP values
- Online learning and model updates
"""

import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

# Optional imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from feature_engineering import AdvancedFeatureEngine
from ml.feature_pipeline import FeaturePipeline
from ml.feature_selector import FeatureSelector
from ml_training_pipeline import MLTrainingPipeline
from ml_inference_engine import OptimizedMLInferenceEngine
from optimized_feature_engine import OptimizedFeatureEngine

logger = logging.getLogger(__name__)


class MLSignalEnhancer:
    """
    ML-powered signal enhancement for trading decisions.

    This class integrates trained ML models with trading signals to:
    - Predict short-term price direction probability
    - Enhance MACD signals with ML confidence
    - Provide Kelly Criterion-based position sizing
    - Monitor model performance and trigger retraining
    - Provide explainability for trading decisions
    """

    def __init__(self, symbol: str = 'BTCUSDT', config: Optional[Dict] = None, min_confidence: float = 0.65):
        """
        Initialize the ML signal enhancer.

        Args:
            symbol: Trading symbol
            config: Configuration dictionary
            min_confidence: Minimum prediction confidence to generate signal (0.5-0.8)
        """
        self.symbol = symbol
        self.config = config or self._get_default_config()
        self.min_confidence = min_confidence

        # Initialize components
        # Initialize advanced feature engineering pipeline
        self.feature_engine = AdvancedFeatureEngine(self.config.get('feature_engineering', {}))
        self.feature_selector = FeatureSelector(method='hybrid')
        self.feature_pipeline = FeaturePipeline(
            feature_engine=self.feature_engine,
            feature_selector=self.feature_selector,
            scaler='robust'
        )

        # Try to load fitted pipeline if available
        try:
            pipeline_path = f"models/{symbol}/feature_pipeline.pkl"
            if os.path.exists(pipeline_path):
                self.feature_pipeline.load(pipeline_path)
                logger.info(f"Loaded fitted feature pipeline from {pipeline_path}")
            else:
                logger.warning(f"No fitted pipeline found at {pipeline_path}")
        except Exception as e:
            logger.warning(f"Could not load fitted pipeline: {e}")
        self.training_pipeline = MLTrainingPipeline(symbol, self.config.get('training', {}))

        # Initialize optimized engines for production inference
        self.optimized_feature_engine = OptimizedFeatureEngine()
        self.optimized_inference_engine = None  # Will be initialized when loading models

        # Track prediction accuracy by confidence level
        self.confidence_buckets = {
            '0.50-0.60': [],
            '0.60-0.70': [],
            '0.70-0.80': [],
            '0.80-0.90': [],
            '0.90-1.00': []
        }

        # Model state
        self.models = {}
        self.model_metadata = {}
        self.feature_columns = []
        self.models_loaded = False

        # Performance tracking
        self.prediction_history = []
        self.max_history = 10000

        # Model monitoring
        self.monitoring_stats = {
            'predictions_made': 0,
            'correct_predictions': 0,
            'total_pnl': 0.0,
            'last_retraining': None,
            'model_age_days': 0
        }

        # Confidence thresholds
        self.confidence_thresholds = self.config['inference']['confidence_thresholds']

        logger.info(f"MLSignalEnhancer initialized for {symbol}")

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'inference': {
                'confidence_thresholds': {
                    'high': 0.65,     # 65%+ confidence = 100% position size
                    'medium': 0.60,   # 60-65% = 70% position size
                    'low': 0.55       # 55-60% = 40% position size, below = no trade
                },
                'kelly_fraction': 0.5,  # Use half Kelly for safety
                'max_position_size': 1.0,  # Maximum position size multiplier
                'fallback_enabled': True
            },
            'monitoring': {
                'retraining_threshold': 0.52,  # Retrain if accuracy drops below 52%
                'max_model_age_days': 30,
                'performance_window': 1000,  # Last 1000 predictions for monitoring
                'alert_threshold': 0.48  # Alert if accuracy below 48%
            },
            'explainability': {
                'shap_enabled': True,
                'feature_importance_tracking': True,
                'max_explanations_stored': 100
            }
        }

    # ==========================================
    # MODEL LOADING AND MANAGEMENT
    # ==========================================

    def load_models(self) -> bool:
        """
        Load trained ML models for inference.

        Returns:
            True if models loaded successfully
        """
        try:
            logger.info(f"Attempting to load ML models for {self.symbol}")
            success = self.training_pipeline.load_models()
            logger.info(f"Model loading success: {success}")

            if success:
                self.models = self.training_pipeline.models
                self.model_metadata = self.training_pipeline.model_metadata
                self.feature_columns = self.training_pipeline.feature_columns
                self.models_loaded = True

                # Initialize optimized inference engine for production use (optional)
                try:
                    # Check if ONNX models exist
                    onnx_paths = [
                        f'models/onnx/lightgbm.onnx',
                        f'models/onnx/xgboost.onnx',
                        f'models/onnx/random_forest.onnx',
                        f'models/onnx/lstm.onnx'
                    ]
                    if all(os.path.exists(path) for path in onnx_paths):
                        model_configs = {name.split('/')[-1].replace('.onnx', ''): path for path in onnx_paths}
                        self.optimized_inference_engine = OptimizedMLInferenceEngine(model_configs)
                        logger.info("✅ Optimized ONNX inference engine initialized")
                    else:
                        logger.info("ONNX models not found - using standard sklearn inference")
                        self.optimized_inference_engine = None
                except Exception as e:
                    logger.warning(f"⚠️  Failed to initialize optimized inference engine: {e}")
                    logger.warning("Using standard sklearn inference")
                    self.optimized_inference_engine = None

                # Update monitoring stats
                if self.model_metadata.get('training_date'):
                    training_date = self.model_metadata['training_date']
                    if isinstance(training_date, str):
                        training_date = datetime.fromisoformat(training_date.replace('Z', '+00:00'))
                    self.monitoring_stats['model_age_days'] = (datetime.now() - training_date).days
                    self.monitoring_stats['last_retraining'] = training_date

                logger.info(f"Loaded ML models: {list(self.models.keys())}")
                logger.info(f"Models trained on {self.model_metadata.get('n_samples', 0)} samples "
                          f"with {len(self.feature_columns)} features")

                return True
            else:
                logger.warning("Failed to load ML models - will use fallback mode")
                return False

        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            return False

    def is_model_available(self) -> bool:
        """
        Check if ML models are available and recent.

        Returns:
            True if models are loaded and not too old
        """
        if not self.models_loaded:
            return False

        # Check model age
        max_age = self.config['monitoring']['max_model_age_days']
        if self.monitoring_stats['model_age_days'] > max_age:
            logger.warning(f"ML models are {self.monitoring_stats['model_age_days']} days old "
                         f"(max allowed: {max_age})")
            return False

        return True

    # ==========================================
    # REAL-TIME INFERENCE
    # ==========================================

    def predict_direction(self, market_data: pd.DataFrame,
                         orderbook_data: Optional[Dict] = None,
                         trade_flow_data: Optional[Dict] = None,
                         funding_rates: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Predict short-term price direction using ML models.

        Args:
            market_data: Recent OHLCV data
            orderbook_data: Current order book state
            trade_flow_data: Recent trade flow metrics
            funding_rates: Current funding rate data

        Returns:
            Dictionary with prediction results
        """
        try:
            start_time = time.time()

            if not self.is_model_available():
                return self._fallback_prediction()

            # Use optimized engines if available (production mode)
            if self.optimized_inference_engine is not None:
                return self._predict_direction_optimized(
                    market_data, orderbook_data, trade_flow_data, funding_rates, start_time
                )

            # Fallback to original method for compatibility
            return self._predict_direction_legacy(
                market_data, orderbook_data, trade_flow_data, funding_rates, start_time
            )

        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return self._fallback_prediction()

    def _predict_direction_optimized(self, market_data: pd.DataFrame,
                                   orderbook_data: Optional[Dict] = None,
                                   trade_flow_data: Optional[Dict] = None,
                                   funding_rates: Optional[List[Dict]] = None,
                                   start_time: float = None) -> Dict[str, Any]:
        """
        Optimized prediction using ONNX models and fast feature computation.
        """
        if start_time is None:
            start_time = time.time()

        # Fast feature computation (<20ms)
        features = self.optimized_feature_engine.compute_features_fast(market_data, self.symbol)

        # Fast ensemble inference (<30ms)
        inference_result = self.optimized_inference_engine.predict_ensemble(features)

        # Calculate confidence metrics
        confidence_metrics = self._calculate_confidence_metrics(inference_result['probability'])

        prediction_result = {
            'direction_probability': inference_result['probability'],
            'predicted_direction': 'UP' if inference_result['probability'] > 0.5 else 'DOWN',
            'confidence_level': confidence_metrics['level'],
            'position_size_multiplier': confidence_metrics['position_multiplier'],
            'kelly_fraction': confidence_metrics['kelly_fraction'],
            'feature_importance': {},  # Not available with ONNX models
            'inference_time_ms': inference_result['latency_ms'],
            'model_available': True,
            'optimization_level': 'onnx',
            'individual_predictions': inference_result['individual_predictions'],
            'ensemble_confidence': inference_result['confidence'],
            'timestamp': datetime.now()
        }

        # Store prediction for monitoring
        self._store_prediction(prediction_result)

        logger.debug(f"Prediction completed - confidence: {inference_result['confidence']:.4f}")
        return prediction_result

    def _predict_direction_legacy(self, market_data: pd.DataFrame,
                                orderbook_data: Optional[Dict] = None,
                                trade_flow_data: Optional[Dict] = None,
                                funding_rates: Optional[List[Dict]] = None,
                                start_time: float = None) -> Dict[str, Any]:
        """
        Legacy prediction method using original sklearn models.
        """
        if start_time is None:
            start_time = time.time()

        # Engineer features
        # Use advanced feature engineering
        features_df = self.feature_engine.create_all_features(market_data)

        # Prepare features for ML models - use all available features if feature_columns not set
        if not self.feature_columns:
            self.feature_columns = [col for col in features_df.columns
                                  if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
            logger.warning(f"No feature columns loaded, using all {len(self.feature_columns)} features")

        feature_values = features_df[self.feature_columns].iloc[-1:].values  # Last row
        feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)

        # Get ensemble prediction
        ensemble_prob = self.training_pipeline.get_ensemble_predictions(feature_values)[0]

        # Calculate confidence metrics
        confidence_metrics = self._calculate_confidence_metrics(ensemble_prob)

        # Get feature importance if available
        feature_importance = self._get_feature_importance(features_df.iloc[-1:], feature_values)

        # Calculate inference time
        inference_time = time.time() - start_time

        prediction_result = {
            'direction_probability': ensemble_prob,
            'predicted_direction': 'UP' if ensemble_prob > 0.5 else 'DOWN',
            'confidence_level': confidence_metrics['level'],
            'position_size_multiplier': confidence_metrics['position_multiplier'],
            'kelly_fraction': confidence_metrics['kelly_fraction'],
            'feature_importance': feature_importance,
            'inference_time_ms': inference_time * 1000,
            'model_available': True,
            'optimization_level': 'legacy',
            'timestamp': datetime.now()
        }

        # Store prediction for monitoring
        self._store_prediction(prediction_result)

        logger.debug(f"Prediction completed - confidence: {inference_result['confidence']:.4f}")
        return prediction_result

    def _fallback_prediction(self) -> Dict[str, Any]:
        """Return fallback prediction when ML models are unavailable."""
        return {
            'direction_probability': 0.5,
            'predicted_direction': 'NEUTRAL',
            'confidence_level': 'none',
            'position_size_multiplier': 0.0,
            'kelly_fraction': 0.0,
            'feature_importance': {},
            'inference_time_ms': 0.0,
            'model_available': False,
            'fallback_reason': 'ML models not available',
            'timestamp': datetime.now()
        }

    def _calculate_confidence_metrics(self, probability: float) -> Dict[str, Any]:
        """
        Calculate confidence level and position sizing based on prediction probability.

        Args:
            probability: Prediction probability (0-1)

        Returns:
            Dictionary with confidence metrics
        """
        thresholds = self.confidence_thresholds

        if probability >= thresholds['high']:
            confidence_level = 'high'
            position_multiplier = 1.0
            kelly_fraction = self._calculate_kelly_fraction(probability)
        elif probability >= thresholds['medium']:
            confidence_level = 'medium'
            position_multiplier = 0.7
            kelly_fraction = self._calculate_kelly_fraction(probability) * 0.7
        elif probability >= thresholds['low']:
            confidence_level = 'low'
            position_multiplier = 0.4
            kelly_fraction = self._calculate_kelly_fraction(probability) * 0.4
        else:
            confidence_level = 'insufficient'
            position_multiplier = 0.0
            kelly_fraction = 0.0

        # Apply maximum position size limit
        position_multiplier = min(position_multiplier, self.config['inference']['max_position_size'])

        return {
            'level': confidence_level,
            'position_multiplier': position_multiplier,
            'kelly_fraction': kelly_fraction
        }

    def _calculate_kelly_fraction(self, win_probability: float, win_loss_ratio: float = 2.0) -> float:
        """
        Calculate Kelly Criterion position size.

        Args:
            win_probability: Probability of winning
            win_loss_ratio: Average win/loss ratio

        Returns:
            Kelly fraction (0-1)
        """
        try:
            # Kelly Formula: f = (p * (b + 1) - 1) / b
            # where p = win probability, b = win/loss ratio
            b = win_loss_ratio
            kelly_f = (win_probability * (b + 1) - 1) / b

            # Apply safety fraction
            kelly_f *= self.config['inference']['kelly_fraction']

            # Ensure non-negative
            kelly_f = max(0, kelly_f)

            return min(kelly_f, 1.0)  # Cap at 100%

        except Exception as e:
            logger.debug(f"Error calculating Kelly fraction: {e}")
            return 0.0

    # ==========================================
    # SIGNAL INTEGRATION
    # ==========================================

    def should_trade(self, macd_signal: str, ml_prediction: Dict[str, Any],
                    current_price: float) -> Tuple[bool, float, str]:
        """
        Determine if should trade based on ML confidence and MACD agreement.

        Args:
            macd_signal: MACD signal ('LONG', 'SHORT', or None)
            ml_prediction: ML prediction results
            current_price: Current market price

        Returns:
            Tuple of (should_trade, adjusted_confidence, reason)
        """
        try:
            if not macd_signal:
                return False, 0.0, 'No MACD signal'

            probability = ml_prediction.get('direction_probability', 0.5)
            confidence = ml_prediction.get('confidence_level', 'none')

            # Convert confidence level to numeric
            confidence_numeric = 0.5
            if isinstance(confidence, str):
                if confidence == 'high':
                    confidence_numeric = 0.8
                elif confidence == 'medium':
                    confidence_numeric = 0.65
                elif confidence == 'low':
                    confidence_numeric = 0.55
            else:
                confidence_numeric = float(confidence) if confidence != 'none' else 0.5

            # FILTER 1: ML Confidence Threshold
            if confidence_numeric < self.min_confidence:
                return False, confidence_numeric, f"ML confidence too low ({confidence_numeric:.2f} < {self.min_confidence})"

            # FILTER 2: ML-MACD Agreement
            ml_direction = 'LONG' if probability > 0.5 else 'SHORT'

            if ml_direction != macd_signal:
                return False, confidence_numeric, f"ML-MACD disagreement (ML: {ml_direction}, MACD: {macd_signal})"

            # FILTER 3: Strong ML Signal (probability far from 0.5)
            if macd_signal == 'LONG':
                if probability < 0.60:  # Require at least 60% probability for LONG
                    return False, confidence_numeric, f"ML probability too weak for LONG ({probability:.2f})"
            else:  # SHORT
                if probability > 0.40:  # Require at most 40% probability for SHORT
                    return False, confidence_numeric, f"ML probability too weak for SHORT ({probability:.2f})"

            # FILTER 4: Model Consensus (all models should agree direction)
            individual_preds = ml_prediction.get('individual_predictions', {})
            if len(individual_preds) > 0:
                long_votes = sum(1 for p in individual_preds.values() if p > 0.5)
                consensus_pct = long_votes / len(individual_preds)

                if ml_direction == 'LONG' and consensus_pct < 0.75:
                    return False, confidence_numeric, f"Insufficient model consensus for LONG ({consensus_pct:.2f})"
                elif ml_direction == 'SHORT' and consensus_pct > 0.25:
                    return False, confidence_numeric, f"Insufficient model consensus for SHORT ({consensus_pct:.2f})"

            # ALL FILTERS PASSED - Calculate adjusted confidence
            adjusted_confidence = confidence_numeric * abs(probability - 0.5) * 2  # Scale by distance from 0.5

            # Enhanced signal info
            enhanced_signal = {
                'signal_type': 'ml_enhanced_filtered',
                'macd_signal': macd_signal,
                'ml_probability': probability,
                'ml_direction': ml_direction,
                'confidence_level': confidence_numeric,
                'adjusted_confidence': adjusted_confidence,
                'position_size_multiplier': ml_prediction.get('position_size_multiplier', 1.0),
                'kelly_fraction': ml_prediction.get('kelly_fraction', 0.0),
                'feature_importance': ml_prediction.get('feature_importance', {}),
                'model_consensus': consensus_pct if len(individual_preds) > 0 else 1.0
            }

            logger.info(f"🎯 ALL FILTERS PASSED - Opening {macd_signal} position with confidence {adjusted_confidence:.2f}")

            return True, adjusted_confidence, "All filters passed"

        except Exception as e:
            logger.error(f"Error in should_trade decision: {e}")
            return False, 0.0, f'Error in signal integration: {e}'

    def record_outcome(self, prediction_confidence: float, actual_outcome: int) -> None:
        """Track prediction accuracy by confidence level"""
        # Determine confidence bucket
        if 0.50 <= prediction_confidence < 0.60:
            bucket = '0.50-0.60'
        elif 0.60 <= prediction_confidence < 0.70:
            bucket = '0.60-0.70'
        elif 0.70 <= prediction_confidence < 0.80:
            bucket = '0.70-0.80'
        elif 0.80 <= prediction_confidence < 0.90:
            bucket = '0.80-0.90'
        else:
            bucket = '0.90-1.00'

        self.confidence_buckets[bucket].append(actual_outcome)

    def get_confidence_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get win rate by confidence bucket"""
        stats = {}
        for bucket, outcomes in self.confidence_buckets.items():
            if len(outcomes) > 0:
                win_rate = sum(outcomes) / len(outcomes)
                stats[bucket] = {
                    'win_rate': win_rate,
                    'count': len(outcomes)
                }
        return stats

    # ==========================================
    # MODEL MONITORING AND PERFORMANCE
    # ==========================================

    def _store_prediction(self, prediction: Dict[str, Any]) -> None:
        """Store prediction for performance monitoring."""
        try:
            prediction_record = {
                'timestamp': prediction['timestamp'],
                'probability': prediction['direction_probability'],
                'predicted_direction': prediction['predicted_direction'],
                'confidence_level': prediction['confidence_level'],
                'actual_outcome': None,  # To be filled later
                'pnl_realized': None,    # To be filled later
                'features': prediction.get('feature_importance', {})
            }

            self.prediction_history.append(prediction_record)

            # Maintain history size
            if len(self.prediction_history) > self.max_history:
                self.prediction_history = self.prediction_history[-self.max_history:]

            self.monitoring_stats['predictions_made'] += 1

        except Exception as e:
            logger.debug(f"Error storing prediction: {e}")

    def update_prediction_outcome(self, prediction_timestamp: datetime,
                                actual_direction: str, pnl: float) -> None:
        """
        Update prediction record with actual outcome for performance tracking.

        Args:
            prediction_timestamp: Timestamp of the prediction
            actual_direction: Actual price direction ('UP' or 'DOWN')
            pnl: Realized P&L from the trade
        """
        try:
            # Find the closest prediction by timestamp
            target_timestamp = prediction_timestamp
            closest_record = None
            min_time_diff = timedelta(minutes=5)  # 5-minute tolerance

            for record in reversed(self.prediction_history):
                time_diff = abs(record['timestamp'] - target_timestamp)
                if time_diff < min_time_diff:
                    closest_record = record
                    min_time_diff = time_diff

            if closest_record:
                # Update outcome
                predicted_direction = closest_record['predicted_direction']
                actual_up = actual_direction == 'UP'

                closest_record['actual_outcome'] = actual_direction
                closest_record['pnl_realized'] = pnl

                # Check if prediction was correct
                predicted_up = predicted_direction == 'UP'
                was_correct = (predicted_up == actual_up)

                if was_correct:
                    self.monitoring_stats['correct_predictions'] += 1

                self.monitoring_stats['total_pnl'] += pnl

                logger.debug(f"Updated prediction outcome: {predicted_direction} -> {actual_direction}, "
                           f"Correct: {was_correct}, PnL: ${pnl:.2f}")

        except Exception as e:
            logger.debug(f"Error updating prediction outcome: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current model performance metrics."""
        try:
            predictions_made = self.monitoring_stats['predictions_made']

            if predictions_made == 0:
                return {'status': 'no_predictions_yet'}

            correct_predictions = self.monitoring_stats['correct_predictions']
            accuracy = correct_predictions / predictions_made

            # Recent performance (last N predictions)
            recent_window = min(100, len(self.prediction_history))
            recent_predictions = self.prediction_history[-recent_window:]
            recent_correct = sum(1 for p in recent_predictions if p.get('actual_outcome'))

            if recent_predictions:
                recent_accuracy = recent_correct / len(recent_predictions)
            else:
                recent_accuracy = 0.0

            return {
                'total_predictions': predictions_made,
                'correct_predictions': correct_predictions,
                'overall_accuracy': accuracy,
                'recent_accuracy': recent_accuracy,
                'total_pnl': self.monitoring_stats['total_pnl'],
                'model_age_days': self.monitoring_stats['model_age_days'],
                'needs_retraining': self._should_retrain(),
                'alert_triggered': accuracy < self.config['monitoring']['alert_threshold']
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'error': str(e)}

    def _should_retrain(self) -> bool:
        """Determine if model retraining is needed."""
        try:
            metrics = self.get_performance_metrics()

            if metrics.get('overall_accuracy', 0) < self.config['monitoring']['retraining_threshold']:
                return True

            if self.monitoring_stats['model_age_days'] > self.config['monitoring']['max_model_age_days']:
                return True

            return False

        except Exception as e:
            logger.debug(f"Error checking retraining need: {e}")
            return False

    # ==========================================
    # EXPLAINABILITY
    # ==========================================

    def _get_feature_importance(self, features_df: pd.DataFrame, feature_values: np.ndarray) -> Dict[str, float]:
        """Get feature importance for the current prediction."""
        try:
            if not SHAP_AVAILABLE or not self.models:
                return {}

            # Use the best performing model for explanations (LightGBM if available)
            explainer_model = self.models.get('lightgbm') or list(self.models.values())[0]

            if hasattr(explainer_model, 'feature_importances_'):
                # Tree-based model feature importance
                importance_scores = explainer_model.feature_importances_
                feature_importance = dict(zip(self.feature_columns, importance_scores))

                # Get top 5 most important features
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                return dict(top_features)
            else:
                return {}

        except Exception as e:
            logger.debug(f"Error calculating feature importance: {e}")
            return {}

    def get_prediction_explanation(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed explanation for a prediction.

        Args:
            prediction: Prediction result dictionary

        Returns:
            Dictionary with explanation details
        """
        try:
            explanation = {
                'prediction_summary': {
                    'direction': prediction['predicted_direction'],
                    'probability': prediction['direction_probability'],
                    'confidence': prediction['confidence_level']
                },
                'key_factors': prediction.get('feature_importance', {}),
                'model_health': self.get_performance_metrics(),
                'recommendations': []
            }

            # Add recommendations based on prediction
            probability = prediction['direction_probability']

            if probability > 0.7:
                explanation['recommendations'].append("Strong bullish signal - consider full position")
            elif probability > 0.6:
                explanation['recommendations'].append("Moderate bullish signal - use partial position")
            elif probability < 0.4:
                explanation['recommendations'].append("Moderate bearish signal - use partial position")
            elif probability < 0.3:
                explanation['recommendations'].append("Strong bearish signal - consider full position")
            else:
                explanation['recommendations'].append("Neutral signal - consider reducing position size")

            # Add model health warnings
            health = explanation['model_health']
            if health.get('alert_triggered', False):
                explanation['recommendations'].append("⚠️ Model performance below threshold - consider manual oversight")

            if health.get('needs_retraining', False):
                explanation['recommendations'].append("🔄 Model retraining recommended")

            return explanation

        except Exception as e:
            logger.error(f"Error generating prediction explanation: {e}")
            return {'error': str(e)}

    # ==========================================
    # MODEL MAINTENANCE
    # ==========================================

    def retrain_models(self) -> bool:
        """
        Retrain ML models with new data.

        Returns:
            True if retraining successful
        """
        try:
            logger.info("Starting ML model retraining...")

            # Run full training pipeline
            results = self.training_pipeline.run_full_pipeline()

            # Update local models
            success = self.load_models()

            if success:
                self.monitoring_stats['last_retraining'] = datetime.now()
                self.monitoring_stats['model_age_days'] = 0

                logger.info("ML model retraining completed successfully")
                return True
            else:
                logger.error("ML model retraining failed")
                return False

        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
            return False

    # ==========================================
    # UTILITY METHODS
    # ==========================================

    def reset_monitoring(self) -> None:
        """Reset performance monitoring statistics."""
        self.prediction_history.clear()
        self.monitoring_stats = {
            'predictions_made': 0,
            'correct_predictions': 0,
            'total_pnl': 0.0,
            'last_retraining': self.monitoring_stats.get('last_retraining'),
            'model_age_days': self.monitoring_stats.get('model_age_days', 0)
        }
        logger.info("Performance monitoring reset")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'models_loaded': list(self.models.keys()) if self.models else [],
            'model_metadata': self.model_metadata,
            'feature_columns': self.feature_columns,
            'performance_metrics': self.get_performance_metrics(),
            'config': self.config
        }
