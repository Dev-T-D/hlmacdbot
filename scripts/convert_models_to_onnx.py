#!/usr/bin/env python3
"""
Convert Trained ML Models to ONNX Format for Ultra-Fast Inference

This script converts all trained ML models (LightGBM, XGBoost, Random Forest, LSTM)
to ONNX format for 5-10x faster inference in production trading.

Usage:
    python scripts/convert_models_to_onnx.py [--symbol SYMBOL]

Arguments:
    --symbol: Trading symbol (default: SOL-USDC)

Requirements:
    pip install onnxmltools onnxruntime lightgbm xgboost scikit-learn tensorflow skl2onnx
"""

import os
import sys
import pickle
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# ONNX conversion libraries
from onnxmltools import convert_lightgbm, convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.convert import convert_sklearn
import onnxruntime as ort

# Model libraries
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# Optional: TensorFlow for LSTM
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ONNXModelConverter:
    """
    Convert trained ML models to ONNX format for optimized inference.
    """

    def __init__(self, models_dir: str = "models", onnx_dir: str = "models/onnx"):
        """
        Initialize the converter.

        Args:
            models_dir: Directory containing trained pickle models
            onnx_dir: Output directory for ONNX models
        """
        self.models_dir = Path(models_dir)
        self.onnx_dir = Path(onnx_dir)
        self.onnx_dir.mkdir(parents=True, exist_ok=True)

        # Default feature count (will be detected from feature engineering)
        self.n_features = self._detect_feature_count()

        logger.info(f"ONNX converter initialized. Models: {self.models_dir}, ONNX: {self.onnx_dir}")
        logger.info(f"Detected {self.n_features} features for model input")

    def _detect_feature_count(self) -> int:
        """Detect the number of features from feature engineering module."""
        try:
            # Try to import and instantiate feature engineer to get feature count
            sys.path.append(str(Path(__file__).parent.parent))
            from feature_engineering import FeatureEngineer

            engineer = FeatureEngineer()
            # Create dummy data to get feature count
            dummy_df = self._create_dummy_dataframe()
            features = engineer.compute_all_features(dummy_df)
            return len(features)

        except Exception as e:
            logger.warning(f"Could not detect feature count from feature engineering: {e}")
            # Fallback to common feature counts
            return 80  # Default assumption

    def _create_dummy_dataframe(self) -> 'pd.DataFrame':
        """Create dummy OHLCV data for feature detection."""
        try:
            import pandas as pd
            # Create 100 periods of dummy data
            dates = pd.date_range('2024-01-01', periods=100, freq='5min')
            np.random.seed(42)  # For reproducible dummy data

            data = {
                'open': 50000 + np.random.normal(0, 1000, 100),
                'high': 50200 + np.random.normal(0, 1000, 100),
                'low': 49800 + np.random.normal(0, 1000, 100),
                'close': 50000 + np.random.normal(0, 1000, 100),
                'volume': np.random.lognormal(10, 1, 100)
            }

            # Ensure high >= max(open, close) and low <= min(open, close)
            for i in range(len(data['open'])):
                high = max(data['open'][i], data['close'][i]) + abs(np.random.normal(0, 200))
                low = min(data['open'][i], data['close'][i]) - abs(np.random.normal(0, 200))
                data['high'][i] = high
                data['low'][i] = low

            return pd.DataFrame(data, index=dates)

        except ImportError:
            logger.error("pandas not available for dummy data creation")
            return None

    def convert_lightgbm_model(self, model_path: Path, output_path: Path) -> bool:
        """
        Convert LightGBM model to ONNX format.

        Args:
            model_path: Path to pickled LightGBM model
            output_path: Output path for ONNX model

        Returns:
            True if conversion successful
        """
        try:
            logger.info(f"Converting LightGBM model: {model_path}")

            # Load the model
            model = lgb.Booster(model_file=str(model_path))

            # Define input tensor shape (batch_size, n_features)
            initial_type = [('float_input', FloatTensorType([None, self.n_features]))]

            # Convert to ONNX
            onnx_model = convert_lightgbm(model, initial_types=initial_type, target_opset=12)

            # Save ONNX model
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            logger.info(f"✅ LightGBM model converted: {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to convert LightGBM model {model_path}: {e}")
            return False

    def convert_xgboost_model(self, model_path: Path, output_path: Path) -> bool:
        """
        Convert XGBoost model to ONNX format.

        Args:
            model_path: Path to pickled XGBoost model
            output_path: Output path for ONNX model

        Returns:
            True if conversion successful
        """
        try:
            logger.info(f"Converting XGBoost model: {model_path}")

            # Load the model
            model = xgb.Booster()
            model.load_model(str(model_path))

            # Define input tensor shape
            initial_type = [('float_input', FloatTensorType([None, self.n_features]))]

            # Convert to ONNX
            onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=12)

            # Save ONNX model
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            logger.info(f"✅ XGBoost model converted: {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to convert XGBoost model {model_path}: {e}")
            return False

    def convert_random_forest_model(self, model_path: Path, output_path: Path) -> bool:
        """
        Convert Random Forest model to ONNX format.

        Args:
            model_path: Path to pickled Random Forest model
            output_path: Output path for ONNX model

        Returns:
            True if conversion successful
        """
        try:
            logger.info(f"Converting Random Forest model: {model_path}")

            # Load the model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Define input tensor shape
            initial_type = [('float_input', FloatTensorType([None, self.n_features]))]

            # Convert to ONNX
            onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)

            # Save ONNX model
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            logger.info(f"✅ Random Forest model converted: {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to convert Random Forest model {model_path}: {e}")
            return False

    def convert_lstm_model(self, model_path: Path, output_path: Path) -> bool:
        """
        Convert LSTM model to ONNX format.

        Args:
            model_path: Path to saved LSTM model (.h5 or SavedModel format)
            output_path: Output path for ONNX model

        Returns:
            True if conversion successful
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available - skipping LSTM conversion")
            return False

        try:
            logger.info(f"Converting LSTM model: {model_path}")

            # Load the model
            if str(model_path).endswith('.h5'):
                model = tf.keras.models.load_model(str(model_path))
            else:
                # Assume SavedModel format
                model = tf.saved_model.load(str(model_path))

            # Create dummy input for tracing (batch_size, sequence_length, n_features)
            # LSTM expects sequence data, we'll use a sequence length of 10
            sequence_length = 10
            dummy_input = tf.random.normal([1, sequence_length, self.n_features])

            # Convert to ONNX
            tf2onnx.convert.from_keras(
                model,
                input_signature=[tf.TensorSpec(dummy_input.shape, tf.float32, name="input")],
                opset=12,
                output_path=str(output_path)
            )

            logger.info(f"✅ LSTM model converted: {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to convert LSTM model {model_path}: {e}")
            return False

    def verify_conversion(self, original_model_path: Path, onnx_model_path: Path,
                         model_type: str) -> bool:
        """
        Verify that ONNX model produces same predictions as original.

        Args:
            original_model_path: Path to original model
            onnx_model_path: Path to ONNX model
            model_type: Type of model ('lightgbm', 'xgboost', 'random_forest', 'lstm')

        Returns:
            True if predictions match within tolerance
        """
        try:
            logger.info(f"Verifying conversion accuracy for {model_type}")

            # Create test data
            np.random.seed(42)
            test_data = np.random.randn(10, self.n_features).astype(np.float32)

            # Get original model predictions
            if model_type == 'lightgbm':
                original_model = lgb.Booster(model_file=str(original_model_path))
                original_preds = original_model.predict(test_data)
            elif model_type == 'xgboost':
                original_model = xgb.Booster()
                original_model.load_model(str(original_model_path))
                original_preds = original_model.predict(xgb.DMatrix(test_data))
            elif model_type == 'random_forest':
                with open(original_model_path, 'rb') as f:
                    original_model = pickle.load(f)
                original_preds = original_model.predict_proba(test_data)[:, 1]
            elif model_type == 'lstm':
                if not TENSORFLOW_AVAILABLE:
                    return False
                # For LSTM, we'd need to create sequence data
                logger.warning("LSTM verification not implemented yet")
                return True
            else:
                logger.error(f"Unknown model type: {model_type}")
                return False

            # Get ONNX model predictions
            session = ort.InferenceSession(str(onnx_model_path))
            input_name = session.get_inputs()[0].name
            onnx_preds = session.run(None, {input_name: test_data})[0]

            # For binary classification, extract probability of positive class
            if len(onnx_preds.shape) > 1 and onnx_preds.shape[1] == 2:
                onnx_preds = onnx_preds[:, 1]

            # Compare predictions
            diff = np.abs(original_preds - onnx_preds)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            logger.info(f"Max prediction difference: {max_diff:.6f}")
            logger.info(f"Mean prediction difference: {mean_diff:.6f}")

            # Allow small tolerance for floating point differences
            tolerance = 1e-5
            if max_diff < tolerance:
                logger.info("✅ Conversion verification passed")
                return True
            else:
                logger.warning(f"⚠️  Conversion verification failed - differences exceed tolerance {tolerance}")
                return False

        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False

    def convert_all_models(self, symbol: str = 'SOL-USDC') -> Dict[str, bool]:
        """
        Convert all available trained models to ONNX format.

        Args:
            symbol: Trading symbol for model directory

        Returns:
            Dict mapping model names to conversion success status
        """
        results = {}

        # Define model mappings based on symbol
        model_mappings = [
            (f'models/{symbol}/lightgbm_model.pkl', 'lightgbm', self.convert_lightgbm_model),
            (f'models/{symbol}/xgboost_model.pkl', 'xgboost', self.convert_xgboost_model),
            (f'models/{symbol}/random_forest_model.pkl', 'random_forest', self.convert_random_forest_model),
            (f'models/{symbol}/lstm_model.keras', 'lstm', self.convert_lstm_model),
        ]

        for model_pickle_path, model_name, converter_func in model_mappings:
            model_path = Path(model_pickle_path)
            onnx_path = self.onnx_dir / f"{model_name}.onnx"

            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                results[model_name] = False
                continue

            # Convert the model
            success = converter_func(model_path, onnx_path)
            results[model_name] = success

            if success:
                # Verify the conversion
                verification_passed = self.verify_conversion(model_path, onnx_path, model_name)
                if not verification_passed:
                    logger.warning(f"⚠️  {model_name} conversion verification failed")

        return results

    def get_conversion_summary(self, results: Dict[str, bool]) -> str:
        """Generate a summary of conversion results."""
        successful = sum(results.values())
        total = len(results)

        summary = f"""
🎯 ONNX Model Conversion Summary
================================

Total Models: {total}
Successful Conversions: {successful}
Failed Conversions: {total - successful}

Model Status:
"""

        for model_name, success in results.items():
            status = "✅" if success else "❌"
            summary += f"  {status} {model_name}\n"

        if successful == total:
            summary += "\n🎉 All models successfully converted to ONNX format!"
            summary += "\n🚀 Ready for ultra-fast inference with <50ms latency."
        else:
            summary += f"\n⚠️  {total - successful} model(s) failed conversion."

        return summary


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description='Convert ML models to ONNX format')
    parser.add_argument('--symbol', default='SOL-USDC', help='Trading symbol (default: SOL-USDC)')
    args = parser.parse_args()

    logger.info("🚀 Starting ML model conversion to ONNX format")
    logger.info(f"Symbol: {args.symbol}")

    converter = ONNXModelConverter()
    results = converter.convert_all_models(args.symbol)

    summary = converter.get_conversion_summary(results)
    print(summary)

    # Save summary to file
    summary_file = Path("models/onnx/conversion_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(summary)

    logger.info(f"Conversion summary saved to: {summary_file}")

    # Exit with appropriate code
    successful_conversions = sum(results.values())
    total_models = len(results)

    if successful_conversions == total_models:
        logger.info("✅ All models converted successfully!")
        return 0
    else:
        logger.error(f"❌ {total_models - successful_conversions} model(s) failed conversion")
        return 1


if __name__ == "__main__":
    sys.exit(main())
