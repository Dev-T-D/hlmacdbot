#!/usr/bin/env python3
"""
ONNX Model Accuracy Validation Suite

Verifies that ONNX model conversion preserves prediction accuracy.
Compares predictions between original pickle models and converted ONNX models.

Usage:
    python tests/test_onnx_accuracy.py
"""

import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ML libraries
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# ONNX runtime
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ONNXAccuracyValidator:
    """
    Validates that ONNX models produce identical predictions to original models.
    """

    def __init__(self):
        self.test_data = self._create_test_data()
        self.tolerance = 1e-5  # Acceptable difference between predictions

    def _create_test_data(self) -> np.ndarray:
        """Create test feature data for validation."""
        np.random.seed(42)  # Reproducible results
        return np.random.randn(100, 80).astype(np.float32)  # 100 samples, 80 features

    def load_original_model(self, model_path: Path, model_type: str) -> Any:
        """
        Load original model from pickle file.

        Args:
            model_path: Path to pickled model
            model_type: Type of model ('lightgbm', 'xgboost', 'random_forest')

        Returns:
            Loaded model object
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        return model

    def load_onnx_model(self, model_path: Path) -> ort.InferenceSession:
        """
        Load ONNX model for inference.

        Args:
            model_path: Path to ONNX model

        Returns:
            ONNX inference session
        """
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")

        session = ort.InferenceSession(str(model_path))
        return session

    def get_original_predictions(self, model: Any, model_type: str, features: np.ndarray) -> np.ndarray:
        """
        Get predictions from original model.

        Args:
            model: Original model object
            model_type: Type of model
            features: Feature array

        Returns:
            Prediction array
        """
        if model_type == 'lightgbm':
            predictions = model.predict(features)
        elif model_type == 'xgboost':
            dmatrix = xgb.DMatrix(features)
            predictions = model.predict(dmatrix)
        elif model_type == 'random_forest':
            predictions = model.predict_proba(features)[:, 1]  # Probability of positive class
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return predictions

    def get_onnx_predictions(self, session: ort.InferenceSession, features: np.ndarray) -> np.ndarray:
        """
        Get predictions from ONNX model.

        Args:
            session: ONNX inference session
            features: Feature array

        Returns:
            Prediction array
        """
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: features})[0]

        # Handle different output formats
        if result.shape[1] == 2:  # Binary classification
            predictions = result[:, 1]  # Probability of positive class
        elif result.shape[1] == 1:  # Regression or single output
            predictions = result[:, 0]
            # Apply sigmoid for regression outputs
            predictions = 1.0 / (1.0 + np.exp(-predictions))
        else:
            # Multi-class - take max probability
            predictions = np.max(result, axis=1)

        return predictions

    def validate_model_accuracy(self, model_type: str) -> Dict[str, Any]:
        """
        Validate accuracy for a specific model type.

        Args:
            model_type: Type of model to validate

        Returns:
            Dict with validation results
        """
        logger.info(f"🔍 Validating {model_type} model accuracy...")

        # Define file paths
        pickle_path = Path(f"models/{model_type}_model.pkl")
        onnx_path = Path(f"models/onnx/{model_type}.onnx")

        # Check if files exist
        if not pickle_path.exists():
            return {
                'model_type': model_type,
                'status': 'skipped',
                'reason': f'Pickle model not found: {pickle_path}'
            }

        if not onnx_path.exists():
            return {
                'model_type': model_type,
                'status': 'skipped',
                'reason': f'ONNX model not found: {onnx_path}'
            }

        try:
            # Load models
            original_model = self.load_original_model(pickle_path, model_type)
            onnx_session = self.load_onnx_model(onnx_path)

            # Get predictions
            original_preds = self.get_original_predictions(original_model, model_type, self.test_data)
            onnx_preds = self.get_onnx_predictions(onnx_session, self.test_data)

            # Calculate differences
            differences = np.abs(original_preds - onnx_preds)
            max_diff = float(np.max(differences))
            mean_diff = float(np.mean(differences))
            std_diff = float(np.std(differences))

            # Check if within tolerance
            within_tolerance = max_diff <= self.tolerance

            # Calculate correlation
            correlation = float(np.corrcoef(original_preds, onnx_preds)[0, 1])

            result = {
                'model_type': model_type,
                'status': 'passed' if within_tolerance else 'failed',
                'max_difference': max_diff,
                'mean_difference': mean_diff,
                'std_difference': std_diff,
                'correlation': correlation,
                'tolerance': self.tolerance,
                'samples_tested': len(self.test_data),
                'within_tolerance': within_tolerance
            }

            logger.info(f"  📊 Max diff: {max_diff:.8f}")
            logger.info(f"  📊 Mean diff: {mean_diff:.8f}")
            logger.info(f"  📊 Correlation: {correlation:.6f}")
            logger.info(f"  ✅ Within tolerance: {within_tolerance}")

            return result

        except Exception as e:
            logger.error(f"❌ Validation failed for {model_type}: {e}")
            return {
                'model_type': model_type,
                'status': 'error',
                'error': str(e)
            }

    def validate_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate accuracy for all available models.

        Returns:
            Dict with results for each model type
        """
        logger.info("🚀 Starting ONNX accuracy validation suite")

        model_types = ['lightgbm', 'xgboost', 'random_forest', 'lstm']
        results = {}

        for model_type in model_types:
            results[model_type] = self.validate_model_accuracy(model_type)

        return results

    def generate_accuracy_report(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a comprehensive accuracy report.

        Args:
            results: Validation results from validate_all_models

        Returns:
            Formatted report string
        """
        report = "🔍 ONNX Model Accuracy Validation Report\n"
        report += "=" * 50 + "\n\n"

        # Summary statistics
        total_models = len(results)
        passed_models = sum(1 for r in results.values() if r['status'] == 'passed')
        skipped_models = sum(1 for r in results.values() if r['status'] == 'skipped')
        failed_models = sum(1 for r in results.values() if r['status'] == 'failed')
        error_models = sum(1 for r in results.values() if r['status'] == 'error')

        report += f"Total Models Tested: {total_models}\n"
        report += f"✅ Passed: {passed_models}\n"
        report += f"⏭️  Skipped: {skipped_models}\n"
        report += f"❌ Failed: {failed_models}\n"
        report += f"🚨 Errors: {error_models}\n\n"

        # Detailed results
        report += "📋 DETAILED RESULTS\n"
        report += "-" * 30 + "\n"

        for model_type, result in results.items():
            status_emoji = {
                'passed': '✅',
                'failed': '❌',
                'skipped': '⏭️ ',
                'error': '🚨'
            }.get(result['status'], '❓')

            report += f"{status_emoji} {model_type.upper()}: {result['status'].upper()}\n"

            if result['status'] == 'passed':
                report += f"   Max Diff: {result['max_difference']:.8f}\n"
                report += f"   Mean Diff: {result['mean_difference']:.8f}\n"
                report += f"   Correlation: {result['correlation']:.6f}\n"
            elif result['status'] in ['failed', 'error']:
                if 'error' in result:
                    report += f"   Error: {result['error']}\n"
                else:
                    report += f"   Max Diff: {result['max_difference']:.8f} (tolerance: {result['tolerance']})\n"

            report += "\n"

        # Overall assessment
        report += "🎯 ACCURACY ASSESSMENT\n"

        if passed_models == total_models - skipped_models:
            report += "✅ EXCELLENT: All converted models preserve accuracy!\n"
            report += "   ONNX conversion is mathematically equivalent.\n"
        elif passed_models > 0:
            report += "⚠️  PARTIAL: Some models preserve accuracy, others need investigation.\n"
        else:
            report += "❌ CRITICAL: No models preserve accuracy - conversion issues detected.\n"

        # Recommendations
        report += "\n💡 RECOMMENDATIONS\n"
        if failed_models > 0:
            report += "- Investigate failed model conversions\n"
            report += "- Check ONNX conversion parameters\n"
            report += "- Verify model serialization format\n"

        if error_models > 0:
            report += "- Fix model loading errors\n"
            report += "- Ensure all dependencies are installed\n"

        if skipped_models > 0:
            report += "- Generate missing models using training pipeline\n"
            report += "- Run: python scripts/convert_models_to_onnx.py\n"

        return report

    def assert_accuracy_requirements(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Assert that accuracy requirements are met.

        Args:
            results: Validation results

        Raises:
            AssertionError: If accuracy requirements are not met
        """
        # At least one model should be available and pass validation
        available_models = [r for r in results.values() if r['status'] != 'skipped']
        passed_models = [r for r in available_models if r['status'] == 'passed']

        assert len(available_models) > 0, "No models available for validation"

        # At least 50% of available models should pass
        pass_rate = len(passed_models) / len(available_models)
        assert pass_rate >= 0.5, f"Only {pass_rate:.1%} of models passed accuracy validation"

        # All passed models should have high correlation (>0.999)
        for result in passed_models:
            assert result['correlation'] > 0.999, \
                f"{result['model_type']} correlation {result['correlation']:.6f} too low"


def run_accuracy_tests():
    """Run the complete accuracy validation suite."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    validator = ONNXAccuracyValidator()

    try:
        # Run validation
        results = validator.validate_all_models()

        # Generate report
        report = validator.generate_accuracy_report(results)
        print("\n" + "="*80)
        print("🔍 ONNX ACCURACY VALIDATION RESULTS")
        print("="*80)
        print(report)

        # Assert requirements
        validator.assert_accuracy_requirements(results)

        print("\n✅ All accuracy requirements met!")
        return True

    except AssertionError as e:
        print(f"\n❌ Accuracy requirement not met: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    success = run_accuracy_tests()
    sys.exit(0 if success else 1)
