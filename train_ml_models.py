#!/usr/bin/env python3
"""
ML Model Training Script

This script runs the complete ML training pipeline for price direction prediction.
Trains ensemble models (LightGBM, XGBoost, Random Forest, LSTM) and saves them for production use.

Usage:
    python train_ml_models.py --symbol BTCUSDT --days 180

Requirements:
    - Historical market data access
    - Sufficient RAM for model training
    - Optional: GPU for LSTM training acceleration
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_training_pipeline import MLTrainingPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ML models for price direction prediction')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--days', type=int, default=180, help='Days of historical data to use')
    parser.add_argument('--config', help='Path to custom config file')
    parser.add_argument('--optimize', action='store_true', help='Run hyperparameter optimization')
    parser.add_argument('--skip-validation', action='store_true', help='Skip model validation')

    args = parser.parse_args()

    try:
        logger.info("🚀 Starting ML Model Training")
        logger.info(f"Symbol: {args.symbol}")
        logger.info(f"Training days: {args.days}")

        # Initialize training pipeline
        config = None
        if args.config and os.path.exists(args.config):
            import json
            with open(args.config) as f:
                config = json.load(f)
            logger.info(f"Loaded config from {args.config}")

        pipeline = MLTrainingPipeline(args.symbol, config)

        # Run data collection
        logger.info("📊 Collecting training data...")
        training_data = pipeline.collect_training_data()

        if training_data.empty:
            logger.error("❌ No training data collected")
            return 1

        logger.info(f"✅ Collected {len(training_data)} training samples")

        # Run hyperparameter optimization if requested
        if args.optimize:
            logger.info("🔍 Running hyperparameter optimization...")

            # Optimize each model type
            for model_type in ['lightgbm', 'xgboost', 'random_forest']:
                try:
                    best_params = pipeline.optimize_hyperparameters(model_type)
                    logger.info(f"✅ {model_type} best params: {best_params}")

                    # Update config with best parameters
                    if config is None:
                        config = pipeline.config

                    config['models'][model_type].update(best_params)

                except Exception as e:
                    logger.warning(f"⚠️ Failed to optimize {model_type}: {e}")

        # Train models
        logger.info("🤖 Training ML models...")
        training_results = pipeline.train_models(training_data)

        if not training_results.get('models'):
            logger.error("❌ Model training failed")
            return 1

        logger.info(f"✅ Trained {len(training_results['models'])} models")

        # Validate models
        if not args.skip_validation:
            logger.info("📈 Validating models...")
            validation_results = pipeline.validate_models()

            # Print validation summary
            print("\n" + "="*60)
            print("MODEL VALIDATION RESULTS")
            print("="*60)

            for model_name, metrics in validation_results.items():
                if isinstance(metrics, dict) and 'error' not in metrics:
                    print(".4f"
                          ".4f"
                          ".4f"
                          ".4f"
                          ".4f")
                else:
                    print(f"{model_name:15} - ERROR: {metrics.get('error', 'Unknown')}")

            print("="*60)

            # Check if ensemble performance is good enough
            ensemble_metrics = validation_results.get('lightgbm', {})  # Use LightGBM as proxy for ensemble
            if ensemble_metrics.get('auc', 0) < 0.52:
                logger.warning("⚠️ Model AUC below 0.52 - consider more training data or feature engineering")
            else:
                logger.info("✅ Model performance acceptable")

        # Save training summary
        summary = {
            'training_timestamp': datetime.now().isoformat(),
            'symbol': args.symbol,
            'training_days': args.days,
            'n_samples': len(training_data),
            'models_trained': list(training_results.get('models', {}).keys()),
            'validation_results': validation_results if not args.skip_validation else None,
            'hyperparameter_optimization': args.optimize
        }

        summary_file = f"models/{args.symbol}/training_summary.json"
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)

        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"📝 Training summary saved to {summary_file}")

        # Final success message
        print("\n" + "🎉 ML MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("   Models saved and ready for production use")
        print(f"   Model directory: models/{args.symbol}/")
        print("   Update your trading bot config to enable ML enhancement")
        return 0

    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
