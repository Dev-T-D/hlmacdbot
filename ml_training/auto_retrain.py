import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timezone, timedelta
import schedule
import time
import joblib

from data_collection.data_manager import DataManager
from feature_engineering import AdvancedFeatureEngine
from ml.feature_selector import FeatureSelector
from ml.feature_pipeline import FeaturePipeline
from ml_training_pipeline import MLTrainingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoRetrainer:
    """
    Automated ML model retraining system

    Features:
    - Scheduled retraining (daily/weekly/monthly)
    - Data quality checks before training
    - Model performance validation
    - Automatic deployment of better models
    - Training history tracking
    """

    def __init__(self, config_path: str = 'config/retrain_config.json'):
        self.data_manager = DataManager()
        self.feature_engine = AdvancedFeatureEngine()
        self.config = self._load_config(config_path)

        # Training history
        self.training_history = []

    def _load_config(self, config_path):
        """Load retraining configuration"""
        import json

        default_config = {
            'symbols': ['BTC', 'ETH', 'SOL'],
            'timeframes': ['1h'],
            'retrain_schedule': 'daily',  # daily, weekly, monthly
            'min_new_samples': 100,  # Minimum new samples before retraining
            'min_accuracy_threshold': 0.55,  # Minimum accuracy to deploy
            'validation_split': 0.2,
            'models_to_train': ['lightgbm', 'xgboost', 'random_forest'],
            'n_features': 50
        }

        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        except FileNotFoundError:
            logger.warning(f"Config not found, using defaults")

        return default_config

    def start_scheduled_retraining(self):
        """Start scheduled retraining based on config"""
        schedule_type = self.config['retrain_schedule']

        logger.info(f"🔄 Starting automated retraining ({schedule_type})")

        if schedule_type == 'daily':
            # Retrain every day at 2 AM UTC
            schedule.every().day.at("02:00").do(self.retrain_all_models)
        elif schedule_type == 'weekly':
            # Retrain every Monday at 2 AM UTC
            schedule.every().monday.at("02:00").do(self.retrain_all_models)
        elif schedule_type == 'monthly':
            # Retrain on 1st of every month at 2 AM UTC
            schedule.every().day.at("02:00").do(self._monthly_retrain_check)

        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def _monthly_retrain_check(self):
        """Check if it's the 1st of the month"""
        if datetime.now(timezone.utc).day == 1:
            self.retrain_all_models()

    def retrain_all_models(self):
        """Retrain models for all configured symbols/timeframes"""
        logger.info(f"\n{'='*80}")
        logger.info("🚀 STARTING AUTOMATED MODEL RETRAINING")
        logger.info(f"{'='*80}")

        start_time = datetime.now(timezone.utc)

        for symbol in self.config['symbols']:
            for timeframe in self.config['timeframes']:
                try:
                    self.retrain_model(symbol, timeframe)
                except Exception as e:
                    logger.error(f"Error retraining {symbol} {timeframe}: {e}")

        duration = datetime.now(timezone.utc) - start_time
        logger.info(f"\n✅ Retraining complete! Duration: {duration}")

    def retrain_model(self, symbol: str, timeframe: str):
        """Retrain model for specific symbol/timeframe"""
        logger.info(f"\n📊 Retraining {symbol} {timeframe}")

        # STEP 1: Check data availability
        X, y = self.data_manager.get_training_data(symbol, timeframe)

        if len(X) < self.config['min_new_samples']:
            logger.warning(f"⚠️  Not enough new data ({len(X)} samples)")
            return

        logger.info(f"   Training samples: {len(X)}")

        # STEP 2: Generate features
        logger.info("   Generating features...")
        features_df = self.feature_engine.create_all_features(X)
        features_df = features_df.dropna()

        # Align labels
        y = y.loc[features_df.index]

        logger.info(f"   Features: {len(features_df.columns)}")
        logger.info(f"   Samples after cleaning: {len(features_df)}")

        # STEP 3: Feature selection
        logger.info("   Selecting features...")
        selector = FeatureSelector(method='hybrid')
        selected_features = selector.select_features(
            features_df, y,
            n_features=self.config['n_features']
        )

        X_selected = features_df[selected_features]

        # STEP 4: Train-test split (time-based)
        split_idx = int(len(X_selected) * (1 - self.config['validation_split']))

        X_train = X_selected.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_val = X_selected.iloc[split_idx:]
        y_val = y.iloc[split_idx:]

        logger.info(f"   Train: {len(X_train)}, Validation: {len(X_val)}")

        # STEP 5: Train models
        logger.info("   Training models...")

        pipeline = MLTrainingPipeline()
        results = pipeline.train_models_with_split_data(X_train, y_train, X_val, y_val)

        # STEP 6: Evaluate and decide deployment
        best_model = None
        best_accuracy = 0

        for model_name, metrics in results.items():
            accuracy = metrics.get('accuracy', 0)

            logger.info(f"   {model_name}: {accuracy:.4f} accuracy")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name

        # STEP 7: Deploy if better than threshold
        if best_accuracy >= self.config['min_accuracy_threshold']:
            logger.info(f"✅ Deploying {best_model} (accuracy: {best_accuracy:.4f})")

            self._deploy_model(symbol, timeframe, best_model, results[best_model])

            # Record training
            self.training_history.append({
                'timestamp': datetime.now(timezone.utc),
                'symbol': symbol,
                'timeframe': timeframe,
                'model': best_model,
                'accuracy': best_accuracy,
                'samples': len(X_train),
                'deployed': True
            })
        else:
            logger.warning(f"⚠️  Model accuracy too low ({best_accuracy:.4f} < {self.config['min_accuracy_threshold']})")
            logger.warning(f"   Keeping existing model")

            self.training_history.append({
                'timestamp': datetime.now(timezone.utc),
                'symbol': symbol,
                'timeframe': timeframe,
                'model': best_model,
                'accuracy': best_accuracy,
                'samples': len(X_train),
                'deployed': False
            })

    def _deploy_model(self, symbol: str, timeframe: str, model_name: str, metrics: dict):
        """Deploy trained model to production"""
        # Create deployment directory
        deploy_dir = Path('models') / symbol / 'production'
        deploy_dir.mkdir(parents=True, exist_ok=True)

        # Copy model files
        source_dir = Path('models') / symbol / model_name

        if source_dir.exists():
            import shutil

            # Backup existing production model
            if (deploy_dir / f"{model_name}.pkl").exists():
                backup_path = deploy_dir / f"{model_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                shutil.copy(deploy_dir / f"{model_name}.pkl", backup_path)

            # Deploy new model
            shutil.copy(source_dir / f"{model_name}.pkl", deploy_dir / f"{model_name}.pkl")

            # Save metrics
            metrics_path = deploy_dir / 'metrics.json'
            import json
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"💾 Model deployed: {deploy_dir}")

    def get_training_statistics(self) -> pd.DataFrame:
        """Get retraining statistics"""
        if not self.training_history:
            return pd.DataFrame()

        df = pd.DataFrame(self.training_history)
        return df

# Run automated retraining
if __name__ == "__main__":
    retrainer = AutoRetrainer()

    # Option 1: Run once immediately
    # retrainer.retrain_all_models()

    # Option 2: Start scheduled retraining
    retrainer.start_scheduled_retraining()
