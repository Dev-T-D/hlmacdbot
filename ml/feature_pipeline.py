"""
End-to-End Feature Engineering Pipeline

This module provides a complete pipeline for feature engineering, selection, scaling,
and preprocessing for ML trading models.

Pipeline steps:
1. Feature Engineering (150+ features)
2. Feature Selection (reduce dimensionality)
3. Missing Value Imputation
4. Feature Scaling (RobustScaler)
5. Pipeline Serialization/Deserialization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

from feature_engineering import AdvancedFeatureEngine
from ml.feature_selector import FeatureSelector

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    End-to-end feature engineering pipeline.

    This class orchestrates the complete feature engineering process:
    - Feature creation from raw OHLCV data
    - Automated feature selection
    - Data preprocessing (imputation, scaling)
    - Pipeline persistence and loading
    """

    def __init__(self, feature_engine: Optional[AdvancedFeatureEngine] = None,
                 feature_selector: Optional[FeatureSelector] = None,
                 scaler: str = 'robust'):
        """
        Initialize the feature pipeline.

        Args:
            feature_engine: Feature engineering instance
            feature_selector: Feature selection instance
            scaler: Scaling method ('standard', 'robust', or None)
        """
        self.feature_engine = feature_engine or AdvancedFeatureEngine()
        self.feature_selector = feature_selector or FeatureSelector(method='hybrid')
        self.selected_features = []

        # Choose scaler
        if scaler == 'standard':
            self.scaler = StandardScaler()
        elif scaler == 'robust':
            self.scaler = RobustScaler()  # Better for outliers
        else:
            self.scaler = None

        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False

        logger.info("FeaturePipeline initialized")
        logger.info(f"Scaler: {scaler}, Selector: {self.feature_selector.method}")

    def fit(self, df: pd.DataFrame, target: pd.Series,
            n_features: int = 50, remove_collinear: bool = True) -> 'FeaturePipeline':
        """
        Fit the feature pipeline.

        Args:
            df: Raw OHLCV DataFrame
            target: Target variable (1 for price up, 0 for down)
            n_features: Number of features to select
            remove_collinear: Whether to remove highly collinear features

        Returns:
            Self for method chaining
        """
        logger.info("🔧 Fitting feature pipeline...")

        # 1. Engineer all features
        logger.debug("   Step 1: Engineering features...")
        features_df = self.feature_engine.create_all_features(df)

        # 2. Remove rows with NaN (from indicators)
        logger.debug("   Step 2: Handling missing values...")
        features_df = features_df.dropna()
        target = target.loc[features_df.index]

        if len(features_df) == 0:
            raise ValueError("No valid data after feature engineering")

        logger.info(f"   Processed {len(features_df)} samples with {len(features_df.columns)} initial features")

        # 3. Select best features
        logger.debug("   Step 3: Selecting features...")
        self.selected_features = self.feature_selector.select_features(
            features_df, target, n_features
        )

        # 4. Remove collinear features if requested
        if remove_collinear:
            logger.debug("   Step 4: Removing collinear features...")
            X_selected = features_df[self.selected_features]
            non_collinear = self.feature_selector.remove_collinear_features(X_selected)
            self.selected_features = non_collinear

        # 5. Fit imputer (for any remaining NaN in production)
        X_selected = features_df[self.selected_features]
        self.imputer.fit(X_selected)

        # 6. Fit scaler
        if self.scaler:
            X_imputed = self.imputer.transform(X_selected)
            self.scaler.fit(X_imputed)

        self.is_fitted = True

        logger.info(f"✅ Pipeline fitted with {len(self.selected_features)} features")
        logger.info(f"   Final features: {self.selected_features[:5]}...")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data through the pipeline.

        Args:
            df: Raw OHLCV DataFrame

        Returns:
            Processed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        # 1. Engineer features
        features_df = self.feature_engine.create_all_features(df)

        # 2. Select features
        X = features_df[self.selected_features]

        # 3. Impute
        X_imputed = self.imputer.transform(X)

        # 4. Scale
        if self.scaler:
            X_scaled = self.scaler.transform(X_imputed)
            return pd.DataFrame(
                X_scaled,
                columns=self.selected_features,
                index=X.index
            )

        return pd.DataFrame(
            X_imputed,
            columns=self.selected_features,
            index=X.index
        )

    def fit_transform(self, df: pd.DataFrame, target: pd.Series,
                     n_features: int = 50, remove_collinear: bool = True) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(df, target, n_features, remove_collinear)
        return self.transform(df)

    def save(self, filepath: str) -> None:
        """Save pipeline to disk"""
        pipeline_data = {
            'selected_features': self.selected_features,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_engine_config': self.feature_engine.config,
            'feature_selector_method': self.feature_selector.method,
            'feature_scores': self.feature_selector.feature_scores,
            'is_fitted': self.is_fitted
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline_data, filepath)
        logger.info(f"💾 Pipeline saved: {filepath}")

    def load(self, filepath: str) -> 'FeaturePipeline':
        """Load pipeline from disk"""
        pipeline_data = joblib.load(filepath)

        self.selected_features = pipeline_data['selected_features']
        self.scaler = pipeline_data['scaler']
        self.imputer = pipeline_data['imputer']

        # Recreate feature engine and selector with saved config
        self.feature_engine = AdvancedFeatureEngine(pipeline_data['feature_engine_config'])
        self.feature_selector = FeatureSelector(pipeline_data['feature_selector_method'])
        self.feature_selector.feature_scores = pipeline_data['feature_scores']

        self.is_fitted = pipeline_data['is_fitted']

        logger.info(f"📂 Pipeline loaded: {filepath}")
        logger.info(f"   {len(self.selected_features)} features loaded")

        return self

    def get_feature_names(self) -> List[str]:
        """Get list of selected feature names"""
        return self.selected_features.copy()

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_selector.feature_scores.copy()

    def analyze_pipeline(self) -> Dict[str, Any]:
        """Analyze the fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted")

        analysis = {
            'n_features': len(self.selected_features),
            'feature_names': self.selected_features,
            'scaler_type': type(self.scaler).__name__ if self.scaler else None,
            'imputer_strategy': self.imputer.strategy,
            'selector_method': self.feature_selector.method,
            'feature_engine_config': self.feature_engine.config
        }

        # Category analysis
        analysis['category_analysis'] = self.feature_selector.analyze_feature_groups(
            pd.DataFrame(columns=self.selected_features)  # Dummy df for analysis
        )

        # Importance statistics
        if self.feature_selector.feature_scores:
            scores = list(self.feature_selector.feature_scores.values())
            analysis['importance_stats'] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'top_5': sorted(self.feature_selector.feature_scores.items(),
                              key=lambda x: x[1], reverse=True)[:5]
            }

        return analysis

    def create_feature_report(self, save_path: Optional[str] = None) -> str:
        """Create a comprehensive feature analysis report"""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted")

        analysis = self.analyze_pipeline()

        report = "=" * 80 + "\n"
        report += "FEATURE ENGINEERING PIPELINE REPORT\n"
        report += "=" * 80 + "\n\n"

        report += f"Pipeline Status: {'Fitted' if self.is_fitted else 'Not fitted'}\n"
        report += f"Selected Features: {len(self.selected_features)}\n"
        report += f"Feature Selector: {analysis['selector_method']}\n"
        report += f"Scaler: {analysis.get('scaler_type', 'None')}\n"
        report += f"Imputer Strategy: {analysis['imputer_strategy']}\n\n"

        # Feature categories
        report += "FEATURE CATEGORIES:\n"
        report += "-" * 40 + "\n"
        if 'category_analysis' in analysis:
            for cat, info in sorted(analysis['category_analysis'].items(),
                                  key=lambda x: x[1]['avg_score'], reverse=True):
                report += ".3f"
        report += "\n"

        # Top features
        report += "TOP 10 FEATURES:\n"
        report += "-" * 40 + "\n"
        if 'importance_stats' in analysis and 'top_5' in analysis['importance_stats']:
            for i, (feature, score) in enumerate(analysis['importance_stats']['top_5'], 1):
                report += f"{i:2d}. {feature:<30} {score:.4f}\n"
        report += "\n"

        # Importance statistics
        if 'importance_stats' in analysis:
            stats = analysis['importance_stats']
            report += "IMPORTANCE STATISTICS:\n"
            report += "-" * 40 + "\n"
            report += ".4f"
            report += ".4f"
            report += ".4f"
            report += ".4f"
            report += "\n"

        # All selected features
        report += "ALL SELECTED FEATURES:\n"
        report += "-" * 40 + "\n"
        for i, feature in enumerate(self.selected_features, 1):
            score = self.feature_selector.feature_scores.get(feature, 0)
            report += f"{i:3d}. {feature:<35} {score:.4f}\n"

        report += "\n" + "=" * 80 + "\n"

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"📄 Feature report saved: {save_path}")

        return report
