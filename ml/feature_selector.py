"""
Automated Feature Selection for ML Trading Models

This module provides comprehensive feature selection methods to reduce dimensionality,
prevent overfitting, and identify the most predictive features for trading signals.

Methods implemented:
- Feature importance (LightGBM, Random Forest)
- Correlation-based selection (target and inter-feature)
- Mutual information selection
- Recursive Feature Elimination (RFE)
- Hybrid approach combining multiple methods
- Collinearity removal
- Feature category analysis
- Importance visualization
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, RFECV
)
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any
import logging

# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Automated feature selection to reduce dimensionality and prevent overfitting.

    This class implements multiple feature selection strategies:
    - importance: Tree-based feature importance
    - correlation: Low intercorrelation + high target correlation
    - mutual_info: Mutual information with target
    - rfe: Recursive Feature Elimination
    - hybrid: Combination of multiple methods
    """

    def __init__(self, method: str = 'importance'):
        """
        Initialize feature selector.

        Args:
            method: Selection method ('importance', 'correlation', 'mutual_info', 'rfe', 'hybrid')
        """
        self.method = method
        self.selected_features = []
        self.feature_scores = {}
        self.selection_results = {}

        # Validate method
        valid_methods = ['importance', 'correlation', 'mutual_info', 'rfe', 'hybrid']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")

        logger.info(f"FeatureSelector initialized with method: {method}")

    def select_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 50) -> List[str]:
        """
        Select best features using specified method.

        Args:
            X: Feature DataFrame
            y: Target variable
            n_features: Number of features to select

        Returns:
            List of selected feature names
        """
        logger.info(f"🔍 Selecting {n_features} features using {self.method} method...")

        if self.method == 'importance':
            self.selected_features = self._select_by_importance(X, y, n_features)
        elif self.method == 'correlation':
            self.selected_features = self._select_by_correlation(X, y, n_features)
        elif self.method == 'mutual_info':
            self.selected_features = self._select_by_mutual_info(X, y, n_features)
        elif self.method == 'rfe':
            self.selected_features = self._select_by_rfe(X, y, n_features)
        elif self.method == 'hybrid':
            self.selected_features = self._select_hybrid(X, y, n_features)

        logger.info(f"✅ Selected {len(self.selected_features)} features: {self.selected_features[:5]}...")

        return self.selected_features

    def _select_by_importance(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """Select features using LightGBM feature importance"""
        logger.debug("Selecting features by tree importance...")

        # Train LightGBM to get feature importance
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            num_leaves=31,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )

        model.fit(X, y)

        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.feature_scores = dict(zip(importance_df['feature'], importance_df['importance']))
        self.selection_results['importance'] = importance_df

        selected = importance_df.head(n_features)['feature'].tolist()

        logger.debug(f"Top 5 features by importance: {selected[:5]}")

        return selected

    def _select_by_correlation(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """Select features with low intercorrelation but high target correlation"""
        logger.debug("Selecting features by correlation...")

        # Calculate correlation with target
        target_corr = X.corrwith(y).abs().sort_values(ascending=False)

        selected = []
        corr_matrix = X.corr().abs()

        for feature in target_corr.index:
            if len(selected) >= n_features:
                break

            # Check correlation with already selected features
            if len(selected) == 0:
                selected.append(feature)
            else:
                # Only add if not highly correlated with existing features
                max_corr = corr_matrix.loc[feature, selected].max()
                if max_corr < 0.8:  # Correlation threshold
                    selected.append(feature)

        self.feature_scores = target_corr.to_dict()
        self.selection_results['correlation'] = target_corr

        logger.debug(f"Selected {len(selected)} low-correlation features")

        return selected

    def _select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """Select features using mutual information"""
        logger.debug("Selecting features by mutual information...")

        mi_scores = mutual_info_classif(X, y, random_state=42, n_jobs=-1)

        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)

        self.feature_scores = dict(zip(mi_df['feature'], mi_df['mi_score']))
        self.selection_results['mutual_info'] = mi_df

        selected = mi_df.head(n_features)['feature'].tolist()

        logger.debug(f"Selected top {n_features} features by mutual information")

        return selected

    def _select_by_rfe(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """Select features using Recursive Feature Elimination"""
        logger.debug(f"Selecting features by RFE (this may take a while)...")

        estimator = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )

        rfe = RFE(estimator, n_features_to_select=n_features, step=10, verbose=0)
        rfe.fit(X, y)

        selected = X.columns[rfe.support_].tolist()

        # Get feature rankings
        rankings = dict(zip(X.columns, rfe.ranking_))
        self.feature_scores = {k: 1.0/v for k, v in rankings.items()}  # Inverse ranking
        self.selection_results['rfe'] = pd.DataFrame({
            'feature': X.columns,
            'ranking': rfe.ranking_,
            'selected': rfe.support_
        })

        logger.debug(f"Selected {n_features} features by RFE")

        return selected

    def _select_hybrid(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """Hybrid approach: combine multiple methods"""
        logger.debug("Selecting features using hybrid approach...")

        # Get features from multiple methods
        importance_features = set(self._select_by_importance(X, y, n_features * 2))
        mi_features = set(self._select_by_mutual_info(X, y, n_features * 2))

        # Take intersection (features selected by both methods)
        common_features = list(importance_features & mi_features)

        # If not enough, add top-ranked from union
        if len(common_features) < n_features:
            union_features = list(importance_features | mi_features)

            # Rank by combined score
            combined_scores = {}
            for feat in union_features:
                imp_score = self.feature_scores.get(feat, 0)
                # Get MI score from previous run
                combined_scores[feat] = imp_score

            # Sort and select top features
            sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            selected = [f[0] for f in sorted_features[:n_features]]
        else:
            selected = common_features[:n_features]

        logger.debug(f"Hybrid selection: {len(common_features)} features agreed by both methods")

        return selected

    def remove_collinear_features(self, X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """
        Remove highly collinear features.

        Args:
            X: Feature DataFrame
            threshold: Correlation threshold for removal

        Returns:
            List of features with low collinearity
        """
        logger.debug(f"Removing collinear features (threshold={threshold})...")

        corr_matrix = X.corr().abs()

        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        remaining_features = [f for f in X.columns if f not in to_drop]

        logger.debug(f"Removed {len(to_drop)} collinear features, {len(remaining_features)} remaining")

        return remaining_features

    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None) -> None:
        """Plot top N most important features"""
        if not MATPLOTLIB_AVAILABLE:
            logger.info("Matplotlib not available - skipping feature importance plot")
            return

        if not self.feature_scores:
            logger.warning("No feature scores available. Run select_features() first.")
            return

        # Sort features by score
        sorted_features = sorted(
            self.feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        features, scores = zip(*sorted_features)

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance ({self.method})')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Feature importance plot saved: {save_path}")
        else:
            plt.show()

    def analyze_feature_groups(self, X: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze features by category"""
        categories = {
            'price': ['ema', 'price', 'roc', 'sma'],
            'volume': ['volume', 'obv', 'mfi', 'vwap'],
            'volatility': ['atr', 'bb', 'vol', 'kc'],
            'momentum': ['rsi', 'macd', 'stoch', 'williams', 'cci'],
            'trend': ['adx', 'aroon', 'psar', 'supertrend'],
            'pattern': ['pattern', 'candle', 'doji'],
            'time': ['hour', 'day', 'weekend'],
            'statistical': ['zscore', 'skew', 'kurtosis', 'entropy']
        }

        category_scores = {}

        for category, keywords in categories.items():
            # Find features matching category
            matching_features = [
                f for f in self.selected_features
                if any(kw in f.lower() for kw in keywords)
            ]

            if matching_features:
                avg_score = np.mean([
                    self.feature_scores.get(f, 0) for f in matching_features
                ])
                category_scores[category] = {
                    'count': len(matching_features),
                    'avg_score': avg_score,
                    'features': matching_features
                }

        logger.info("\n📊 Feature Category Analysis:")
        for cat, info in sorted(category_scores.items(), key=lambda x: x[1]['avg_score'], reverse=True):
            logger.info(".3f")

        return category_scores

    def get_feature_stability(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
        """Analyze feature stability between train and test sets"""
        logger.debug("Analyzing feature stability...")

        # Select features on training set
        train_features = self.select_features(X_train, y_train)

        # Calculate importance on test set
        test_selector = FeatureSelector(method=self.method)
        test_features = test_selector.select_features(X_test, y_test)

        # Calculate stability metrics
        intersection = set(train_features) & set(test_features)
        union = set(train_features) | set(test_features)

        stability_score = len(intersection) / len(union) if union else 0

        # Feature ranking stability
        common_features = list(intersection)
        if common_features:
            train_ranks = {f: i for i, f in enumerate(train_features)}
            test_ranks = {f: i for i, f in enumerate(test_features)}

            rank_correlation = np.corrcoef(
                [train_ranks[f] for f in common_features],
                [test_ranks[f] for f in common_features]
            )[0, 1]
        else:
            rank_correlation = 0

        stability_metrics = {
            'stability_score': stability_score,
            'rank_correlation': rank_correlation,
            'common_features': len(intersection),
            'total_train_features': len(train_features),
            'total_test_features': len(test_features)
        }

        logger.info(".3f")
        logger.info(".3f")

        return stability_metrics

    def compare_methods(self, X: pd.DataFrame, y: pd.Series,
                       n_features: int = 50) -> pd.DataFrame:
        """Compare different selection methods"""
        logger.info("Comparing feature selection methods...")

        methods = ['importance', 'correlation', 'mutual_info', 'rfe']
        results = []

        for method in methods:
            selector = FeatureSelector(method=method)
            features = selector.select_features(X, y, n_features)

            # Calculate average importance score
            avg_score = np.mean([selector.feature_scores.get(f, 0) for f in features])

            results.append({
                'method': method,
                'n_features': len(features),
                'avg_score': avg_score,
                'top_features': features[:5]
            })

        comparison_df = pd.DataFrame(results)
        logger.info("\n📊 Method Comparison:")
        logger.info(comparison_df.to_string(index=False))

        return comparison_df
