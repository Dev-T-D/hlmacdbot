"""
Advanced Feature Engineering for ML-Enhanced Trading Strategy

This module provides comprehensive feature engineering for financial time series prediction,
creating 150+ advanced features from technical indicators, market microstructure, and temporal patterns.

Key Features:
- Price-Based Features: Multiple EMAs, ROC, momentum, price ratios, gaps
- Volume-Based Features: OBV, MFI, VWAP, volume ratios, accumulation/distribution
- Volatility Features: ATR, Bollinger Bands, Keltner Channels, historical/realized vol
- Momentum Features: RSI, MACD variations, Stochastic, Williams %R, CCI, Ultimate Oscillator
- Trend Features: ADX, Aroon, Parabolic SAR, Supertrend, linear regression slopes
- Pattern Recognition: Candlestick patterns, consecutive moves, divergence detection
- Microstructure Features: Bid-ask spread, order book imbalance, trade flow
- Time-Based Features: Cyclical encoding, funding rate proximity, weekend detection
- Statistical Features: Z-score, skewness, kurtosis, entropy, Hurst exponent
- Interaction Features: Volume-price correlation, momentum-volume confirmation

Real-time computation optimized for live trading with Numba JIT compilation where possible.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from input_sanitizer import InputSanitizer

# Optional imports for advanced features
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback without JIT
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)

# Log optional dependency status
if not TALIB_AVAILABLE:
    logger.warning("TA-Lib not available - some pattern features disabled")


# Global helper functions for Numba compatibility
@jit(nopython=True)
def _calculate_hurst_exponent_global(prices: np.ndarray) -> float:
    """Calculate Hurst exponent for trend persistence (global function for Numba compatibility)"""
    if len(prices) < 20:
        return 0.5  # Random walk

    lags = np.arange(2, min(20, len(prices)//2))
    tau = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag >= len(prices):
            tau[i] = np.nan
        else:
            diff = prices[lag:] - prices[:-lag]
            tau[i] = np.std(diff)

    # Remove NaN values
    valid_idx = ~np.isnan(tau)
    lags = lags[valid_idx]
    tau = tau[valid_idx]

    if len(lags) < 2:
        return 0.5

    # Linear regression on log-log plot
    log_lags = np.log(lags)
    log_tau = np.log(tau)

    slope = np.polyfit(log_lags, log_tau, 1)[0]
    hurst = slope * 2.0

    return max(0.0, min(1.0, hurst))  # Clamp to [0, 1]


class AdvancedFeatureEngine:
    """
    Advanced feature engineering for ML-based trading signals.

    This class creates 150+ features from multiple domains:
    - Price-based features (30+ features)
    - Volume-based features (15+ features)
    - Volatility features (20+ features)
    - Momentum features (25+ features)
    - Trend features (20+ features)
    - Pattern recognition (10+ features)
    - Microstructure features (10+ features)
    - Time-based features (10+ features)
    - Statistical features (10+ features)
    - Interaction features (10+ features)

    All features are designed for real-time computation and ML model training.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the advanced feature engineer.

        Args:
            config: Configuration dictionary with feature parameters
        """
        self.config = config or self._get_default_config()
        self.feature_names = []

        # Feature calculation parameters
        self.ema_periods = [3, 5, 8, 13, 21, 34, 55, 89, 144, 200]
        self.volume_periods = [5, 10, 20, 50]
        self.volatility_periods = [7, 14, 21]
        self.momentum_periods = [7, 14, 21, 28]
        self.trend_periods = [10, 20, 50]

        # Initialize caches for performance
        self._cache = {}

        logger.info("AdvancedFeatureEngine initialized")
        logger.info(f"Numba JIT: {'Enabled' if NUMBA_AVAILABLE else 'Disabled'}")
        logger.info(f"TA-Lib: {'Enabled' if TALIB_AVAILABLE else 'Disabled'}")

    def _get_default_config(self) -> Dict:
        """Get default configuration parameters."""
        return {
            'feature_engineering': {
                'enabled_categories': ['price', 'volume', 'volatility', 'momentum', 'trend', 'pattern', 'time', 'statistical', 'interaction'],
                'numba_enabled': True,
                'talib_enabled': True,
                'cache_enabled': True,
                'max_cache_size': 1000
            }
        }

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set from OHLCV data

        Args:
            df: DataFrame with columns: open, high, low, close, volume

        Returns:
            DataFrame with all engineered features
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("DataFrame must contain OHLCV columns")

        logger.info(f"🔧 Creating features from {len(df)} candles")

        features_df = df.copy()

        # 1. PRICE-BASED FEATURES
        if 'price' in self.config['feature_engineering']['enabled_categories']:
            logger.debug("Adding price-based features...")
            features_df = self._add_price_features(features_df)

        # 2. VOLUME-BASED FEATURES
        if 'volume' in self.config['feature_engineering']['enabled_categories']:
            logger.debug("Adding volume-based features...")
            features_df = self._add_volume_features(features_df)

        # 3. VOLATILITY FEATURES
        if 'volatility' in self.config['feature_engineering']['enabled_categories']:
            logger.debug("Adding volatility features...")
            features_df = self._add_volatility_features(features_df)

        # 4. MOMENTUM FEATURES
        if 'momentum' in self.config['feature_engineering']['enabled_categories']:
            logger.debug("Adding momentum features...")
            features_df = self._add_momentum_features(features_df)

        # 5. TREND FEATURES
        if 'trend' in self.config['feature_engineering']['enabled_categories']:
            logger.debug("Adding trend features...")
            features_df = self._add_trend_features(features_df)

        # 6. PATTERN RECOGNITION
        if 'pattern' in self.config['feature_engineering']['enabled_categories']:
            logger.debug("Adding pattern recognition features...")
            features_df = self._add_pattern_features(features_df)

        # 7. MICROSTRUCTURE FEATURES (placeholder)
        if 'microstructure' in self.config['feature_engineering']['enabled_categories']:
            logger.debug("Adding microstructure features...")
            features_df = self._add_microstructure_features(features_df)

        # 8. TIME-BASED FEATURES
        if 'time' in self.config['feature_engineering']['enabled_categories']:
            logger.debug("Adding time-based features...")
            features_df = self._add_time_features(features_df)

        # 9. STATISTICAL FEATURES
        if 'statistical' in self.config['feature_engineering']['enabled_categories']:
            logger.debug("Adding statistical features...")
            features_df = self._add_statistical_features(features_df)

        # 10. INTERACTION FEATURES
        if 'interaction' in self.config['feature_engineering']['enabled_categories']:
            logger.debug("Adding interaction features...")
            features_df = self._add_interaction_features(features_df)

        # Count total features created
        original_cols = ['open', 'high', 'low', 'close', 'volume']
        total_features = len(features_df.columns) - len(original_cols)
        self.feature_names = [col for col in features_df.columns if col not in original_cols]

        logger.info(f"✅ Created {total_features} features across {len(self.config['feature_engineering']['enabled_categories'])} categories")

        return features_df

    # ==========================================
    # PRICE-BASED FEATURES
    # ==========================================

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced price-based features"""

        # Multiple EMA periods
        for period in self.ema_periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # Price ratios (current price vs various EMAs)
        # Only use periods that exist in ema_periods
        ratio_periods = [p for p in [21, 50, 200] if p in self.ema_periods]
        for period in ratio_periods:
            df[f'price_ema{period}_ratio'] = df['close'] / df[f'ema_{period}']

        # EMA cross signals (only if required periods exist)
        if 8 in self.ema_periods and 21 in self.ema_periods:
            df['ema_cross_fast_slow'] = (df['ema_8'] > df['ema_21']).astype(int)
            df['ema_cross_momentum'] = df['ema_8'] - df['ema_21']

        # Price momentum (ROC over multiple periods)
        for period in [1, 3, 5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(period)

        # Price acceleration (second derivative)
        df['price_acceleration'] = df['close'].pct_change().diff()

        # Typical price and weighted close
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['weighted_close'] = (df['high'] + df['low'] + df['close'] * 2) / 4

        # Range indicators
        df['high_low_range'] = df['high'] - df['low']
        df['high_low_range_pct'] = (df['high'] - df['low']) / df['close']

        # Gap detection
        df['open_close_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        logger.debug(f"Added {len([col for col in df.columns if 'ema_' in col or 'roc_' in col or 'price_' in col])} price features")
        return df

    # ==========================================
    # VOLUME-BASED FEATURES
    # ==========================================

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced volume-based features"""

        # Volume moving averages
        for period in self.volume_periods:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()

        # Volume ratios
        df['volume_ratio_5_20'] = df['volume_sma_5'] / df['volume_sma_20']
        df['volume_ratio_current'] = df['volume'] / df['volume_sma_20']

        # Volume momentum
        df['volume_momentum'] = df['volume'].pct_change(5)

        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20).mean()
        df['obv_signal'] = df['obv'] - df['obv_ema']

        # Volume-Weighted Average Price (VWAP)
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['price_vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']

        # Money Flow Index
        df['mfi'] = self._calculate_mfi(df)

        # Volume spike detection
        df['volume_spike'] = (df['volume'] > df['volume_sma_20'] * 2).astype(int)

        # Accumulation/Distribution
        df['ad'] = self._calculate_accumulation_distribution(df)

        logger.debug(f"Added {len([col for col in df.columns if 'volume' in col or 'obv' in col or 'vwap' in col])} volume features")
        return df

    # ==========================================
    # VOLATILITY FEATURES
    # ==========================================

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced volatility features"""

        # ATR (Average True Range) - multiple periods
        for period in self.volatility_periods:
            df[f'atr_{period}'] = self._calculate_atr(df['high'].values, df['low'].values, df['close'].values, period)

        # ATR percentile (normalized volatility)
        df['atr_percentile'] = df['atr_14'].rolling(100).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) > 1 else 50
        )

        # Bollinger Bands - multiple periods
        for period in [10, 20, 50]:
            bb = self._calculate_bollinger_bands(df, period)
            df[f'bb_upper_{period}'] = bb['upper']
            df[f'bb_lower_{period}'] = bb['lower']
            df[f'bb_width_{period}'] = (bb['upper'] - bb['lower']) / bb['middle']
            df[f'bb_position_{period}'] = (df['close'] - bb['lower']) / (bb['upper'] - bb['lower'])

        # Keltner Channels
        kc = self._calculate_keltner_channels(df)
        df['kc_upper'] = kc['upper']
        df['kc_lower'] = kc['lower']
        df['kc_position'] = (df['close'] - kc['lower']) / (kc['upper'] - kc['lower'])

        # Historical volatility (realized volatility)
        for period in self.trend_periods:
            df[f'realized_vol_{period}'] = df['close'].pct_change().rolling(period).std() * np.sqrt(252)

        # Volatility ratio (short-term vs long-term)
        df['vol_ratio'] = df['realized_vol_10'] / df['realized_vol_50']

        # Parkinson volatility (uses high-low range)
        df['parkinson_vol'] = self._calculate_parkinson_volatility(df['high'].values, df['low'].values)

        # Garman-Klass volatility (more efficient estimator)
        df['garman_klass_vol'] = self._calculate_garman_klass_volatility(df['open'].values, df['high'].values, df['low'].values, df['close'].values)

        logger.debug(f"Added {len([col for col in df.columns if 'atr_' in col or 'bb_' in col or 'vol' in col])} volatility features")
        return df

    # ==========================================
    # MOMENTUM FEATURES
    # ==========================================

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced momentum indicators"""

        # RSI (Relative Strength Index) - multiple periods
        for period in self.momentum_periods:
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'].values, period)

        # RSI divergence detection
        df['rsi_divergence'] = self._detect_rsi_divergence(df)

        # Stochastic Oscillator
        for period in [14, 21]:
            stoch = self._calculate_stochastic(df, period)
            df[f'stoch_k_{period}'] = stoch['k']
            df[f'stoch_d_{period}'] = stoch['d']

        # Williams %R
        for period in [14, 28]:
            df[f'williams_r_{period}'] = self._calculate_williams_r(df, period)

        # CCI (Commodity Channel Index)
        df['cci'] = self._calculate_cci(df)

        # Ultimate Oscillator
        df['ultimate_oscillator'] = self._calculate_ultimate_oscillator(df)

        # MACD variations
        macd_configs = [(8, 21, 5), (12, 26, 9), (16, 30, 13)]
        for fast, slow, signal in macd_configs:
            macd = self._calculate_macd(df, fast, slow, signal)
            df[f'macd_{fast}_{slow}'] = macd['macd']
            df[f'macd_signal_{fast}_{slow}'] = macd['signal']
            df[f'macd_hist_{fast}_{slow}'] = macd['histogram']

        # Rate of Change (ROC) with smoothing
        df['roc_smooth'] = df['close'].pct_change(10).rolling(5).mean()

        # Momentum (price change over period)
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)

        logger.debug(f"Added {len([col for col in df.columns if 'rsi_' in col or 'macd_' in col or 'stoch_' in col])} momentum features")
        return df

    # ==========================================
    # TREND FEATURES
    # ==========================================

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced trend identification features"""

        # ADX (Average Directional Index) - trend strength
        for period in [14, 21]:
            df[f'adx_{period}'] = self._calculate_adx(df, period)

        # +DI and -DI (Directional Indicators)
        di = self._calculate_directional_indicators(df)
        df['di_plus'] = di['plus']
        df['di_minus'] = di['minus']
        df['di_diff'] = di['plus'] - di['minus']

        # Aroon Indicator
        aroon = self._calculate_aroon(df)
        df['aroon_up'] = aroon['up']
        df['aroon_down'] = aroon['down']
        df['aroon_oscillator'] = aroon['up'] - aroon['down']

        # Parabolic SAR
        df['psar'] = self._calculate_parabolic_sar(df)
        df['psar_signal'] = (df['close'] > df['psar']).astype(int)

        # Linear regression slope (trend direction)
        for period in self.trend_periods:
            df[f'lr_slope_{period}'] = df['close'].rolling(period).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == period else np.nan
            )

        # R-squared of linear regression (trend strength)
        df['lr_r2'] = df['close'].rolling(20).apply(
            lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] ** 2 if len(x) == 20 else np.nan
        )

        # Supertrend indicator
        df['supertrend'] = self._calculate_supertrend(df)
        df['supertrend_signal'] = (df['close'] > df['supertrend']).astype(int)

        # Trend quality (consistency of trend direction)
        df['trend_quality'] = df['close'].pct_change().rolling(20).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) == 20 else 0.5
        )

        logger.debug(f"Added {len([col for col in df.columns if 'adx_' in col or 'aroon_' in col or 'trend_' in col])} trend features")
        return df

    # ==========================================
    # PATTERN RECOGNITION FEATURES
    # ==========================================

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Candlestick pattern recognition"""

        # Use TA-Lib for pattern recognition if available
        if TALIB_AVAILABLE:
            try:
                # Doji patterns
                df['pattern_doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])

                # Engulfing patterns
                df['pattern_engulfing_bull'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])

                # Hammer patterns
                df['pattern_hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])

                # Morning/Evening star
                df['pattern_morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
                df['pattern_evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])

                # Three white soldiers / Three black crows
                df['pattern_three_white_soldiers'] = talib.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
                df['pattern_three_black_crows'] = talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])

            except Exception as e:
                logger.warning(f"TA-Lib pattern recognition failed: {e}")

        # Custom pattern features
        df['candle_body'] = abs(df['close'] - df['open'])
        df['candle_body_pct'] = df['candle_body'] / df['close']
        df['candle_direction'] = np.sign(df['close'] - df['open'])

        # Upper and lower shadows
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['shadow_ratio'] = df['upper_shadow'] / (df['lower_shadow'] + 1e-10)

        # Consecutive candle patterns
        df['consecutive_green'] = (df['close'] > df['open']).rolling(5).sum()
        df['consecutive_red'] = (df['close'] < df['open']).rolling(5).sum()

        logger.debug(f"Added {len([col for col in df.columns if 'pattern_' in col or 'candle_' in col])} pattern features")
        return df

    # ==========================================
    # MICROSTRUCTURE FEATURES
    # ==========================================

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure features (placeholders if no orderbook data)"""

        # Bid-ask spread (placeholder - would be populated from orderbook)
        df['spread'] = 0.001  # Placeholder
        df['spread_pct'] = df['spread'] / df['close']

        # Order book imbalance (placeholder)
        df['orderbook_imbalance'] = 0.0  # Would be calculated from orderbook levels

        # Trade classification (placeholder)
        df['buy_volume'] = df['volume'] * 0.5  # Placeholder
        df['sell_volume'] = df['volume'] * 0.5  # Placeholder
        df['volume_imbalance'] = (df['buy_volume'] - df['sell_volume']) / df['volume']

        # Price impact estimation
        df['price_impact'] = (df['high'] - df['low']) / df['volume']

        logger.debug("Added microstructure features (placeholders)")
        return df

    # ==========================================
    # TIME-BASED FEATURES
    # ==========================================

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features"""

        # Hour of day (cyclical encoding)
        if df.index.dtype == 'datetime64[ns]' or hasattr(df.index, 'hour'):
            df['hour'] = df.index.hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

            # Day of week (cyclical encoding)
            df['day_of_week'] = df.index.dayofweek
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

            # Is weekend
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

            # Time since market open/close (for stocks)
            # For crypto (24/7), use funding time proximity
            df['funding_hour'] = ((df['hour'] + 8) % 24)  # Funding at 0, 8, 16 UTC
            df['time_to_funding'] = df['funding_hour'].apply(
                lambda x: min(x % 8, 8 - (x % 8))
            )
        else:
            # Fallback if no datetime index
            df['hour'] = 12
            df['hour_sin'] = 0.0
            df['hour_cos'] = 1.0
            df['day_of_week'] = 0
            df['day_sin'] = 0.0
            df['day_cos'] = 1.0
            df['is_weekend'] = 0
            df['time_to_funding'] = 4

        logger.debug(f"Added {len([col for col in df.columns if 'hour' in col or 'day' in col or 'funding' in col])} time features")
        return df

    # ==========================================
    # STATISTICAL FEATURES
    # ==========================================

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical features"""

        # Z-score (standardized returns)
        for period in [20, 50]:
            returns = df['close'].pct_change()
            mean = returns.rolling(period).mean()
            std = returns.rolling(period).std()
            df[f'zscore_{period}'] = (returns - mean) / std

        # Skewness of returns
        df['returns_skew'] = df['close'].pct_change().rolling(20).skew()

        # Kurtosis of returns (tail risk)
        df['returns_kurtosis'] = df['close'].pct_change().rolling(20).kurt()

        # Autocorrelation (mean reversion indicator)
        df['returns_autocorr'] = df['close'].pct_change().rolling(20).apply(
            lambda x: x.autocorr() if len(x.dropna()) > 1 else 0.0
        )

        # Hurst exponent (trend persistence) - disabled due to Numba polyfit limitation
        # df['hurst_exponent'] = df['close'].rolling(100).apply(
        #     lambda x: _calculate_hurst_exponent_global(x.values) if len(x.dropna()) == 100 else np.nan
        # )
        df['hurst_exponent'] = 0.5  # Default random walk value

        # Entropy (randomness measure)
        df['entropy'] = df['close'].pct_change().rolling(20).apply(
            lambda x: stats.entropy(np.histogram(x.dropna(), bins=10)[0] + 1) if len(x.dropna()) > 1 else 0.0
        )

        logger.debug(f"Added {len([col for col in df.columns if 'zscore' in col or 'skew' in col or 'hurst' in col])} statistical features")
        return df

    # ==========================================
    # INTERACTION FEATURES
    # ==========================================

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature interactions and ratios"""

        # Volume-Price interaction
        df['volume_price_corr'] = df['volume'].rolling(20).corr(df['close'])

        # Volatility-Volume interaction
        df['vol_volume_ratio'] = df['atr_14'] * df['volume_ratio_current']

        # Momentum-Volume confirmation
        df['momentum_volume_conf'] = df['rsi_14'] * df['volume_ratio_current']

        # Trend-Volatility interaction
        df['trend_vol_product'] = df['adx_14'] * df['atr_percentile']

        # Multi-timeframe alignment (only use EMAs that exist)
        tf_alignment = 0
        if 'ema_8' in df.columns and 'ema_21' in df.columns:
            tf_alignment += (df['ema_8'] > df['ema_21']).astype(int)
        if 'ema_21' in df.columns and 'ema_50' in df.columns:
            tf_alignment += (df['ema_21'] > df['ema_50']).astype(int)
        if 'ema_50' in df.columns and 'ema_200' in df.columns:
            tf_alignment += (df['ema_50'] > df['ema_200']).astype(int)
        df['tf_alignment'] = tf_alignment

        logger.debug(f"Added {len([col for col in df.columns if 'corr' in col or 'ratio' in col or 'product' in col])} interaction features")
        return df

    # ==========================================
    # HELPER CALCULATION METHODS
    # ==========================================

    @staticmethod
    @jit(nopython=True)
    def _calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Numba-optimized RSI calculation"""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)

        rsi_values = np.full(len(prices), np.nan)
        gains = np.zeros(len(prices))
        losses = np.zeros(len(prices))

        # Calculate price changes
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            gains[i] = max(change, 0)
            losses[i] = max(-change, 0)

        # Calculate initial averages
        avg_gain = np.mean(gains[1:period+1])
        avg_loss = np.mean(losses[1:period+1])

        for i in range(period, len(prices)):
            if avg_loss == 0:
                rsi_values[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_values[i] = 100 - (100 / (1 + rs))

            # Update averages for next iteration
            if i < len(prices) - 1:
                avg_gain = (avg_gain * (period - 1) + gains[i+1]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i+1]) / period

        return rsi_values

    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD with signal line and histogram"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line

        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }

    @staticmethod
    @jit(nopython=True)
    def _calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Numba-optimized ATR calculation"""
        if len(high) < period + 1:
            return np.full(len(high), np.nan)

        atr_values = np.full(len(high), np.nan)
        tr_values = np.zeros(len(high))

        # Calculate True Range
        tr_values[1:] = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        # Calculate ATR
        atr_values[period] = np.mean(tr_values[1:period+1])

        for i in range(period + 1, len(high)):
            atr_values[i] = (atr_values[i-1] * (period - 1) + tr_values[i]) / period

        return atr_values

    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = df['close'].rolling(period).mean()
        std_dev = df['close'].rolling(period).std()

        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }

    def _calculate_keltner_channels(self, df: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Keltner Channels"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        middle = typical_price.rolling(period).mean()
        atr = self._calculate_atr(df['high'].values, df['low'].values, df['close'].values, period)

        upper = middle + (atr * multiplier)
        lower = middle - (atr * multiplier)

        return {
            'upper': pd.Series(upper, index=df.index),
            'middle': middle,
            'lower': pd.Series(lower, index=df.index)
        }

    @staticmethod
    @jit(nopython=True)
    def _calculate_parkinson_volatility(high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """Calculate Parkinson volatility estimator"""
        if len(high) != len(low):
            return np.full(len(high), np.nan)

        log_hl_ratio = np.log(high / low)
        parkinson_vol = np.sqrt((1 / (4 * np.log(2))) * np.square(log_hl_ratio))

        return parkinson_vol

    @staticmethod
    @jit(nopython=True)
    def _calculate_garman_klass_volatility(open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate Garman-Klass volatility estimator"""
        if len(open) != len(high) or len(high) != len(low) or len(low) != len(close):
            return np.full(len(open), np.nan)

        log_ho = np.log(high / open)
        log_lo = np.log(low / open)
        log_co = np.log(close / open)

        garman_klass_vol = np.sqrt(
            0.5 * np.square(log_ho - log_lo) -
            (2 * np.log(2) - 1) * np.square(log_co)
        )

        return garman_klass_vol

    def _detect_rsi_divergence(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Detect RSI divergence patterns"""
        rsi = df[f'rsi_{period}']
        price = df['close']

        # Bullish divergence: price makes lower low, RSI makes higher low
        price_ll = (price < price.shift(5)) & (price.shift(5) < price.shift(10))
        rsi_hl = (rsi > rsi.shift(5)) & (rsi.shift(5) > rsi.shift(10))
        bullish_div = (price_ll & rsi_hl).astype(int)

        # Bearish divergence: price makes higher high, RSI makes lower high
        price_hh = (price > price.shift(5)) & (price.shift(5) > price.shift(10))
        rsi_lh = (rsi < rsi.shift(5)) & (rsi.shift(5) < rsi.shift(10))
        bearish_div = (price_hh & rsi_lh).astype(int) * -1

        return bullish_div + bearish_div

    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = df['low'].rolling(period).min()
        highest_high = df['high'].rolling(period).max()

        k = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(3).mean()

        return {'k': k, 'd': d}

    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = df['high'].rolling(period).max()
        lowest_low = df['low'].rolling(period).min()

        williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))

        return williams_r

    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))

        cci = (typical_price - sma) / (0.015 * mad)

        return cci

    def _calculate_ultimate_oscillator(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Ultimate Oscillator"""
        # Simplified version - full implementation would be more complex
        bp = df['close'] - np.minimum(df['low'], df['close'].shift(1))
        tr = np.maximum(df['high'] - df['low'],
                       np.maximum(np.abs(df['high'] - df['close'].shift(1)),
                                np.abs(df['low'] - df['close'].shift(1))))

        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()

        uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7

        return uo

    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(period).sum()
        negative_mf = negative_flow.rolling(period).sum()

        mfi = 100 - (100 / (1 + positive_mf / negative_mf))

        return mfi

    def _calculate_accumulation_distribution(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0)  # Handle division by zero
        ad = clv * df['volume']
        ad = ad.cumsum()

        return ad

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()

        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        atr = pd.Series(self._calculate_atr(df['high'].values, df['low'].values, df['close'].values, period), index=df.index)

        pos_di = 100 * (pos_dm.rolling(period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(period).mean() / atr)

        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(period).mean()

        return adx

    def _calculate_directional_indicators(self, df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """Calculate Directional Indicators (+DI and -DI)"""
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()

        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        atr = pd.Series(self._calculate_atr(df['high'].values, df['low'].values, df['close'].values, period), index=df.index)

        pos_di = 100 * (pos_dm.rolling(period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(period).mean() / atr)

        return {'plus': pos_di, 'minus': neg_di}

    def _calculate_aroon(self, df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """Calculate Aroon Indicator"""
        aroon_up = 100 * (period - df['high'].rolling(period).apply(lambda x: period - np.argmax(x))) / period
        aroon_down = 100 * (period - df['low'].rolling(period).apply(lambda x: period - np.argmin(x))) / period

        return {'up': aroon_up, 'down': aroon_down}

    def _calculate_parabolic_sar(self, df: pd.DataFrame, af: float = 0.02, max_af: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR"""
        psar = df['close'].copy()
        psar_bull = True
        af_value = af
        ep = df['low'].iloc[0] if psar_bull else df['high'].iloc[0]

        for i in range(2, len(df)):
            psar.iloc[i] = psar.iloc[i-1] + af_value * (ep - psar.iloc[i-1])

            if psar_bull:
                if df['low'].iloc[i] <= psar.iloc[i]:
                    psar_bull = False
                    psar.iloc[i] = ep
                    ep = df['high'].iloc[i]
                    af_value = af
                else:
                    if df['high'].iloc[i] > ep:
                        ep = df['high'].iloc[i]
                        af_value = min(af_value + af, max_af)
            else:
                if df['high'].iloc[i] >= psar.iloc[i]:
                    psar_bull = True
                    psar.iloc[i] = ep
                    ep = df['low'].iloc[i]
                    af_value = af
                else:
                    if df['low'].iloc[i] < ep:
                        ep = df['low'].iloc[i]
                        af_value = min(af_value + af, max_af)

        return psar

    def _calculate_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3) -> pd.Series:
        """Calculate Supertrend indicator"""
        hl2 = (df['high'] + df['low']) / 2
        atr = pd.Series(self._calculate_atr(df['high'].values, df['low'].values, df['close'].values, period), index=df.index)

        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        supertrend = pd.Series(index=df.index, dtype=float)

        for i in range(len(df)):
            if i == 0:
                supertrend.iloc[i] = upper_band.iloc[i]
            else:
                if df['close'].iloc[i-1] <= supertrend.iloc[i-1]:
                    # Bearish trend
                    supertrend.iloc[i] = max(upper_band.iloc[i], supertrend.iloc[i-1])
                else:
                    # Bullish trend
                    supertrend.iloc[i] = min(lower_band.iloc[i], supertrend.iloc[i-1])

        return supertrend

