"""
Feature Engineering for ML-Enhanced Trading Strategy

This module provides comprehensive feature engineering for financial time series prediction,
creating 80+ features from technical indicators, market microstructure, and temporal patterns.

Key Features:
- Technical Indicators: Trend, momentum, volatility, volume-based features
- Market Microstructure: Order book imbalance, trade flow, liquidity metrics
- Temporal Features: Time of day, market regime, volatility measures
- Derived Features: Ratios, classifications, and normalized metrics
- Real-time computation for live trading
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from input_sanitizer import InputSanitizer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for ML-based trading signals.

    This class creates features from multiple domains:
    - Technical analysis (50+ indicators)
    - Market microstructure (20+ metrics)
    - Temporal patterns (10+ features)
    - Derived relationships and ratios

    All features are designed for real-time computation and ML model training.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the feature engineer.

        Args:
            config: Configuration dictionary with feature parameters
        """
        self.config = config or self._get_default_config()

        # Feature calculation parameters
        self.ema_periods = self.config['technical']['ema_periods']
        self.rsi_periods = self.config['technical']['rsi_periods']
        self.bb_period = self.config['technical']['bollinger_period']
        self.bb_std = self.config['technical']['bollinger_std']
        self.atr_period = self.config['technical']['atr_period']
        self.adx_period = self.config['technical']['adx_period']

        # Initialize state for incremental calculations
        self._ema_cache = {}
        self._rsi_cache = {}
        self._atr_cache = {}
        self._adx_cache = {}

        logger.info("FeatureEngineer initialized")

    def _get_default_config(self) -> Dict:
        """Get default configuration parameters."""
        return {
            'technical': {
                'ema_periods': [9, 21, 50, 200],
                'rsi_periods': [14, 21],
                'bollinger_period': 20,
                'bollinger_std': 2.0,
                'atr_period': 14,
                'adx_period': 14,
                'stoch_period': 14,
                'williams_period': 14,
                'aroon_period': 14,
                'obv_period': 20,
                'volume_sma_periods': [10, 20, 50]
            },
            'microstructure': {
                'orderbook_depths': [5, 10, 20],
                'trade_flow_windows': [60, 300, 900],  # 1m, 5m, 15m in seconds
                'large_trade_multiplier': 2.0,
                'spread_threshold_bps': 50
            },
            'temporal': {
                'volatility_windows': [60, 240, 1440],  # 1h, 4h, 24h in minutes
                'funding_rate_periods': [1, 8]  # Current and 8h average
            },
            'derived': {
                'momentum_ratios': [(5, 15), (15, 60), (60, 240)],  # Minutes pairs
                'volatility_regime_thresholds': [0.01, 0.03],  # Low/high thresholds
                'trend_strength_thresholds': [20, 40]  # ADX thresholds
            }
        }

    # ==========================================
    # TECHNICAL INDICATOR FEATURES (50+ features)
    # ==========================================

    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicator features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with technical features added
        """
        try:
            # Make a copy to avoid modifying original
            features_df = df.copy()

            # Trend indicators
            features_df = self._add_trend_features(features_df)

            # Momentum indicators
            features_df = self._add_momentum_features(features_df)

            # Volatility indicators
            features_df = self._add_volatility_features(features_df)

            # Volume indicators
            features_df = self._add_volume_features(features_df)

            # Price pattern features
            features_df = self._add_price_pattern_features(features_df)

            logger.debug(f"Calculated {len(features_df.columns) - len(df.columns)} technical features")
            return features_df

        except Exception as e:
            logger.error(f"Error calculating technical features: {e}")
            return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based technical features."""
        try:
            # EMA features
            for period in self.ema_periods:
                df[f'ema_{period}'] = self._calculate_ema(df['close'], period)
                # EMA slope (rate of change)
                df[f'ema_{period}_slope'] = df[f'ema_{period}'].pct_change(5)
                # Distance from EMA
                df[f'price_vs_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']

            # MACD features (multiple periods)
            macd_periods = [(12, 26, 9), (8, 21, 8), (19, 39, 9)]
            for fast, slow, signal in macd_periods:
                macd_line, signal_line, histogram = self._calculate_macd(df['close'], fast, slow, signal)
                df[f'macd_{fast}_{slow}_{signal}'] = macd_line
                df[f'macd_signal_{fast}_{slow}_{signal}'] = signal_line
                df[f'macd_hist_{fast}_{slow}_{signal}'] = histogram
                # MACD momentum
                df[f'macd_hist_momentum_{fast}_{slow}_{signal}'] = histogram.pct_change(3)

            # ADX (Average Directional Index)
            adx, plus_di, minus_di = self._calculate_adx(df)
            df['adx'] = adx
            df['plus_di'] = plus_di
            df['minus_di'] = minus_di
            df['adx_trend_strength'] = adx / 50.0  # Normalize to 0-2 range

            # Aroon Oscillator
            aroon_up, aroon_down = self._calculate_aroon(df, self.adx_period)
            df['aroon_up'] = aroon_up
            df['aroon_down'] = aroon_down
            df['aroon_oscillator'] = aroon_up - aroon_down

            return df

        except Exception as e:
            logger.debug(f"Error calculating trend features: {e}")
            return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based technical features."""
        try:
            # RSI features
            for period in self.rsi_periods:
                df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
                # RSI divergence (simplified)
                df[f'rsi_divergence_{period}'] = df[f'rsi_{period}'].pct_change(10)

            # Stochastic Oscillator
            stoch_k, stoch_d = self._calculate_stochastic(df)
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            df['stoch_diff'] = stoch_k - stoch_d

            # Williams %R
            df['williams_r'] = self._calculate_williams_r(df)

            # Rate of Change (ROC)
            roc_periods = [10, 20, 30]
            for period in roc_periods:
                df[f'roc_{period}'] = df['close'].pct_change(period)
                # ROC momentum
                df[f'roc_momentum_{period}'] = df[f'roc_{period}'].pct_change(5)

            # Price momentum (various timeframes)
            momentum_periods = [3, 5, 10, 20]
            for period in momentum_periods:
                df[f'momentum_{period}'] = df['close'].pct_change(period)
                # Momentum acceleration
                df[f'momentum_accel_{period}'] = df[f'momentum_{period}'].pct_change(3)

            return df

        except Exception as e:
            logger.debug(f"Error calculating momentum features: {e}")
            return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based technical features."""
        try:
            # ATR (Average True Range)
            df['atr'] = self._calculate_atr(df)
            # Normalized ATR (as % of price)
            df['atr_normalized'] = df['atr'] / df['close']

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

            # Bollinger Band squeeze (narrow bands indicate low volatility)
            df['bb_squeeze'] = df['bb_width'].rolling(20).mean()

            # Keltner Channels
            kc_upper, kc_middle, kc_lower = self._calculate_keltner_channels(df)
            df['kc_upper'] = kc_upper
            df['kc_middle'] = kc_middle
            df['kc_lower'] = kc_lower
            df['kc_position'] = (df['close'] - kc_lower) / (kc_upper - kc_lower)

            # Historical Volatility
            hv_periods = [10, 20, 30]
            for period in hv_periods:
                df[f'hv_{period}'] = self._calculate_historical_volatility(df, period)

            # Price range features
            df['daily_range'] = (df['high'] - df['low']) / df['close']
            df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
            df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']

            return df

        except Exception as e:
            logger.debug(f"Error calculating volatility features: {e}")
            return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based technical features."""
        try:
            # On-Balance Volume (OBV)
            df['obv'] = self._calculate_obv(df)
            # OBV trend
            df['obv_trend'] = df['obv'].pct_change(10)

            # Volume Rate of Change
            volume_roc_periods = [5, 10, 20]
            for period in volume_roc_periods:
                df[f'volume_roc_{period}'] = df['volume'].pct_change(period)

            # Volume SMA ratios
            for period in self.config['technical']['volume_sma_periods']:
                df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
                df[f'volume_vs_sma_{period}'] = df['volume'] / df[f'volume_sma_{period}']

            # VWAP (Volume Weighted Average Price)
            df['vwap'] = self._calculate_vwap(df)
            df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']

            # Volume-Price Trend (VPT)
            df['vpt'] = self._calculate_vpt(df)

            # Chaikin Money Flow (CMF)
            df['cmf'] = self._calculate_chaikin_money_flow(df)

            # Volume Oscillator
            df['volume_oscillator'] = df['volume'].rolling(5).mean() / df['volume'].rolling(10).mean()

            return df

        except Exception as e:
            logger.debug(f"Error calculating volume features: {e}")
            return df

    def _add_price_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern recognition features."""
        try:
            # Higher highs and higher lows (trend strength)
            df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
            df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
            df['lower_high'] = (df['high'] < df['high'].shift(1)).astype(int)
            df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)

            # Pivot points
            df['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
            df['resistance_1'] = 2 * df['pivot_point'] - df['low']
            df['support_1'] = 2 * df['pivot_point'] - df['high']

            # Distance to pivot levels
            df['distance_to_resistance'] = (df['resistance_1'] - df['close']) / df['close']
            df['distance_to_support'] = (df['close'] - df['support_1']) / df['close']

            # Candlestick patterns (simplified)
            df['bullish_engulfing'] = (
                (df['close'] > df['open']) &
                (df['open'].shift(1) > df['close'].shift(1)) &
                (df['close'] > df['open'].shift(1)) &
                (df['open'] < df['close'].shift(1))
            ).astype(int)

            df['bearish_engulfing'] = (
                (df['close'] < df['open']) &
                (df['open'].shift(1) < df['close'].shift(1)) &
                (df['close'] < df['open'].shift(1)) &
                (df['open'] > df['close'].shift(1))
            ).astype(int)

            # Doji pattern
            df['doji'] = ((abs(df['close'] - df['open']) / (df['high'] - df['low'])) < 0.1).astype(int)

            # Price acceleration
            df['price_acceleration'] = df['close'].pct_change().pct_change()

            return df

        except Exception as e:
            logger.debug(f"Error calculating price pattern features: {e}")
            return df

    # ==========================================
    # MARKET MICROSTRUCTURE FEATURES (20+ features)
    # ==========================================

    def calculate_microstructure_features(self, df: pd.DataFrame,
                                        orderbook_data: Optional[Dict] = None,
                                        trade_flow_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate market microstructure features from order book and trade flow data.

        Args:
            df: DataFrame with OHLCV data
            orderbook_data: Real-time order book data
            trade_flow_data: Recent trade flow metrics

        Returns:
            DataFrame with microstructure features added
        """
        try:
            features_df = df.copy()

            # Order book imbalance features
            if orderbook_data:
                features_df = self._add_orderbook_features(features_df, orderbook_data)

            # Trade flow features
            if trade_flow_data:
                features_df = self._add_trade_flow_features(features_df, trade_flow_data)

            # Liquidity and spread features
            features_df = self._add_liquidity_features(features_df, orderbook_data)

            # Spoofing and manipulation detection
            features_df = self._add_manipulation_features(features_df, orderbook_data)

            logger.debug(f"Calculated microstructure features")
            return features_df

        except Exception as e:
            logger.error(f"Error calculating microstructure features: {e}")
            return df

    def _add_orderbook_features(self, df: pd.DataFrame, orderbook_data: Dict) -> pd.DataFrame:
        """Add order book imbalance and depth features."""
        try:
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            spread_bps = orderbook_data.get('spread_bps', 0)

            # Order book imbalance at different depths
            for depth in self.config['microstructure']['orderbook_depths']:
                if len(bids) >= depth and len(asks) >= depth:
                    bid_volume = sum(qty for _, qty in bids[:depth])
                    ask_volume = sum(qty for _, qty in asks[:depth])
                    total_volume = bid_volume + ask_volume

                    if total_volume > 0:
                        imbalance = (bid_volume - ask_volume) / total_volume
                        df[f'ob_imbalance_{depth}'] = imbalance
                        df[f'bid_volume_{depth}'] = bid_volume
                        df[f'ask_volume_{depth}'] = ask_volume

            # Spread features
            df['spread_bps'] = spread_bps
            df['spread_normalized'] = spread_bps / 100.0  # Normalize for ML

            # Order book depth changes (if historical data available)
            if len(df) > 1:
                df['bid_depth_change'] = df.get('bid_volume_10', 0) - df.get('bid_volume_10', 0).shift(1)
                df['ask_depth_change'] = df.get('ask_volume_10', 0) - df.get('ask_volume_10', 0).shift(1)

            # Order book slope (liquidity concentration)
            if len(bids) >= 5:
                bid_prices = [price for price, _ in bids[:5]]
                bid_volumes = [qty for _, qty in bids[:5]]
                if len(bid_prices) > 1:
                    df['bid_slope'] = np.polyfit(range(len(bid_prices)), bid_prices, 1)[0]

            if len(asks) >= 5:
                ask_prices = [price for price, _ in asks[:5]]
                ask_volumes = [qty for _, qty in asks[:5]]
                if len(ask_prices) > 1:
                    df['ask_slope'] = np.polyfit(range(len(ask_prices)), ask_prices, 1)[0]

            return df

        except Exception as e:
            logger.debug(f"Error calculating orderbook features: {e}")
            return df

    def _add_trade_flow_features(self, df: pd.DataFrame, trade_flow_data: Dict) -> pd.DataFrame:
        """Add trade flow analysis features."""
        try:
            # Net trade flow at different timeframes
            for window in self.config['microstructure']['trade_flow_windows']:
                window_name = f"{window//60}m" if window >= 60 else f"{window}s"
                net_flow = trade_flow_data.get(f'net_flow_{window_name}', 0)
                df[f'net_trade_flow_{window_name}'] = net_flow

                # Trade flow momentum
                flow_momentum = trade_flow_data.get(f'flow_momentum_{window_name}', 0)
                df[f'trade_flow_momentum_{window_name}'] = flow_momentum

            # Aggression ratio (buy vs sell pressure)
            aggression_ratio = trade_flow_data.get('aggression_ratio', 1.0)
            df['aggression_ratio'] = aggression_ratio

            # Large trade frequency
            large_trades = trade_flow_data.get('large_trades_count', 0)
            df['large_trade_frequency'] = large_trades

            # Buy/sell pressure ratios
            buy_pressure = trade_flow_data.get('buy_pressure', 0.5)
            sell_pressure = trade_flow_data.get('sell_pressure', 0.5)
            df['buy_pressure'] = buy_pressure
            df['sell_pressure'] = sell_pressure
            df['pressure_imbalance'] = buy_pressure - sell_pressure

            return df

        except Exception as e:
            logger.debug(f"Error calculating trade flow features: {e}")
            return df

    def _add_liquidity_features(self, df: pd.DataFrame, orderbook_data: Optional[Dict]) -> pd.DataFrame:
        """Add liquidity and market depth features."""
        try:
            # Market depth ratios
            if orderbook_data:
                total_bid_volume = sum(qty for _, qty in orderbook_data.get('bids', [])[:20])
                total_ask_volume = sum(qty for _, qty in orderbook_data.get('asks', [])[:20])

                if total_bid_volume + total_ask_volume > 0:
                    df['market_depth_ratio'] = total_bid_volume / (total_bid_volume + total_ask_volume)

                # Liquidity score (inverse of spread, scaled by depth)
                spread_bps = orderbook_data.get('spread_bps', 100)
                liquidity_score = (total_bid_volume + total_ask_volume) / (spread_bps + 1)
                df['liquidity_score'] = liquidity_score

            # Volume to spread ratio (higher = better liquidity)
            if 'volume' in df.columns and 'spread_bps' in df.columns:
                df['volume_spread_ratio'] = df['volume'] / (df['spread_bps'] + 1)

            return df

        except Exception as e:
            logger.debug(f"Error calculating liquidity features: {e}")
            return df

    def _add_manipulation_features(self, df: pd.DataFrame, orderbook_data: Optional[Dict]) -> pd.DataFrame:
        """Add features to detect potential market manipulation."""
        try:
            # Spoofing detection (simplified)
            if orderbook_data and len(orderbook_data.get('bids', [])) > 0:
                # Check for unusually large orders far from mid price
                mid_price = orderbook_data.get('mid_price', 0)
                large_bid_distances = []

                for price, qty in orderbook_data['bids'][:10]:
                    distance_pct = abs(price - mid_price) / mid_price
                    if qty > 100 and distance_pct > 0.001:  # 0.1% away
                        large_bid_distances.append(distance_pct)

                df['potential_spoofing_bids'] = len(large_bid_distances)

            # Order book concentration (liquidity magnets)
            if orderbook_data:
                bids = orderbook_data.get('bids', [])
                if bids:
                    total_bid_volume = sum(qty for _, qty in bids[:20])
                    largest_bid_pct = max(qty for _, qty in bids[:5]) / total_bid_volume if total_bid_volume > 0 else 0
                    df['bid_concentration'] = largest_bid_pct

            return df

        except Exception as e:
            logger.debug(f"Error calculating manipulation features: {e}")
            return df

    # ==========================================
    # TEMPORAL FEATURES (10+ features)
    # ==========================================

    def calculate_temporal_features(self, df: pd.DataFrame,
                                  funding_rates: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Calculate time-based and market regime features.

        Args:
            df: DataFrame with OHLCV data and timestamps
            funding_rates: Recent funding rate data

        Returns:
            DataFrame with temporal features added
        """
        try:
            features_df = df.copy()

            # Extract time components
            if 'timestamp' in df.columns:
                timestamps = pd.to_datetime(df['timestamp'])

                # Hour of day (0-23)
                features_df['hour_of_day'] = timestamps.dt.hour

                # Day of week (0=Monday, 6=Sunday)
                features_df['day_of_week'] = timestamps.dt.dayofweek

                # One-hot encode day of week
                for day in range(7):
                    features_df[f'day_{day}'] = (features_df['day_of_week'] == day).astype(int)

                # Time since market open (assuming 24/7 crypto market)
                features_df['minutes_since_midnight'] = timestamps.dt.hour * 60 + timestamps.dt.minute

                # Market session (Asia, Europe, US)
                features_df['asia_session'] = ((features_df['hour_of_day'] >= 0) & (features_df['hour_of_day'] < 8)).astype(int)
                features_df['europe_session'] = ((features_df['hour_of_day'] >= 8) & (features_df['hour_of_day'] < 16)).astype(int)
                features_df['us_session'] = ((features_df['hour_of_day'] >= 16) & (features_df['hour_of_day'] < 24)).astype(int)

            # Volatility regime features
            features_df = self._add_volatility_regime_features(features_df)

            # Funding rate features
            if funding_rates:
                features_df = self._add_funding_rate_features(features_df, funding_rates)

            logger.debug(f"Calculated temporal features")
            return features_df

        except Exception as e:
            logger.error(f"Error calculating temporal features: {e}")
            return df

    def _add_volatility_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility regime classification features."""
        try:
            # Calculate volatility at different timeframes
            for window_minutes in self.config['temporal']['volatility_windows']:
                window_name = f"{window_minutes//60}h" if window_minutes >= 60 else f"{window_minutes}m"

                # Rolling volatility (standard deviation of returns)
                returns = df['close'].pct_change()
                volatility = returns.rolling(window=window_minutes).std()
                df[f'volatility_{window_name}'] = volatility

                # Volatility regime classification
                vol_thresholds = self.config['derived']['volatility_regime_thresholds']
                df[f'vol_regime_{window_name}'] = np.select(
                    [
                        volatility < vol_thresholds[0],
                        (volatility >= vol_thresholds[0]) & (volatility < vol_thresholds[1]),
                        volatility >= vol_thresholds[1]
                    ],
                    [0, 1, 2],  # 0=low, 1=medium, 2=high
                    default=1
                )

            # Volatility of volatility
            df['vol_of_vol'] = df.get('volatility_1h', pd.Series()).pct_change().rolling(60).std()

            return df

        except Exception as e:
            logger.debug(f"Error calculating volatility regime features: {e}")
            return df

    def _add_funding_rate_features(self, df: pd.DataFrame, funding_rates: List[Dict]) -> pd.DataFrame:
        """Add funding rate based features."""
        try:
            if not funding_rates:
                return df

            # Current funding rate
            current_funding = funding_rates[0].get('funding_rate', 0) if funding_rates else 0
            df['funding_rate_current'] = current_funding

            # Average funding rate over different periods
            for periods in self.config['temporal']['funding_rate_periods']:
                if len(funding_rates) >= periods:
                    avg_funding = np.mean([fr.get('funding_rate', 0) for fr in funding_rates[:periods]])
                    df[f'funding_rate_avg_{periods}h'] = avg_funding

            # Funding rate momentum
            if len(funding_rates) >= 2:
                funding_momentum = funding_rates[0].get('funding_rate', 0) - funding_rates[1].get('funding_rate', 0)
                df['funding_rate_momentum'] = funding_momentum

            return df

        except Exception as e:
            logger.debug(f"Error calculating funding rate features: {e}")
            return df

    # ==========================================
    # DERIVED FEATURES (ratios, classifications, etc.)
    # ==========================================

    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived features from existing technical and microstructure data.

        Args:
            df: DataFrame with base features already calculated

        Returns:
            DataFrame with derived features added
        """
        try:
            features_df = df.copy()

            # Momentum ratios
            features_df = self._add_momentum_ratios(features_df)

            # Trend strength classifications
            features_df = self._add_trend_classifications(features_df)

            # Support/resistance relationships
            features_df = self._add_support_resistance_features(features_df)

            # Composite indicators
            features_df = self._add_composite_indicators(features_df)

            logger.debug(f"Calculated derived features")
            return features_df

        except Exception as e:
            logger.error(f"Error calculating derived features: {e}")
            return df

    def _add_momentum_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum ratio features."""
        try:
            for short_period, long_period in self.config['derived']['momentum_ratios']:
                short_col = f'momentum_{short_period}'
                long_col = f'momentum_{long_period}'

                if short_col in df.columns and long_col in df.columns:
                    ratio_col = f'momentum_ratio_{short_period}m_{long_period}m'
                    df[ratio_col] = df[short_col] / (df[long_col] + 1e-8)  # Avoid division by zero

                    # Momentum divergence
                    df[f'momentum_divergence_{short_period}m_{long_period}m'] = (
                        df[short_col] - df[long_col]
                    )

            return df

        except Exception as e:
            logger.debug(f"Error calculating momentum ratios: {e}")
            return df

    def _add_trend_classifications(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend strength classifications based on ADX."""
        try:
            if 'adx' in df.columns:
                adx_thresholds = self.config['derived']['trend_strength_thresholds']

                # Trend strength classification
                df['trend_strength_class'] = np.select(
                    [
                        df['adx'] < adx_thresholds[0],
                        (df['adx'] >= adx_thresholds[0]) & (df['adx'] < adx_thresholds[1]),
                        df['adx'] >= adx_thresholds[1]
                    ],
                    [0, 1, 2],  # 0=weak, 1=moderate, 2=strong
                    default=0
                )

                # Trend direction from ADX components
                df['trend_direction'] = np.sign(df.get('plus_di', 0) - df.get('minus_di', 0))

            return df

        except Exception as e:
            logger.debug(f"Error calculating trend classifications: {e}")
            return df

    def _add_support_resistance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add support and resistance relationship features."""
        try:
            # Distance to key levels (normalized)
            if 'close' in df.columns:
                # Round number proximity (psychological levels)
                df['nearest_100'] = (df['close'] // 100) * 100
                df['distance_to_round_number'] = abs(df['close'] - df['nearest_100']) / df['close']

                # Fibonacci retracement levels (simplified)
                recent_high = df['high'].rolling(50).max()
                recent_low = df['low'].rolling(50).min()
                price_range = recent_high - recent_low

                for fib_level in [0.236, 0.382, 0.5, 0.618, 0.786]:
                    fib_price = recent_low + price_range * fib_level
                    distance = abs(df['close'] - fib_price) / df['close']
                    df[f'fib_distance_{int(fib_level*1000)}'] = distance

            return df

        except Exception as e:
            logger.debug(f"Error calculating support/resistance features: {e}")
            return df

    def _add_composite_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add composite indicators combining multiple signals."""
        try:
            # Trend-momentum composite
            if 'adx' in df.columns and 'rsi_14' in df.columns:
                # Trend-following strength
                df['trend_momentum_score'] = (df['adx'] / 50.0) * (df['rsi_14'] - 50) / 50.0

            # Volume-price composite
            if 'obv_trend' in df.columns and 'vwap_distance' in df.columns:
                df['volume_price_alignment'] = df['obv_trend'] * df['vwap_distance']

            # Volatility-adjusted momentum
            if 'momentum_10' in df.columns and 'atr_normalized' in df.columns:
                df['vol_adjusted_momentum'] = df['momentum_10'] / (df['atr_normalized'] + 1e-8)

            # Market regime indicator (combining trend and volatility)
            if 'trend_strength_class' in df.columns:
                vol_regime = df.get('vol_regime_1h', 1)
                df['market_regime'] = df['trend_strength_class'] * 3 + vol_regime  # 0-8 scale

            return df

        except Exception as e:
            logger.debug(f"Error calculating composite indicators: {e}")
            return df

    # ==========================================
    # INDICATOR CALCULATION HELPERS
    # ==========================================

    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Average Directional Index."""
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = high - high.shift(1)
        minus_dm = low.shift(1) - low

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Smoothed averages
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx, plus_di, minus_di

    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = df['close'].rolling(window=period).mean()
        std_dev = df['close'].rolling(window=period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, sma, lower

    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = df['low'].rolling(window=period).min()
        highest_high = df['high'].rolling(window=period).max()

        k = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=3).mean()

        return k, d

    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()

        return -100 * ((highest_high - df['close']) / (highest_high - lowest_low))

    def _calculate_aroon(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate Aroon Oscillator."""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()

        aroon_up = 100 * (period - (period - df['high'].rolling(window=period).argmax())) / period
        aroon_down = 100 * (period - (period - df['low'].rolling(window=period).argmin())) / period

        return aroon_up, aroon_down

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['volume'].iloc[0]

        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return obv

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap

    def _calculate_vpt(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume-Price Trend."""
        price_change = df['close'].pct_change()
        vpt = (price_change * df['volume']).cumsum()
        return vpt

    def _calculate_chaikin_money_flow(self, df: pd.DataFrame, period: int = 21) -> pd.Series:
        """Calculate Chaikin Money Flow."""
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        money_flow_volume = money_flow_multiplier * df['volume']
        cmf = money_flow_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        return cmf

    def _calculate_keltner_channels(self, df: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        middle = typical_price.rolling(window=period).mean()
        atr = self._calculate_atr(df, period)
        upper = middle + (atr * multiplier)
        lower = middle - (atr * multiplier)
        return upper, middle, lower

    def _calculate_historical_volatility(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Historical Volatility."""
        returns = df['close'].pct_change()
        return returns.rolling(window=period).std() * np.sqrt(252)  # Annualized

    # ==========================================
    # MAIN FEATURE ENGINEERING PIPELINE
    # ==========================================

    def engineer_features(self, market_data: pd.DataFrame,
                         orderbook_data: Optional[Dict] = None,
                         trade_flow_data: Optional[Dict] = None,
                         funding_rates: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Main feature engineering pipeline.

        Args:
            market_data: OHLCV DataFrame
            orderbook_data: Real-time order book data
            trade_flow_data: Trade flow metrics
            funding_rates: Funding rate data

        Returns:
            DataFrame with all engineered features
        """
        try:
            logger.info("Starting comprehensive feature engineering...")

            # Start with market data
            features_df = market_data.copy()

            # Calculate feature categories
            features_df = self.calculate_technical_features(features_df)
            features_df = self.calculate_microstructure_features(
                features_df, orderbook_data, trade_flow_data
            )
            features_df = self.calculate_temporal_features(features_df, funding_rates)
            features_df = self.calculate_derived_features(features_df)

            # Remove NaN values and infinite values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

            # Ensure all features are numeric
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            features_df = features_df[numeric_cols]

            logger.info(f"Feature engineering complete: {len(features_df.columns)} features created")

            return features_df

        except Exception as e:
            logger.error(f"Error in feature engineering pipeline: {e}")
            # Return basic features on error
            return market_data.select_dtypes(include=[np.number]).fillna(0)

    def get_feature_importance_template(self) -> Dict[str, str]:
        """
        Get template for feature importance analysis.

        Returns:
            Dictionary mapping feature names to categories
        """
        return {
            # Technical indicators
            'ema_9': 'trend', 'ema_21': 'trend', 'ema_50': 'trend', 'ema_200': 'trend',
            'macd_12_26_9': 'trend', 'rsi_14': 'momentum', 'stoch_k': 'momentum',
            'atr': 'volatility', 'bb_position': 'volatility', 'obv': 'volume',
            'vwap_distance': 'volume',

            # Microstructure features
            'ob_imbalance_10': 'microstructure', 'spread_bps': 'microstructure',
            'net_trade_flow_5m': 'microstructure', 'aggression_ratio': 'microstructure',

            # Temporal features
            'hour_of_day': 'temporal', 'day_of_week': 'temporal',
            'volatility_1h': 'temporal', 'funding_rate_current': 'temporal',

            # Derived features
            'trend_strength_class': 'derived', 'momentum_ratio_5m_15m': 'derived',
            'market_regime': 'derived'
        }
