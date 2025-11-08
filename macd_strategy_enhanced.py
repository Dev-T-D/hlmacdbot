"""
Enhanced MACD Trading Strategy with Advanced Filters

Comprehensive MACD strategy with multi-timeframe analysis, volume confirmation,
volatility filters, market regime detection, and adaptive parameters.

Features:
- Multi-timeframe trend confirmation
- Volume analysis and surge detection
- Volatility filters (ATR, Bollinger Bands)
- Market regime classification (ADX-based)
- Adaptive parameter adjustment
- Advanced entry/exit filters
- Risk-adjusted position sizing
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Union, Any, List
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from constants import (
    MACD_MIN_CANDLES_BUFFER,
    MIN_STOP_DISTANCE_PCT,
    MAX_STOP_DISTANCE_PCT
)
from exceptions import IndicatorCalculationError, EntrySignalError
from input_sanitizer import InputSanitizer

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class AdaptiveParameters:
    """Adaptive parameters that adjust based on performance."""
    base_fast_length: int = 12
    base_slow_length: int = 26
    base_signal_length: int = 9
    current_fast_length: int = 12
    current_slow_length: int = 26
    current_signal_length: int = 9
    performance_window: int = 20  # trades to evaluate
    min_win_rate_threshold: float = 0.40
    pause_trades_threshold: float = 0.35  # pause if win rate below this
    pause_duration_minutes: int = 60
    last_parameter_adjustment: Optional[datetime] = None
    paused_until: Optional[datetime] = None


@dataclass
class MarketCondition:
    """Current market condition analysis."""
    regime: MarketRegime
    adx_value: float
    trend_strength: float
    volatility_ratio: float  # current ATR / average ATR
    volume_ratio: float  # current volume / average volume
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    fibonacci_levels: Dict[str, float] = field(default_factory=dict)


class EnhancedMACDStrategy:
    """
    Enhanced MACD Strategy with Advanced Filters and Multi-Timeframe Analysis.

    Includes comprehensive market analysis, risk management, and adaptive parameters.
    """

    def __init__(self,
                 # MACD Parameters
                 fast_length: int = 12,
                 slow_length: int = 26,
                 signal_length: int = 9,

                 # Risk Management
                 risk_reward_ratio: float = 2.0,
                 base_position_size_pct: float = 0.05,

                 # Multi-Timeframe Settings
                 higher_timeframe_multiplier: int = 12,  # e.g., 12x for 5m -> 1h
                 require_higher_tf_alignment: bool = True,

                 # Volume Analysis
                 require_volume_confirmation: bool = True,
                 volume_period: int = 20,
                 min_volume_multiplier: float = 1.2,  # require 1.2x average volume
                 volume_surge_threshold: float = 3.0,  # 3x average = extreme caution

                 # Volatility Filters
                 use_atr_filter: bool = True,
                 atr_period: int = 14,
                 max_volatility_multiplier: float = 3.0,  # skip if ATR > 3x average
                 use_bollinger_filter: bool = True,
                 bollinger_period: int = 20,
                 bollinger_std: float = 2.0,

                 # Market Regime Detection
                 use_market_regime_filter: bool = True,
                 adx_period: int = 14,
                 adx_trending_threshold: float = 25.0,
                 adx_ranging_threshold: float = 20.0,

                 # Additional Filters
                 use_rsi_divergence: bool = True,
                 rsi_period: int = 14,
                 use_support_resistance: bool = True,
                 sr_lookback_periods: int = 50,
                 use_fibonacci_levels: bool = True,
                 use_round_number_filter: bool = True,
                 round_number_tolerance: float = 0.001,  # 0.1% tolerance
                 use_time_filter: bool = True,
                 trading_hours_start: int = 0,  # 0 = midnight UTC
                 trading_hours_end: int = 23,  # 23 = 11 PM UTC

                 # Exit Enhancements
                 use_partial_profits: bool = True,
                 partial_profit_ratio: float = 0.5,  # close 50% at 1:1 RR
                 partial_profit_target: float = 1.0,  # take partial at 1:1
                 use_time_based_exit: bool = True,
                 max_trade_duration_hours: int = 24,
                 use_break_even_stop: bool = True,
                 break_even_activation_rr: float = 1.0,  # move SL to BE after +1R

                 # Adaptive Parameters
                 use_adaptive_parameters: bool = True,

                 # Legacy compatibility
                 rsi_oversold: float = 30.0,
                 rsi_overbought: float = 70.0,
                 min_histogram_strength: float = 0.0,
                 min_trend_strength: float = 0.0,
                 strict_long_conditions: bool = True,
                 disable_long_trades: bool = False):

        # Initialize basic parameters
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.signal_length = signal_length
        self.risk_reward_ratio = risk_reward_ratio
        self.base_position_size_pct = base_position_size_pct

        # Multi-timeframe settings
        self.higher_timeframe_multiplier = higher_timeframe_multiplier
        self.require_higher_tf_alignment = require_higher_tf_alignment

        # Volume analysis
        self.require_volume_confirmation = require_volume_confirmation
        self.volume_period = volume_period
        self.min_volume_multiplier = min_volume_multiplier
        self.volume_surge_threshold = volume_surge_threshold

        # Volatility filters
        self.use_atr_filter = use_atr_filter
        self.atr_period = atr_period
        self.max_volatility_multiplier = max_volatility_multiplier
        self.use_bollinger_filter = use_bollinger_filter
        self.bollinger_period = bollinger_period
        self.bollinger_std = bollinger_std

        # Market regime detection
        self.use_market_regime_filter = use_market_regime_filter
        self.adx_period = adx_period
        self.adx_trending_threshold = adx_trending_threshold
        self.adx_ranging_threshold = adx_ranging_threshold

        # Additional filters
        self.use_rsi_divergence = use_rsi_divergence
        self.rsi_period = rsi_period
        self.use_support_resistance = use_support_resistance
        self.sr_lookback_periods = sr_lookback_periods
        self.use_fibonacci_levels = use_fibonacci_levels
        self.use_round_number_filter = use_round_number_filter
        self.round_number_tolerance = round_number_tolerance
        self.use_time_filter = use_time_filter
        self.trading_hours_start = trading_hours_start
        self.trading_hours_end = trading_hours_end

        # Exit enhancements
        self.use_partial_profits = use_partial_profits
        self.partial_profit_ratio = partial_profit_ratio
        self.partial_profit_target = partial_profit_target
        self.use_time_based_exit = use_time_based_exit
        self.max_trade_duration_hours = max_trade_duration_hours
        self.use_break_even_stop = use_break_even_stop
        self.break_even_activation_rr = break_even_activation_rr

        # Adaptive parameters
        self.use_adaptive_parameters = use_adaptive_parameters
        self.adaptive_params = AdaptiveParameters(
            base_fast_length=fast_length,
            base_slow_length=slow_length,
            base_signal_length=signal_length
        )

        # Legacy parameters for compatibility
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.min_histogram_strength = min_histogram_strength
        self.min_trend_strength = min_trend_strength
        self.strict_long_conditions = strict_long_conditions
        self.disable_long_trades = disable_long_trades

        # Internal state
        self.min_candles = max(
            MACD_MIN_CANDLES_BUFFER,
            slow_length + signal_length,
            volume_period,
            atr_period,
            adx_period,
            bollinger_period,
            sr_lookback_periods
        )

        # Performance tracking for adaptive parameters
        self.recent_performance: List[Dict[str, Any]] = []
        self.max_performance_history = 100

        logger.info("Enhanced MACD Strategy initialized with advanced filters")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators needed for the strategy."""
        try:
            df = df.copy()

            # Basic MACD calculation
            fast_ema = df['close'].ewm(span=self.fast_length, adjust=False).mean()
            slow_ema = df['close'].ewm(span=self.slow_length, adjust=False).mean()
            df['macd'] = fast_ema - slow_ema
            df['signal'] = df['macd'].ewm(span=self.signal_length, adjust=False).mean()
            df['histogram'] = df['macd'] - df['signal']

            # RSI for momentum
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # Volume analysis
            if 'volume' in df.columns:
                df['volume_ma'] = df['volume'].rolling(window=self.volume_period).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma']

            # ATR for volatility
            if self.use_atr_filter:
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['atr'] = tr.rolling(window=self.atr_period).mean()
                df['atr_ratio'] = df['atr'] / df['atr'].rolling(window=self.atr_period * 2).mean()

            # Bollinger Bands
            if self.use_bollinger_filter:
                df['bb_middle'] = df['close'].rolling(window=self.bollinger_period).mean()
                df['bb_std'] = df['close'].rolling(window=self.bollinger_period).std()
                df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * self.bollinger_std)
                df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * self.bollinger_std)
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

            # ADX for trend strength
            if self.use_market_regime_filter:
                df = self._calculate_adx(df)

            # Support/Resistance levels
            if self.use_support_resistance:
                df = self._calculate_support_resistance(df)

            # Fibonacci levels (requires recent swing points)
            if self.use_fibonacci_levels:
                df = self._calculate_fibonacci_levels(df)

            return df

        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            raise IndicatorCalculationError(f"Indicator calculation failed: {e}")

    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average Directional Index (ADX) for trend strength."""
        try:
            # True Range
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )

            # Directional Movement
            df['dm_plus'] = np.where(
                (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                np.maximum(df['high'] - df['high'].shift(1), 0),
                0
            )
            df['dm_minus'] = np.where(
                (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                np.maximum(df['low'].shift(1) - df['low'], 0),
                0
            )

            # Smoothed averages
            df['tr_smooth'] = df['tr'].rolling(window=self.adx_period).mean()
            df['dm_plus_smooth'] = df['dm_plus'].rolling(window=self.adx_period).mean()
            df['dm_minus_smooth'] = df['dm_minus'].rolling(window=self.adx_period).mean()

            # Directional Indicators
            df['di_plus'] = (df['dm_plus_smooth'] / df['tr_smooth']) * 100
            df['di_minus'] = (df['dm_minus_smooth'] / df['tr_smooth']) * 100

            # ADX
            df['dx'] = (abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])) * 100
            df['adx'] = df['dx'].rolling(window=self.adx_period).mean()

            return df

        except Exception as e:
            logger.warning(f"Failed to calculate ADX: {e}")
            return df

    def _calculate_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate support and resistance levels."""
        try:
            # Simple pivot points - can be enhanced with more sophisticated analysis
            lookback = self.sr_lookback_periods

            # Recent highs and lows
            df['recent_high'] = df['high'].rolling(window=lookback).max()
            df['recent_low'] = df['low'].rolling(window=lookback).min()

            # Support: recent low + some buffer
            df['support_level'] = df['recent_low'] * 1.002  # 0.2% above recent low

            # Resistance: recent high - some buffer
            df['resistance_level'] = df['recent_high'] * 0.998  # 0.2% below recent high

            return df

        except Exception as e:
            logger.warning(f"Failed to calculate support/resistance: {e}")
            return df

    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Fibonacci retracement levels from recent swing points."""
        try:
            # Simple implementation - find recent high/low swing
            lookback = 20  # Look for swings in last 20 candles

            # Find recent swing high and low
            recent_high = df['high'].rolling(window=lookback).max()
            recent_low = df['low'].rolling(window=lookback).min()
            recent_range = recent_high - recent_low

            # Fibonacci levels
            df['fib_236'] = recent_high - (recent_range * 0.236)
            df['fib_382'] = recent_high - (recent_range * 0.382)
            df['fib_500'] = recent_high - (recent_range * 0.500)
            df['fib_618'] = recent_high - (recent_range * 0.618)

            return df

        except Exception as e:
            logger.warning(f"Failed to calculate Fibonacci levels: {e}")
            return df

    def get_market_condition(self, df: pd.DataFrame) -> MarketCondition:
        """Analyze current market conditions."""
        try:
            current = df.iloc[-1]

            # Determine market regime
            adx = current.get('adx', 20.0)

            if adx > self.adx_trending_threshold:
                # Determine trend direction
                di_plus = current.get('di_plus', 0)
                di_minus = current.get('di_minus', 0)
                if di_plus > di_minus:
                    regime = MarketRegime.TRENDING_UP
                else:
                    regime = MarketRegime.TRENDING_DOWN
            elif adx < self.adx_ranging_threshold:
                regime = MarketRegime.RANGING
            else:
                regime = MarketRegime.HIGH_VOLATILITY

            # Volatility analysis
            volatility_ratio = current.get('atr_ratio', 1.0)

            # Volume analysis
            volume_ratio = current.get('volume_ratio', 1.0)

            # Support/Resistance levels
            support_level = current.get('support_level')
            resistance_level = current.get('resistance_level')

            # Fibonacci levels
            fib_levels = {}
            for level in ['fib_236', 'fib_382', 'fib_500', 'fib_618']:
                if level in current and pd.notna(current[level]):
                    fib_levels[level] = current[level]

            return MarketCondition(
                regime=regime,
                adx_value=adx,
                trend_strength=adx,
                volatility_ratio=volatility_ratio,
                volume_ratio=volume_ratio,
                support_level=support_level,
                resistance_level=resistance_level,
                fibonacci_levels=fib_levels
            )

        except Exception as e:
            logger.warning(f"Failed to analyze market condition: {e}")
            return MarketCondition(
                regime=MarketRegime.RANGING,
                adx_value=20.0,
                trend_strength=20.0,
                volatility_ratio=1.0,
                volume_ratio=1.0
            )

    def check_multi_timeframe_alignment(self, df: pd.DataFrame, higher_tf_df: Optional[pd.DataFrame] = None) -> Tuple[bool, str]:
        """
        Check if current timeframe aligns with higher timeframe trend.

        Args:
            df: Current timeframe data
            higher_tf_df: Higher timeframe data (if available)

        Returns:
            Tuple of (aligned, reason)
        """
        if not self.require_higher_tf_alignment:
            return True, "Higher timeframe alignment not required"

        try:
            # For now, use a simple trend check on current data
            # In production, this would analyze higher_tf_df
            current = df.iloc[-1]

            # Simple trend check using EMA alignment
            ema_50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
            ema_200 = df['close'].ewm(span=200, adjust=False).mean().iloc[-1]

            if len(df) < 200:
                return True, "Insufficient data for trend analysis"

            if ema_50 > ema_200 and current['close'] > ema_50:
                trend = "bullish"
            elif ema_50 < ema_200 and current['close'] < ema_50:
                trend = "bearish"
            else:
                trend = "neutral"

            return trend != "neutral", f"Higher timeframe trend: {trend}"

        except Exception as e:
            logger.warning(f"Multi-timeframe check failed: {e}")
            return True, "Multi-timeframe check failed, proceeding"

    def check_volume_conditions(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check volume-based entry conditions."""
        if not self.require_volume_confirmation:
            return True, "Volume confirmation not required"

        try:
            current = df.iloc[-1]

            if 'volume_ratio' not in current or pd.isna(current['volume_ratio']):
                return False, "Volume data not available"

            volume_ratio = current['volume_ratio']

            # Check for volume surge (extreme caution)
            if volume_ratio > self.volume_surge_threshold:
                return False, f"Volume surge detected ({volume_ratio:.1f}x average)"

            # Check minimum volume requirement
            if volume_ratio < self.min_volume_multiplier:
                return False, f"Insufficient volume ({volume_ratio:.1f}x < {self.min_volume_multiplier:.1f}x required)"

            return True, f"Volume confirmed ({volume_ratio:.1f}x average)"

        except Exception as e:
            return False, f"Volume check failed: {e}"

    def check_volatility_conditions(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check volatility-based entry conditions."""
        try:
            current = df.iloc[-1]

            # ATR-based volatility filter
            if self.use_atr_filter and 'atr_ratio' in current and pd.notna(current['atr_ratio']):
                atr_ratio = current['atr_ratio']
                if atr_ratio > self.max_volatility_multiplier:
                    return False, f"High volatility (ATR {atr_ratio:.1f}x normal)"

            # Bollinger Band position filter
            if self.use_bollinger_filter and 'bb_position' in current and pd.notna(current['bb_position']):
                bb_pos = current['bb_position']
                # Avoid entries near bands in ranging markets
                market_condition = self.get_market_condition(df)
                if market_condition.regime == MarketRegime.RANGING:
                    if bb_pos < 0.2 or bb_pos > 0.8:  # Near lower/upper bands
                        return False, f"Near Bollinger Band in ranging market (position: {bb_pos:.2f})"

            return True, "Volatility conditions met"

        except Exception as e:
            return False, f"Volatility check failed: {e}"

    def check_additional_filters(self, df: pd.DataFrame, signal_type: str) -> Tuple[bool, str]:
        """Check additional entry filters."""
        try:
            current = df.iloc[-1]

            # Time filter
            if self.use_time_filter:
                current_hour = pd.Timestamp.now().hour
                if not (self.trading_hours_start <= current_hour <= self.trading_hours_end):
                    return False, f"Outside trading hours ({current_hour}:00)"

            # Round number filter
            if self.use_round_number_filter:
                price = current['close']
                round_numbers = [10000, 50000, 100000]  # Common round numbers

                for round_num in round_numbers:
                    tolerance = round_num * self.round_number_tolerance
                    if abs(price - round_num) <= tolerance:
                        return False, f"Near round number {round_num} (price: {price:.2f})"

            # Support/Resistance filter
            if self.use_support_resistance and 'support_level' in current and 'resistance_level' in current:
                if signal_type == 'LONG' and current['close'] <= current['support_level']:
                    return False, "Price near support level"
                elif signal_type == 'SHORT' and current['close'] >= current['resistance_level']:
                    return False, "Price near resistance level"

            # Fibonacci level filter (avoid entries at fib levels in ranging markets)
            market_condition = self.get_market_condition(df)
            if self.use_fibonacci_levels and market_condition.regime == MarketRegime.RANGING:
                for fib_level, fib_price in market_condition.fibonacci_levels.items():
                    if abs(current['close'] - fib_price) / current['close'] < 0.005:  # Within 0.5%
                        return False, f"Near Fibonacci level {fib_level}"

            # RSI divergence (simplified check)
            if self.use_rsi_divergence and len(df) >= 10:
                recent_prices = df['close'].tail(10)
                recent_rsi = df['rsi'].tail(10)

                # Simple divergence check - price making higher high, RSI making lower high
                if (signal_type == 'LONG' and
                    recent_prices.iloc[-1] > recent_prices.iloc[-3] and
                    recent_rsi.iloc[-1] < recent_rsi.iloc[-3]):
                    return False, "RSI divergence detected (bullish signal weakened)"

                # Price making lower low, RSI making higher low
                elif (signal_type == 'SHORT' and
                      recent_prices.iloc[-1] < recent_prices.iloc[-3] and
                      recent_rsi.iloc[-1] > recent_rsi.iloc[-3]):
                    return False, "RSI divergence detected (bearish signal weakened)"

            return True, "All additional filters passed"

        except Exception as e:
            return False, f"Additional filters check failed: {e}"

    def check_entry_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Union[str, float, Any]]]:
        """
        Enhanced entry signal check with all advanced filters.
        """
        if len(df) < self.min_candles:
            return None

        try:
            # Check if trading is paused due to poor performance
            if self._is_trading_paused():
                logger.info("Trading paused due to poor recent performance")
                return None

            # Calculate indicators
            df = self.calculate_indicators(df)
            current = df.iloc[-1]

            # Basic MACD signal detection (similar to original)
            bullish_cross, bearish_cross = self.detect_crossover(df)

            # Get market condition
            market_condition = self.get_market_condition(df)

            # Check multi-timeframe alignment
            mtf_aligned, mtf_reason = self.check_multi_timeframe_alignment(df)
            if not mtf_aligned:
                logger.debug(f"Multi-timeframe not aligned: {mtf_reason}")
                return None

            # Check volume conditions
            volume_ok, volume_reason = self.check_volume_conditions(df)
            if not volume_ok:
                logger.debug(f"Volume conditions not met: {volume_reason}")
                return None

            # Check volatility conditions
            volatility_ok, volatility_reason = self.check_volatility_conditions(df)
            if not volatility_ok:
                logger.debug(f"Volatility conditions not met: {volatility_reason}")
                return None

            # LONG SIGNAL CHECK
            if bullish_cross and not self.disable_long_trades:
                # Additional filters for LONG
                filters_ok, filter_reason = self.check_additional_filters(df, 'LONG')
                if not filters_ok:
                    logger.debug(f"LONG filters not met: {filter_reason}")
                    return None

                # Market regime adjustments
                position_size_pct = self._calculate_position_size(df, 'LONG', market_condition)

                return {
                    'signal': 'LONG',
                    'price': current['close'],
                    'stop_loss': self.calculate_stop_loss(current['close'], 'LONG', market_condition),
                    'take_profit': self.calculate_take_profit(current['close'], 'LONG'),
                    'position_size_pct': position_size_pct,
                    'confidence': self._calculate_signal_confidence(df, 'LONG'),
                    'market_regime': market_condition.regime.value,
                    'indicators': {
                        'macd': current['macd'],
                        'signal': current['signal'],
                        'histogram': current['histogram'],
                        'rsi': current.get('rsi', 50),
                        'adx': market_condition.adx_value
                    }
                }

            # SHORT SIGNAL CHECK
            elif bearish_cross:
                # Additional filters for SHORT
                filters_ok, filter_reason = self.check_additional_filters(df, 'SHORT')
                if not filters_ok:
                    logger.debug(f"SHORT filters not met: {filter_reason}")
                    return None

                # Market regime adjustments
                position_size_pct = self._calculate_position_size(df, 'SHORT', market_condition)

                return {
                    'signal': 'SHORT',
                    'price': current['close'],
                    'stop_loss': self.calculate_stop_loss(current['close'], 'SHORT', market_condition),
                    'take_profit': self.calculate_take_profit(current['close'], 'SHORT'),
                    'position_size_pct': position_size_pct,
                    'confidence': self._calculate_signal_confidence(df, 'SHORT'),
                    'market_regime': market_condition.regime.value,
                    'indicators': {
                        'macd': current['macd'],
                        'signal': current['signal'],
                        'histogram': current['histogram'],
                        'rsi': current.get('rsi', 50),
                        'adx': market_condition.adx_value
                    }
                }

            return None

        except Exception as e:
            logger.error(f"Entry signal check failed: {e}")
            return None

    def detect_crossover(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """Detect MACD crossovers (same as original)."""
        if len(df) < 2:
            return False, False

        current = df.iloc[-1]
        previous = df.iloc[-2]

        bullish_cross = (current['macd'] > current['signal'] and
                        previous['macd'] <= previous['signal'])

        bearish_cross = (current['macd'] < current['signal'] and
                        previous['macd'] >= previous['signal'])

        return bullish_cross, bearish_cross

    def _calculate_position_size(self, df: pd.DataFrame, signal_type: str, market_condition: MarketCondition) -> float:
        """Calculate position size based on volatility and market conditions."""
        base_size = self.base_position_size_pct

        # Reduce size in high volatility
        if market_condition.volatility_ratio > 2.0:
            base_size *= 0.7  # 30% reduction

        # Adjust based on market regime
        if market_condition.regime == MarketRegime.TRENDING_UP and signal_type == 'LONG':
            base_size *= 1.2  # Increase size in trending market with trend
        elif market_condition.regime == MarketRegime.TRENDING_DOWN and signal_type == 'SHORT':
            base_size *= 1.2
        elif market_condition.regime == MarketRegime.RANGING:
            base_size *= 0.8  # Reduce size in ranging markets

        return min(base_size, 0.1)  # Cap at 10%

    def _calculate_signal_confidence(self, df: pd.DataFrame, signal_type: str) -> float:
        """Calculate signal confidence score (0-1)."""
        confidence = 0.5  # Base confidence

        try:
            current = df.iloc[-1]

            # Histogram strength
            hist_strength = abs(current.get('histogram', 0)) / abs(current.get('close', 1))
            confidence += min(hist_strength * 10, 0.2)  # Max +0.2

            # RSI confirmation
            rsi = current.get('rsi', 50)
            if signal_type == 'LONG' and rsi > 50:
                confidence += 0.1
            elif signal_type == 'SHORT' and rsi < 50:
                confidence += 0.1

            # Volume confirmation
            volume_ratio = current.get('volume_ratio', 1.0)
            if volume_ratio > 1.5:
                confidence += 0.1

            # ADX strength (trend confirmation)
            adx = current.get('adx', 20)
            if adx > 25:
                confidence += 0.1

        except Exception:
            pass

        return min(confidence, 1.0)

    def calculate_stop_loss(self, entry_price: float, signal_type: str, market_condition: MarketCondition) -> float:
        """Calculate stop loss with volatility adjustment."""
        # Base stop distance (2% for crypto)
        base_stop_pct = 0.02

        # Increase stop in high volatility
        if market_condition.volatility_ratio > 1.5:
            base_stop_pct *= 1.5

        # Wider stops in trending markets
        if market_condition.regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            base_stop_pct *= 1.3

        if signal_type == 'LONG':
            return entry_price * (1 - base_stop_pct)
        else:
            return entry_price * (1 + base_stop_pct)

    def calculate_take_profit(self, entry_price: float, signal_type: str) -> float:
        """Calculate take profit based on risk-reward ratio."""
        risk = abs(entry_price - self.calculate_stop_loss(entry_price, signal_type,
                                                        MarketCondition(MarketRegime.RANGING, 20, 20, 1.0, 1.0)))
        reward = risk * self.risk_reward_ratio

        if signal_type == 'LONG':
            return entry_price + reward
        else:
            return entry_price - reward

    def check_exit_signal(self, df: pd.DataFrame, position_type: str, entry_time: datetime,
                         entry_price: float, stop_loss: float) -> Tuple[bool, str]:
        """
        Enhanced exit signal check with partial profits and time-based exits.
        """
        try:
            current = df.iloc[-1]
            current_price = current['close']

            # Time-based exit
            if self.use_time_based_exit:
                trade_duration = datetime.now() - entry_time
                if trade_duration.total_seconds() / 3600 > self.max_trade_duration_hours:
                    return True, f"Time-based exit after {self.max_trade_duration_hours} hours"

            # Basic MACD exit (opposite crossover)
            bullish_cross, bearish_cross = self.detect_crossover(df)

            if position_type == 'LONG' and bearish_cross:
                return True, "MACD bearish crossover"
            elif position_type == 'SHORT' and bullish_cross:
                return True, "MACD bullish crossover"

            # Partial profit taking
            if self.use_partial_profits:
                current_rr = abs(current_price - entry_price) / abs(entry_price - stop_loss)

                if current_rr >= self.partial_profit_target:
                    # This would trigger partial close in trading logic
                    logger.info(f"Partial profit target reached: {current_rr:.2f}R")

            # Break-even stop activation
            if self.use_break_even_stop:
                current_rr = abs(current_price - entry_price) / abs(entry_price - stop_loss)

                if current_rr >= self.break_even_activation_rr:
                    # Move stop loss to break-even
                    new_sl = entry_price
                    logger.info(f"Break-even stop activated at {new_sl}")

            return False, "No exit signal"

        except Exception as e:
            logger.error(f"Exit signal check failed: {e}")
            return False, f"Exit check error: {e}"

    def _is_trading_paused(self) -> bool:
        """Check if trading is paused due to poor performance."""
        if not self.use_adaptive_parameters:
            return False

        if self.adaptive_params.paused_until:
            if datetime.now() < self.adaptive_params.paused_until:
                return True
            else:
                # Pause expired, reset
                self.adaptive_params.paused_until = None

        return False

    def update_performance(self, trade_result: Dict[str, Any]):
        """Update strategy performance for adaptive parameters."""
        if not self.use_adaptive_parameters:
            return

        self.recent_performance.append(trade_result)

        # Keep only recent performance
        if len(self.recent_performance) > self.max_performance_history:
            self.recent_performance = self.recent_performance[-self.max_performance_history:]

        # Check if we need to pause trading
        if len(self.recent_performance) >= self.adaptive_params.performance_window:
            recent_trades = self.recent_performance[-self.adaptive_params.performance_window:]
            win_rate = sum(1 for t in recent_trades if t.get('pnl', 0) > 0) / len(recent_trades)

            if win_rate < self.adaptive_params.pause_trades_threshold:
                self.adaptive_params.paused_until = datetime.now() + timedelta(
                    minutes=self.adaptive_params.pause_duration_minutes
                )
                logger.warning(f"Trading paused due to low win rate: {win_rate:.1%}")

    def get_strategy_config(self) -> Dict[str, Any]:
        """Get current strategy configuration for logging."""
        return {
            'fast_length': self.adaptive_params.current_fast_length,
            'slow_length': self.adaptive_params.current_slow_length,
            'signal_length': self.adaptive_params.current_signal_length,
            'risk_reward_ratio': self.risk_reward_ratio,
            'higher_tf_multiplier': self.higher_timeframe_multiplier,
            'require_volume_confirmation': self.require_volume_confirmation,
            'use_atr_filter': self.use_atr_filter,
            'use_market_regime_filter': self.use_market_regime_filter,
            'use_adaptive_parameters': self.use_adaptive_parameters,
            'trading_paused': self._is_trading_paused()
        }
