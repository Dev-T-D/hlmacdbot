"""
MACD Overlay Trading Strategy

Adapted from backtesting to live trading

"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Union, Any
import logging

from constants import (
    MACD_MIN_CANDLES_BUFFER,
    MIN_STOP_DISTANCE_PCT,
    MAX_STOP_DISTANCE_PCT
)
from exceptions import IndicatorCalculationError, EntrySignalError
from input_sanitizer import InputSanitizer

logger = logging.getLogger(__name__)


class MACDStrategy:
    """MACD Overlay Strategy for Live Trading"""
    
    def __init__(self, 
                 fast_length: int = 12,
                 slow_length: int = 26, 
                 signal_length: int = 9,
                 risk_reward_ratio: float = 2.0,
                 rsi_period: int = 14,
                 rsi_oversold: float = 30.0,
                 rsi_overbought: float = 70.0,
                 min_histogram_strength: float = 0.0,
                 require_volume_confirmation: bool = True,
                 volume_period: int = 20,
                 min_trend_strength: float = 0.0,
                 strict_long_conditions: bool = True,
                 disable_long_trades: bool = False):
        """
        Initialize MACD Strategy with improved filters
        
        Args:
            fast_length: Fast EMA period (default: 12)
            slow_length: Slow EMA period (default: 26)
            signal_length: Signal line period (default: 9)
            risk_reward_ratio: Risk/reward ratio for TP (default: 2.0)
            rsi_period: RSI period for momentum filter (default: 14)
            rsi_oversold: RSI oversold level (default: 30.0)
            rsi_overbought: RSI overbought level (default: 70.0)
            min_histogram_strength: Minimum histogram value for entry (default: 0.0)
            require_volume_confirmation: Require volume above average (default: True)
            volume_period: Period for volume average (default: 20)
            min_trend_strength: Minimum MACD-signal distance for trend strength (default: 0.0)
            strict_long_conditions: Use stricter conditions for LONG entries (default: True)
            disable_long_trades: Disable LONG trades entirely (default: False)
        """
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.signal_length = signal_length
        self.risk_reward_ratio = risk_reward_ratio
        
        # Improved filters
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.min_histogram_strength = min_histogram_strength
        self.require_volume_confirmation = require_volume_confirmation
        self.volume_period = volume_period
        self.min_trend_strength = min_trend_strength
        self.strict_long_conditions = strict_long_conditions
        self.disable_long_trades = disable_long_trades
        
        # Minimum candles needed for calculation (increased for RSI)
        self.min_candles = max(slow_length, signal_length, rsi_period, volume_period) + MACD_MIN_CANDLES_BUFFER
        
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD indicators with validation
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            DataFrame with MACD indicators added
            
        Raises:
            ValueError: If indicators contain invalid values (NaN/Inf)
        """
        df = df.copy()
        
        # Validate input data
        if df['close'].isna().any():
            nan_count = df['close'].isna().sum()
            nan_indices = df[df['close'].isna()].index.tolist()[:10]  # Show first 10
            logger.warning(
                f"Found {nan_count} NaN value(s) in 'close' column "
                f"(indices: {nan_indices}{'...' if nan_count > 10 else ''}) - "
                f"attempting forward/backward fill"
            )
            # Use ffill() and bfill() instead of deprecated method parameter
            df['close'] = df['close'].ffill().bfill()
            # If still NaN after fill, raise error
            if df['close'].isna().any():
                remaining_nan = df['close'].isna().sum()
                remaining_indices = df[df['close'].isna()].index.tolist()
                raise IndicatorCalculationError(
                    f"Cannot fill all NaN values in 'close' column. "
                    f"After fill attempt, {remaining_nan} NaN value(s) remain "
                    f"at indices: {remaining_indices}. "
                    f"This indicates data quality issues - check data source."
                )
        
        if (df['close'] <= 0).any():
            invalid_count = (df['close'] <= 0).sum()
            invalid_indices = df[df['close'] <= 0].index.tolist()[:10]
            invalid_values = df.loc[df['close'] <= 0, 'close'].tolist()[:10]
            raise IndicatorCalculationError(
                f"Found {invalid_count} non-positive price value(s) in 'close' column. "
                f"Indices: {invalid_indices}{'...' if invalid_count > 10 else ''}, "
                f"Values: {invalid_values}{'...' if invalid_count > 10 else ''}. "
                f"Prices must be positive - check data source for errors."
            )
        
        # Calculate MACD
        fast_ema = self.calculate_ema(df['close'], self.fast_length)
        slow_ema = self.calculate_ema(df['close'], self.slow_length)
        
        df['fast_ema'] = fast_ema
        df['slow_ema'] = slow_ema
        df['macd'] = fast_ema - slow_ema
        df['signal'] = self.calculate_ema(df['macd'], self.signal_length)
        df['histogram'] = df['macd'] - df['signal']
        
        # Calculate RSI for momentum filter
        df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
        
        # Calculate volume moving average for volume confirmation
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=self.volume_period).mean()
        else:
            df['volume_ma'] = pd.Series([0] * len(df), index=df.index)
        
        # Validate calculated indicators for NaN/Inf
        indicator_columns = ['fast_ema', 'slow_ema', 'macd', 'signal', 'histogram', 'rsi', 'volume_ma']
        for col in indicator_columns:
            nan_count = df[col].isna().sum()
            inf_count = np.isinf(df[col]).sum()
            
            if nan_count > 0:
                logger.warning(
                    f"Found {nan_count} NaN values in '{col}' - "
                    f"this is normal for initial {self.slow_length} candles"
                )
                # Fill NaN values with 0 for initial periods (before enough data)
                df[col] = df[col].fillna(0)
            
            if inf_count > 0:
                inf_indices = df[np.isinf(df[col])].index.tolist()[:10]
                logger.error(
                    f"Found {inf_count} Inf value(s) in '{col}' "
                    f"(indices: {inf_indices}{'...' if inf_count > 10 else ''}) - "
                    f"data quality issue!"
                )
                raise IndicatorCalculationError(
                    f"Invalid indicator calculation: '{col}' contains {inf_count} Inf value(s). "
                    f"Inf values found at indices: {inf_indices}{'...' if inf_count > 10 else ''}. "
                    f"This indicates a data quality problem - check for zero or invalid values "
                    f"in input data (e.g., zero prices, division by zero)."
                )
        
        # Additional validation: Check for reasonable values
        # MACD and signal should be finite numbers
        if not np.isfinite(df['macd']).all():
            invalid_indices = df[~np.isfinite(df['macd'])].index.tolist()
            invalid_count = len(invalid_indices)
            invalid_values = df.loc[~np.isfinite(df['macd']), 'macd'].tolist()[:10]
            logger.error(
                f"Non-finite MACD values: {invalid_count} found at indices "
                f"{invalid_indices[:10]}{'...' if invalid_count > 10 else ''}"
            )
            raise IndicatorCalculationError(
                f"MACD calculation produced {invalid_count} non-finite value(s). "
                f"Indices: {invalid_indices[:10]}{'...' if invalid_count > 10 else ''}, "
                f"Values: {invalid_values}{'...' if invalid_count > 10 else ''}. "
                f"This may indicate: 1) Insufficient data (need at least {self.slow_length} candles), "
                f"2) Invalid input prices, or 3) Calculation overflow. "
                f"Check input data quality and ensure sufficient historical data."
            )
        
        if not np.isfinite(df['signal']).all():
            invalid_indices = df[~np.isfinite(df['signal'])].index.tolist()
            invalid_count = len(invalid_indices)
            invalid_values = df.loc[~np.isfinite(df['signal']), 'signal'].tolist()[:10]
            logger.error(
                f"Non-finite Signal values: {invalid_count} found at indices "
                f"{invalid_indices[:10]}{'...' if invalid_count > 10 else ''}"
            )
            raise IndicatorCalculationError(
                f"Signal calculation produced {invalid_count} non-finite value(s). "
                f"Indices: {invalid_indices[:10]}{'...' if invalid_count > 10 else ''}, "
                f"Values: {invalid_values}{'...' if invalid_count > 10 else ''}. "
                f"This may indicate: 1) Invalid MACD values (check MACD calculation), "
                f"2) Insufficient data (need at least {self.signal_length} candles for signal), "
                f"or 3) Calculation overflow. Check MACD values and input data quality."
            )
        
        return df
    
    def calculate_rsi(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: Price series (typically close prices)
            period: RSI period (default: 14)
            
        Returns:
            RSI series
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def is_bullish_candle(self, row: pd.Series) -> bool:
        """Check if candle is bullish"""
        return row['close'] > row['open']
    
    def is_bearish_candle(self, row: pd.Series) -> bool:
        """Check if candle is bearish"""
        return row['close'] < row['open']
    
    def detect_crossover(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """
        Detect MACD crossovers using last 2 candles
        
        Returns:
            Tuple of (bullish_cross, bearish_cross)
        """
        if len(df) < 2:
            return False, False
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        bullish_cross = (previous['macd'] <= previous['signal']) and (current['macd'] > current['signal'])
        bearish_cross = (previous['macd'] >= previous['signal']) and (current['macd'] < current['signal'])
        
        return bullish_cross, bearish_cross
    
    def calculate_stop_loss_take_profit(self, 
                                       entry_price: float,
                                       position_type: str,
                                       macd: float,
                                       signal: float) -> Dict[str, float]:
        """
        Calculate stop loss and take profit levels with validation
        
        Args:
            entry_price: Entry price
            position_type: "LONG" or "SHORT"
            macd: Current MACD value
            signal: Current Signal value
            
        Returns:
            Dict with 'stop_loss', 'take_profit', 'risk'
            
        Raises:
            ValueError: If calculated stop loss or take profit is invalid
        """
        # Sanitize and validate inputs
        entry_price = InputSanitizer.sanitize_price(entry_price, 'entry_price')
        
        if position_type not in ["LONG", "SHORT"]:
            raise EntrySignalError(
                f"Invalid position type: '{position_type}'. "
                f"Must be 'LONG' or 'SHORT' (got '{position_type}'). "
                f"Check signal generation logic."
            )
        
        # Validate MACD and signal are finite
        if not np.isfinite(macd) or not np.isfinite(signal):
            macd_status = "finite" if np.isfinite(macd) else f"non-finite ({macd})"
            signal_status = "finite" if np.isfinite(signal) else f"non-finite ({signal})"
            raise EntrySignalError(
                f"Invalid MACD indicator values: MACD={macd_status}, Signal={signal_status}. "
                f"Both values must be finite numbers. "
                f"This may indicate: 1) Insufficient data for indicator calculation, "
                f"2) Invalid market data, or 3) Calculation error. "
                f"Check indicator calculation and input data quality."
            )
        
        # Improved stop loss calculation: Use percentage-based approach
        # This prevents premature stops from MACD-signal noise
        # Use 1% of entry price as base stop distance (wider stops = fewer premature exits)
        base_stop_pct = 0.01  # 1% of entry price
        
        # Also consider MACD-signal difference, but don't let it make stops too tight
        macd_signal_diff = abs(macd - signal)
        macd_signal_pct = abs(macd_signal_diff / entry_price) if entry_price > 0 else 0
        
        # Use the larger of: base percentage or MACD-signal based (but not too tight)
        # This ensures stops are wide enough to avoid noise
        stop_pct = max(base_stop_pct, macd_signal_pct * 0.5)  # Scale MACD diff down
        
        # Minimum and maximum stop loss distances (as percentage of entry price)
        min_stop_distance_pct = MIN_STOP_DISTANCE_PCT
        max_stop_distance_pct = MAX_STOP_DISTANCE_PCT
        
        # Small epsilon for floating-point comparison tolerance
        epsilon = 1e-6
        
        # Clamp stop percentage to reasonable bounds
        stop_pct = max(min_stop_distance_pct, min(stop_pct, max_stop_distance_pct))
        
        # Calculate risk amount from percentage
        risk_amount = entry_price * stop_pct
        
        # Calculate stop loss and take profit
        if position_type == "LONG":
            stop_loss = entry_price - risk_amount
            take_profit = entry_price + (risk_amount * self.risk_reward_ratio)
        else:  # SHORT
            stop_loss = entry_price + risk_amount
            take_profit = entry_price - (risk_amount * self.risk_reward_ratio)
        
        # Validate stop loss is positive and reasonable
        if stop_loss <= 0:
            raise ValueError(
                f"Invalid stop loss: {stop_loss:.8f}. "
                f"Stop loss must be positive (got {stop_loss:.8f}). "
                f"Entry price: {entry_price:.8f}, Risk amount: {risk_amount:.8f}, "
                f"Position type: {position_type}. "
                f"This may indicate: 1) Risk amount too large relative to entry price, "
                f"2) Invalid MACD/Signal values, or 3) Calculation error. "
                f"Check risk calculation and ensure entry price is valid."
            )
        
        # Validate stop loss distance is reasonable
        # Use small epsilon to handle floating-point precision issues
        stop_distance_pct = abs(stop_loss - entry_price) / entry_price
        if stop_distance_pct < (min_stop_distance_pct - epsilon):
            raise ValueError(
                f"Stop loss too close to entry price. "
                f"Distance: {stop_distance_pct*100:.3f}%, Minimum required: {min_stop_distance_pct*100:.1f}%. "
                f"Entry: {entry_price:.8f}, Stop loss: {stop_loss:.8f}, "
                f"Position type: {position_type}. "
                f"Stop loss must be at least {min_stop_distance_pct*100:.1f}% away from entry to avoid "
                f"premature stops due to normal price volatility."
            )
        if stop_distance_pct > max_stop_distance_pct:
            raise ValueError(
                f"Stop loss too far from entry price. "
                f"Distance: {stop_distance_pct*100:.2f}%, Maximum allowed: {max_stop_distance_pct*100:.1f}%. "
                f"Entry: {entry_price:.8f}, Stop loss: {stop_loss:.8f}, "
                f"Position type: {position_type}. "
                f"Stop loss exceeds maximum distance of {max_stop_distance_pct*100:.1f}% - "
                f"this may indicate excessive risk or invalid MACD/Signal values."
            )
        
        # Validate take profit is positive
        if take_profit <= 0:
            raise ValueError(
                f"Invalid take profit: {take_profit:.8f}. "
                f"Take profit must be positive (got {take_profit:.8f}). "
                f"Entry price: {entry_price:.8f}, Risk amount: {risk_amount:.8f}, "
                f"Risk/reward ratio: {self.risk_reward_ratio}, Position type: {position_type}. "
                f"This may indicate: 1) Risk amount too large, 2) Invalid risk/reward ratio, "
                f"or 3) Calculation error. Check risk calculation parameters."
            )
        
        # Validate take profit is reasonable distance from entry
        # Use small epsilon to handle floating-point precision issues
        tp_distance_pct = abs(take_profit - entry_price) / entry_price
        if tp_distance_pct < (min_stop_distance_pct - epsilon):
            logger.warning(
                f"Take profit very close to entry: {tp_distance_pct*100:.3f}%. "
                f"This may result in premature exits."
            )
        
        # Validate risk/reward ratio is maintained
        actual_rr = abs(take_profit - entry_price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
        if actual_rr < 1.0:
            logger.warning(
                f"Risk/reward ratio is less than 1:1 ({actual_rr:.2f}). "
                f"Consider adjusting strategy parameters."
            )
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk': risk_amount
        }
    
    def check_entry_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Union[str, float, Any]]]:
        """
        Check for entry signals
        
        Args:
            df: DataFrame with OHLCV and MACD indicators
            
        Returns:
            Dict with signal details or None
        """
        if len(df) < self.min_candles:
            logger.warning(
                f"Insufficient data for entry signal check: {len(df)} candles available, "
                f"need at least {self.min_candles} candles "
                f"(slow_length={self.slow_length}, signal_length={self.signal_length}). "
                f"Waiting for more market data..."
            )
            return None
        
        # Calculate indicators if not present
        if 'macd' not in df.columns:
            df = self.calculate_indicators(df)
        
        current = df.iloc[-1]
        
        # Validate current indicator values are finite
        required_indicators = ['macd', 'signal', 'histogram', 'close']
        for indicator in required_indicators:
            if indicator not in current:
                logger.error(
                    f"Missing required indicator '{indicator}' in DataFrame. "
                    f"Available columns: {list(current.index)}. "
                    f"Ensure indicators are calculated before checking entry signals."
                )
                return None
            
            value = current[indicator]
            if pd.isna(value) or not np.isfinite(value):
                logger.warning(
                    f"Invalid {indicator} value: {value} (NaN or non-finite). "
                    f"Candle index: {current.name if hasattr(current, 'name') else 'unknown'}. "
                    f"Skipping signal check - ensure indicators are calculated with sufficient data."
                )
                return None
        
        bullish_cross, bearish_cross = self.detect_crossover(df)
        
        # Get histogram value (normalized by price for better comparison)
        histogram_value = current['histogram']
        histogram_normalized = abs(histogram_value / current['close']) if current['close'] > 0 else 0
        
        # Check histogram strength threshold
        histogram_strong_enough = abs(histogram_value) >= self.min_histogram_strength
        
        # Check trend strength (MACD-signal distance)
        trend_strength = abs(current['macd'] - current['signal'])
        trend_strength_normalized = abs(trend_strength / current['close']) if current['close'] > 0 else 0
        trend_strong_enough = trend_strength_normalized >= self.min_trend_strength
        
        # Get RSI for momentum filter
        rsi = current.get('rsi', 50.0)  # Default to neutral if RSI not available
        rsi_valid = pd.notna(rsi) and np.isfinite(rsi)
        
        # Volume confirmation
        volume_confirmed = True
        if self.require_volume_confirmation and 'volume' in current and 'volume_ma' in current:
            volume_ma = current.get('volume_ma', 0)
            current_volume = current.get('volume', 0)
            if pd.notna(volume_ma) and volume_ma > 0:
                volume_confirmed = current_volume >= volume_ma * 0.8  # At least 80% of average volume
        
        bullish_overlay = current['histogram'] > 0
        bearish_overlay = current['histogram'] < 0
        
        # LONG ENTRY CONDITIONS (IMPROVED)
        # Requirements:
        # 1. Bullish overlay (histogram > 0)
        # 2. Histogram strength threshold
        # 3. Trend strength threshold
        # 4. Price above slow EMA
        # 5. Bullish candle
        # 6. Bullish crossover
        # 7. RSI momentum filter (RSI > 50 for LONG, or RSI < oversold for mean reversion)
        # 8. Volume confirmation (if enabled)
        price_above_slow_ema = current['close'] > current.get('slow_ema', current['close'])
        
        # RSI conditions for LONG
        rsi_bullish = True
        if rsi_valid:
            if self.strict_long_conditions:
                # Stricter: RSI should be between 40-70 (not overbought, but showing momentum)
                rsi_bullish = 40.0 <= rsi <= 70.0
            else:
                # Standard: RSI > 50 (momentum) or RSI < oversold (mean reversion)
                rsi_bullish = rsi > 50.0 or rsi < self.rsi_oversold
        
        long_conditions = [
            not self.disable_long_trades,  # Check if LONG trades are disabled
            bullish_overlay,
            histogram_strong_enough,
            trend_strong_enough,
            price_above_slow_ema,
            self.is_bullish_candle(current),
            bullish_cross,
            rsi_bullish,
            volume_confirmed
        ]
        
        if all(long_conditions):
            levels = self.calculate_stop_loss_take_profit(
                current['close'], "LONG", current['macd'], current['signal']
            )
            
            return {
                'type': 'LONG',
                'entry_price': current['close'],
                'stop_loss': levels['stop_loss'],
                'take_profit': levels['take_profit'],
                'risk': levels['risk'],
                'timestamp': current.get('timestamp', current.name)
            }
        
        # SHORT ENTRY CONDITIONS (IMPROVED)
        # Requirements:
        # 1. Bearish overlay (histogram < 0)
        # 2. Histogram strength threshold
        # 3. Trend strength threshold
        # 4. Price below slow EMA
        # 5. Bearish candle
        # 6. Bearish crossover
        # 7. RSI momentum filter (RSI < 50 for SHORT, or RSI > overbought for mean reversion)
        # 8. Volume confirmation (if enabled)
        price_below_slow_ema = current['close'] < current.get('slow_ema', current['close'])
        
        # RSI conditions for SHORT
        rsi_bearish = True
        if rsi_valid:
            # RSI < 50 (momentum) or RSI > overbought (mean reversion)
            rsi_bearish = rsi < 50.0 or rsi > self.rsi_overbought
        
        short_conditions = [
            bearish_overlay,
            histogram_strong_enough,
            trend_strong_enough,
            price_below_slow_ema,
            self.is_bearish_candle(current),
            bearish_cross,
            rsi_bearish,
            volume_confirmed
        ]
        
        if all(short_conditions):
            levels = self.calculate_stop_loss_take_profit(
                current['close'], "SHORT", current['macd'], current['signal']
            )
            
            return {
                'type': 'SHORT',
                'entry_price': current['close'],
                'stop_loss': levels['stop_loss'],
                'take_profit': levels['take_profit'],
                'risk': levels['risk'],
                'timestamp': current.get('timestamp', current.name)
            }
        
        return None
    
    def check_higher_timeframe_trend(self, df_higher_tf: pd.DataFrame, 
                                     signal_type: str) -> Tuple[bool, str]:
        """
        Check higher timeframe trend alignment
        
        Args:
            df_higher_tf: DataFrame with higher timeframe data (must have indicators calculated)
            signal_type: "LONG" or "SHORT" - the signal from lower timeframe
            
        Returns:
            Tuple of (is_aligned, reason)
            is_aligned: True if higher timeframe trend supports the signal
            reason: Explanation of trend status
        """
        if df_higher_tf.empty or len(df_higher_tf) < self.min_candles:
            return False, "Insufficient higher timeframe data"
        
        # Ensure indicators are calculated
        if 'macd' not in df_higher_tf.columns:
            df_higher_tf = self.calculate_indicators(df_higher_tf)
        
        current = df_higher_tf.iloc[-1]
        
        # Validate indicators
        required_indicators = ['macd', 'signal', 'histogram', 'close']
        for indicator in required_indicators:
            if indicator not in current:
                return False, f"Missing {indicator} in higher timeframe"
            value = current[indicator]
            if pd.isna(value) or not np.isfinite(value):
                return False, f"Invalid {indicator} in higher timeframe"
        
        # Check trend based on MACD histogram and price position
        histogram = current['histogram']
        price_above_slow_ema = current['close'] > current.get('slow_ema', current['close'])
        price_below_slow_ema = current['close'] < current.get('slow_ema', current['close'])
        
        # Determine higher timeframe trend
        # Bullish trend: histogram > 0 AND price above slow EMA
        # Bearish trend: histogram < 0 AND price below slow EMA
        higher_tf_bullish = histogram > 0 and price_above_slow_ema
        higher_tf_bearish = histogram < 0 and price_below_slow_ema
        
        # Check alignment
        if signal_type == "LONG":
            if higher_tf_bullish:
                return True, "Higher timeframe trend is bullish - supports LONG entry"
            elif higher_tf_bearish:
                return False, "Higher timeframe trend is bearish - conflicts with LONG entry"
            else:
                # Neutral/uncertain trend
                return False, "Higher timeframe trend is neutral - no clear direction"
        
        elif signal_type == "SHORT":
            if higher_tf_bearish:
                return True, "Higher timeframe trend is bearish - supports SHORT entry"
            elif higher_tf_bullish:
                return False, "Higher timeframe trend is bullish - conflicts with SHORT entry"
            else:
                # Neutral/uncertain trend
                return False, "Higher timeframe trend is neutral - no clear direction"
        
        return False, "Unknown signal type"
    
    def check_exit_signal(self, df: pd.DataFrame, position_type: str) -> Tuple[bool, str]:
        """
        Check for exit signals based on current position
        
        Args:
            df: DataFrame with OHLCV and MACD indicators
            position_type: "LONG" or "SHORT"
            
        Returns:
            Tuple of (should_exit, reason)
        """
        if len(df) < 2:
            return False, ""
        
        # Calculate indicators if not present
        if 'macd' not in df.columns:
            df = self.calculate_indicators(df)
        
        current = df.iloc[-1]
        bullish_cross, bearish_cross = self.detect_crossover(df)
        
        if position_type == "LONG":
            # Improved exit logic: Require confirmation before exiting
            # Exit on bearish crossover (strong signal)
            if bearish_cross:
                return True, "Bearish Crossover"
            # Exit if histogram turns negative AND RSI shows weakness
            elif current['histogram'] < 0:
                rsi = current.get('rsi', 50.0)
                if pd.notna(rsi) and rsi < 45.0:  # RSI below 45 confirms weakness
                    return True, "Histogram Negative + RSI Weakness"
                # Otherwise, wait for stronger confirmation (crossover)
        
        elif position_type == "SHORT":
            # Improved exit logic: Require confirmation before exiting
            # Exit on bullish crossover (strong signal)
            if bullish_cross:
                return True, "Bullish Crossover"
            # Exit if histogram turns positive AND RSI shows strength
            elif current['histogram'] > 0:
                rsi = current.get('rsi', 50.0)
                if pd.notna(rsi) and rsi > 55.0:  # RSI above 55 confirms strength
                    return True, "Histogram Positive + RSI Strength"
                # Otherwise, wait for stronger confirmation (crossover)
        
        return False, ""
    
    def get_indicator_values(self, df: pd.DataFrame) -> Dict[str, Union[float, bool]]:
        """Get current indicator values for logging"""
        if len(df) == 0:
            return {}
        
        if 'macd' not in df.columns:
            df = self.calculate_indicators(df)
        
        current = df.iloc[-1]
        
        rsi = current.get('rsi', None)
        return {
            'close': current['close'],
            'macd': current['macd'],
            'signal': current['signal'],
            'histogram': current['histogram'],
            'rsi': rsi if pd.notna(rsi) else None,
            'bullish_overlay': current['histogram'] > 0,
            'bearish_overlay': current['histogram'] < 0
        }

