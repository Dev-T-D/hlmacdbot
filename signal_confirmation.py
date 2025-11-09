"""
Signal Confirmation Module

Advanced filtering for trading signals with multi-timeframe confirmation,
volume quality checks, and market regime suitability analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MultiTimeframeConfirmation:
    """Ensure signal alignment across multiple timeframes"""

    def __init__(self, primary_tf='1h', confirmation_tfs=['4h', '1d']):
        self.primary_tf = primary_tf
        self.confirmation_tfs = confirmation_tfs
        self.tf_cache = {}

    def check_alignment(self, symbol, signal_direction, hyperliquid_client):
        """
        Check if signal aligns with higher timeframes

        Args:
            symbol: Trading pair (e.g., 'BNBUSDT')
            signal_direction: 'LONG' or 'SHORT'
            hyperliquid_client: Client to fetch multi-timeframe data

        Returns:
            (aligned: bool, alignment_score: float, details: dict)
        """
        alignments = {}

        for tf in self.confirmation_tfs:
            # Fetch higher timeframe data
            htf_data = self._get_timeframe_data(symbol, tf, hyperliquid_client)

            if htf_data is None:
                continue

            # Check trend direction on higher timeframe
            htf_trend = self._determine_trend(htf_data)

            # Check if aligned
            is_aligned = (htf_trend == signal_direction)
            alignments[tf] = {
                'trend': htf_trend,
                'aligned': is_aligned
            }

        # Calculate alignment score
        if len(alignments) == 0:
            return False, 0.0, {}

        alignment_score = sum(a['aligned'] for a in alignments.values()) / len(alignments)
        all_aligned = all(a['aligned'] for a in alignments.values())

        return all_aligned, alignment_score, alignments

    def _get_timeframe_data(self, symbol, timeframe, client):
        """Fetch and cache timeframe data"""
        cache_key = f"{symbol}_{timeframe}"

        # Check cache (expire after 1 hour)
        if cache_key in self.tf_cache:
            cached_data, timestamp = self.tf_cache[cache_key]
            if (pd.Timestamp.now() - timestamp).total_seconds() < 3600:
                return cached_data

        # Fetch fresh data
        try:
            # Use the client's method to get historical data
            data = client.get_historical_klines(symbol, timeframe, limit=200)
            if not data:
                return None

            df = pd.DataFrame(data)
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Cache it
            self.tf_cache[cache_key] = (df, pd.Timestamp.now())

            return df
        except Exception as e:
            logger.warning(f"Error fetching {timeframe} data: {e}")
            return None

    def _determine_trend(self, df):
        """Determine trend direction from dataframe"""
        # Calculate EMAs
        ema_50 = df['close'].ewm(span=50).mean()
        ema_200 = df['close'].ewm(span=200).mean()

        # Current price vs EMAs
        current_price = df['close'].iloc[-1]
        current_ema_50 = ema_50.iloc[-1]
        current_ema_200 = ema_200.iloc[-1]

        # Trend determination
        if current_price > current_ema_50 > current_ema_200:
            return 'LONG'
        elif current_price < current_ema_50 < current_ema_200:
            return 'SHORT'
        else:
            return 'NEUTRAL'


class VolumeQualityFilter:
    """Filter trades based on volume quality"""

    def check_volume_quality(self, df, min_volume_ratio=1.5):
        """
        Check if current volume supports the signal

        Args:
            df: DataFrame with volume data
            min_volume_ratio: Minimum ratio of current volume to average

        Returns:
            (quality_ok: bool, volume_ratio: float, reason: str)
        """
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]

        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        # FILTER 1: Volume spike required
        if volume_ratio < min_volume_ratio:
            return False, volume_ratio, f"Volume too low ({volume_ratio:.2f}x avg, need {min_volume_ratio}x)"

        # FILTER 2: Check volume trend (increasing volume = stronger signal)
        volume_trend = df['volume'].rolling(5).mean().iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]

        if volume_trend < 0.8:
            return False, volume_ratio, f"Volume trend declining ({volume_trend:.2f})"

        # FILTER 3: No volume anomalies (avoid wash trading)
        volume_std = df['volume'].rolling(20).std().iloc[-1]
        volume_zscore = (current_volume - avg_volume) / volume_std if volume_std > 0 else 0

        if volume_zscore > 5:  # Extreme outlier
            return False, volume_ratio, f"Volume anomaly detected (z-score: {volume_zscore:.2f})"

        return True, volume_ratio, "Volume quality good"


class MarketRegimeFilter:
    """Filter trades based on current market regime"""

    def __init__(self):
        self.favorable_regimes = {
            'LONG': ['BULL_TREND', 'BREAKOUT_PENDING'],
            'SHORT': ['BEAR_TREND', 'HIGH_VOLATILITY']
        }

    def check_regime_suitability(self, current_regime, signal_direction):
        """
        Check if current market regime is suitable for signal

        Args:
            current_regime: Current regime from regime_detector
            signal_direction: 'LONG' or 'SHORT'

        Returns:
            (suitable: bool, reason: str)
        """
        favorable = self.favorable_regimes.get(signal_direction, [])

        if current_regime in favorable:
            return True, f"Favorable regime for {signal_direction}: {current_regime}"

        # RANGING market: Skip all trades (low win rate in ranging)
        if current_regime == 'RANGING':
            return False, "Ranging market - low probability trades"

        # LOW_VOLATILITY: Skip all trades (no movement)
        if current_regime == 'LOW_VOLATILITY':
            return False, "Low volatility - insufficient price movement"

        # Unfavorable regime
        return False, f"Unfavorable regime for {signal_direction}: {current_regime}"
