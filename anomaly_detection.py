"""
Anomaly Detection for Trading Bot

Statistical anomaly detection to identify unusual patterns and potential issues.
Uses multiple detection methods: Z-score, Isolation Forest, and time-series analysis.

Detects anomalies in:
- Trade frequency and volume
- API response times
- Position sizes
- Win/loss patterns
- Market data anomalies
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    IsolationForest = None
    StandardScaler = None

logger = logging.getLogger(__name__)


@dataclass
class AnomalyScore:
    """Anomaly detection result."""
    metric_name: str
    value: float
    score: float  # Anomaly score (higher = more anomalous)
    threshold: float
    is_anomaly: bool
    method: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metric_name': self.metric_name,
            'value': self.value,
            'score': self.score,
            'threshold': self.threshold,
            'is_anomaly': self.is_anomaly,
            'method': self.method,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context
        }


@dataclass
class AnomalyMetrics:
    """Container for various anomaly detection results."""
    trade_frequency_anomaly: Optional[AnomalyScore] = None
    trade_volume_anomaly: Optional[AnomalyScore] = None
    api_latency_anomaly: Optional[AnomalyScore] = None
    position_size_anomaly: Optional[AnomalyScore] = None
    win_rate_anomaly: Optional[AnomalyScore] = None
    pnl_anomaly: Optional[AnomalyScore] = None
    market_data_anomaly: Optional[AnomalyScore] = None

    def get_all_anomalies(self) -> List[AnomalyScore]:
        """Get all detected anomalies."""
        anomalies = []
        for field_name in self.__dataclass_fields__:
            anomaly = getattr(self, field_name)
            if anomaly and anomaly.is_anomaly:
                anomalies.append(anomaly)
        return anomalies

    def has_anomalies(self) -> bool:
        """Check if any anomalies were detected."""
        return len(self.get_all_anomalies()) > 0


class StatisticalAnomalyDetector:
    """
    Statistical anomaly detection using Z-score and moving averages.

    Detects anomalies based on standard deviation from mean.
    """

    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.data_windows: Dict[str, deque] = {}

    def add_data_point(self, metric_name: str, value: float, timestamp: Optional[datetime] = None) -> AnomalyScore:
        """Add data point and check for anomalies."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Initialize window if needed
        if metric_name not in self.data_windows:
            self.data_windows[metric_name] = deque(maxlen=self.window_size)

        # Add value to window
        self.data_windows[metric_name].append(value)

        # Calculate anomaly score
        window = list(self.data_windows[metric_name])
        if len(window) < 3:  # Need minimum data for meaningful statistics
            return AnomalyScore(
                metric_name=metric_name,
                value=value,
                score=0.0,
                threshold=self.z_threshold,
                is_anomaly=False,
                method="z_score",
                timestamp=timestamp
            )

        mean = np.mean(window)
        std = np.std(window)

        if std == 0:
            z_score = 0.0
        else:
            z_score = abs(value - mean) / std

        is_anomaly = z_score > self.z_threshold

        return AnomalyScore(
            metric_name=metric_name,
            value=value,
            score=z_score,
            threshold=self.z_threshold,
            is_anomaly=is_anomaly,
            method="z_score",
            timestamp=timestamp
        )


class IsolationForestDetector:
    """
    Machine learning-based anomaly detection using Isolation Forest.

    Better at detecting complex patterns and multivariate anomalies.
    """

    def __init__(self, window_size: int = 100, contamination: float = 0.1):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for IsolationForest anomaly detection")

        self.window_size = window_size
        self.contamination = contamination
        self.data_windows: Dict[str, deque] = {}
        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}

    def add_data_point(self, metric_name: str, value: float, timestamp: Optional[datetime] = None) -> AnomalyScore:
        """Add data point and check for anomalies using Isolation Forest."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Initialize window and model if needed
        if metric_name not in self.data_windows:
            self.data_windows[metric_name] = deque(maxlen=self.window_size)
            self.models[metric_name] = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            self.scalers[metric_name] = StandardScaler()

        # Add value to window
        self.data_windows[metric_name].append(value)

        # Need enough data to train model
        window = list(self.data_windows[metric_name])
        if len(window) < 10:
            return AnomalyScore(
                metric_name=metric_name,
                value=value,
                score=0.0,
                threshold=0.5,  # Isolation Forest returns -1 for anomalies, 1 for normal
                is_anomaly=False,
                method="isolation_forest",
                timestamp=timestamp
            )

        # Prepare data for model
        X = np.array(window).reshape(-1, 1)
        X_scaled = self.scalers[metric_name].fit_transform(X)

        # Fit model and predict
        self.models[metric_name].fit(X_scaled)
        predictions = self.models[metric_name].predict(X_scaled)

        # Last prediction is for current value
        is_anomaly = predictions[-1] == -1
        anomaly_score = 1.0 if is_anomaly else 0.0  # Simplified score

        return AnomalyScore(
            metric_name=metric_name,
            value=value,
            score=anomaly_score,
            threshold=0.5,
            is_anomaly=is_anomaly,
            method="isolation_forest",
            timestamp=timestamp
        )


class TimeSeriesAnomalyDetector:
    """
    Time-series based anomaly detection using moving averages and trend analysis.
    """

    def __init__(self, window_size: int = 20, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.data_windows: Dict[str, deque] = {}
        self.moving_averages: Dict[str, float] = {}
        self.trends: Dict[str, List[float]] = {}

    def add_data_point(self, metric_name: str, value: float, timestamp: Optional[datetime] = None) -> AnomalyScore:
        """Add data point and detect time-series anomalies."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Initialize tracking
        if metric_name not in self.data_windows:
            self.data_windows[metric_name] = deque(maxlen=self.window_size)
            self.trends[metric_name] = []

        # Add value
        self.data_windows[metric_name].append(value)

        window = list(self.data_windows[metric_name])
        if len(window) < self.window_size:
            return AnomalyScore(
                metric_name=metric_name,
                value=value,
                score=0.0,
                threshold=self.sensitivity,
                is_anomaly=False,
                method="time_series",
                timestamp=timestamp
            )

        # Calculate moving average
        current_ma = np.mean(window)
        self.moving_averages[metric_name] = current_ma

        # Calculate trend (simple linear regression slope)
        x = np.arange(len(window))
        y = np.array(window)
        slope = np.polyfit(x, y, 1)[0]
        self.trends[metric_name].append(slope)

        # Keep only recent trends
        if len(self.trends[metric_name]) > 10:
            self.trends[metric_name] = self.trends[metric_name][-10:]

        # Detect anomalies based on deviation from moving average and trend changes
        deviation = abs(value - current_ma) / current_ma if current_ma != 0 else 0

        # Trend change anomaly (sudden direction change)
        trend_anomaly = False
        if len(self.trends[metric_name]) >= 3:
            recent_trends = self.trends[metric_name][-3:]
            if len(set(np.sign(recent_trends))) > 1:  # Mixed directions
                trend_change = abs(recent_trends[-1] - recent_trends[-2])
                avg_trend = np.mean(recent_trends[:-1])
                if avg_trend != 0:
                    trend_anomaly = trend_change / abs(avg_trend) > 2.0

        # Combined anomaly score
        anomaly_score = deviation
        if trend_anomaly:
            anomaly_score += 1.0

        is_anomaly = anomaly_score > self.sensitivity

        return AnomalyScore(
            metric_name=metric_name,
            value=value,
            score=anomaly_score,
            threshold=self.sensitivity,
            is_anomaly=is_anomaly,
            method="time_series",
            timestamp=timestamp,
            context={
                'moving_average': current_ma,
                'deviation': deviation,
                'trend_anomaly': trend_anomaly
            }
        )


class TradingAnomalyDetector:
    """
    Comprehensive anomaly detection for trading bot operations.

    Combines multiple detection methods to identify unusual patterns in:
    - Trading activity
    - API performance
    - Risk metrics
    - Market data
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.last_check = datetime.now(timezone.utc)

        # Initialize detectors
        self.statistical_detector = StatisticalAnomalyDetector(
            window_size=self.config.get('statistical_window', 100),
            z_threshold=self.config.get('z_threshold', 3.0)
        )

        if SKLEARN_AVAILABLE:
            try:
                self.ml_detector = IsolationForestDetector(
                    window_size=self.config.get('ml_window', 100),
                    contamination=self.config.get('contamination', 0.1)
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ML detector: {e}")
                self.ml_detector = None
        else:
            logger.warning("scikit-learn not available, skipping ML-based anomaly detection")
            self.ml_detector = None

        self.time_series_detector = TimeSeriesAnomalyDetector(
            window_size=self.config.get('time_series_window', 20),
            sensitivity=self.config.get('time_series_sensitivity', 2.0)
        )

        # Historical data for context
        self.recent_trades: deque = deque(maxlen=1000)
        self.api_latencies: deque = deque(maxlen=1000)
        self.position_sizes: deque = deque(maxlen=1000)

        logger.info("Trading anomaly detector initialized")

    def check_trading_anomalies(self, **metrics) -> AnomalyMetrics:
        """
        Check for anomalies in trading-related metrics.

        Args:
            metrics: Dictionary containing various trading metrics
        """
        current_time = datetime.now(timezone.utc)
        results = AnomalyMetrics()

        # Trade frequency anomaly
        trades_per_minute = metrics.get('trades_per_minute', 0)
        if trades_per_minute > 0:
            results.trade_frequency_anomaly = self.statistical_detector.add_data_point(
                'trades_per_minute', trades_per_minute, current_time
            )

        # Trade volume anomaly
        trade_volume = metrics.get('trade_volume', 0)
        if trade_volume > 0:
            results.trade_volume_anomaly = self.statistical_detector.add_data_point(
                'trade_volume', trade_volume, current_time
            )

        # API latency anomaly
        api_latency = metrics.get('api_latency_seconds', 0)
        if api_latency > 0:
            results.api_latency_anomaly = self.time_series_detector.add_data_point(
                'api_latency', api_latency, current_time
            )

        # Position size anomaly
        position_size = metrics.get('position_size_percent', 0)
        if position_size > 0:
            results.position_size_anomaly = self.statistical_detector.add_data_point(
                'position_size_percent', position_size, current_time
            )

        # Win rate anomaly
        win_rate = metrics.get('win_rate', 0)
        if 0 <= win_rate <= 1:
            results.win_rate_anomaly = self.statistical_detector.add_data_point(
                'win_rate', win_rate, current_time
            )

        # P&L anomaly
        pnl = metrics.get('pnl', 0)
        if pnl != 0:
            results.pnl_anomaly = self.time_series_detector.add_data_point(
                'pnl', pnl, current_time
            )

        return results

    def check_market_data_anomalies(self, price: float, volume: float, symbol: str) -> Optional[AnomalyScore]:
        """Check for anomalies in market data."""
        # Price anomaly detection
        price_anomaly = self.statistical_detector.add_data_point(
            f'price_{symbol}', price, datetime.now(timezone.utc)
        )

        # Volume anomaly detection
        volume_anomaly = self.statistical_detector.add_data_point(
            f'volume_{symbol}', volume, datetime.now(timezone.utc)
        )

        # Return the more anomalous of the two
        if price_anomaly.score > volume_anomaly.score:
            price_anomaly.metric_name = f'market_data_{symbol}'
            price_anomaly.context.update({'data_type': 'price', 'volume': volume})
            return price_anomaly if price_anomaly.is_anomaly else None
        else:
            volume_anomaly.metric_name = f'market_data_{symbol}'
            volume_anomaly.context.update({'data_type': 'volume', 'price': price})
            return volume_anomaly if volume_anomaly.is_anomaly else None

    def detect_trade_streaks(self, recent_trades: List[Dict[str, Any]], min_streak: int = 5) -> Optional[AnomalyScore]:
        """
        Detect unusual winning or losing streaks.

        Args:
            recent_trades: List of recent trade dictionaries with 'pnl' field
            min_streak: Minimum streak length to consider anomalous
        """
        if len(recent_trades) < min_streak:
            return None

        # Extract P&L outcomes
        outcomes = [1 if trade.get('pnl', 0) > 0 else -1 for trade in recent_trades[-20:]]  # Last 20 trades

        # Find current streak
        if not outcomes:
            return None

        current_outcome = outcomes[-1]
        streak_length = 0

        for outcome in reversed(outcomes):
            if outcome == current_outcome:
                streak_length += 1
            else:
                break

        # Check if streak is anomalous
        if streak_length >= min_streak:
            # Calculate expected streak probability (simplified)
            win_rate = sum(1 for o in outcomes if o > 0) / len(outcomes)
            expected_prob = win_rate if current_outcome > 0 else (1 - win_rate)
            anomaly_score = streak_length * (1 - expected_prob)  # Higher score = more anomalous

            return AnomalyScore(
                metric_name='trade_streak',
                value=streak_length,
                score=anomaly_score,
                threshold=min_streak,
                is_anomaly=anomaly_score > min_streak * 0.5,
                method='streak_analysis',
                timestamp=datetime.now(timezone.utc),
                context={
                    'streak_type': 'winning' if current_outcome > 0 else 'losing',
                    'win_rate': win_rate,
                    'expected_probability': expected_prob
                }
            )

        return None

    def detect_api_error_spikes(self, error_rate: float, time_window_minutes: int = 5) -> Optional[AnomalyScore]:
        """Detect spikes in API error rates."""
        return self.time_series_detector.add_data_point(
            'api_error_rate', error_rate, datetime.now(timezone.utc)
        )

    def detect_position_concentration(self, positions: List[Dict[str, Any]]) -> Optional[AnomalyScore]:
        """
        Detect unusual concentration in position sizes.

        Args:
            positions: List of position dictionaries with 'size' and 'symbol' fields
        """
        if not positions:
            return None

        total_exposure = sum(abs(pos.get('size', 0)) for pos in positions)
        if total_exposure == 0:
            return None

        # Calculate concentration (largest position / total exposure)
        position_sizes = [abs(pos.get('size', 0)) for pos in positions]
        max_position = max(position_sizes)
        concentration = max_position / total_exposure

        # Check for anomalous concentration
        concentration_anomaly = self.statistical_detector.add_data_point(
            'position_concentration', concentration, datetime.now(timezone.utc)
        )

        if concentration_anomaly.is_anomaly:
            concentration_anomaly.context.update({
                'max_position_size': max_position,
                'total_exposure': total_exposure,
                'position_count': len(positions)
            })

        return concentration_anomaly if concentration_anomaly.is_anomaly else None

    def get_anomaly_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of anomalies detected in the last N hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # This would need to be implemented with proper storage
        # For now, return basic structure
        return {
            'time_range': f'{hours} hours',
            'total_anomalies': 0,
            'anomaly_types': {},
            'severity_distribution': {},
            'most_common_anomaly': None,
            'recent_anomalies': []
        }

    def update_historical_data(self, trades: List[Dict[str, Any]] = None,
                             api_latencies: List[float] = None,
                             position_sizes: List[float] = None):
        """Update historical data for better anomaly detection."""
        if trades:
            self.recent_trades.extend(trades)

        if api_latencies:
            self.api_latencies.extend(api_latencies)

        if position_sizes:
            self.position_sizes.extend(position_sizes)

    def reset_detectors(self):
        """Reset all anomaly detectors (useful for testing or after configuration changes)."""
        self.statistical_detector = StatisticalAnomalyDetector(
            window_size=self.config.get('statistical_window', 100),
            z_threshold=self.config.get('z_threshold', 3.0)
        )

        self.time_series_detector = TimeSeriesAnomalyDetector(
            window_size=self.config.get('time_series_window', 20),
            sensitivity=self.config.get('time_series_sensitivity', 2.0)
        )

        if self.ml_detector:
            self.ml_detector = IsolationForestDetector(
                window_size=self.config.get('ml_window', 100),
                contamination=self.config.get('contamination', 0.1)
            )

        logger.info("Anomaly detectors reset")


# Global anomaly detector instance
_anomaly_detector: Optional[TradingAnomalyDetector] = None


def get_anomaly_detector() -> TradingAnomalyDetector:
    """Get global anomaly detector instance."""
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = TradingAnomalyDetector()
    return _anomaly_detector


def initialize_anomaly_detector(config: Optional[Dict[str, Any]] = None) -> TradingAnomalyDetector:
    """Initialize global anomaly detector."""
    global _anomaly_detector
    _anomaly_detector = TradingAnomalyDetector(config)
    return _anomaly_detector


# Convenience functions for common anomaly checks
def check_trading_anomalies(**metrics) -> AnomalyMetrics:
    """Convenience function to check trading anomalies."""
    return get_anomaly_detector().check_trading_anomalies(**metrics)


def check_market_data_anomalies(price: float, volume: float, symbol: str) -> Optional[AnomalyScore]:
    """Convenience function to check market data anomalies."""
    return get_anomaly_detector().check_market_data_anomalies(price, volume, symbol)


def detect_trade_streaks(recent_trades: List[Dict[str, Any]], min_streak: int = 5) -> Optional[AnomalyScore]:
    """Convenience function to detect trade streaks."""
    return get_anomaly_detector().detect_trade_streaks(recent_trades, min_streak)


def detect_api_error_spikes(error_rate: float, time_window_minutes: int = 5) -> Optional[AnomalyScore]:
    """Convenience function to detect API error spikes."""
    return get_anomaly_detector().detect_api_error_spikes(error_rate, time_window_minutes)


def detect_position_concentration(positions: List[Dict[str, Any]]) -> Optional[AnomalyScore]:
    """Convenience function to detect position concentration anomalies."""
    return get_anomaly_detector().detect_position_concentration(positions)
