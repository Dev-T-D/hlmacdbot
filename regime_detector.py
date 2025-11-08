"""
Market Regime Detection System

This module implements comprehensive market regime detection using multiple technical
indicators and statistical methods to classify current market conditions and adapt
trading strategies accordingly.

Key Features:
- 6 distinct regime classifications (Bull/Bear Trend, Ranging, High/Low Volatility, Breakout)
- Multi-timeframe regime analysis
- Hidden Markov Model for regime prediction
- GARCH volatility forecasting
- Regime persistence tracking
- Statistical regime validation
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

# Optional imports
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Enumeration of market regimes."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT_PENDING = "breakout_pending"
    NEUTRAL = "neutral"


@dataclass
class RegimeMetrics:
    """Metrics for regime classification."""
    trend_strength: float = 0.0
    volatility_level: float = 0.0
    range_bound_score: float = 0.0
    breakout_probability: float = 0.0
    momentum_score: float = 0.0
    mean_reversion_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'trend_strength': self.trend_strength,
            'volatility_level': self.volatility_level,
            'range_bound_score': self.range_bound_score,
            'breakout_probability': self.breakout_probability,
            'momentum_score': self.momentum_score,
            'mean_reversion_score': self.mean_reversion_score
        }


@dataclass
class RegimeAnalysis:
    """Complete regime analysis result."""
    primary_regime: MarketRegime
    confidence_score: float
    secondary_regime: Optional[MarketRegime] = None
    transition_probability: float = 0.0
    persistence_score: float = 0.0
    metrics: RegimeMetrics = field(default_factory=RegimeMetrics)
    timeframe_alignment: str = "unknown"
    volatility_forecast: Optional[float] = None
    regime_duration: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'primary_regime': self.primary_regime.value,
            'confidence_score': self.confidence_score,
            'secondary_regime': self.secondary_regime.value if self.secondary_regime else None,
            'transition_probability': self.transition_probability,
            'persistence_score': self.persistence_score,
            'metrics': self.metrics.to_dict(),
            'timeframe_alignment': self.timeframe_alignment,
            'volatility_forecast': self.volatility_forecast,
            'regime_duration': self.regime_duration
        }


class VolatilityForecaster:
    """GARCH-based volatility forecasting."""

    def __init__(self):
        """Initialize volatility forecaster."""
        self.fitted_model = None
        self.last_forecast = None
        self.forecast_timestamp = None

    def fit_model(self, returns: np.ndarray, p: int = 1, q: int = 1) -> bool:
        """
        Fit GARCH model to returns data.

        Args:
            returns: Array of returns
            p: GARCH p parameter
            q: GARCH q parameter

        Returns:
            True if model fitted successfully
        """
        if not ARCH_AVAILABLE:
            logger.warning("ARCH library not available - volatility forecasting disabled")
            return False

        try:
            # Ensure we have enough data
            if len(returns) < 50:
                logger.warning("Insufficient data for GARCH model fitting")
                return False

            # Fit GARCH(1,1) model
            model = arch_model(returns, vol='Garch', p=p, q=q, rescale=False)
            self.fitted_model = model.fit(disp='off')

            logger.info(f"GARCH({p},{q}) model fitted successfully")
            return True

        except Exception as e:
            logger.error(f"Error fitting GARCH model: {e}")
            return False

    def forecast_volatility(self, horizon: int = 1) -> Optional[float]:
        """
        Forecast future volatility.

        Args:
            horizon: Number of periods to forecast

        Returns:
            Forecasted volatility (annualized if daily data)
        """
        if self.fitted_model is None:
            return None

        try:
            # Generate forecast
            forecast = self.fitted_model.forecast(horizon=horizon)

            # Extract variance forecast and convert to volatility
            variance_forecast = forecast.variance.values[-1, 0]
            volatility_forecast = np.sqrt(variance_forecast)

            # Annualize if using daily returns (assuming 252 trading days)
            if horizon == 1:
                volatility_forecast *= np.sqrt(252)

            self.last_forecast = volatility_forecast
            self.forecast_timestamp = datetime.now()

            return volatility_forecast

        except Exception as e:
            logger.error(f"Error forecasting volatility: {e}")
            return None


class HMMRegimeDetector:
    """Hidden Markov Model for regime detection."""

    def __init__(self, n_states: int = 3):
        """
        Initialize HMM regime detector.

        Args:
            n_states: Number of hidden states (regimes)
        """
        self.n_states = n_states
        self.model = None
        self.state_labels = {}  # Map HMM states to market regimes

    def fit(self, returns: np.ndarray, n_iter: int = 100) -> bool:
        """
        Fit HMM to returns data.

        Args:
            returns: Array of returns
            n_iter: Maximum iterations for fitting

        Returns:
            True if model fitted successfully
        """
        if not HMMLEARN_AVAILABLE:
            logger.warning("hmmlearn not available - HMM regime detection disabled")
            return False

        try:
            if len(returns) < 100:
                logger.warning("Insufficient data for HMM fitting")
                return False

            # Initialize and fit HMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=n_iter,
                random_state=42
            )

            # Fit the model
            self.model.fit(returns.reshape(-1, 1))

            # Label states based on their mean returns
            state_means = []
            for state in range(self.n_states):
                state_data = returns[self.model.predict(returns.reshape(-1, 1)) == state]
                if len(state_data) > 0:
                    state_means.append(np.mean(state_data))
                else:
                    state_means.append(0.0)

            # Sort states by mean return and assign regime labels
            sorted_indices = np.argsort(state_means)
            self.state_labels = {
                sorted_indices[0]: MarketRegime.BEAR_TREND,  # Lowest returns
                sorted_indices[-1]: MarketRegime.BULL_TREND,  # Highest returns
            }

            # Middle state(s) for ranging
            for i in range(1, self.n_states - 1):
                self.state_labels[sorted_indices[i]] = MarketRegime.RANGING

            logger.info(f"HMM fitted with {self.n_states} states")
            return True

        except Exception as e:
            logger.error(f"Error fitting HMM: {e}")
            return False

    def predict_regime(self, returns: np.ndarray) -> Optional[MarketRegime]:
        """
        Predict current regime using HMM.

        Args:
            returns: Recent returns data

        Returns:
            Predicted regime or None if prediction fails
        """
        if self.model is None or not returns.size:
            return None

        try:
            # Predict state for most recent data
            states = self.model.predict(returns.reshape(-1, 1))
            current_state = states[-1]

            return self.state_labels.get(current_state, MarketRegime.NEUTRAL)

        except Exception as e:
            logger.debug(f"Error predicting regime with HMM: {e}")
            return None


class RegimePersistenceTracker:
    """Track regime persistence and transitions."""

    def __init__(self, min_persistence: int = 3, max_history: int = 1000):
        """
        Initialize persistence tracker.

        Args:
            min_persistence: Minimum consecutive observations for regime confirmation
            max_history: Maximum history to keep
        """
        self.min_persistence = min_persistence
        self.max_history = max_history
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []
        self.transition_matrix: Dict[MarketRegime, Dict[MarketRegime, int]] = defaultdict(lambda: defaultdict(int))

    def add_observation(self, regime: MarketRegime, timestamp: Optional[datetime] = None) -> None:
        """
        Add regime observation to history.

        Args:
            regime: Detected regime
            timestamp: Observation timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Add to history
        self.regime_history.append((timestamp, regime))

        # Maintain history size
        if len(self.regime_history) > self.max_history:
            self.regime_history = self.regime_history[-self.max_history:]

        # Update transition matrix
        if len(self.regime_history) >= 2:
            prev_regime = self.regime_history[-2][1]
            self.transition_matrix[prev_regime][regime] += 1

    def get_current_regime_persistence(self) -> Tuple[MarketRegime, int]:
        """
        Get current regime and its persistence.

        Returns:
            Tuple of (current_regime, persistence_count)
        """
        if not self.regime_history:
            return MarketRegime.NEUTRAL, 0

        current_regime = self.regime_history[-1][1]
        persistence = 0

        # Count consecutive occurrences
        for timestamp, regime in reversed(self.regime_history):
            if regime == current_regime:
                persistence += 1
            else:
                break

        return current_regime, persistence

    def should_confirm_regime(self, regime: MarketRegime) -> bool:
        """
        Check if regime should be confirmed based on persistence.

        Args:
            regime: Regime to check

        Returns:
            True if regime has sufficient persistence
        """
        current_regime, persistence = self.get_current_regime_persistence()
        return current_regime == regime and persistence >= self.min_persistence

    def get_transition_probability(self, from_regime: MarketRegime,
                                 to_regime: MarketRegime) -> float:
        """
        Get transition probability between regimes.

        Args:
            from_regime: Source regime
            to_regime: Target regime

        Returns:
            Transition probability (0-1)
        """
        if from_regime not in self.transition_matrix:
            return 0.0

        total_transitions = sum(self.transition_matrix[from_regime].values())
        if total_transitions == 0:
            return 0.0

        return self.transition_matrix[from_regime][to_regime] / total_transitions

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get regime transition statistics."""
        total_observations = len(self.regime_history)

        if total_observations < 2:
            return {'total_observations': total_observations}

        # Calculate regime frequencies
        regime_counts = defaultdict(int)
        for _, regime in self.regime_history:
            regime_counts[regime] += 1

        regime_frequencies = {
            regime.value: count / total_observations
            for regime, count in regime_counts.items()
        }

        # Calculate average persistence
        persistence_lengths = []
        current_persistence = 1

        for i in range(1, len(self.regime_history)):
            if self.regime_history[i][1] == self.regime_history[i-1][1]:
                current_persistence += 1
            else:
                persistence_lengths.append(current_persistence)
                current_persistence = 1

        persistence_lengths.append(current_persistence)
        avg_persistence = np.mean(persistence_lengths) if persistence_lengths else 0

        return {
            'total_observations': total_observations,
            'regime_frequencies': regime_frequencies,
            'avg_regime_persistence': avg_persistence,
            'most_common_regime': max(regime_counts, key=regime_counts.get).value if regime_counts else None
        }


class RegimeDetector:
    """
    Comprehensive market regime detection system.

    This class combines multiple technical indicators and statistical methods
    to classify current market conditions and provide regime-based trading guidance.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize regime detector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()

        # Initialize components
        self.volatility_forecaster = VolatilityForecaster()
        self.hmm_detector = HMMRegimeDetector(n_states=self.config['hmm']['n_states'])
        self.persistence_tracker = RegimePersistenceTracker(
            min_persistence=self.config['persistence']['min_periods'],
            max_history=self.config['persistence']['max_history']
        )

        # Regime detection parameters
        self.trend_threshold = self.config['thresholds']['trend_strength']
        self.volatility_threshold_high = self.config['thresholds']['volatility_high']
        self.volatility_threshold_low = self.config['thresholds']['volatility_low']
        self.range_bound_adx_threshold = self.config['thresholds']['range_bound_adx']
        self.breakout_percentile = self.config['thresholds']['breakout_bb_percentile']

        # Historical data for analysis
        self.price_history: List[pd.DataFrame] = []
        self.max_price_history = 1000

        # Regime transition tracking
        self.last_regime = MarketRegime.NEUTRAL
        self.regime_start_time = datetime.now()

        logger.info("RegimeDetector initialized")

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'thresholds': {
                'trend_strength': 25,  # ADX threshold for trending
                'volatility_high': 1.5,  # ATR multiplier for high volatility
                'volatility_low': 0.7,   # ATR multiplier for low volatility
                'range_bound_adx': 20,   # ADX threshold for ranging
                'breakout_bb_percentile': 20  # BB width percentile for breakout
            },
            'hmm': {
                'n_states': 3,
                'enabled': True
            },
            'persistence': {
                'min_periods': 3,  # Minimum consecutive periods for confirmation
                'max_history': 1000
            },
            'multi_timeframe': {
                'enabled': True,
                'timeframes': ['5m', '1h', '4h']
            },
            'volatility_forecast': {
                'enabled': True,
                'horizon': 1,
                'garch_p': 1,
                'garch_q': 1
            }
        }

    def detect_regime(self, df: pd.DataFrame, symbol: str = "unknown") -> RegimeAnalysis:
        """
        Detect current market regime from price data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol (for logging)

        Returns:
            Complete regime analysis
        """
        try:
            if df.empty or len(df) < 50:
                return RegimeAnalysis(
                    primary_regime=MarketRegime.NEUTRAL,
                    confidence_score=0.0,
                    metrics=RegimeMetrics()
                )

            # Calculate regime metrics
            metrics = self._calculate_regime_metrics(df)

            # Determine primary regime
            primary_regime = self._classify_regime(metrics)

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(metrics, primary_regime)

            # Check secondary regime
            secondary_regime = self._detect_secondary_regime(metrics, primary_regime)

            # Calculate transition probability
            transition_prob = self.persistence_tracker.get_transition_probability(
                self.last_regime, primary_regime
            )

            # Get persistence score
            _, persistence = self.persistence_tracker.get_current_regime_persistence()
            persistence_score = min(persistence / self.config['persistence']['min_periods'], 1.0)

            # HMM regime prediction
            hmm_regime = None
            if self.config['hmm']['enabled'] and self.hmm_detector.model is not None:
                returns = df['close'].pct_change().dropna().values[-50:]  # Last 50 periods
                if len(returns) >= 10:
                    hmm_regime = self.hmm_detector.predict_regime(returns)

            # Use HMM as secondary if available and different
            if hmm_regime and hmm_regime != primary_regime:
                secondary_regime = hmm_regime

            # Get volatility forecast
            volatility_forecast = None
            if self.config['volatility_forecast']['enabled']:
                returns = df['close'].pct_change().dropna().values
                if len(returns) >= 100:
                    # Fit model if not already fitted
                    if self.volatility_forecaster.fitted_model is None:
                        self.volatility_forecaster.fit_model(returns)

                    # Get forecast
                    volatility_forecast = self.volatility_forecaster.forecast_volatility(
                        horizon=self.config['volatility_forecast']['horizon']
                    )

            # Calculate regime duration
            regime_duration = 0
            if primary_regime == self.last_regime:
                regime_duration = int((datetime.now() - self.regime_start_time).total_seconds() / 60)
            else:
                self.last_regime = primary_regime
                self.regime_start_time = datetime.now()

            # Add to history for persistence tracking
            self.persistence_tracker.add_observation(primary_regime)

            # Store price data
            self.price_history.append(df.copy())
            if len(self.price_history) > self.max_price_history:
                self.price_history = self.price_history[-self.max_price_history:]

            analysis = RegimeAnalysis(
                primary_regime=primary_regime,
                confidence_score=confidence_score,
                secondary_regime=secondary_regime,
                transition_probability=transition_prob,
                persistence_score=persistence_score,
                metrics=metrics,
                timeframe_alignment="single",  # Will be updated with multi-timeframe
                volatility_forecast=volatility_forecast,
                regime_duration=regime_duration
            )

            logger.debug(f"Regime detected for {symbol}: {primary_regime.value} "
                        ".3f"
            return analysis

        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return RegimeAnalysis(
                primary_regime=MarketRegime.NEUTRAL,
                confidence_score=0.0,
                metrics=RegimeMetrics()
            )

    def _calculate_regime_metrics(self, df: pd.DataFrame) -> RegimeMetrics:
        """Calculate comprehensive regime metrics."""
        try:
            metrics = RegimeMetrics()

            # Trend strength (ADX)
            metrics.trend_strength = self._calculate_adx(df)

            # Volatility level (normalized ATR)
            current_atr = self._calculate_atr(df)
            avg_atr = df.get('atr', pd.Series()).rolling(50).mean().iloc[-1] if 'atr' in df.columns else current_atr
            metrics.volatility_level = current_atr / avg_atr if avg_atr > 0 else 1.0

            # Range-bound score (inverse of ADX)
            metrics.range_bound_score = 1.0 - (metrics.trend_strength / 50.0)

            # Breakout probability (Bollinger Band squeeze)
            bb_width = self._calculate_bb_width(df)
            bb_width_percentile = self._calculate_bb_width_percentile(df, bb_width)
            metrics.breakout_probability = 1.0 - (bb_width_percentile / 100.0)

            # Momentum score
            momentum = df['close'].pct_change(20).iloc[-1]
            metrics.momentum_score = momentum

            # Mean reversion score (distance from moving averages)
            ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
            ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
            current_price = df['close'].iloc[-1]

            ma_distance = abs(current_price - (ema_20 + ema_50) / 2) / current_price
            metrics.mean_reversion_score = 1.0 - ma_distance

            return metrics

        except Exception as e:
            logger.debug(f"Error calculating regime metrics: {e}")
            return RegimeMetrics()

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index."""
        try:
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

            return adx.iloc[-1] if not adx.empty else 25.0

        except Exception as e:
            logger.debug(f"Error calculating ADX: {e}")
            return 25.0

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            high = df['high']
            low = df['low']
            close = df['close']

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr = tr.rolling(window=period).mean()
            return atr.iloc[-1] if not atr.empty else df['close'].iloc[-1] * 0.02

        except Exception as e:
            logger.debug(f"Error calculating ATR: {e}")
            return df['close'].iloc[-1] * 0.02

    def _calculate_bb_width(self, df: pd.DataFrame, period: int = 20, std: float = 2.0) -> float:
        """Calculate Bollinger Band width."""
        try:
            close = df['close']
            sma = close.rolling(window=period).mean()
            std_dev = close.rolling(window=period).std()
            upper = sma + (std_dev * std)
            lower = sma - (std_dev * std)

            width = (upper - lower) / sma
            return width.iloc[-1] if not width.empty else 0.05

        except Exception as e:
            logger.debug(f"Error calculating BB width: {e}")
            return 0.05

    def _calculate_bb_width_percentile(self, df: pd.DataFrame, current_width: float) -> float:
        """Calculate percentile of current BB width vs historical."""
        try:
            # Calculate historical BB widths
            widths = []
            for i in range(20, len(df)):
                window = df.iloc[i-20:i]
                width = self._calculate_bb_width(window)
                widths.append(width)

            if not widths:
                return 50.0

            # Calculate percentile
            percentile = np.sum(np.array(widths) <= current_width) / len(widths) * 100
            return percentile

        except Exception as e:
            logger.debug(f"Error calculating BB width percentile: {e}")
            return 50.0

    def _classify_regime(self, metrics: RegimeMetrics) -> MarketRegime:
        """Classify market regime based on metrics."""
        try:
            current_price = 0  # Would need price data, simplified for now

            # Trend regimes
            if metrics.trend_strength > self.trend_threshold:
                # Determine direction based on momentum
                if metrics.momentum_score > 0.01:  # Positive momentum
                    return MarketRegime.BULL_TREND
                elif metrics.momentum_score < -0.01:  # Negative momentum
                    return MarketRegime.BEAR_TREND

            # Range-bound regime
            if metrics.trend_strength < self.range_bound_adx_threshold:
                return MarketRegime.RANGING

            # Volatility regimes
            if metrics.volatility_level > self.volatility_threshold_high:
                return MarketRegime.HIGH_VOLATILITY
            elif metrics.volatility_level < self.volatility_threshold_low:
                return MarketRegime.LOW_VOLATILITY

            # Breakout pending
            if metrics.breakout_probability > (self.breakout_percentile / 100.0):
                return MarketRegime.BREAKOUT_PENDING

            return MarketRegime.NEUTRAL

        except Exception as e:
            logger.debug(f"Error classifying regime: {e}")
            return MarketRegime.NEUTRAL

    def _calculate_confidence_score(self, metrics: RegimeMetrics, regime: MarketRegime) -> float:
        """Calculate confidence score for regime classification."""
        try:
            base_confidence = 0.5  # Base confidence

            # Add confidence based on metric strength
            if regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
                base_confidence += metrics.trend_strength / 100.0
                base_confidence += abs(metrics.momentum_score) * 10

            elif regime == MarketRegime.RANGING:
                base_confidence += metrics.range_bound_score

            elif regime == MarketRegime.HIGH_VOLATILITY:
                base_confidence += (metrics.volatility_level - 1.0) / 2.0

            elif regime == MarketRegime.LOW_VOLATILITY:
                base_confidence += (1.0 - metrics.volatility_level) / 2.0

            elif regime == MarketRegime.BREAKOUT_PENDING:
                base_confidence += metrics.breakout_probability

            return min(base_confidence, 1.0)

        except Exception as e:
            logger.debug(f"Error calculating confidence score: {e}")
            return 0.5

    def _detect_secondary_regime(self, metrics: RegimeMetrics,
                               primary_regime: MarketRegime) -> Optional[MarketRegime]:
        """Detect secondary regime characteristics."""
        try:
            # Check for secondary characteristics
            if primary_regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
                # Check volatility
                if metrics.volatility_level > self.volatility_threshold_high:
                    return MarketRegime.HIGH_VOLATILITY
                elif metrics.volatility_level < self.volatility_threshold_low:
                    return MarketRegime.LOW_VOLATILITY

            elif primary_regime == MarketRegime.RANGING:
                # Check for breakout potential
                if metrics.breakout_probability > 0.7:
                    return MarketRegime.BREAKOUT_PENDING

            return None

        except Exception as e:
            logger.debug(f"Error detecting secondary regime: {e}")
            return None

    def detect_multi_timeframe_regime(self, timeframes_data: Dict[str, pd.DataFrame],
                                    symbol: str = "unknown") -> Tuple[RegimeAnalysis, str]:
        """
        Detect regime across multiple timeframes.

        Args:
            timeframes_data: Dictionary of timeframe -> DataFrame
            symbol: Trading symbol

        Returns:
            Tuple of (analysis, alignment_description)
        """
        try:
            if not self.config['multi_timeframe']['enabled']:
                # Fall back to single timeframe
                primary_tf = self.config['multi_timeframe']['timeframes'][0]
                df = timeframes_data.get(primary_tf)
                if df is not None:
                    analysis = self.detect_regime(df, symbol)
                    return analysis, "single_timeframe"

            regime_votes = defaultdict(int)
            analyses = {}

            # Detect regime for each timeframe
            for timeframe, df in timeframes_data.items():
                if df is not None and len(df) >= 50:
                    analysis = self.detect_regime(df, f"{symbol}_{timeframe}")
                    analyses[timeframe] = analysis
                    regime_votes[analysis.primary_regime] += 1

            # Find consensus regime
            if regime_votes:
                consensus_regime = max(regime_votes, key=regime_votes.get)
                consensus_votes = regime_votes[consensus_regime]
                total_votes = sum(regime_votes.values())

                # Determine alignment level
                if consensus_votes == total_votes:
                    alignment = "perfect"
                    confidence_multiplier = 1.2
                elif consensus_votes >= total_votes * 0.7:
                    alignment = "strong"
                    confidence_multiplier = 1.1
                elif consensus_votes >= total_votes * 0.5:
                    alignment = "moderate"
                    confidence_multiplier = 1.0
                else:
                    alignment = "weak"
                    confidence_multiplier = 0.8

                # Use the shortest timeframe analysis as base
                primary_tf = min(analyses.keys(),
                               key=lambda x: self._timeframe_to_minutes(x))
                base_analysis = analyses[primary_tf]

                # Adjust confidence based on alignment
                adjusted_confidence = min(base_analysis.confidence_score * confidence_multiplier, 1.0)

                # Create multi-timeframe analysis
                multi_analysis = RegimeAnalysis(
                    primary_regime=consensus_regime,
                    confidence_score=adjusted_confidence,
                    secondary_regime=base_analysis.secondary_regime,
                    transition_probability=base_analysis.transition_probability,
                    persistence_score=base_analysis.persistence_score,
                    metrics=base_analysis.metrics,
                    timeframe_alignment=alignment,
                    volatility_forecast=base_analysis.volatility_forecast,
                    regime_duration=base_analysis.regime_duration
                )

                return multi_analysis, alignment
            else:
                # No valid analyses
                return RegimeAnalysis(
                    primary_regime=MarketRegime.NEUTRAL,
                    confidence_score=0.0
                ), "no_data"

        except Exception as e:
            logger.error(f"Error in multi-timeframe regime detection: {e}")
            return RegimeAnalysis(
                primary_regime=MarketRegime.NEUTRAL,
                confidence_score=0.0
            ), "error"

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        timeframe_map = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720, '1d': 1440
        }
        return timeframe_map.get(timeframe, 60)

    def fit_historical_models(self, df: pd.DataFrame) -> bool:
        """
        Fit statistical models using historical data.

        Args:
            df: Historical price data

        Returns:
            True if models fitted successfully
        """
        try:
            returns = df['close'].pct_change().dropna().values

            if len(returns) < 200:
                logger.warning("Insufficient historical data for model fitting")
                return False

            # Fit volatility model
            vol_success = self.volatility_forecaster.fit_model(returns)

            # Fit HMM model
            hmm_success = self.hmm_detector.fit(returns)

            success_count = sum([vol_success, hmm_success])
            logger.info(f"Successfully fitted {success_count}/2 statistical models")

            return success_count > 0

        except Exception as e:
            logger.error(f"Error fitting historical models: {e}")
            return False

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get comprehensive regime detection statistics."""
        return {
            'persistence_stats': self.persistence_tracker.get_regime_statistics(),
            'volatility_forecast_available': self.volatility_forecaster.fitted_model is not None,
            'hmm_available': self.hmm_detector.model is not None,
            'price_history_length': len(self.price_history),
            'current_regime': self.last_regime.value,
            'regime_duration_minutes': int((datetime.now() - self.regime_start_time).total_seconds() / 60)
        }

    def reset(self) -> None:
        """Reset regime detector state."""
        self.price_history.clear()
        self.persistence_tracker = RegimePersistenceTracker(
            min_persistence=self.config['persistence']['min_periods'],
            max_history=self.config['persistence']['max_history']
        )
        self.last_regime = MarketRegime.NEUTRAL
        self.regime_start_time = datetime.now()

        logger.info("Regime detector reset")
