"""
Maximum Adverse Excursion (MAE) Analyzer

This module analyzes the maximum adverse excursion for optimal stop-loss placement.
MAE measures how far price moves against a position before the trade becomes profitable.

Key Features:
- Track MAE for each trade in real-time
- Calculate optimal stop-loss distance based on historical MAE
- Dynamic stop-loss adjustment based on MAE patterns
- Statistical analysis of MAE distributions
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MAERecord:
    """Record of Maximum Adverse Excursion for a single trade."""
    trade_id: str
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    max_adverse_excursion: float = 0.0  # Maximum % move against position
    max_adverse_price: float = 0.0      # Price at maximum excursion
    mae_time: Optional[datetime] = None   # When MAE occurred
    final_pnl_pct: float = 0.0
    was_winner: bool = False
    duration_minutes: float = 0.0

    def update_mae(self, current_price: float, current_time: datetime) -> None:
        """Update MAE calculation with current price."""
        if self.side == 'LONG':
            # For long positions, adverse moves are downward
            adverse_move = (self.entry_price - current_price) / self.entry_price
        else:
            # For short positions, adverse moves are upward
            adverse_move = (current_price - self.entry_price) / self.entry_price

        # Only track adverse moves (positive values)
        adverse_move = max(0, adverse_move)

        if adverse_move > self.max_adverse_excursion:
            self.max_adverse_excursion = adverse_move
            self.max_adverse_price = current_price
            self.mae_time = current_time

    def finalize_trade(self, exit_price: float, exit_time: datetime) -> None:
        """Finalize trade record with exit details."""
        self.exit_price = exit_price
        self.exit_time = exit_time

        if self.side == 'LONG':
            self.final_pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            self.final_pnl_pct = (self.entry_price - exit_price) / self.entry_price

        self.was_winner = self.final_pnl_pct > 0

        if self.entry_time and self.exit_time:
            self.duration_minutes = (self.exit_time - self.entry_time).total_seconds() / 60

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'max_adverse_excursion': self.max_adverse_excursion,
            'max_adverse_price': self.max_adverse_price,
            'mae_time': self.mae_time.isoformat() if self.mae_time else None,
            'final_pnl_pct': self.final_pnl_pct,
            'was_winner': self.was_winner,
            'duration_minutes': self.duration_minutes
        }


@dataclass
class MAEAnalysis:
    """Statistical analysis of MAE data."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # MAE statistics
    avg_mae_all: float = 0.0
    avg_mae_winners: float = 0.0
    avg_mae_losers: float = 0.0
    median_mae: float = 0.0
    mae_75th_percentile: float = 0.0
    mae_90th_percentile: float = 0.0
    mae_95th_percentile: float = 0.0

    # Optimal stop distances
    recommended_stop_pct: float = 0.0  # Based on 75th percentile + buffer
    conservative_stop_pct: float = 0.0  # Based on 90th percentile + buffer
    tight_stop_pct: float = 0.0        # Based on 50th percentile

    # MAE distribution
    mae_histogram_bins: List[float] = field(default_factory=list)
    mae_histogram_counts: List[int] = field(default_factory=list)

    # MAE by trade duration
    mae_by_duration: Dict[str, float] = field(default_factory=dict)

    def calculate_optimal_stops(self, buffer_pct: float = 0.25) -> None:
        """Calculate optimal stop-loss distances based on MAE analysis."""
        if self.total_trades < 10:
            # Insufficient data - use conservative defaults
            self.recommended_stop_pct = 0.02  # 2%
            self.conservative_stop_pct = 0.03  # 3%
            self.tight_stop_pct = 0.01        # 1%
            return

        # Recommended stop: 75th percentile + buffer
        self.recommended_stop_pct = self.mae_75th_percentile * (1 + buffer_pct)

        # Conservative stop: 90th percentile + buffer
        self.conservative_stop_pct = self.mae_90th_percentile * (1 + buffer_pct)

        # Tight stop: median + small buffer
        self.tight_stop_pct = self.median_mae * (1 + buffer_pct * 0.5)

        # Ensure minimum stops
        min_stop = 0.005  # 0.5% minimum
        self.recommended_stop_pct = max(self.recommended_stop_pct, min_stop)
        self.conservative_stop_pct = max(self.conservative_stop_pct, min_stop)
        self.tight_stop_pct = max(self.tight_stop_pct, min_stop)


class MAEAnalyzer:
    """
    Maximum Adverse Excursion analyzer for optimal stop-loss placement.

    MAE measures the maximum percentage move against a position during its lifetime.
    By analyzing historical MAE patterns, we can set stop-losses that protect capital
    without being stopped out by normal market noise.

    Key insights:
    - Most trades experience some adverse excursion before becoming profitable
    - Optimal stops should be beyond typical MAE levels
    - MAE patterns differ by timeframe, volatility, and market conditions
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize the MAE analyzer.

        Args:
            max_history: Maximum number of trades to keep in history
        """
        self.max_history = max_history
        self.mae_records: List[MAERecord] = []
        self.active_trades: Dict[str, MAERecord] = {}

        # Analysis cache
        self._analysis_cache: Optional[MAEAnalysis] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_validity_minutes = 30  # Recalculate every 30 minutes

        logger.info("MAEAnalyzer initialized")

    def start_trade_tracking(self, trade_id: str, symbol: str, side: str,
                           entry_price: float, entry_time: Optional[datetime] = None) -> None:
        """
        Start tracking MAE for a new trade.

        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            entry_price: Entry price
            entry_time: Entry timestamp (defaults to now)
        """
        if entry_time is None:
            entry_time = datetime.now()

        mae_record = MAERecord(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_time=entry_time
        )

        self.active_trades[trade_id] = mae_record

        # Invalidate analysis cache
        self._analysis_cache = None

        logger.debug(f"Started MAE tracking for trade {trade_id} at ${entry_price:.2f}")

    def update_trade_price(self, trade_id: str, current_price: float,
                          current_time: Optional[datetime] = None) -> None:
        """
        Update MAE calculation with current price.

        Args:
            trade_id: Trade identifier
            current_price: Current market price
            current_time: Current timestamp (defaults to now)
        """
        if current_time is None:
            current_time = datetime.now()

        if trade_id in self.active_trades:
            self.active_trades[trade_id].update_mae(current_price, current_time)

    def finalize_trade(self, trade_id: str, exit_price: float,
                      exit_time: Optional[datetime] = None) -> None:
        """
        Finalize trade and store MAE record.

        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            exit_time: Exit timestamp (defaults to now)
        """
        if exit_time is None:
            exit_time = datetime.now()

        if trade_id in self.active_trades:
            mae_record = self.active_trades[trade_id]
            mae_record.finalize_trade(exit_price, exit_time)

            # Move to completed trades
            self.mae_records.append(mae_record)
            del self.active_trades[trade_id]

            # Maintain history size
            if len(self.mae_records) > self.max_history:
                self.mae_records = self.mae_records[-self.max_history:]

            # Invalidate analysis cache
            self._analysis_cache = None

            logger.debug(f"Finalized MAE tracking for trade {trade_id}: "
                        f"MAE={mae_record.max_adverse_excursion:.4f}, "
                        f"P&L={mae_record.final_pnl_pct:.4f}")

    def get_optimal_stop_distance(self, percentile: float = 75,
                                buffer_pct: float = 0.25) -> float:
        """
        Get optimal stop-loss distance based on historical MAE analysis.

        Args:
            percentile: Percentile of MAE distribution to use (default 75th)
            buffer_pct: Additional buffer above the percentile

        Returns:
            Recommended stop distance as percentage
        """
        analysis = self.get_mae_analysis()

        if analysis.total_trades < 10:
            # Insufficient data - return conservative default
            return 0.02  # 2%

        # Select percentile
        if percentile >= 90:
            base_stop = analysis.mae_90th_percentile
        elif percentile >= 75:
            base_stop = analysis.mae_75th_percentile
        else:
            base_stop = analysis.median_mae

        # Add buffer
        optimal_stop = base_stop * (1 + buffer_pct)

        # Ensure minimum stop
        return max(optimal_stop, 0.005)  # 0.5% minimum

    def get_dynamic_stop_distance(self, trade_id: str, current_profit_r: float,
                                base_volatility: float = 0.01) -> float:
        """
        Get dynamic stop distance based on trade progress and MAE history.

        Args:
            trade_id: Current trade identifier
            current_profit_r: Current profit in risk units (e.g., 0.5R, 1R, 2R)
            base_volatility: Base volatility measure (ATR or similar)

        Returns:
            Recommended stop distance as percentage
        """
        analysis = self.get_mae_analysis()

        if trade_id not in self.active_trades:
            return self.get_optimal_stop_distance()

        current_mae = self.active_trades[trade_id].max_adverse_excursion

        # Dynamic stop logic
        if current_profit_r < 0.5:
            # Early stage - use base stop
            base_stop = self.get_optimal_stop_distance(75)
            return base_stop

        elif current_profit_r < 1.0:
            # Break-even stage - move stop to entry
            return 0.0  # Break-even stop

        elif current_profit_r < 2.0:
            # Profit stage - tight trailing stop
            return base_volatility * 0.5

        else:
            # Large profit stage - wider trailing stop
            return base_volatility * 1.5

    def get_mae_analysis(self) -> MAEAnalysis:
        """
        Get comprehensive MAE analysis.

        Returns:
            MAEAnalysis object with statistical analysis
        """
        # Check cache validity
        now = datetime.now()
        if (self._analysis_cache is not None and
            self._cache_timestamp is not None and
            (now - self._cache_timestamp).total_seconds() < self._cache_validity_minutes * 60):
            return self._analysis_cache

        # Perform analysis
        analysis = self._calculate_mae_analysis()

        # Cache result
        self._analysis_cache = analysis
        self._cache_timestamp = now

        return analysis

    def _calculate_mae_analysis(self) -> MAEAnalysis:
        """Calculate comprehensive MAE statistics."""
        if not self.mae_records:
            return MAEAnalysis()

        # Extract MAE values
        all_maes = [record.max_adverse_excursion for record in self.mae_records]
        winner_maes = [record.max_adverse_excursion for record in self.mae_records if record.was_winner]
        loser_maes = [record.max_adverse_excursion for record in self.mae_records if not record.was_winner]

        analysis = MAEAnalysis(
            total_trades=len(self.mae_records),
            winning_trades=len(winner_maes),
            losing_trades=len(loser_maes)
        )

        if all_maes:
            analysis.avg_mae_all = np.mean(all_maes)
            analysis.median_mae = np.median(all_maes)
            analysis.mae_75th_percentile = np.percentile(all_maes, 75)
            analysis.mae_90th_percentile = np.percentile(all_maes, 90)
            analysis.mae_95th_percentile = np.percentile(all_maes, 95)

        if winner_maes:
            analysis.avg_mae_winners = np.mean(winner_maes)

        if loser_maes:
            analysis.avg_mae_losers = np.mean(loser_maes)

        # Calculate optimal stops
        analysis.calculate_optimal_stops()

        # MAE distribution histogram
        if all_maes:
            hist_bins = np.linspace(0, max(all_maes), 20)
            hist_counts, _ = np.histogram(all_maes, bins=hist_bins)
            analysis.mae_histogram_bins = hist_bins.tolist()
            analysis.mae_histogram_counts = hist_counts.tolist()

        # MAE by duration
        duration_bins = [(0, 5), (5, 15), (15, 60), (60, 240)]  # minutes
        for min_dur, max_dur in duration_bins:
            bin_records = [r for r in self.mae_records
                          if min_dur <= r.duration_minutes < max_dur]
            if bin_records:
                bin_maes = [r.max_adverse_excursion for r in bin_records]
                analysis.mae_by_duration[f"{min_dur}-{max_dur}min"] = np.mean(bin_maes)

        return analysis

    def get_mae_insights(self) -> Dict[str, Union[float, str, List]]:
        """Get key insights from MAE analysis."""
        analysis = self.get_mae_analysis()

        insights = {
            'total_trades_analyzed': analysis.total_trades,
            'win_rate': analysis.winning_trades / analysis.total_trades if analysis.total_trades > 0 else 0,

            'mae_statistics': {
                'average_mae': analysis.avg_mae_all,
                'median_mae': analysis.median_mae,
                'mae_75th_percentile': analysis.mae_75th_percentile,
                'mae_90th_percentile': analysis.mae_90th_percentile
            },

            'optimal_stops': {
                'recommended_stop_pct': analysis.recommended_stop_pct,
                'conservative_stop_pct': analysis.conservative_stop_pct,
                'tight_stop_pct': analysis.tight_stop_pct
            },

            'mae_patterns': {
                'winners_avg_mae': analysis.avg_mae_winners,
                'losers_avg_mae': analysis.avg_mae_losers,
                'mae_by_duration': analysis.mae_by_duration
            }
        }

        # Add insights based on analysis
        if analysis.total_trades >= 50:
            if analysis.avg_mae_winners > analysis.avg_mae_losers:
                insights['key_insight'] = "Winning trades typically experience more adverse excursion, suggesting patience is rewarded"
            else:
                insights['key_insight'] = "Winning trades have lower MAE, suggesting tighter stops may be optimal"

            if analysis.mae_75th_percentile < 0.02:
                insights['stop_recommendation'] = "Tight stops (2%) appear optimal based on MAE analysis"
            elif analysis.mae_75th_percentile < 0.05:
                insights['stop_recommendation'] = "Moderate stops (3-5%) recommended"
            else:
                insights['stop_recommendation'] = "Wide stops (>5%) needed for this strategy"

        return insights

    def get_trade_mae_status(self, trade_id: str) -> Optional[Dict]:
        """Get current MAE status for an active trade."""
        if trade_id not in self.active_trades:
            return None

        record = self.active_trades[trade_id]
        analysis = self.get_mae_analysis()

        return {
            'current_mae': record.max_adverse_excursion,
            'mae_vs_median': record.max_adverse_excursion - analysis.median_mae,
            'mae_vs_75th': record.max_adverse_excursion - analysis.mae_75th_percentile,
            'recommended_stop': analysis.recommended_stop_pct,
            'time_elapsed_minutes': (datetime.now() - record.entry_time).total_seconds() / 60
        }

    def export_mae_data(self, filepath: str) -> None:
        """Export MAE data for external analysis."""
        try:
            data = [record.to_dict() for record in self.mae_records]

            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)

            logger.info(f"Exported {len(data)} MAE records to {filepath}")

        except Exception as e:
            logger.error(f"Error exporting MAE data: {e}")

    def reset_analysis(self) -> None:
        """Reset MAE analysis cache."""
        self._analysis_cache = None
        self._cache_timestamp = None
        logger.info("MAE analysis cache reset")

    def get_stop_adjustment_factor(self, current_mae: float,
                                 target_percentile: float = 75) -> float:
        """
        Calculate stop adjustment factor based on current MAE vs historical.

        Args:
            current_mae: Current MAE for the trade
            target_percentile: Target percentile for comparison

        Returns:
            Adjustment factor (>1 means widen stop, <1 means tighten)
        """
        analysis = self.get_mae_analysis()

        if analysis.total_trades < 10:
            return 1.0

        # Get target MAE level
        if target_percentile >= 90:
            target_mae = analysis.mae_90th_percentile
        elif target_percentile >= 75:
            target_mae = analysis.mae_75th_percentile
        else:
            target_mae = analysis.median_mae

        if target_mae == 0:
            return 1.0

        # Calculate adjustment factor
        adjustment = current_mae / target_mae

        # Limit extreme adjustments
        return np.clip(adjustment, 0.5, 2.0)
