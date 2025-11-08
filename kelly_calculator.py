"""
Kelly Criterion Calculator for Optimal Position Sizing

This module implements the Kelly Criterion for optimal position sizing in trading.
The Kelly formula maximizes the long-term growth rate while minimizing the risk of ruin.

Key Features:
- Full Kelly and fractional Kelly calculations
- Dynamic parameter updates from trading history
- Risk of ruin estimation
- Monte Carlo simulation for risk assessment
- Integration with ML confidence scores
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KellyParameters:
    """Parameters for Kelly Criterion calculation."""
    win_rate: float = 0.0  # Probability of winning trades (p)
    avg_win: float = 0.0   # Average win size (as % of capital)
    avg_loss: float = 0.0  # Average loss size (as % of capital)
    reward_risk_ratio: float = 2.0  # Average win / average loss ratio
    kelly_fraction: float = 0.25  # Fraction of Kelly to use (typically 0.25 = Quarter Kelly)
    max_position_pct: float = 0.10  # Maximum position size as % of capital
    min_position_pct: float = 0.001  # Minimum position size as % of capital

    def __post_init__(self):
        """Validate parameters after initialization."""
        if not (0 <= self.win_rate <= 1):
            raise ValueError(f"win_rate must be between 0 and 1, got {self.win_rate}")
        if self.avg_win < 0:
            raise ValueError(f"avg_win must be non-negative, got {self.avg_win}")
        if self.avg_loss < 0:
            raise ValueError(f"avg_loss must be non-negative, got {self.avg_loss}")
        if not (0 < self.kelly_fraction <= 1):
            raise ValueError(f"kelly_fraction must be between 0 and 1, got {self.kelly_fraction}")


@dataclass
class KellyResult:
    """Result of Kelly Criterion calculation."""
    full_kelly_pct: float = 0.0      # Full Kelly percentage
    fractional_kelly_pct: float = 0.0  # Fractional Kelly percentage
    recommended_position_pct: float = 0.0  # Final recommended position size
    risk_of_ruin: float = 0.0        # Estimated probability of ruin
    expected_growth_rate: float = 0.0  # Expected geometric growth rate
    parameters: KellyParameters = field(default_factory=KellyParameters)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'full_kelly_pct': self.full_kelly_pct,
            'fractional_kelly_pct': self.fractional_kelly_pct,
            'recommended_position_pct': self.recommended_position_pct,
            'risk_of_ruin': self.risk_of_ruin,
            'expected_growth_rate': self.expected_growth_rate,
            'parameters': {
                'win_rate': self.parameters.win_rate,
                'avg_win': self.parameters.avg_win,
                'avg_loss': self.parameters.avg_loss,
                'reward_risk_ratio': self.parameters.reward_risk_ratio,
                'kelly_fraction': self.parameters.kelly_fraction,
                'max_position_pct': self.parameters.max_position_pct,
                'min_position_pct': self.parameters.min_position_pct
            }
        }


class KellyCalculator:
    """
    Kelly Criterion calculator for optimal position sizing.

    The Kelly Criterion maximizes the long-term growth rate by finding the optimal
    fraction of capital to risk on each trade. It balances growth against the risk of ruin.

    Formula: f = (p * b - q) / b
    where:
    - f = fraction of capital to risk
    - p = probability of winning
    - q = probability of losing (1 - p)
    - b = ratio of average win to average loss
    """

    def __init__(self, initial_parameters: Optional[KellyParameters] = None):
        """
        Initialize the Kelly calculator.

        Args:
            initial_parameters: Initial Kelly parameters (optional)
        """
        self.parameters = initial_parameters or KellyParameters()
        self.trade_history: List[Dict] = []
        self.max_history = 1000  # Keep last 1000 trades for statistics

        # Exponential moving average parameters for updating statistics
        self.ema_alpha = 0.1  # Weight for recent trades (higher = more responsive)

        logger.info("KellyCalculator initialized")

    def calculate_kelly_fraction(self, win_rate: Optional[float] = None,
                               avg_win: Optional[float] = None,
                               avg_loss: Optional[float] = None,
                               reward_risk_ratio: Optional[float] = None) -> KellyResult:
        """
        Calculate the Kelly fraction for optimal position sizing.

        Args:
            win_rate: Probability of winning (overrides instance parameter)
            avg_win: Average win size as % of capital (overrides instance parameter)
            avg_loss: Average loss size as % of capital (overrides instance parameter)
            reward_risk_ratio: Average win/loss ratio (overrides instance parameter)

        Returns:
            KellyResult with calculated position sizing
        """
        # Use provided parameters or instance parameters
        p = win_rate if win_rate is not None else self.parameters.win_rate
        avg_win_pct = avg_win if avg_win is not None else self.parameters.avg_win
        avg_loss_pct = avg_loss if avg_loss is not None else self.parameters.avg_loss
        b = reward_risk_ratio if reward_risk_ratio is not None else self.parameters.reward_risk_ratio

        # Validate inputs
        if p <= 0 or p >= 1:
            logger.warning(f"Invalid win rate: {p}, using conservative estimate")
            p = 0.5  # Conservative default

        if avg_loss_pct <= 0:
            logger.warning(f"Invalid average loss: {avg_loss_pct}, using conservative estimate")
            avg_loss_pct = 0.02  # 2% conservative default

        if b <= 0:
            logger.warning(f"Invalid reward/risk ratio: {b}, using conservative estimate")
            b = 1.5  # Conservative default

        # Calculate full Kelly fraction
        # f = (p * b - q) / b where q = 1 - p
        q = 1 - p
        full_kelly_pct = (p * b - q) / b

        # Apply Kelly fraction (typically 0.25 for safety)
        fractional_kelly_pct = full_kelly_pct * self.parameters.kelly_fraction

        # Apply limits
        recommended_position_pct = np.clip(
            fractional_kelly_pct,
            self.parameters.min_position_pct,
            self.parameters.max_position_pct
        )

        # Calculate additional metrics
        risk_of_ruin = self._calculate_risk_of_ruin(p, b)
        expected_growth_rate = self._calculate_expected_growth_rate(full_kelly_pct, p, b)

        result = KellyResult(
            full_kelly_pct=full_kelly_pct,
            fractional_kelly_pct=fractional_kelly_pct,
            recommended_position_pct=recommended_position_pct,
            risk_of_ruin=risk_of_ruin,
            expected_growth_rate=expected_growth_rate,
            parameters=KellyParameters(
                win_rate=p,
                avg_win=avg_win_pct,
                avg_loss=avg_loss_pct,
                reward_risk_ratio=b,
                kelly_fraction=self.parameters.kelly_fraction,
                max_position_pct=self.parameters.max_position_pct,
                min_position_pct=self.parameters.min_position_pct
            )
        )

        logger.debug(f"Kelly calculation: p={p:.3f}, b={b:.2f}, full_kelly={full_kelly_pct:.4f}, "
                    f"fractional={fractional_kelly_pct:.4f}, recommended={recommended_position_pct:.4f}")

        return result

    def calculate_position_size(self, account_balance: float,
                              kelly_result: Optional[KellyResult] = None,
                              ml_confidence: float = 1.0) -> Dict[str, Union[float, str]]:
        """
        Calculate actual position size in dollars.

        Args:
            account_balance: Current account balance
            kelly_result: Pre-calculated Kelly result (optional)
            ml_confidence: ML model confidence multiplier (0-1)

        Returns:
            Dictionary with position sizing details
        """
        if kelly_result is None:
            kelly_result = self.calculate_kelly_fraction()

        # Apply ML confidence adjustment
        confidence_adjusted_pct = kelly_result.recommended_position_pct * ml_confidence

        # Ensure within bounds
        final_position_pct = np.clip(
            confidence_adjusted_pct,
            self.parameters.min_position_pct,
            self.parameters.max_position_pct
        )

        position_size = account_balance * final_position_pct

        return {
            'position_size_usd': position_size,
            'position_size_pct': final_position_pct,
            'kelly_fraction': kelly_result.fractional_kelly_pct,
            'ml_confidence': ml_confidence,
            'risk_of_ruin': kelly_result.risk_of_ruin,
            'expected_growth_rate': kelly_result.expected_growth_rate
        }

    def update_statistics(self, trade_result: Dict) -> None:
        """
        Update Kelly parameters with new trade result using exponential moving average.

        Args:
            trade_result: Dictionary with trade details
                Required keys: 'pnl_pct' (profit/loss as % of capital), 'win' (boolean)
        """
        pnl_pct = trade_result.get('pnl_pct', 0)
        is_win = trade_result.get('win', pnl_pct > 0)

        # Add to trade history
        trade_record = {
            'timestamp': datetime.now(),
            'pnl_pct': pnl_pct,
            'win': is_win,
            'trade_result': trade_result
        }
        self.trade_history.append(trade_record)

        # Maintain history size
        if len(self.trade_history) > self.max_history:
            self.trade_history = self.trade_history[-self.max_history:]

        # Update parameters using exponential moving average
        if len(self.trade_history) >= 2:  # Need at least 2 trades for meaningful stats
            self._update_ema_statistics()

        logger.debug(f"Updated Kelly statistics with trade: pnl={pnl_pct:.4f}%, win={is_win}")

    def _update_ema_statistics(self) -> None:
        """Update Kelly parameters using exponential moving average of recent trades."""
        recent_trades = self.trade_history[-100:]  # Use last 100 trades for statistics

        if len(recent_trades) < 2:
            return

        # Calculate wins and losses
        wins = [t['pnl_pct'] for t in recent_trades if t['win']]
        losses = [abs(t['pnl_pct']) for t in recent_trades if not t['win']]

        if not wins or not losses:
            return

        # Update win rate with EMA
        new_win_rate = len(wins) / len(recent_trades)
        self.parameters.win_rate = self._ema_update(
            self.parameters.win_rate, new_win_rate, self.ema_alpha
        )

        # Update average win with EMA
        new_avg_win = np.mean(wins)
        if self.parameters.avg_win == 0:
            self.parameters.avg_win = new_avg_win
        else:
            self.parameters.avg_win = self._ema_update(
                self.parameters.avg_win, new_avg_win, self.ema_alpha
            )

        # Update average loss with EMA
        new_avg_loss = np.mean(losses)
        if self.parameters.avg_loss == 0:
            self.parameters.avg_loss = new_avg_loss
        else:
            self.parameters.avg_loss = self._ema_update(
                self.parameters.avg_loss, new_avg_loss, self.ema_alpha
            )

        # Update reward/risk ratio
        if self.parameters.avg_loss > 0:
            new_rr_ratio = self.parameters.avg_win / self.parameters.avg_loss
            if self.parameters.reward_risk_ratio == 2.0:  # Default value
                self.parameters.reward_risk_ratio = new_rr_ratio
            else:
                self.parameters.reward_risk_ratio = self._ema_update(
                    self.parameters.reward_risk_ratio, new_rr_ratio, self.ema_alpha
                )

    def _ema_update(self, current_value: float, new_value: float, alpha: float) -> float:
        """Update value using exponential moving average."""
        return alpha * new_value + (1 - alpha) * current_value

    def _calculate_risk_of_ruin(self, win_rate: float, reward_risk_ratio: float,
                               num_trades: int = 1000) -> float:
        """
        Calculate the probability of ruin using simplified approximation.

        This uses a simplified formula. For more accurate calculation,
        use Monte Carlo simulation.

        Args:
            win_rate: Probability of winning
            reward_risk_ratio: Average win/loss ratio
            num_trades: Number of trades to simulate

        Returns:
            Estimated probability of ruin
        """
        try:
            # Simplified approximation for risk of ruin
            # More accurate calculation would use Monte Carlo
            if win_rate <= 0.5:
                # If win rate <= 50%, risk of ruin approaches 1
                return 0.99

            # Use formula: ROR ≈ (q/p)^N where q=1-p, but this is simplified
            # For practical purposes, use a conservative estimate
            loss_rate = 1 - win_rate

            if loss_rate >= win_rate:
                return 0.95  # High risk of ruin

            # Simplified calculation based on Kelly principles
            # Risk of ruin decreases as edge increases
            edge = win_rate * reward_risk_ratio - loss_rate

            if edge <= 0:
                return 0.90  # Negative edge = high risk

            # Rough approximation: lower edge = higher risk
            risk_of_ruin = max(0.01, np.exp(-edge * 10))

            return min(risk_of_ruin, 0.99)

        except Exception as e:
            logger.debug(f"Error calculating risk of ruin: {e}")
            return 0.50  # Conservative default

    def _calculate_expected_growth_rate(self, kelly_fraction: float,
                                      win_rate: float, reward_risk_ratio: float) -> float:
        """
        Calculate expected geometric growth rate.

        The Kelly criterion maximizes the expected logarithmic growth rate.

        Args:
            kelly_fraction: Kelly fraction used
            win_rate: Probability of winning
            reward_risk_ratio: Average win/loss ratio

        Returns:
            Expected annual growth rate (approximate)
        """
        try:
            # Expected growth rate per trade: G = p * ln(1 + f * b) + q * ln(1 - f)
            # where q = 1 - p

            p = win_rate
            q = 1 - p
            f = kelly_fraction
            b = reward_risk_ratio

            if f * b >= 1:  # Would lead to infinite growth (theoretical)
                return 0.0

            growth_per_trade = p * np.log(1 + f * b) + q * np.log(1 - f)

            # Approximate annual growth (assuming ~250 trading days/year, 5 trades/day)
            trades_per_year = 250 * 5
            annual_growth_rate = np.exp(growth_per_trade * trades_per_year) - 1

            return annual_growth_rate

        except Exception as e:
            logger.debug(f"Error calculating expected growth rate: {e}")
            return 0.0

    def run_monte_carlo_simulation(self, initial_balance: float = 10000,
                                 num_simulations: int = 1000,
                                 num_trades: int = 1000,
                                 ruin_threshold: float = 0.1) -> Dict[str, float]:
        """
        Run Monte Carlo simulation to estimate risk of ruin and performance distribution.

        Args:
            initial_balance: Starting account balance
            num_simulations: Number of Monte Carlo simulations
            num_trades: Number of trades per simulation
            ruin_threshold: Fraction of initial balance considered "ruin"

        Returns:
            Dictionary with simulation results
        """
        try:
            ruin_balance = initial_balance * ruin_threshold

            # Get current parameters
            p = self.parameters.win_rate
            avg_win_pct = self.parameters.avg_win
            avg_loss_pct = self.parameters.avg_loss

            if p == 0 or avg_loss_pct == 0:
                logger.warning("Insufficient trading history for Monte Carlo simulation")
                return {
                    'risk_of_ruin': 0.5,
                    'median_final_balance': initial_balance,
                    'worst_case_balance': initial_balance,
                    'best_case_balance': initial_balance
                }

            ruin_count = 0
            final_balances = []

            for sim in range(num_simulations):
                balance = initial_balance

                for trade in range(num_trades):
                    # Simulate trade outcome
                    is_win = np.random.random() < p

                    if is_win:
                        pnl_pct = avg_win_pct
                    else:
                        pnl_pct = -avg_loss_pct

                    # Apply P&L
                    balance *= (1 + pnl_pct)

                    # Check for ruin
                    if balance <= ruin_balance:
                        ruin_count += 1
                        break

                final_balances.append(balance)

            # Calculate statistics
            risk_of_ruin = ruin_count / num_simulations
            median_final_balance = np.median(final_balances)
            worst_case_balance = np.min(final_balances)
            best_case_balance = np.max(final_balances)

            return {
                'risk_of_ruin': risk_of_ruin,
                'median_final_balance': median_final_balance,
                'worst_case_balance': worst_case_balance,
                'best_case_balance': best_case_balance,
                'simulations_run': num_simulations,
                'trades_per_simulation': num_trades
            }

        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            return {
                'risk_of_ruin': 0.5,
                'error': str(e)
            }

    def get_statistics_summary(self) -> Dict[str, Union[float, int]]:
        """Get summary of current Kelly statistics."""
        total_trades = len(self.trade_history)
        if total_trades == 0:
            return {'total_trades': 0}

        wins = sum(1 for t in self.trade_history if t['win'])
        win_rate = wins / total_trades

        winning_trades = [t['pnl_pct'] for t in self.trade_history if t['win']]
        losing_trades = [abs(t['pnl_pct']) for t in self.trade_history if not t['win']]

        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        reward_risk_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # Calculate current Kelly
        kelly_result = self.calculate_kelly_fraction()

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'reward_risk_ratio': reward_risk_ratio,
            'current_kelly_pct': kelly_result.recommended_position_pct,
            'risk_of_ruin': kelly_result.risk_of_ruin,
            'expected_growth_rate': kelly_result.expected_growth_rate
        }

    def reset_statistics(self) -> None:
        """Reset all statistics and trade history."""
        self.parameters = KellyParameters()
        self.trade_history.clear()
        logger.info("Kelly calculator statistics reset")

    def save_state(self, filepath: str) -> None:
        """Save Kelly calculator state to file."""
        try:
            import pickle
            state = {
                'parameters': self.parameters,
                'trade_history': self.trade_history[-100:],  # Save last 100 trades
                'max_history': self.max_history,
                'ema_alpha': self.ema_alpha
            }

            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            logger.info(f"Kelly calculator state saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving Kelly calculator state: {e}")

    def load_state(self, filepath: str) -> None:
        """Load Kelly calculator state from file."""
        try:
            import pickle

            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            self.parameters = state.get('parameters', KellyParameters())
            self.trade_history = state.get('trade_history', [])
            self.max_history = state.get('max_history', 1000)
            self.ema_alpha = state.get('ema_alpha', 0.1)

            logger.info(f"Kelly calculator state loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading Kelly calculator state: {e}")
            # Continue with default state
