"""
Advanced Risk Manager with Comprehensive Position Sizing and Risk Controls

This module implements sophisticated risk management for trading, combining:
- Kelly Criterion position sizing
- Volatility-adjusted sizing
- Equity curve protection
- Correlation-based diversification
- Time-based risk adjustments
- Maximum risk limits and circuit breakers
- Dynamic stop-loss management
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

from kelly_calculator import KellyCalculator, KellyResult
from mae_analyzer import MAEAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Comprehensive risk limits configuration."""
    # Per-trade limits
    max_position_size_pct: float = 0.05  # Max 5% of capital per trade
    max_risk_per_trade_pct: float = 0.02  # Max 2% risk per trade

    # Daily limits
    max_daily_loss_pct: float = 0.05  # Max 5% daily loss
    max_daily_trades: int = 10

    # Weekly limits
    max_weekly_loss_pct: float = 0.15  # Max 15% weekly loss

    # Portfolio limits
    max_portfolio_leverage: float = 5.0  # Max 5x total leverage
    max_concurrent_positions: int = 3
    max_correlation_threshold: float = 0.7  # Max correlation between positions

    # Circuit breaker thresholds
    circuit_breaker_daily_loss_pct: float = 0.08  # 8% daily loss triggers circuit breaker
    circuit_breaker_weekly_loss_pct: float = 0.20  # 20% weekly loss triggers circuit breaker

    # Recovery settings
    drawdown_recovery_threshold: float = 0.10  # Resume after 10% drawdown recovery
    cool_down_period_minutes: int = 60  # Wait period after circuit breaker


@dataclass
class RiskState:
    """Current risk state of the portfolio."""
    account_balance: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    peak_balance: float = 0.0
    current_drawdown_pct: float = 0.0

    # Position tracking
    active_positions: List[Dict] = field(default_factory=list)
    daily_trades_count: int = 0
    last_trade_time: Optional[datetime] = None

    # Circuit breaker state
    circuit_breaker_triggered: bool = False
    circuit_breaker_end_time: Optional[datetime] = None
    last_reset_time: datetime = field(default_factory=datetime.now)

    def update_balance(self, new_balance: float) -> None:
        """Update account balance and drawdown calculations."""
        old_balance = self.account_balance
        self.account_balance = new_balance

        # Update peak balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

        # Calculate drawdown
        if self.peak_balance > 0:
            self.current_drawdown_pct = (self.peak_balance - new_balance) / self.peak_balance

    def add_position(self, position: Dict) -> None:
        """Add active position to tracking."""
        self.active_positions.append(position)
        self.daily_trades_count += 1
        self.last_trade_time = datetime.now()

    def remove_position(self, position_id: str) -> None:
        """Remove position from tracking."""
        self.active_positions = [
            pos for pos in self.active_positions
            if pos.get('id') != position_id
        ]

    def get_portfolio_risk(self) -> Dict[str, float]:
        """Calculate current portfolio risk metrics."""
        if not self.active_positions:
            return {'total_risk_pct': 0.0, 'leverage': 0.0}

        total_exposure = sum(pos.get('size_usd', 0) for pos in self.active_positions)
        total_risk = sum(pos.get('risk_usd', 0) for pos in self.active_positions)

        leverage = total_exposure / self.account_balance if self.account_balance > 0 else 0
        total_risk_pct = total_risk / self.account_balance if self.account_balance > 0 else 0

        return {
            'total_risk_pct': total_risk_pct,
            'leverage': leverage,
            'total_exposure': total_exposure,
            'position_count': len(self.active_positions)
        }


@dataclass
class PositionSizingResult:
    """Result of position sizing calculation."""
    position_size_usd: float = 0.0
    position_size_pct: float = 0.0
    risk_amount_usd: float = 0.0
    risk_amount_pct: float = 0.0

    # Sizing factors applied
    kelly_factor: float = 1.0
    volatility_factor: float = 1.0
    equity_curve_factor: float = 1.0
    correlation_factor: float = 1.0
    time_factor: float = 1.0

    # Risk metrics
    expected_risk_of_ruin: float = 0.0
    expected_growth_rate: float = 0.0

    # Rejection reasons
    rejected: bool = False
    rejection_reason: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'position_size_usd': self.position_size_usd,
            'position_size_pct': self.position_size_pct,
            'risk_amount_usd': self.risk_amount_usd,
            'risk_amount_pct': self.risk_amount_pct,
            'factors_applied': {
                'kelly': self.kelly_factor,
                'volatility': self.volatility_factor,
                'equity_curve': self.equity_curve_factor,
                'correlation': self.correlation_factor,
                'time': self.time_factor
            },
            'risk_metrics': {
                'risk_of_ruin': self.expected_risk_of_ruin,
                'growth_rate': self.expected_growth_rate
            },
            'rejected': self.rejected,
            'rejection_reason': self.rejection_reason
        }


class AdvancedRiskManager:
    """
    Advanced risk manager with comprehensive position sizing and risk controls.

    This class implements sophisticated risk management techniques:
    - Kelly Criterion for optimal position sizing
    - Volatility-adjusted sizing
    - Equity curve protection
    - Correlation-based diversification
    - Time-based risk adjustments
    - Circuit breakers and risk limits
    - Dynamic stop-loss management
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the advanced risk manager.

        Args:
            config: Configuration dictionary with risk parameters
        """
        self.config = config or self._get_default_config()

        # Initialize components
        self.kelly_calculator = KellyCalculator()
        self.mae_analyzer = MAEAnalyzer()

        # Risk limits and state
        self.risk_limits = RiskLimits(**self.config.get('limits', {}))
        self.risk_state = RiskState()

        # Performance tracking
        self.daily_performance: List[Dict] = []
        self.weekly_performance: List[Dict] = []

        # Market condition tracking
        self.current_volatility = 0.01  # Default 1%
        self.market_correlations: Dict[str, Dict[str, float]] = {}

        logger.info("AdvancedRiskManager initialized")

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'limits': {
                'max_position_size_pct': 0.05,
                'max_risk_per_trade_pct': 0.02,
                'max_daily_loss_pct': 0.05,
                'max_daily_trades': 10,
                'max_weekly_loss_pct': 0.15,
                'max_portfolio_leverage': 5.0,
                'max_concurrent_positions': 3,
                'circuit_breaker_daily_loss_pct': 0.08,
                'circuit_breaker_weekly_loss_pct': 0.20
            },
            'kelly': {
                'fraction': 0.25,  # Quarter Kelly
                'min_trades_for_estimate': 10
            },
            'volatility': {
                'adjustment_enabled': True,
                'base_atr_period': 14,
                'max_volatility_adjustment': 0.5  # Max 50% size reduction
            },
            'equity_curve': {
                'protection_enabled': True,
                'drawdown_levels': [0.05, 0.10, 0.15, 0.20],
                'reduction_factors': [0.75, 0.50, 0.25, 0.10]
            },
            'correlation': {
                'check_enabled': True,
                'max_correlation': 0.7,
                'reduction_factor': 0.5
            },
            'time_based': {
                'weekend_reduction': 0.5,
                'funding_window_reduction': 0.8,
                'low_liquidity_reduction': 0.7
            }
        }

    # ==========================================
    # POSITION SIZING
    # ==========================================

    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float,
                              account_balance: float, current_positions: List[Dict] = None,
                              ml_confidence: float = 1.0, market_data: Dict = None) -> PositionSizingResult:
        """
        Calculate optimal position size using comprehensive risk analysis.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            account_balance: Current account balance
            current_positions: List of current active positions
            ml_confidence: ML model confidence (0-1)
            market_data: Current market data (volatility, etc.)

        Returns:
            PositionSizingResult with calculated size and risk metrics
        """
        try:
            # Initialize result
            result = PositionSizingResult()

            # Check circuit breaker
            if self._is_circuit_breaker_active():
                result.rejected = True
                result.rejection_reason = "Circuit breaker active"
                return result

            # Check basic risk limits
            limit_check = self._check_basic_limits(account_balance, current_positions or [])
            if not limit_check['allowed']:
                result.rejected = True
                result.rejection_reason = limit_check['reason']
                return result

            # Calculate base Kelly position size
            kelly_result = self.kelly_calculator.calculate_kelly_fraction()
            base_position_pct = kelly_result.recommended_position_pct

            # Apply ML confidence adjustment
            result.kelly_factor = min(base_position_pct * ml_confidence, self.risk_limits.max_position_size_pct)

            # Apply volatility adjustment
            if market_data and self.config['volatility']['adjustment_enabled']:
                vol_factor = self._calculate_volatility_adjustment(market_data)
                result.volatility_factor = vol_factor
                result.kelly_factor *= vol_factor

            # Apply equity curve protection
            if self.config['equity_curve']['protection_enabled']:
                equity_factor = self._calculate_equity_curve_adjustment()
                result.equity_curve_factor = equity_factor
                result.kelly_factor *= equity_factor

            # Apply correlation adjustment
            if self.config['correlation']['check_enabled'] and current_positions:
                corr_factor = self._calculate_correlation_adjustment(symbol, current_positions)
                result.correlation_factor = corr_factor
                result.kelly_factor *= corr_factor

            # Apply time-based adjustment
            time_factor = self._calculate_time_based_adjustment()
            result.time_factor = time_factor
            result.kelly_factor *= time_factor

            # Calculate final position size
            result.position_size_pct = result.kelly_factor
            result.position_size_usd = account_balance * result.position_size_pct

            # Calculate risk amount
            risk_per_share = abs(entry_price - stop_loss)
            position_size_shares = result.position_size_usd / entry_price
            result.risk_amount_usd = risk_per_share * position_size_shares
            result.risk_amount_pct = result.risk_amount_usd / account_balance

            # Check risk per trade limit
            if result.risk_amount_pct > self.risk_limits.max_risk_per_trade_pct:
                reduction_factor = self.risk_limits.max_risk_per_trade_pct / result.risk_amount_pct
                result.position_size_pct *= reduction_factor
                result.position_size_usd *= reduction_factor
                result.risk_amount_usd *= reduction_factor
                result.risk_amount_pct *= reduction_factor

            # Set risk metrics
            result.expected_risk_of_ruin = kelly_result.risk_of_ruin
            result.expected_growth_rate = kelly_result.expected_growth_rate

            # Final validation
            if result.position_size_pct < 0.001:  # Less than 0.1%
                result.rejected = True
                result.rejection_reason = "Position size too small"
                return result

            logger.info(f"Position sizing for {symbol}: ${result.position_size_usd:.2f} "
                       f"({result.position_size_pct:.3f}%), Risk: ${result.risk_amount_usd:.2f}")

            return result

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            result.rejected = True
            result.rejection_reason = f"Calculation error: {e}"
            return result

    def _calculate_volatility_adjustment(self, market_data: Dict) -> float:
        """Calculate volatility-based position size adjustment."""
        try:
            # Get current ATR or volatility measure
            current_atr = market_data.get('atr', self.current_volatility)
            average_atr = market_data.get('avg_atr', self.current_volatility)

            if average_atr == 0:
                return 1.0

            # Inverse relationship: higher volatility = smaller position
            adjustment = average_atr / current_atr

            # Limit extreme adjustments
            max_adjustment = self.config['volatility']['max_volatility_adjustment']
            adjustment = np.clip(adjustment, 1 - max_adjustment, 1 + max_adjustment)

            return adjustment

        except Exception as e:
            logger.debug(f"Error calculating volatility adjustment: {e}")
            return 1.0

    def _calculate_equity_curve_adjustment(self) -> float:
        """Calculate equity curve-based position size adjustment."""
        try:
            drawdown = self.risk_state.current_drawdown_pct
            levels = self.config['equity_curve']['drawdown_levels']
            factors = self.config['equity_curve']['reduction_factors']

            # Find appropriate reduction factor
            for level, factor in zip(levels, factors):
                if drawdown >= level:
                    return factor

            return 1.0  # No adjustment needed

        except Exception as e:
            logger.debug(f"Error calculating equity curve adjustment: {e}")
            return 1.0

    def _calculate_correlation_adjustment(self, symbol: str, current_positions: List[Dict]) -> float:
        """Calculate correlation-based position size adjustment."""
        try:
            max_correlation = 0

            # Check correlation with existing positions
            for position in current_positions:
                pos_symbol = position.get('symbol', '')
                if pos_symbol != symbol:
                    correlation = self.market_correlations.get(symbol, {}).get(pos_symbol, 0)
                    max_correlation = max(max_correlation, abs(correlation))

            # Apply adjustment if correlation is too high
            if max_correlation > self.config['correlation']['max_correlation']:
                return self.config['correlation']['reduction_factor']

            return 1.0

        except Exception as e:
            logger.debug(f"Error calculating correlation adjustment: {e}")
            return 1.0

    def _calculate_time_based_adjustment(self) -> float:
        """Calculate time-based position size adjustment."""
        try:
            now = datetime.now()
            hour = now.hour
            weekday = now.weekday()  # 0=Monday, 6=Sunday

            # Weekend reduction
            if weekday >= 5:  # Saturday/Sunday
                return self.config['time_based']['weekend_reduction']

            # Low liquidity hours (00:00-04:00 UTC)
            if 0 <= hour < 4:
                return self.config['time_based']['low_liquidity_reduction']

            # Funding rate windows (reduce 30 min before/after)
            # Hyperliquid funding at 00:00, 08:00, 16:00 UTC
            if hour in [23, 0, 7, 8, 15, 16]:
                return self.config['time_based']['funding_window_reduction']

            return 1.0

        except Exception as e:
            logger.debug(f"Error calculating time adjustment: {e}")
            return 1.0

    # ==========================================
    # RISK LIMITS AND CIRCUIT BREAKERS
    # ==========================================

    def _check_basic_limits(self, account_balance: float, current_positions: List[Dict]) -> Dict:
        """Check basic risk limits before position sizing."""
        try:
            # Check daily loss limit
            daily_loss_pct = abs(self.risk_state.daily_pnl) / account_balance if account_balance > 0 else 0
            if daily_loss_pct >= self.risk_limits.max_daily_loss_pct:
                return {'allowed': False, 'reason': f'Daily loss limit exceeded ({daily_loss_pct:.1%})'}

            # Check weekly loss limit
            weekly_loss_pct = abs(self.risk_state.weekly_pnl) / account_balance if account_balance > 0 else 0
            if weekly_loss_pct >= self.risk_limits.max_weekly_loss_pct:
                return {'allowed': False, 'reason': f'Weekly loss limit exceeded ({weekly_loss_pct:.1%})'}

            # Check daily trade limit
            if self.risk_state.daily_trades_count >= self.risk_limits.max_daily_trades:
                return {'allowed': False, 'reason': f'Daily trade limit exceeded ({self.risk_state.daily_trades_count})'}

            # Check concurrent positions limit
            if len(current_positions) >= self.risk_limits.max_concurrent_positions:
                return {'allowed': False, 'reason': f'Max concurrent positions exceeded ({len(current_positions)})'}

            # Check portfolio leverage
            portfolio_risk = self.risk_state.get_portfolio_risk()
            if portfolio_risk['leverage'] >= self.risk_limits.max_portfolio_leverage:
                return {'allowed': False, 'reason': f'Portfolio leverage limit exceeded ({portfolio_risk["leverage"]:.1f}x)'}

            return {'allowed': True}

        except Exception as e:
            logger.error(f"Error checking basic limits: {e}")
            return {'allowed': False, 'reason': f'Limit check error: {e}'}

    def _is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is currently active."""
        if not self.risk_state.circuit_breaker_triggered:
            return False

        # Check if cool-down period has expired
        if self.risk_state.circuit_breaker_end_time:
            if datetime.now() >= self.risk_state.circuit_breaker_end_time:
                # Reset circuit breaker
                self.risk_state.circuit_breaker_triggered = False
                self.risk_state.circuit_breaker_end_time = None
                logger.info("Circuit breaker cool-down period expired - trading resumed")
                return False

        return True

    def check_circuit_breaker_conditions(self, account_balance: float) -> bool:
        """
        Check if circuit breaker conditions are met.

        Args:
            account_balance: Current account balance

        Returns:
            True if circuit breaker should be triggered
        """
        try:
            # Calculate current daily and weekly P&L
            daily_loss_pct = abs(self.risk_state.daily_pnl) / account_balance if account_balance > 0 else 0
            weekly_loss_pct = abs(self.risk_state.weekly_pnl) / account_balance if account_balance > 0 else 0

            # Check circuit breaker thresholds
            if (daily_loss_pct >= self.risk_limits.circuit_breaker_daily_loss_pct or
                weekly_loss_pct >= self.risk_limits.circuit_breaker_weekly_loss_pct):

                # Trigger circuit breaker
                self.risk_state.circuit_breaker_triggered = True
                cool_down_minutes = self.risk_limits.cool_down_period_minutes
                self.risk_state.circuit_breaker_end_time = datetime.now() + timedelta(minutes=cool_down_minutes)

                logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED: "
                              f"Daily Loss: {daily_loss_pct:.1%}, Weekly Loss: {weekly_loss_pct:.1%} "
                              f"Cool-down: {cool_down_minutes} minutes")

                return True

            return False

        except Exception as e:
            logger.error(f"Error checking circuit breaker conditions: {e}")
            return False

    # ==========================================
    # DYNAMIC STOP-LOSS MANAGEMENT
    # ==========================================

    def calculate_dynamic_stop(self, trade_id: str, entry_price: float,
                              current_price: float, position_side: str,
                              volatility: float = 0.01) -> Dict[str, Any]:
        """
        Calculate dynamic stop-loss based on trade progress and MAE analysis.

        Args:
            trade_id: Trade identifier
            entry_price: Entry price
            current_price: Current price
            position_side: 'LONG' or 'SHORT'
            volatility: Current volatility measure

        Returns:
            Dictionary with stop loss recommendation
        """
        try:
            # Calculate current profit in risk units
            if position_side == 'LONG':
                current_pnl_pct = (current_price - entry_price) / entry_price
            else:
                current_pnl_pct = (entry_price - current_price) / entry_price

            # Get base risk amount (assume 2% initial risk)
            base_risk_pct = 0.02
            current_profit_r = current_pnl_pct / base_risk_pct

            # Get dynamic stop from MAE analyzer
            optimal_stop_pct = self.mae_analyzer.get_dynamic_stop_distance(
                trade_id=trade_id,
                current_profit_r=current_profit_r,
                base_volatility=volatility
            )

            # Calculate stop price
            if position_side == 'LONG':
                stop_price = current_price * (1 - optimal_stop_pct)
            else:
                stop_price = current_price * (1 + optimal_stop_pct)

            return {
                'stop_price': stop_price,
                'stop_distance_pct': optimal_stop_pct,
                'current_profit_r': current_profit_r,
                'stop_type': 'dynamic_mae_based'
            }

        except Exception as e:
            logger.error(f"Error calculating dynamic stop: {e}")
            # Return conservative default
            if position_side == 'LONG':
                stop_price = entry_price * 0.98  # 2% below entry
            else:
                stop_price = entry_price * 1.02  # 2% above entry

            return {
                'stop_price': stop_price,
                'stop_distance_pct': 0.02,
                'current_profit_r': 0.0,
                'stop_type': 'conservative_default'
            }

    # ==========================================
    # PERFORMANCE TRACKING AND MONITORING
    # ==========================================

    def update_performance(self, pnl: float, trade_result: Dict = None) -> None:
        """
        Update performance tracking with new trade result.

        Args:
            pnl: Profit/loss from the trade
            trade_result: Additional trade details
        """
        try:
            # Update risk state
            self.risk_state.daily_pnl += pnl

            # Update Kelly calculator statistics
            if trade_result:
                self.kelly_calculator.update_statistics(trade_result)

            # Update MAE analyzer if trade details available
            if trade_result and 'trade_id' in trade_result:
                # This would be called when trade is closed
                pass

            # Check circuit breaker conditions
            self.check_circuit_breaker_conditions(self.risk_state.account_balance)

        except Exception as e:
            logger.error(f"Error updating performance: {e}")

    def reset_daily_stats(self, account_balance: float) -> None:
        """Reset daily performance statistics."""
        try:
            # Store previous day performance
            if self.risk_state.daily_pnl != 0:
                daily_record = {
                    'date': datetime.now().date(),
                    'pnl': self.risk_state.daily_pnl,
                    'trades': self.risk_state.daily_trades_count,
                    'balance': account_balance
                }
                self.daily_performance.append(daily_record)

            # Reset daily counters
            self.risk_state.daily_pnl = 0
            self.risk_state.daily_trades_count = 0
            self.risk_state.last_reset_time = datetime.now()

            # Update account balance
            self.risk_state.update_balance(account_balance)

            logger.info(f"Daily stats reset - Balance: ${account_balance:.2f}, "
                       f"Drawdown: {self.risk_state.current_drawdown_pct:.1%}")

        except Exception as e:
            logger.error(f"Error resetting daily stats: {e}")

    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard."""
        try:
            portfolio_risk = self.risk_state.get_portfolio_risk()
            kelly_stats = self.kelly_calculator.get_statistics_summary()
            mae_insights = self.mae_analyzer.get_mae_insights()

            dashboard = {
                'portfolio_health': {
                    'account_balance': self.risk_state.account_balance,
                    'peak_balance': self.risk_state.peak_balance,
                    'current_drawdown_pct': self.risk_state.current_drawdown_pct,
                    'daily_pnl': self.risk_state.daily_pnl,
                    'weekly_pnl': self.risk_state.weekly_pnl
                },
                'portfolio_risk': portfolio_risk,
                'kelly_statistics': kelly_stats,
                'mae_analysis': mae_insights,
                'risk_limits': {
                    'daily_loss_limit_pct': self.risk_limits.max_daily_loss_pct,
                    'daily_loss_current_pct': abs(self.risk_state.daily_pnl) / self.risk_state.account_balance if self.risk_state.account_balance > 0 else 0,
                    'max_concurrent_positions': self.risk_limits.max_concurrent_positions,
                    'current_positions': len(self.risk_state.active_positions),
                    'circuit_breaker_active': self.risk_state.circuit_breaker_triggered
                },
                'trading_activity': {
                    'daily_trades': self.risk_state.daily_trades_count,
                    'active_positions': len(self.risk_state.active_positions),
                    'last_trade_time': self.risk_state.last_trade_time
                }
            }

            return dashboard

        except Exception as e:
            logger.error(f"Error generating risk dashboard: {e}")
            return {'error': str(e)}

    def get_monte_carlo_risk_assessment(self, num_simulations: int = 1000) -> Dict[str, float]:
        """
        Run Monte Carlo simulation for risk of ruin assessment.

        Args:
            num_simulations: Number of Monte Carlo simulations

        Returns:
            Risk assessment results
        """
        try:
            return self.kelly_calculator.run_monte_carlo_simulation(
                initial_balance=self.risk_state.account_balance,
                num_simulations=num_simulations
            )
        except Exception as e:
            logger.error(f"Error running Monte Carlo assessment: {e}")
            return {'error': str(e)}

    # ==========================================
    # UTILITY METHODS
    # ==========================================

    def update_market_conditions(self, volatility: float = None,
                               correlations: Dict[str, Dict[str, float]] = None) -> None:
        """
        Update market condition variables.

        Args:
            volatility: Current market volatility
            correlations: Asset correlation matrix
        """
        if volatility is not None:
            self.current_volatility = volatility

        if correlations is not None:
            self.market_correlations = correlations

    def save_state(self, filepath: str) -> None:
        """Save risk manager state to file."""
        try:
            import pickle
            state = {
                'risk_limits': self.risk_limits,
                'risk_state': self.risk_state,
                'config': self.config,
                'daily_performance': self.daily_performance[-100:],  # Last 100 days
                'kelly_calculator': self.kelly_calculator,
                'mae_analyzer': self.mae_analyzer
            }

            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            logger.info(f"Risk manager state saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving risk manager state: {e}")

    def load_state(self, filepath: str) -> None:
        """Load risk manager state from file."""
        try:
            import pickle

            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            self.risk_limits = state.get('risk_limits', self.risk_limits)
            self.risk_state = state.get('risk_state', self.risk_state)
            self.config = state.get('config', self.config)
            self.daily_performance = state.get('daily_performance', [])

            # Load component states
            if 'kelly_calculator' in state:
                self.kelly_calculator = state['kelly_calculator']
            if 'mae_analyzer' in state:
                self.mae_analyzer = state['mae_analyzer']

            logger.info(f"Risk manager state loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading risk manager state: {e}")

    def reset_risk_state(self) -> None:
        """Reset risk state for fresh start."""
        self.risk_state = RiskState()
        self.kelly_calculator.reset_statistics()
        self.mae_analyzer.reset_analysis()
        self.daily_performance.clear()
        logger.info("Risk manager state reset")
