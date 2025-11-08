"""
Adaptive Strategy System

This module implements regime-adaptive trading strategies that dynamically adjust
parameters based on detected market conditions. Each regime uses optimized
parameters for maximum effectiveness.

Key Features:
- Regime-specific strategy parameters
- Dynamic MACD settings per regime
- Risk/reward ratio adaptation
- Position sizing adjustments
- Stop-loss optimization
- Strategy switching (trend-following vs mean-reversion)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any

from regime_detector import MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class StrategyParameters:
    """Complete set of strategy parameters for a regime."""
    # Core strategy settings
    strategy_type: str = "MACD_OVERLAY"  # MACD_OVERLAY, MEAN_REVERSION, BREAKOUT, NONE

    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # RSI parameters
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # Bollinger Band parameters (for mean reversion)
    bb_period: int = 20
    bb_std: float = 2.0

    # Risk management
    risk_reward_ratio: float = 2.0
    position_size_multiplier: float = 1.0
    trailing_stop_pct: float = 2.0
    min_histogram_strength: float = 0.0

    # Entry filters
    require_volume_confirmation: bool = True
    volume_period: int = 20
    min_trend_strength: float = 0.0
    strict_long_conditions: bool = True
    disable_long_trades: bool = False
    disable_short_trades: bool = False

    # Advanced filters
    use_adx_filter: bool = True
    adx_period: int = 14
    adx_threshold: float = 20

    # Time-based filters
    time_filters_enabled: bool = False
    allowed_hours: List[int] = field(default_factory=lambda: list(range(24)))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'strategy_type': self.strategy_type,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'rsi_period': self.rsi_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'bb_period': self.bb_period,
            'bb_std': self.bb_std,
            'risk_reward_ratio': self.risk_reward_ratio,
            'position_size_multiplier': self.position_size_multiplier,
            'trailing_stop_pct': self.trailing_stop_pct,
            'min_histogram_strength': self.min_histogram_strength,
            'require_volume_confirmation': self.require_volume_confirmation,
            'volume_period': self.volume_period,
            'min_trend_strength': self.min_trend_strength,
            'strict_long_conditions': self.strict_long_conditions,
            'disable_long_trades': self.disable_long_trades,
            'disable_short_trades': self.disable_short_trades,
            'use_adx_filter': self.use_adx_filter,
            'adx_period': self.adx_period,
            'adx_threshold': self.adx_threshold,
            'time_filters_enabled': self.time_filters_enabled,
            'allowed_hours': self.allowed_hours
        }


class RegimePerformanceTracker:
    """Track strategy performance by regime."""

    def __init__(self, max_history: int = 1000):
        """
        Initialize performance tracker.

        Args:
            max_history: Maximum trades to keep in history
        """
        self.max_history = max_history
        self.regime_performance: Dict[MarketRegime, List[Dict]] = {
            regime: [] for regime in MarketRegime
        }

    def record_trade(self, regime: MarketRegime, trade_result: Dict) -> None:
        """
        Record trade result for regime analysis.

        Args:
            regime: Market regime during trade
            trade_result: Trade details (pnl, duration, etc.)
        """
        self.regime_performance[regime].append(trade_result)

        # Maintain history size
        if len(self.regime_performance[regime]) > self.max_history:
            self.regime_performance[regime] = self.regime_performance[regime][-self.max_history:]

    def get_regime_profitability(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Get profitability metrics for a regime.

        Args:
            regime: Market regime to analyze

        Returns:
            Dictionary with performance metrics
        """
        trades = self.regime_performance[regime]

        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'total_return': 0.0
            }

        # Calculate metrics
        total_trades = len(trades)
        wins = [t for t in trades if t.get('pnl_pct', 0) > 0]
        losses = [t for t in trades if t.get('pnl_pct', 0) <= 0]

        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        avg_win = sum(t['pnl_pct'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl_pct'] for t in losses) / len(losses) if losses else 0

        # Profit factor
        total_wins = sum(t['pnl_pct'] for t in wins)
        total_losses = abs(sum(t['pnl_pct'] for t in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Sharpe ratio (simplified)
        returns = [t['pnl_pct'] for t in trades]
        if len(returns) > 1:
            avg_return = sum(returns) / len(returns)
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe_ratio = avg_return / std_return * (252 ** 0.5) if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        total_return = sum(t['pnl_pct'] for t in trades)

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return
        }

    def get_best_regime(self) -> Tuple[MarketRegime, Dict[str, float]]:
        """
        Get the best performing regime.

        Returns:
            Tuple of (best_regime, performance_metrics)
        """
        best_regime = None
        best_score = -float('inf')
        best_metrics = {}

        for regime in MarketRegime:
            metrics = self.get_regime_profitability(regime)
            score = metrics['sharpe_ratio']  # Use Sharpe ratio as primary metric

            if score > best_score and metrics['total_trades'] >= 10:
                best_score = score
                best_regime = regime
                best_metrics = metrics

        return (best_regime or MarketRegime.NEUTRAL, best_metrics)


class AdaptiveStrategy:
    """
    Adaptive strategy system that adjusts parameters based on market regime.

    This class implements regime-aware trading strategies with optimized parameters
    for different market conditions. It dynamically switches between trend-following,
    mean-reversion, and breakout strategies based on detected market regimes.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize adaptive strategy system.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()

        # Initialize performance tracking
        self.performance_tracker = RegimePerformanceTracker()

        # Current regime and parameters
        self.current_regime = MarketRegime.NEUTRAL
        self.current_parameters = self._get_default_parameters()

        # Regime transition tracking
        self.regime_history: List[Tuple[MarketRegime, StrategyParameters]] = []

        logger.info("AdaptiveStrategy initialized")

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'regime_parameters': {
                'bull_trend': {
                    'strategy_type': 'MACD_OVERLAY',
                    'macd_fast': 8, 'macd_slow': 21, 'macd_signal': 5,
                    'risk_reward_ratio': 3.0,
                    'position_size_multiplier': 1.2,
                    'trailing_stop_pct': 2.0,
                    'min_histogram_strength': 0.0005,
                    'strict_long_conditions': True,
                    'disable_short_trades': True
                },
                'bear_trend': {
                    'strategy_type': 'MACD_OVERLAY',
                    'macd_fast': 8, 'macd_slow': 21, 'macd_signal': 5,
                    'risk_reward_ratio': 3.0,
                    'position_size_multiplier': 1.0,
                    'trailing_stop_pct': 2.0,
                    'min_histogram_strength': 0.0005,
                    'strict_long_conditions': True,
                    'disable_long_trades': True
                },
                'ranging': {
                    'strategy_type': 'MEAN_REVERSION',
                    'bb_period': 20, 'bb_std': 2.0,
                    'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
                    'risk_reward_ratio': 1.5,
                    'position_size_multiplier': 0.8,
                    'trailing_stop_pct': 1.0,
                    'require_volume_confirmation': False
                },
                'high_volatility': {
                    'strategy_type': 'BREAKOUT',
                    'risk_reward_ratio': 2.5,
                    'position_size_multiplier': 0.5,
                    'trailing_stop_pct': 3.0,
                    'min_histogram_strength': 0.001,
                    'use_adx_filter': False
                },
                'low_volatility': {
                    'strategy_type': 'NONE',
                    'position_size_multiplier': 0.0
                },
                'breakout_pending': {
                    'strategy_type': 'BREAKOUT',
                    'risk_reward_ratio': 3.0,
                    'position_size_multiplier': 1.5,
                    'trailing_stop_pct': 1.5,
                    'min_histogram_strength': 0.001
                }
            },
            'transition_rules': {
                'min_regime_persistence': 3,
                'allow_partial_transitions': True,
                'smooth_transitions': True
            },
            'performance_adaptation': {
                'enabled': True,
                'min_trades_for_adaptation': 50,
                'adaptation_rate': 0.1
            }
        }

    def _get_default_parameters(self) -> StrategyParameters:
        """Get default strategy parameters."""
        return StrategyParameters()

    def adapt_to_regime(self, regime: MarketRegime,
                        regime_confidence: float = 1.0) -> StrategyParameters:
        """
        Adapt strategy parameters for the detected regime.

        Args:
            regime: Detected market regime
            regime_confidence: Confidence in regime detection (0-1)

        Returns:
            Optimized strategy parameters for the regime
        """
        try:
            # Get base parameters for regime
            regime_key = regime.value
            base_params = self.config['regime_parameters'].get(regime_key, {})

            # Create parameter object
            parameters = StrategyParameters(**base_params)

            # Apply confidence-based adjustments
            if regime_confidence < 0.7:
                # Reduce position size for low confidence
                parameters.position_size_multiplier *= regime_confidence
                # Use more conservative risk/reward
                parameters.risk_reward_ratio = min(parameters.risk_reward_ratio, 2.0)

            # Apply performance-based adaptations
            if self.config['performance_adaptation']['enabled']:
                parameters = self._apply_performance_adaptations(parameters, regime)

            # Update current state
            self.current_regime = regime
            self.current_parameters = parameters

            # Track regime transitions
            self.regime_history.append((regime, parameters))

            logger.info(f"Adapted strategy for regime: {regime.value} "
                       f"(confidence: {regime_confidence:.2f})")

            return parameters

        except Exception as e:
            logger.error(f"Error adapting to regime {regime}: {e}")
            return self._get_default_parameters()

    def _apply_performance_adaptations(self, parameters: StrategyParameters,
                                      regime: MarketRegime) -> StrategyParameters:
        """
        Apply performance-based parameter adjustments.

        Args:
            parameters: Base regime parameters
            regime: Current regime

        Returns:
            Adjusted parameters
        """
        try:
            performance = self.performance_tracker.get_regime_profitability(regime)

            if performance['total_trades'] < self.config['performance_adaptation']['min_trades_for_adaptation']:
                return parameters

            adaptation_rate = self.config['performance_adaptation']['adaptation_rate']

            # Adjust position size based on win rate
            if performance['win_rate'] > 0.6:
                # Good performance - increase size slightly
                parameters.position_size_multiplier *= (1 + adaptation_rate)
            elif performance['win_rate'] < 0.4:
                # Poor performance - reduce size
                parameters.position_size_multiplier *= (1 - adaptation_rate)

            # Adjust risk/reward based on profit factor
            if performance['profit_factor'] > 2.0:
                # Profitable - can be more aggressive
                parameters.risk_reward_ratio *= (1 + adaptation_rate)
            elif performance['profit_factor'] < 1.2:
                # Not profitable - be more conservative
                parameters.risk_reward_ratio *= (1 - adaptation_rate)

            # Adjust trailing stop based on volatility of returns
            if performance.get('sharpe_ratio', 0) < 0.5:
                # High volatility - wider stops
                parameters.trailing_stop_pct *= (1 + adaptation_rate)
            elif performance.get('sharpe_ratio', 0) > 1.5:
                # Low volatility - tighter stops
                parameters.trailing_stop_pct *= (1 - adaptation_rate)

            return parameters

        except Exception as e:
            logger.debug(f"Error applying performance adaptations: {e}")
            return parameters

    def should_trade_regime(self, regime: MarketRegime,
                           signal_type: str = "any") -> Tuple[bool, str]:
        """
        Determine if trading should occur in current regime.

        Args:
            regime: Current market regime
            signal_type: Type of signal ('long', 'short', 'any')

        Returns:
            Tuple of (should_trade, reason)
        """
        try:
            parameters = self.current_parameters

            # Check if strategy is disabled for this regime
            if parameters.strategy_type == "NONE":
                return False, f"Trading disabled in {regime.value} regime"

            # Check trade direction restrictions
            if signal_type == "long" and parameters.disable_long_trades:
                return False, "Long trades disabled in current regime"
            elif signal_type == "short" and parameters.disable_short_trades:
                return False, "Short trades disabled in current regime"

            # Check minimum regime persistence
            if len(self.regime_history) < self.config['transition_rules']['min_regime_persistence']:
                return False, f"Regime not persistent enough ({len(self.regime_history)} periods)"

            return True, "Trading allowed"

        except Exception as e:
            logger.error(f"Error checking trade permission: {e}")
            return False, f"Error: {e}"

    def get_regime_parameters(self, regime: MarketRegime) -> StrategyParameters:
        """
        Get optimized parameters for a specific regime.

        Args:
            regime: Market regime

        Returns:
            Strategy parameters optimized for the regime
        """
        regime_key = regime.value
        base_params = self.config['regime_parameters'].get(regime_key, {})
        return StrategyParameters(**base_params)

    def handle_regime_transition(self, old_regime: MarketRegime,
                                new_regime: MarketRegime) -> Dict[str, Any]:
        """
        Handle transition between regimes.

        Args:
            old_regime: Previous regime
            new_regime: New regime

        Returns:
            Transition actions and recommendations
        """
        try:
            actions = {
                'close_positions': False,
                'adjust_stops': False,
                'change_strategy': True,
                'reason': f"Regime changed from {old_regime.value} to {new_regime.value}"
            }

            # Determine transition actions based on regime change
            if old_regime == new_regime:
                actions['change_strategy'] = False
                actions['reason'] = "Same regime - no changes needed"
                return actions

            # Major regime changes requiring position closure
            major_changes = [
                (MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND),
                (MarketRegime.BEAR_TREND, MarketRegime.BULL_TREND),
                (MarketRegime.RANGING, MarketRegime.BULL_TREND),
                (MarketRegime.RANGING, MarketRegime.BEAR_TREND),
                (MarketRegime.LOW_VOLATILITY, MarketRegime.HIGH_VOLATILITY),
            ]

            if (old_regime, new_regime) in major_changes:
                actions['close_positions'] = True
                actions['adjust_stops'] = True
                actions['reason'] += " - Major regime change, closing positions recommended"

            # Strategy type changes
            old_strategy = self.get_regime_parameters(old_regime).strategy_type
            new_strategy = self.get_regime_parameters(new_regime).strategy_type

            if old_strategy != new_strategy:
                actions['change_strategy'] = True
                actions['reason'] += f" - Strategy change: {old_strategy} → {new_strategy}"

            return actions

        except Exception as e:
            logger.error(f"Error handling regime transition: {e}")
            return {
                'close_positions': False,
                'adjust_stops': False,
                'change_strategy': True,
                'reason': f"Error in transition handling: {e}"
            }

    def record_trade_result(self, regime: MarketRegime, trade_result: Dict) -> None:
        """
        Record trade result for performance analysis.

        Args:
            regime: Regime during trade
            trade_result: Trade details and outcome
        """
        try:
            self.performance_tracker.record_trade(regime, trade_result)
            logger.debug(f"Recorded trade result for {regime.value} regime")
        except Exception as e:
            logger.debug(f"Error recording trade result: {e}")

    def get_regime_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive regime performance report."""
        try:
            report = {
                'overall_best_regime': None,
                'regime_performance': {},
                'current_regime': self.current_regime.value,
                'regime_transitions': len(self.regime_history)
            }

            # Get performance for each regime
            for regime in MarketRegime:
                performance = self.performance_tracker.get_regime_profitability(regime)
                report['regime_performance'][regime.value] = performance

            # Find best regime
            best_regime, best_metrics = self.performance_tracker.get_best_regime()
            report['overall_best_regime'] = {
                'regime': best_regime.value,
                'sharpe_ratio': best_metrics.get('sharpe_ratio', 0),
                'win_rate': best_metrics.get('win_rate', 0),
                'total_trades': best_metrics.get('total_trades', 0)
            }

            return report

        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}

    def optimize_regime_parameters(self, regime: MarketRegime,
                                 historical_data: List[Dict]) -> StrategyParameters:
        """
        Optimize parameters for a regime using historical data.

        Args:
            regime: Regime to optimize for
            historical_data: Historical trade data

        Returns:
            Optimized parameters
        """
        try:
            # This is a placeholder for more sophisticated optimization
            # In a real implementation, this would use techniques like:
            # - Bayesian optimization
            # - Genetic algorithms
            # - Walk-forward analysis

            base_params = self.get_regime_parameters(regime)

            # Simple optimization based on historical performance
            if historical_data:
                # Analyze what worked in this regime historically
                regime_trades = [t for t in historical_data if t.get('regime') == regime.value]

                if len(regime_trades) >= 20:
                    # Adjust parameters based on what worked
                    profitable_trades = [t for t in regime_trades if t.get('pnl_pct', 0) > 0]

                    if len(profitable_trades) > len(regime_trades) * 0.6:
                        # Good performance - can be more aggressive
                        base_params.position_size_multiplier *= 1.1
                        base_params.risk_reward_ratio *= 1.05

            return base_params

        except Exception as e:
            logger.error(f"Error optimizing regime parameters: {e}")
            return self.get_regime_parameters(regime)

    def get_strategy_recommendation(self, regime: MarketRegime,
                                   current_conditions: Dict) -> Dict[str, Any]:
        """
        Get comprehensive strategy recommendation for current conditions.

        Args:
            regime: Current market regime
            current_conditions: Current market conditions

        Returns:
            Strategy recommendation with parameters and rationale
        """
        try:
            parameters = self.adapt_to_regime(regime)
            performance = self.performance_tracker.get_regime_profitability(regime)

            recommendation = {
                'regime': regime.value,
                'strategy_type': parameters.strategy_type,
                'parameters': parameters.to_dict(),
                'expected_performance': performance,
                'confidence_level': 'high' if performance.get('total_trades', 0) >= 50 else 'low',
                'risk_level': self._assess_risk_level(parameters, current_conditions),
                'rationale': self._generate_rationale(regime, parameters, performance)
            }

            return recommendation

        except Exception as e:
            logger.error(f"Error generating strategy recommendation: {e}")
            return {
                'regime': regime.value,
                'error': str(e),
                'fallback_parameters': self._get_default_parameters().to_dict()
            }

    def _assess_risk_level(self, parameters: StrategyParameters,
                          conditions: Dict) -> str:
        """Assess risk level of strategy parameters."""
        try:
            risk_score = 0

            # Position size risk
            if parameters.position_size_multiplier > 1.5:
                risk_score += 2
            elif parameters.position_size_multiplier > 1.0:
                risk_score += 1

            # Risk/reward risk
            if parameters.risk_reward_ratio > 3.0:
                risk_score += 2
            elif parameters.risk_reward_ratio > 2.0:
                risk_score += 1

            # Stop loss risk
            if parameters.trailing_stop_pct < 1.0:
                risk_score += 2
            elif parameters.trailing_stop_pct < 2.0:
                risk_score += 1

            # Market condition risk
            volatility = conditions.get('volatility', 0.01)
            if volatility > 0.05:  # 5% daily volatility
                risk_score += 1

            # Determine risk level
            if risk_score >= 5:
                return "very_high"
            elif risk_score >= 3:
                return "high"
            elif risk_score >= 1:
                return "moderate"
            else:
                return "low"

        except Exception as e:
            logger.debug(f"Error assessing risk level: {e}")
            return "unknown"

    def _generate_rationale(self, regime: MarketRegime,
                           parameters: StrategyParameters,
                           performance: Dict) -> str:
        """Generate rationale for strategy recommendation."""
        try:
            rationale_parts = []

            # Regime-specific rationale
            if regime == MarketRegime.BULL_TREND:
                rationale_parts.append("Bull trend detected - using trend-following with aggressive targets")
            elif regime == MarketRegime.BEAR_TREND:
                rationale_parts.append("Bear trend detected - using trend-following with short bias")
            elif regime == MarketRegime.RANGING:
                rationale_parts.append("Ranging market - switching to mean-reversion strategy")
            elif regime == MarketRegime.HIGH_VOLATILITY:
                rationale_parts.append("High volatility - reducing position size and widening stops")
            elif regime == MarketRegime.LOW_VOLATILITY:
                rationale_parts.append("Low volatility - trading suspended due to poor risk/reward")
            elif regime == MarketRegime.BREAKOUT_PENDING:
                rationale_parts.append("Breakout pending - increasing position size for potential large moves")

            # Performance-based rationale
            if performance.get('total_trades', 0) >= 20:
                win_rate = performance.get('win_rate', 0)
                profit_factor = performance.get('profit_factor', 0)

                if win_rate > 0.6 and profit_factor > 1.5:
                    rationale_parts.append(".1%")
                elif win_rate < 0.4 or profit_factor < 1.2:
                    rationale_parts.append("Historical performance in this regime is poor - using conservative parameters")

            return ". ".join(rationale_parts)

        except Exception as e:
            logger.debug(f"Error generating rationale: {e}")
            return f"Strategy adapted for {regime.value} regime"

    def reset(self) -> None:
        """Reset adaptive strategy state."""
        self.current_regime = MarketRegime.NEUTRAL
        self.current_parameters = self._get_default_parameters()
        self.regime_history.clear()

        logger.info("Adaptive strategy reset")
