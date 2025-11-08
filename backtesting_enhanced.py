"""
Enhanced Backtesting Framework for Trading Strategies

Advanced backtesting with walk-forward optimization, Monte Carlo simulation,
robustness testing, and comprehensive performance analysis.

Features:
- Walk-forward analysis to prevent overfitting
- Monte Carlo simulation for risk assessment
- Transaction cost modeling
- Risk-adjusted performance metrics
- Parameter sensitivity analysis
- Out-of-sample validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
import random
from pathlib import Path
import json

from macd_strategy_enhanced import EnhancedMACDStrategy, MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    calmar_ratio: float
    sortino_ratio: float
    expectancy: float
    kelly_criterion: float
    trades: List[Dict[str, Any]]
    equity_curve: List[float]
    drawdown_curve: List[float]


@dataclass
class WalkForwardResult:
    """Results from walk-forward optimization."""
    in_sample_results: List[BacktestResult]
    out_of_sample_results: List[BacktestResult]
    parameter_sets: List[Dict[str, Any]]
    overall_performance: Dict[str, float]
    robustness_score: float


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    simulations: int
    confidence_intervals: Dict[str, Tuple[float, float]]
    value_at_risk: Dict[str, float]
    expected_shortfall: Dict[str, float]
    probability_of_loss: Dict[str, float]
    max_drawdown_distribution: List[float]
    return_distribution: List[float]


class EnhancedBacktester:
    """
    Enhanced backtesting framework with advanced analysis capabilities.
    """

    def __init__(self,
                 initial_balance: float = 10000.0,
                 commission_per_trade: float = 0.001,  # 0.1%
                 slippage_pct: float = 0.0005,  # 0.05%
                 enable_fractional_shares: bool = True):

        self.initial_balance = initial_balance
        self.commission_per_trade = commission_per_trade
        self.slippage_pct = slippage_pct
        self.enable_fractional_shares = enable_fractional_shares

    def run_backtest(self,
                    df: pd.DataFrame,
                    strategy: EnhancedMACDStrategy,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> BacktestResult:
        """
        Run a single backtest with enhanced strategy.
        """
        # Filter data by date range
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        if len(df) < 100:
            raise ValueError("Insufficient data for backtesting")

        # Initialize tracking variables
        balance = self.initial_balance
        equity_curve = [balance]
        drawdown_curve = [0.0]
        trades = []
        peak_balance = balance
        max_drawdown = 0.0

        # Track open position
        open_position = None

        for i in range(len(df)):
            current_candle = df.iloc[i:i+1]
            current_time = df.index[i]

            # Check for entry signals
            if open_position is None:
                signal = strategy.check_entry_signal(current_candle)

                if signal:
                    # Open position
                    entry_price = signal['price'] * (1 + self.slippage_pct)  # Add slippage
                    position_size = balance * signal['position_size_pct']
                    quantity = position_size / entry_price

                    if not self.enable_fractional_shares:
                        quantity = int(quantity)

                    actual_position_size = quantity * entry_price
                    commission = actual_position_size * self.commission_per_trade

                    open_position = {
                        'type': signal['signal'],
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'entry_time': current_time,
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'commission': commission,
                        'market_regime': signal.get('market_regime', 'unknown')
                    }

                    balance -= commission

            # Check for exit signals
            elif open_position:
                exit_signal, exit_reason = strategy.check_exit_signal(
                    current_candle,
                    open_position['type'],
                    open_position['entry_time'],
                    open_position['entry_price'],
                    open_position['stop_loss']
                )

                current_price = current_candle.iloc[0]['close']

                # Check stop loss and take profit
                should_exit = False
                if open_position['type'] == 'LONG':
                    if current_price <= open_position['stop_loss']:
                        should_exit = True
                        exit_reason = 'stop_loss'
                    elif current_price >= open_position['take_profit']:
                        should_exit = True
                        exit_reason = 'take_profit'
                else:  # SHORT
                    if current_price >= open_position['stop_loss']:
                        should_exit = True
                        exit_reason = 'stop_loss'
                    elif current_price <= open_position['take_profit']:
                        should_exit = True
                        exit_reason = 'take_profit'

                if exit_signal or should_exit:
                    # Close position
                    exit_price = current_price * (1 - self.slippage_pct)  # Subtract slippage for exit
                    exit_commission = abs(quantity * exit_price) * self.commission_per_trade

                    if open_position['type'] == 'LONG':
                        pnl = (exit_price - open_position['entry_price']) * open_position['quantity']
                    else:
                        pnl = (open_position['entry_price'] - exit_price) * open_position['quantity']

                    pnl -= exit_commission
                    balance += pnl

                    # Record trade
                    trade = {
                        'entry_time': open_position['entry_time'],
                        'exit_time': current_time,
                        'type': open_position['type'],
                        'entry_price': open_position['entry_price'],
                        'exit_price': exit_price,
                        'quantity': open_position['quantity'],
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'commission': open_position['commission'] + exit_commission,
                        'market_regime': open_position['market_regime'],
                        'duration': (current_time - open_position['entry_time']).total_seconds() / 3600  # hours
                    }

                    trades.append(trade)
                    strategy.update_performance(trade)

                    open_position = None

            # Update equity curve and drawdown
            current_equity = balance
            if open_position:
                unrealized_pnl = self._calculate_unrealized_pnl(open_position, current_price)
                current_equity += unrealized_pnl

            equity_curve.append(current_equity)

            # Calculate drawdown
            peak_balance = max(peak_balance, current_equity)
            current_drawdown = (peak_balance - current_equity) / peak_balance
            max_drawdown = max(max_drawdown, current_drawdown)
            drawdown_curve.append(current_drawdown)

        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(trades, equity_curve, max_drawdown)

        return BacktestResult(
            total_return=metrics['total_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=max_drawdown,
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            total_trades=len(trades),
            avg_trade_duration=metrics['avg_trade_duration'],
            calmar_ratio=metrics['calmar_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            expectancy=metrics['expectancy'],
            kelly_criterion=metrics['kelly_criterion'],
            trades=trades,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve
        )

    def _calculate_unrealized_pnl(self, position: Dict[str, Any], current_price: float) -> float:
        """Calculate unrealized P&L for open position."""
        if position['type'] == 'LONG':
            return (current_price - position['entry_price']) * position['quantity']
        else:
            return (position['entry_price'] - current_price) * position['quantity']

    def _calculate_performance_metrics(self, trades: List[Dict[str, Any]],
                                     equity_curve: List[float], max_drawdown: float) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not trades:
            return {
                'total_return': 0.0, 'sharpe_ratio': 0.0, 'win_rate': 0.0,
                'profit_factor': 0.0, 'avg_trade_duration': 0.0,
                'calmar_ratio': 0.0, 'sortino_ratio': 0.0, 'expectancy': 0.0, 'kelly_criterion': 0.0
            }

        # Basic metrics
        total_pnl = sum(t['pnl'] for t in trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0

        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Duration analysis
        durations = [t['duration'] for t in trades]
        avg_trade_duration = np.mean(durations) if durations else 0

        # Risk-adjusted metrics
        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = returns[~np.isnan(returns)]  # Remove NaN values

        if len(returns) > 1:
            # Sharpe ratio (assuming daily returns, annualize)
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365) if np.std(returns) > 0 else 0

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(365) if len(downside_returns) > 0 else 0
        else:
            sharpe_ratio = sortino_ratio = 0

        # Calmar ratio
        calmar_ratio = total_pnl / max_drawdown if max_drawdown > 0 else 0

        # Expectancy
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Kelly Criterion
        if avg_win > 0 and avg_loss > 0:
            kelly_criterion = win_rate - ((1 - win_rate) * (avg_loss / avg_win))
        else:
            kelly_criterion = 0

        total_return = (equity_curve[-1] - self.initial_balance) / self.initial_balance

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_trade_duration,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'expectancy': expectancy,
            'kelly_criterion': kelly_criterion
        }

    def walk_forward_optimization(self,
                                df: pd.DataFrame,
                                parameter_ranges: Dict[str, List[Any]],
                                in_sample_periods: int = 6,
                                out_of_sample_periods: int = 2,
                                step_months: int = 3) -> WalkForwardResult:
        """
        Perform walk-forward optimization to prevent overfitting.

        Args:
            df: Historical data
            parameter_ranges: Dictionary of parameter names to lists of values to test
            in_sample_periods: Number of periods for in-sample optimization
            out_of_sample_periods: Number of periods for out-of-sample validation
            step_months: How many months to advance each step
        """
        if df.empty:
            raise ValueError("DataFrame is empty")

        # Generate all parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_ranges)

        in_sample_results = []
        out_of_sample_results = []
        tested_params = []

        # Calculate total periods needed
        total_periods = in_sample_periods + out_of_sample_periods
        step_size = pd.DateOffset(months=step_months)

        start_date = df.index.min()
        end_date = df.index.max()

        current_start = start_date

        while current_start + (total_periods * step_size) <= end_date:
            # Define periods
            in_sample_end = current_start + (in_sample_periods * step_size)
            out_of_sample_end = in_sample_end + (out_of_sample_periods * step_size)

            # In-sample optimization
            best_params = None
            best_sharpe = -float('inf')

            for params in param_combinations:
                strategy = EnhancedMACDStrategy(**params)
                try:
                    result = self.run_backtest(df, strategy, current_start, in_sample_end)
                    if result.sharpe_ratio > best_sharpe:
                        best_sharpe = result.sharpe_ratio
                        best_params = params
                except Exception as e:
                    logger.warning(f"Failed backtest with params {params}: {e}")
                    continue

            if best_params:
                tested_params.append(best_params)

                # Out-of-sample validation
                strategy = EnhancedMACDStrategy(**best_params)
                oos_result = self.run_backtest(df, strategy, in_sample_end, out_of_sample_end)

                # Store results
                in_sample_results.append(self.run_backtest(df, strategy, current_start, in_sample_end))
                out_of_sample_results.append(oos_result)

            current_start += step_size

        # Calculate overall performance
        overall_performance = self._calculate_overall_performance(out_of_sample_results)
        robustness_score = self._calculate_robustness_score(out_of_sample_results)

        return WalkForwardResult(
            in_sample_results=in_sample_results,
            out_of_sample_results=out_of_sample_results,
            parameter_sets=tested_params,
            overall_performance=overall_performance,
            robustness_score=robustness_score
        )

    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters."""
        import itertools

        keys = list(parameter_ranges.keys())
        values = list(parameter_ranges.values())

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _calculate_overall_performance(self, results: List[BacktestResult]) -> Dict[str, float]:
        """Calculate overall performance across multiple backtests."""
        if not results:
            return {}

        total_returns = [r.total_return for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results]
        max_drawdowns = [r.max_drawdown for r in results]

        return {
            'avg_total_return': np.mean(total_returns),
            'median_total_return': np.median(total_returns),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'return_std': np.std(total_returns),
            'sharpe_std': np.std(sharpe_ratios)
        }

    def _calculate_robustness_score(self, results: List[BacktestResult]) -> float:
        """Calculate robustness score (0-1, higher is better)."""
        if not results:
            return 0.0

        # Count profitable periods
        profitable_periods = sum(1 for r in results if r.total_return > 0)
        profitability_rate = profitable_periods / len(results)

        # Average Sharpe ratio
        avg_sharpe = np.mean([r.sharpe_ratio for r in results])

        # Consistency score (lower standard deviation of returns)
        returns = [r.total_return for r in results]
        return_consistency = 1 / (1 + np.std(returns)) if returns else 0

        # Combined score
        robustness = (profitability_rate * 0.4 + (avg_sharpe / 3) * 0.4 + return_consistency * 0.2)
        return min(robustness, 1.0)

    def monte_carlo_simulation(self,
                             df: pd.DataFrame,
                             strategy: EnhancedMACDStrategy,
                             num_simulations: int = 1000,
                             confidence_levels: List[float] = None) -> MonteCarloResult:
        """
        Run Monte Carlo simulation to assess strategy robustness.

        Args:
            df: Historical data
            strategy: Trading strategy
            num_simulations: Number of simulations to run
            confidence_levels: Confidence levels for VaR calculation
        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]

        all_returns = []
        all_max_drawdowns = []

        # Get original result for reference
        original_result = self.run_backtest(df, strategy)

        for i in range(num_simulations):
            # Create bootstrapped sample
            bootstrap_df = self._bootstrap_sample(df)

            try:
                # Run backtest on bootstrapped data
                result = self.run_backtest(bootstrap_df, strategy)

                all_returns.append(result.total_return)
                all_max_drawdowns.append(result.max_drawdown)

            except Exception as e:
                logger.warning(f"Monte Carlo simulation {i+1} failed: {e}")
                continue

        if not all_returns:
            raise ValueError("No successful Monte Carlo simulations")

        # Calculate confidence intervals and VaR
        confidence_intervals = {}
        value_at_risk = {}
        expected_shortfall = {}

        for conf_level in confidence_levels:
            # Confidence intervals
            lower_bound = np.percentile(all_returns, (1 - conf_level) * 100)
            upper_bound = np.percentile(all_returns, conf_level * 100)
            confidence_intervals[f"{int(conf_level*100)}%"] = (lower_bound, upper_bound)

            # Value at Risk (VaR)
            var_level = (1 - conf_level) * 100
            var = np.percentile(all_returns, var_level)
            value_at_risk[f"{int(conf_level*100)}%"] = var

            # Expected Shortfall (CVaR)
            losses = [r for r in all_returns if r <= var]
            es = np.mean(losses) if losses else var
            expected_shortfall[f"{int(conf_level*100)}%"] = es

        # Probability of loss
        probability_of_loss = {}
        for conf_level in confidence_levels:
            prob_loss = sum(1 for r in all_returns if r < 0) / len(all_returns)
            probability_of_loss[f"{int(conf_level*100)}%"] = prob_loss

        return MonteCarloResult(
            simulations=len(all_returns),
            confidence_intervals=confidence_intervals,
            value_at_risk=value_at_risk,
            expected_shortfall=expected_shortfall,
            probability_of_loss=probability_of_loss,
            max_drawdown_distribution=all_max_drawdowns,
            return_distribution=all_returns
        )

    def _bootstrap_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a bootstrap sample from historical data."""
        # Sample with replacement
        sample_indices = np.random.choice(len(df), size=len(df), replace=True)
        bootstrap_df = df.iloc[sample_indices].sort_index()

        # Add some noise to prevent overfitting to specific patterns
        # Slight randomization of OHLC values (±0.1%)
        noise_factor = 0.001
        for col in ['open', 'high', 'low', 'close']:
            if col in bootstrap_df.columns:
                noise = np.random.normal(0, noise_factor, len(bootstrap_df))
                bootstrap_df[col] = bootstrap_df[col] * (1 + noise)

        # Ensure OHLC relationships are maintained
        for idx in bootstrap_df.index:
            row = bootstrap_df.loc[idx]
            bootstrap_df.loc[idx, 'high'] = max(row['open'], row['high'], row['low'], row['close'])
            bootstrap_df.loc[idx, 'low'] = min(row['open'], row['high'], row['low'], row['close'])

        return bootstrap_df

    def parameter_sensitivity_analysis(self,
                                    df: pd.DataFrame,
                                    base_params: Dict[str, Any],
                                    parameter_ranges: Dict[str, List[float]],
                                    metric: str = 'sharpe_ratio') -> Dict[str, List[Tuple[float, float]]]:
        """
        Analyze how sensitive performance is to parameter changes.

        Args:
            df: Historical data
            base_params: Base parameter set
            parameter_ranges: Parameters to test with their ranges
            metric: Performance metric to analyze
        """
        sensitivity_results = {}

        for param_name, param_values in parameter_ranges.items():
            results = []

            for param_value in param_values:
                # Create parameter set with this value
                test_params = base_params.copy()
                test_params[param_name] = param_value

                try:
                    strategy = EnhancedMACDStrategy(**test_params)
                    result = self.run_backtest(df, strategy)

                    # Get the metric value
                    metric_value = getattr(result, metric)
                    results.append((param_value, metric_value))

                except Exception as e:
                    logger.warning(f"Failed sensitivity test for {param_name}={param_value}: {e}")
                    continue

            sensitivity_results[param_name] = results

        return sensitivity_results

    def save_results(self, result: BacktestResult, filename: str):
        """Save backtest results to JSON file."""
        result_dict = {
            'total_return': result.total_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'total_trades': result.total_trades,
            'avg_trade_duration': result.avg_trade_duration,
            'calmar_ratio': result.calmar_ratio,
            'sortino_ratio': result.sortino_ratio,
            'expectancy': result.expectancy,
            'kelly_criterion': result.kelly_criterion,
            'trades': result.trades,
            'equity_curve_length': len(result.equity_curve),
            'drawdown_curve_length': len(result.drawdown_curve)
        }

        with open(filename, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)

        logger.info(f"Backtest results saved to {filename}")

    def load_results(self, filename: str) -> BacktestResult:
        """Load backtest results from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)

        # Reconstruct trades with datetime objects
        trades = []
        for trade in data['trades']:
            trade_copy = trade.copy()
            trade_copy['entry_time'] = pd.to_datetime(trade['entry_time'])
            trade_copy['exit_time'] = pd.to_datetime(trade['exit_time'])
            trades.append(trade_copy)

        return BacktestResult(
            total_return=data['total_return'],
            sharpe_ratio=data['sharpe_ratio'],
            max_drawdown=data['max_drawdown'],
            win_rate=data['win_rate'],
            profit_factor=data['profit_factor'],
            total_trades=data['total_trades'],
            avg_trade_duration=data['avg_trade_duration'],
            calmar_ratio=data['calmar_ratio'],
            sortino_ratio=data['sortino_ratio'],
            expectancy=data['expectancy'],
            kelly_criterion=data['kelly_criterion'],
            trades=trades,
            equity_curve=[0.0] * data['equity_curve_length'],  # Placeholder
            drawdown_curve=[0.0] * data['drawdown_curve_length']  # Placeholder
        )


def compare_strategies(df: pd.DataFrame,
                      strategies: Dict[str, EnhancedMACDStrategy],
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> Dict[str, BacktestResult]:
    """
    Compare multiple strategies on the same data.

    Args:
        df: Historical data
        strategies: Dictionary of strategy names to strategy instances
        start_date: Start date for backtest
        end_date: End date for backtest

    Returns:
        Dictionary of strategy names to backtest results
    """
    backtester = EnhancedBacktester()
    results = {}

    for name, strategy in strategies.items():
        try:
            logger.info(f"Running backtest for strategy: {name}")
            result = backtester.run_backtest(df, strategy, start_date, end_date)
            results[name] = result
            logger.info(f"Strategy {name}: Return={result.total_return:.2%}, Sharpe={result.sharpe_ratio:.2f}")

        except Exception as e:
            logger.error(f"Failed to backtest strategy {name}: {e}")

    return results


def generate_performance_report(results: Dict[str, BacktestResult],
                              filename: str = "performance_report.html"):
    """
    Generate HTML performance report comparing multiple strategies.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Strategy Performance Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
            .metric-card {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ color: #7f8c8d; font-size: 14px; }}
            .positive {{ color: #27ae60; }}
            .negative {{ color: #e74c3c; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Strategy Performance Comparison Report</h1>
        <p>Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Strategy Overview</h2>
        <div class="metric-grid">
    """

    # Add strategy cards
    for name, result in results.items():
        color_class = "positive" if result.total_return > 0 else "negative"
        html_content += f"""
            <div class="metric-card">
                <div class="metric-value {color_class}">{result.total_return:.1%}</div>
                <div class="metric-label">{name} - Total Return</div>
            </div>
        """

    html_content += """
        </div>

        <h2>Detailed Metrics</h2>
        <table>
            <tr>
                <th>Strategy</th>
                <th>Total Return</th>
                <th>Sharpe Ratio</th>
                <th>Max Drawdown</th>
                <th>Win Rate</th>
                <th>Profit Factor</th>
                <th>Total Trades</th>
                <th>Expectancy</th>
            </tr>
    """

    for name, result in results.items():
        html_content += f"""
            <tr>
                <td>{name}</td>
                <td>{result.total_return:.2%}</td>
                <td>{result.sharpe_ratio:.2f}</td>
                <td>{result.max_drawdown:.2%}</td>
                <td>{result.win_rate:.1%}</td>
                <td>{result.profit_factor:.2f}</td>
                <td>{result.total_trades}</td>
                <td>${result.expectancy:.2f}</td>
            </tr>
        """

    html_content += """
        </table>
    </body>
    </html>
    """

    with open(filename, 'w') as f:
        f.write(html_content)

    logger.info(f"Performance report saved to {filename}")
