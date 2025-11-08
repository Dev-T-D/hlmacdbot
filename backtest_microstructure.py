"""
Backtesting Script for Order Flow and Market Microstructure Analysis

This script compares trading performance with and without order flow filters
to validate the effectiveness of microstructure-based signal enhancement.

Usage:
    python backtest_microstructure.py --symbol BTCUSDT --start-date 2024-01-01 --end-date 2024-02-01
"""

import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from hyperliquid_client import HyperliquidClient
from macd_strategy import MACDStrategy
from order_flow_analyzer import OrderFlowAnalyzer
from risk_manager import RiskManager

logger = logging.getLogger(__name__)


class MicrostructureBacktester:
    """
    Backtester for comparing trading strategies with and without microstructure analysis.

    This class simulates historical trading with different filter combinations to
    measure the impact of order flow analysis on trading performance.
    """

    def __init__(self, symbol: str, start_date: datetime, end_date: datetime,
                 config: Optional[Dict] = None):
        """
        Initialize the microstructure backtester.

        Args:
            symbol: Trading symbol
            start_date: Backtest start date
            end_date: Backtest end date
            config: Backtest configuration
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.config = config or self._get_default_config()

        # Initialize components
        self.client = HyperliquidClient(
            private_key="dummy",  # Not needed for historical data
            wallet_address="dummy",
            testnet=True
        )

        # Strategy configurations to test
        self.strategies = {
            'baseline': {
                'name': 'MACD Only',
                'use_microstructure': False,
                'description': 'Original MACD strategy without microstructure filters'
            },
            'orderbook_only': {
                'name': 'MACD + Order Book',
                'use_microstructure': True,
                'filters': ['orderbook'],
                'description': 'MACD with order book imbalance filter only'
            },
            'trade_flow_only': {
                'name': 'MACD + Trade Flow',
                'use_microstructure': True,
                'filters': ['trade_flow'],
                'description': 'MACD with trade flow confirmation only'
            },
            'volume_profile_only': {
                'name': 'MACD + Volume Profile',
                'use_microstructure': True,
                'filters': ['volume_profile'],
                'description': 'MACD with volume profile confirmation only'
            },
            'full_microstructure': {
                'name': 'MACD + Full Microstructure',
                'use_microstructure': True,
                'filters': ['orderbook', 'trade_flow', 'volume_profile'],
                'description': 'MACD with all microstructure filters'
            }
        }

        # Backtest results storage
        self.results = {}

        logger.info(f"Initialized microstructure backtester for {symbol} "
                   f"from {start_date.date()} to {end_date.date()}")

    def _get_default_config(self) -> Dict:
        """Get default backtest configuration."""
        return {
            'strategy': {
                'fast_length': 12,
                'slow_length': 26,
                'signal_length': 9,
                'risk_reward_ratio': 2.0
            },
            'risk': {
                'max_position_size_pct': 0.05,
                'max_daily_loss_pct': 0.03,
                'leverage': 5
            },
            'microstructure': {
                'imbalance_threshold': 0.3,
                'depth_levels': 10
            },
            'backtest': {
                'initial_balance': 10000.0,
                'commission_pct': 0.1,  # 0.1% round trip
                'slippage_bps': 5,      # 5 basis points
                'max_trades_per_day': 10
            }
        }

    def run_backtest(self) -> Dict[str, Dict]:
        """
        Run backtest for all strategy configurations.

        Returns:
            Dictionary with backtest results for each strategy
        """
        logger.info("Starting microstructure backtest...")

        # Load historical data
        historical_data = self._load_historical_data()
        if historical_data.empty:
            raise ValueError("No historical data available for backtesting")

        logger.info(f"Loaded {len(historical_data)} candles of historical data")

        # Run backtest for each strategy
        for strategy_name, strategy_config in self.strategies.items():
            logger.info(f"Running backtest for: {strategy_config['name']}")
            try:
                result = self._run_single_backtest(historical_data, strategy_config)
                self.results[strategy_name] = result
                logger.info(f"✅ {strategy_name}: {result['total_trades']} trades, "
                          f"P&L: ${result['total_pnl']:.2f}, Win Rate: {result['win_rate']:.1%}")
            except Exception as e:
                logger.error(f"❌ Failed backtest for {strategy_name}: {e}")
                self.results[strategy_name] = {'error': str(e)}

        # Generate comparison report
        comparison = self._generate_comparison_report()

        logger.info("Microstructure backtest completed")
        return {
            'results': self.results,
            'comparison': comparison
        }

    def _load_historical_data(self) -> pd.DataFrame:
        """Load historical market data for backtesting."""
        try:
            # Calculate date range
            days = (self.end_date - self.start_date).days

            # Load data in chunks to avoid rate limits
            all_data = []
            current_date = self.start_date

            while current_date < self.end_date:
                chunk_end = min(current_date + timedelta(days=30), self.end_date)

                # In a real implementation, this would fetch from exchange or database
                # For now, generate synthetic data for demonstration
                chunk_data = self._generate_synthetic_data(current_date, chunk_end)
                all_data.append(chunk_data)

                current_date = chunk_end

            if not all_data:
                return pd.DataFrame()

            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)

            return combined_data

        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()

    def _generate_synthetic_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate synthetic market data for backtesting (for demonstration)."""
        # This is a simplified synthetic data generator
        # In production, this would load real historical data

        hours = int((end_date - start_date).total_seconds() / 3600)
        timestamps = pd.date_range(start=start_date, periods=hours, freq='1H')

        # Generate realistic price movements
        np.random.seed(42)  # For reproducible results

        # Start with realistic BTC price
        base_price = 45000.0

        # Generate random walk with trend
        price_changes = np.random.normal(0, 0.02, len(timestamps))  # 2% daily volatility
        trend = np.linspace(0, 0.1, len(timestamps))  # Slight upward trend

        prices = base_price * (1 + np.cumsum(price_changes + trend/100))

        # Generate OHLCV data
        data = []
        for i, (ts, price) in enumerate(zip(timestamps, prices)):
            # Add some intrabar volatility
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = price * (1 + np.random.normal(0, 0.002))
            close = price

            # Generate volume (higher during volatile periods)
            volatility = abs(price_changes[i] if i < len(price_changes) else 0)
            base_volume = 1000 + np.random.exponential(500)
            volume = base_volume * (1 + volatility * 10)

            data.append({
                'timestamp': ts,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        return pd.DataFrame(data)

    def _run_single_backtest(self, data: pd.DataFrame, strategy_config: Dict) -> Dict:
        """Run backtest for a single strategy configuration."""
        # Initialize strategy
        strategy = MACDStrategy(
            fast_length=self.config['strategy']['fast_length'],
            slow_length=self.config['strategy']['slow_length'],
            signal_length=self.config['strategy']['signal_length'],
            risk_reward_ratio=self.config['strategy']['risk_reward_ratio']
        )

        # Initialize order flow analyzer if needed
        order_flow = None
        if strategy_config.get('use_microstructure', False):
            order_flow = OrderFlowAnalyzer(
                symbol=self.symbol,
                config=self.config.get('microstructure', {})
            )

        # Initialize risk manager
        risk_manager = RiskManager(
            max_position_size_pct=self.config['risk']['max_position_size_pct'],
            max_daily_loss_pct=self.config['risk']['max_daily_loss_pct'],
            max_trades_per_day=self.config['backtest']['max_trades_per_day'],
            leverage=self.config['risk']['leverage']
        )

        # Backtest state
        balance = self.config['backtest']['initial_balance']
        position = None
        trades = []
        daily_trades = 0
        last_trade_date = None

        # Process each candle
        for idx, row in data.iterrows():
            current_date = row['timestamp'].date()

            # Reset daily counters
            if last_trade_date != current_date:
                daily_trades = 0
                last_trade_date = current_date

            # Create OHLCV dict for strategy
            candle = {
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            }

            # Update order flow analyzer with market data
            if order_flow:
                # Simulate order book updates (simplified)
                mid_price = (row['bid'] + row['ask']) / 2 if 'bid' in row and 'ask' in row else row['close']
                # Generate synthetic order book
                bids = [[mid_price * (1 - i * 0.001), 100] for i in range(10)]
                asks = [[mid_price * (1 + i * 0.001), 100] for i in range(10)]

                order_flow.update_orderbook(bids, asks)

                # Simulate trade data
                trade = {
                    'price': row['close'],
                    'quantity': row['volume'] * 0.01,  # Simplified
                    'side': 'buy' if row['close'] > row['open'] else 'sell',
                    'timestamp': row['timestamp'].timestamp()
                }
                order_flow.process_trade(trade)

            # Calculate indicators
            df_slice = data.iloc[max(0, idx-50):idx+1].copy()  # Last 50 candles
            df_indicators = strategy.calculate_indicators(df_slice)

            if df_indicators.empty or len(df_indicators) < 2:
                continue

            current_candle = df_indicators.iloc[-1]

            # Check for position closure
            if position:
                should_close, reason = self._check_backtest_exit_conditions(
                    position, current_candle['close'], strategy, df_indicators
                )

                if should_close:
                    pnl = self._calculate_pnl(position, current_candle['close'])
                    balance += pnl

                    trade_record = {
                        'entry_time': position['entry_time'],
                        'exit_time': row['timestamp'],
                        'side': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_candle['close'],
                        'quantity': position['quantity'],
                        'pnl': pnl,
                        'reason': reason,
                        'commission': abs(pnl) * self.config['backtest']['commission_pct'] / 100
                    }
                    trades.append(trade_record)

                    position = None
                    continue

            # Check for new signals (only if no position)
            if not position and daily_trades < self.config['backtest']['max_trades_per_day']:
                signal = strategy.check_entry_signal(df_indicators)

                if signal:
                    # Apply microstructure filters if enabled
                    if strategy_config.get('use_microstructure', False) and order_flow:
                        enhanced_signal = self._apply_microstructure_filters(
                            signal, order_flow, strategy_config.get('filters', [])
                        )
                        if not enhanced_signal:
                            continue  # Signal rejected by microstructure filters
                        signal = enhanced_signal

                    # Calculate position size
                    stop_loss = signal.get('stop_loss')
                    if stop_loss:
                        size_info = risk_manager.calculate_position_size(
                            balance=balance,
                            entry_price=signal['entry_price'],
                            stop_loss=stop_loss,
                            min_qty=0.001
                        )

                        if size_info and size_info['quantity'] > 0:
                            # Apply slippage and commission
                            entry_price = signal['entry_price'] * (1 + self.config['backtest']['slippage_bps'] / 10000)

                            position = {
                                'type': signal['type'],
                                'entry_price': entry_price,
                                'stop_loss': signal.get('stop_loss'),
                                'take_profit': signal.get('take_profit'),
                                'quantity': size_info['quantity'],
                                'entry_time': row['timestamp']
                            }

                            daily_trades += 1

                            logger.debug(f"Opened {signal['type']} position at ${entry_price:.2f}")

        # Calculate final results
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_pnl = sum(t['pnl'] for t in trades)
        total_commission = sum(t['commission'] for t in trades)
        net_pnl = total_pnl - total_commission

        # Calculate additional metrics
        if trades:
            avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = abs(np.mean([t['pnl'] for t in trades if t['pnl'] <= 0])) if winning_trades < total_trades else 0
            profit_factor = (avg_win * winning_trades) / (avg_loss * (total_trades - winning_trades)) if avg_loss > 0 else float('inf')
        else:
            avg_win = avg_loss = profit_factor = 0

        return {
            'strategy_name': strategy_config['name'],
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_commission': total_commission,
            'net_pnl': net_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_balance': balance,
            'return_pct': (balance / self.config['backtest']['initial_balance'] - 1) * 100,
            'trades': trades[:100]  # Store first 100 trades for analysis
        }

    def _apply_microstructure_filters(self, signal: Dict, order_flow: OrderFlowAnalyzer,
                                    filters: List[str]) -> Optional[Dict]:
        """Apply microstructure filters to trading signal."""
        direction = signal.get('type', '')
        entry_price = signal.get('entry_price', 0)

        # Check order book imbalance
        if 'orderbook' in filters:
            imbalance_data = order_flow.calculate_orderbook_imbalance()
            should_trade, reason = order_flow.should_trade_based_on_imbalance(
                direction, imbalance_data
            )
            if not should_trade:
                return None

        # Check trade flow
        if 'trade_flow' in filters:
            flow_ok, flow_reason = order_flow.get_trade_flow_signal(direction)
            if not flow_ok:
                return None

        # Check volume profile
        if 'volume_profile' in filters:
            volume_ok, volume_reason = order_flow.get_volume_profile_signal(
                direction, entry_price
            )
            if not volume_ok:
                return None

        # Signal passed all filters
        return signal

    def _check_backtest_exit_conditions(self, position: Dict, current_price: float,
                                      strategy: MACDStrategy, df: pd.DataFrame) -> tuple:
        """Check exit conditions for backtest."""
        # Check stop loss and take profit
        if position['type'] == 'LONG':
            if current_price <= position.get('stop_loss', 0):
                return True, 'Stop Loss'
            if current_price >= position.get('take_profit', float('inf')):
                return True, 'Take Profit'
        else:  # SHORT
            if current_price >= position.get('stop_loss', float('inf')):
                return True, 'Stop Loss'
            if current_price <= position.get('take_profit', 0):
                return True, 'Take Profit'

        # Check strategy exit signals
        should_exit, reason = strategy.check_exit_signal(df, position['type'])
        if should_exit:
            return True, reason

        return False, ''

    def _calculate_pnl(self, position: Dict, exit_price: float) -> float:
        """Calculate profit/loss for a closed position."""
        entry_price = position['entry_price']
        quantity = position['quantity']

        if position['type'] == 'LONG':
            gross_pnl = (exit_price - entry_price) * quantity
        else:  # SHORT
            gross_pnl = (entry_price - exit_price) * quantity

        return gross_pnl

    def _generate_comparison_report(self) -> Dict:
        """Generate comparison report across all strategies."""
        if not self.results:
            return {}

        # Extract key metrics
        comparison = {}
        baseline_result = self.results.get('baseline', {})

        for strategy_name, result in self.results.items():
            if 'error' in result:
                continue

            comparison[strategy_name] = {
                'name': result.get('strategy_name', strategy_name),
                'total_trades': result.get('total_trades', 0),
                'win_rate': result.get('win_rate', 0),
                'net_pnl': result.get('net_pnl', 0),
                'profit_factor': result.get('profit_factor', 0),
                'return_pct': result.get('return_pct', 0),
            }

            # Calculate improvement over baseline
            if baseline_result and 'error' not in baseline_result:
                baseline_pnl = baseline_result.get('net_pnl', 0)
                baseline_win_rate = baseline_result.get('win_rate', 0)

                pnl_improvement = result.get('net_pnl', 0) - baseline_pnl
                win_rate_improvement = result.get('win_rate', 0) - baseline_win_rate

                comparison[strategy_name].update({
                    'pnl_improvement': pnl_improvement,
                    'win_rate_improvement': win_rate_improvement,
                    'pnl_improvement_pct': (pnl_improvement / abs(baseline_pnl) * 100) if baseline_pnl != 0 else 0
                })

        return comparison

    def print_results(self) -> None:
        """Print formatted backtest results."""
        if not self.results:
            print("No backtest results available")
            return

        print("\n" + "="*80)
        print("MICROSTRUCTURE BACKTEST RESULTS")
        print("="*80)

        print("<30")
        print("-"*80)

        for strategy_name, result in self.results.items():
            if 'error' in result:
                print("<30"                continue

            name = result.get('strategy_name', strategy_name)
            trades = result.get('total_trades', 0)
            win_rate = result.get('win_rate', 0)
            pnl = result.get('net_pnl', 0)
            profit_factor = result.get('profit_factor', 0)

            print("<30"
                  "<8"
                  "<8.1%"
                  "<12.2f"
                  "<10.2f")

        if self.results.get('comparison'):
            print("\n" + "-"*80)
            print("IMPROVEMENT OVER BASELINE (MACD Only)")
            print("-"*80)

            for strategy_name, comp in self.results['comparison'].items():
                if strategy_name == 'baseline':
                    continue

                name = comp.get('name', strategy_name)
                pnl_imp = comp.get('pnl_improvement', 0)
                win_imp = comp.get('win_rate_improvement', 0)

                print("<30"
                      "<+8.2f"
                      "<+6.1%")

        print("="*80)


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description='Order Flow Microstructure Backtester')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', help='Configuration file path')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Parse dates
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

        # Load config if provided
        config = None
        if args.config:
            import json
            with open(args.config) as f:
                config = json.load(f)

        # Run backtest
        backtester = MicrostructureBacktester(args.symbol, start_date, end_date, config)
        results = backtester.run_backtest()

        # Print results
        backtester.print_results()

        # Save detailed results
        import json
        output_file = f"microstructure_backtest_{args.symbol}_{start_date.date()}_{end_date.date()}.json"
        with open(output_file, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = {}
            for k, v in results.items():
                if isinstance(v, dict):
                    json_results[k] = {}
                    for k2, v2 in v.items():
                        if isinstance(v2, np.floating):
                            json_results[k][k2] = float(v2)
                        elif isinstance(v2, np.integer):
                            json_results[k][k2] = int(v2)
                        else:
                            json_results[k][k2] = v2
                else:
                    json_results[k] = v

            json.dump(json_results, f, indent=2, default=str)

        print(f"\nDetailed results saved to: {output_file}")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


if __name__ == "__main__":
    main()
