#!/usr/bin/env python3

"""Compare Exit Strategy Performance"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from exit_strategies import PartialProfitTaker, VolatilityAdaptiveTrailingStop, TimeBasedExit

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExitStrategyBacktester:
    """Backtest different exit strategies"""

    def __init__(self):
        self.results = {}

    def simulate_trade(self, entry_price, stop_loss, side, exit_strategy, market_data):
        """
        Simulate a single trade with given exit strategy

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            side: 'LONG' or 'SHORT'
            exit_strategy: Dict with strategy configuration
            market_data: DataFrame with OHLCV data

        Returns:
            Dict with trade results
        """
        # Initialize position
        position = {
            'type': side,
            'entry_price': entry_price,
            'quantity': 1000,  # Fixed for simulation
            'original_quantity': 1000,
            'stop_loss': stop_loss,
            'entry_time': datetime.now(),
            'unrealized_pnl': 0
        }

        # Initialize exit strategies
        partial_profit = PartialProfitTaker(levels=[0.5, 1.0, 1.5, 2.0]) if exit_strategy.get('partial_profit', False) else None
        volatility_trailing = VolatilityAdaptiveTrailingStop(atr_multiplier=2.5) if exit_strategy.get('trailing_stop') == 'volatility' else None
        time_exit = TimeBasedExit(max_hold_hours=24) if exit_strategy.get('time_exit', 24) > 0 else None

        # Fixed trailing stop for comparison
        fixed_trailing_distance = abs(entry_price - stop_loss) * 0.5  # 50% of risk

        total_pnl = 0
        remaining_quantity = position['quantity']
        exit_reason = "Still open"

        # Simulate through each candle
        for idx, row in market_data.iterrows():
            current_price = row['close']
            current_time = datetime.now()  # Simplified for backtest

            # Update position P&L
            if side == 'LONG':
                unrealized_pnl = (current_price - entry_price) * remaining_quantity
            else:
                unrealized_pnl = (entry_price - current_price) * remaining_quantity

            position['unrealized_pnl'] = unrealized_pnl

            # Check partial profit taking
            if partial_profit and remaining_quantity > 0:
                actions = partial_profit.check_profit_levels(position, current_price)
                for action in actions:
                    # Calculate P&L for partial close
                    if side == 'LONG':
                        pnl = (current_price - entry_price) * action['quantity']
                    else:
                        pnl = (entry_price - current_price) * action['quantity']

                    total_pnl += pnl
                    remaining_quantity -= action['quantity']

                    if remaining_quantity <= 0:
                        exit_reason = f"Partial Profit {action['r_multiple']}R"
                        break

            # Check volatility-adaptive trailing stop
            if volatility_trailing and remaining_quantity > 0:
                new_stop = volatility_trailing.update_trailing_stop(position, current_price, market_data.loc[:idx])
                if new_stop != position['stop_loss']:
                    position['stop_loss'] = new_stop

            # Check stop loss
            stop_hit = False
            if side == 'LONG' and current_price <= position['stop_loss']:
                stop_hit = True
                exit_reason = "Stop Loss"
            elif side == 'SHORT' and current_price >= position['stop_loss']:
                stop_hit = True
                exit_reason = "Stop Loss"

            if stop_hit:
                # Close remaining position
                if side == 'LONG':
                    pnl = (current_price - entry_price) * remaining_quantity
                else:
                    pnl = (entry_price - current_price) * remaining_quantity

                total_pnl += pnl
                remaining_quantity = 0
                break

            # Check time-based exit (simplified)
            holding_hours = (current_time - position['entry_time']).total_seconds() / 3600
            if holding_hours >= exit_strategy.get('time_exit', 24):
                exit_reason = "Time Exit"

                # Close remaining position
                if side == 'LONG':
                    pnl = (current_price - entry_price) * remaining_quantity
                else:
                    pnl = (entry_price - current_price) * remaining_quantity

                total_pnl += pnl
                remaining_quantity = 0
                break

            # Check if reached end of data (simplified exit)
            if idx == len(market_data) - 1 and remaining_quantity > 0:
                exit_reason = "End of Data"

                # Close remaining position at market
                if side == 'LONG':
                    pnl = (current_price - entry_price) * remaining_quantity
                else:
                    pnl = (entry_price - current_price) * remaining_quantity

                total_pnl += pnl
                remaining_quantity = 0

        # Calculate R-multiple
        risk_amount = abs(entry_price - stop_loss)
        r_multiple = total_pnl / risk_amount if risk_amount > 0 else 0

        return {
            'pnl': total_pnl,
            'r_multiple': r_multiple,
            'exit_reason': exit_reason,
            'remaining_quantity': remaining_quantity
        }


def compare_exit_strategies(df):
    """Compare different exit strategies"""

    strategies = {
        'baseline': {
            'partial_profit': False,
            'trailing_stop': 'fixed',
            'time_exit': 24
        },
        'partial_profit': {
            'partial_profit': True,
            'trailing_stop': 'fixed',
            'time_exit': 24
        },
        'volatility_trail': {
            'partial_profit': False,
            'trailing_stop': 'volatility',
            'time_exit': 24
        },
        'combined': {
            'partial_profit': True,
            'trailing_stop': 'volatility',
            'time_exit': 24
        }
    }

    backtester = ExitStrategyBacktester()
    results = {}

    for name, config in strategies.items():
        logger.info(f"\n🔧 Testing: {name}")

        trades = []

        # Simulate trades from historical data
        # For simplicity, we'll simulate a few trades
        for i in range(min(10, len(df) - 100)):  # Limit to 10 trades for speed
            start_idx = i * 50  # Space trades apart
            if start_idx + 100 >= len(df):
                break

            market_segment = df.iloc[start_idx:start_idx+100]

            # Simulate entry (simplified)
            entry_price = market_segment.iloc[0]['close']
            stop_loss = entry_price * 0.98  # 2% stop loss
            side = 'LONG'

            # Run trade simulation
            trade_result = backtester.simulate_trade(
                entry_price, stop_loss, side, config, market_segment
            )

            trades.append(trade_result)

        # Calculate metrics
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]

            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
            total_return = sum(t['pnl'] for t in trades)
            avg_r_multiple = sum(t['r_multiple'] for t in trades) / len(trades) if trades else 0

            results[name] = {
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_return': total_return,
                'avg_r_multiple': avg_r_multiple,
                'total_trades': len(trades)
            }

            logger.info(".2%")
            logger.info(".2f")
            logger.info(f"   Total Trades: {len(trades)}")
        else:
            results[name] = {
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'total_return': 0,
                'avg_r_multiple': 0,
                'total_trades': 0
            }

    # Compare results
    logger.info("\n📊 EXIT STRATEGY COMPARISON:")
    logger.info("=" * 80)

    comparison_df = pd.DataFrame(results).T
    logger.info(comparison_df.to_string())

    # Find best
    if results:
        best_strategy = max(results.keys(), key=lambda x: results[x]['total_return'])
        logger.info(f"\n🏆 Best Strategy: {best_strategy}")
        logger.info(".2f")
        logger.info(".2f")

    return results


def main():
    print("🚀 EXIT STRATEGY BACKTEST COMPARISON")
    print("=" * 50)

    # Load backtest data
    try:
        df = pd.read_csv('bnb_ml_backtest_trades.csv')
        logger.info(f"Loaded {len(df)} historical trades")
    except FileNotFoundError:
        logger.error("Trade data not found. Run backtest first.")
        return

    # For exit strategy testing, we need price data
    # Let's create some synthetic price data for testing
    import numpy as np

    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=5000, freq='5min')
    prices = 500 + np.cumsum(np.random.randn(5000) * 0.5)  # Random walk

    price_df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices + np.random.randn(5000) * 0.1,
        'volume': np.random.randint(1000, 10000, 5000)
    })

    # Add ATR for volatility trailing
    price_df['high_low'] = price_df['high'] - price_df['low']
    price_df['high_close'] = (price_df['high'] - price_df['close'].shift()).abs()
    price_df['low_close'] = (price_df['low'] - price_df['close'].shift()).abs()

    true_range = pd.concat([
        price_df['high_low'],
        price_df['high_close'],
        price_df['low_close']
    ], axis=1).max(axis=1)

    price_df['atr_14'] = true_range.rolling(14).mean()

    logger.info(f"Created synthetic price data: {len(price_df)} candles")

    # Run comparison
    results = compare_exit_strategies(price_df)

    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('exit_strategy_comparison.csv')
    logger.info("💾 Results saved: exit_strategy_comparison.csv")


if __name__ == "__main__":
    main()
