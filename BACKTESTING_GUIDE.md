# Backtesting Guide

## Overview

The backtesting system provides realistic simulation of trading strategies with:
- **Slippage modeling**: Market orders execute with configurable slippage
- **Trading fees**: Separate maker/taker fees
- **Realistic execution**: Orders execute at candle close ± slippage
- **Risk management**: Respects daily limits, position sizing
- **Performance metrics**: Comprehensive statistics

## Quick Start

### 1. Prepare Historical Data

Create a CSV file with OHLCV data. The backtester supports multiple column name formats:

**Standard format:**
```csv
timestamp,open,high,low,close,volume
1609459200000,29000.0,29500.0,28800.0,29300.0,1000.5
1609545600000,29300.0,29800.0,29100.0,29600.0,1200.3
...
```

**Capitalized format (also supported):**
```csv
Datetime,Open,High,Low,Close,Volume
2024-05-22 01:30:00,619.38,619.78,615.65,617.23,121.927
2024-05-22 02:00:00,616.9,618.42,614.76,615.02,159.519
...
```

**Required columns (case-insensitive):**
- `timestamp`/`datetime`/`date`: Timestamp (Unix milliseconds, Unix seconds, or datetime string)
- `open`, `high`, `low`, `close`: Price data
- `volume`: Trading volume

**Note:** Additional columns (like `Support`, `Resistance`) are ignored automatically.

### 2. Run Backtest

**Basic usage:**
```bash
python3 run_backtest.py --data historical_data.csv
```

**With configuration file:**
```bash
python3 run_backtest.py --data historical_data.csv --config config.json
```

**With custom parameters:**
```bash
python3 run_backtest.py \
    --data historical_data.csv \
    --initial-balance 10000 \
    --slippage 0.001 \
    --taker-fee 0.0004 \
    --leverage 10
```

**Quick backtest with BNB data (example):**
```bash
# Using the provided BNB 30m data
./run_bnb_backtest.sh

# Or manually:
python3 run_backtest.py \
    --data data/BNB_30m_5000.csv \
    --config config/config.json \
    --initial-balance 10000 \
    --leverage 10 \
    --export-trades data/bnb_backtest_trades.csv \
    --export-equity data/bnb_backtest_equity.csv
```

**Export results:**
```bash
python3 run_backtest.py \
    --data historical_data.csv \
    --export-trades trades.csv \
    --export-equity equity_curve.csv
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data` | Path to historical data CSV (required) | - |
| `--config` | Path to config.json (optional) | - |
| `--initial-balance` | Starting balance | 10000 |
| `--slippage` | Slippage percentage (0.001 = 0.1%) | 0.001 |
| `--maker-fee` | Maker fee rate (0.0002 = 0.02%) | 0.0002 |
| `--taker-fee` | Taker fee rate (0.0004 = 0.04%) | 0.0004 |
| `--leverage` | Trading leverage | 10 |
| `--start-date` | Start date filter (YYYY-MM-DD) | - |
| `--end-date` | End date filter (YYYY-MM-DD) | - |
| `--export-trades` | Export trades to CSV | - |
| `--export-equity` | Export equity curve to CSV | - |

## Configuration

The backtester reads settings from `config.json`:

```json
{
  "strategy": {
    "fast_length": 12,
    "slow_length": 26,
    "signal_length": 9,
    "risk_reward_ratio": 2.0
  },
  "risk": {
    "leverage": 10,
    "max_position_size_pct": 0.1,
    "max_daily_loss_pct": 0.05,
    "max_trades_per_day": 10,
    "trailing_stop": {
      "enabled": true,
      "trail_percent": 2.0,
      "activation_percent": 1.0,
      "update_threshold_percent": 0.5
    }
  }
}
```

## How It Works

### Order Execution

1. **Entry Orders**: Market orders execute at candle close + slippage
   - LONG: Buy at `close_price * (1 + slippage_pct)`
   - SHORT: Sell at `close_price * (1 - slippage_pct)`

2. **Exit Orders**: Market orders execute at candle close - slippage
   - LONG: Sell at `close_price * (1 - slippage_pct)`
   - SHORT: Buy at `close_price * (1 + slippage_pct)`

### Fees

- **Maker fees**: Applied to limit orders (default: 0.02%)
- **Taker fees**: Applied to market orders (default: 0.04%)

Fees are deducted from balance on both entry and exit.

### Slippage

Slippage simulates real-world execution:
- Entry slippage: Slightly worse price (costs more)
- Exit slippage: Slightly worse price (receives less)

Default slippage: 0.1% per trade.

### Risk Management

The backtester respects all risk limits:
- **Daily loss limit**: Stops trading if daily loss exceeds limit
- **Trade limit**: Maximum trades per day
- **Position sizing**: Calculated based on stop loss distance
- **Trailing stop**: Optional trailing stop-loss

## Performance Metrics

The backtester calculates:

- **Total Trades**: Number of completed trades
- **Win Rate**: Percentage of profitable trades
- **Total Return**: Absolute and percentage return
- **Average Win/Loss**: Average profit/loss per trade
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted return metric
- **Max Drawdown**: Largest peak-to-trough decline
- **Average Trade Duration**: Average time in position

## Example Output

```
======================================================================
STARTING BACKTEST
======================================================================
Initial Balance: $10,000.00
Period: 2024-01-01 to 2024-12-31
Total Candles: 8760
======================================================================
...
======================================================================
BACKTEST COMPLETE
======================================================================
Total Trades:        45
Winning Trades:      28
Losing Trades:       17
Win Rate:            62.22%
Total P&L:           $2,450.50
Total Return:        $2,450.50 (+24.51%)
Final Balance:       $12,450.50
Average Win:         $245.30
Average Loss:        -$142.80
Profit Factor:       1.72
Sharpe Ratio:        1.45
Max Drawdown:         $850.00 (8.50%)
Avg Trade Duration:   12.5 hours
======================================================================
```

## Exporting Results

### Trade Journal

Export all trades to CSV:

```bash
python3 run_backtest.py --data data.csv --export-trades trades.csv
```

**Columns:**
- `entry_time`, `exit_time`: Trade timestamps
- `position_type`: LONG or SHORT
- `entry_price`, `exit_price`: Execution prices
- `quantity`: Position size
- `entry_slippage`, `exit_slippage`: Slippage amounts
- `entry_fee`, `exit_fee`: Trading fees
- `pnl`, `pnl_pct`: Profit/loss
- `exit_reason`: Why position was closed
- `leverage`: Leverage used

### Equity Curve

Export balance over time:

```bash
python3 run_backtest.py --data data.csv --export-equity equity.csv
```

Useful for visualizing performance and drawdowns.

## Programmatic Usage

You can also use the backtester programmatically:

```python
from backtester import Backtester, BacktestConfig
import pandas as pd

# Load data
df = pd.read_csv('historical_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Configure backtest
config = BacktestConfig(
    initial_balance=10000,
    slippage_pct=0.001,
    taker_fee=0.0004,
    leverage=10,
    trailing_stop_enabled=True
)

# Run backtest
backtester = Backtester(config)
results = backtester.run_backtest(df)

# Access results
print(f"Win Rate: {results['win_rate']:.2f}%")
print(f"Total Return: {results['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

# Export trades
backtester.export_trades('trades.csv')
```

## Tips

1. **Use realistic slippage**: Higher volatility = higher slippage
2. **Test different timeframes**: Strategy may perform differently on different timeframes
3. **Filter by date**: Test specific periods (bull/bear markets)
4. **Compare with/without trailing stop**: See impact on performance
5. **Analyze trade journal**: Identify patterns in winning/losing trades
6. **Check drawdowns**: Ensure max drawdown is acceptable
7. **Validate with paper trading**: Backtest → Paper trade → Live trade

## Limitations

- **No multi-timeframe**: Multi-timeframe analysis requires separate higher TF data
- **No partial fills**: Orders execute fully or not at all
- **No order book simulation**: Uses candle close prices
- **No latency**: Instant execution (unrealistic for HFT)
- **No funding rates**: For perpetual futures (can be added)

## Next Steps

1. Run backtest on historical data
2. Analyze results and optimize parameters
3. Test on different time periods
4. Compare with live trading results
5. Iterate and improve strategy

