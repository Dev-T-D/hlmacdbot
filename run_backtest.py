#!/usr/bin/env python3
"""
Backtest Runner Script

Run backtests on historical data with configurable parameters.

Usage:
    python3 run_backtest.py --data data.csv --config config.json
    python3 run_backtest.py --data data.csv --initial-balance 10000 --slippage 0.001
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from backtester import Backtester, BacktestConfig
from macd_strategy import MACDStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_historical_data(filepath: str) -> pd.DataFrame:
    """
    Load historical data from CSV file
    
    Supports multiple formats:
    - timestamp,open,high,low,close,volume (standard)
    - Datetime,Open,High,Low,Close,Volume (capitalized)
    - datetime,open,high,low,close,volume (lowercase datetime)
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with OHLCV data (standardized column names)
    """
    try:
        df = pd.read_csv(filepath)
        
        # Normalize column names (case-insensitive mapping)
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ['timestamp', 'datetime', 'date', 'time']:
                column_mapping[col] = 'timestamp'
            elif col_lower == 'open':
                column_mapping[col] = 'open'
            elif col_lower == 'high':
                column_mapping[col] = 'high'
            elif col_lower == 'low':
                column_mapping[col] = 'low'
            elif col_lower == 'close':
                column_mapping[col] = 'close'
            elif col_lower == 'volume':
                column_mapping[col] = 'volume'
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Found columns: {list(df.columns)}. "
                f"Expected: {required_columns}"
            )
        
        # Convert timestamp to datetime
        if df['timestamp'].dtype == 'int64' or df['timestamp'].dtype == 'float64':
            # Try milliseconds first, then seconds
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            except (ValueError, OverflowError):
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN values in critical columns
        initial_len = len(df)
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'timestamp'])
        if len(df) < initial_len:
            logger.warning(f"Removed {initial_len - len(df)} rows with NaN values")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} candles from {filepath}")
        logger.info(f"Date range: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
        logger.info(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise


def create_backtest_config(args, config_dict: Optional[dict] = None) -> BacktestConfig:
    """
    Create BacktestConfig from arguments and/or config file
    
    Args:
        args: Command line arguments
        config_dict: Optional config dictionary from file
        
    Returns:
        BacktestConfig instance
    """
    # Start with defaults
    backtest_config = BacktestConfig()
    
    # Override with config file if provided
    if config_dict:
        strategy_config = config_dict.get('strategy', {})
        risk_config = config_dict.get('risk', {})
        trading_config = config_dict.get('trading', {})
        
        backtest_config.leverage = risk_config.get('leverage', backtest_config.leverage)
        backtest_config.max_position_size_pct = risk_config.get('max_position_size_pct', backtest_config.max_position_size_pct)
        backtest_config.max_daily_loss_pct = risk_config.get('max_daily_loss_pct', backtest_config.max_daily_loss_pct)
        backtest_config.max_trades_per_day = risk_config.get('max_trades_per_day', backtest_config.max_trades_per_day)
        
        # Trailing stop
        trailing_config = risk_config.get('trailing_stop', {})
        backtest_config.trailing_stop_enabled = trailing_config.get('enabled', False)
        if backtest_config.trailing_stop_enabled:
            backtest_config.trailing_stop_config = {
                'trail_percent': trailing_config.get('trail_percent', 2.0),
                'activation_percent': trailing_config.get('activation_percent', 1.0),
                'update_threshold_percent': trailing_config.get('update_threshold_percent', 0.5)
            }
        
        # Multi-timeframe
        multi_tf_config = config_dict.get('multi_timeframe', {})
        backtest_config.multi_timeframe_enabled = multi_tf_config.get('enabled', False)
        backtest_config.higher_timeframe = multi_tf_config.get('higher_timeframe')
        
        # Strategy configuration
        backtest_config.strategy_config = strategy_config
    
    # Override with command line arguments
    if args.initial_balance:
        backtest_config.initial_balance = args.initial_balance
    if args.slippage:
        backtest_config.slippage_pct = args.slippage
    if args.maker_fee:
        backtest_config.maker_fee = args.maker_fee
    if args.taker_fee:
        backtest_config.taker_fee = args.taker_fee
    if args.leverage:
        backtest_config.leverage = args.leverage
    
    return backtest_config


def main():
    parser = argparse.ArgumentParser(description='Run backtest on historical data')
    parser.add_argument('--data', required=True, help='Path to historical data CSV file')
    parser.add_argument('--config', help='Path to config.json (optional)')
    parser.add_argument('--initial-balance', type=float, help='Initial balance (default: 10000)')
    parser.add_argument('--slippage', type=float, help='Slippage percentage (default: 0.001 = 0.1%%)')
    parser.add_argument('--maker-fee', type=float, help='Maker fee rate (default: 0.0002 = 0.02%%)')
    parser.add_argument('--taker-fee', type=float, help='Taker fee rate (default: 0.0004 = 0.04%%)')
    parser.add_argument('--leverage', type=int, help='Leverage (default: 10)')
    parser.add_argument('--start-date', help='Start date filter (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date filter (YYYY-MM-DD)')
    parser.add_argument('--export-trades', help='Export trades to CSV file')
    parser.add_argument('--export-equity', help='Export equity curve to CSV file')
    
    args = parser.parse_args()
    
    # Load config if provided
    config_dict = None
    if args.config:
        config_dict = load_config_from_file(args.config)
    
    # Create backtest config
    backtest_config = create_backtest_config(args, config_dict)
    
    # Load historical data
    df = load_historical_data(args.data)
    
    # Parse date filters
    start_date = None
    end_date = None
    if args.start_date:
        start_date = pd.to_datetime(args.start_date)
    if args.end_date:
        end_date = pd.to_datetime(args.end_date)
    
    # Create backtester
    backtester = Backtester(backtest_config)
    
    # Run backtest
    try:
        results = backtester.run_backtest(df, start_date=start_date, end_date=end_date)
        
        # Export trades if requested
        if args.export_trades:
            backtester.export_trades(args.export_trades)
            logger.info(f"Trades exported to {args.export_trades}")
        
        # Export equity curve if requested
        if args.export_equity:
            equity_df = pd.DataFrame({
                'balance': results['equity_curve']
            })
            equity_df.to_csv(args.export_equity, index=False)
            logger.info(f"Equity curve exported to {args.export_equity}")
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Win Rate: {results['win_rate']:.2f}%")
        logger.info(f"Total Return: {results['total_return_pct']:+.2f}%")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

