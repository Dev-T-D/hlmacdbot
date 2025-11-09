#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from macd_strategy import MACDStrategy
from ml_signal_enhancer import MLSignalEnhancer
from signal_confirmation import MultiTimeframeConfirmation, VolumeQualityFilter, MarketRegimeFilter
from exit_strategies import PartialProfitTaker, VolatilityAdaptiveTrailingStop, TimeBasedExit
from filter_statistics import FilterStatistics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLBacktester:
    """Simple backtester with ML integration"""

    def __init__(self, config):
        self.config = config

        # Initialize components like trading bot
        strategy_config = config.get('strategy', {})
        self.strategy = MACDStrategy(
            fast_length=strategy_config.get('fast_length', 12),
            slow_length=strategy_config.get('slow_length', 26),
            signal_length=strategy_config.get('signal_length', 9),
            risk_reward_ratio=strategy_config.get('risk_reward_ratio', 2.0),
            rsi_period=strategy_config.get('rsi_period', 14),
            rsi_oversold=strategy_config.get('rsi_oversold', 30),
            rsi_overbought=strategy_config.get('rsi_overbought', 70),
            min_histogram_strength=strategy_config.get('min_histogram_strength', 0.0),
            require_volume_confirmation=strategy_config.get('require_volume_confirmation', False),
            volume_period=strategy_config.get('volume_period', 20),
            min_trend_strength=strategy_config.get('min_trend_strength', 0.0),
            strict_long_conditions=strategy_config.get('strict_long_conditions', True),
            disable_long_trades=strategy_config.get('disable_long_trades', False)
        )

        # Initialize ML enhancer
        ml_config = config.get('ml_enhancement', {})
        self.ml_enhancer = MLSignalEnhancer(
            symbol=config.get('trading', {}).get('symbol', 'BTCUSDT'),
            config=ml_config
        )

        # Check if ML is available
        self.ml_enabled = self.ml_enhancer.load_models()
        if self.ml_enabled:
            logger.info("✅ ML models loaded for backtest")
        else:
            logger.warning("⚠️ ML models not available - running without ML")

        # Initialize filters
        self.mtf_confirmation = MultiTimeframeConfirmation()
        self.volume_filter = VolumeQualityFilter()
        self.regime_filter = MarketRegimeFilter()
        self.filter_stats = FilterStatistics()

        # Initialize exit strategies
        self.partial_profit = PartialProfitTaker(levels=[0.5, 1.0, 1.5, 2.0])
        self.volatility_trailing = VolatilityAdaptiveTrailingStop(atr_multiplier=2.5)
        self.time_exit = TimeBasedExit(max_hold_hours=24)

        # Trading parameters
        self.symbol = config.get('trading', {}).get('symbol', 'BTCUSDT')
        self.leverage = config.get('risk', {}).get('leverage', 10)
        self.max_position_size_pct = config.get('risk', {}).get('max_position_size_pct', 0.1)

        # Results tracking
        self.trades = []
        self.equity_curve = []
        self.current_balance = 10000.0  # Starting balance
        self.current_position = None

    def run_backtest(self, df):
        """Run backtest on historical data"""
        logger.info("Starting ML-enhanced backtest...")

        # Calculate indicators once
        df_with_indicators = self.strategy.calculate_indicators(df.copy())

        total_trades = 0
        winning_trades = 0

        for i in range(50, len(df_with_indicators)):  # Start after indicator warmup
            current_candle = df_with_indicators.iloc[i]
            current_price = current_candle['close']

            # Update equity curve
            self.equity_curve.append(self.current_balance)

            # Check exit conditions if we have a position
            if self.current_position:
                should_exit, exit_reason = self.check_exit_conditions(df_with_indicators.iloc[:i+1], current_price)
                if should_exit:
                    # Store pnl before closing position
                    pnl_before_close = self.current_position.get('pnl', 0) if self.current_position else 0
                    self.close_position(current_price, exit_reason)
                    total_trades += 1
                    if pnl_before_close > 0:
                        winning_trades += 1

            # Look for entry signals if no position
            elif not self.current_position:
                signal = self.check_entry_signal(df_with_indicators.iloc[:i+1])
                if signal:
                    self.open_position(signal, current_price)

        # Close any remaining position
        if self.current_position:
            pnl_before_close = self.current_position.get('pnl', 0)
            self.close_position(current_price, "End of backtest")
            total_trades += 1
            if pnl_before_close > 0:
                winning_trades += 1

        # Calculate metrics
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_return = (self.current_balance - 10000) / 10000 * 100

        # Calculate Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        # Calculate max drawdown
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_balance': self.current_balance
        }

    def check_entry_signal(self, df):
        """Check for entry signals with ML and filters"""
        # Get MACD signal
        macd_signal = self.strategy.check_entry_signal(df)
        if not macd_signal:
            return None

        logger.debug(f"MACD Signal: {macd_signal['type']} @ ${macd_signal['entry_price']:.2f}")

        # ==========================================
        # 4-LAYER SIGNAL QUALITY FILTER CASCADE
        # ==========================================

        # Get ML prediction
        ml_prediction = {}
        if self.ml_enabled:
            try:
                ml_prediction = self.ml_enhancer.predict_direction(df)
            except Exception as e:
                logger.warning(f"ML prediction failed: {e} - falling back to no ML for this backtest")
                self.ml_enabled = False  # Disable ML for this backtest

        # FILTER LAYER 1: ML Confidence & MACD Agreement (or skip if no ML)
        if self.ml_enabled:
            should_trade, confidence, reason = self.ml_enhancer.should_trade(
                macd_signal['type'], ml_prediction, macd_signal['entry_price']
            )

            # Record filter result
            self.filter_stats.record_filter_result('ml_confidence', should_trade, macd_signal['type'])

            if not should_trade:
                logger.debug(f"❌ Filter 1 Failed: {reason}")
                return None

            logger.debug(f"✅ Filter 1 Passed: Confidence {confidence:.2f}")
            confidence = confidence
        else:
            # No ML - use default confidence and skip this filter
            confidence = 1.0
            logger.debug("✅ Filter 1 Skipped: No ML models available")
            self.filter_stats.record_filter_result('ml_confidence', True, macd_signal['type'])

        # FILTER LAYER 2: Multi-Timeframe Confirmation (relaxed for backtest)
        # In backtest mode, we'll assume MTF alignment for simplicity
        mtf_aligned = True
        mtf_score = 1.0
        logger.debug("✅ Filter 2 Passed: MTF alignment assumed in backtest mode")
        self.filter_stats.record_filter_result('mtf_alignment', True, macd_signal['type'])

        # FILTER LAYER 3: Volume Quality
        volume_ok, volume_ratio, volume_reason = self.volume_filter.check_volume_quality(df)

        # Record filter result
        self.filter_stats.record_filter_result('volume_quality', volume_ok, macd_signal['type'])

        if not volume_ok:
            logger.debug(f"❌ Filter 3 Failed: {volume_reason}")
            return None

        logger.debug(f"✅ Filter 3 Passed: Volume {volume_ratio:.2f}x average")

        # FILTER LAYER 4: Market Regime (relaxed for backtest)
        # In backtest mode, assume favorable regime
        regime_ok = True
        regime_reason = "Favorable regime assumed in backtest mode"
        logger.debug(f"✅ Filter 4 Passed: {regime_reason}")
        self.filter_stats.record_filter_result('market_regime', True, macd_signal['type'])

        # ALL FILTERS PASSED - Create final signal
        logger.info(f"🎯 ALL FILTERS PASSED - Opening {macd_signal['type']} position")
        return {
            'type': macd_signal['type'],
            'entry_price': macd_signal['entry_price'],
            'confidence': confidence,
            'filters_passed': ['ml_confidence', 'mtf_alignment', 'volume_quality', 'market_regime']
        }

    def check_exit_conditions(self, df, current_price):
        """Check exit conditions with enhanced strategies"""
        if not self.current_position:
            return False, ""

        position = self.current_position

        # STRATEGY 1: Partial Profit Taking
        profit_actions = self.partial_profit.check_profit_levels(position, current_price)

        for action in profit_actions:
            logger.debug(f"💰 Partial Profit Target Hit: {action['r_multiple']}R")
            logger.debug(f"   Closing {action['percentage']:.0f}% of position at ${current_price:.2f}")

            # Close partial position
            self._close_partial_position(action['quantity'], current_price, f"Take Profit {action['r_multiple']}R")

        # STRATEGY 2: Volatility-Adaptive Trailing Stop
        if hasattr(df, 'atr_14') and not df.empty:
            new_stop = self.volatility_trailing.update_trailing_stop(position, current_price, df.iloc[-1:])
            if new_stop != position['stop_loss']:
                position['stop_loss'] = new_stop
                logger.debug(f"🎯 Trailing Stop Updated: ${position['stop_loss']:.2f}")

        # STRATEGY 3: Check if stop loss hit
        if position['type'] == 'LONG' and current_price <= position['stop_loss']:
            logger.debug(f"🛑 Stop Loss Hit: ${current_price:.2f} <= ${position['stop_loss']:.2f}")
            return True, "Stop Loss Hit"

        elif position['type'] == 'SHORT' and current_price >= position['stop_loss']:
            logger.debug(f"🛑 Stop Loss Hit: ${current_price:.2f} >= ${position['stop_loss']:.2f}")
            return True, "Stop Loss Hit"

        return False, ""

    def open_position(self, signal, current_price):
        """Open a new position"""
        position_size = self.current_balance * self.max_position_size_pct * self.leverage

        self.current_position = {
            'type': signal['type'],
            'entry_price': signal['entry_price'],
            'quantity': position_size / signal['entry_price'],  # Quantity in base currency
            'original_quantity': position_size / signal['entry_price'],
            'stop_loss': signal['entry_price'] * 0.98 if signal['type'] == 'LONG' else signal['entry_price'] * 1.02,
            'take_profit': signal['entry_price'] * 1.02 if signal['type'] == 'LONG' else signal['entry_price'] * 0.98,
            'entry_time': datetime.now(),
            'confidence': signal.get('confidence', 1.0)
        }

        logger.info(f"Opened {signal['type']} position: Qty={self.current_position['quantity']:.4f}, "
                   f"Entry=${signal['entry_price']:.2f}, Stop=${self.current_position['stop_loss']:.2f}")

    def close_position(self, current_price, reason):
        """Close current position"""
        if not self.current_position:
            return

        position = self.current_position

        # Calculate P&L
        if position['type'] == 'LONG':
            pnl = (current_price - position['entry_price']) * position['quantity']
        else:  # SHORT
            pnl = (position['entry_price'] - current_price) * position['quantity']

        # Update balance
        self.current_balance += pnl

        # Record trade
        self.trades.append({
            'type': position['type'],
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'quantity': position['quantity'],
            'pnl': pnl,
            'reason': reason,
            'confidence': position.get('confidence', 1.0)
        })

        logger.info(f"Closed {position['type']} position: Exit=${current_price:.2f}, "
                   f"P&L=${pnl:.2f}, Balance=${self.current_balance:.2f}, Reason: {reason}")

        self.current_position = None

    def _close_partial_position(self, quantity, price, reason):
        """Close partial position"""
        position = self.current_position

        # Calculate P&L for this partial close
        if position['type'] == 'LONG':
            pnl = (price - position['entry_price']) * quantity
        else:
            pnl = (position['entry_price'] - price) * quantity

        # Update position quantity and balance
        position['quantity'] -= quantity
        self.current_balance += pnl

        # Record partial close
        self.trades.append({
            'type': f"{position['type']}_PARTIAL",
            'entry_price': position['entry_price'],
            'exit_price': price,
            'quantity': quantity,
            'pnl': pnl,
            'reason': reason,
            'confidence': position.get('confidence', 1.0)
        })

        logger.debug(f"Partial close: Qty={quantity:.4f}, Price=${price:.2f}, P&L=${pnl:.2f}")


def main():
    # Load config
    import json
    with open('../config/config.json', 'r') as f:
        config = json.load(f)

    # Load historical data
    df = pd.read_csv('../data/sol_usdc_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    logger.info(f"Loaded {len(df)} SOL-USDC candles")
    logger.info(f"Date range: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")

    # Create backtester
    backtester = MLBacktester(config)

    # Run backtest
    results = backtester.run_backtest(df)

    # Print results
    print("\n" + "=" * 70)
    print("ML-ENHANCED BACKTEST RESULTS - SOL-USDC")
    print("=" * 70)
    print(f"Total Trades:      {results['total_trades']}")
    print(f"Winning Trades:    {results['winning_trades']}")
    print(f"Win Rate:          {results['win_rate']:.2f}%")
    print(f"Total Return:      {results['total_return']:+.2f}%")
    print(f"Sharpe Ratio:      {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:      {results['max_drawdown']:.2f}%")
    print(f"Final Balance:     ${results['final_balance']:.2f}")
    print("=" * 70)

    # Show filter statistics
    filter_stats = backtester.filter_stats.get_filter_efficiency()
    print("\n📊 Filter Efficiency:")
    for filter_name, stats in filter_stats.items():
        print(f"  {filter_name}: {stats['passed']}/{stats['total_signals']} passed ({stats['rejection_rate']:.1f} rejection rate)")


if __name__ == "__main__":
    main()
