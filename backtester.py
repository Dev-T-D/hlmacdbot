"""
Advanced Backtesting System

Features:
- Slippage modeling
- Trading fees
- Realistic order execution
- Performance metrics
- Trade journal
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum

from macd_strategy import MACDStrategy
from risk_manager import RiskManager, TrailingStopLoss
from exceptions import DailyLimitError

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class Trade:
    """Represents a completed trade"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    position_type: str  # "LONG" or "SHORT"
    entry_slippage: float = 0.0
    exit_slippage: float = 0.0
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    leverage: int = 1


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_balance: float = 10000.0
    maker_fee: float = 0.0002  # 0.02% maker fee
    taker_fee: float = 0.0004  # 0.04% taker fee (market orders)
    slippage_pct: float = 0.001  # 0.1% slippage for market orders
    leverage: int = 10
    max_position_size_pct: float = 0.1
    max_daily_loss_pct: float = 0.05
    max_trades_per_day: int = 10
    trailing_stop_enabled: bool = False
    trailing_stop_config: Dict = field(default_factory=lambda: {
        'trail_percent': 2.0,
        'activation_percent': 1.0,
        'update_threshold_percent': 0.5
    })
    multi_timeframe_enabled: bool = False
    higher_timeframe: Optional[str] = None
    # Strategy parameters
    strategy_config: Dict = field(default_factory=dict)


class Backtester:
    """
    Advanced backtesting engine with slippage, fees, and realistic execution
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtester
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        
        # Initialize strategy with config parameters
        strategy_params = config.strategy_config if hasattr(config, 'strategy_config') else {}
        self.strategy = MACDStrategy(
            fast_length=strategy_params.get('fast_length', 12),
            slow_length=strategy_params.get('slow_length', 26),
            signal_length=strategy_params.get('signal_length', 9),
            risk_reward_ratio=strategy_params.get('risk_reward_ratio', 2.0),
            rsi_period=strategy_params.get('rsi_period', 14),
            rsi_oversold=strategy_params.get('rsi_oversold', 30.0),
            rsi_overbought=strategy_params.get('rsi_overbought', 70.0),
            min_histogram_strength=strategy_params.get('min_histogram_strength', 0.0),
            require_volume_confirmation=strategy_params.get('require_volume_confirmation', True),
            volume_period=strategy_params.get('volume_period', 20),
            min_trend_strength=strategy_params.get('min_trend_strength', 0.0),
            strict_long_conditions=strategy_params.get('strict_long_conditions', True),
            disable_long_trades=strategy_params.get('disable_long_trades', False),
        )
        
        self.risk_manager = RiskManager(
            max_position_size_pct=config.max_position_size_pct,
            max_daily_loss_pct=config.max_daily_loss_pct,
            max_trades_per_day=config.max_trades_per_day,
            leverage=config.leverage,
            exchange='hyperliquid'
        )
        
        # State
        self.balance = config.initial_balance
        self.initial_balance = config.initial_balance
        self.current_position: Optional[Dict] = None
        self.trailing_stop: Optional[TrailingStopLoss] = None
        
        if config.trailing_stop_enabled:
            self.trailing_stop = TrailingStopLoss(
                trail_percent=config.trailing_stop_config['trail_percent'],
                activation_percent=config.trailing_stop_config['activation_percent'],
                update_threshold_percent=config.trailing_stop_config['update_threshold_percent']
            )
        
        # Trade journal
        self.trades: List[Trade] = []
        self.daily_pnl: List[float] = []
        self.daily_balance: List[float] = []
        
        # Performance tracking
        self.equity_curve: List[float] = [config.initial_balance]
        self.max_drawdown: float = 0.0
        self.max_drawdown_pct: float = 0.0
        self.peak_balance: float = config.initial_balance
        
        # Order queue (for realistic execution)
        self.pending_orders: List[Dict] = []
        
    def calculate_slippage(self, price: float, order_type: OrderType, 
                          is_entry: bool = True) -> float:
        """
        Calculate slippage for an order
        
        Args:
            price: Base price
            order_type: Market or limit order
            is_entry: True for entry, False for exit
            
        Returns:
            Slippage amount (positive = worse execution)
        """
        if order_type == OrderType.LIMIT:
            return 0.0  # Limit orders execute at limit price (no slippage)
        
        # Market orders have slippage
        # Entry slippage: buy at slightly higher price (slippage_pct)
        # Exit slippage: sell at slightly lower price (slippage_pct)
        slippage_multiplier = 1.0 if is_entry else -1.0
        slippage = price * self.config.slippage_pct * slippage_multiplier
        
        return slippage
    
    def calculate_fee(self, notional_value: float, order_type: OrderType) -> float:
        """
        Calculate trading fee
        
        Args:
            notional_value: Order notional value (price * quantity)
            order_type: Market (taker) or limit (maker) order
            
        Returns:
            Fee amount
        """
        fee_rate = self.config.taker_fee if order_type == OrderType.MARKET else self.config.maker_fee
        return notional_value * fee_rate
    
    def execute_order(self, price: float, quantity: float, order_type: OrderType,
                     is_entry: bool = True) -> Tuple[float, float, float]:
        """
        Execute an order with slippage and fees
        
        Args:
            price: Base price
            quantity: Order quantity
            order_type: Market or limit order
            is_entry: True for entry, False for exit
            
        Returns:
            Tuple of (execution_price, slippage, fee)
        """
        slippage = self.calculate_slippage(price, order_type, is_entry)
        execution_price = price + slippage
        
        notional_value = execution_price * quantity
        fee = self.calculate_fee(notional_value, order_type)
        
        return execution_price, slippage, fee
    
    def open_position(self, signal: Dict, current_candle: pd.Series, 
                     df: pd.DataFrame, candle_index: int) -> bool:
        """
        Open a position based on signal
        
        Args:
            signal: Entry signal dictionary
            current_candle: Current candle data
            df: Full DataFrame
            candle_index: Current candle index
            
        Returns:
            True if position opened successfully
        """
        if self.current_position is not None:
            return False  # Already have position
        
        # Calculate position size
        try:
            size_info = self.risk_manager.calculate_position_size(
                balance=self.balance,
                entry_price=signal['entry_price'],
                stop_loss=signal['stop_loss']
            )
        except ValueError as e:
            logger.debug(f"Cannot open position: {e}")
            return False
        
        quantity = size_info['quantity']
        entry_price = signal['entry_price']
        
        # Execute order (market order executes at current candle's close + slippage)
        execution_price, slippage, fee = self.execute_order(
            price=current_candle['close'],
            quantity=quantity,
            order_type=OrderType.MARKET,
            is_entry=True
        )
        
        # Check if we have enough balance (including fees)
        notional_value = execution_price * quantity
        total_cost = notional_value / self.config.leverage + fee  # Margin + fee
        
        if total_cost > self.balance:
            logger.debug(f"Insufficient balance: need ${total_cost:.2f}, have ${self.balance:.2f}")
            return False
        
        # Open position
        self.current_position = {
            'type': signal['type'],
            'entry_price': execution_price,
            'quantity': quantity,
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'entry_time': current_candle.get('timestamp', candle_index),
            'entry_slippage': slippage,
            'entry_fee': fee,
            'margin_used': total_cost - fee
        }
        
        # Deduct margin and fee from balance
        self.balance -= total_cost
        
        # Initialize trailing stop if enabled
        if self.trailing_stop:
            self.trailing_stop.initialize_position(
                entry_price=execution_price,
                initial_stop_loss=signal['stop_loss'],
                position_type=signal['type']
            )
        
        logger.debug(
            f"Position opened: {signal['type']} {quantity} @ ${execution_price:.2f} "
            f"(slippage: ${slippage:.2f}, fee: ${fee:.2f})"
        )
        
        return True
    
    def close_position(self, exit_price: float, exit_reason: str, 
                      current_candle: pd.Series) -> Optional[Trade]:
        """
        Close current position
        
        Args:
            exit_price: Exit price
            exit_reason: Reason for exit
            current_candle: Current candle data
            
        Returns:
            Trade object if position was closed, None otherwise
        """
        if self.current_position is None:
            return None
        
        position = self.current_position
        quantity = position['quantity']
        entry_price = position['entry_price']
        position_type = position['type']
        
        # Execute exit order (market order executes at current candle's close - slippage)
        execution_price, slippage, fee = self.execute_order(
            price=current_candle['close'],
            quantity=quantity,
            order_type=OrderType.MARKET,
            is_entry=False
        )
        
        # Calculate P&L
        if position_type == "LONG":
            pnl = (execution_price - entry_price) * quantity
        else:  # SHORT
            pnl = (entry_price - execution_price) * quantity
        
        # Subtract fees
        total_fees = position['entry_fee'] + fee
        net_pnl = pnl - total_fees
        
        # Calculate P&L percentage based on margin used
        margin_used = position['margin_used']
        pnl_pct = (net_pnl / margin_used) * 100 if margin_used > 0 else 0.0
        
        # Return margin and add P&L
        self.balance += margin_used + net_pnl
        
        # Create trade record
        trade = Trade(
            entry_time=position['entry_time'],
            exit_time=current_candle.get('timestamp', datetime.now()),
            entry_price=entry_price,
            exit_price=execution_price,
            quantity=quantity,
            position_type=position_type,
            entry_slippage=position['entry_slippage'],
            exit_slippage=slippage,
            entry_fee=position['entry_fee'],
            exit_fee=fee,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            leverage=self.config.leverage
        )
        
        self.trades.append(trade)
        
        # Update equity curve
        self.equity_curve.append(self.balance)
        
        # Update max drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        else:
            drawdown = self.peak_balance - self.balance
            drawdown_pct = (drawdown / self.peak_balance) * 100
            if drawdown_pct > self.max_drawdown_pct:
                self.max_drawdown = drawdown
                self.max_drawdown_pct = drawdown_pct
        
        logger.debug(
            f"Position closed: {position_type} @ ${execution_price:.2f} "
            f"(slippage: ${slippage:.2f}, fee: ${fee:.2f}, P&L: ${net_pnl:.2f})"
        )
        
        # Clear position
        self.current_position = None
        if self.trailing_stop:
            self.trailing_stop.reset()
        
        return trade
    
    def check_exit_conditions(self, df: pd.DataFrame, candle_index: int) -> Tuple[bool, str, float]:
        """
        Check if position should be closed
        
        Args:
            df: DataFrame with indicators
            candle_index: Current candle index
            
        Returns:
            Tuple of (should_exit, reason, exit_price)
        """
        if self.current_position is None:
            return False, "", 0.0
        
        current_candle = df.iloc[candle_index]
        current_price = current_candle['close']
        position = self.current_position
        position_type = position['type']
        
        # Check stop loss
        stop_loss = position['stop_loss']
        if position_type == "LONG" and current_price <= stop_loss:
            return True, "Stop Loss", stop_loss
        elif position_type == "SHORT" and current_price >= stop_loss:
            return True, "Stop Loss", stop_loss
        
        # Check take profit
        take_profit = position['take_profit']
        if position_type == "LONG" and current_price >= take_profit:
            return True, "Take Profit", take_profit
        elif position_type == "SHORT" and current_price <= take_profit:
            return True, "Take Profit", take_profit
        
        # Check trailing stop if enabled
        if self.trailing_stop:
            stop_updated, new_stop_loss, _ = self.trailing_stop.update(current_price)
            if stop_updated:
                position['stop_loss'] = new_stop_loss
            
            stop_hit, reason = self.trailing_stop.check_stop_hit(current_price)
            if stop_hit:
                return True, reason, new_stop_loss
        
        # Check MACD exit signal
        if candle_index >= 1:  # Need at least 2 candles for crossover detection
            should_exit, exit_reason = self.strategy.check_exit_signal(
                df.iloc[:candle_index+1], 
                position_type
            )
            if should_exit:
                return True, exit_reason, current_price
        
        return False, "", 0.0
    
    def run_backtest(self, df: pd.DataFrame, start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            df: DataFrame with OHLCV data
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("=" * 70)
        logger.info("STARTING BACKTEST")
        logger.info("=" * 70)
        logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
        logger.info(f"Period: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
        logger.info(f"Total Candles: {len(df)}")
        logger.info("=" * 70)
        
        # Filter by date range if specified
        if start_date or end_date:
            if 'timestamp' in df.columns:
                if start_date:
                    df = df[df['timestamp'] >= start_date]
                if end_date:
                    df = df[df['timestamp'] <= end_date]
        
        if len(df) < self.strategy.min_candles:
            raise ValueError(f"Insufficient data: {len(df)} candles, need at least {self.strategy.min_candles}")
        
        # Calculate indicators for entire dataset
        df = self.strategy.calculate_indicators(df)
        
        # Reset state
        self.balance = self.initial_balance
        self.current_position = None
        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0
        
        # Track daily stats
        current_date = None
        daily_pnl = 0.0
        daily_trades = 0
        
        # Iterate through candles
        for i in range(self.strategy.min_candles, len(df)):
            current_candle = df.iloc[i]
            # Extract date from timestamp (handle both datetime and timestamp formats)
            timestamp_value = current_candle.get('timestamp')
            if pd.isna(timestamp_value):
                timestamp_value = df.index[i] if i < len(df.index) else i
            candle_date = pd.to_datetime(timestamp_value).date()
            
            # Reset daily stats if new day (check BEFORE processing trades)
            if current_date is not None and current_date != candle_date:
                # Save previous day's stats
                self.daily_pnl.append(daily_pnl)
                self.daily_balance.append(self.balance)
                logger.debug(f"Day reset: {current_date} -> {candle_date}, Trades: {daily_trades}, P&L: ${daily_pnl:.2f}")
            
            # Reset for new day (or first day)
            if current_date != candle_date:
                current_date = candle_date
                daily_pnl = 0.0
                daily_trades = 0
                # Reset risk manager daily stats
                self.risk_manager.reset_daily_stats(self.balance)
                self.risk_manager.daily_trades = 0  # Reset trade count
                logger.debug(f"Daily reset: {candle_date}, Balance: ${self.balance:.2f}")
            
            # Check exit conditions if we have a position
            if self.current_position:
                should_exit, exit_reason, exit_price = self.check_exit_conditions(df, i)
                
                if should_exit:
                    trade = self.close_position(exit_price, exit_reason, current_candle)
                    if trade:
                        daily_pnl += trade.pnl
                        daily_trades += 1
                        # update_daily_pnl already increments daily_trades, so don't double-count
                        self.risk_manager.update_daily_pnl(trade.pnl)
            
            # Check for entry signals if no position
            if self.current_position is None:
                # Check risk limits
                allowed, reason = self.risk_manager.check_risk_limits(self.balance)
                
                if allowed:
                    # Get signal from current data up to this point
                    signal = self.strategy.check_entry_signal(df.iloc[:i+1])
                    
                    if signal:
                        # Multi-timeframe check if enabled
                        # Note: Multi-timeframe requires separate higher TF data
                        # For simplicity, skip multi-TF check in backtest
                        # (would need to fetch higher TF data separately)
                        
                        # Try to open position
                        if self.open_position(signal, current_candle, df, i):
                            daily_trades += 1
                            # Note: daily_trades count is managed by risk_manager.update_daily_pnl()
                            # when position closes, so we don't increment here for entry
                            # Note: P&L will be tracked when position closes
        
        # Close any open position at end
        if self.current_position:
            final_price = df.iloc[-1]['close']
            trade = self.close_position(final_price, "Backtest End", df.iloc[-1])
            if trade:
                daily_pnl += trade.pnl
                self.daily_pnl.append(daily_pnl)
                self.daily_balance.append(self.balance)
        
        # Calculate performance metrics
        results = self.calculate_performance_metrics()
        
        logger.info("=" * 70)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 70)
        self.print_results(results)
        
        return results
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'final_balance': self.balance,
                'initial_balance': self.initial_balance,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'avg_trade_duration_hours': 0.0,
                'trades': [],
                'equity_curve': self.equity_curve
            }
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0.0
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_return = self.balance - self.initial_balance
        total_return_pct = (total_return / self.initial_balance) * 100
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
        
        # Sharpe ratio (simplified - using daily returns)
        if len(self.daily_pnl) > 1:
            daily_returns = np.diff(self.daily_balance) / self.daily_balance[:-1]
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown (already calculated)
        max_dd = self.max_drawdown
        max_dd_pct = self.max_drawdown_pct
        
        # Average trade duration
        if self.trades:
            durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in self.trades]
            avg_duration_hours = np.mean(durations)
        else:
            avg_duration_hours = 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'final_balance': self.balance,
            'initial_balance': self.initial_balance,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'avg_trade_duration_hours': avg_duration_hours,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }
    
    def print_results(self, results: Dict) -> None:
        """Print backtest results"""
        logger.info(f"Total Trades:        {results['total_trades']}")
        logger.info(f"Winning Trades:      {results['winning_trades']}")
        logger.info(f"Losing Trades:       {results['losing_trades']}")
        logger.info(f"Win Rate:            {results['win_rate']:.2f}%")
        logger.info(f"Total P&L:           ${results['total_pnl']:,.2f}")
        logger.info(f"Total Return:        ${results['total_return']:,.2f} ({results['total_return_pct']:+.2f}%)")
        logger.info(f"Final Balance:       ${results['final_balance']:,.2f}")
        logger.info(f"Average Win:          ${results['avg_win']:,.2f}")
        logger.info(f"Average Loss:         ${results['avg_loss']:,.2f}")
        logger.info(f"Profit Factor:        {results['profit_factor']:.2f}")
        logger.info(f"Sharpe Ratio:         {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown:         ${results['max_drawdown']:,.2f} ({results['max_drawdown_pct']:.2f}%)")
        logger.info(f"Avg Trade Duration:   {results['avg_trade_duration_hours']:.1f} hours")
        logger.info("=" * 70)
    
    def export_trades(self, filename: str) -> None:
        """
        Export trade journal to CSV
        
        Args:
            filename: Output filename
        """
        if not self.trades:
            logger.warning("No trades to export")
            return
        
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'position_type': trade.position_type,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'entry_slippage': trade.entry_slippage,
                'exit_slippage': trade.exit_slippage,
                'entry_fee': trade.entry_fee,
                'exit_fee': trade.exit_fee,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'exit_reason': trade.exit_reason,
                'leverage': trade.leverage
            })
        
        df_trades = pd.DataFrame(trades_data)
        df_trades.to_csv(filename, index=False)
        logger.info(f"Trade journal exported to {filename}")

