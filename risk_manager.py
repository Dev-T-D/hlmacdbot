"""
Risk Management Module

Handles position sizing, risk limits, safety checks, and trailing stop-loss

"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime

from constants import MIN_STOP_DISTANCE_PCT
from exceptions import (
    RiskManagementError,
    PositionSizeError,
    DailyLimitError
)
from input_sanitizer import InputSanitizer

logger = logging.getLogger(__name__)


class TrailingStopLoss:
    """Trailing Stop-Loss Manager"""
    
    def __init__(self,
                 trail_percent: float = 2.0,
                 activation_percent: float = 1.0,
                 update_threshold_percent: float = 0.5):
        """
        Initialize Trailing Stop-Loss
        
        Args:
            trail_percent: Distance to trail behind best price (2.0 = 2%)
            activation_percent: Profit threshold to activate trailing (1.0 = 1%)
            update_threshold_percent: Minimum movement to trigger update (0.5 = 0.5%)
        """
        self.trail_percent = trail_percent / 100.0  # Convert to decimal
        self.activation_percent = activation_percent / 100.0
        self.update_threshold_percent = update_threshold_percent / 100.0
        
        # Tracking variables
        self.is_active = False
        self.best_price = None
        self.current_stop_loss = None
        self.entry_price = None
        self.position_type = None
        self.last_update_time = None
        
        logger.debug(f"Trailing Stop-Loss initialized: Trail={trail_percent}%, "
                   f"Activation={activation_percent}%, "
                   f"Update Threshold={update_threshold_percent}%")
    
    def initialize_position(self,
                          entry_price: float,
                          initial_stop_loss: float,
                          position_type: str):
        """
        Initialize trailing stop for a new position
        
        Args:
            entry_price: Position entry price
            initial_stop_loss: Initial stop loss price
            position_type: "LONG" or "SHORT"
        """
        self.entry_price = entry_price
        self.current_stop_loss = initial_stop_loss
        self.position_type = position_type
        self.best_price = entry_price
        self.is_active = False
        self.last_update_time = datetime.now()
        
        logger.debug(f"Trailing stop initialized for {position_type} position: "
                   f"Entry=${entry_price:.2f}, Initial SL=${initial_stop_loss:.2f}")
    
    def update(self, current_price: float) -> Tuple[bool, float, str]:
        """
        Update trailing stop based on current price
        
        Args:
            current_price: Current market price
            
        Returns:
            Tuple of (stop_updated, new_stop_loss, update_message)
        """
        if not self.entry_price or not self.position_type:
            return False, self.current_stop_loss, "Not initialized"
        
        # Calculate current profit percentage
        if self.position_type == "LONG":
            profit_pct = (current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            profit_pct = (self.entry_price - current_price) / self.entry_price
        
        # Check if we should activate trailing
        if not self.is_active:
            if profit_pct >= self.activation_percent:
                self.is_active = True
                logger.info(f"🟢 Trailing stop ACTIVATED at {profit_pct*100:.2f}% profit")
            else:
                return False, self.current_stop_loss, "Waiting for activation threshold"
        
        # Update best price and trailing stop
        stop_updated = False
        update_message = ""
        
        if self.position_type == "LONG":
            # For LONG: trail below the highest price
            if current_price > self.best_price:
                old_best = self.best_price
                self.best_price = current_price
                
                # Calculate new stop loss
                new_stop_loss = self.best_price * (1 - self.trail_percent)
                
                # Only update if movement exceeds threshold
                if new_stop_loss > self.current_stop_loss:
                    stop_change_pct = ((new_stop_loss - self.current_stop_loss) / 
                                      self.current_stop_loss)
                    
                    if stop_change_pct >= self.update_threshold_percent:
                        old_stop = self.current_stop_loss
                        self.current_stop_loss = new_stop_loss
                        self.last_update_time = datetime.now()
                        stop_updated = True
                        
                        update_message = (f"LONG stop trailed: ${old_stop:.2f} → "
                                        f"${new_stop_loss:.2f} (Best: ${old_best:.2f} "
                                        f"→ ${self.best_price:.2f})")
                        logger.info(f"📈 {update_message}")
        
        else:  # SHORT
            # For SHORT: trail above the lowest price
            if current_price < self.best_price:
                old_best = self.best_price
                self.best_price = current_price
                
                # Calculate new stop loss
                new_stop_loss = self.best_price * (1 + self.trail_percent)
                
                # Only update if movement exceeds threshold
                if new_stop_loss < self.current_stop_loss:
                    stop_change_pct = ((self.current_stop_loss - new_stop_loss) / 
                                      self.current_stop_loss)
                    
                    if stop_change_pct >= self.update_threshold_percent:
                        old_stop = self.current_stop_loss
                        self.current_stop_loss = new_stop_loss
                        self.last_update_time = datetime.now()
                        stop_updated = True
                        
                        update_message = (f"SHORT stop trailed: ${old_stop:.2f} → "
                                        f"${new_stop_loss:.2f} (Best: ${old_best:.2f} "
                                        f"→ ${self.best_price:.2f})")
                        logger.info(f"📉 {update_message}")
        
        return stop_updated, self.current_stop_loss, update_message
    
    def check_stop_hit(self, current_price: float) -> Tuple[bool, str]:
        """
        Check if trailing stop has been hit
        
        Args:
            current_price: Current market price
            
        Returns:
            Tuple of (stop_hit, reason)
        """
        if not self.is_active or not self.current_stop_loss:
            return False, ""
        
        if self.position_type == "LONG":
            if current_price <= self.current_stop_loss:
                return True, f"Trailing Stop Hit (${self.current_stop_loss:.2f})"
        else:  # SHORT
            if current_price >= self.current_stop_loss:
                return True, f"Trailing Stop Hit (${self.current_stop_loss:.2f})"
        
        return False, ""
    
    def get_status(self) -> Dict:
        """Get current trailing stop status"""
        if not self.entry_price:
            return {"initialized": False}
        
        return {
            "initialized": True,
            "active": self.is_active,
            "position_type": self.position_type,
            "entry_price": self.entry_price,
            "best_price": self.best_price,
            "current_stop_loss": self.current_stop_loss,
            "trail_percent": self.trail_percent * 100,
            "last_update": self.last_update_time.isoformat() if self.last_update_time else None
        }
    
    def reset(self):
        """Reset trailing stop for new position"""
        self.is_active = False
        self.best_price = None
        self.current_stop_loss = None
        self.entry_price = None
        self.position_type = None
        self.last_update_time = None


class RiskManager:
    """Risk Management for Futures Trading"""
    
    # Exchange-specific limits (defaults - should be overridden per exchange)
    EXCHANGE_LIMITS = {
        'bitunix': {
            'min_qty': 0.001,
            'max_qty': 1000000.0,
            'qty_precision': 3,
            'min_notional': 5.0,  # Minimum $5 USDT
            'max_notional': 10000000.0,  # Maximum $10M USDT
        },
        'hyperliquid': {
            'min_qty': 0.001,
            'max_qty': 1000000.0,
            'qty_precision': 3,
            'min_notional': 1.0,  # Minimum $1 USDT
            'max_notional': 10000000.0,  # Maximum $10M USDT
        }
    }
    
    def __init__(self,
                 max_position_size_pct: float = 0.1,
                 max_daily_loss_pct: float = 0.05,
                 max_trades_per_day: int = 10,
                 leverage: int = 10,
                 exchange: str = 'hyperliquid'):
        """
        Initialize Risk Manager
        
        Args:
            max_position_size_pct: Maximum position size as % of equity (0.1 = 10%)
            max_daily_loss_pct: Maximum daily loss as % of equity (0.05 = 5%)
            max_trades_per_day: Maximum number of trades per day
            leverage: Leverage multiplier
            exchange: Exchange name ('hyperliquid' or 'bitunix') for limit validation
        """
        self.max_position_size_pct = max_position_size_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_trades_per_day = max_trades_per_day
        self.leverage = leverage
        self.exchange = exchange.lower()
        
        # Get exchange-specific limits
        self.exchange_limits = self.EXCHANGE_LIMITS.get(
            self.exchange, 
            self.EXCHANGE_LIMITS['hyperliquid']  # Default to Hyperliquid limits
        )
        
        # Track daily stats
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.starting_balance = 0.0
        
    def reset_daily_stats(self, current_balance: float):
        """Reset daily tracking stats"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.starting_balance = current_balance
        logger.debug(f"Daily stats reset. Starting balance: ${current_balance:,.2f}")
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking"""
        self.daily_pnl += pnl
        self.daily_trades += 1
        logger.debug(f"Daily P&L: ${self.daily_pnl:,.2f} | Trades: {self.daily_trades}")
    
    def calculate_total_exposure(self, positions: list) -> Dict:
        """
        Calculate total exposure and risk from existing positions
        
        Args:
            positions: List of position dicts, each with 'entry_price', 'quantity', 'stop_loss', 'type'
            
        Returns:
            Dict with total_notional_value, total_risk, total_risk_pct
        """
        total_notional_value = 0.0
        total_risk = 0.0
        
        for pos in positions:
            if not pos:
                continue
                
            entry_price = float(pos.get('entry_price', 0))
            quantity = float(pos.get('quantity', 0))
            stop_loss = float(pos.get('stop_loss', 0))
            
            if entry_price > 0 and quantity > 0 and stop_loss > 0:
                notional_value = entry_price * quantity
                risk_per_unit = abs(entry_price - stop_loss)
                position_risk = quantity * risk_per_unit
                
                total_notional_value += notional_value
                total_risk += position_risk
        
        return {
            'total_notional_value': total_notional_value,
            'total_risk': total_risk,
            'total_risk_pct': 0.0  # Will be calculated with balance
        }
    
    def calculate_position_size(self, 
                               balance: float,
                               entry_price: float,
                               stop_loss: float,
                               min_qty: Optional[float] = None,
                               qty_precision: Optional[int] = None,
                               existing_positions: Optional[list] = None) -> Dict:
        """
        Calculate position size based on risk management rules with exchange limit validation
        
        Args:
            balance: Available balance (USDT)
            entry_price: Entry price
            stop_loss: Stop loss price
            min_qty: Minimum order quantity (overrides exchange default if provided)
            qty_precision: Quantity decimal precision (overrides exchange default if provided)
            existing_positions: Optional list of existing position dicts to account for exposure
            
        Returns:
            Dict with quantity, notional value, and risk details
            
        Raises:
            ValueError: If calculated position size violates exchange limits
        """
        # Sanitize and validate inputs
        balance = InputSanitizer.sanitize_price(balance, 'balance')
        entry_price = InputSanitizer.sanitize_price(entry_price, 'entry_price')
        stop_loss = InputSanitizer.sanitize_price(stop_loss, 'stop_loss')
        
        # Calculate total exposure from existing positions
        existing_exposure = {'total_notional_value': 0.0, 'total_risk': 0.0}
        if existing_positions:
            existing_exposure = self.calculate_total_exposure(existing_positions)
            existing_exposure['total_risk_pct'] = (existing_exposure['total_risk'] / balance * 100) if balance > 0 else 0.0
            
            logger.debug(
                f"Existing exposure: Notional=${existing_exposure['total_notional_value']:,.2f}, "
                f"Risk=${existing_exposure['total_risk']:,.2f} ({existing_exposure['total_risk_pct']:.2f}%)"
            )
        
        # Use exchange limits or provided overrides
        exchange_min_qty = self.exchange_limits['min_qty']
        exchange_max_qty = self.exchange_limits['max_qty']
        exchange_qty_precision = self.exchange_limits['qty_precision']
        exchange_min_notional = self.exchange_limits['min_notional']
        exchange_max_notional = self.exchange_limits['max_notional']
        
        effective_min_qty = min_qty if min_qty is not None else exchange_min_qty
        effective_qty_precision = qty_precision if qty_precision is not None else exchange_qty_precision
        
        # Calculate maximum position value based on % of equity
        # Account for existing exposure by reducing available balance proportionally
        max_position_value = balance * self.max_position_size_pct * self.leverage
        
        # If we have existing positions, reduce available position size to account for them
        # This ensures we don't exceed total risk limits across all positions
        if existing_exposure['total_notional_value'] > 0:
            # Calculate what percentage of max position size is already used
            max_total_notional = balance * self.max_position_size_pct * self.leverage
            used_pct = existing_exposure['total_notional_value'] / max_total_notional if max_total_notional > 0 else 0
            
            # Reduce available position size proportionally
            if used_pct >= 1.0:
                logger.warning(
                    f"Existing positions already use {used_pct*100:.1f}% of max position size. "
                    f"Cannot open new position without exceeding limits."
                )
                raise PositionSizeError(
                    f"Cannot open new position: existing exposure exceeds maximum allowed. "
                    f"Existing exposure: ${existing_exposure['total_notional_value']:,.2f}, "
                    f"Maximum allowed: ${max_total_notional:,.2f}, "
                    f"Excess: ${existing_exposure['total_notional_value'] - max_total_notional:,.2f}. "
                    f"Balance: ${balance:.2f}, Max position size: {self.max_position_size_pct*100:.1f}%, "
                    f"Leverage: {self.leverage}x. "
                    f"Close existing positions or increase max_position_size_pct to open new position."
                )
            
            # Reduce available position size
            available_pct = 1.0 - used_pct
            max_position_value = max_position_value * available_pct
            
            logger.debug(
                f"Accounting for existing exposure: {used_pct*100:.1f}% used, "
                f"{available_pct*100:.1f}% available for new position"
            )
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            raise PositionSizeError(
                f"Invalid risk per unit: {risk_per_unit:.8f}. "
                f"Entry price: {entry_price:.8f}, Stop loss: {stop_loss:.8f}. "
                f"Entry price and stop loss are too close or identical (difference: {risk_per_unit:.8f}). "
                f"Stop loss must be at least {MIN_STOP_DISTANCE_PCT*100:.1f}% away from entry price. "
                f"Check stop loss calculation in strategy."
            )
        
        # Position size based on max position value
        quantity_by_value = max_position_value / entry_price
        
        # Calculate actual risk
        position_risk = quantity_by_value * risk_per_unit
        risk_pct = (position_risk / balance) * 100
        
        # Round to precision
        quantity = round(quantity_by_value, effective_qty_precision)
        
        # Validate against exchange minimum quantity
        if quantity < effective_min_qty:
            logger.warning(
                f"Calculated quantity {quantity} below exchange minimum {effective_min_qty}. "
                f"Adjusting to minimum."
            )
            quantity = effective_min_qty
        
        # Validate against exchange maximum quantity
        if quantity > exchange_max_qty:
            logger.warning(
                f"Calculated quantity {quantity} exceeds exchange maximum {exchange_max_qty}. "
                f"Capping at maximum."
            )
            quantity = exchange_max_qty
            # Recalculate notional value with capped quantity
            max_position_value = quantity * entry_price
        
        # Calculate notional value
        notional_value = quantity * entry_price
        
        # Validate against exchange minimum notional value
        if notional_value < exchange_min_notional:
            raise PositionSizeError(
                f"Position size below exchange minimum. "
                f"Notional value: ${notional_value:.2f}, Minimum required: ${exchange_min_notional:.2f}, "
                f"Shortfall: ${exchange_min_notional - notional_value:.2f}. "
                f"Entry price: {entry_price:.8f}, Quantity: {quantity:.8f}. "
                f"Increase position size to meet exchange minimum order size."
            )
        
        # Validate against exchange maximum notional value
        if notional_value > exchange_max_notional:
            logger.warning(
                f"Position notional value ${notional_value:.2f} exceeds exchange maximum "
                f"${exchange_max_notional:.2f}. Capping position size."
            )
            # Recalculate quantity to meet max notional
            quantity = exchange_max_notional / entry_price
            quantity = round(quantity, effective_qty_precision)
            notional_value = quantity * entry_price
            # Recalculate risk with adjusted quantity
            position_risk = quantity * risk_per_unit
            risk_pct = (position_risk / balance) * 100
        
        # Final validation: Ensure quantity still meets minimum after adjustments
        if quantity < effective_min_qty:
            raise PositionSizeError(
                f"Cannot create position meeting exchange minimum requirements. "
                f"Minimum quantity: {effective_min_qty}, Minimum notional: ${exchange_min_notional:.2f}, "
                f"Entry price: ${entry_price:.8f}, Calculated quantity: {quantity:.8f}. "
                f"Position size is too small after adjustments. "
                f"Check: 1) Entry price validity, 2) Exchange minimums, 3) Position sizing logic."
            )
        
        # Validate quantity precision (step size)
        # Round to ensure it matches exchange step size requirements
        step_size = 10 ** (-effective_qty_precision)
        quantity = round(quantity / step_size) * step_size
        quantity = round(quantity, effective_qty_precision)
        
        # Final notional value with precision-adjusted quantity
        notional_value = quantity * entry_price
        
        logger.debug(
            f"Position size calculated: Qty={quantity}, Notional=${notional_value:.2f}, "
            f"Risk=${position_risk:.2f} ({risk_pct:.2f}%), "
            f"Exchange: {self.exchange}"
        )
        
        # Calculate total risk including new position
        total_risk_with_new = existing_exposure['total_risk'] + position_risk
        total_risk_pct_with_new = (total_risk_with_new / balance * 100) if balance > 0 else 0.0
        
        return {
            'quantity': quantity,
            'notional_value': notional_value,
            'position_risk': position_risk,
            'risk_pct': risk_pct,
            'leverage_used': notional_value / balance if balance > 0 else 0,
            'existing_exposure': {
                'total_notional_value': existing_exposure['total_notional_value'],
                'total_risk': existing_exposure['total_risk'],
                'total_risk_pct': existing_exposure.get('total_risk_pct', 0.0)
            },
            'total_risk_with_new': total_risk_with_new,
            'total_risk_pct_with_new': total_risk_pct_with_new,
            'exchange_limits_applied': {
                'min_qty': effective_min_qty,
                'max_qty': exchange_max_qty,
                'min_notional': exchange_min_notional,
                'max_notional': exchange_max_notional,
                'qty_precision': effective_qty_precision
            }
        }
    
    def check_risk_limits(self, balance: float) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on risk limits
        
        Args:
            balance: Current account balance
            
        Returns:
            Tuple of (allowed, reason)
        """
        # Check daily loss limit
        if self.starting_balance > 0:
            daily_loss_pct = abs(self.daily_pnl) / self.starting_balance
            
            if self.daily_pnl < 0 and daily_loss_pct >= self.max_daily_loss_pct:
                raise DailyLimitError(
                    f"Daily loss limit reached. "
                    f"Daily P&L: ${self.daily_pnl:.2f} ({daily_loss_pct*100:.2f}% of starting balance), "
                    f"Limit: {self.max_daily_loss_pct*100:.2f}%. "
                    f"Starting balance: ${self.starting_balance:.2f}, Current balance: ${balance:.2f}. "
                    f"Trading paused until daily reset. Adjust max_daily_loss_pct if needed."
                )
        
        # Check daily trade limit
        if self.daily_trades >= self.max_trades_per_day:
                raise DailyLimitError(
                    f"Daily trade limit reached. "
                    f"Trades today: {self.daily_trades}, Maximum allowed: {self.max_trades_per_day}. "
                    f"Trading paused until daily reset. Adjust max_trades_per_day in config if needed."
                )
        
        # Check minimum balance
        if balance < 10:  # Minimum $10 USDT
                raise DailyLimitError(
                    f"Account balance too low. "
                    f"Balance: ${balance:.2f}, Minimum required: $10.00. "
                    f"Insufficient funds for trading. Deposit more funds to continue."
                )
        
        return True, "OK"
    
    def validate_order(self, 
                      balance: float,
                      quantity: float,
                      entry_price: float,
                      stop_loss: float,
                      take_profit: float) -> Tuple[bool, str]:
        """
        Validate order parameters
        
        Args:
            balance: Current balance
            quantity: Order quantity
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Tuple of (valid, reason)
        """
        # Check basic risk limits first
        allowed, reason = self.check_risk_limits(balance)
        if not allowed:
            return False, reason
        
        # Validate prices
        if entry_price <= 0:
            return False, "Invalid entry price"
        
        if stop_loss <= 0:
            return False, "Invalid stop loss"
        
        if take_profit <= 0:
            return False, "Invalid take profit"
        
        # Check quantity
        if quantity <= 0:
            return False, "Invalid quantity"
        
        # Check notional value
        notional = quantity * entry_price
        if notional > balance * self.leverage:
            return False, f"Position too large: ${notional:.2f} > ${balance * self.leverage:.2f}"
        
        # Validate risk/reward makes sense
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk <= 0:
            return False, "Stop loss too close to entry"
        
        if reward / risk < 1.0:  # At least 1:1 R:R
            return False, f"Risk/Reward too low: {reward/risk:.2f}"
        
        return True, "Order validated"
    
    def get_risk_summary(self, balance: float) -> Dict:
        """Get current risk management summary"""
        daily_loss_pct = 0
        if self.starting_balance > 0:
            daily_loss_pct = (self.daily_pnl / self.starting_balance) * 100
        
        return {
            'balance': balance,
            'daily_pnl': self.daily_pnl,
            'daily_loss_pct': daily_loss_pct,
            'daily_trades': self.daily_trades,
            'max_trades_remaining': max(0, self.max_trades_per_day - self.daily_trades),
            'leverage': self.leverage,
            'max_position_pct': self.max_position_size_pct * 100,
            'max_daily_loss_pct': self.max_daily_loss_pct * 100
        }

