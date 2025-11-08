"""
Position Circuit Breaker - Risk Management Safety System

Prevents catastrophic losses by enforcing position size limits and circuit breakers.
Automatically blocks trades that exceed safety thresholds.

Features:
- Maximum position size limits (percentage of account balance)
- Maximum daily loss limits with automatic trading suspension
- Position concentration limits (prevent over-allocation to single asset)
- Sanity checks for all order parameters
- Emergency liquidation triggers
- Real-time monitoring and alerts
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerLimits:
    """Configuration for circuit breaker limits."""
    max_position_size_pct: float = 0.10  # 10% of account balance per position
    max_total_exposure_pct: float = 0.50  # 50% total exposure across all positions
    max_daily_loss_pct: float = 0.10     # 10% daily loss limit
    max_single_asset_pct: float = 0.25   # 25% allocation to single asset
    min_order_size_usd: float = 10.0     # Minimum order size in USD
    max_order_size_pct: float = 0.05     # 5% of balance per order
    sanity_check_enabled: bool = True    # Enable pre-order sanity checks


class PositionCircuitBreaker:
    """
    Circuit breaker system for position risk management.

    Monitors position sizes, account exposure, and enforces safety limits
    to prevent catastrophic losses from oversized or concentrated positions.
    """

    def __init__(self, limits: Optional[CircuitBreakerLimits] = None):
        """
        Initialize position circuit breaker.

        Args:
            limits: Circuit breaker limit configuration
        """
        self.limits = limits or CircuitBreakerLimits()

        # Daily tracking
        self.daily_start_balance = 0.0
        self.daily_trades: List[Dict[str, Any]] = []
        self.daily_pnl = 0.0
        self.trading_suspended = False
        self.suspension_reason = ""

        # Real-time monitoring
        self.current_positions: Dict[str, Dict[str, Any]] = {}
        self.account_balance = 0.0

        # Circuit breaker state
        self.circuit_breaker_tripped = False
        self.tripped_time: Optional[datetime] = None
        self.tripped_reason = ""

        logger.info(
            f"Position Circuit Breaker initialized with limits: "
            f"max_pos={self.limits.max_position_size_pct*100}%, "
            f"max_exposure={self.limits.max_total_exposure_pct*100}%, "
            f"max_daily_loss={self.limits.max_daily_loss_pct*100}%"
        )

    def update_account_balance(self, balance: float) -> None:
        """Update current account balance."""
        self.account_balance = balance

        # Reset daily tracking if this is a new day
        # (This would be called by the main bot on daily reset)

    def update_positions(self, positions: List[Dict[str, Any]]) -> None:
        """Update current positions tracking."""
        self.current_positions = {pos.get('symbol', ''): pos for pos in positions}

    def validate_position_size(
        self,
        symbol: str,
        position_size: float,
        entry_price: float,
        account_balance: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Validate position size against circuit breaker limits.

        Args:
            symbol: Trading symbol
            position_size: Position size in base currency
            entry_price: Entry price
            account_balance: Current account balance (uses self.account_balance if None)

        Returns:
            Tuple of (is_valid, error_message)
        """
        balance = account_balance or self.account_balance

        if balance <= 0:
            return False, "Invalid account balance for position validation"

        # Calculate position value in USD
        position_value = position_size * entry_price

        # Check minimum order size
        if position_value < self.limits.min_order_size_usd:
            return False, ".2f"

        # Check maximum position size limit
        max_position_value = balance * self.limits.max_position_size_pct
        if position_value > max_position_value:
            return False, ".2f"

        # Check maximum order size as percentage of balance
        if position_value > balance * self.limits.max_order_size_pct:
            return False, ".2f"

        # Check single asset concentration limit
        existing_position = self.current_positions.get(symbol, {})
        existing_size = abs(float(existing_position.get('size', 0)))
        existing_value = existing_size * float(existing_position.get('entryPrice', entry_price))

        total_asset_value = existing_value + position_value
        max_asset_value = balance * self.limits.max_single_asset_pct

        if total_asset_value > max_asset_value:
            return False, ".2f"

        return True, "Position size validated"

    def validate_total_exposure(self, account_balance: Optional[float] = None) -> Tuple[bool, str]:
        """
        Validate total account exposure against limits.

        Args:
            account_balance: Current account balance (uses self.account_balance if None)

        Returns:
            Tuple of (is_valid, error_message)
        """
        balance = account_balance or self.account_balance

        if balance <= 0:
            return False, "Invalid account balance for exposure validation"

        # Calculate total exposure
        total_exposure = 0.0
        for pos in self.current_positions.values():
            size = abs(float(pos.get('size', 0)))
            entry_price = float(pos.get('entryPrice', 0))
            total_exposure += size * entry_price

        # Check total exposure limit
        max_exposure = balance * self.limits.max_total_exposure_pct

        if total_exposure > max_exposure:
            return False, ".2f"

        return True, "Total exposure validated"

    def validate_daily_loss_limit(self, current_pnl: float, start_balance: float) -> Tuple[bool, str]:
        """
        Validate daily loss limit.

        Args:
            current_pnl: Current daily P&L
            start_balance: Starting balance for the day

        Returns:
            Tuple of (is_valid, error_message)
        """
        if start_balance <= 0:
            return False, "Invalid starting balance for daily loss validation"

        # Calculate loss percentage
        loss_pct = abs(current_pnl) / start_balance if current_pnl < 0 else 0

        if loss_pct > self.limits.max_daily_loss_pct:
            self.trading_suspended = True
            self.suspension_reason = ".2f"
            return False, self.suspension_reason

        return True, "Daily loss limit validated"

    def perform_sanity_checks(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str,
        account_balance: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Perform comprehensive sanity checks on order parameters.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            price: Order price
            order_type: Order type (LIMIT/MARKET)
            account_balance: Account balance (uses self.account_balance if None)

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.limits.sanity_check_enabled:
            return True, "Sanity checks disabled"

        balance = account_balance or self.account_balance

        # Basic parameter validation
        if not symbol or not isinstance(symbol, str):
            return False, "Invalid symbol"

        if side not in ['BUY', 'SELL']:
            return False, "Invalid order side"

        if quantity <= 0:
            return False, "Invalid order quantity"

        if price <= 0:
            return False, "Invalid order price"

        if order_type not in ['LIMIT', 'MARKET']:
            return False, "Invalid order type"

        # Quantity sanity checks
        if quantity < 0.000001:  # Minimum reasonable quantity
            return False, "Order quantity too small"

        if quantity > 1000000:  # Maximum reasonable quantity
            return False, "Order quantity unreasonably large"

        # Price sanity checks
        if price < 0.000001:  # Minimum reasonable price
            return False, "Order price too low"

        if price > 10000000:  # Maximum reasonable price
            return False, "Order price unreasonably high"

        # Account balance checks
        if balance > 0:
            order_value = quantity * price
            if order_value > balance * 0.9:  # 90% of balance sanity check
                return False, ".2f"

        # Symbol-specific checks (basic)
        if not any(char.isalpha() for char in symbol):
            return False, "Symbol must contain alphabetic characters"

        return True, "All sanity checks passed"

    def check_circuit_breaker_status(self) -> Tuple[bool, str]:
        """
        Check if circuit breaker is currently tripped.

        Returns:
            Tuple of (is_tripped, reason)
        """
        if self.circuit_breaker_tripped:
            return True, self.tripped_reason

        # Check trading suspension
        if self.trading_suspended:
            return True, self.suspension_reason

        return False, "Circuit breaker not tripped"

    def trip_circuit_breaker(self, reason: str) -> None:
        """Manually trip the circuit breaker."""
        self.circuit_breaker_tripped = True
        self.tripped_time = datetime.now(timezone.utc)
        self.tripped_reason = reason

        logger.critical("=" * 60)
        logger.critical("🚨 CIRCUIT BREAKER TRIPPED 🚨")
        logger.critical("=" * 60)
        logger.critical(f"Reason: {reason}")
        logger.critical(f"Time: {self.tripped_time.isoformat()}")
        logger.critical("All trading suspended until manual reset")
        logger.critical("=" * 60)

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker."""
        self.circuit_breaker_tripped = False
        self.tripped_time = None
        self.tripped_reason = ""

        # Also reset trading suspension
        self.trading_suspended = False
        self.suspension_reason = ""

        logger.info("Circuit breaker reset - trading can resume")

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "circuit_breaker_tripped": self.circuit_breaker_tripped,
            "tripped_reason": self.tripped_reason,
            "tripped_time": self.tripped_time.isoformat() if self.tripped_time else None,
            "trading_suspended": self.trading_suspended,
            "suspension_reason": self.suspension_reason,
            "current_positions_count": len(self.current_positions),
            "account_balance": self.account_balance,
            "limits": {
                "max_position_size_pct": self.limits.max_position_size_pct,
                "max_total_exposure_pct": self.limits.max_total_exposure_pct,
                "max_daily_loss_pct": self.limits.max_daily_loss_pct,
                "max_single_asset_pct": self.limits.max_single_asset_pct,
                "min_order_size_usd": self.limits.min_order_size_usd,
                "max_order_size_pct": self.limits.max_order_size_pct,
            }
        }

    def validate_order_before_execution(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "LIMIT",
        account_balance: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Comprehensive pre-execution order validation.

        Combines all circuit breaker checks into a single validation.

        Returns:
            Tuple of (can_execute, reason)
        """
        # Check circuit breaker status first
        is_tripped, reason = self.check_circuit_breaker_status()
        if is_tripped:
            return False, f"Circuit breaker tripped: {reason}"

        # Update account balance if provided
        if account_balance is not None:
            self.account_balance = account_balance

        # Perform sanity checks
        valid, reason = self.perform_sanity_checks(symbol, side, quantity, price, order_type)
        if not valid:
            return False, f"Sanity check failed: {reason}"

        # Validate position size
        if order_type == "LIMIT":  # Only check for limit orders with known prices
            valid, reason = self.validate_position_size(symbol, quantity, price)
            if not valid:
                return False, f"Position size validation failed: {reason}"

        # Validate total exposure
        valid, reason = self.validate_total_exposure()
        if not valid:
            return False, f"Total exposure validation failed: {reason}"

        return True, "Order validation passed"


class EmergencyLiquidationTrigger:
    """
    Emergency liquidation trigger system.

    Monitors for emergency conditions and triggers position liquidation
    when safety thresholds are breached.
    """

    def __init__(self, circuit_breaker: PositionCircuitBreaker):
        """
        Initialize emergency liquidation trigger.

        Args:
            circuit_breaker: Position circuit breaker instance
        """
        self.circuit_breaker = circuit_breaker
        self.liquidation_active = False

    def check_emergency_conditions(self) -> List[str]:
        """
        Check for emergency liquidation conditions.

        Returns:
            List of emergency condition messages (empty if no emergencies)
        """
        emergencies = []

        # Check account balance integrity
        if self.circuit_breaker.account_balance <= 0:
            emergencies.append("Account balance is zero or negative")

        # Check for extreme position concentrations
        for symbol, position in self.circuit_breaker.current_positions.items():
            size = abs(float(position.get('size', 0)))
            entry_price = float(position.get('entryPrice', 0))
            position_value = size * entry_price

            # Emergency: Single position > 80% of account
            if (position_value / max(self.circuit_breaker.account_balance, 1)) > 0.8:
                emergencies.append(f"Position {symbol} exceeds 80% of account balance")

        # Check for extreme total exposure
        total_exposure = sum(
            abs(float(pos.get('size', 0))) * float(pos.get('entryPrice', 0))
            for pos in self.circuit_breaker.current_positions.values()
        )

        if self.circuit_breaker.account_balance > 0:
            exposure_ratio = total_exposure / self.circuit_breaker.account_balance

            # Emergency: Total exposure > 90% of account
            if exposure_ratio > 0.9:
                emergencies.append(f"Total exposure {exposure_ratio:.1%} exceeds 90% threshold")

        return emergencies

    def should_trigger_emergency_liquidation(self) -> Tuple[bool, str]:
        """
        Determine if emergency liquidation should be triggered.

        Returns:
            Tuple of (should_trigger, reason)
        """
        emergencies = self.check_emergency_conditions()

        if emergencies:
            reason = "; ".join(emergencies)
            return True, f"Emergency conditions detected: {reason}"

        return False, "No emergency conditions"


# Global circuit breaker instance
_circuit_breaker: Optional[PositionCircuitBreaker] = None


def get_position_circuit_breaker() -> Optional[PositionCircuitBreaker]:
    """Get global position circuit breaker instance."""
    return _circuit_breaker


def initialize_position_circuit_breaker(limits: Optional[CircuitBreakerLimits] = None) -> PositionCircuitBreaker:
    """Initialize global position circuit breaker."""
    global _circuit_breaker
    _circuit_breaker = PositionCircuitBreaker(limits)
    return _circuit_breaker
