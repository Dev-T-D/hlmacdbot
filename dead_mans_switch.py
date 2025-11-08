"""
Dead Man's Switch - Emergency Position Protection

Automatically closes all positions if the trading bot becomes unresponsive.
Critical safety mechanism to prevent unlimited losses during bot failures.

Features:
- Heartbeat monitoring with configurable timeout
- Automatic position closure on timeout
- Emergency liquidation with market orders
- Alert notifications and audit logging
- Graceful recovery mechanisms
"""

import asyncio
import threading
import time
import logging
from typing import Optional, Callable, Dict, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class DeadMansSwitch:
    """
    Dead Man's Switch for emergency position protection.

    Monitors bot heartbeat and automatically closes positions if the bot
    becomes unresponsive for longer than the configured timeout period.

    This prevents unlimited losses if the bot crashes, loses network connectivity,
    or encounters critical errors that prevent normal stop-loss execution.
    """

    def __init__(
        self,
        timeout_minutes: int = 5,
        check_interval_seconds: int = 30,
        emergency_callback: Optional[Callable] = None,
        alert_callback: Optional[Callable] = None,
        enabled: bool = True
    ):
        """
        Initialize dead man's switch.

        Args:
            timeout_minutes: Minutes without heartbeat before triggering emergency
            check_interval_seconds: How often to check heartbeat status
            emergency_callback: Function to call during emergency (closes positions)
            alert_callback: Function to call for alerts (email, SMS, etc.)
            enabled: Whether the dead man's switch is active
        """
        self.timeout_minutes = timeout_minutes
        self.check_interval_seconds = check_interval_seconds
        self.emergency_callback = emergency_callback
        self.alert_callback = alert_callback
        self.enabled = enabled

        # Heartbeat tracking
        self.last_heartbeat = time.time()
        self.heartbeat_count = 0

        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.running = False

        # Emergency state
        self.emergency_triggered = False
        self.emergency_time: Optional[float] = None

        # Lock for thread safety
        self.lock = threading.Lock()

        logger.info(
            f"Dead Man's Switch initialized: timeout={timeout_minutes}min, "
            f"check_interval={check_interval_seconds}s, enabled={enabled}"
        )

    def start(self) -> None:
        """Start the dead man's switch monitoring."""
        if not self.enabled:
            logger.info("Dead Man's Switch disabled - not starting monitoring")
            return

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Dead Man's Switch already running")
            return

        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_loop,
            name="DeadMansSwitch",
            daemon=True
        )
        self.monitoring_thread.start()

        logger.info("Dead Man's Switch monitoring started")

    def stop(self) -> None:
        """Stop the dead man's switch monitoring."""
        self.running = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
            if self.monitoring_thread.is_alive():
                logger.warning("Dead Man's Switch monitoring thread did not stop gracefully")

        logger.info("Dead Man's Switch monitoring stopped")

    def heartbeat(self) -> None:
        """Update the heartbeat timestamp."""
        with self.lock:
            self.last_heartbeat = time.time()
            self.heartbeat_count += 1

            # Reset emergency state if we recovered
            if self.emergency_triggered:
                logger.info("Dead Man's Switch: Bot recovered from emergency state")
                self.emergency_triggered = False
                self.emergency_time = None

    def is_healthy(self) -> bool:
        """Check if the bot is healthy (recent heartbeat)."""
        with self.lock:
            time_since_heartbeat = time.time() - self.last_heartbeat
            return time_since_heartbeat < (self.timeout_minutes * 60)

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the dead man's switch."""
        with self.lock:
            time_since_heartbeat = time.time() - self.last_heartbeat
            healthy = time_since_heartbeat < (self.timeout_minutes * 60)

            return {
                "enabled": self.enabled,
                "healthy": healthy,
                "last_heartbeat": datetime.fromtimestamp(self.last_heartbeat, timezone.utc).isoformat(),
                "time_since_heartbeat_seconds": time_since_heartbeat,
                "timeout_minutes": self.timeout_minutes,
                "emergency_triggered": self.emergency_triggered,
                "emergency_time": datetime.fromtimestamp(self.emergency_time, timezone.utc).isoformat() if self.emergency_time else None,
                "heartbeat_count": self.heartbeat_count,
                "running": self.running,
            }

    def _monitor_loop(self) -> None:
        """Main monitoring loop that runs in background thread."""
        logger.info("Dead Man's Switch monitoring loop started")

        while self.running:
            try:
                time.sleep(self.check_interval_seconds)

                if not self.enabled:
                    continue

                with self.lock:
                    time_since_heartbeat = time.time() - self.last_heartbeat

                    # Check if we've exceeded the timeout
                    if time_since_heartbeat >= (self.timeout_minutes * 60) and not self.emergency_triggered:
                        self._trigger_emergency(time_since_heartbeat)

                    # Send warning alerts before emergency
                    elif time_since_heartbeat >= (self.timeout_minutes * 60 * 0.8):  # 80% of timeout
                        self._send_warning_alert(time_since_heartbeat)

            except Exception as e:
                logger.error(f"Dead Man's Switch monitoring error: {e}")
                # Continue monitoring despite errors

        logger.info("Dead Man's Switch monitoring loop stopped")

    def _trigger_emergency(self, time_since_heartbeat: float) -> None:
        """Trigger emergency position closure."""
        self.emergency_triggered = True
        self.emergency_time = time.time()

        emergency_info = {
            "trigger_time": datetime.fromtimestamp(self.emergency_time, timezone.utc).isoformat(),
            "time_since_heartbeat": time_since_heartbeat,
            "timeout_minutes": self.timeout_minutes,
            "last_heartbeat": datetime.fromtimestamp(self.last_heartbeat, timezone.utc).isoformat(),
        }

        logger.critical("=" * 80)
        logger.critical("🚨 DEAD MAN'S SWITCH ACTIVATED 🚨")
        logger.critical("=" * 80)
        logger.critical(f"Bot unresponsive for {time_since_heartbeat:.1f} seconds")
        logger.critical(f"Timeout threshold: {self.timeout_minutes * 60} seconds")
        logger.critical("Initiating emergency position closure...")
        logger.critical("=" * 80)

        # Send alert notification
        if self.alert_callback:
            try:
                self.alert_callback("DEAD_MAN_SWITCH_TRIGGERED", emergency_info)
            except Exception as e:
                logger.error(f"Failed to send emergency alert: {e}")

        # Execute emergency position closure
        if self.emergency_callback:
            try:
                logger.critical("Executing emergency position closure...")
                success = self.emergency_callback()
                if success:
                    logger.critical("✅ Emergency position closure completed successfully")
                else:
                    logger.critical("❌ Emergency position closure failed")
            except Exception as e:
                logger.critical(f"❌ Emergency position closure error: {e}")
        else:
            logger.critical("❌ No emergency callback configured - manual intervention required!")

    def _send_warning_alert(self, time_since_heartbeat: float) -> None:
        """Send warning alert before emergency triggers."""
        warning_info = {
            "time_since_heartbeat": time_since_heartbeat,
            "timeout_minutes": self.timeout_minutes,
            "warning_threshold": self.timeout_minutes * 60 * 0.8,
            "last_heartbeat": datetime.fromtimestamp(self.last_heartbeat, timezone.utc).isoformat(),
        }

        logger.warning("=" * 60)
        logger.warning("⚠️  DEAD MAN'S SWITCH WARNING ⚠️")
        logger.warning("=" * 60)
        logger.warning(f"Bot unresponsive for {time_since_heartbeat:.1f} seconds")
        logger.warning(f"Emergency triggers in {(self.timeout_minutes * 60) - time_since_heartbeat:.1f} seconds")
        logger.warning("Ensure bot is functioning properly")
        logger.warning("=" * 60)

        # Send alert notification
        if self.alert_callback:
            try:
                self.alert_callback("DEAD_MAN_SWITCH_WARNING", warning_info)
            except Exception as e:
                logger.error(f"Failed to send warning alert: {e}")


class EmergencyLiquidationHandler:
    """
    Handles emergency position liquidation with safety checks.

    Ensures positions are closed safely during emergency situations,
    with validation and fallback mechanisms.
    """

    def __init__(self, client, audit_logger):
        """
        Initialize emergency liquidation handler.

        Args:
            client: Trading API client
            audit_logger: Audit logger instance
        """
        self.client = client
        self.audit_logger = audit_logger

    async def emergency_close_all_positions(self) -> bool:
        """
        Emergency close all open positions.

        Uses market orders for immediate execution, regardless of price.
        Includes comprehensive error handling and logging.

        Returns:
            True if all positions closed successfully, False otherwise
        """
        try:
            logger.critical("🔴 EMERGENCY LIQUIDATION: Starting position closure")

            # Get all positions
            account_info, positions, orders = await self.client.batch_get_account_data()

            if not positions:
                logger.critical("No open positions found - emergency liquidation complete")
                return True

            logger.critical(f"Found {len(positions)} open positions to close")

            success_count = 0
            failure_count = 0

            # Close each position
            for position in positions:
                try:
                    symbol = position.get("symbol", "").replace("USDT", "").replace("USDC", "")
                    side = "SHORT" if position.get("side") == "LONG" else "LONG"
                    size = abs(float(position.get("size", 0)))

                    if size <= 0:
                        continue

                    # Place market order to close position
                    order_result = await self.client.place_order(
                        symbol=symbol,
                        side=side,
                        quantity=size,
                        order_type="MARKET",
                        reduce_only=True
                    )

                    if order_result.get("status") == "ok":
                        logger.critical(f"✅ Emergency closed position: {symbol} {side} {size}")
                        success_count += 1

                        # Audit log the emergency closure
                        self.audit_logger.log_trade(
                            symbol=symbol,
                            side=side,
                            quantity=size,
                            price="MARKET",  # Emergency market order
                            order_type="MARKET",
                            strategy="EMERGENCY_LIQUIDATION",
                            pnl=0,  # Will be calculated by exchange
                            reason="Dead Man's Switch Emergency",
                        )
                    else:
                        logger.critical(f"❌ Failed to close position {symbol}: {order_result}")
                        failure_count += 1

                except Exception as e:
                    logger.critical(f"❌ Error closing position {position}: {e}")
                    failure_count += 1

            # Summary
            logger.critical("=" * 60)
            logger.critical("EMERGENCY LIQUIDATION SUMMARY")
            logger.critical("=" * 60)
            logger.critical(f"Positions found: {len(positions)}")
            logger.critical(f"Successfully closed: {success_count}")
            logger.critical(f"Failed to close: {failure_count}")
            logger.critical("=" * 60)

            return failure_count == 0

        except Exception as e:
            logger.critical(f"❌ Emergency liquidation failed with error: {e}")
            return False

    async def validate_emergency_closure(self) -> Dict[str, Any]:
        """
        Validate that emergency closure was successful.

        Checks that all positions are actually closed and logs results.

        Returns:
            Validation results dictionary
        """
        try:
            # Wait a moment for orders to execute
            await asyncio.sleep(2)

            # Check current positions
            account_info, positions, orders = await self.client.batch_get_account_data()

            validation = {
                "positions_remaining": len(positions),
                "open_orders": len(orders),
                "validation_time": datetime.now(timezone.utc).isoformat(),
                "success": len(positions) == 0,
            }

            if validation["success"]:
                logger.critical("✅ Emergency liquidation validation: SUCCESS - All positions closed")
            else:
                logger.critical(f"❌ Emergency liquidation validation: FAILED - {len(positions)} positions still open")
                for pos in positions:
                    logger.critical(f"  Remaining position: {pos}")

            return validation

        except Exception as e:
            logger.error(f"Emergency liquidation validation failed: {e}")
            return {
                "error": str(e),
                "validation_time": datetime.now(timezone.utc).isoformat(),
                "success": False,
            }


# Global dead man's switch instance
_dead_man_switch: Optional[DeadMansSwitch] = None


def get_dead_man_switch() -> Optional[DeadMansSwitch]:
    """Get global dead man's switch instance."""
    return _dead_man_switch


def initialize_dead_man_switch(
    timeout_minutes: int = 5,
    emergency_callback: Optional[Callable] = None,
    alert_callback: Optional[Callable] = None,
    enabled: bool = True
) -> DeadMansSwitch:
    """Initialize global dead man's switch."""
    global _dead_man_switch
    _dead_man_switch = DeadMansSwitch(
        timeout_minutes=timeout_minutes,
        emergency_callback=emergency_callback,
        alert_callback=alert_callback,
        enabled=enabled
    )
    return _dead_man_switch
