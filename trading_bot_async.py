"""
Async High-Performance Trading Bot for Hyperliquid

Optimized for concurrent operations, caching, and real-time trading.
Uses async/await patterns for maximum performance.

"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from audit_logger import AuditLogger
from cache_manager import CacheManager, initialize_cache_manager, get_cache_manager
from constants import (
    CACHE_MAX_AGE_SECONDS,
    DEFAULT_ACTIVATION_PERCENT,
    DEFAULT_BALANCE_DRY_RUN,
    DEFAULT_CANDLES_TO_FETCH,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DEFAULT_STATUS_CHECK_INTERVAL,
    DEFAULT_TRAIL_PERCENT,
    DEFAULT_UPDATE_THRESHOLD_PERCENT,
    HIGHER_TF_CACHE_MAX_AGE_SECONDS,
    HIGHER_TF_CANDLES_NEEDED,
    MAX_INCREMENTAL_CANDLES,
    MIN_CANDLES_FOR_MACD,
    MIN_ENTRY_PRICE_DIFF,
    MIN_INCREMENTAL_CANDLES,
    MIN_QUANTITY,
    POSITION_QTY_DIFF_THRESHOLD_PCT,
    QUANTITY_PRECISION,
)
from exceptions import (
    ConfigurationError,
    DailyLimitError,
    ExchangeAPIError,
    ExchangeError,
    ExchangeNetworkError,
    ExchangeTimeoutError,
    IndicatorCalculationError,
    EntrySignalError,
    MarketDataError,
    MarketDataUnavailableError,
    MarketDataValidationError,
    OrderError,
    OrderExecutionError,
    OrderValidationError,
    PositionError,
    PositionSizeError,
    RiskManagementError,
    StrategyError,
    TradingBotError,
)
from hyperliquid_client_async import AsyncHyperliquidClient
from hyperliquid_websocket import HyperliquidWebSocketClient, WEBSOCKETS_AVAILABLE
from input_sanitizer import InputSanitizer
from macd_strategy import MACDStrategy
from order_tracker import OrderStatus, OrderTracker
from risk_manager import RiskManager, TrailingStopLoss

logger = logging.getLogger(__name__)


class AsyncTradingBot:
    """High-Performance Async Trading Bot for Hyperliquid"""

    def __init__(self, config_path: str = "config/config.json"):
        """Initialize async trading bot with configuration."""
        self.config = self._load_config(config_path)

        # Initialize credential manager
        from credential_manager import CredentialManager
        self.credential_manager = CredentialManager(use_keyring=True)
        self._load_credentials()

        # Initialize audit logger
        audit_log_path = self.config.get("audit_log_path", "logs/audit.log")
        self.audit_logger = AuditLogger(log_file=audit_log_path)

        # Verify audit log integrity
        is_valid, errors = self.audit_logger.verify_log_integrity()
        if not is_valid:
            logger.warning(f"⚠️  Audit log integrity check failed: {errors}")

        # Validate configuration
        self._validate_config()

        # Initialize async Hyperliquid client
        self.client = AsyncHyperliquidClient(
            private_key=self.config["private_key"],
            wallet_address=self.config["wallet_address"],
            testnet=self.config.get("testnet", True),
            max_concurrent_requests=10,
            connection_pool_size=20
        )

        # WebSocket client for real-time data
        self.websocket_enabled = InputSanitizer.sanitize_boolean(
            self.config.get("websocket", {}).get("enabled", False), "websocket.enabled"
        )
        self.ws_client: Optional[HyperliquidWebSocketClient] = None
        self.current_price: Optional[float] = None

        if self.websocket_enabled:
            if not WEBSOCKETS_AVAILABLE:
                logger.warning("WebSocket enabled but 'websockets' library not installed. Falling back to REST polling.")
                self.websocket_enabled = False
            else:
                websocket_config = self.config.get("websocket", {})
                reconnect_interval = websocket_config.get("reconnect_interval", 5.0)
                max_reconnect_attempts = websocket_config.get("max_reconnect_attempts", 10)

                self.ws_client = HyperliquidWebSocketClient(
                    testnet=self.config.get("testnet", True),
                    reconnect_interval=reconnect_interval,
                    max_reconnect_attempts=max_reconnect_attempts
                )
                logger.info("WebSocket client initialized")

        # Trading parameters
        self.symbol = InputSanitizer.sanitize_symbol(self.config["trading"]["symbol"])
        self.timeframe = InputSanitizer.sanitize_timeframe(self.config["trading"]["timeframe"])
        self.check_interval = InputSanitizer.sanitize_check_interval(
            self.config["trading"]["check_interval"]
        )
        self.dry_run = InputSanitizer.sanitize_boolean(
            self.config["trading"].get("dry_run", True), "dry_run"
        )

        # Strategy and risk management
        self.strategy = self._initialize_strategy()
        self.risk_manager = self._initialize_risk_manager()
        self.trailing_stop_enabled = self.config["risk"]["trailing_stop"]["enabled"]
        self.trailing_stop: Optional[TrailingStopLoss] = None

        # Order tracking
        order_tracking_config = self.config.get("order_tracking", {})
        self.order_tracker = OrderTracker(
            max_retries=order_tracking_config.get("max_retries", DEFAULT_MAX_RETRIES),
            retry_delay=order_tracking_config.get("retry_delay", DEFAULT_RETRY_DELAY),
            status_check_interval=order_tracking_config.get(
                "status_check_interval", DEFAULT_STATUS_CHECK_INTERVAL
            ),
        )

        # Multi-timeframe analysis
        multi_timeframe_config = self.config.get("multi_timeframe", {})
        self.multi_timeframe_enabled = InputSanitizer.sanitize_boolean(
            multi_timeframe_config.get("enabled", False), "multi_timeframe.enabled"
        )
        self.higher_timeframe = multi_timeframe_config.get("higher_timeframe", "4h")

        # State management
        self.current_position: Optional[Dict] = None
        self.last_daily_reset = datetime.now(timezone.utc).date()

        # Cache manager
        self.cache_manager = CacheManager(
            redis_url=self.config.get("cache", {}).get("redis_url"),
            max_memory_cache_size=self.config.get("cache", {}).get("max_memory_size", 1000),
            default_ttl=self.config.get("cache", {}).get("default_ttl", 300)
        )

        # Async control
        self._running = False
        self._shutdown_event = asyncio.Event()

        logger.info("Async trading bot initialized")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in config file: {e}")

    def _load_credentials(self):
        """Load credentials securely."""
        creds = self.credential_manager.get_hyperliquid_credentials(self.config)
        if not creds['private_key'] or not creds['wallet_address']:
            raise ConfigurationError("Missing credentials. Set via environment variables or keyring.")

        self.config["private_key"] = creds['private_key']
        self.config["wallet_address"] = creds['wallet_address']

    def _validate_config(self):
        """Validate configuration structure."""
        required_fields = ["private_key", "wallet_address", "testnet", "trading", "strategy", "risk"]
        for field in required_fields:
            if field not in self.config:
                raise ConfigurationError(f"Missing required field: {field}")

        trading = self.config["trading"]
        required_trading = ["symbol", "timeframe", "check_interval"]
        for field in required_trading:
            if field not in trading:
                raise ConfigurationError(f"Missing trading field: {field}")

    def _initialize_strategy(self) -> MACDStrategy:
        """Initialize MACD strategy."""
        strategy_config = self.config["strategy"]
        return MACDStrategy(
            fast_length=strategy_config["fast_length"],
            slow_length=strategy_config["slow_length"],
            signal_length=strategy_config["signal_length"],
            risk_reward_ratio=strategy_config["risk_reward_ratio"],
            rsi_period=strategy_config.get("rsi_period", 14),
            rsi_oversold=strategy_config.get("rsi_oversold", 30.0),
            rsi_overbought=strategy_config.get("rsi_overbought", 65.0),
            min_histogram_strength=strategy_config.get("min_histogram_strength", 0.0),
            require_volume_confirmation=strategy_config.get("require_volume_confirmation", False),
            volume_period=strategy_config.get("volume_period", 20),
            min_trend_strength=strategy_config.get("min_trend_strength", 0.0),
            strict_long_conditions=strategy_config.get("strict_long_conditions", True),
            disable_long_trades=strategy_config.get("disable_long_trades", False),
        )

    def _initialize_risk_manager(self) -> RiskManager:
        """Initialize risk manager."""
        risk_config = self.config["risk"]
        return RiskManager(
            max_daily_loss_pct=risk_config["max_daily_loss_pct"],
            max_position_size_pct=risk_config["max_position_size_pct"],
            max_trades_per_day=risk_config["max_trades_per_day"],
            leverage=risk_config["leverage"],
        )

    async def setup_leverage(self):
        """Setup leverage for trading."""
        try:
            leverage = self.risk_manager.leverage
            if not self.dry_run:
                await self.client.set_leverage(self.symbol, leverage)
                logger.info(f"✅ Leverage set to {leverage}x")
            else:
                logger.info(f"🔶 DRY RUN - Would set leverage to {leverage}x")
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")

    async def get_market_data(self) -> pd.DataFrame:
        """Get market data with high-performance caching."""
        cache_key = CacheManager.make_market_data_key(self.symbol, self.timeframe, DEFAULT_CANDLES_TO_FETCH)

        # Check cache first
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Using cached market data for {cache_key}")
            # Convert back to DataFrame
            return pd.DataFrame(cached_data)

        # Fetch fresh data
        try:
            klines = await self.client.get_klines(
                self.symbol, self.timeframe, limit=DEFAULT_CANDLES_TO_FETCH
            )

            if not klines:
                raise MarketDataUnavailableError("No kline data received")

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df.astype({
                'timestamp': 'int64',
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64'
            })

            # Validate DataFrame
            if df.empty or len(df) < MIN_CANDLES_FOR_MACD:
                raise MarketDataValidationError(
                    f"Insufficient data: got {len(df)} candles, need at least {MIN_CANDLES_FOR_MACD}"
                )

            # Cache the result (convert to dict for JSON serialization)
            await self.cache_manager.set(cache_key, df.to_dict('records'), CACHE_MAX_AGE_SECONDS)

            logger.debug(f"Fetched {len(df)} candles for {self.symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            raise MarketDataError(f"Market data fetch failed: {e}") from e

    async def get_account_balance(self) -> float:
        """Get account balance with high-performance caching."""
        cache_key = CacheManager.make_account_data_key(self.client.wallet_address)

        # Check cache first
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Using cached account balance for {self.client.wallet_address}")
            return float(cached_data.get("balance", "0"))

        # Fetch fresh data
        try:
            account_info = await self.client.get_account_info()
            balance = float(account_info.get("balance", "0"))

            # Cache the result (shorter TTL for account data)
            await self.cache_manager.set(cache_key, account_info, 30)  # 30 seconds

            return balance

        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            raise ValueError(f"Account balance unavailable: {e}") from e

    async def get_position(self) -> Optional[Dict]:
        """Get current position with high-performance caching."""
        cache_key = CacheManager.make_position_key(self.client.wallet_address, self.symbol)

        # Check cache first
        cached_position = await self.cache_manager.get(cache_key)
        if cached_position is not None:
            logger.debug(f"Using cached position for {self.symbol}")
            return cached_position

        # Fetch fresh data using batch API
        try:
            account_info, positions, orders = await self.client.batch_get_account_data()

            # Cache individual positions
            for pos in positions:
                pos_cache_key = CacheManager.make_position_key(self.client.wallet_address, pos.get("symbol", ""))
                await self.cache_manager.set(pos_cache_key, pos, 10)  # 10 seconds

            # Find position for our symbol
            for pos in positions:
                if pos.get("symbol") == self.symbol:
                    return pos

            return None

        except Exception as e:
            logger.error(f"Failed to get position: {e}")
            return None

    async def get_higher_timeframe_data(self) -> Optional[pd.DataFrame]:
        """Get higher timeframe data for multi-timeframe analysis."""
        try:
            klines = await self.client.get_klines(
                self.symbol, self.higher_timeframe, limit=HIGHER_TF_CANDLES_NEEDED
            )

            if not klines or len(klines) < 50:  # Need sufficient higher TF data
                return None

            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df.astype({
                'timestamp': 'int64',
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64'
            })

            return df

        except Exception as e:
            logger.warning(f"Failed to get higher timeframe data: {e}")
            return None

    async def _sync_position_with_exchange(self) -> None:
        """Sync position with exchange."""
        try:
            exchange_position = await self.get_position()

            if exchange_position:
                # Position exists on exchange
                if not self.current_position:
                    # We don't have it locally, load it
                    self.current_position = exchange_position
                    self._load_position_state()
                    logger.info(f"📥 Synced position from exchange: {exchange_position}")
                else:
                    # Compare with local position
                    local_size = self.current_position.get("size", 0)
                    exchange_size = exchange_position.get("size", 0)

                    if abs(local_size - exchange_size) > POSITION_QTY_DIFF_THRESHOLD_PCT * local_size:
                        logger.warning(
                            f"Position size mismatch: local={local_size}, exchange={exchange_size}. "
                            "Updating to exchange position."
                        )
                        self.current_position = exchange_position
                        self._load_position_state()
            else:
                # No position on exchange
                if self.current_position:
                    logger.warning("Exchange shows no position but we have local position. Clearing local position.")
                    self.current_position = None
                    self.trailing_stop = None

        except Exception as e:
            logger.error(f"Failed to sync position: {e}")

    async def run_trading_cycle(self) -> None:
        """Execute one async trading cycle."""
        try:
            # Get market data
            df = await self.get_market_data()
            current_price = df.iloc[-1]["close"]

            # Update current price from WebSocket if available
            if self.current_price:
                current_price = self.current_price

            # Check order statuses and sync position concurrently
            await asyncio.gather(
                self._check_order_statuses(),
                self._sync_position_with_exchange()
            )

            # Calculate indicators only when needed
            indicators_calculated = False

            if self.current_position:
                # Calculate indicators for exit signals
                if not indicators_calculated:
                    df = self.strategy.calculate_indicators(df)
                    indicators_calculated = True

                indicators = self.strategy.get_indicator_values(df)
                logger.info(
                    f"💹 {self.symbol}: ${current_price:,.2f} | "
                    f"MACD: {indicators['macd']:.4f} | "
                    f"Signal: {indicators['signal']:.4f} | "
                    f"Hist: {indicators['histogram']:.4f}"
                )

                # Check exit conditions
                should_exit, exit_reason = self.check_exit_conditions(df)
                if should_exit:
                    await self.close_position(exit_reason, current_price)

            else:
                # Calculate indicators for entry signals
                if not indicators_calculated:
                    df = self.strategy.calculate_indicators(df)
                    indicators_calculated = True

                indicators = self.strategy.get_indicator_values(df)
                logger.info(
                    f"💹 {self.symbol}: ${current_price:,.2f} | "
                    f"MACD: {indicators['macd']:.4f} | "
                    f"Signal: {indicators['signal']:.4f} | "
                    f"Hist: {indicators['histogram']:.4f}"
                )

                # Look for entry signals
                signal = self.strategy.check_entry_signal(df)
                if signal:
                    # Multi-timeframe analysis
                    if self.multi_timeframe_enabled:
                        df_higher_tf = await self.get_higher_timeframe_data()
                        if df_higher_tf is not None:
                            is_aligned, reason = self.strategy.check_higher_timeframe_trend(
                                df_higher_tf, signal["type"]
                            )
                            if not is_aligned:
                                logger.info(f"⏸️ Entry signal rejected: {reason}")
                                signal = None
                            else:
                                logger.info(f"✅ Multi-timeframe confirmed: {reason}")

                    if signal:
                        await self.enter_position(signal, current_price)

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)

    async def enter_position(self, signal: Dict, current_price: float) -> None:
        """Enter position based on signal."""
        try:
            # Calculate position size
            account_balance = await self.get_account_balance()
            position_size = self.risk_manager.calculate_position_size(
                account_balance, current_price, signal["stop_loss"]
            )

            if position_size < MIN_QUANTITY:
                logger.warning(f"Position size too small: {position_size} < {MIN_QUANTITY}")
                return

            # Place order
            side = "BUY" if signal["type"] == "LONG" else "SELL"
            order_result = await self.client.place_order(
                symbol=self.symbol,
                side=side,
                quantity=position_size,
                price=current_price,
                order_type="LIMIT"
            )

            if order_result.get("status") == "ok":
                # Create position record
                self.current_position = {
                    "symbol": self.symbol,
                    "type": signal["type"],
                    "entry_price": current_price,
                    "stop_loss": signal["stop_loss"],
                    "take_profit": signal["take_profit"],
                    "size": position_size,
                    "order_id": order_result.get("orderId"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                # Initialize trailing stop if enabled
                if self.trailing_stop_enabled:
                    self.trailing_stop = TrailingStopLoss(
                        initial_stop_loss=signal["stop_loss"],
                        trail_percent=self.config["risk"]["trailing_stop"]["trail_percent"],
                        activation_percent=self.config["risk"]["trailing_stop"]["activation_percent"],
                        update_threshold_percent=self.config["risk"]["trailing_stop"]["update_threshold_percent"],
                    )

                # Save position state
                self._save_position_state()

                # Log trade
                self.audit_logger.log_trade(
                    symbol=self.symbol,
                    side=side,
                    quantity=position_size,
                    price=current_price,
                    order_type="LIMIT",
                    strategy="MACD",
                    indicators=signal.get("indicators", {}),
                )

                logger.info(f"🎯 Entered {signal['type']} position: {position_size} @ ${current_price:,.2f}")

        except Exception as e:
            logger.error(f"Failed to enter position: {e}")

    async def close_position(self, reason: str, current_price: float) -> None:
        """Close current position."""
        if not self.current_position:
            return

        try:
            position_size = self.current_position["size"]
            side = "SELL" if self.current_position["type"] == "LONG" else "BUY"

            # Place closing order
            order_result = await self.client.place_order(
                symbol=self.symbol,
                side=side,
                quantity=position_size,
                price=current_price,
                order_type="MARKET",
                reduce_only=True
            )

            if order_result.get("status") == "ok":
                # Calculate P&L
                entry_price = self.current_position["entry_price"]
                pnl = (current_price - entry_price) * position_size if self.current_position["type"] == "LONG" else (entry_price - current_price) * position_size

                # Log trade
                self.audit_logger.log_trade(
                    symbol=self.symbol,
                    side=side,
                    quantity=position_size,
                    price=current_price,
                    order_type="MARKET",
                    strategy="MACD",
                    pnl=pnl,
                    reason=reason,
                )

                logger.info(f"💰 Closed {self.current_position['type']} position: {reason} | P&L: ${pnl:,.2f}")

                # Clear position
                self.current_position = None
                self.trailing_stop = None
                self._save_position_state()

        except Exception as e:
            logger.error(f"Failed to close position: {e}")

    def check_exit_conditions(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check exit conditions."""
        if not self.current_position:
            return False, ""

        # Get latest indicator values
        indicators = self.strategy.get_indicator_values(df)
        current_price = df.iloc[-1]["close"]

        # Check stop loss and take profit
        stop_loss = self.current_position["stop_loss"]
        take_profit = self.current_position["take_profit"]

        if self.current_position["type"] == "LONG":
            if current_price <= stop_loss:
                return True, f"Stop Loss hit (${current_price:,.2f} <= ${stop_loss:,.2f})"
            elif current_price >= take_profit:
                return True, f"Take Profit hit (${current_price:,.2f} >= ${take_profit:,.2f})"
        else:  # SHORT
            if current_price >= stop_loss:
                return True, f"Stop Loss hit (${current_price:,.2f} >= ${stop_loss:,.2f})"
            elif current_price <= take_profit:
                return True, f"Take Profit hit (${current_price:,.2f} <= ${take_profit:,.2f})"

        # Check MACD exit signals
        histogram = indicators["histogram"]
        prev_histogram = df.iloc[-2]["histogram"] if len(df) > 1 else 0

        # Exit on MACD histogram reversal (bearish divergence for long, bullish for short)
        if self.current_position["type"] == "LONG" and histogram < 0 and prev_histogram > 0:
            return True, "MACD bearish divergence"
        elif self.current_position["type"] == "SHORT" and histogram > 0 and prev_histogram < 0:
            return True, "MACD bullish divergence"

        return False, ""

    async def _check_order_statuses(self) -> None:
        """Check status of pending orders."""
        # Implementation for order status checking
        pass

    def _load_position_state(self) -> None:
        """Load position state from file."""
        try:
            if os.path.exists("state/position.json"):
                with open("state/position.json", "r") as f:
                    self.current_position = json.load(f)
                logger.info("Position state loaded from file")
        except Exception as e:
            logger.error(f"Failed to load position state: {e}")

    async def _warm_cache(self) -> None:
        """Warm cache with frequently accessed data."""
        logger.info("Warming cache with initial data...")

        warmup_data = {}

        try:
            # Warm asset metadata (long TTL)
            metadata_key = CacheManager.make_asset_metadata_key()
            metadata = await self.client.get_asset_metadata()
            warmup_data[metadata_key] = (metadata, 3600)  # 1 hour

            # Warm clearinghouse state (medium TTL)
            clearinghouse_key = CacheManager.make_clearinghouse_key(self.client.wallet_address)
            clearinghouse_state = await self.client.get_clearinghouse_state()
            warmup_data[clearinghouse_key] = (clearinghouse_state, 300)  # 5 minutes

            # Note: Market data will be cached on first access
            # Account data will be cached on first access

            await self.cache_manager.warm_cache(warmup_data)
            logger.info(f"Cache warmed with {len(warmup_data)} entries")

        except Exception as e:
            logger.warning(f"Cache warming failed: {e}")

    def _save_position_state(self) -> None:
        """Save position state to file."""
        try:
            os.makedirs("state", exist_ok=True)
            with open("state/position.json", "w") as f:
                json.dump(self.current_position, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save position state: {e}")

    async def run(self) -> None:
        """Main async bot loop."""
        logger.info("🚀 Starting async trading bot...")

        # Initialize cache manager
        await self.cache_manager.connect()

        # Cache warming - preload frequently accessed data
        await self._warm_cache()

        # Setup leverage
        await self.setup_leverage()

        # Setup WebSocket if enabled
        if self.websocket_enabled and self.ws_client:
            self.ws_client.start()

        # Initialize daily stats
        try:
            initial_balance = await self.get_account_balance()
            self.risk_manager.reset_daily_stats(initial_balance)
        except ValueError as e:
            logger.error(f"❌ Cannot start bot without account balance: {e}")
            return

        self._running = True

        try:
            while self._running and not self._shutdown_event.is_set():
                # Check daily reset
                current_date = datetime.now(timezone.utc).date()
                if current_date > self.last_daily_reset:
                    logger.info("\n" + "=" * 60)
                    logger.info("🌅 NEW TRADING DAY (UTC)")
                    logger.info("=" * 60)
                    try:
                        daily_balance = await self.get_account_balance()
                        self.risk_manager.reset_daily_stats(daily_balance)
                    except ValueError as e:
                        logger.error(f"Cannot reset daily stats: {e}")
                    self.last_daily_reset = current_date

                # Process WebSocket messages
                if self.websocket_enabled and self.ws_client:
                    self.ws_client.process_messages()

                # Run trading cycle
                utc_now = datetime.now(timezone.utc)
                logger.info("\n" + "-" * 60)
                logger.info(f"⏰ UTC: {utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                logger.info("-" * 60)

                await self.run_trading_cycle()

                # Wait before next cycle
                logger.info(f"⏳ Sleeping for {self.check_interval} seconds...\n")
                await asyncio.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("\n🛑 Bot stopped by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            # Cleanup
            if self.websocket_enabled and self.ws_client:
                logger.info("Closing WebSocket connection...")
                self.ws_client.stop()

            # Close cache manager
            await self.cache_manager.disconnect()

            # Close client
            await self.client.close()

            logger.info("👋 Async bot shutdown complete")

    async def shutdown(self) -> None:
        """Shutdown the bot gracefully."""
        logger.info("Initiating graceful shutdown...")
        self._running = False
        self._shutdown_event.set()

        # Close current position if any
        if self.current_position:
            try:
                df = await self.get_market_data()
                current_price = df.iloc[-1]["close"]
                await self.close_position("Bot Shutdown", current_price)
            except Exception as e:
                logger.error(f"Error closing position during shutdown: {e}")


async def main():
    """Main async entry point."""
    # Setup logging
    from trading_bot import setup_logging
    setup_logging()

    # Create and run bot
    bot = AsyncTradingBot("config/config.json")

    try:
        await bot.run()
    except KeyboardInterrupt:
        await bot.shutdown()
    finally:
        await bot.client.close()


if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Run async bot
    asyncio.run(main())
