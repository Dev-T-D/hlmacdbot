"""
MACD Futures Trading Bot.

Main bot script with complete trading logic.
Supports Hyperliquid exchange.

"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from audit_logger import AuditLogger
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
from credential_manager import CredentialManager
from exceptions import (
    ConfigurationError,
    DailyLimitError,
    ExchangeAPIError,
    ExchangeError,
    ExchangeNetworkError,
    ExchangeTimeoutError,
    IndicatorCalculationError,
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
from hyperliquid_client import HyperliquidClient
from hyperliquid_websocket import HyperliquidWebSocketClient, WEBSOCKETS_AVAILABLE
from input_sanitizer import InputSanitizer
from macd_strategy import MACDStrategy
from order_tracker import OrderStatus, OrderTracker
from risk_manager import RiskManager, TrailingStopLoss

# Setup logging with rotation and different levels per handler
def setup_logging():
    """
    Configure logging with rotation and different levels per handler.

    - File handler: DEBUG level with rotation (10MB max, 5 backups)
    - Console handler: INFO level (less verbose for terminal)
    - Error file handler: ERROR level only (separate error log)
    """
    # Ensure logs directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers filter

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Detailed format for file logs
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Simpler format for console logs
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # File handler: DEBUG level with rotation (10MB per file, keep 5 backups)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "bot.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Error file handler: ERROR level only (separate error log)
    error_handler = RotatingFileHandler(
        os.path.join(log_dir, "bot_errors.log"),
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)

    # Console handler: INFO level (less verbose)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Log configuration
    root_logger.info("Logging configured:")
    root_logger.info(f"  - File log: {os.path.join(log_dir, 'bot.log')} (DEBUG)")
    root_logger.info(f"  - Error log: {os.path.join(log_dir, 'bot_errors.log')} (ERROR)")
    root_logger.info("  - Console: INFO level")
    root_logger.info("  - Rotation: 10MB per file, 5 backups")


# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


class TradingBot:
    """
    MACD Futures Trading Bot for Hyperliquid Exchange.

    This class implements a complete automated trading system featuring:
    - Advanced MACD-based trend following strategy
    - Real-time WebSocket data streaming with REST API fallback
    - Comprehensive risk management with trailing stops
    - Enterprise-grade reliability with circuit breakers and state recovery
    - Detailed audit logging and tamper detection
    - Multi-channel alerting and monitoring

    The bot operates continuously, analyzing market conditions and executing trades
    based on configurable strategy parameters and risk limits.

    Attributes:
        config (dict): Complete configuration dictionary with all bot settings
        client (HyperliquidClient): Exchange API client for order execution
        strategy (MACDStrategy): Trading strategy implementation
        risk_manager (RiskManager): Position sizing and risk control
        audit_logger (AuditLogger): Tamper-protected audit logging
        websocket_enabled (bool): Whether WebSocket streaming is active
        ws_client (Optional[HyperliquidWebSocketClient]): WebSocket client instance
        current_price (Optional[float]): Latest price from WebSocket stream

    Key Features:
        - **Strategy**: MACD crossover signals with trend confirmation
        - **Risk Management**: Configurable position sizing, stop-loss, take-profit
        - **Data Sources**: WebSocket real-time + REST API historical
        - **Reliability**: Automatic recovery, state persistence, error handling
        - **Monitoring**: Comprehensive logging, metrics, and alerting
        - **Security**: Encrypted credentials, input validation, audit trails

    Example:
        >>> bot = TradingBot('config/config.json')
        >>> bot.run()
        Starting MACD Trading Bot v2.0 (Hyperliquid)
        Testnet: True, Dry Run: False
        Connected to Hyperliquid exchange
        WebSocket enabled - real-time data streaming active
        Trading loop started - monitoring BTCUSDT
    """

    def __init__(self, config_path: str = "config/config.json"):
        """
        Initialize the trading bot with comprehensive setup and validation.

        This method performs complete bot initialization including:
        - Configuration loading and validation
        - Secure credential management
        - Exchange connectivity testing
        - WebSocket client setup (if enabled)
        - Strategy and risk manager initialization
        - Audit logging verification
        - Health checks and startup validation

        Args:
            config_path (str): Path to JSON configuration file. Defaults to 'config/config.json'.
                The configuration file should contain all required sections:
                - exchange: Hyperliquid connection settings
                - trading: Symbol, timeframe, intervals
                - strategy: MACD parameters and filters
                - risk: Position sizing and loss limits
                - websocket: Real-time data settings (optional)

        Raises:
            ConfigurationError: If config file is missing, invalid, or incomplete
            CredentialError: If required credentials cannot be loaded
            ExchangeError: If exchange connectivity cannot be established
            InitializationError: If any component fails to initialize properly

        Example:
            >>> # Initialize with default config
            >>> bot = TradingBot()

            >>> # Initialize with custom config
            >>> bot = TradingBot('config/production.json')
        """
        # Load and validate configuration
        self.config = self._load_config(config_path)

        # Initialize credential manager for secure credential storage
        self.credential_manager = CredentialManager(use_keyring=True)

        # Initialize audit logger for tamper-protected trade logging
        audit_log_path = self.config.get("audit_log_path", "logs/audit.log")
        self.audit_logger = AuditLogger(log_file=audit_log_path)

        # Verify audit log integrity on startup
        is_valid, errors = self.audit_logger.verify_log_integrity()
        if not is_valid:
            logger.warning(f"⚠️  Audit log integrity check failed: {errors}")
            logger.warning("Some log entries may have been tampered with!")

        # Load credentials from secure storage (keyring > env vars > config file)
        self._load_credentials()

        # Validate configuration structure
        self._validate_config()

        # Determine exchange type (sanitized) - Hyperliquid only
        exchange = self.config.get("exchange", "hyperliquid").lower().strip()
        if exchange != "hyperliquid":
            raise ConfigurationError(
                f"Invalid exchange: '{exchange}'. Only 'hyperliquid' is supported. "
                f"Bitunix support has been removed."
            )

        # Initialize Hyperliquid client (with sanitized inputs)
        private_key = InputSanitizer.sanitize_private_key(self.config["private_key"])
        wallet_address = InputSanitizer.sanitize_wallet_address(self.config["wallet_address"])
        testnet = InputSanitizer.sanitize_boolean(self.config.get("testnet", True), "testnet")

        self.client = HyperliquidClient(
            private_key=private_key, wallet_address=wallet_address, testnet=testnet
        )
        self.exchange_name = "Hyperliquid"
        self.testnet = testnet

        # WebSocket client for real-time data (optional)
        websocket_config = self.config.get("websocket", {})
        self.websocket_enabled = InputSanitizer.sanitize_boolean(
            websocket_config.get("enabled", False), "websocket.enabled"
        )
        self.ws_client: Optional[HyperliquidWebSocketClient] = None
        self.current_price: Optional[float] = None  # Real-time price from WebSocket

        if self.websocket_enabled:
            if not WEBSOCKETS_AVAILABLE:
                logger.warning(
                    "WebSocket enabled but 'websockets' library not installed. "
                    "Install with: pip install websockets. Falling back to REST polling."
                )
                self.websocket_enabled = False
            else:
                reconnect_interval = websocket_config.get("reconnect_interval", 5.0)
                max_reconnect_attempts = websocket_config.get("max_reconnect_attempts", 10)
                try:
                    self.ws_client = HyperliquidWebSocketClient(
                        testnet=testnet,
                        reconnect_interval=float(reconnect_interval),
                        max_reconnect_attempts=int(max_reconnect_attempts),
                    )
                    logger.info("WebSocket client initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize WebSocket client: {e}")
                    logger.warning("Falling back to REST API polling")
                    self.websocket_enabled = False
                    self.ws_client = None

        # Sanitize strategy parameters
        fast_length = InputSanitizer.sanitize_macd_length(
            self.config["strategy"]["fast_length"], "fast_length"
        )
        slow_length = InputSanitizer.sanitize_macd_length(
            self.config["strategy"]["slow_length"], "slow_length"
        )
        signal_length = InputSanitizer.sanitize_macd_length(
            self.config["strategy"]["signal_length"], "signal_length"
        )
        risk_reward_ratio = InputSanitizer.sanitize_risk_reward_ratio(
            self.config["strategy"]["risk_reward_ratio"]
        )

        # Get improved strategy filters (with defaults)
        strategy_config = self.config.get("strategy", {})
        rsi_period = strategy_config.get("rsi_period", 14)
        rsi_oversold = strategy_config.get("rsi_oversold", 30.0)
        rsi_overbought = strategy_config.get("rsi_overbought", 70.0)
        min_histogram_strength = strategy_config.get("min_histogram_strength", 0.0)
        require_volume_confirmation = strategy_config.get("require_volume_confirmation", True)
        volume_period = strategy_config.get("volume_period", 20)
        min_trend_strength = strategy_config.get("min_trend_strength", 0.0)
        strict_long_conditions = strategy_config.get("strict_long_conditions", True)
        disable_long_trades = strategy_config.get("disable_long_trades", False)

        self.strategy = MACDStrategy(
            fast_length=fast_length,
            slow_length=slow_length,
            signal_length=signal_length,
            risk_reward_ratio=risk_reward_ratio,
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            min_histogram_strength=min_histogram_strength,
            require_volume_confirmation=require_volume_confirmation,
            volume_period=volume_period,
            min_trend_strength=min_trend_strength,
            strict_long_conditions=strict_long_conditions,
            disable_long_trades=disable_long_trades,
        )

        # Sanitize risk parameters
        leverage = InputSanitizer.sanitize_leverage(self.config["risk"]["leverage"], exchange)
        max_position_size_pct = InputSanitizer.sanitize_percentage(
            self.config["risk"]["max_position_size_pct"], "max_position_size_pct"
        )
        max_daily_loss_pct = InputSanitizer.sanitize_percentage(
            self.config["risk"]["max_daily_loss_pct"], "max_daily_loss_pct"
        )
        max_trades_per_day = InputSanitizer.sanitize_max_trades_per_day(
            self.config["risk"]["max_trades_per_day"]
        )

        self.risk_manager = RiskManager(
            max_position_size_pct=max_position_size_pct,
            max_daily_loss_pct=max_daily_loss_pct,
            max_trades_per_day=max_trades_per_day,
            leverage=leverage,
            exchange=exchange,  # Pass exchange name for limit validation
        )

        # Initialize trailing stop-loss (if enabled) with sanitized inputs
        trailing_config = self.config["risk"].get("trailing_stop", {})
        self.trailing_stop_enabled = InputSanitizer.sanitize_boolean(
            trailing_config.get("enabled", False), "trailing_stop.enabled"
        )

        if self.trailing_stop_enabled:
            # Handle percentage values that might be provided as percentages (e.g., 2.0 = 2%)
            trail_val = trailing_config.get("trail_percent", DEFAULT_TRAIL_PERCENT)
            if isinstance(trail_val, (int, float)):
                if trail_val > 1:
                    trail_percent = trail_val / 100.0  # Convert 2.0 -> 0.02
                else:
                    trail_percent = trail_val
            else:
                trail_percent = DEFAULT_TRAIL_PERCENT / 100.0

            trail_percent = InputSanitizer.sanitize_percentage(
                trail_percent, "trail_percent", min_value=0.001, max_value=0.20
            )

            activation_val = trailing_config.get("activation_percent", DEFAULT_ACTIVATION_PERCENT)
            if isinstance(activation_val, (int, float)):
                if activation_val > 1:
                    activation_percent = activation_val / 100.0
                else:
                    activation_percent = activation_val
            else:
                activation_percent = DEFAULT_ACTIVATION_PERCENT / 100.0

            activation_percent = InputSanitizer.sanitize_percentage(
                activation_percent, "activation_percent", min_value=0.001, max_value=0.20
            )

            update_val = trailing_config.get(
                "update_threshold_percent", DEFAULT_UPDATE_THRESHOLD_PERCENT
            )
            if isinstance(update_val, (int, float)):
                if update_val > 1:
                    update_threshold_percent = update_val / 100.0
                else:
                    update_threshold_percent = update_val
            else:
                update_threshold_percent = DEFAULT_UPDATE_THRESHOLD_PERCENT / 100.0

            update_threshold_percent = InputSanitizer.sanitize_percentage(
                update_threshold_percent,
                "update_threshold_percent",
                min_value=0.001,
                max_value=0.20,
            )

            self.trailing_stop = TrailingStopLoss(
                trail_percent=trail_percent,
                activation_percent=activation_percent,
                update_threshold_percent=update_threshold_percent,
            )
        else:
            self.trailing_stop = None

        # Bot state (sanitized inputs)
        self.symbol = InputSanitizer.sanitize_symbol(self.config["trading"]["symbol"])
        self.timeframe = InputSanitizer.sanitize_timeframe(self.config["trading"]["timeframe"])
        self.check_interval = InputSanitizer.sanitize_check_interval(
            self.config["trading"]["check_interval"]
        )
        self.dry_run = InputSanitizer.sanitize_boolean(
            self.config["trading"].get("dry_run", True), "dry_run"
        )

        self.current_position = None
        self.last_check_time = None
        # Use UTC for daily reset to ensure consistent behavior across timezones
        self.last_daily_reset = datetime.now(timezone.utc).date()

        # Market data cache for performance optimization
        self.market_data_cache: Optional[pd.DataFrame] = None
        self.cache_timestamp: Optional[datetime] = None
        self.cache_max_age_seconds = CACHE_MAX_AGE_SECONDS

        # Multi-timeframe analysis (sanitized inputs)
        multi_tf_config = self.config.get("multi_timeframe", {})
        self.multi_timeframe_enabled = InputSanitizer.sanitize_boolean(
            multi_tf_config.get("enabled", False), "multi_timeframe.enabled"
        )
        if self.multi_timeframe_enabled:
            self.higher_timeframe = InputSanitizer.sanitize_timeframe(
                multi_tf_config.get("higher_timeframe", "4h")
            )
        else:
            self.higher_timeframe = None
        # Higher timeframe cache (longer TTL since it changes less frequently)
        self.higher_tf_cache: Optional[pd.DataFrame] = None
        self.higher_tf_cache_timestamp: Optional[datetime] = None
        self.higher_tf_cache_max_age_seconds = HIGHER_TF_CACHE_MAX_AGE_SECONDS

        # Position persistence
        self.state_file_path = self.config.get("state_file", "data/position_state.json")
        # Ensure data directory exists
        state_dir = os.path.dirname(self.state_file_path)
        if state_dir and not os.path.exists(state_dir):
            os.makedirs(state_dir, exist_ok=True)

        # Load saved position state if available
        self._load_position_state()

        # Health monitoring
        self.start_time = datetime.now(timezone.utc)
        self.last_cycle_time: Optional[datetime] = None
        self.cycle_count = 0
        self.recent_errors: List[Dict] = []  # Store last 20 errors
        self.max_recent_errors = 20

        # Order tracking and retry
        order_tracking_config = self.config.get("order_tracking", {})
        max_retries = order_tracking_config.get("max_retries", DEFAULT_MAX_RETRIES)
        retry_delay = order_tracking_config.get("retry_delay", DEFAULT_RETRY_DELAY)
        self.order_tracker = OrderTracker(
            max_retries=max_retries,
            retry_delay=retry_delay,
            status_check_interval=order_tracking_config.get(
                "status_check_interval", DEFAULT_STATUS_CHECK_INTERVAL
            ),
        )

        # Initialize health monitor if enabled
        self.health_monitor = None
        health_config = self.config.get("health_monitor", {})
        if health_config.get("enabled", False):
            try:
                from health_monitor import HealthMonitor

                port = health_config.get("port", 8080)
                host = health_config.get("host", "127.0.0.1")
                self.health_monitor = HealthMonitor(self, port=port, host=host)
                self.health_monitor.start()
            except ImportError:
                logger.warning(
                    "Flask not available - health monitor disabled. Install with: pip install flask"
                )
            except Exception as e:
                logger.warning(f"Failed to start health monitor: {e}")

        logger.info("=" * 60)
        logger.info(f"{self.exchange_name.upper()} MACD FUTURES TRADING BOT INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Exchange: {self.exchange_name}")
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Timeframe: {self.timeframe}")
        logger.info(f"Leverage: {self.config['risk']['leverage']}x")
        logger.info(f"Dry Run Mode: {self.dry_run}")
        logger.info(f"Risk per trade: {self.config['risk']['max_position_size_pct']*100}%")
        logger.info(f"Trailing Stop: {'ENABLED' if self.trailing_stop_enabled else 'DISABLED'}")
        if self.trailing_stop_enabled:
            logger.info(f"  - Trail Distance: {trailing_config.get('trail_percent', 2.0)}%")
            logger.info(f"  - Activation: {trailing_config.get('activation_percent', 1.0)}%")
        logger.info(f"WebSocket: {'ENABLED' if self.websocket_enabled else 'DISABLED'}")
        if self.websocket_enabled:
            logger.info("  - Real-time price updates enabled")
            logger.info("  - Fallback to REST API if WebSocket disconnects")
        logger.info("=" * 60)

    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration file with comprehensive error handling.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
            ValueError: If config file is empty or invalid
        """
        # Convert to absolute path for consistent handling
        config_path = os.path.abspath(config_path)

        # Validate path is within project directory (prevent path traversal attacks)
        # Get project root directory (directory containing this file)
        project_root = os.path.dirname(os.path.abspath(__file__))

        # Ensure config path is within project root
        try:
            # Get real paths to handle symlinks
            config_path_real = os.path.realpath(config_path)
            project_root_real = os.path.realpath(project_root)

            # Check if config path is within project root
            if not config_path_real.startswith(project_root_real):
                raise ValueError(
                    f"Config file path must be within project directory. "
                    f"Project root: {project_root_real}, "
                    f"Config path: {config_path_real}"
                )
        except (OSError, ValueError) as e:
            # If path validation fails, raise clear error
            if isinstance(e, ValueError):
                raise
            raise ConfigurationError(
                f"Invalid config file path: {config_path}. " f"Path validation failed: {e}"
            ) from e

        # Try to load the file
        try:
            with open(config_path, "r") as f:
                config_content = f.read().strip()

                if not config_content:
                    raise ValueError(
                        f"Configuration file '{config_path}' is empty. "
                        f"Please check your config file."
                    )

                try:
                    config = json.loads(config_content)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON in configuration file '{config_path}': {e}. "
                        f"Please check the file syntax."
                    ) from e

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found: '{config_path}'. "
                f"Please create config/config.json from config/config.example.json"
            ) from None
        except PermissionError:
            raise PermissionError(
                f"Permission denied reading configuration file: '{config_path}'. "
                f"Please check file permissions."
            ) from None
        except Exception as e:
            raise ValueError(f"Error reading configuration file '{config_path}': {e}") from e

        # Validate it's a dictionary
        if not isinstance(config, dict):
            raise ValueError(
                f"Configuration file must contain a JSON object (dict). "
                f"Got: {type(config).__name__}"
            )

        return config

    def _load_credentials(self) -> None:
        """
        Load credentials from secure storage with fallback chain.

        Priority order:
        1. System keyring (most secure)
        2. Environment variables
        3. Config file (with warnings)

        Uses CredentialManager for secure credential retrieval.
        """
        exchange = self.config.get("exchange", "hyperliquid").lower()

        if exchange != "hyperliquid":
            raise ConfigurationError(
                f"Invalid exchange: '{exchange}'. Only 'hyperliquid' is supported. "
                f"Bitunix support has been removed."
            )

        if exchange == "hyperliquid":
            # Get Hyperliquid credentials using credential manager
            creds = self.credential_manager.get_hyperliquid_credentials(self.config)

            # Validate and set credentials
            if creds["private_key"]:
                self.config["private_key"] = creds["private_key"]
                # Check if it's a placeholder
                if (
                    self.config["private_key"].startswith("0x0000")
                    or "example" in self.config["private_key"].lower()
                ):
                    raise ConfigurationError(
                        "Invalid private_key: appears to be a placeholder. "
                        "Please set credentials using:\n"
                        "  - System keyring: python manage_credentials.py set "
                        "hyperliquid_private_key <value>\n"
                        "  - Environment variable: export "
                        "HYPERLIQUID_PRIVATE_KEY=<value>\n"
                        "  - Config file: update config/config.json"
                    )
            else:
                raise ConfigurationError(
                    "Hyperliquid private_key not found. Please set credentials using:\n"
                    "  - System keyring: python manage_credentials.py set "
                    "hyperliquid_private_key <value>\n"
                    "  - Environment variable: export "
                    "HYPERLIQUID_PRIVATE_KEY=<value>\n"
                    "  - Config file: update config/config.json"
                )

            if creds["wallet_address"]:
                self.config["wallet_address"] = creds["wallet_address"]
                # Check if it's a placeholder
                if (
                    self.config["wallet_address"].startswith("0x0000")
                    or "example" in self.config["wallet_address"].lower()
                ):
                    raise ConfigurationError(
                        "Invalid wallet_address: appears to be a placeholder. "
                        "Please set credentials using:\n"
                        "  - System keyring: python manage_credentials.py set "
                        "hyperliquid_wallet_address <value>\n"
                        "  - Environment variable: export "
                        "HYPERLIQUID_WALLET_ADDRESS=<value>\n"
                        "  - Config file: update config/config.json"
                    )
            else:
                raise ConfigurationError(
                    "Hyperliquid wallet_address not found. Please set credentials using:\n"
                    "  - System keyring: python manage_credentials.py set "
                    "hyperliquid_wallet_address <value>\n"
                    "  - Environment variable: export "
                    "HYPERLIQUID_WALLET_ADDRESS=<value>\n"
                    "  - Config file: update config/config.json"
                )

    def _validate_exchange_config(self) -> str:
        """
        Validate exchange configuration.

        Returns:
            Exchange name (lowercase)

        Raises:
            ValueError: If exchange is missing or invalid
        """
        exchange = self.config.get("exchange", "").lower()

        if not exchange:
            raise ValueError(
                "Missing required field: 'exchange'. "
                "Must be 'hyperliquid' (Bitunix support has been removed). "
                "Set in config.json or via environment variable."
            )

        if exchange != "hyperliquid":
            raise ConfigurationError(
                f"Invalid exchange '{exchange}'. "
                f"Only 'hyperliquid' is supported (Bitunix support has been removed). "
                f"Check config.json exchange field."
            )

        return exchange

    def _validate_exchange_credentials(self, exchange: str) -> None:
        """
        Validate exchange-specific credentials.

        Args:
            exchange: Exchange name ('hyperliquid' or 'bitunix')

        Raises:
            ValueError: If required credentials are missing
        """
        if exchange == "hyperliquid":
            if "private_key" not in self.config or not self.config["private_key"]:
                raise ValueError(
                    "Missing required field for Hyperliquid: 'private_key'. "
                    "Set HYPERLIQUID_PRIVATE_KEY environment variable or add to config.json. "
                    "Private key must start with 0x and be 66 characters."
                )
            if "wallet_address" not in self.config or not self.config["wallet_address"]:
                raise ValueError(
                    "Missing required field for Hyperliquid: 'wallet_address'. "
                    "Set HYPERLIQUID_WALLET_ADDRESS environment variable or add to config.json. "
                    "Wallet address must start with 0x and be 42 characters."
                )

    def _validate_trading_config(self) -> None:
        """
        Validate trading configuration section.

        Raises:
            ValueError: If trading section is missing or invalid
        """
        if "trading" not in self.config:
            raise ValueError(
                "Missing required section: 'trading'. "
                "Add trading section to config.json with symbol, timeframe, and check_interval."
            )

        trading = self.config["trading"]
        required_trading_fields = ["symbol", "timeframe", "check_interval"]
        missing_fields = [f for f in required_trading_fields if f not in trading]

        if missing_fields:
            raise ConfigurationError(
                f"Missing required fields in 'trading' section: {', '.join(missing_fields)}. "
                f"Required fields: {', '.join(required_trading_fields)}. "
                f"Add missing fields to config.json."
            )

        # Validate timeframe format
        timeframe = trading.get("timeframe", "")
        valid_timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]
        if timeframe and timeframe.lower() not in valid_timeframes:
            logger.warning(
                f"Unusual timeframe '{timeframe}' - supported: {', '.join(valid_timeframes)}. "
                f"Bot may not work correctly with unsupported timeframes."
            )

        # Validate check_interval is reasonable
        check_interval = trading.get("check_interval", 0)
        if check_interval < 60:
            logger.warning(
                f"Check interval {check_interval}s is very short (< 60s). "
                f"This may cause excessive API calls and rate limiting."
            )

    def _validate_strategy_config(self) -> None:
        """
        Validate strategy configuration section.

        Raises:
            ValueError: If strategy section is missing or invalid
        """
        if "strategy" not in self.config:
            raise ValueError(
                "Missing required section: 'strategy'. "
                "Add strategy section to config.json with MACD parameters."
            )

        strategy = self.config["strategy"]
        required_strategy_fields = [
            "fast_length",
            "slow_length",
            "signal_length",
            "risk_reward_ratio",
        ]
        missing_fields = [f for f in required_strategy_fields if f not in strategy]

        if missing_fields:
            raise ConfigurationError(
                f"Missing required fields in 'strategy' section: {', '.join(missing_fields)}. "
                f"Required fields: {', '.join(required_strategy_fields)}. "
                f"Add missing fields to config.json."
            )

        # Validate strategy parameters are reasonable
        fast_length = strategy.get("fast_length", 0)
        slow_length = strategy.get("slow_length", 0)

        if fast_length >= slow_length:
            raise ConfigurationError(
                f"Invalid MACD parameters: fast_length ({fast_length}) must be less than "
                f"slow_length ({slow_length}). "
                f"Check strategy configuration."
            )

        if risk_reward_ratio := strategy.get("risk_reward_ratio", 0):
            if risk_reward_ratio < 1.0:
                logger.warning(
                    f"Risk/reward ratio {risk_reward_ratio} is less than 1:1. "
                    f"This means risking more than potential profit - not recommended."
                )

    def _validate_risk_config(self) -> None:
        """
        Validate risk management configuration section.

        Raises:
            ValueError: If risk section is missing or invalid
        """
        if "risk" not in self.config:
            raise ValueError(
                "Missing required section: 'risk'. "
                "Add risk section to config.json with leverage and risk limits."
            )

        risk = self.config["risk"]
        required_risk_fields = [
            "leverage",
            "max_position_size_pct",
            "max_daily_loss_pct",
            "max_trades_per_day",
        ]
        missing_fields = [f for f in required_risk_fields if f not in risk]

        if missing_fields:
            raise ConfigurationError(
                f"Missing required fields in 'risk' section: {', '.join(missing_fields)}. "
                f"Required fields: {', '.join(required_risk_fields)}. "
                f"Add missing fields to config.json."
            )

        # Validate numeric values are positive
        leverage = risk.get("leverage", 0)
        if leverage <= 0:
            raise ConfigurationError(
                f"Invalid 'risk.leverage': {leverage}. "
                f"Leverage must be positive (got {leverage}). "
                f"Typical values: 1-50 for Hyperliquid, 1-125 for Bitunix."
            )

        max_position_size_pct = risk.get("max_position_size_pct", 0)
        if max_position_size_pct <= 0:
            raise ConfigurationError(
                f"Invalid 'risk.max_position_size_pct': {max_position_size_pct}. "
                f"Must be positive (got {max_position_size_pct}). "
                f"Typical values: 0.05-0.20 (5%-20% of equity)."
            )
        if max_position_size_pct > 1.0:
            pct_value = max_position_size_pct * 100
            raise ConfigurationError(
                f"Invalid 'risk.max_position_size_pct': {max_position_size_pct} "
                f"({pct_value}%). Cannot exceed 100% (1.0). "
                f"Current value: {pct_value}%. "
                f"Reduce to a reasonable value (typically 5-20%)."
            )

        max_daily_loss_pct = risk.get("max_daily_loss_pct", 0)
        if max_daily_loss_pct <= 0:
            raise ConfigurationError(
                f"Invalid 'risk.max_daily_loss_pct': {max_daily_loss_pct}. "
                f"Must be positive (got {max_daily_loss_pct}). "
                f"Typical values: 0.02-0.10 (2%-10% of equity)."
            )
        if max_daily_loss_pct > 1.0:
            pct_value = max_daily_loss_pct * 100
            raise ConfigurationError(
                f"Invalid 'risk.max_daily_loss_pct': {max_daily_loss_pct} "
                f"({pct_value}%). Cannot exceed 100% (1.0). "
                f"Current value: {pct_value}%. "
                f"Reduce to a reasonable value (typically 2-10%)."
            )

        max_trades_per_day = risk.get("max_trades_per_day", 0)
        if max_trades_per_day <= 0:
            raise ConfigurationError(
                f"Invalid 'risk.max_trades_per_day': {max_trades_per_day}. "
                f"Must be positive integer (got {max_trades_per_day}). "
                f"Typical values: 5-20 trades per day."
            )
        if not isinstance(max_trades_per_day, int):
            raise ConfigurationError(
                f"Invalid 'risk.max_trades_per_day': {max_trades_per_day}. "
                f"Must be an integer (got {type(max_trades_per_day).__name__})."
            )

    def _validate_config(self) -> None:
        """
        Validate complete configuration structure and required fields.

        This method orchestrates validation by calling specialized validation methods
        for each configuration section. Validation is performed in order:
        1. Exchange configuration
        2. Exchange-specific credentials
        3. Trading configuration
        4. Strategy configuration
        5. Risk management configuration

        Raises:
            ValueError: If any required fields are missing or invalid
        """
        # Validate exchange configuration
        exchange = self._validate_exchange_config()

        # Validate exchange-specific credentials
        self._validate_exchange_credentials(exchange)

        # Validate trading configuration
        self._validate_trading_config()

        # Validate strategy configuration
        self._validate_strategy_config()

        # Validate risk management configuration
        self._validate_risk_config()

        logger.debug("Configuration validation passed")

    def _get_timeframe_seconds(self, timeframe: Optional[str] = None) -> int:
        """
        Convert timeframe string to seconds.

        Args:
            timeframe: Timeframe string (defaults to self.timeframe)

        Returns:
            Number of seconds in one candle
        """
        if timeframe is None:
            timeframe = self.timeframe

        timeframe_map = {
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "6h": 21600,
            "12h": 43200,
            "1d": 86400,
        }

        return timeframe_map.get(
            timeframe.lower() if timeframe else self.timeframe.lower(), 300
        )  # Default to 5m

    def _is_cache_valid(self) -> bool:
        """
        Check if cached market data is still valid.

        Returns:
            True if cache exists and is not expired
        """
        if self.market_data_cache is None or self.cache_timestamp is None:
            return False

        # Check if cache is too old
        cache_age = (datetime.now(timezone.utc) - self.cache_timestamp).total_seconds()
        if cache_age > self.cache_max_age_seconds:
            logger.debug(f"Cache expired (age: {cache_age:.1f}s)")
            return False

        # Check if we have enough data
        if len(self.market_data_cache) < MIN_CANDLES_FOR_MACD:
            logger.debug("Cache has insufficient data")
            return False

        return True

    def _validate_market_data(self, df: pd.DataFrame, min_candles: int = None) -> bool:
        """
        Validate market data DataFrame.

        Args:
            df: DataFrame to validate
            min_candles: Minimum number of candles required (None = use strategy minimum)

        Returns:
            True if valid, False otherwise
        """
        if df is None:
            logger.error("Market data DataFrame is None")
            return False

        if df.empty:
            logger.error("Market data DataFrame is empty")
            return False

        # Check minimum required candles
        if min_candles is None:
            # Use strategy's minimum requirement
            min_candles = self.strategy.min_candles

        if len(df) < min_candles:
            logger.error(
                f"Insufficient market data: {len(df)} candles, "
                f"need at least {min_candles} for strategy calculation"
            )
            return False

        # Validate required columns exist
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        # Check for NaN values in critical columns
        critical_columns = ["open", "high", "low", "close"]
        for col in critical_columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                logger.error(
                    f"Found {nan_count} NaN values in '{col}' column. "
                    f"Data quality insufficient for trading."
                )
                return False

        # Validate price data is reasonable (positive values)
        if (df[critical_columns] <= 0).any().any():
            logger.error("Found non-positive price values in market data")
            return False

        # Validate timestamps are in order (most recent last)
        if len(df) > 1:
            timestamps = pd.to_datetime(df["timestamp"])
            if not timestamps.is_monotonic_increasing:
                logger.warning("Timestamps are not in chronological order")
                # Sort by timestamp
                df.sort_values("timestamp", inplace=True)
                df.reset_index(drop=True, inplace=True)

        return True

    def get_market_data(self) -> pd.DataFrame:
        """
        Fetch market data from exchange with validation and caching.

        Optimized approach:
        - Uses cached data if still valid
        - Fetches only new candles incrementally when possible
        - Falls back to full fetch only when necessary

        Returns:
            Validated DataFrame with market data

        Raises:
            ValueError: If market data is invalid or unavailable
        """
        # Check if we can use cached data
        if self._is_cache_valid():
            logger.debug(f"Using cached market data ({len(self.market_data_cache)} candles)")
            return self.market_data_cache.copy()

        # Determine how many candles to fetch
        if self.market_data_cache is not None and len(self.market_data_cache) > 0:
            # Calculate time since last cache update
            time_since_cache = (datetime.now(timezone.utc) - self.cache_timestamp).total_seconds()
            timeframe_seconds = self._get_timeframe_seconds()

            # Calculate expected new candles (add buffer of 2 candles for safety)
            expected_new_candles = int(time_since_cache / timeframe_seconds) + 2

            # Limit to reasonable range
            candles_to_fetch = min(
                max(expected_new_candles, MIN_INCREMENTAL_CANDLES), MAX_INCREMENTAL_CANDLES
            )
            logger.debug(f"Incremental fetch: requesting {candles_to_fetch} new candles")
        else:
            # No cache or empty cache - fetch full dataset
            candles_to_fetch = DEFAULT_CANDLES_TO_FETCH
            logger.debug(
                f"Full fetch: requesting {DEFAULT_CANDLES_TO_FETCH} candles (no valid cache)"
            )

        # Fetch data from exchange
        try:
            klines = self.client.get_klines(
                symbol=self.symbol, interval=self.timeframe, limit=candles_to_fetch
            )

            if not klines or len(klines) == 0:
                return self._handle_fetch_failure("No klines data returned")

            # Convert to DataFrame
            df_new = pd.DataFrame(klines)
            df_new.columns = ["timestamp", "open", "high", "low", "close", "volume"]

            # Convert to numeric
            for col in ["open", "high", "low", "close", "volume"]:
                df_new[col] = pd.to_numeric(df_new[col], errors="coerce")

            df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms")

            # Merge with cache if we did incremental fetch
            if candles_to_fetch < DEFAULT_CANDLES_TO_FETCH and self.market_data_cache is not None:
                # Combine old and new data
                df_combined = pd.concat([self.market_data_cache, df_new], ignore_index=True)

                # Remove duplicates (keep latest)
                df_combined = df_combined.drop_duplicates(subset=["timestamp"], keep="last")

                # Sort by timestamp
                df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)

                # Keep only last N candles to prevent unlimited growth
                if len(df_combined) > DEFAULT_CANDLES_TO_FETCH:
                    df_combined = df_combined.iloc[-DEFAULT_CANDLES_TO_FETCH:].reset_index(
                        drop=True
                    )

                df = df_combined
                logger.debug(f"Merged cache + {len(df_new)} new candles = {len(df)} total")
            else:
                df = df_new
                logger.debug(f"Fetched {len(df)} candles for {self.symbol}")

            # Validate the data
            if not self._validate_market_data(df):
                return self._handle_fetch_failure("Data validation failed")

            # Update cache
            self.market_data_cache = df.copy()
            self.cache_timestamp = datetime.now(timezone.utc)

            return df

        except Exception as e:
            logger.warning(f"Error fetching klines: {e}")
            return self._handle_fetch_failure(f"API error: {e}")

    def _handle_fetch_failure(self, reason: str, prefer_stale_cache: bool = True) -> pd.DataFrame:
        """
        Handle market data fetch failures with fallback strategies.

        Fallback priority:
        1. Use stale cache (if available and prefer_stale_cache=True)
        2. Use ticker fallback (single candle from current price)
        3. Raise ValueError if all fallbacks fail

        Args:
            reason: Reason for the fetch failure (for logging)
            prefer_stale_cache: If True, prefer stale cache over ticker fallback

        Returns:
            DataFrame with market data from fallback source

        Raises:
            ValueError: If all fallback sources fail
        """
        # First try: Use stale cache if available
        if prefer_stale_cache and self.market_data_cache is not None:
            logger.warning(f"{reason} - using stale cache as fallback")
            return self.market_data_cache.copy()

        # Second try: Use ticker fallback
        df_fallback = self._get_ticker_fallback()
        if df_fallback is not None:
            logger.warning(f"{reason} - using ticker fallback (limited functionality)")
            return df_fallback

        # Last resort: Try stale cache even if not preferred (if available)
        if not prefer_stale_cache and self.market_data_cache is not None:
            logger.warning(f"{reason} - using stale cache as last resort")
            return self.market_data_cache.copy()

        # All fallbacks failed
        cache_status = (
            "(unavailable)"
            if self.market_data_cache is None
            else "(available but not used)"
        )
        ticker_status = (
            "(unavailable)" if df_fallback is None else "(available)"
        )
        raise MarketDataUnavailableError(
            f"All market data sources failed. Reason: {reason}. "
            f"Attempted fallbacks: 1) Stale cache {cache_status}, "
            f"2) Ticker fallback {ticker_status}. "
            f"Cannot proceed without market data. "
            f"Check: 1) Exchange API connectivity, 2) Symbol '{self.symbol}' "
            f"validity, 3) Network connection, 4) Exchange API status."
        )

    def _use_fallback_data(self) -> pd.DataFrame:
        """
        Attempt to use ticker fallback data.

        Returns:
            DataFrame from ticker fallback

        Raises:
            ValueError: If fallback also fails
        """
        return self._handle_fetch_failure("Fallback requested", prefer_stale_cache=False)

    def clear_market_data_cache(self) -> None:
        """
        Clear the market data cache to force fresh data fetch.

        Useful when:
        - Detecting data anomalies
        - After extended downtime
        - Manual refresh needed
        """
        self.market_data_cache = None
        self.cache_timestamp = None
        logger.debug("Market data cache cleared - next fetch will be full refresh")

    def get_higher_timeframe_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch higher timeframe market data for multi-timeframe analysis.

        Returns:
            DataFrame with higher timeframe data, or None if multi-timeframe disabled
        """
        if not self.multi_timeframe_enabled or not self.higher_timeframe:
            return None

        # Check cache validity
        if self.higher_tf_cache is not None and self.higher_tf_cache_timestamp:
            cache_age = (
                datetime.now(timezone.utc) - self.higher_tf_cache_timestamp
            ).total_seconds()
            if cache_age < self.higher_tf_cache_max_age_seconds:
                logger.debug(
                    f"Using cached higher timeframe data ({len(self.higher_tf_cache)} candles)"
                )
                return self.higher_tf_cache.copy()

        try:
            # Fetch higher timeframe data (need fewer candles since timeframe is larger)
            # For example: if trading on 1h and checking 4h, we need ~50 candles
            # for 4h = 200 hours = ~8 days
            candles_needed = HIGHER_TF_CANDLES_NEEDED

            logger.debug(
                f"Fetching {candles_needed} candles for higher timeframe {self.higher_timeframe}"
            )

            klines = self.client.get_klines(
                symbol=self.symbol, interval=self.higher_timeframe, limit=candles_needed
            )

            if not klines or len(klines) == 0:
                logger.warning(f"No higher timeframe data available for {self.higher_timeframe}")
                return None

            # Convert to DataFrame
            df_higher_tf = pd.DataFrame(klines)
            df_higher_tf.columns = ["timestamp", "open", "high", "low", "close", "volume"]

            # Convert to numeric
            for col in ["open", "high", "low", "close", "volume"]:
                df_higher_tf[col] = pd.to_numeric(df_higher_tf[col], errors="coerce")

            df_higher_tf["timestamp"] = pd.to_datetime(df_higher_tf["timestamp"], unit="ms")

            # Validate
            if not self._validate_market_data(df_higher_tf, min_candles=self.strategy.min_candles):
                logger.warning("Higher timeframe data validation failed")
                return None

            # Update cache
            self.higher_tf_cache = df_higher_tf.copy()
            self.higher_tf_cache_timestamp = datetime.now(timezone.utc)

            logger.debug(
                f"Fetched {len(df_higher_tf)} candles for higher timeframe {self.higher_timeframe}"
            )

            return df_higher_tf

        except Exception as e:
            logger.error(f"Error fetching higher timeframe data: {e}")
            # Return cached data if available (even if stale)
            if self.higher_tf_cache is not None:
                logger.warning("Using stale higher timeframe cache due to fetch error")
                return self.higher_tf_cache.copy()
            return None

    def _save_position_state(self) -> None:
        """
        Save current position state and trailing stop to file.

        Persists position data so bot can resume after restart without losing
        position tracking. Includes trailing stop state if enabled.
        """
        try:
            state_data = {
                "position": None,
                "trailing_stop": None,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "symbol": self.symbol,
                "exchange": self.exchange_name.lower(),
            }

            # Save position if exists
            if self.current_position:
                # Convert datetime objects to ISO format strings for JSON serialization
                position_copy = self.current_position.copy()
                if "entry_time" in position_copy and isinstance(
                    position_copy["entry_time"], datetime
                ):
                    position_copy["entry_time"] = position_copy["entry_time"].isoformat()
                state_data["position"] = position_copy

            # Save trailing stop state if enabled and initialized
            if self.trailing_stop_enabled and self.trailing_stop:
                trailing_stop_status = self.trailing_stop.get_status()
                if trailing_stop_status.get("initialized", False):
                    # Convert datetime to ISO format if present
                    if trailing_stop_status.get("last_update"):
                        if isinstance(trailing_stop_status["last_update"], datetime):
                            trailing_stop_status["last_update"] = trailing_stop_status[
                                "last_update"
                            ].isoformat()
                    state_data["trailing_stop"] = trailing_stop_status

            # Write to file atomically (write to temp file, then rename)
            temp_file = self.state_file_path + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(state_data, f, indent=2)

            # Atomic rename (works on Unix/Linux/Mac, Windows may need different approach)
            if os.name == "nt":  # Windows
                if os.path.exists(self.state_file_path):
                    os.remove(self.state_file_path)
                os.rename(temp_file, self.state_file_path)
            else:  # Unix/Linux/Mac
                os.rename(temp_file, self.state_file_path)

            logger.debug(f"Position state saved to {self.state_file_path}")

        except Exception as e:
            logger.error(f"Error saving position state: {e}")
            # Don't raise - position tracking should continue even if save fails

    def _load_position_state(self) -> None:
        """
        Load saved position state from file.

        Restores position and trailing stop state on bot restart.
        Validates loaded state against exchange position to ensure consistency.
        """
        try:
            if not os.path.exists(self.state_file_path):
                logger.debug("No saved position state found - starting fresh")
                return

            with open(self.state_file_path, "r") as f:
                state_data = json.load(f)

            # Validate state file matches current symbol and exchange
            saved_symbol = state_data.get("symbol", "")
            saved_exchange = state_data.get("exchange", "").lower()

            if saved_symbol != self.symbol:
                logger.warning(
                    f"Saved position state is for symbol '{saved_symbol}', "
                    f"but bot is configured for '{self.symbol}'. Ignoring saved state."
                )
                return

            if saved_exchange != self.exchange_name.lower():
                logger.warning(
                    f"Saved position state is for exchange '{saved_exchange}', "
                    f"but bot is configured for '{self.exchange_name}'. Ignoring saved state."
                )
                return

            # Restore position
            saved_position = state_data.get("position")
            if saved_position:
                # Convert ISO format strings back to datetime objects
                if "entry_time" in saved_position and isinstance(saved_position["entry_time"], str):
                    try:
                        saved_position["entry_time"] = datetime.fromisoformat(
                            saved_position["entry_time"].replace("Z", "+00:00")
                        )
                    except (ValueError, AttributeError):
                        # Fallback to current time if parsing fails
                        saved_position["entry_time"] = datetime.now(timezone.utc)

                self.current_position = saved_position
                logger.info(
                    f"✅ Restored position from saved state: "
                    f"{saved_position.get('type', 'UNKNOWN')} "
                    f"@ ${saved_position.get('entry_price', 0):.2f}, "
                    f"Qty: {saved_position.get('quantity', 0)}"
                )

            # Restore trailing stop state if enabled
            if self.trailing_stop_enabled and self.trailing_stop:
                saved_trailing_stop = state_data.get("trailing_stop")
                if saved_trailing_stop and saved_trailing_stop.get("initialized", False):
                    try:
                        # Restore trailing stop state
                        self.trailing_stop.is_active = saved_trailing_stop.get("active", False)
                        self.trailing_stop.position_type = saved_trailing_stop.get("position_type")
                        self.trailing_stop.entry_price = saved_trailing_stop.get("entry_price")
                        self.trailing_stop.best_price = saved_trailing_stop.get("best_price")
                        self.trailing_stop.current_stop_loss = saved_trailing_stop.get(
                            "current_stop_loss"
                        )

                        # Convert ISO format string back to datetime
                        if saved_trailing_stop.get("last_update"):
                            last_update_str = saved_trailing_stop["last_update"]
                            if isinstance(last_update_str, str):
                                try:
                                    self.trailing_stop.last_update_time = datetime.fromisoformat(
                                        last_update_str.replace("Z", "+00:00")
                                    )
                                except (ValueError, AttributeError):
                                    self.trailing_stop.last_update_time = None

                        logger.info(
                            f"✅ Restored trailing stop state: "
                            f"{'ACTIVE' if self.trailing_stop.is_active else 'INACTIVE'}, "
                            f"Stop: ${self.trailing_stop.current_stop_loss:.2f}"
                        )
                    except Exception as e:
                        logger.warning(f"Error restoring trailing stop state: {e}")
                        # Reset trailing stop if restoration fails
                        if self.trailing_stop:
                            self.trailing_stop.reset()

            # Validate restored state against exchange
            logger.debug("Validating restored position against exchange...")
            self._sync_position_with_exchange()

            saved_at = state_data.get("saved_at", "unknown")
            logger.debug(f"Position state loaded (saved at: {saved_at})")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in position state file: {e}")
            # Backup corrupted file
            backup_path = self.state_file_path + ".corrupted"
            try:
                os.rename(self.state_file_path, backup_path)
                logger.debug(f"Corrupted state file backed up to {backup_path}")
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Error loading position state: {e}")
            # Don't raise - bot should continue even if state load fails

    def _record_error(self, error_message: str, exc_info: bool = False) -> None:
        """
        Record error for health monitoring.

        Args:
            error_message: Error message string
            exc_info: Whether to include exception info
        """
        error_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": error_message,
        }

        self.recent_errors.append(error_entry)

        # Keep only last N errors
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors = self.recent_errors[-self.max_recent_errors :]

    def _get_ticker_fallback(self) -> Optional[pd.DataFrame]:
        """
        Fallback method to get current price from ticker.

        Returns:
            DataFrame with single candle, or None if failed
        """
        try:
            ticker = self.client.get_ticker(self.symbol)
            if ticker and "markPrice" in ticker:
                current_price = float(ticker["markPrice"])
                # Create a simple 1-candle DataFrame
                df = pd.DataFrame(
                    [
                        {
                            "timestamp": pd.Timestamp.now(),
                            "open": current_price,
                            "high": current_price,
                            "low": current_price,
                            "close": current_price,
                            "volume": 1.0,
                        }
                    ]
                )
                logger.debug(f"Using ticker data fallback: ${current_price:,.2f}")
                # Note: This fallback data is insufficient for strategy calculation
                # but can be used for dry run testing
                return df
        except Exception as e:
            logger.error(f"Fallback ticker data also failed: {e}")

        return None

    def get_account_balance(self) -> float:
        """
        Get available balance in USDT.

        Returns:
            Account balance in USDT

        Raises:
            ValueError: If balance cannot be retrieved in live trading mode
        """
        try:
            account_info = self.client.get_account_info()

            if account_info and "balance" in account_info:
                balance = float(account_info["balance"])

                # Validate balance is reasonable
                if balance < 0:
                    raise ValueError(
                        f"Invalid balance retrieved from exchange: ${balance:,.2f}. "
                        f"Balance cannot be negative. "
                        f"Exchange: {self.exchange_name}, Symbol: {self.symbol}. "
                        f"Check exchange API response and account status."
                    )

                logger.debug(f"Account Balance: ${balance:,.2f} USDT")
                return balance

            # Could not retrieve balance
            if self.dry_run:
                logger.warning(
                    "⚠️ Could not retrieve balance from exchange. "
                    "Using default $1000.00 for DRY RUN mode only."
                )
                logger.warning(
                    "⚠️ WARNING: This default balance may not reflect actual account state. "
                    "Verify API connection before live trading!"
                )
                return DEFAULT_BALANCE_DRY_RUN  # Only allow default in dry run mode
            else:
                raise ValueError(
                    f"Cannot retrieve account balance in LIVE TRADING mode. "
                    f"Exchange: {self.exchange_name}, Symbol: {self.symbol}. "
                    f"API response: {account_info if account_info else 'None'}. "
                    f"Response missing 'balance' field or empty. "
                    f"This is required for risk management. "
                    f"Check: 1) Exchange API connectivity, 2) Authentication credentials, "
                    f"3) Account permissions, 4) Exchange API status."
                )

        except ValueError:
            # Re-raise ValueError (our validation errors)
            raise
        except Exception as e:
            # API call failed
            if self.dry_run:
                logger.warning(
                    f"⚠️ Account endpoint failed: {e}. "
                    "Using default $1000.00 for DRY RUN mode only."
                )
                logger.warning(
                    "⚠️ WARNING: This default balance may not reflect actual account state. "
                    "Verify API connection before live trading!"
                )
                return DEFAULT_BALANCE_DRY_RUN  # Only allow default in dry run mode
            else:
                raise ValueError(
                    f"Failed to retrieve account balance in LIVE TRADING mode: {e}. "
                    "This is required for risk management. "
                    "Please check your API connection and credentials."
                ) from e

    def check_existing_position(self) -> Optional[Dict]:
        """Check if there's an existing position."""
        try:
            position = self.client.get_position(self.symbol)

            if position and float(position.get("holdAmount", 0)) != 0:
                logger.debug(f"Existing position found: {position}")
                return position

            return None

        except Exception as e:
            logger.error(f"Error checking position: {e}")
            return None

    def _sync_position_with_exchange(self) -> None:
        """
        Synchronize tracked position with exchange position.

        Compares exchange position with tracked position and only updates
        if there's a meaningful difference. Preserves trailing stop state
        and position metadata.
        """
        try:
            exchange_position = self.check_existing_position()

            # No position on exchange
            if exchange_position is None:
                if self.current_position is not None:
                    logger.debug("Position closed on exchange - clearing tracked position")
                    self.current_position = None
                    # Reset trailing stop
                    if self.trailing_stop_enabled and self.trailing_stop:
                        self.trailing_stop.reset()
                    self._save_position_state()
                return

            # Position exists on exchange
            # Handle different exchange formats (Bitunix vs Hyperliquid)
            exchange_qty = float(
                exchange_position.get("holdAmount", exchange_position.get("holdQty", 0))
            )
            exchange_side = exchange_position.get("side", exchange_position.get("positionSide", ""))
            # Try different field names for entry price
            exchange_entry_price = float(
                exchange_position.get(
                    "entryPrice",
                    exchange_position.get("avgPrice", exchange_position.get("openPrice", 0)),
                )
            )

            # No tracked position - initialize from exchange
            if self.current_position is None:
                logger.debug("New position detected on exchange - initializing tracking")
                # Determine position type from exchange side
                position_type = exchange_side if exchange_side in ["LONG", "SHORT"] else "LONG"

                # Create position dict with exchange data
                self.current_position = {
                    "type": position_type,
                    "entry_price": exchange_entry_price,
                    "quantity": exchange_qty,
                    "stop_loss": (
                        exchange_entry_price * 0.98
                        if position_type == "LONG"
                        else exchange_entry_price * 1.02
                    ),  # Default SL
                    "take_profit": (
                        exchange_entry_price * 1.02
                        if position_type == "LONG"
                        else exchange_entry_price * 0.98
                    ),  # Default TP
                    "entry_time": datetime.now(),
                    "from_exchange": True,  # Flag to indicate this came from exchange
                }

                # Initialize trailing stop if enabled
                if self.trailing_stop_enabled and self.trailing_stop:
                    self.trailing_stop.initialize_position(
                        entry_price=exchange_entry_price,
                        initial_stop_loss=self.current_position["stop_loss"],
                        position_type=position_type,
                    )

                logger.warning(
                    f"⚠️ Position initialized from exchange. "
                    f"TP/SL may not match exchange settings. "
                    f"Entry: ${exchange_entry_price:.2f}, Qty: {exchange_qty}"
                )
                self._save_position_state()
                return

            # Compare tracked position with exchange position
            tracked_qty = float(self.current_position.get("quantity", 0))
            tracked_type = self.current_position.get("type", "")

            # Check if position changed significantly
            qty_diff = abs(exchange_qty - tracked_qty)
            qty_diff_pct = (qty_diff / tracked_qty * 100) if tracked_qty > 0 else 100

            # Position closed or significantly changed (more than threshold % difference)
            if exchange_qty == 0 or qty_diff_pct > POSITION_QTY_DIFF_THRESHOLD_PCT:
                if exchange_qty == 0:
                    logger.debug("Position closed on exchange - clearing tracked position")
                    self.current_position = None
                    if self.trailing_stop_enabled and self.trailing_stop:
                        self.trailing_stop.reset()
                    self._save_position_state()
                else:
                    logger.warning(
                        f"Position quantity changed significantly: "
                        f"{tracked_qty} → {exchange_qty} ({qty_diff_pct:.1f}% change)"
                    )
                    # Log position update to audit log
                    try:
                        self.audit_logger.log_position_update(
                            symbol=self.symbol,
                            position_type=tracked_type or exchange_side,
                            quantity=exchange_qty,
                            entry_price=exchange_entry_price,
                            mark_price=exchange_entry_price,  # Use entry price as approximation
                            unrealized_pnl=0.0,  # Will be updated by exchange
                            reason=f"Position quantity changed: {tracked_qty} → {exchange_qty}",
                        )
                    except Exception as e:
                        logger.error(f"Failed to write audit log entry: {e}")

                    # Update quantity but preserve other metadata
                    self.current_position["quantity"] = exchange_qty
                    self.current_position["entry_price"] = exchange_entry_price
                    # Ensure trailing stop is initialized if enabled
                    if self.trailing_stop_enabled and self.trailing_stop:
                        if self.trailing_stop.entry_price is None:
                            logger.debug(
                                "Initializing trailing stop after position quantity change"
                            )
                            self.trailing_stop.initialize_position(
                                entry_price=exchange_entry_price,
                                initial_stop_loss=self.current_position.get(
                                    "stop_loss",
                                    (
                                        exchange_entry_price * 0.98
                                        if tracked_type == "LONG"
                                        else exchange_entry_price * 1.02
                                    ),
                                ),
                                position_type=tracked_type or exchange_side,
                            )
                    self._save_position_state()
                return

            # Position type mismatch (shouldn't happen, but check anyway)
            if tracked_type and exchange_side and tracked_type != exchange_side:
                logger.error(
                    f"Position type mismatch: tracked={tracked_type}, exchange={exchange_side}. "
                    f"This should not happen!"
                )
                # Update to match exchange
                self.current_position["type"] = exchange_side
                # Re-initialize trailing stop with correct position type
                if self.trailing_stop_enabled and self.trailing_stop:
                    logger.debug("Re-initializing trailing stop due to position type change")
                    self.trailing_stop.initialize_position(
                        entry_price=exchange_entry_price,
                        initial_stop_loss=self.current_position.get(
                            "stop_loss",
                            (
                                exchange_entry_price * 0.98
                                if exchange_side == "LONG"
                                else exchange_entry_price * 1.02
                            ),
                        ),
                        position_type=exchange_side,
                    )
                return

            # Position is the same - preserve tracked position metadata
            # Ensure trailing stop is initialized if enabled
            if self.trailing_stop_enabled and self.trailing_stop:
                # Check if trailing stop needs initialization
                if self.trailing_stop.entry_price is None:
                    logger.debug("Initializing trailing stop for existing tracked position")
                    self.trailing_stop.initialize_position(
                        entry_price=self.current_position.get("entry_price", exchange_entry_price),
                        initial_stop_loss=self.current_position.get(
                            "stop_loss",
                            (
                                exchange_entry_price * 0.98
                                if tracked_type == "LONG"
                                else exchange_entry_price * 1.02
                            ),
                        ),
                        position_type=tracked_type or exchange_side,
                    )

            # Only update entry price if it's significantly different (partial fills)
            entry_diff = abs(exchange_entry_price - self.current_position.get("entry_price", 0))
            if entry_diff > MIN_ENTRY_PRICE_DIFF:
                logger.debug(
                    f"Updating entry price: ${self.current_position.get('entry_price', 0):.2f} → "
                    f"${exchange_entry_price:.2f}"
                )
                self.current_position["entry_price"] = exchange_entry_price
                self._save_position_state()

            # Position is essentially the same - no update needed
            # Trailing stop state is preserved

        except Exception as e:
            logger.error(f"Error syncing position with exchange: {e}", exc_info=True)
            # Don't update current_position on error - preserve existing state

    def _prepare_entry_order(self, signal: Dict, balance: float) -> Optional[Dict]:
        """
        Prepare entry order by calculating position size and validating order.

        Args:
            signal: Entry signal from strategy
            balance: Current account balance

        Returns:
            Dictionary with order details if valid, None otherwise
        """
        try:
            entry_price = signal["entry_price"]
            stop_loss = signal["stop_loss"]
            take_profit = signal["take_profit"]
            position_type = signal["type"]

            # Calculate position size, accounting for existing positions
            existing_positions = [self.current_position] if self.current_position else []
            size_info = self.risk_manager.calculate_position_size(
                balance=balance,
                entry_price=entry_price,
                stop_loss=stop_loss,
                min_qty=MIN_QUANTITY,
                qty_precision=QUANTITY_PRECISION,
                existing_positions=existing_positions,
            )

            # Validate order (may raise OrderValidationError)
            try:
                valid, reason = self.risk_manager.validate_order(
                    balance=balance,
                    quantity=size_info["quantity"],
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )

                if not valid:
                    logger.warning(f"Order validation failed: {reason}")
                    return None
            except (PositionSizeError, DailyLimitError) as e:
                logger.warning(f"Risk check failed: {e}")
                return None

            return {
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_type": position_type,
                "quantity": str(size_info["quantity"]),
                "size_info": size_info,
            }
        except Exception as e:
            logger.error(f"Error preparing entry order: {e}")
            return None

    def _log_entry_order_details(self, order_details: Dict) -> None:
        """
        Log entry order details in a formatted way.

        Args:
            order_details: Dictionary containing order details
        """
        position_type = order_details["position_type"]
        entry_price = order_details["entry_price"]
        stop_loss = order_details["stop_loss"]
        take_profit = order_details["take_profit"]
        quantity = order_details["quantity"]
        size_info = order_details["size_info"]

        logger.info("=" * 60)
        logger.info(f"🎯 {position_type} ENTRY SIGNAL")
        logger.info("=" * 60)
        logger.info(f"Entry Price: ${entry_price:,.2f}")
        logger.info(f"Stop Loss:   ${stop_loss:,.2f}")
        logger.info(f"Take Profit: ${take_profit:,.2f}")
        logger.info(f"Quantity:    {quantity}")
        logger.info(f"Notional:    ${size_info['notional_value']:,.2f}")
        logger.info(
            f"Risk:        ${size_info['position_risk']:,.2f} ({size_info['risk_pct']:.2f}%)"
        )
        logger.info("=" * 60)

    def _handle_dry_run_entry(self, order_details: Dict) -> bool:
        """
        Handle entry order in dry run mode (simulate without placing real order).

        Args:
            order_details: Dictionary containing order details

        Returns:
            True if simulated successfully
        """
        logger.info("🔶 DRY RUN MODE - Order not executed")

        self.current_position = {
            "type": order_details["position_type"],
            "entry_price": order_details["entry_price"],
            "quantity": float(order_details["quantity"]),
            "stop_loss": order_details["stop_loss"],
            "take_profit": order_details["take_profit"],
            "entry_time": datetime.now(timezone.utc),
        }

        # Initialize trailing stop for this position
        if self.trailing_stop_enabled and self.trailing_stop:
            self.trailing_stop.initialize_position(
                entry_price=order_details["entry_price"],
                initial_stop_loss=order_details["stop_loss"],
                position_type=order_details["position_type"],
            )

        self._save_position_state()
        return True

    def _build_order_info(self, order_details: Dict) -> Dict:
        """
        Build order information dictionary for exchange API.

        Args:
            order_details: Dictionary containing order details

        Returns:
            Dictionary formatted for exchange API
        """
        position_type = order_details["position_type"]
        side = "BUY" if position_type == "LONG" else "SELL"

        return {
            "symbol": self.symbol,
            "side": side,
            "trade_side": "OPEN",
            "order_type": "MARKET",
            "quantity": order_details["quantity"],
            "price": None,
            "take_profit": str(order_details["take_profit"]),
            "stop_loss": str(order_details["stop_loss"]),
            "entry_price": order_details["entry_price"],
            "position_type": position_type,
            "size_info": order_details["size_info"],
        }

    def place_entry_order(self, signal: Dict, balance: float) -> bool:
        """
        Place entry order based on signal.

        This method orchestrates the entry order placement process:
        1. Prepares order (calculates size, validates)
        2. Logs order details
        3. Handles dry run mode or places real order

        Args:
            signal: Entry signal from strategy
            balance: Current account balance

        Returns:
            True if order placed successfully
        """
        try:
            # Prepare order (calculate size, validate)
            order_details = self._prepare_entry_order(signal, balance)
            if order_details is None:
                return False

            # Log order details
            self._log_entry_order_details(order_details)

            # Handle dry run mode
            if self.dry_run:
                return self._handle_dry_run_entry(order_details)

            # Build order info for exchange API
            order_info = self._build_order_info(order_details)

            # Place order with retry logic
            success = self._place_order_with_retry(order_info)

            if success:
                # Log trade entry to audit log
                try:
                    self.audit_logger.log_trade_entry(
                        symbol=self.symbol,
                        position_type=order_details["position_type"],
                        entry_price=order_details["entry_price"],
                        quantity=float(order_details["quantity"]),
                        stop_loss=order_details["stop_loss"],
                        take_profit=order_details["take_profit"],
                        order_id=order_info.get("orderId"),
                        balance=balance,
                        leverage=self.risk_manager.leverage,
                        dry_run=self.dry_run,
                    )
                except Exception as e:
                    logger.error(f"Failed to write audit log entry: {e}")

                # Order placed successfully - position will be updated by status check
                return True
            else:
                logger.error("❌ Failed to place order after retries")
                return False

        except Exception as e:
            logger.error(f"Error placing entry order: {e}")
            self._record_error(f"Error placing entry order: {e}")
            return False

    def _place_order_with_retry(self, order_info: Dict) -> bool:
        """
        Place order with retry logic.

        Args:
            order_info: Order information dictionary

        Returns:
            True if order placed successfully, False otherwise
        """
        max_attempts = self.order_tracker.max_retries + 1
        tracking_id = None

        for attempt in range(max_attempts):
            try:
                # Place order
                response = self.client.place_order(
                    symbol=order_info["symbol"],
                    side=order_info["side"],
                    trade_side=order_info["trade_side"],
                    order_type=order_info["order_type"],
                    quantity=order_info["quantity"],
                    take_profit=order_info.get("take_profit"),
                    stop_loss=order_info.get("stop_loss"),
                )

                order_id = (
                    response.get("data", {}).get("orderId") if response.get("code") == 0 else None
                )

                # Track order
                if tracking_id is None:
                    tracking_id = self.order_tracker.track_order(order_id, order_info)
                else:
                    # Update existing tracking
                    if tracking_id in self.order_tracker.active_orders:
                        self.order_tracker.active_orders[tracking_id]["order_id"] = order_id
                        self.order_tracker.active_orders[tracking_id]["updated_at"] = datetime.now(
                            timezone.utc
                        )

                if response.get("code") == 0:
                    logger.info(f"✅ Order placed successfully (orderId: {order_id})")

                    # Update status to submitted
                    if tracking_id:
                        self.order_tracker.update_order_status(tracking_id, OrderStatus.SUBMITTED)

                    # For market orders, assume filled immediately (check will verify)
                    # Initialize position with expected values
                    if order_info["trade_side"] == "OPEN":
                        self.current_position = {
                            "type": order_info["position_type"],
                            "entry_price": order_info["entry_price"],
                            "quantity": float(order_info["quantity"]),
                            "stop_loss": float(order_info["stop_loss"]),
                            "take_profit": float(order_info["take_profit"]),
                            "entry_time": datetime.now(),
                            "order_id": order_id,
                            "tracking_id": tracking_id,
                        }

                        # Initialize trailing stop
                        if self.trailing_stop_enabled and self.trailing_stop:
                            self.trailing_stop.initialize_position(
                                entry_price=order_info["entry_price"],
                                initial_stop_loss=float(order_info["stop_loss"]),
                                position_type=order_info["position_type"],
                            )

                        self._save_position_state()

                    return True
                else:
                    error_msg = response.get("msg", "Unknown error")
                    logger.warning(
                        f"⚠️ Order placement failed "
                        f"(attempt {attempt + 1}/{max_attempts}): {error_msg}"
                    )

                    if tracking_id:
                        self.order_tracker.update_order_status(
                            tracking_id, OrderStatus.FAILED, error_message=error_msg
                        )

                    # Retry if not last attempt
                    if attempt < max_attempts - 1:
                        logger.info(f"🔄 Retrying order in {self.order_tracker.retry_delay}s...")
                        time.sleep(self.order_tracker.retry_delay)
                        continue
                    else:
                        logger.error(f"❌ Order failed after {max_attempts} attempts")
                        return False

            except Exception as e:
                logger.error(f"Error placing order (attempt {attempt + 1}/{max_attempts}): {e}")
                self._record_error(f"Order placement error: {e}")

                if tracking_id:
                    self.order_tracker.update_order_status(
                        tracking_id, OrderStatus.FAILED, error_message=str(e)
                    )

                # Retry if not last attempt
                if attempt < max_attempts - 1:
                    logger.info(f"🔄 Retrying order in {self.order_tracker.retry_delay}s...")
                    time.sleep(self.order_tracker.retry_delay)
                    continue
                else:
                    return False

        return False

    def _check_order_statuses(self) -> None:
        """
        Check status of active orders and handle partial fills.

        Called periodically to verify order status and update positions.
        """
        active_orders = self.order_tracker.get_active_orders(symbol=self.symbol)

        if not active_orders:
            return

        for order_record in active_orders:
            tracking_id = order_record["tracking_id"]

            try:
                # Check order status with exchange
                status = self.order_tracker.check_order_status(self.client, tracking_id)

                if status == OrderStatus.PARTIAL:
                    # Handle partial fill
                    self._handle_partial_fill(order_record)
                elif status == OrderStatus.FILLED:
                    # Order fully filled - verify position
                    logger.info(f"✅ Order {tracking_id} fully filled")
                    # Position should already be synced, but verify
                    self._sync_position_with_exchange()
                elif status == OrderStatus.FAILED:
                    # Order failed - check if should retry
                    if self.order_tracker.should_retry(tracking_id):
                        logger.info(f"🔄 Retrying failed order {tracking_id}")
                        retry_info = self.order_tracker.mark_for_retry(tracking_id)
                        # Retry order placement
                        self._place_order_with_retry(retry_info)

            except Exception as e:
                logger.error(f"Error checking order status for {tracking_id}: {e}")
                self._record_error(f"Order status check error: {e}")

    def _handle_partial_fill(self, order_record: Dict) -> None:
        """
        Handle partial order fill.

        Args:
            order_record: Order tracking record
        """
        tracking_id = order_record["tracking_id"]
        filled_qty = order_record.get("filled_quantity", 0)
        original_qty = float(order_record.get("quantity", 0))

        logger.info(
            f"📊 Partial fill detected: {filled_qty}/{original_qty} for order {tracking_id}"
        )

        # Update position quantity if this is an entry order
        if order_record.get("trade_side") == "OPEN" and self.current_position:
            # Adjust position quantity based on filled amount
            # Note: This is approximate - actual position sync will correct it
            fill_ratio = filled_qty / original_qty if original_qty > 0 else 0

            if fill_ratio > 0:
                logger.info(f"Adjusting position quantity by fill ratio: {fill_ratio:.2%}")
                # Position will be corrected by sync on next cycle

        # Sync position to get accurate fill status
        self._sync_position_with_exchange()

    def _check_dry_run_tp_sl(self, current_price: float) -> tuple:
        """
        Check if TP/SL levels are hit in dry run mode.

        Args:
            current_price: Current market price

        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        if not self.dry_run or not self.current_position:
            return False, ""

        # Use trailing stop loss if active, otherwise use original stop loss
        stop_loss_to_check = self.current_position["stop_loss"]
        if (
            self.trailing_stop_enabled
            and self.trailing_stop
            and self.trailing_stop.current_stop_loss
        ):
            stop_loss_to_check = self.trailing_stop.current_stop_loss

        position_type = self.current_position["type"]
        take_profit = self.current_position["take_profit"]

        if position_type == "LONG":
            if current_price >= take_profit:
                return True, "Take Profit Hit (Manual)"
            elif current_price <= stop_loss_to_check:
                return True, "Stop Loss Hit (Manual)"
        else:  # SHORT
            if current_price <= take_profit:
                return True, "Take Profit Hit (Manual)"
            elif current_price >= stop_loss_to_check:
                return True, "Stop Loss Hit (Manual)"

        return False, ""

    def check_exit_conditions(self, df: pd.DataFrame) -> tuple:
        """Check if position should be closed."""
        if not self.current_position:
            return False, ""

        # Validate DataFrame is not empty before accessing
        if df.empty:
            logger.error("Cannot check exit conditions: DataFrame is empty")
            return False, ""

        # Use WebSocket price if available (more real-time), otherwise use DataFrame
        if self.current_price is not None:
            current_price = self.current_price
        else:
            current_price = df.iloc[-1]["close"]

        # Check trailing stop first (highest priority if enabled)
        if self.trailing_stop_enabled and self.trailing_stop:
            stop_hit, reason = self.trailing_stop.check_stop_hit(current_price)
            if stop_hit:
                return True, reason

        # Check strategy exit signals
        should_exit, reason = self.strategy.check_exit_signal(df, self.current_position["type"])

        if should_exit:
            return True, reason

        # In dry run, manually check TP/SL from price (using current_position values)
        should_exit, reason = self._check_dry_run_tp_sl(current_price)
        if should_exit:
            return True, reason

        return False, ""

    def close_position(self, reason: str, current_price: Optional[float] = None) -> bool:
        """
        Close current position with input sanitization.

        Args:
            reason: Reason for closing position (sanitized)
            current_price: Current market price (optional, sanitized if provided)

        Returns:
            True if position closed successfully, False otherwise
        """
        if not self.current_position:
            return False

        # Sanitize reason string
        if not isinstance(reason, str):
            reason = str(reason)
        reason = reason.strip()[:200]  # Limit length and strip

        try:
            logger.info("=" * 60)
            logger.info(f"🚪 CLOSING {self.current_position['type']} POSITION")
            logger.info(f"Reason: {reason}")
            logger.info("=" * 60)

            # Sanitize current_price if provided
            if current_price is not None:
                current_price = InputSanitizer.sanitize_price(current_price, "current_price")

            # Calculate P&L
            entry_price = InputSanitizer.sanitize_price(
                self.current_position["entry_price"], "entry_price"
            )
            quantity = InputSanitizer.sanitize_quantity(
                self.current_position["quantity"], "quantity"
            )
            leverage = self.risk_manager.leverage

            if self.current_position["type"] == "LONG":
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity

            # Calculate P&L percentage based on margin used (not notional value)
            # Margin used = notional_value / leverage = (entry_price * quantity) / leverage
            notional_value = entry_price * quantity
            margin_used = notional_value / leverage if leverage > 0 else notional_value

            # P&L percentage relative to margin used (accounts for leverage)
            pnl_pct = (pnl / margin_used) * 100 if margin_used > 0 else 0.0

            # Also calculate price movement percentage for reference
            price_move_pct = (
                ((current_price - entry_price) / entry_price * 100)
                if self.current_position["type"] == "LONG"
                else ((entry_price - current_price) / entry_price * 100)
            )

            logger.info(f"Entry Price:     ${entry_price:,.2f}")
            logger.info(f"Exit Price:      ${current_price:,.2f}")
            logger.info(f"Quantity:        {quantity}")
            logger.info(f"Leverage:        {leverage}x")
            logger.info(f"Margin Used:     ${margin_used:,.2f}")
            logger.info(f"Price Move:      {price_move_pct:+.2f}%")
            logger.info(f"P&L:             ${pnl:,.2f} ({pnl_pct:+.2f}% on margin)")
            logger.info("=" * 60)

            # Update risk manager
            self.risk_manager.update_daily_pnl(pnl)

            # Log trade exit to audit log
            try:
                self.audit_logger.log_trade_exit(
                    symbol=self.symbol,
                    position_type=self.current_position["type"],
                    entry_price=entry_price,
                    exit_price=current_price,
                    quantity=quantity,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    reason=reason,
                    order_id=None,  # Will be updated when order is placed
                    leverage=leverage,
                    margin_used=margin_used,
                    dry_run=self.dry_run,
                )
            except Exception as e:
                logger.error(f"Failed to write audit log entry: {e}")

            if self.dry_run:
                logger.info("🔶 DRY RUN MODE - Position closed virtually")
                self.current_position = None

                # Reset trailing stop
                if self.trailing_stop_enabled and self.trailing_stop:
                    self.trailing_stop.reset()

                self._save_position_state()
                return True

            # Close position on exchange with retry logic
            # Note: close_position() may use different endpoints, so we track it separately
            max_attempts = self.order_tracker.max_retries + 1

            for attempt in range(max_attempts):
                try:
                    response = self.client.close_position(self.symbol, quantity=str(quantity))

                    order_id = (
                        response.get("data", {}).get("orderId")
                        if response.get("code") == 0
                        else None
                    )

                    # Track close order
                    order_info = {
                        "symbol": self.symbol,
                        "side": "SELL" if self.current_position["type"] == "LONG" else "BUY",
                        "trade_side": "CLOSE",
                        "order_type": "MARKET",
                        "quantity": str(quantity),
                        "price": None,
                        "close_reason": reason,
                    }

                    tracking_id = self.order_tracker.track_order(order_id, order_info)

                    if response.get("code") == 0:
                        logger.info(
                            f"✅ Position close order placed successfully (orderId: {order_id})"
                        )

                        # Log order status change to audit log
                        try:
                            self.audit_logger.log_order_status_change(
                                order_id=order_id or "unknown",
                                symbol=self.symbol,
                                status="SUBMITTED",
                                details={"tracking_id": tracking_id},
                            )
                        except Exception as e:
                            logger.error(f"Failed to write audit log entry: {e}")

                        # Update status
                        self.order_tracker.update_order_status(tracking_id, OrderStatus.SUBMITTED)

                        # For market close orders, assume filled immediately
                        self.order_tracker.update_order_status(tracking_id, OrderStatus.FILLED)

                        # Log order filled status to audit log
                        try:
                            self.audit_logger.log_order_status_change(
                                order_id=order_id or "unknown",
                                symbol=self.symbol,
                                status="FILLED",
                                previous_status="SUBMITTED",
                                details={"tracking_id": tracking_id},
                            )
                        except Exception as e:
                            logger.error(f"Failed to write audit log entry: {e}")

                        self.current_position = None

                        # Reset trailing stop
                        if self.trailing_stop_enabled and self.trailing_stop:
                            self.trailing_stop.reset()

                        self._save_position_state()
                        return True
                    else:
                        error_msg = response.get("msg", "Unknown error")
                        logger.warning(
                            f"⚠️ Close order failed "
                            f"(attempt {attempt + 1}/{max_attempts}): {error_msg}"
                        )

                        if tracking_id:
                            self.order_tracker.update_order_status(
                                tracking_id, OrderStatus.FAILED, error_message=error_msg
                            )

                        # Retry if not last attempt
                        if attempt < max_attempts - 1:
                            logger.info(
                                f"🔄 Retrying close order in {self.order_tracker.retry_delay}s..."
                            )
                            time.sleep(self.order_tracker.retry_delay)
                            continue
                        else:
                            logger.error(
                                f"❌ Failed to close position after {max_attempts} attempts"
                            )
                            return False

                except Exception as e:
                    logger.error(
                        f"Error closing position (attempt {attempt + 1}/{max_attempts}): {e}"
                    )
                    self._record_error(f"Close position error: {e}")

                    # Retry if not last attempt
                    if attempt < max_attempts - 1:
                        logger.info(
                            f"🔄 Retrying close order in {self.order_tracker.retry_delay}s..."
                        )
                        time.sleep(self.order_tracker.retry_delay)
                        continue
                    else:
                        return False

            return False

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False

    def setup_leverage(self):
        """Set leverage for the trading pair."""
        try:
            if self.dry_run:
                logger.info(f"🔶 DRY RUN - Would set leverage to {self.risk_manager.leverage}x")
                return

            response = self.client.set_leverage(
                symbol=self.symbol, leverage=self.risk_manager.leverage
            )

            if response.get("code") == 0:
                logger.info(f"✅ Leverage set to {self.risk_manager.leverage}x")
            else:
                logger.warning(f"Could not set leverage: {response}")

        except Exception as e:
            logger.error(f"Error setting leverage: {e}")

    def run_trading_cycle(self):
        """Execute one trading cycle."""
        try:
            # Get market data with validation
            try:
                df = self.get_market_data()
            except ValueError as e:
                logger.error(f"Cannot proceed without valid market data: {e}")
                logger.warning("Skipping this trading cycle - will retry next interval")
                return

            # Additional safety check (should not be needed after validation, but keep for safety)
            if df.empty:
                logger.error("Received empty DataFrame despite validation - this should not happen")
                return

            # Get current price (no indicators needed for this)
            current_price = df.iloc[-1]["close"]

            # Check order statuses and handle partial fills
            self._check_order_statuses()

            # Sync position with exchange (preserves trailing stop state)
            # This doesn't require indicators, so we do it first
            self._sync_position_with_exchange()

            # Only calculate indicators when we actually need them:
            # - If we have a position: need for exit signals
            # - If we don't have a position: need for entry signals
            # This avoids unnecessary CPU usage when just syncing or doing other operations
            indicators_calculated = False

            if self.current_position:
                # Calculate indicators only when needed (for exit signals)
                if not indicators_calculated:
                    df = self.strategy.calculate_indicators(df)
                    indicators_calculated = True

                    # Validate DataFrame is still not empty after indicator calculation
                    if df.empty:
                        logger.error(
                            "DataFrame became empty after indicator calculation - cannot proceed"
                        )
                        return

                # Get indicator values for logging
                indicators = self.strategy.get_indicator_values(df)
                logger.info(
                    f"💹 {self.symbol}: ${current_price:,.2f} | "
                    f"MACD: {indicators['macd']:.4f} | "
                    f"Signal: {indicators['signal']:.4f} | "
                    f"Hist: {indicators['histogram']:.4f}"
                )

                # Update trailing stop if enabled
                if self.trailing_stop_enabled and self.trailing_stop:
                    stop_updated, new_stop_loss, update_msg = self.trailing_stop.update(
                        current_price
                    )

                    # Save state if trailing stop was updated
                    if stop_updated:
                        self._save_position_state()

                    # If stop was updated and we're not in dry run, update on exchange
                    if stop_updated and not self.dry_run:
                        try:
                            self.client.update_stop_loss(
                                symbol=self.symbol,
                                new_stop_loss=str(new_stop_loss),
                                take_profit=str(self.current_position["take_profit"]),
                            )
                        except Exception as e:
                            logger.error(f"Failed to update stop-loss on exchange: {e}")

                    # Update position's stop_loss value
                    if stop_updated:
                        self.current_position["stop_loss"] = new_stop_loss

                # Check exit conditions (requires indicators)
                should_exit, exit_reason = self.check_exit_conditions(df)

                if should_exit:
                    self.close_position(exit_reason, current_price)
                else:
                    position_info = f"📊 Holding {self.current_position['type']} position"

                    # Add trailing stop status to log
                    if self.trailing_stop_enabled and self.trailing_stop:
                        ts_status = self.trailing_stop.get_status()
                        if ts_status["initialized"]:
                            position_info += (
                                f" | Trailing: "
                                f"{'ACTIVE' if ts_status['active'] else 'INACTIVE'}"
                            )
                            if ts_status["active"]:
                                position_info += f" (SL: ${ts_status['current_stop_loss']:.2f})"

                    logger.info(position_info)

            else:
                # Calculate indicators only when needed (for entry signals)
                if not indicators_calculated:
                    df = self.strategy.calculate_indicators(df)
                    indicators_calculated = True

                    # Validate DataFrame is still not empty after indicator calculation
                    if df.empty:
                        logger.error(
                            "DataFrame became empty after indicator calculation - cannot proceed"
                        )
                        return

                # Get indicator values for logging
                indicators = self.strategy.get_indicator_values(df)
                logger.info(
                    f"💹 {self.symbol}: ${current_price:,.2f} | "
                    f"MACD: {indicators['macd']:.4f} | "
                    f"Signal: {indicators['signal']:.4f} | "
                    f"Hist: {indicators['histogram']:.4f}"
                )

                # Look for entry signals (requires indicators)
                signal = self.strategy.check_entry_signal(df)

                if signal:
                    # Multi-timeframe analysis: Check higher timeframe trend
                    if self.multi_timeframe_enabled:
                        df_higher_tf = self.get_higher_timeframe_data()

                        if df_higher_tf is not None:
                            is_aligned, reason = self.strategy.check_higher_timeframe_trend(
                                df_higher_tf, signal["type"]
                            )

                            if not is_aligned:
                                logger.info(
                                    f"⏸️ Entry signal rejected by multi-timeframe "
                                    f"analysis: {reason}"
                                )
                                logger.info(
                                    f"   Signal: {signal['type']} @ ${signal['entry_price']:.2f}"
                                )
                                logger.info(f"   Higher TF ({self.higher_timeframe}): {reason}")
                                # Skip entry - higher timeframe doesn't support it
                                signal = None
                            else:
                                logger.info(
                                    f"✅ Multi-timeframe analysis confirms entry signal: {reason}"
                                )
                                logger.info(
                                    f"   Signal: {signal['type']} @ ${signal['entry_price']:.2f}"
                                )
                                logger.info(f"   Higher TF ({self.higher_timeframe}): {reason}")
                        else:
                            logger.warning(
                                "⚠️ Multi-timeframe enabled but higher timeframe "
                                "data unavailable - proceeding without confirmation"
                            )
                            # Continue with signal if higher TF data unavailable
                            # (don't block trading)

                if signal:
                    # Get balance
                    try:
                        balance = self.get_account_balance()
                    except ValueError as e:
                        logger.error(f"Cannot proceed without account balance: {e}")
                        logger.warning("Skipping entry signal - will retry next cycle")
                        return

                    if balance > 0:
                        # Check risk limits
                        allowed, reason = self.risk_manager.check_risk_limits(balance)

                        if allowed:
                            self.place_entry_order(signal, balance)
                        else:
                            logger.warning(f"⚠️ Trading not allowed: {reason}")
                else:
                    logger.info("👀 No entry signals")

            # Print risk summary
            try:
                balance = self.get_account_balance()
                risk_summary = self.risk_manager.get_risk_summary(balance)
                max_trades = self.config['risk']['max_trades_per_day']
                logger.info(
                    f"📊 Daily P&L: ${risk_summary['daily_pnl']:,.2f} "
                    f"({risk_summary['daily_loss_pct']:+.2f}%) | "
                    f"Trades: {risk_summary['daily_trades']}/{max_trades}"
                )
            except ValueError as e:
                logger.warning(f"Cannot get risk summary - balance unavailable: {e}")

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)
            # Track error for health monitoring
            self._record_error(str(e), exc_info=True)
        finally:
            # Update cycle tracking
            self.last_cycle_time = datetime.now(timezone.utc)
            self.cycle_count += 1

    def _setup_websocket(self):
        """Initialize WebSocket subscriptions."""
        if not self.websocket_enabled or not self.ws_client:
            return

        try:
            # Get coin symbol (e.g., "BTC" from "BTCUSDT" or "SOL" from "SOL-USDC")
            coin = self.symbol.replace("USDT", "").replace("USDC", "").replace("-USDC", "").replace("-USDT", "").replace("-USD", "")

            # Start WebSocket client
            self.ws_client.start()

            # Wait a moment for connection
            time.sleep(1)

            # Subscribe to trades for real-time price updates
            if self.ws_client.is_connected():
                # Use asyncio to subscribe
                import asyncio

                async def subscribe():
                    await self.ws_client.subscribe_trades(
                        coin=coin, callback=self._on_trade_update
                    )
                    logger.info(f"✅ Subscribed to WebSocket trades for {coin}")

                # Run subscription in WebSocket's event loop
                if self.ws_client.ws_loop and self.ws_client.ws_loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        subscribe(), self.ws_client.ws_loop
                    )
                else:
                    logger.warning("WebSocket event loop not running, subscription skipped")

            logger.info("WebSocket setup complete")
        except Exception as e:
            logger.error(f"Error setting up WebSocket: {e}")
            logger.warning("Continuing with REST API polling only")
            self.websocket_enabled = False

    def _on_trade_update(self, trade_data: Dict):
        """
        Handle trade updates from WebSocket.

        Args:
            trade_data: Trade data from WebSocket
        """
        try:
            # Extract price from trade data
            price_str = trade_data.get("data", {}).get("px") or trade_data.get("px")
            if not price_str:
                return

            price = float(price_str)
            coin = (
                trade_data.get("data", {}).get("coin")
                or trade_data.get("coin", "")
            )

            # Update current price for stop-loss/take-profit checks
            expected_coin = self.symbol.replace("USDT", "").replace("USDC", "").replace("-USDC", "").replace("-USDT", "").replace("-USD", "")
            if coin == expected_coin:
                self.current_price = price
                logger.debug(f"📡 Real-time price update: ${price:.2f}")

                # Check if we need to update trailing stop or exit conditions
                if self.current_position:
                    self._check_exit_conditions_realtime(price)
        except Exception as e:
            logger.error(f"Error processing trade update: {e}")

    def _check_exit_conditions_realtime(self, current_price: float):
        """
        Check exit conditions with real-time price from WebSocket.

        Args:
            current_price: Current market price from WebSocket
        """
        if not self.current_position:
            return

        try:
            # Quick check for stop-loss/take-profit hits
            stop_loss = self.current_position.get("stop_loss")
            take_profit = self.current_position.get("take_profit")

            if stop_loss and current_price <= stop_loss:
                logger.warning(f"🛑 Stop-loss hit at ${current_price:.2f} (WebSocket)")
                self.close_position("Stop Loss Hit (WebSocket)", current_price)
            elif take_profit and current_price >= take_profit:
                logger.info(f"🎯 Take-profit hit at ${current_price:.2f} (WebSocket)")
                self.close_position("Take Profit Hit (WebSocket)", current_price)
            elif self.trailing_stop_enabled and self.trailing_stop:
                # Update trailing stop with real-time price
                self.trailing_stop.update_price(current_price)
        except Exception as e:
            logger.error(f"Error checking exit conditions (real-time): {e}")

    def run(self):
        """
        Execute the main trading bot loop with comprehensive error handling and monitoring.

        This method implements the core trading logic that runs continuously:
        1. **Initialization**: Sets up leverage, WebSocket connections, and daily stats
        2. **Trading Loop**: Continuously monitors market conditions and executes trades
        3. **Daily Reset**: Automatically resets daily statistics at UTC midnight
        4. **Error Handling**: Comprehensive exception handling with graceful recovery
        5. **Monitoring**: Logs system status, balance updates, and trading activity

        The bot operates on a configurable check interval, analyzing market data,
        generating trading signals, and executing orders while maintaining strict
        risk management controls.

        Key Behaviors:
        - **Continuous Operation**: Runs indefinitely until interrupted (Ctrl+C)
        - **Daily Statistics**: Resets P&L tracking at UTC midnight
        - **Error Recovery**: Handles temporary API issues and continues operation
        - **Position Monitoring**: Tracks open positions and manages risk
        - **Real-time Updates**: Processes WebSocket data when available
        - **Audit Logging**: Records all trading decisions and executions

        The loop performs these steps in each cycle:
        1. Process any pending WebSocket messages
        2. Check for daily reset (UTC midnight)
        3. Fetch current account balance
        4. Retrieve market data (OHLCV candles)
        5. Calculate technical indicators
        6. Generate trading signals
        7. Execute trades with risk management
        8. Update position monitoring
        9. Log system status and metrics

        Raises:
            KeyboardInterrupt: When user stops the bot (Ctrl+C)
            SystemExit: On critical errors requiring shutdown
            Exception: For unexpected errors (logged and handled)

        Example:
            >>> bot = TradingBot()
            >>> try:
            ...     bot.run()
            ... except KeyboardInterrupt:
            ...     print("Bot stopped by user")
        """
        logger.info("🚀 Starting trading bot...")

        # Setup leverage
        self.setup_leverage()

        # Setup WebSocket if enabled
        if self.websocket_enabled:
            self._setup_websocket()

        # Initialize daily stats - require balance at startup
        try:
            initial_balance = self.get_account_balance()
            self.risk_manager.reset_daily_stats(initial_balance)
        except ValueError as e:
            logger.error(f"❌ Cannot start bot without account balance: {e}")
            logger.error("Bot startup failed. Please check your API connection and credentials.")
            raise

        try:
            while True:
                # Process WebSocket messages if enabled
                if self.websocket_enabled and self.ws_client:
                    self.ws_client.process_messages()

                # Check if new day (reset daily stats)
                # Use UTC for consistent daily reset timing across all timezones
                current_date = datetime.now(timezone.utc).date()
                if current_date > self.last_daily_reset:
                    logger.info("\n" + "=" * 60)
                    logger.info("🌅 NEW TRADING DAY (UTC)")
                    logger.info("=" * 60)
                    try:
                        daily_balance = self.get_account_balance()
                        self.risk_manager.reset_daily_stats(daily_balance)
                    except ValueError as e:
                        logger.error(f"Cannot reset daily stats - balance unavailable: {e}")
                        logger.warning(
                            "Continuing with previous day's stats - will retry next cycle"
                        )
                    self.last_daily_reset = current_date

                # Run trading cycle
                logger.info("\n" + "-" * 60)
                # Log both UTC and local time for clarity
                utc_now = datetime.now(timezone.utc)
                logger.info(
                    f"⏰ UTC: {utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')} | "
                    f"Local: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                logger.info("-" * 60)

                self.run_trading_cycle()

                # Wait before next check
                logger.info(f"⏳ Sleeping for {self.check_interval} seconds...\n")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("\n🛑 Bot stopped by user")

            # Cleanup WebSocket connection
            if self.websocket_enabled and self.ws_client:
                logger.info("Closing WebSocket connection...")
                try:
                    self.ws_client.stop()
                    logger.info("WebSocket connection closed")
                except Exception as e:
                    logger.error(f"Error closing WebSocket: {e}")

            # Close any open positions
            if self.current_position:
                logger.info("Closing open position...")
                try:
                    # Use WebSocket price if available, otherwise fetch from REST
                    current_price = None
                    if self.current_price:
                        current_price = self.current_price
                        logger.info(f"Using WebSocket price: ${current_price:.2f}")
                    else:
                        df = self.get_market_data()
                        if not df.empty:
                            current_price = df.iloc[-1]["close"]
                    if current_price:
                        self.close_position("Bot Shutdown", current_price)
                    else:
                        logger.warning(
                            "Cannot get market data for position close - position may remain open"
                        )
                except (
                    MarketDataError,
                    MarketDataUnavailableError,
                    MarketDataValidationError,
                ) as e:
                    logger.error(f"Cannot get market data for position close: {e}")
                    logger.warning("Position may remain open - please check exchange manually")
                except ValueError as e:
                    logger.error(f"Validation error during position close: {e}")
                    logger.warning("Position may remain open - please check exchange manually")

        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)

        finally:
            logger.info("👋 Bot shutdown complete")


if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Start bot
    bot = TradingBot("config/config.json")
    bot.run()
