"""
Trading Bot Constants

Centralized constants to replace magic numbers throughout the codebase.
"""

# Cache Configuration
CACHE_MAX_AGE_SECONDS = 60  # Market data cache validity duration
HIGHER_TF_CACHE_MAX_AGE_SECONDS = 300  # Higher timeframe cache (5 minutes)
METADATA_CACHE_TTL = 3600  # Asset metadata cache TTL (1 hour)
CLEARINGHOUSE_CACHE_TTL = 5  # Clearinghouse state cache TTL (5 seconds)

# Market Data Fetching
DEFAULT_CANDLES_TO_FETCH = 200  # Full dataset fetch
MIN_INCREMENTAL_CANDLES = 5  # Minimum candles for incremental fetch
MAX_INCREMENTAL_CANDLES = 50  # Maximum candles for incremental fetch
MIN_CANDLES_FOR_MACD = 50  # Minimum candles needed for MACD calculation
HIGHER_TF_CANDLES_NEEDED = 50  # Candles needed for higher timeframe analysis

# Position and Order Limits
MIN_QUANTITY = 0.001  # Minimum order quantity
QUANTITY_PRECISION = 3  # Decimal precision for quantities
MIN_ENTRY_PRICE_DIFF = 0.01  # Minimum entry price difference to update (1 cent)
POSITION_QTY_DIFF_THRESHOLD_PCT = 5.0  # Position quantity difference threshold (5%)

# Default Values
DEFAULT_BALANCE_DRY_RUN = 1000.0  # Default balance for dry run mode
DEFAULT_CHECK_INTERVAL = 60  # Default check interval in seconds
DEFAULT_TIMEOUT = 10  # Default HTTP request timeout in seconds

# Connection Pooling
POOL_CONNECTIONS = 10  # Number of connection pools to cache
POOL_MAXSIZE = 20  # Maximum connections per pool

# Risk Management Defaults
DEFAULT_LEVERAGE = 10
DEFAULT_MAX_POSITION_SIZE_PCT = 0.1  # 10%
DEFAULT_MAX_DAILY_LOSS_PCT = 0.05  # 5%
DEFAULT_MAX_TRADES_PER_DAY = 10

# Trailing Stop Defaults
DEFAULT_TRAIL_PERCENT = 2.0  # 2%
DEFAULT_ACTIVATION_PERCENT = 1.0  # 1%
DEFAULT_UPDATE_THRESHOLD_PERCENT = 0.5  # 0.5%

# Stop Loss Limits
MIN_STOP_DISTANCE_PCT = 0.001  # 0.1% minimum stop distance
MAX_STOP_DISTANCE_PCT = 0.10  # 10% maximum stop distance

# MACD Strategy
MACD_MIN_CANDLES_BUFFER = 10  # Buffer added to slow_length/signal_length for min_candles

# Backtesting Defaults
BACKTEST_INITIAL_BALANCE = 10000.0
BACKTEST_MAKER_FEE = 0.0002  # 0.02%
BACKTEST_TAKER_FEE = 0.0004  # 0.04%
BACKTEST_SLIPPAGE_PCT = 0.001  # 0.1%

# Order Tracking
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0  # seconds
DEFAULT_STATUS_CHECK_INTERVAL = 5.0  # seconds
ORDER_HISTORY_MAX_SIZE = 100  # Maximum orders to keep in history

# Health Monitor
DEFAULT_HEALTH_PORT = 8080
DEFAULT_HEALTH_HOST = '127.0.0.1'
RECENT_ERRORS_MAX = 10  # Maximum recent errors to track

# API Key Validation
MIN_API_KEY_LENGTH = 10  # Minimum API key length for validation

# Percentage Conversion
PERCENT_TO_DECIMAL = 100.0  # Multiply by this to convert percentage to decimal

