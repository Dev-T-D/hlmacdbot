# API Reference

This comprehensive API reference documents all classes, methods, and functions in the Hyperliquid MACD trading bot. Each component includes detailed parameter descriptions, return values, exceptions, and usage examples.

## 📚 Table of Contents

- [Core Classes](#core-classes)
  - [TradingBot](#tradingbot)
  - [EnhancedMACDStrategy](#enhancedmacdstrategy)
  - [HyperliquidClient](#hyperliquidclient)
  - [ResilienceManager](#resiliencemanager)
- [Infrastructure Classes](#infrastructure-classes)
  - [MetricsManager](#metricsmanager)
  - [AlertManager](#alertmanager)
  - [StructuredLogger](#structuredlogger)
- [Security Classes](#security-classes)
  - [SecureKeyStorage](#securekeystorage)
  - [AuditLogger](#auditlogger)
- [Utility Classes](#utility-classes)
  - [InputSanitizer](#inputsanitizer)
  - [RiskManager](#riskmanager)
- [Data Types](#data-types)
- [Error Handling](#error-handling)

## 🏗️ Core Classes

### TradingBot

The main orchestration engine that coordinates trading activities, risk management, and system monitoring.

#### Constructor

```python
TradingBot(config: dict, strategy: EnhancedMACDStrategy, client: HyperliquidClient)
```

**Parameters:**
- `config` (dict): Configuration dictionary containing trading parameters
- `strategy` (EnhancedMACDStrategy): Trading strategy instance
- `client` (HyperliquidClient): Exchange API client instance

**Example:**
```python
config = {
    'trading': {'symbol': 'BTCUSDT', 'dry_run': True},
    'risk': {'max_position_size_pct': 0.05}
}
strategy = EnhancedMACDStrategy(config)
client = HyperliquidClient(config)
bot = TradingBot(config, strategy, client)
```

#### Methods

##### `run() -> None`
Main trading loop that executes the trading strategy.

**Returns:** None

**Raises:**
- `KeyboardInterrupt`: When stopped by user
- `SystemExit`: On critical errors
- `Exception`: For unexpected errors

**Example:**
```python
try:
    bot.run()
except KeyboardInterrupt:
    logger.info("Bot stopped by user")
```

##### `check_trading_signals() -> Optional[TradeSignal]`
Analyzes market conditions and generates trading signals.

**Returns:**
- `TradeSignal`: Trading signal with entry/exit information
- `None`: No valid signal found

**Raises:**
- `StrategyError`: Strategy calculation failures
- `MarketDataError`: Missing or invalid market data

**Example:**
```python
signal = bot.check_trading_signals()
if signal:
    print(f"Signal: {signal.direction} at {signal.entry_price}")
    bot.execute_signal(signal)
```

##### `execute_signal(signal: TradeSignal) -> bool`
Executes a trading signal by placing orders.

**Parameters:**
- `signal` (TradeSignal): Trading signal to execute

**Returns:**
- `bool`: True if order executed successfully

**Raises:**
- `OrderError`: Order placement failures
- `RiskLimitError`: Risk management violations

**Example:**
```python
signal = TradeSignal(
    direction='LONG',
    entry_price=45000.0,
    stop_loss=44000.0,
    take_profit=47000.0,
    quantity=0.001
)
success = bot.execute_signal(signal)
```

##### `manage_positions() -> None`
Manages open positions including stop-loss updates and exits.

**Returns:** None

**Raises:**
- `PositionError`: Position management failures
- `ExchangeError`: API communication errors

**Example:**
```python
bot.manage_positions()  # Called in main loop
```

##### `get_account_status() -> AccountStatus`
Retrieves current account balance and position information.

**Returns:**
- `AccountStatus`: Account status with balance and positions

**Raises:**
- `ExchangeError`: API communication failures
- `AuthenticationError`: Invalid credentials

**Example:**
```python
status = bot.get_account_status()
print(f"Balance: ${status.balance:.2f}")
print(f"Positions: {len(status.positions)}")
```

### EnhancedMACDStrategy

Advanced trading strategy with multi-timeframe analysis, volume confirmation, and adaptive parameters.

#### Constructor

```python
EnhancedMACDStrategy(config: dict)
```

**Parameters:**
- `config` (dict): Strategy configuration parameters

**Example:**
```python
config = {
    'strategy': {
        'fast_length': 12,
        'slow_length': 26,
        'signal_length': 9,
        'use_multi_timeframe': True
    }
}
strategy = EnhancedMACDStrategy(config)
```

#### Methods

##### `calculate_indicators(df: pd.DataFrame) -> pd.DataFrame`
Calculates all technical indicators used by the strategy.

**Parameters:**
- `df` (pd.DataFrame): OHLCV data with required columns

**Returns:**
- `pd.DataFrame`: DataFrame with calculated indicators

**Raises:**
- `IndicatorError`: Calculation failures
- `DataError`: Invalid input data

**Example:**
```python
df = client.get_klines('BTCUSDT', '5m', 100)
df_with_indicators = strategy.calculate_indicators(df)
print(df_with_indicators.columns.tolist())
```

##### `check_entry_signal(df: pd.DataFrame, position_type: str) -> bool`
Evaluates entry conditions for a specific position type.

**Parameters:**
- `df` (pd.DataFrame): Market data with indicators
- `position_type` (str): 'LONG' or 'SHORT'

**Returns:**
- `bool`: True if entry conditions met

**Raises:**
- `StrategyError`: Signal evaluation failures

**Example:**
```python
df = client.get_klines('BTCUSDT', '5m', 100)
df = strategy.calculate_indicators(df)

if strategy.check_entry_signal(df, 'LONG'):
    print("Long entry signal detected")
```

##### `check_exit_signal(df: pd.DataFrame, position: Position) -> Optional[str]`
Evaluates exit conditions for an open position.

**Parameters:**
- `df` (pd.DataFrame): Current market data
- `position` (Position): Current position information

**Returns:**
- `str`: Exit reason ('stop_loss', 'take_profit', 'signal_exit')
- `None`: No exit signal

**Example:**
```python
position = Position(
    symbol='BTCUSDT',
    side='LONG',
    quantity=0.001,
    entry_price=45000.0
)

exit_reason = strategy.check_exit_signal(df, position)
if exit_reason:
    print(f"Exit signal: {exit_reason}")
```

##### `calculate_position_size(balance: float, risk_pct: float, entry_price: float, stop_loss: float) -> float`
Calculates position size based on risk management parameters.

**Parameters:**
- `balance` (float): Account balance
- `risk_pct` (float): Risk percentage per trade
- `entry_price` (float): Entry price
- `stop_loss` (float): Stop loss price

**Returns:**
- `float`: Position size in base currency

**Example:**
```python
balance = 10000.0
risk_pct = 0.02  # 2% risk
entry_price = 45000.0
stop_loss = 44000.0  # 2% stop loss

size = strategy.calculate_position_size(balance, risk_pct, entry_price, stop_loss)
print(f"Position size: {size} BTC")
```

##### `get_market_condition() -> str`
Analyzes current market regime using ADX and other indicators.

**Returns:**
- `str`: Market condition ('TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'HIGH_VOLATILITY')

**Example:**
```python
regime = strategy.get_market_condition()
print(f"Current regime: {regime}")

if regime == 'TRENDING_UP':
    # Adjust strategy for uptrend
    strategy.adjust_for_trend('bullish')
```

### HyperliquidClient

Exchange API client with comprehensive error handling and rate limiting.

#### Constructor

```python
HyperliquidClient(config: dict)
```

**Parameters:**
- `config` (dict): Client configuration with API credentials

**Example:**
```python
config = {
    'exchange': 'hyperliquid',
    'private_key': '0x...',
    'wallet_address': '0x...',
    'testnet': True
}
client = HyperliquidClient(config)
```

#### Methods

##### `get_ticker(symbol: str) -> dict`
Retrieves current ticker information for a symbol.

**Parameters:**
- `symbol` (str): Trading symbol (e.g., 'BTCUSDT')

**Returns:**
- `dict`: Ticker data with price, volume, timestamp

**Raises:**
- `ExchangeError`: API communication failures
- `RateLimitError`: Rate limit exceeded

**Example:**
```python
ticker = client.get_ticker('BTCUSDT')
print(f"BTC Price: ${ticker['price']:.2f}")
print(f"24h Volume: {ticker['volume']:.2f}")
```

##### `get_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame`
Retrieves historical candlestick data.

**Parameters:**
- `symbol` (str): Trading symbol
- `interval` (str): Timeframe ('1m', '5m', '1h', '1d')
- `limit` (int): Number of candles (default: 500, max: 1000)

**Returns:**
- `pd.DataFrame`: OHLCV data with timestamp index

**Raises:**
- `ExchangeError`: API failures
- `DataError`: Invalid parameters

**Example:**
```python
df = client.get_klines('BTCUSDT', '5m', 100)
print(f"Retrieved {len(df)} candles")
print(f"Latest close: ${df['close'].iloc[-1]:.2f}")
```

##### `place_order(order: OrderRequest) -> OrderResponse`
Places a new order on the exchange.

**Parameters:**
- `order` (OrderRequest): Order parameters

**Returns:**
- `OrderResponse`: Order execution result

**Raises:**
- `OrderError`: Order placement failures
- `InsufficientFundsError`: Not enough balance
- `InvalidOrderError`: Invalid order parameters

**Example:**
```python
order = OrderRequest(
    symbol='BTCUSDT',
    side='BUY',
    order_type='LIMIT',
    quantity=0.001,
    price=45000.0
)

response = client.place_order(order)
print(f"Order ID: {response.order_id}")
print(f"Status: {response.status}")
```

##### `cancel_order(order_id: str) -> bool`
Cancels an open order.

**Parameters:**
- `order_id` (str): Order ID to cancel

**Returns:**
- `bool`: True if cancelled successfully

**Raises:**
- `OrderError`: Cancellation failures
- `OrderNotFoundError`: Order doesn't exist

**Example:**
```python
success = client.cancel_order('123456789')
if success:
    print("Order cancelled successfully")
```

##### `get_positions() -> List[Position]`
Retrieves current open positions.

**Returns:**
- `List[Position]`: List of open positions

**Raises:**
- `ExchangeError`: API failures
- `AuthenticationError`: Invalid credentials

**Example:**
```python
positions = client.get_positions()
for pos in positions:
    print(f"{pos.symbol}: {pos.side} {pos.quantity} @ ${pos.entry_price}")
```

##### `get_account_balance() -> AccountBalance`
Retrieves account balance information.

**Returns:**
- `AccountBalance`: Balance information

**Raises:**
- `ExchangeError`: API failures
- `AuthenticationError`: Invalid credentials

**Example:**
```python
balance = client.get_account_balance()
print(f"Total Balance: ${balance.total:.2f}")
print(f"Available: ${balance.available:.2f}")
print(f"Used: ${balance.used:.2f}")
```

### ResilienceManager

System reliability and automatic recovery orchestration.

#### Constructor

```python
ResilienceManager(config: dict)
```

**Parameters:**
- `config` (dict): Resilience configuration

**Example:**
```python
config = {
    'resilience': {
        'circuit_breaker_failure_threshold': 5,
        'max_retries_per_hour': 100,
        'state_db_path': 'data/bot_state.db'
    }
}
resilience = ResilienceManager(config)
```

#### Methods

##### `execute_with_resilience(func: Callable, *args, **kwargs) -> Any`
Executes a function with automatic retry and circuit breaker protection.

**Parameters:**
- `func` (Callable): Function to execute
- `args`: Positional arguments for function
- `kwargs`: Keyword arguments for function

**Returns:**
- `Any`: Function result

**Raises:**
- `CircuitBreakerOpenError`: Circuit breaker is open
- `MaxRetriesExceededError`: All retries exhausted

**Example:**
```python
def risky_api_call():
    return client.get_ticker('BTCUSDT')

try:
    ticker = resilience.execute_with_resilience(risky_api_call)
    print(f"Price: ${ticker['price']}")
except CircuitBreakerOpenError:
    print("Circuit breaker is open - backing off")
```

##### `validate_api_response(response: dict, expected_keys: List[str]) -> bool`
Validates API response structure and data integrity.

**Parameters:**
- `response` (dict): API response to validate
- `expected_keys` (List[str]): Required keys in response

**Returns:**
- `bool`: True if validation passes

**Raises:**
- `ValidationError`: Validation failures

**Example:**
```python
response = client.get_ticker('BTCUSDT')
expected_keys = ['price', 'volume', 'timestamp']

if resilience.validate_api_response(response, expected_keys):
    print("Response is valid")
```

##### `save_state(state_data: dict) -> None`
Saves bot state to persistent storage with atomic transactions.

**Parameters:**
- `state_data` (dict): State data to save

**Returns:** None

**Raises:**
- `StateSaveError`: Save operation failures

**Example:**
```python
state = {
    'positions': bot.get_positions(),
    'last_signal_time': datetime.now().isoformat(),
    'account_balance': client.get_account_balance()
}
resilience.save_state(state)
```

##### `recover_state() -> dict`
Recovers bot state from persistent storage.

**Returns:**
- `dict`: Recovered state data

**Raises:**
- `StateRecoveryError`: Recovery failures

**Example:**
```python
try:
    state = resilience.recover_state()
    bot.restore_from_state(state)
    print("State recovered successfully")
except StateRecoveryError as e:
    print(f"State recovery failed: {e}")
```

## 🏭 Infrastructure Classes

### MetricsManager

Prometheus-compatible metrics collection and exposition.

#### Constructor

```python
MetricsManager(config: dict)
```

**Parameters:**
- `config` (dict): Metrics configuration

**Example:**
```python
config = {
    'metrics': {
        'port': 8000,
        'histogram_buckets': [0.1, 0.5, 1.0, 2.0, 5.0]
    }
}
metrics = MetricsManager(config)
```

#### Methods

##### `record_trade(symbol: str, side: str, pnl: float, duration: float) -> None`
Records a completed trade for metrics.

**Parameters:**
- `symbol` (str): Trading symbol
- `side` (str): Trade direction ('BUY', 'SELL')
- `pnl` (float): Profit/loss amount
- `duration` (float): Trade duration in seconds

**Returns:** None

**Example:**
```python
metrics.record_trade('BTCUSDT', 'BUY', 25.50, 7200)  # $25.50 profit, 2 hours
```

##### `record_api_latency(endpoint: str, latency: float, success: bool) -> None`
Records API call latency and success rate.

**Parameters:**
- `endpoint` (str): API endpoint name
- `latency` (float): Response time in seconds
- `success` (bool): Whether call was successful

**Returns:** None

**Example:**
```python
start_time = time.time()
try:
    ticker = client.get_ticker('BTCUSDT')
    latency = time.time() - start_time
    metrics.record_api_latency('get_ticker', latency, True)
except Exception:
    latency = time.time() - start_time
    metrics.record_api_latency('get_ticker', latency, False)
```

##### `record_system_metrics() -> None`
Records system resource usage metrics.

**Returns:** None

**Example:**
```python
# Called periodically (every 30 seconds)
metrics.record_system_metrics()
```

##### `start_http_server(port: int) -> None`
Starts Prometheus metrics HTTP server.

**Parameters:**
- `port` (int): Port to bind server to

**Returns:** None

**Example:**
```python
metrics.start_http_server(8000)
print("Metrics server running on http://localhost:8000")
```

### AlertManager

Multi-channel notification system for critical events.

#### Constructor

```python
AlertManager(config: dict)
```

**Parameters:**
- `config` (dict): Alert configuration with channel settings

**Example:**
```python
config = {
    'alerting': {
        'channels': {
            'telegram': {
                'enabled': True,
                'bot_token': '123456789:ABC...',
                'chat_ids': ['987654321']
            }
        }
    }
}
alerts = AlertManager(config)
```

#### Methods

##### `send_alert(message: str, severity: str = 'INFO', context: dict = None) -> bool`
Sends an alert through configured channels.

**Parameters:**
- `message` (str): Alert message
- `severity` (str): Alert severity ('CRITICAL', 'WARNING', 'INFO')
- `context` (dict): Additional context data

**Returns:**
- `bool`: True if sent successfully to at least one channel

**Raises:**
- `AlertError`: Sending failures

**Example:**
```python
context = {
    'symbol': 'BTCUSDT',
    'pnl': -150.00,
    'position_size': 0.002
}

success = alerts.send_alert(
    "Stop loss triggered on BTC position",
    severity='CRITICAL',
    context=context
)
```

##### `send_telegram(message: str, chat_id: str = None) -> bool`
Sends message via Telegram bot.

**Parameters:**
- `message` (str): Message to send
- `chat_id` (str): Specific chat ID (uses default if None)

**Returns:**
- `bool`: True if sent successfully

**Example:**
```python
alerts.send_telegram("Bot started successfully")
```

##### `send_email(subject: str, body: str, recipients: List[str] = None) -> bool`
Sends email notification.

**Parameters:**
- `subject` (str): Email subject
- `body` (str): Email body (HTML supported)
- `recipients` (List[str]): Email addresses (uses default if None)

**Returns:**
- `bool`: True if sent successfully

**Example:**
```python
alerts.send_email(
    subject="Daily Trading Summary",
    body="<h1>Daily P&L: +$245.67</h1><p>Win rate: 62%</p>"
)
```

### StructuredLogger

JSON-formatted logging with correlation IDs and context.

#### Constructor

```python
StructuredLogger(name: str, config: dict = None)
```

**Parameters:**
- `name` (str): Logger name
- `config` (dict): Logger configuration

**Example:**
```python
config = {
    'logging': {
        'level': 'INFO',
        'correlation_id': True,
        'json_format': True
    }
}
logger = StructuredLogger('trading_bot', config)
```

#### Methods

##### `info(message: str, **context) -> None`
Logs info-level message with context.

**Parameters:**
- `message` (str): Log message
- `context`: Additional context key-value pairs

**Returns:** None

**Example:**
```python
logger.info("Trade executed",
    symbol='BTCUSDT',
    side='BUY',
    quantity=0.001,
    price=45000.00,
    pnl=0.0
)
```

##### `error(message: str, exc: Exception = None, **context) -> None`
Logs error-level message with exception details.

**Parameters:**
- `message` (str): Error message
- `exc` (Exception): Exception object for automatic formatting
- `context`: Additional context

**Returns:** None

**Example:**
```python
try:
    client.place_order(order)
except Exception as e:
    logger.error("Order placement failed",
        order_id=order.id,
        symbol=order.symbol,
        exc=e
    )
```

##### `set_correlation_id(correlation_id: str) -> None`
Sets correlation ID for request tracing.

**Parameters:**
- `correlation_id` (str): Unique correlation identifier

**Returns:** None

**Example:**
```python
correlation_id = str(uuid.uuid4())
logger.set_correlation_id(correlation_id)

# All subsequent logs will include this correlation ID
logger.info("Processing trade signal")
```

## 🔐 Security Classes

### SecureKeyStorage

Encrypted private key storage with secure memory management.

#### Constructor

```python
SecureKeyStorage(encryption_key: bytes = None)
```

**Parameters:**
- `encryption_key` (bytes): AES encryption key (generated if None)

**Example:**
```python
key_storage = SecureKeyStorage()
key_storage.store_private_key('0xabcdef123456...')
```

#### Methods

##### `store_private_key(private_key: str) -> None`
Securely stores an encrypted private key.

**Parameters:**
- `private_key` (str): Private key to store

**Returns:** None

**Raises:**
- `KeyStorageError`: Storage failures
- `InvalidKeyError`: Invalid key format

**Example:**
```python
private_key = "0xabcdef123456789..."
key_storage.store_private_key(private_key)
```

##### `get_private_key() -> str`
Retrieves decrypted private key from secure storage.

**Returns:**
- `str`: Decrypted private key

**Raises:**
- `KeyStorageError`: Retrieval failures
- `DecryptionError`: Decryption failures

**Example:**
```python
try:
    private_key = key_storage.get_private_key()
    # Use private key for signing
    signature = sign_message(message, private_key)
except DecryptionError:
    logger.error("Failed to decrypt private key")
```

##### `emergency_zeroize() -> None`
Securely erases all stored keys from memory.

**Returns:** None

**Example:**
```python
# Called during emergency shutdown
key_storage.emergency_zeroize()
logger.critical("All keys securely erased")
```

### AuditLogger

Comprehensive security event logging with integrity verification.

#### Constructor

```python
AuditLogger(log_file: str, key: bytes = None)
```

**Parameters:**
- `log_file` (str): Path to audit log file
- `key` (bytes): HMAC key for integrity verification

**Example:**
```python
audit = AuditLogger('logs/audit.log')
```

#### Methods

##### `log_event(event_type: str, user_id: str, details: dict) -> None`
Logs a security event with full context.

**Parameters:**
- `event_type` (str): Event type ('LOGIN', 'TRADE', 'CONFIG_CHANGE')
- `user_id` (str): User or system identifier
- `details` (dict): Event details and context

**Returns:** None

**Raises:**
- `AuditLogError`: Logging failures

**Example:**
```python
audit.log_event('TRADE_EXECUTED', 'bot_instance_1', {
    'symbol': 'BTCUSDT',
    'side': 'BUY',
    'quantity': 0.001,
    'price': 45000.00,
    'timestamp': datetime.now().isoformat()
})
```

##### `verify_integrity() -> bool`
Verifies audit log integrity using HMAC signatures.

**Returns:**
- `bool`: True if log integrity is intact

**Raises:**
- `IntegrityCheckError`: Verification failures

**Example:**
```python
if audit.verify_integrity():
    print("Audit log integrity verified")
else:
    print("Audit log has been tampered with!")
```

## 🛠️ Utility Classes

### InputSanitizer

Input validation and sanitization for security.

#### Constructor

```python
InputSanitizer(config: dict = None)
```

**Parameters:**
- `config` (dict): Sanitization rules configuration

**Example:**
```python
sanitizer = InputSanitizer({
    'max_string_length': 1000,
    'allowed_symbols': ['BTCUSDT', 'ETHUSDT']
})
```

#### Methods

##### `sanitize_price(price: Union[str, float]) -> float`
Validates and sanitizes price inputs.

**Parameters:**
- `price` (Union[str, float]): Price value to sanitize

**Returns:**
- `float`: Sanitized price value

**Raises:**
- `ValidationError`: Invalid price format or range

**Example:**
```python
try:
    clean_price = sanitizer.sanitize_price("45000.50")
    print(f"Sanitized price: ${clean_price:.2f}")
except ValidationError as e:
    print(f"Invalid price: {e}")
```

##### `sanitize_quantity(quantity: Union[str, float], symbol: str) -> float`
Validates quantity against symbol-specific limits.

**Parameters:**
- `quantity` (Union[str, float]): Quantity to sanitize
- `symbol` (str): Trading symbol for validation rules

**Returns:**
- `float`: Sanitized quantity

**Raises:**
- `ValidationError`: Invalid quantity

**Example:**
```python
quantity = sanitizer.sanitize_quantity("0.001", "BTCUSDT")
print(f"Sanitized quantity: {quantity} BTC")
```

##### `sanitize_symbol(symbol: str) -> str`
Validates trading symbol format and availability.

**Parameters:**
- `symbol` (str): Symbol to validate

**Returns:**
- `str`: Sanitized symbol

**Raises:**
- `ValidationError`: Invalid symbol

**Example:**
```python
symbol = sanitizer.sanitize_symbol("btcusdt")  # Case insensitive
print(f"Sanitized symbol: {symbol}")  # BTCUSDT
```

### RiskManager

Position sizing and risk management calculations.

#### Constructor

```python
RiskManager(config: dict)
```

**Parameters:**
- `config` (dict): Risk management configuration

**Example:**
```python
config = {
    'risk': {
        'max_position_size_pct': 0.05,
        'max_daily_loss_pct': 0.03,
        'leverage': 5
    }
}
risk_manager = RiskManager(config)
```

#### Methods

##### `calculate_position_size(balance: float, entry_price: float, stop_loss: float, volatility: float = None) -> float`
Calculates position size based on risk parameters.

**Parameters:**
- `balance` (float): Account balance
- `entry_price` (float): Entry price
- `stop_loss` (float): Stop loss price
- `volatility` (float): Current volatility (optional)

**Returns:**
- `float`: Position size in base currency

**Example:**
```python
balance = 10000.0
entry_price = 45000.0
stop_loss = 44000.0  # 2% risk per trade

size = risk_manager.calculate_position_size(balance, entry_price, stop_loss)
print(f"Position size: {size} BTC")
```

##### `check_daily_loss_limit(current_pnl: float, balance: float) -> bool`
Checks if daily loss limit has been exceeded.

**Parameters:**
- `current_pnl` (float): Current daily P&L
- `balance` (float): Account balance

**Returns:**
- `bool`: True if within limits, False if exceeded

**Example:**
```python
daily_pnl = -350.0  # $350 loss today
balance = 10000.0

if not risk_manager.check_daily_loss_limit(daily_pnl, balance):
    print("Daily loss limit exceeded!")
    # Trigger emergency shutdown
```

##### `adjust_for_volatility(base_size: float, volatility: float) -> float`
Adjusts position size based on current market volatility.

**Parameters:**
- `base_size` (float): Base position size
- `volatility` (float): Current volatility measure

**Returns:**
- `float`: Adjusted position size

**Example:**
```python
base_size = 0.002  # 0.002 BTC
current_volatility = 2.5  # 2.5x normal volatility

adjusted_size = risk_manager.adjust_for_volatility(base_size, current_volatility)
print(f"Volatility-adjusted size: {adjusted_size} BTC")
```

## 📊 Data Types

### Core Data Types

#### TradeSignal
```python
@dataclass
class TradeSignal:
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    confidence: float = 1.0
    reason: str = ""
    context: dict = field(default_factory=dict)
```

#### Position
```python
@dataclass
class Position:
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    quantity: float
    entry_price: float
    current_price: float = None
    unrealized_pnl: float = 0.0
    stop_loss: float = None
    take_profit: float = None
    entry_time: datetime = None
    last_update: datetime = None
    metadata: dict = field(default_factory=dict)
```

#### OrderRequest
```python
@dataclass
class OrderRequest:
    symbol: str
    side: str  # 'BUY' or 'SELL'
    order_type: str  # 'LIMIT', 'MARKET', 'STOP'
    quantity: float
    price: float = None
    stop_price: float = None
    time_in_force: str = 'GTC'
    reduce_only: bool = False
    post_only: bool = False
```

#### AccountBalance
```python
@dataclass
class AccountBalance:
    total: float
    available: float
    used: float
    currency: str = 'USDC'
    timestamp: datetime = None
```

## ⚠️ Error Handling

### Exception Hierarchy

```
TradingBotException (base exception)
├── ExchangeError
│   ├── APIError
│   ├── AuthenticationError
│   ├── RateLimitError
│   └── NetworkError
├── StrategyError
│   ├── IndicatorError
│   ├── SignalError
│   └── DataError
├── OrderError
│   ├── InvalidOrderError
│   ├── InsufficientFundsError
│   ├── OrderNotFoundError
│   └── OrderRejectedError
├── RiskError
│   ├── RiskLimitError
│   ├── PositionSizeError
│   └── LossLimitError
├── SecurityError
│   ├── KeyStorageError
│   ├── EncryptionError
│   ├── DecryptionError
│   └── AccessDeniedError
├── ResilienceError
│   ├── CircuitBreakerOpenError
│   ├── MaxRetriesExceededError
│   ├── StateSaveError
│   └── StateRecoveryError
└── ValidationError
    ├── InputValidationError
    ├── ResponseValidationError
    └── DataValidationError
```

### Error Handling Patterns

#### API Call with Resilience
```python
from resilience import ResilienceManager

resilience = ResilienceManager(config)

try:
    result = resilience.execute_with_resilience(
        client.get_ticker,
        'BTCUSDT'
    )
except CircuitBreakerOpenError:
    logger.warning("Circuit breaker open, using cached data")
    result = get_cached_ticker('BTCUSDT')
except MaxRetriesExceededError:
    logger.error("API call failed after all retries")
    raise SystemExit("Cannot continue without market data")
```

#### Order Placement with Validation
```python
try:
    # Pre-validate order
    sanitizer.sanitize_price(order.price)
    sanitizer.sanitize_quantity(order.quantity, order.symbol)

    # Check risk limits
    if not risk_manager.check_daily_loss_limit(current_pnl, balance):
        raise RiskLimitError("Daily loss limit exceeded")

    # Place order with resilience
    response = resilience.execute_with_resilience(
        client.place_order,
        order
    )

except ValidationError as e:
    logger.error(f"Order validation failed: {e}")
    return False
except RiskLimitError as e:
    logger.warning(f"Risk limit violation: {e}")
    alerts.send_alert(f"Risk limit hit: {e}", severity='CRITICAL')
    return False
except OrderError as e:
    logger.error(f"Order failed: {e}")
    return False
```

#### Strategy Error Recovery
```python
try:
    signal = strategy.check_entry_signal(df, position_type)
except IndicatorError as e:
    logger.error(f"Indicator calculation failed: {e}")
    # Continue with reduced functionality
    signal = None
except StrategyError as e:
    logger.error(f"Strategy error: {e}")
    # Fallback to basic MACD strategy
    signal = basic_macd_check(df, position_type)
```

This comprehensive API reference provides everything needed to understand, extend, and integrate with the Hyperliquid MACD trading bot system. Each class and method includes detailed documentation, parameter descriptions, return values, exception types, and practical usage examples.
