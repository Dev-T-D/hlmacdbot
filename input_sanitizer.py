"""
Input Sanitization Module

Validates and sanitizes all user inputs from config files and API calls
to prevent injection attacks, invalid data, and ensure data integrity.
"""

import re
import logging
from typing import Union, Optional, Any
from decimal import Decimal, InvalidOperation

from exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Sanitizes and validates user inputs"""
    
    # Valid trading pair patterns (Hyperliquid supports USDT, USDC pairs, and dash format like SOL-USDC)
    SYMBOL_PATTERN = re.compile(r'^[A-Z0-9]{2,20}(-USDC|-USDT|-USD)?(USDT|USDC|USD|BTC|ETH)?$')
    
    # Valid timeframe patterns
    VALID_TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w']
    
    # Exchange limits
    HYPERLIQUID_MAX_LEVERAGE = 50
    MIN_LEVERAGE = 1
    MAX_LEVERAGE = 200  # Absolute maximum safety limit
    
    # Price limits (safety bounds)
    MIN_PRICE = 0.00000001  # Minimum price (1 satoshi equivalent)
    MAX_PRICE = 1000000000  # Maximum price ($1B per unit)
    
    # Quantity limits
    MIN_QUANTITY = 0.00000001  # Minimum quantity
    MAX_QUANTITY = 1000000000  # Maximum quantity
    
    # Percentage limits
    MIN_PERCENTAGE = 0.0
    MAX_PERCENTAGE = 1.0  # 100%
    
    # String length limits
    MAX_SYMBOL_LENGTH = 20
    MAX_API_KEY_LENGTH = 200
    MAX_SECRET_KEY_LENGTH = 200
    MAX_WALLET_ADDRESS_LENGTH = 42  # Ethereum address length
    
    # Integer limits
    MIN_CHECK_INTERVAL = 10  # Minimum 10 seconds
    MAX_CHECK_INTERVAL = 86400  # Maximum 24 hours
    MIN_MAX_TRADES_PER_DAY = 1
    MAX_MAX_TRADES_PER_DAY = 1000
    
    @staticmethod
    def sanitize_symbol(symbol: Any) -> str:
        """
        Sanitize and validate trading symbol.
        
        Args:
            symbol: Symbol to sanitize
            
        Returns:
            Sanitized symbol string (uppercase)
            
        Raises:
            ConfigurationError: If symbol is invalid
        """
        if not isinstance(symbol, str):
            raise ConfigurationError(
                f"Invalid symbol type: expected str, got {type(symbol).__name__}. "
                f"Value: {symbol}"
            )
        
        # Strip whitespace and convert to uppercase
        symbol = symbol.strip().upper()
        
        # Check length
        if len(symbol) == 0:
            raise ConfigurationError("Symbol cannot be empty")
        
        if len(symbol) > InputSanitizer.MAX_SYMBOL_LENGTH:
            raise ConfigurationError(
                f"Symbol too long: {len(symbol)} characters (max: {InputSanitizer.MAX_SYMBOL_LENGTH}). "
                f"Symbol: {symbol}"
            )
        
        # Validate format (alphanumeric, optional USDT/USDC/USD/BTC/ETH suffix)
        if not InputSanitizer.SYMBOL_PATTERN.match(symbol):
            raise ConfigurationError(
                f"Invalid symbol format: '{symbol}'. "
                f"Expected format: [A-Z0-9]{2,20}(USDT|USDC|USD|BTC|ETH)? "
                f"Examples: BTCUSDT, ETHUSDC, SOLUSDT, BTCUSDC"
            )
        
        # Check for dangerous characters (prevent injection)
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '`', '$', '(', ')', '{', '}', '[', ']']
        if any(char in symbol for char in dangerous_chars):
            raise ConfigurationError(
                f"Symbol contains invalid characters: '{symbol}'. "
                f"Only alphanumeric characters and trading pair suffixes are allowed"
            )
        
        return symbol
    
    @staticmethod
    def sanitize_timeframe(timeframe: Any) -> str:
        """
        Sanitize and validate timeframe.
        
        Args:
            timeframe: Timeframe to sanitize
            
        Returns:
            Sanitized timeframe string (lowercase)
            
        Raises:
            ConfigurationError: If timeframe is invalid
        """
        if not isinstance(timeframe, str):
            raise ConfigurationError(
                f"Invalid timeframe type: expected str, got {type(timeframe).__name__}. "
                f"Value: {timeframe}"
            )
        
        # Strip whitespace and convert to lowercase
        timeframe = timeframe.strip().lower()
        
        # Validate against known timeframes
        if timeframe not in InputSanitizer.VALID_TIMEFRAMES:
            raise ConfigurationError(
                f"Invalid timeframe: '{timeframe}'. "
                f"Valid timeframes: {', '.join(InputSanitizer.VALID_TIMEFRAMES)}"
            )
        
        return timeframe
    
    @staticmethod
    def sanitize_price(price: Any, field_name: str = "price") -> float:
        """
        Sanitize and validate price value.
        
        Args:
            price: Price to sanitize (can be str, int, float)
            field_name: Name of field for error messages
            
        Returns:
            Validated price as float
            
        Raises:
            ConfigurationError: If price is invalid
        """
        # Convert to float
        try:
            if isinstance(price, str):
                price = price.strip()
                if not price:
                    raise ConfigurationError(f"{field_name} cannot be empty")
            price_float = float(price)
        except (ValueError, TypeError, AttributeError) as e:
            raise ConfigurationError(
                f"Invalid {field_name}: '{price}' cannot be converted to number. "
                f"Type: {type(price).__name__}, Error: {str(e)}"
            ) from e
        
        # Validate range
        if price_float <= 0:
            raise ConfigurationError(
                f"Invalid {field_name}: {price_float} must be positive. "
                f"Got: {price_float}"
            )
        
        if price_float < InputSanitizer.MIN_PRICE:
            raise ConfigurationError(
                f"Invalid {field_name}: {price_float} below minimum {InputSanitizer.MIN_PRICE}. "
                f"This may indicate a unit error (e.g., satoshis instead of BTC)"
            )
        
        if price_float > InputSanitizer.MAX_PRICE:
            raise ConfigurationError(
                f"Invalid {field_name}: {price_float} above maximum {InputSanitizer.MAX_PRICE}. "
                f"This may indicate a unit error or invalid data"
            )
        
        # Check for NaN or Inf
        if not (price_float == price_float):  # NaN check
            raise ConfigurationError(
                f"Invalid {field_name}: NaN (Not a Number) value detected"
            )
        
        if abs(price_float) == float('inf'):
            raise ConfigurationError(
                f"Invalid {field_name}: Infinity value detected"
            )
        
        return price_float
    
    @staticmethod
    def sanitize_quantity(quantity: Any, field_name: str = "quantity") -> float:
        """
        Sanitize and validate quantity value.
        
        Args:
            quantity: Quantity to sanitize (can be str, int, float)
            field_name: Name of field for error messages
            
        Returns:
            Validated quantity as float
            
        Raises:
            ConfigurationError: If quantity is invalid
        """
        # Convert to float
        try:
            if isinstance(quantity, str):
                quantity = quantity.strip()
                if not quantity:
                    raise ConfigurationError(f"{field_name} cannot be empty")
            quantity_float = float(quantity)
        except (ValueError, TypeError, AttributeError) as e:
            raise ConfigurationError(
                f"Invalid {field_name}: '{quantity}' cannot be converted to number. "
                f"Type: {type(quantity).__name__}, Error: {str(e)}"
            ) from e
        
        # Validate range
        if quantity_float <= 0:
            raise ConfigurationError(
                f"Invalid {field_name}: {quantity_float} must be positive. "
                f"Got: {quantity_float}"
            )
        
        if quantity_float < InputSanitizer.MIN_QUANTITY:
            raise ConfigurationError(
                f"Invalid {field_name}: {quantity_float} below minimum {InputSanitizer.MIN_QUANTITY}"
            )
        
        if quantity_float > InputSanitizer.MAX_QUANTITY:
            raise ConfigurationError(
                f"Invalid {field_name}: {quantity_float} above maximum {InputSanitizer.MAX_QUANTITY}"
            )
        
        # Check for NaN or Inf
        if not (quantity_float == quantity_float):  # NaN check
            raise ConfigurationError(
                f"Invalid {field_name}: NaN (Not a Number) value detected"
            )
        
        if abs(quantity_float) == float('inf'):
            raise ConfigurationError(
                f"Invalid {field_name}: Infinity value detected"
            )
        
        return quantity_float
    
    @staticmethod
    def sanitize_percentage(percentage: Any, field_name: str = "percentage", 
                           min_value: float = None, max_value: float = None) -> float:
        """
        Sanitize and validate percentage value (0.0 to 1.0).
        
        Args:
            percentage: Percentage to sanitize (can be str, int, float)
            field_name: Name of field for error messages
            min_value: Optional minimum value (default: MIN_PERCENTAGE)
            max_value: Optional maximum value (default: MAX_PERCENTAGE)
            
        Returns:
            Validated percentage as float
            
        Raises:
            ConfigurationError: If percentage is invalid
        """
        if min_value is None:
            min_value = InputSanitizer.MIN_PERCENTAGE
        if max_value is None:
            max_value = InputSanitizer.MAX_PERCENTAGE
        
        # Convert to float
        try:
            if isinstance(percentage, str):
                percentage = percentage.strip()
                if not percentage:
                    raise ConfigurationError(f"{field_name} cannot be empty")
            percentage_float = float(percentage)
        except (ValueError, TypeError, AttributeError) as e:
            raise ConfigurationError(
                f"Invalid {field_name}: '{percentage}' cannot be converted to number. "
                f"Type: {type(percentage).__name__}, Error: {str(e)}"
            ) from e
        
        # Validate range
        if percentage_float < min_value:
            raise ConfigurationError(
                f"Invalid {field_name}: {percentage_float} below minimum {min_value} "
                f"({min_value * 100:.1f}%). Got: {percentage_float} ({percentage_float * 100:.2f}%)"
            )
        
        if percentage_float > max_value:
            raise ConfigurationError(
                f"Invalid {field_name}: {percentage_float} above maximum {max_value} "
                f"({max_value * 100:.1f}%). Got: {percentage_float} ({percentage_float * 100:.2f}%)"
            )
        
        # Check for NaN or Inf
        if not (percentage_float == percentage_float):  # NaN check
            raise ConfigurationError(
                f"Invalid {field_name}: NaN (Not a Number) value detected"
            )
        
        if abs(percentage_float) == float('inf'):
            raise ConfigurationError(
                f"Invalid {field_name}: Infinity value detected"
            )
        
        return percentage_float
    
    @staticmethod
    def sanitize_leverage(leverage: Any, exchange: str = "hyperliquid") -> int:
        """
        Sanitize and validate leverage value.
        
        Args:
            leverage: Leverage to sanitize (can be str, int, float)
            exchange: Exchange name for limit validation
            
        Returns:
            Validated leverage as int
            
        Raises:
            ConfigurationError: If leverage is invalid
        """
        # Convert to int
        try:
            if isinstance(leverage, str):
                leverage = leverage.strip()
                if not leverage:
                    raise ConfigurationError("leverage cannot be empty")
            leverage_int = int(float(leverage))  # Allow "10.0" -> 10
        except (ValueError, TypeError, AttributeError) as e:
            raise ConfigurationError(
                f"Invalid leverage: '{leverage}' cannot be converted to integer. "
                f"Type: {type(leverage).__name__}, Error: {str(e)}"
            ) from e
        
        # Validate range
        if leverage_int < InputSanitizer.MIN_LEVERAGE:
            raise ConfigurationError(
                f"Invalid leverage: {leverage_int} below minimum {InputSanitizer.MIN_LEVERAGE}x"
            )
        
        if leverage_int > InputSanitizer.MAX_LEVERAGE:
            raise ConfigurationError(
                f"Invalid leverage: {leverage_int} above absolute maximum {InputSanitizer.MAX_LEVERAGE}x. "
                f"This is a safety limit to prevent configuration errors"
            )
        
        # Exchange-specific limits
        exchange_upper = exchange.lower()
        if exchange_upper == "hyperliquid" and leverage_int > InputSanitizer.HYPERLIQUID_MAX_LEVERAGE:
            raise ConfigurationError(
                f"Invalid leverage for Hyperliquid: {leverage_int}x exceeds maximum "
                f"{InputSanitizer.HYPERLIQUID_MAX_LEVERAGE}x"
            )
        # Bitunix support removed - only Hyperliquid is supported
        
        return leverage_int
    
    @staticmethod
    def sanitize_positive_integer(value: Any, field_name: str, 
                                  min_value: int = 1, max_value: int = None) -> int:
        """
        Sanitize and validate positive integer.
        
        Args:
            value: Value to sanitize
            field_name: Name of field for error messages
            min_value: Minimum allowed value
            max_value: Maximum allowed value (None = no maximum)
            
        Returns:
            Validated integer
            
        Raises:
            ConfigurationError: If value is invalid
        """
        # Convert to int
        try:
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    raise ConfigurationError(f"{field_name} cannot be empty")
            value_int = int(float(value))  # Allow "10.0" -> 10
        except (ValueError, TypeError, AttributeError) as e:
            raise ConfigurationError(
                f"Invalid {field_name}: '{value}' cannot be converted to integer. "
                f"Type: {type(value).__name__}, Error: {str(e)}"
            ) from e
        
        # Validate range
        if value_int < min_value:
            raise ConfigurationError(
                f"Invalid {field_name}: {value_int} below minimum {min_value}"
            )
        
        if max_value is not None and value_int > max_value:
            raise ConfigurationError(
                f"Invalid {field_name}: {value_int} above maximum {max_value}"
            )
        
        return value_int
    
    @staticmethod
    def sanitize_check_interval(interval: Any) -> int:
        """
        Sanitize and validate check interval (seconds).
        
        Args:
            interval: Interval to sanitize
            
        Returns:
            Validated interval as int
            
        Raises:
            ConfigurationError: If interval is invalid
        """
        return InputSanitizer.sanitize_positive_integer(
            interval,
            "check_interval",
            min_value=InputSanitizer.MIN_CHECK_INTERVAL,
            max_value=InputSanitizer.MAX_CHECK_INTERVAL
        )
    
    @staticmethod
    def sanitize_max_trades_per_day(value: Any) -> int:
        """
        Sanitize and validate max trades per day.
        
        Args:
            value: Value to sanitize
            
        Returns:
            Validated integer
            
        Raises:
            ConfigurationError: If value is invalid
        """
        return InputSanitizer.sanitize_positive_integer(
            value,
            "max_trades_per_day",
            min_value=InputSanitizer.MIN_MAX_TRADES_PER_DAY,
            max_value=InputSanitizer.MAX_MAX_TRADES_PER_DAY
        )
    
    @staticmethod
    def sanitize_api_key(api_key: Any, field_name: str = "api_key") -> str:
        """
        Sanitize and validate API key.
        
        Args:
            api_key: API key to sanitize
            field_name: Name of field for error messages
            
        Returns:
            Sanitized API key string
            
        Raises:
            ConfigurationError: If API key is invalid
        """
        if not isinstance(api_key, str):
            raise ConfigurationError(
                f"Invalid {field_name} type: expected str, got {type(api_key).__name__}"
            )
        
        # Strip whitespace
        api_key = api_key.strip()
        
        # Check length
        if len(api_key) == 0:
            raise ConfigurationError(f"{field_name} cannot be empty")
        
        if len(api_key) > InputSanitizer.MAX_API_KEY_LENGTH:
            raise ConfigurationError(
                f"{field_name} too long: {len(api_key)} characters (max: {InputSanitizer.MAX_API_KEY_LENGTH})"
            )
        
        # Check for dangerous characters (prevent injection)
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '`', '$', '(', ')', '{', '}', '[', ']', '\n', '\r']
        if any(char in api_key for char in dangerous_chars):
            raise ConfigurationError(
                f"{field_name} contains invalid characters. "
                f"Only alphanumeric characters, hyphens, and underscores are allowed"
            )
        
        return api_key
    
    @staticmethod
    def sanitize_private_key(private_key: Any) -> str:
        """
        Sanitize and validate Ethereum private key.
        
        Args:
            private_key: Private key to sanitize
            
        Returns:
            Sanitized private key string
            
        Raises:
            ConfigurationError: If private key is invalid
        """
        if not isinstance(private_key, str):
            raise ConfigurationError(
                f"Invalid private_key type: expected str, got {type(private_key).__name__}"
            )
        
        # Strip whitespace
        private_key = private_key.strip()
        
        # Check length (0x + 64 hex chars = 66)
        if len(private_key) != 66:
            raise ConfigurationError(
                f"Invalid private_key length: expected 66 characters (0x + 64 hex), "
                f"got {len(private_key)}. "
                f"Private key must start with 0x and be 66 characters total"
            )
        
        # Check prefix
        if not private_key.startswith('0x'):
            raise ConfigurationError(
                f"Invalid private_key format: must start with '0x'. "
                f"Got: {private_key[:10]}..."
            )
        
        # Validate hex characters
        hex_part = private_key[2:]
        if not all(c in '0123456789abcdefABCDEF' for c in hex_part):
            raise ConfigurationError(
                f"Invalid private_key format: contains non-hexadecimal characters. "
                f"Private key must be hexadecimal (0-9, a-f, A-F)"
            )
        
        # Check for obviously invalid keys
        if hex_part.lower() == '0' * 64:
            raise ConfigurationError(
                "Invalid private_key: cannot use zero private key"
            )
        
        if len(set(hex_part.lower())) == 1:
            raise ConfigurationError(
                "Invalid private_key: appears to be a test/placeholder key (all same character)"
            )
        
        return private_key.lower()  # Normalize to lowercase
    
    @staticmethod
    def sanitize_wallet_address(address: Any) -> str:
        """
        Sanitize and validate Ethereum wallet address.
        
        Args:
            address: Wallet address to sanitize
            
        Returns:
            Sanitized wallet address string (lowercase)
            
        Raises:
            ConfigurationError: If wallet address is invalid
        """
        if not isinstance(address, str):
            raise ConfigurationError(
                f"Invalid wallet_address type: expected str, got {type(address).__name__}"
            )
        
        # Strip whitespace
        address = address.strip()
        
        # Check length (0x + 40 hex chars = 42)
        if len(address) != 42:
            raise ConfigurationError(
                f"Invalid wallet_address length: expected 42 characters (0x + 40 hex), "
                f"got {len(address)}. "
                f"Wallet address must start with 0x and be 42 characters total"
            )
        
        # Check prefix
        if not address.startswith('0x'):
            raise ConfigurationError(
                f"Invalid wallet_address format: must start with '0x'. "
                f"Got: {address[:10]}..."
            )
        
        # Validate hex characters
        hex_part = address[2:]
        if not all(c in '0123456789abcdefABCDEF' for c in hex_part):
            raise ConfigurationError(
                f"Invalid wallet_address format: contains non-hexadecimal characters. "
                f"Wallet address must be hexadecimal (0-9, a-f, A-F)"
            )
        
        return address.lower()  # Normalize to lowercase
    
    @staticmethod
    def sanitize_risk_reward_ratio(ratio: Any) -> float:
        """
        Sanitize and validate risk/reward ratio.
        
        Args:
            ratio: Risk/reward ratio to sanitize
            
        Returns:
            Validated ratio as float
            
        Raises:
            ConfigurationError: If ratio is invalid
        """
        # Convert to float
        try:
            if isinstance(ratio, str):
                ratio = ratio.strip()
                if not ratio:
                    raise ConfigurationError("risk_reward_ratio cannot be empty")
            ratio_float = float(ratio)
        except (ValueError, TypeError, AttributeError) as e:
            raise ConfigurationError(
                f"Invalid risk_reward_ratio: '{ratio}' cannot be converted to number. "
                f"Type: {type(ratio).__name__}, Error: {str(e)}"
            ) from e
        
        # Validate range (typically 1.0 to 10.0)
        if ratio_float < 0.1:
            raise ConfigurationError(
                f"Invalid risk_reward_ratio: {ratio_float} below minimum 0.1. "
                f"Risk/reward ratio should be at least 0.1 (risking 10x potential profit)"
            )
        
        if ratio_float > 100.0:
            raise ConfigurationError(
                f"Invalid risk_reward_ratio: {ratio_float} above maximum 100.0. "
                f"This may indicate a configuration error"
            )
        
        # Check for NaN or Inf
        if not (ratio_float == ratio_float):  # NaN check
            raise ConfigurationError(
                "Invalid risk_reward_ratio: NaN (Not a Number) value detected"
            )
        
        if abs(ratio_float) == float('inf'):
            raise ConfigurationError(
                "Invalid risk_reward_ratio: Infinity value detected"
            )
        
        return ratio_float
    
    @staticmethod
    def sanitize_macd_length(length: Any, field_name: str) -> int:
        """
        Sanitize and validate MACD length parameter.
        
        Args:
            length: Length to sanitize
            field_name: Name of field (fast_length, slow_length, signal_length)
            
        Returns:
            Validated length as int
            
        Raises:
            ConfigurationError: If length is invalid
        """
        # Convert to int
        try:
            if isinstance(length, str):
                length = length.strip()
                if not length:
                    raise ConfigurationError(f"{field_name} cannot be empty")
            length_int = int(float(length))  # Allow "12.0" -> 12
        except (ValueError, TypeError, AttributeError) as e:
            raise ConfigurationError(
                f"Invalid {field_name}: '{length}' cannot be converted to integer. "
                f"Type: {type(length).__name__}, Error: {str(e)}"
            ) from e
        
        # Validate range (typically 1-200)
        if length_int < 1:
            raise ConfigurationError(
                f"Invalid {field_name}: {length_int} below minimum 1"
            )
        
        if length_int > 200:
            raise ConfigurationError(
                f"Invalid {field_name}: {length_int} above maximum 200. "
                f"This may indicate a configuration error"
            )
        
        return length_int
    
    @staticmethod
    def sanitize_boolean(value: Any, field_name: str, default: bool = False) -> bool:
        """
        Sanitize and validate boolean value.
        
        Args:
            value: Value to sanitize
            field_name: Name of field for error messages
            default: Default value if value is None or empty
            
        Returns:
            Validated boolean
            
        Raises:
            ConfigurationError: If value cannot be converted to boolean
        """
        if value is None:
            return default
        
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            value = value.strip().lower()
            if value in ('true', '1', 'yes', 'on'):
                return True
            elif value in ('false', '0', 'no', 'off', ''):
                return False
            else:
                raise ConfigurationError(
                    f"Invalid {field_name}: '{value}' cannot be converted to boolean. "
                    f"Expected: true/false, 1/0, yes/no, on/off"
                )
        
        if isinstance(value, (int, float)):
            return bool(value)
        
        raise ConfigurationError(
            f"Invalid {field_name} type: cannot convert {type(value).__name__} to boolean. "
            f"Value: {value}"
        )
    
    @staticmethod
    def sanitize_host(host: Any) -> str:
        """
        Sanitize and validate host address.
        
        Args:
            host: Host address to sanitize
            
        Returns:
            Sanitized host string
            
        Raises:
            ConfigurationError: If host is invalid
        """
        if not isinstance(host, str):
            raise ConfigurationError(
                f"Invalid host type: expected str, got {type(host).__name__}"
            )
        
        host = host.strip()
        
        # Valid host patterns
        valid_patterns = [
            r'^127\.0\.0\.1$',  # localhost IPv4
            r'^0\.0\.0\.0$',     # all interfaces IPv4
            r'^localhost$',      # localhost hostname
            r'^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$',  # IPv4
        ]
        
        if not any(re.match(pattern, host) for pattern in valid_patterns):
            raise ConfigurationError(
                f"Invalid host: '{host}'. "
                f"Valid hosts: 127.0.0.1 (localhost), 0.0.0.0 (all interfaces), "
                f"or valid IPv4 address"
            )
        
        return host
    
    @staticmethod
    def sanitize_port(port: Any) -> int:
        """
        Sanitize and validate port number.
        
        Args:
            port: Port to sanitize
            
        Returns:
            Validated port as int
            
        Raises:
            ConfigurationError: If port is invalid
        """
        # Convert to int
        try:
            if isinstance(port, str):
                port = port.strip()
                if not port:
                    raise ConfigurationError("port cannot be empty")
            port_int = int(float(port))
        except (ValueError, TypeError, AttributeError) as e:
            raise ConfigurationError(
                f"Invalid port: '{port}' cannot be converted to integer. "
                f"Type: {type(port).__name__}, Error: {str(e)}"
            ) from e
        
        # Validate range (1-65535)
        if port_int < 1:
            raise ConfigurationError(
                f"Invalid port: {port_int} below minimum 1"
            )
        
        if port_int > 65535:
            raise ConfigurationError(
                f"Invalid port: {port_int} above maximum 65535"
            )
        
        # Warn about common restricted ports
        restricted_ports = [22, 23, 25, 53, 80, 443, 3306, 5432]
        if port_int in restricted_ports:
            logger.warning(
                f"Port {port_int} is commonly used by system services. "
                f"Consider using a different port (e.g., 8080, 3000, 5000)"
            )
        
        return port_int

