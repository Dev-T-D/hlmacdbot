"""
Configuration Validator

Validates configuration files and credentials for the trading bot
Supports Hyperliquid exchange configuration (Bitunix support removed)

"""

import json
import os
import re
import logging
from typing import Dict, Tuple, Optional
from eth_utils import is_address, to_checksum_address

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates trading bot configuration"""
    
    # Valid exchanges
    VALID_EXCHANGES = ["hyperliquid"]
    
    # Valid timeframes
    VALID_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    
    # Valid symbols (extend as needed)
    VALID_SYMBOLS = [
        "BTCUSDT", "BTC",
        "ETHUSDT", "ETH",
        "SOLUSDT", "SOL"
    ]
    
    def __init__(self, config_path: str = "config/config.json"):
        """
        Initialize validator
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = None
        self.errors = []
        self.warnings = []
    
    def load_config(self) -> Dict:
        """
        Load configuration from file and environment variables
        
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config is invalid JSON
        """
        # Load from file
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Copy config.example.json to config.json and update with your credentials"
            )
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in configuration file: {e}",
                e.doc, e.pos
            )
        
        # Override with environment variables if present
        self._load_env_overrides()
        
        return self.config
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables"""
        env_mappings = {
            "HYPERLIQUID_PRIVATE_KEY": ("private_key", None),
            "HYPERLIQUID_WALLET_ADDRESS": ("wallet_address", None),
        }
        
        for env_var, (config_key, section) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                if section:
                    self.config[section][config_key] = value
                else:
                    self.config[config_key] = value
                logger.info(f"Loaded {config_key} from environment variable {env_var}")
    
    def validate(self) -> Tuple[bool, list, list]:
        """
        Validate complete configuration
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        if self.config is None:
            self.load_config()
        
        # Validate exchange type
        self._validate_exchange()
        
        # Validate credentials based on exchange
        exchange = self.config.get("exchange", "").lower()
        
        if exchange == "hyperliquid":
            self._validate_hyperliquid_credentials()
        elif exchange:
            self.errors.append(
                f"Invalid exchange '{exchange}'. Only 'hyperliquid' is supported. "
                f"Bitunix support has been removed."
            )
        
        # Validate trading parameters
        self._validate_trading_params()
        
        # Validate strategy parameters
        self._validate_strategy_params()
        
        # Validate risk parameters
        self._validate_risk_params()
        
        # Security checks
        self._security_checks()
        
        is_valid = len(self.errors) == 0
        
        return is_valid, self.errors, self.warnings
    
    def _validate_exchange(self):
        """Validate exchange configuration"""
        exchange = self.config.get("exchange", "").lower()
        
        if not exchange:
            self.errors.append("Exchange not specified in configuration")
        elif exchange not in self.VALID_EXCHANGES:
            self.errors.append(
                f"Invalid exchange '{exchange}'. Must be one of: {', '.join(self.VALID_EXCHANGES)}"
            )
    
    def _validate_hyperliquid_credentials(self):
        """Validate Hyperliquid-specific credentials"""
        private_key = self.config.get("private_key", "")
        wallet_address = self.config.get("wallet_address", "")
        
        # Validate private key
        if not private_key:
            self.errors.append("Hyperliquid private_key is required")
        else:
            if not self._is_valid_private_key(private_key):
                self.errors.append(
                    "Invalid private_key format. Must be 64 hex characters with optional 0x prefix"
                )
            
            # Check for example/placeholder keys
            if private_key == "0x0000000000000000000000000000000000000000000000000000000000000000":
                self.errors.append(
                    "Using placeholder private_key. Please set your actual Hyperliquid private key"
                )
        
        # Validate wallet address
        if not wallet_address:
            self.errors.append("Hyperliquid wallet_address is required")
        else:
            if not self._is_valid_eth_address(wallet_address):
                self.errors.append(
                    "Invalid wallet_address format. Must be a valid Ethereum address (0x...)"
                )
            
            # Check for example/placeholder address
            if wallet_address == "0x0000000000000000000000000000000000000000":
                self.errors.append(
                    "Using placeholder wallet_address. Please set your actual Hyperliquid wallet address"
                )
            
            # Check checksum
            try:
                if wallet_address != to_checksum_address(wallet_address):
                    self.warnings.append(
                        f"Wallet address is not checksummed. Consider using: {to_checksum_address(wallet_address)}"
                    )
            except Exception:
                pass
    
    def _validate_trading_params(self):
        """Validate trading parameters"""
        trading = self.config.get("trading", {})
        
        # Symbol
        symbol = trading.get("symbol", "")
        if not symbol:
            self.errors.append("Trading symbol is required")
        elif symbol not in self.VALID_SYMBOLS:
            self.warnings.append(
                f"Symbol '{symbol}' not in known list. Ensure it's supported by the exchange"
            )
        
        # Timeframe
        timeframe = trading.get("timeframe", "")
        if not timeframe:
            self.errors.append("Trading timeframe is required")
        elif timeframe not in self.VALID_TIMEFRAMES:
            self.errors.append(
                f"Invalid timeframe '{timeframe}'. Must be one of: {', '.join(self.VALID_TIMEFRAMES)}"
            )
        
        # Check interval
        check_interval = trading.get("check_interval")
        if check_interval is None:
            self.errors.append("Trading check_interval is required")
        elif not isinstance(check_interval, (int, float)) or check_interval <= 0:
            self.errors.append("check_interval must be a positive number")
        elif check_interval < 60:
            self.warnings.append(
                f"check_interval is very short ({check_interval}s). May cause rate limiting"
            )
        
        # Dry run
        dry_run = trading.get("dry_run")
        if dry_run is None:
            self.warnings.append("dry_run not specified, defaulting to True")
        elif not dry_run and self.config.get("testnet"):
            self.warnings.append(
                "dry_run=false on testnet. Consider using dry_run=true for testing"
            )
    
    def _validate_strategy_params(self):
        """Validate strategy parameters"""
        strategy = self.config.get("strategy", {})
        
        params_to_check = {
            "fast_length": (1, 100),
            "slow_length": (1, 200),
            "signal_length": (1, 50),
            "risk_reward_ratio": (0.5, 10.0)
        }
        
        for param, (min_val, max_val) in params_to_check.items():
            value = strategy.get(param)
            if value is None:
                self.errors.append(f"Strategy parameter '{param}' is required")
            elif not isinstance(value, (int, float)):
                self.errors.append(f"Strategy parameter '{param}' must be a number")
            elif not (min_val <= value <= max_val):
                self.errors.append(
                    f"Strategy parameter '{param}' must be between {min_val} and {max_val}"
                )
        
        # Validate MACD relationships
        fast = strategy.get("fast_length", 0)
        slow = strategy.get("slow_length", 0)
        if fast and slow and fast >= slow:
            self.errors.append(
                f"fast_length ({fast}) must be less than slow_length ({slow})"
            )
    
    def _validate_risk_params(self):
        """Validate risk management parameters"""
        risk = self.config.get("risk", {})
        
        # Leverage
        leverage = risk.get("leverage")
        if leverage is None:
            self.errors.append("Risk parameter 'leverage' is required")
        elif not isinstance(leverage, int) or leverage < 1:
            self.errors.append("Leverage must be an integer >= 1")
        else:
            exchange = self.config.get("exchange", "").lower()
            max_leverage = 50 if exchange == "hyperliquid" else 125
            if leverage > max_leverage:
                self.errors.append(
                    f"Leverage {leverage} exceeds maximum for {exchange} ({max_leverage}x)"
                )
            if leverage > 20:
                self.warnings.append(
                    f"High leverage ({leverage}x) increases risk significantly"
                )
        
        # Position size percentage
        max_pos_pct = risk.get("max_position_size_pct")
        if max_pos_pct is None:
            self.errors.append("Risk parameter 'max_position_size_pct' is required")
        elif not (0 < max_pos_pct <= 1):
            self.errors.append("max_position_size_pct must be between 0 and 1")
        elif max_pos_pct > 0.2:
            self.warnings.append(
                f"Large position size ({max_pos_pct*100}% of equity) increases risk"
            )
        
        # Daily loss percentage
        daily_loss_pct = risk.get("max_daily_loss_pct")
        if daily_loss_pct is None:
            self.errors.append("Risk parameter 'max_daily_loss_pct' is required")
        elif not (0 < daily_loss_pct <= 1):
            self.errors.append("max_daily_loss_pct must be between 0 and 1")
        
        # Max trades per day
        max_trades = risk.get("max_trades_per_day")
        if max_trades is None:
            self.errors.append("Risk parameter 'max_trades_per_day' is required")
        elif not isinstance(max_trades, int) or max_trades < 1:
            self.errors.append("max_trades_per_day must be an integer >= 1")
        
        # Trailing stop
        trailing = risk.get("trailing_stop", {})
        if trailing.get("enabled"):
            self._validate_trailing_stop(trailing)
    
    def _validate_trailing_stop(self, trailing: Dict):
        """Validate trailing stop parameters"""
        params_to_check = {
            "trail_percent": (0.1, 10.0),
            "activation_percent": (0.1, 10.0),
            "update_threshold_percent": (0.1, 5.0)
        }
        
        for param, (min_val, max_val) in params_to_check.items():
            value = trailing.get(param)
            if value is None:
                self.warnings.append(f"Trailing stop parameter '{param}' not set")
            elif not isinstance(value, (int, float)):
                self.errors.append(f"Trailing stop '{param}' must be a number")
            elif not (min_val <= value <= max_val):
                self.warnings.append(
                    f"Trailing stop '{param}' ({value}) is outside typical range ({min_val}-{max_val})"
                )
    
    def _security_checks(self):
        """Perform security checks"""
        # Check if using example config
        if self.config_path.endswith("config.example.json"):
            self.errors.append(
                "Using config.example.json. Copy to config.json and update with real credentials"
            )
        
        # Check for production settings
        if not self.config.get("testnet") and not self.config.get("trading", {}).get("dry_run"):
            self.warnings.append(
                "⚠️  PRODUCTION MODE: testnet=false and dry_run=false. Real money at risk!"
            )
        
        # Check file permissions (Unix-like systems)
        try:
            import stat
            st = os.stat(self.config_path)
            if st.st_mode & stat.S_IRGRP or st.st_mode & stat.S_IROTH:
                self.warnings.append(
                    f"Config file {self.config_path} is readable by others. "
                    f"Run: chmod 600 {self.config_path}"
                )
        except Exception:
            pass  # Skip on Windows or if check fails
    
    @staticmethod
    def _is_valid_private_key(key: str) -> bool:
        """
        Validate Ethereum private key format
        
        Args:
            key: Private key string
            
        Returns:
            True if valid format
        """
        # Remove 0x prefix if present
        if key.startswith('0x'):
            key = key[2:]
        
        # Check if 64 hex characters
        if len(key) != 64:
            return False
        
        # Check if valid hex
        try:
            int(key, 16)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def _is_valid_eth_address(address: str) -> bool:
        """
        Validate Ethereum address format
        
        Args:
            address: Ethereum address
            
        Returns:
            True if valid format
        """
        if not address.startswith('0x'):
            return False
        
        if len(address) != 42:
            return False
        
        try:
            # Use eth_utils for proper validation
            return is_address(address)
        except Exception:
            return False
    
    def print_validation_results(self):
        """Print validation results to console"""
        print("\n" + "=" * 70)
        print("CONFIGURATION VALIDATION RESULTS")
        print("=" * 70)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ Configuration is valid!")
        elif not self.errors:
            print("\n✅ Configuration is valid (with warnings)")
        else:
            print("\n❌ Configuration has errors that must be fixed")
        
        print("=" * 70 + "\n")


def validate_config(config_path: str = "config/config.json") -> Tuple[bool, Dict]:
    """
    Validate configuration file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (is_valid, config_dict)
    """
    validator = ConfigValidator(config_path)
    
    try:
        config = validator.load_config()
        is_valid, errors, warnings = validator.validate()
        validator.print_validation_results()
        
        if not is_valid:
            raise ValueError(f"Configuration validation failed with {len(errors)} error(s)")
        
        return is_valid, config
        
    except Exception as e:
        print(f"\n❌ Configuration error: {e}\n")
        raise


if __name__ == "__main__":
    # Test the validator
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.json"
    
    try:
        is_valid, config = validate_config(config_path)
        
        if is_valid:
            print(f"✅ Configuration loaded successfully for {config.get('exchange')} exchange")
            sys.exit(0)
        else:
            print("❌ Configuration validation failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

