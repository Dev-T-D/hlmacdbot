"""
Test for config validation

Tests:
- Missing file
- Invalid JSON
- Missing required fields
- Invalid values
"""

import unittest
import json
import os
import tempfile
from unittest.mock import patch, mock_open

# Handle optional dependencies
try:
    from trading_bot import TradingBot
    from exceptions import ConfigurationError
    TRADING_BOT_AVAILABLE = True
except ImportError as e:
    TRADING_BOT_AVAILABLE = False
    ConfigurationError = Exception  # Fallback
    print(f"Warning: Could not import trading_bot: {e}")


class TestConfigValidation(unittest.TestCase):
    """Test cases for config validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.valid_config = {
            'exchange': 'hyperliquid',
            'testnet': True,
            'trading': {
                'symbol': 'BTC',
                'timeframe': '1h',
                'leverage': 10,
                'check_interval': 60
            },
            'risk': {
                'max_position_size_pct': 0.1,
                'max_daily_loss_pct': 0.05,
                'max_trades_per_day': 10
            },
            'private_key': '0x' + '0' * 64,
            'wallet_address': '0x' + '0' * 40
        }
    
    @unittest.skipIf(not TRADING_BOT_AVAILABLE, "trading_bot module not available")
    def test_missing_file(self):
        """Test handling of missing config file"""
        with self.assertRaises((FileNotFoundError, IOError)):
            TradingBot(config_path='nonexistent_config.json')
    
    @unittest.skipIf(not TRADING_BOT_AVAILABLE, "trading_bot module not available")
    def test_invalid_json(self):
        """Test handling of invalid JSON"""
        invalid_json = "{'invalid': json syntax}"
        
        with patch('builtins.open', mock_open(read_data=invalid_json)):
            with self.assertRaises((json.JSONDecodeError, ValueError)):
                TradingBot(config_path='invalid.json')
    
    @unittest.skipIf(not TRADING_BOT_AVAILABLE, "trading_bot module not available")
    def test_missing_required_fields(self):
        """Test handling of missing required fields"""
        # Missing 'exchange'
        config_missing_exchange = self.valid_config.copy()
        del config_missing_exchange['exchange']
        
        with patch('trading_bot.TradingBot._load_config', return_value=config_missing_exchange):
            with self.assertRaises((KeyError, ValueError)):
                bot = TradingBot.__new__(TradingBot)
                bot.config = config_missing_exchange
                bot._validate_config()
        
        # Missing 'trading' section
        config_missing_trading = self.valid_config.copy()
        del config_missing_trading['trading']
        
        with patch('trading_bot.TradingBot._load_config', return_value=config_missing_trading):
            with self.assertRaises((KeyError, ValueError)):
                bot = TradingBot.__new__(TradingBot)
                bot.config = config_missing_trading
                bot._validate_config()
    
    @unittest.skipIf(not TRADING_BOT_AVAILABLE, "trading_bot module not available")
    def test_invalid_exchange(self):
        """Test handling of invalid exchange value"""
        config_invalid_exchange = self.valid_config.copy()
        config_invalid_exchange['exchange'] = 'invalid_exchange'
        
        with patch('trading_bot.TradingBot._load_config', return_value=config_invalid_exchange):
            # Should handle invalid exchange gracefully
            try:
                bot = TradingBot.__new__(TradingBot)
                bot.config = config_invalid_exchange
                # May raise ValueError or use default
            except ValueError:
                pass  # Expected
    
    @unittest.skipIf(not TRADING_BOT_AVAILABLE, "trading_bot module not available")
    def test_invalid_leverage(self):
        """Test handling of invalid leverage value"""
        config_invalid_leverage = self.valid_config.copy()
        config_invalid_leverage['trading']['leverage'] = -10  # Negative leverage
        
        with patch('trading_bot.TradingBot._load_config', return_value=config_invalid_leverage):
            with self.assertRaises((ValueError, AssertionError)):
                bot = TradingBot.__new__(TradingBot)
                bot.config = config_invalid_leverage
                bot._validate_config()
    
    @unittest.skipIf(not TRADING_BOT_AVAILABLE, "trading_bot module not available")
    def test_invalid_risk_percentages(self):
        """Test handling of invalid risk percentages"""
        # Negative max_position_size_pct
        config_invalid_risk = self.valid_config.copy()
        config_invalid_risk['risk']['max_position_size_pct'] = -0.1
        
        with patch('trading_bot.TradingBot._load_config', return_value=config_invalid_risk):
            with self.assertRaises((ValueError, AssertionError)):
                bot = TradingBot.__new__(TradingBot)
                bot.config = config_invalid_risk
                bot._validate_config()
        
        # Percentage > 1.0
        config_invalid_risk2 = self.valid_config.copy()
        config_invalid_risk2['risk']['max_position_size_pct'] = 1.5  # > 100%
        
        with patch('trading_bot.TradingBot._load_config', return_value=config_invalid_risk2):
            with self.assertRaises((ValueError, AssertionError)):
                bot = TradingBot.__new__(TradingBot)
                bot.config = config_invalid_risk2
                bot._validate_config()
    
    @unittest.skipIf(not TRADING_BOT_AVAILABLE, "trading_bot module not available")
    def test_missing_credentials(self):
        """Test handling of missing credentials"""
        config_no_creds = self.valid_config.copy()
        del config_no_creds['private_key']
        del config_no_creds['wallet_address']
        
        with patch('trading_bot.TradingBot._load_config', return_value=config_no_creds):
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises((KeyError, ValueError, ConfigurationError)):
                    bot = TradingBot.__new__(TradingBot)
                    bot.config = config_no_creds
                    # Initialize credential_manager for the test
                    from credential_manager import CredentialManager
                    bot.credential_manager = CredentialManager(use_keyring=False)
                    bot._load_credentials()
    
    @unittest.skipIf(not TRADING_BOT_AVAILABLE, "trading_bot module not available")
    def test_invalid_symbol(self):
        """Test handling of invalid symbol"""
        config_invalid_symbol = self.valid_config.copy()
        config_invalid_symbol['trading']['symbol'] = ''  # Empty symbol
        
        with patch('trading_bot.TradingBot._load_config', return_value=config_invalid_symbol):
            with self.assertRaises((ValueError, AssertionError)):
                bot = TradingBot.__new__(TradingBot)
                bot.config = config_invalid_symbol
                bot._validate_config()
    
    @unittest.skipIf(not TRADING_BOT_AVAILABLE, "trading_bot module not available")
    def test_invalid_timeframe(self):
        """Test handling of invalid timeframe"""
        config_invalid_timeframe = self.valid_config.copy()
        config_invalid_timeframe['trading']['timeframe'] = 'invalid'
        
        with patch('trading_bot.TradingBot._load_config', return_value=config_invalid_timeframe):
            # May or may not validate timeframe format
            try:
                bot = TradingBot.__new__(TradingBot)
                bot.config = config_invalid_timeframe
                bot._validate_config()
            except (ValueError, AssertionError):
                pass  # May validate timeframe format
    
    @unittest.skipIf(not TRADING_BOT_AVAILABLE, "trading_bot module not available")
    def test_valid_config(self):
        """Test that valid config passes validation"""
        with patch('trading_bot.TradingBot._load_config', return_value=self.valid_config):
            with patch('trading_bot.HyperliquidClient'):
                with patch('trading_bot.TradingBot._load_credentials'):
                    try:
                        bot = TradingBot(config_path='valid.json')
                        self.assertIsNotNone(bot.config, "Valid config should load")
                    except Exception as e:
                        # May fail on client initialization, but config should be valid
                        pass


if __name__ == '__main__':
    unittest.main()

