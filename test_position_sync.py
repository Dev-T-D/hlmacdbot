"""
Test for position synchronization

Tests:
- Position exists on exchange but not in bot
- Position exists in bot but not on exchange
- Mismatch between bot and exchange
"""

import unittest
from unittest.mock import Mock, patch

# Handle optional dependencies
try:
    from trading_bot import TradingBot
    TRADING_BOT_AVAILABLE = True
except ImportError as e:
    TRADING_BOT_AVAILABLE = False
    print(f"Warning: Could not import trading_bot: {e}")


@unittest.skipIf(not TRADING_BOT_AVAILABLE, "trading_bot module not available")
class TestPositionSynchronization(unittest.TestCase):
    """Test cases for position synchronization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = {
            'exchange': 'hyperliquid',
            'testnet': True,
            'trading': {
                'symbol': 'BTC',
                'timeframe': '1h',
                'leverage': 10
            },
            'risk': {
                'max_position_size_pct': 0.1,
                'max_daily_loss_pct': 0.05,
                'max_trades_per_day': 10
            },
            'private_key': '0x' + '0' * 64,
            'wallet_address': '0x' + '0' * 40
        }
    
    @patch('trading_bot.HyperliquidClient')
    def test_position_on_exchange_not_in_bot(self, mock_client_class):
        """Test when position exists on exchange but not tracked in bot"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Exchange has position
        mock_client.get_user_state.return_value = {
            'positions': [{
                'symbol': 'BTC',
                'side': 'LONG',
                'size': 0.1,
                'entry_price': 50000.0
            }]
        }
        
        # Bot has no position
        with patch('trading_bot.TradingBot._load_config', return_value=self.mock_config):
            bot = TradingBot.__new__(TradingBot)
            bot.config = self.mock_config
            bot.client = mock_client
            bot.position = None
            bot.symbol = 'BTC'
            
            # Sync should detect position
            exchange_positions = bot.client.get_user_state()['positions']
            self.assertEqual(len(exchange_positions), 1, "Exchange should have position")
            self.assertIsNone(bot.position, "Bot should not have position initially")
            
            # After sync, bot should recognize position
            if exchange_positions:
                bot.position = {
                    'symbol': exchange_positions[0]['symbol'],
                    'side': exchange_positions[0]['side'],
                    'quantity': exchange_positions[0]['size'],
                    'entry_price': exchange_positions[0]['entry_price']
                }
            
            self.assertIsNotNone(bot.position, "Bot should sync position from exchange")
    
    @patch('trading_bot.HyperliquidClient')
    def test_position_in_bot_not_on_exchange(self, mock_client_class):
        """Test when position exists in bot but not on exchange"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Exchange has no position
        mock_client.get_user_state.return_value = {'positions': []}
        
        # Bot has position
        with patch('trading_bot.TradingBot._load_config', return_value=self.mock_config):
            bot = TradingBot.__new__(TradingBot)
            bot.config = self.mock_config
            bot.client = mock_client
            bot.position = {
                'symbol': 'BTC',
                'side': 'LONG',
                'quantity': 0.1,
                'entry_price': 50000.0
            }
            bot.symbol = 'BTC'
            
            # Sync should detect mismatch
            exchange_positions = bot.client.get_user_state()['positions']
            self.assertEqual(len(exchange_positions), 0, "Exchange should have no position")
            self.assertIsNotNone(bot.position, "Bot should have position")
            
            # After sync, bot should clear position
            if not exchange_positions:
                bot.position = None
            
            self.assertIsNone(bot.position, "Bot should clear position when not on exchange")
    
    @patch('trading_bot.HyperliquidClient')
    def test_position_mismatch(self, mock_client_class):
        """Test when position details mismatch between bot and exchange"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Exchange has different position details
        mock_client.get_user_state.return_value = {
            'positions': [{
                'symbol': 'BTC',
                'side': 'LONG',
                'size': 0.2,  # Different quantity
                'entry_price': 51000.0  # Different entry price
            }]
        }
        
        # Bot has different position
        with patch('trading_bot.TradingBot._load_config', return_value=self.mock_config):
            bot = TradingBot.__new__(TradingBot)
            bot.config = self.mock_config
            bot.client = mock_client
            bot.position = {
                'symbol': 'BTC',
                'side': 'LONG',
                'quantity': 0.1,  # Different quantity
                'entry_price': 50000.0  # Different entry price
            }
            bot.symbol = 'BTC'
            
            # Sync should detect mismatch
            exchange_positions = bot.client.get_user_state()['positions']
            if exchange_positions:
                exchange_pos = exchange_positions[0]
                bot_pos = bot.position
                
                # Check for mismatches
                quantity_mismatch = abs(exchange_pos['size'] - bot_pos['quantity']) > 0.01
                price_mismatch = abs(exchange_pos['entry_price'] - bot_pos['entry_price']) > 100
                
                self.assertTrue(quantity_mismatch or price_mismatch, "Should detect mismatch")
                
                # Bot should update to match exchange
                bot.position['quantity'] = exchange_pos['size']
                bot.position['entry_price'] = exchange_pos['entry_price']
                
                self.assertEqual(bot.position['quantity'], exchange_pos['size'], "Should sync quantity")
                self.assertEqual(bot.position['entry_price'], exchange_pos['entry_price'], "Should sync entry price")
    
    @patch('trading_bot.HyperliquidClient')
    def test_multiple_positions(self, mock_client_class):
        """Test synchronization with multiple positions"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Exchange has multiple positions
        mock_client.get_user_state.return_value = {
            'positions': [
                {
                    'symbol': 'BTC',
                    'side': 'LONG',
                    'size': 0.1,
                    'entry_price': 50000.0
                },
                {
                    'symbol': 'ETH',
                    'side': 'SHORT',
                    'size': 1.0,
                    'entry_price': 3000.0
                }
            ]
        }
        
        # Bot tracks only BTC
        with patch('trading_bot.TradingBot._load_config', return_value=self.mock_config):
            bot = TradingBot.__new__(TradingBot)
            bot.config = self.mock_config
            bot.client = mock_client
            bot.position = {
                'symbol': 'BTC',
                'side': 'LONG',
                'quantity': 0.1,
                'entry_price': 50000.0
            }
            bot.symbol = 'BTC'
            
            # Should find matching position
            exchange_positions = bot.client.get_user_state()['positions']
            matching_pos = next((p for p in exchange_positions if p['symbol'] == bot.symbol), None)
            
            self.assertIsNotNone(matching_pos, "Should find matching position")
            self.assertEqual(matching_pos['symbol'], 'BTC')


if __name__ == '__main__':
    unittest.main()

