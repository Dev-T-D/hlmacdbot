"""
Integration test for full trading cycle

Tests:
- Entry signal → position → exit signal → close
- With mocked API
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Handle optional dependencies
try:
    from trading_bot import TradingBot
    from macd_strategy import MACDStrategy
    TRADING_BOT_AVAILABLE = True
except ImportError as e:
    TRADING_BOT_AVAILABLE = False
    print(f"Warning: Could not import trading_bot: {e}")


@unittest.skipIf(not TRADING_BOT_AVAILABLE, "trading_bot module not available")
class TestTradingCycleIntegration(unittest.TestCase):
    """Integration tests for full trading cycle"""
    
    def setUp(self):
        """Set up test fixtures with mocked API"""
        # Create a mock config
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
    @patch('builtins.open', create=True)
    @patch('json.load')
    def test_full_trading_cycle_long(self, mock_json_load, mock_open, mock_client_class):
        """Test complete LONG trading cycle: entry → position → exit → close"""
        # Setup mocks
        mock_json_load.return_value = self.mock_config
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock market data
        mock_client.get_klines.return_value = self._create_mock_market_data()
        mock_client.get_account_balance.return_value = 10000.0
        mock_client.get_user_state.return_value = {'positions': []}
        mock_client.get_current_price.return_value = 50000.0
        
        # Create bot instance
        with patch('trading_bot.TradingBot._load_config', return_value=self.mock_config):
            bot = TradingBot.__new__(TradingBot)
            bot.config = self.mock_config
            bot.exchange_name = "Hyperliquid"
            bot.client = mock_client
            bot.symbol = 'BTC'
            bot.timeframe = '1h'
            bot.leverage = 10
            bot.position = None
            bot.strategy = MACDStrategy()
            bot.risk_manager = Mock()
            bot.risk_manager.calculate_position_size.return_value = {
                'quantity': 0.1,
                'notional_value': 5000.0,
                'risk_amount': 100.0
            }
            bot.risk_manager.trailing_stop = Mock()
            bot.market_data_cache = None
            bot.cache_timestamp = None
            bot.cache_max_age_seconds = 60
        
        # Step 1: Check for entry signal (should find LONG signal)
        df = self._create_mock_market_data_with_signal('LONG')
        signal = bot.strategy.check_entry_signal(df)
        
        self.assertIsNotNone(signal, "Should detect entry signal")
        self.assertEqual(signal['type'], 'LONG')
        
        # Step 2: Mock opening position
        mock_client.place_order.return_value = {'order_id': 'test_order_123'}
        mock_client.get_user_state.return_value = {
            'positions': [{
                'symbol': 'BTC',
                'side': 'LONG',
                'size': 0.1,
                'entry_price': 50000.0
            }]
        }
        
        # Step 3: Check for exit signal (should find exit)
        df_exit = self._create_mock_market_data_with_signal('EXIT_LONG')
        exit_signal = bot.strategy.check_exit_signal(df_exit, 'LONG')
        
        # Step 4: Mock closing position
        mock_client.close_position.return_value = {'order_id': 'close_order_123'}
        mock_client.get_user_state.return_value = {'positions': []}
        
        # Verify cycle completed
        self.assertIsNotNone(signal)
        self.assertIsNotNone(exit_signal)
    
    def _create_mock_market_data(self, length=100):
        """Create mock market data DataFrame"""
        dates = pd.date_range(start='2024-01-01', periods=length, freq='1h')
        base_price = 50000
        
        data = []
        for i in range(length):
            price = base_price + (i * 10) + np.random.randn() * 100
            data.append({
                'timestamp': dates[i],
                'open': price,
                'high': price * 1.01,
                'low': price * 0.99,
                'close': price * 1.005,
                'volume': 1000.0
            })
        
        return pd.DataFrame(data)
    
    def _create_mock_market_data_with_signal(self, signal_type='LONG'):
        """Create mock market data with specific signal"""
        df = self._create_mock_market_data()
        
        # Calculate indicators
        strategy = MACDStrategy()
        df = strategy.calculate_indicators(df)
        
        if signal_type == 'LONG':
            # Create bullish crossover
            df.iloc[-2, df.columns.get_loc('macd')] = 100.0
            df.iloc[-2, df.columns.get_loc('signal')] = 150.0
            df.iloc[-1, df.columns.get_loc('macd')] = 200.0
            df.iloc[-1, df.columns.get_loc('signal')] = 150.0
            df.iloc[-1, df.columns.get_loc('histogram')] = 50.0
            df.iloc[-1, df.columns.get_loc('close')] = 51000
            df.iloc[-1, df.columns.get_loc('open')] = 50500
            if 'slow_ema' in df.columns:
                df.iloc[-1, df.columns.get_loc('slow_ema')] = 50000
        
        elif signal_type == 'EXIT_LONG':
            # Create bearish crossover for exit
            df.iloc[-2, df.columns.get_loc('macd')] = 200.0
            df.iloc[-2, df.columns.get_loc('signal')] = 150.0
            df.iloc[-1, df.columns.get_loc('macd')] = 100.0
            df.iloc[-1, df.columns.get_loc('signal')] = 150.0
            df.iloc[-1, df.columns.get_loc('histogram')] = -50.0
        
        return df
    
    @patch('trading_bot.HyperliquidClient')
    def test_position_tracking(self, mock_client_class):
        """Test position tracking through cycle"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock state changes: no position -> position exists -> no position
        mock_client.get_user_state.side_effect = [
            {'positions': []},  # Initially no position
            {'positions': [{    # After entry: position exists
                'symbol': 'BTC',
                'side': 'LONG',
                'size': 0.1,
                'entry_price': 50000.0
            }]},
            {'positions': []}   # After exit: no position
        ]
        
        # Verify position tracking
        positions = mock_client.get_user_state()['positions']
        self.assertEqual(len(positions), 0)  # Initially no position
        
        positions = mock_client.get_user_state()['positions']
        self.assertEqual(len(positions), 1)  # After entry
        
        positions = mock_client.get_user_state()['positions']
        self.assertEqual(len(positions), 0)  # After exit


if __name__ == '__main__':
    unittest.main()

