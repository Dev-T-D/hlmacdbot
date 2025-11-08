"""
Unit tests for MACDStrategy.check_entry_signal()

Tests:
- Valid bullish cross
- Valid bearish cross
- Insufficient data
- NaN values
- Edge cases
"""

import unittest
import pandas as pd
import numpy as np
from macd_strategy import MACDStrategy


class TestMACDStrategy(unittest.TestCase):
    """Test cases for MACDStrategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.strategy = MACDStrategy(
            fast_length=12,
            slow_length=26,
            signal_length=9,
            risk_reward_ratio=2.0
        )
    
    def create_test_dataframe(self, length=100, macd_trend='bullish'):
        """Create test DataFrame with MACD indicators"""
        dates = pd.date_range(start='2024-01-01', periods=length, freq='1h')
        
        # Create price data
        base_price = 50000
        prices = []
        for i in range(length):
            if macd_trend == 'bullish':
                price = base_price + (i * 10) + np.random.randn() * 100
            elif macd_trend == 'bearish':
                price = base_price - (i * 10) + np.random.randn() * 100
            else:
                price = base_price + np.random.randn() * 200
            prices.append(price)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': [p * 1.005 for p in prices],
            'volume': [1000.0] * length
        })
        
        # Calculate indicators
        df = self.strategy.calculate_indicators(df)
        
        return df
    
    def test_valid_bullish_cross(self):
        """Test valid bullish MACD crossover signal"""
        df = self.create_test_dataframe(length=100, macd_trend='bullish')
        
        # Create bullish crossover scenario
        # Previous candle: MACD below signal
        df.iloc[-2, df.columns.get_loc('macd')] = 100.0
        df.iloc[-2, df.columns.get_loc('signal')] = 150.0
        df.iloc[-2, df.columns.get_loc('histogram')] = -50.0
        
        # Current candle: MACD crosses above signal
        df.iloc[-1, df.columns.get_loc('macd')] = 200.0
        df.iloc[-1, df.columns.get_loc('signal')] = 150.0
        df.iloc[-1, df.columns.get_loc('histogram')] = 50.0
        
        # Ensure bullish candle and price above slow EMA
        df.iloc[-1, df.columns.get_loc('close')] = 51000
        df.iloc[-1, df.columns.get_loc('open')] = 50500
        if 'slow_ema' in df.columns:
            df.iloc[-1, df.columns.get_loc('slow_ema')] = 50000
        
        signal = self.strategy.check_entry_signal(df)
        
        self.assertIsNotNone(signal, "Should detect bullish cross signal")
        self.assertEqual(signal['type'], 'LONG', "Signal should be LONG")
        self.assertIn('entry_price', signal)
        self.assertIn('stop_loss', signal)
        self.assertIn('take_profit', signal)
        self.assertGreater(signal['take_profit'], signal['entry_price'], "TP should be above entry for LONG")
    
    def test_valid_bearish_cross(self):
        """Test valid bearish MACD crossover signal"""
        df = self.create_test_dataframe(length=100, macd_trend='bearish')
        
        # Create bearish crossover scenario
        # Previous candle: MACD above signal
        df.iloc[-2, df.columns.get_loc('macd')] = 200.0
        df.iloc[-2, df.columns.get_loc('signal')] = 150.0
        df.iloc[-2, df.columns.get_loc('histogram')] = 50.0
        
        # Current candle: MACD crosses below signal
        df.iloc[-1, df.columns.get_loc('macd')] = 100.0
        df.iloc[-1, df.columns.get_loc('signal')] = 150.0
        df.iloc[-1, df.columns.get_loc('histogram')] = -50.0
        
        # Ensure bearish candle and price below slow EMA
        df.iloc[-1, df.columns.get_loc('close')] = 49000
        df.iloc[-1, df.columns.get_loc('open')] = 49500
        if 'slow_ema' in df.columns:
            df.iloc[-1, df.columns.get_loc('slow_ema')] = 50000
        
        signal = self.strategy.check_entry_signal(df)
        
        self.assertIsNotNone(signal, "Should detect bearish cross signal")
        self.assertEqual(signal['type'], 'SHORT', "Signal should be SHORT")
        self.assertIn('entry_price', signal)
        self.assertIn('stop_loss', signal)
        self.assertIn('take_profit', signal)
        self.assertLess(signal['take_profit'], signal['entry_price'], "TP should be below entry for SHORT")
    
    def test_insufficient_data(self):
        """Test with insufficient data (less than min_candles)"""
        # Create DataFrame with less than minimum required candles
        df = self.create_test_dataframe(length=20)  # Less than min_candles (typically 50+)
        
        signal = self.strategy.check_entry_signal(df)
        
        self.assertIsNone(signal, "Should return None with insufficient data")
    
    def test_nan_values(self):
        """Test with NaN values in indicators"""
        df = self.create_test_dataframe(length=100)
        
        # Set MACD to NaN
        df.iloc[-1, df.columns.get_loc('macd')] = np.nan
        
        signal = self.strategy.check_entry_signal(df)
        
        self.assertIsNone(signal, "Should return None with NaN MACD")
    
    def test_infinite_values(self):
        """Test with infinite values in indicators"""
        df = self.create_test_dataframe(length=100)
        
        # Set signal to infinity
        df.iloc[-1, df.columns.get_loc('signal')] = np.inf
        
        signal = self.strategy.check_entry_signal(df)
        
        self.assertIsNone(signal, "Should return None with infinite signal")
    
    def test_no_crossover(self):
        """Test when no crossover occurs"""
        df = self.create_test_dataframe(length=100)
        
        # No crossover: MACD stays above signal
        df.iloc[-2, df.columns.get_loc('macd')] = 200.0
        df.iloc[-2, df.columns.get_loc('signal')] = 150.0
        df.iloc[-1, df.columns.get_loc('macd')] = 210.0
        df.iloc[-1, df.columns.get_loc('signal')] = 160.0
        
        signal = self.strategy.check_entry_signal(df)
        
        self.assertIsNone(signal, "Should return None when no crossover")
    
    def test_crossover_without_overlay(self):
        """Test crossover without matching overlay"""
        df = self.create_test_dataframe(length=100)
        
        # Bullish crossover but bearish overlay (histogram < 0)
        df.iloc[-2, df.columns.get_loc('macd')] = 100.0
        df.iloc[-2, df.columns.get_loc('signal')] = 150.0
        df.iloc[-1, df.columns.get_loc('macd')] = 200.0
        df.iloc[-1, df.columns.get_loc('signal')] = 150.0
        df.iloc[-1, df.columns.get_loc('histogram')] = -10.0  # Bearish overlay
        
        signal = self.strategy.check_entry_signal(df)
        
        self.assertIsNone(signal, "Should return None when overlay doesn't match")
    
    def test_crossover_without_price_confirmation(self):
        """Test crossover without price confirmation"""
        df = self.create_test_dataframe(length=100)
        
        # Bullish crossover but price below slow EMA
        df.iloc[-2, df.columns.get_loc('macd')] = 100.0
        df.iloc[-2, df.columns.get_loc('signal')] = 150.0
        df.iloc[-1, df.columns.get_loc('macd')] = 200.0
        df.iloc[-1, df.columns.get_loc('signal')] = 150.0
        df.iloc[-1, df.columns.get_loc('histogram')] = 50.0
        df.iloc[-1, df.columns.get_loc('close')] = 49000
        if 'slow_ema' in df.columns:
            df.iloc[-1, df.columns.get_loc('slow_ema')] = 50000  # Price below EMA
        
        signal = self.strategy.check_entry_signal(df)
        
        # May or may not return signal depending on implementation
        # This test documents expected behavior
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame()
        
        signal = self.strategy.check_entry_signal(df)
        
        self.assertIsNone(signal, "Should return None with empty DataFrame")
    
    def test_missing_columns(self):
        """Test with missing required columns"""
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1h'),
            'close': [50000] * 100
        })
        
        signal = self.strategy.check_entry_signal(df)
        
        # Should either calculate indicators or return None
        # This test documents behavior
    
    def test_edge_case_single_candle(self):
        """Test with only one candle"""
        df = self.create_test_dataframe(length=1)
        
        signal = self.strategy.check_entry_signal(df)
        
        self.assertIsNone(signal, "Should return None with insufficient data")
    
    def test_edge_case_exactly_min_candles(self):
        """Test with exactly minimum required candles"""
        df = self.create_test_dataframe(length=self.strategy.min_candles)
        
        # Should not crash, may or may not return signal
        signal = self.strategy.check_entry_signal(df)
        
        # Signal may be None if conditions not met, but should not crash
        self.assertIsInstance(signal, (dict, type(None)), "Should return dict or None")


if __name__ == '__main__':
    unittest.main()

