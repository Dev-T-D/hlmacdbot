"""
Performance test for indicator calculations

Tests:
- Large datasets (1000+ candles)
- Measure execution time
"""

import unittest
import time
import pandas as pd
import numpy as np
from macd_strategy import MACDStrategy


class TestIndicatorPerformance(unittest.TestCase):
    """Performance tests for indicator calculations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.strategy = MACDStrategy(
            fast_length=12,
            slow_length=26,
            signal_length=9,
            risk_reward_ratio=2.0
        )
    
    def create_large_dataset(self, length=1000):
        """Create large dataset for performance testing"""
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
    
    def test_large_dataset_1000_candles(self):
        """Test performance with 1000 candles"""
        df = self.create_large_dataset(length=1000)
        
        start_time = time.time()
        df_with_indicators = self.strategy.calculate_indicators(df)
        elapsed_time = time.time() - start_time
        
        self.assertIn('macd', df_with_indicators.columns, "Should calculate MACD")
        self.assertIn('signal', df_with_indicators.columns, "Should calculate signal")
        self.assertIn('histogram', df_with_indicators.columns, "Should calculate histogram")
        
        # Performance assertion: should complete in reasonable time (< 1 second)
        self.assertLess(elapsed_time, 1.0, f"Should complete in < 1 second, took {elapsed_time:.3f}s")
        
        print(f"\n1000 candles: {elapsed_time:.3f} seconds")
    
    def test_large_dataset_5000_candles(self):
        """Test performance with 5000 candles"""
        df = self.create_large_dataset(length=5000)
        
        start_time = time.time()
        df_with_indicators = self.strategy.calculate_indicators(df)
        elapsed_time = time.time() - start_time
        
        self.assertIn('macd', df_with_indicators.columns, "Should calculate MACD")
        
        # Performance assertion: should complete in reasonable time (< 5 seconds)
        self.assertLess(elapsed_time, 5.0, f"Should complete in < 5 seconds, took {elapsed_time:.3f}s")
        
        print(f"\n5000 candles: {elapsed_time:.3f} seconds")
    
    def test_large_dataset_10000_candles(self):
        """Test performance with 10000 candles"""
        df = self.create_large_dataset(length=10000)
        
        start_time = time.time()
        df_with_indicators = self.strategy.calculate_indicators(df)
        elapsed_time = time.time() - start_time
        
        self.assertIn('macd', df_with_indicators.columns, "Should calculate MACD")
        
        # Performance assertion: should complete in reasonable time (< 10 seconds)
        self.assertLess(elapsed_time, 10.0, f"Should complete in < 10 seconds, took {elapsed_time:.3f}s")
        
        print(f"\n10000 candles: {elapsed_time:.3f} seconds")
    
    def test_multiple_calculations(self):
        """Test performance of multiple calculations"""
        df = self.create_large_dataset(length=1000)
        
        # First calculation
        start_time = time.time()
        df1 = self.strategy.calculate_indicators(df)
        first_time = time.time() - start_time
        
        # Second calculation (should be similar)
        start_time = time.time()
        df2 = self.strategy.calculate_indicators(df)
        second_time = time.time() - start_time
        
        # Times should be similar (within 50%)
        time_diff = abs(first_time - second_time) / max(first_time, second_time)
        self.assertLess(time_diff, 0.5, "Multiple calculations should have similar performance")
        
        print(f"\nFirst calculation: {first_time:.3f}s")
        print(f"Second calculation: {second_time:.3f}s")
    
    def test_signal_check_performance(self):
        """Test performance of signal checking"""
        df = self.create_large_dataset(length=1000)
        df = self.strategy.calculate_indicators(df)
        
        start_time = time.time()
        signal = self.strategy.check_entry_signal(df)
        elapsed_time = time.time() - start_time
        
        # Signal check should be very fast (< 0.1 seconds)
        self.assertLess(elapsed_time, 0.1, f"Signal check should be fast, took {elapsed_time:.3f}s")
        
        print(f"\nSignal check: {elapsed_time:.3f} seconds")
    
    def test_incremental_calculation(self):
        """Test performance of incremental calculations"""
        # Calculate for 1000 candles
        df1 = self.create_large_dataset(length=1000)
        start_time = time.time()
        df1 = self.strategy.calculate_indicators(df1)
        time1 = time.time() - start_time
        
        # Calculate for 1001 candles (add one)
        df2 = self.create_large_dataset(length=1001)
        start_time = time.time()
        df2 = self.strategy.calculate_indicators(df2)
        time2 = time.time() - start_time
        
        # Time difference should be small (adding one candle)
        time_diff = abs(time2 - time1)
        self.assertLess(time_diff, 0.1, "Adding one candle should not significantly increase calculation time")
        
        print(f"\n1000 candles: {time1:.3f}s")
        print(f"1001 candles: {time2:.3f}s")
        print(f"Difference: {time_diff:.3f}s")


if __name__ == '__main__':
    unittest.main()

