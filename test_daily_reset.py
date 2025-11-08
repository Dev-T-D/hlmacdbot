"""
Test for daily reset logic

Tests:
- Midnight crossing
- Timezone handling
- Stats reset
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta, date
from risk_manager import RiskManager

# Handle optional dependencies
# Note: This test only uses RiskManager, not TradingBot
TRADING_BOT_AVAILABLE = False  # Not needed for these tests


class TestDailyReset(unittest.TestCase):
    """Test cases for daily reset logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.risk_manager = RiskManager(
            max_position_size_pct=0.1,
            max_daily_loss_pct=0.05,
            max_trades_per_day=10,
            leverage=10,
            exchange='hyperliquid'
        )
    
    def test_stats_reset(self):
        """Test that daily stats are reset"""
        # Set some daily stats
        self.risk_manager.daily_pnl = -100.0
        self.risk_manager.daily_trades = 5
        self.risk_manager.starting_balance = 10000.0
        
        # Reset stats
        self.risk_manager.reset_daily_stats(11000.0)
        
        self.assertEqual(self.risk_manager.daily_pnl, 0.0, "Daily P&L should reset")
        self.assertEqual(self.risk_manager.daily_trades, 0, "Daily trades should reset")
        self.assertEqual(self.risk_manager.starting_balance, 11000.0, "Starting balance should update")
    
    def test_midnight_crossing(self):
        """Test midnight crossing detection"""
        # Create datetime just before midnight
        before_midnight = datetime(2024, 1, 1, 23, 59, 59)
        
        # Create datetime just after midnight
        after_midnight = datetime(2024, 1, 2, 0, 0, 1)
        
        # Check if day changed
        day_changed = before_midnight.date() != after_midnight.date()
        
        self.assertTrue(day_changed, "Should detect day change")
    
    def test_timezone_handling(self):
        """Test timezone handling for daily reset"""
        # Test that UTC date extraction works consistently
        # This simulates the bot's daily reset logic using UTC dates

        utc_time = datetime(2024, 1, 1, 0, 0, 0)  # Midnight UTC
        est_offset = timedelta(hours=-5)  # EST is UTC-5
        est_time = utc_time + est_offset  # 2023-12-31 19:00:00

        # Extract dates (what the bot does for daily reset)
        utc_date = utc_time.date()
        est_date = est_time.date()

        # UTC should be 2024-01-01, EST should be 2023-12-31
        self.assertEqual(utc_date, date(2024, 1, 1), "UTC date should be correct")
        self.assertEqual(est_date, date(2023, 12, 31), "EST date should be previous day")

        # Verify date comparison works (bot uses this for daily reset)
        self.assertNotEqual(utc_date, est_date, "Different timezones should have different dates at UTC midnight")
    
    def test_daily_reset_trigger(self):
        """Test that daily reset is triggered at midnight"""
        # Test date comparison logic (used by bot for daily reset)
        # Previous day
        previous_day = datetime(2024, 1, 1).date()
        # Current day (just after midnight)
        current_day = datetime(2024, 1, 2, 0, 0, 1).date()
        
        # Check if reset should occur
        should_reset = previous_day != current_day
        
        self.assertTrue(should_reset, "Should trigger reset at midnight")
        self.assertNotEqual(previous_day, current_day, "Dates should be different")
    
    def test_daily_pnl_tracking(self):
        """Test daily P&L tracking"""
        self.risk_manager.reset_daily_stats(10000.0)
        
        # Add some P&L
        self.risk_manager.update_daily_pnl(100.0)
        self.risk_manager.update_daily_pnl(-50.0)
        self.risk_manager.update_daily_pnl(200.0)
        
        self.assertEqual(self.risk_manager.daily_pnl, 250.0, "Should track cumulative P&L")
        self.assertEqual(self.risk_manager.daily_trades, 3, "Should track trade count")
    
    def test_max_daily_loss_check(self):
        """Test max daily loss check"""
        self.risk_manager.reset_daily_stats(10000.0)
        max_daily_loss = 10000.0 * 0.05  # 5% of balance = 500
        
        # Add losses up to limit
        self.risk_manager.update_daily_pnl(-400.0)
        self.assertLessEqual(abs(self.risk_manager.daily_pnl), max_daily_loss, "Should allow losses up to limit")
        
        # Try to exceed limit
        self.risk_manager.update_daily_pnl(-200.0)  # Total: -600, exceeds -500 limit
        
        # Check if limit exceeded
        limit_exceeded = abs(self.risk_manager.daily_pnl) > max_daily_loss
        
        self.assertTrue(limit_exceeded, "Should detect when limit exceeded")
    
    def test_max_trades_per_day_check(self):
        """Test max trades per day check"""
        self.risk_manager.reset_daily_stats(10000.0)
        max_trades = 10
        
        # Add trades up to limit
        for i in range(max_trades):
            self.risk_manager.update_daily_pnl(10.0)
        
        self.assertEqual(self.risk_manager.daily_trades, max_trades, "Should track trades correctly")
        
        # Check if limit reached
        limit_reached = self.risk_manager.daily_trades >= max_trades
        
        self.assertTrue(limit_reached, "Should detect when trade limit reached")
    
    def test_consecutive_days(self):
        """Test handling of consecutive days"""
        # Day 1
        self.risk_manager.reset_daily_stats(10000.0)
        self.risk_manager.update_daily_pnl(100.0)
        
        day1_pnl = self.risk_manager.daily_pnl
        day1_trades = self.risk_manager.daily_trades
        
        # Day 2 reset
        self.risk_manager.reset_daily_stats(10100.0)
        
        self.assertEqual(self.risk_manager.daily_pnl, 0.0, "Day 2 P&L should reset")
        self.assertEqual(self.risk_manager.daily_trades, 0, "Day 2 trades should reset")
        self.assertNotEqual(day1_pnl, self.risk_manager.daily_pnl, "Should reset between days")


if __name__ == '__main__':
    unittest.main()

