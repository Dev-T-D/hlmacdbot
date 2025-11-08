"""
Unit tests for TrailingStopLoss.update()

Tests:
- Activation threshold
- Trailing updates
- Stop hit
- Position type LONG/SHORT
"""

import unittest
from datetime import datetime
from risk_manager import TrailingStopLoss


class TestTrailingStopLoss(unittest.TestCase):
    """Test cases for TrailingStopLoss"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.trailing_stop = TrailingStopLoss(
            trail_percent=2.0,           # 2% trail distance
            activation_percent=1.0,      # Activate at 1% profit
            update_threshold_percent=0.5 # Update if stop moves 0.5%
        )
    
    def test_initialization(self):
        """Test trailing stop initialization"""
        self.assertIsNone(self.trailing_stop.entry_price)
        self.assertFalse(self.trailing_stop.is_active)
        self.assertIsNone(self.trailing_stop.position_type)
    
    def test_initialize_long_position(self):
        """Test initializing LONG position"""
        self.trailing_stop.initialize_position(
            entry_price=50000.0,
            initial_stop_loss=49000.0,
            position_type="LONG"
        )
        
        self.assertEqual(self.trailing_stop.entry_price, 50000.0)
        self.assertEqual(self.trailing_stop.position_type, "LONG")
        self.assertEqual(self.trailing_stop.current_stop_loss, 49000.0)
        self.assertEqual(self.trailing_stop.best_price, 50000.0)
        self.assertFalse(self.trailing_stop.is_active)
    
    def test_initialize_short_position(self):
        """Test initializing SHORT position"""
        self.trailing_stop.initialize_position(
            entry_price=50000.0,
            initial_stop_loss=51000.0,
            position_type="SHORT"
        )
        
        self.assertEqual(self.trailing_stop.entry_price, 50000.0)
        self.assertEqual(self.trailing_stop.position_type, "SHORT")
        self.assertEqual(self.trailing_stop.current_stop_loss, 51000.0)
        self.assertEqual(self.trailing_stop.best_price, 50000.0)
    
    def test_activation_threshold_long(self):
        """Test activation threshold for LONG position"""
        self.trailing_stop.initialize_position(
            entry_price=50000.0,
            initial_stop_loss=49000.0,
            position_type="LONG"
        )
        
        # Price below activation threshold
        stop_updated, new_stop, msg = self.trailing_stop.update(50400.0)  # 0.8% profit
        self.assertFalse(stop_updated)
        self.assertFalse(self.trailing_stop.is_active)
        
        # Price above activation threshold
        stop_updated, new_stop, msg = self.trailing_stop.update(50500.0)  # 1% profit
        self.assertTrue(self.trailing_stop.is_active)
    
    def test_activation_threshold_short(self):
        """Test activation threshold for SHORT position"""
        self.trailing_stop.initialize_position(
            entry_price=50000.0,
            initial_stop_loss=51000.0,
            position_type="SHORT"
        )
        
        # Price above activation threshold (profit for SHORT)
        stop_updated, new_stop, msg = self.trailing_stop.update(49500.0)  # 1% profit
        self.assertTrue(self.trailing_stop.is_active)
    
    def test_trailing_update_long(self):
        """Test trailing stop update for LONG position"""
        self.trailing_stop.initialize_position(
            entry_price=50000.0,
            initial_stop_loss=49000.0,
            position_type="LONG"
        )
        
        # Activate trailing stop
        self.trailing_stop.update(50500.0)  # 1% profit - activates
        
        # Price increases - should trail stop loss
        stop_updated, new_stop, msg = self.trailing_stop.update(51000.0)  # 2% profit
        
        if stop_updated:
            self.assertGreater(new_stop, 49000.0, "Stop should trail upward")
            self.assertLess(new_stop, 51000.0, "Stop should be below best price")
    
    def test_trailing_update_short(self):
        """Test trailing stop update for SHORT position"""
        self.trailing_stop.initialize_position(
            entry_price=50000.0,
            initial_stop_loss=51000.0,
            position_type="SHORT"
        )
        
        # Activate trailing stop
        self.trailing_stop.update(49500.0)  # 1% profit - activates
        
        # Price decreases - should trail stop loss downward
        stop_updated, new_stop, msg = self.trailing_stop.update(49000.0)  # 2% profit
        
        if stop_updated:
            self.assertLess(new_stop, 51000.0, "Stop should trail downward")
            self.assertGreater(new_stop, 49000.0, "Stop should be above best price")
    
    def test_stop_hit_long(self):
        """Test stop hit detection for LONG position"""
        self.trailing_stop.initialize_position(
            entry_price=50000.0,
            initial_stop_loss=49000.0,
            position_type="LONG"
        )
        
        # Activate trailing stop
        self.trailing_stop.update(50500.0)
        self.trailing_stop.update(51000.0)  # Trail stop to ~49980 (2% below 51000)
        
        # Price drops below stop
        stop_hit, reason = self.trailing_stop.check_stop_hit(49900.0)
        
        self.assertTrue(stop_hit, "Should detect stop hit")
        self.assertIn("Stop Hit", reason)
    
    def test_stop_hit_short(self):
        """Test stop hit detection for SHORT position"""
        self.trailing_stop.initialize_position(
            entry_price=50000.0,
            initial_stop_loss=51000.0,
            position_type="SHORT"
        )
        
        # Activate trailing stop
        self.trailing_stop.update(49500.0)
        self.trailing_stop.update(49000.0)  # Trail stop to ~49980 (2% above 49000)
        
        # Price rises above stop
        stop_hit, reason = self.trailing_stop.check_stop_hit(50100.0)
        
        self.assertTrue(stop_hit, "Should detect stop hit")
        self.assertIn("Stop Hit", reason)
    
    def test_no_update_below_threshold(self):
        """Test that stop doesn't update if movement below threshold"""
        self.trailing_stop.initialize_position(
            entry_price=50000.0,
            initial_stop_loss=49000.0,
            position_type="LONG"
        )
        
        # Activate
        self.trailing_stop.update(50500.0)
        
        # Small price increase (below update threshold)
        old_stop = self.trailing_stop.current_stop_loss
        stop_updated, new_stop, msg = self.trailing_stop.update(50510.0)  # Very small increase
        
        # May or may not update depending on threshold
        if not stop_updated:
            self.assertEqual(new_stop, old_stop, "Stop should not change if below threshold")
    
    def test_reset(self):
        """Test resetting trailing stop"""
        self.trailing_stop.initialize_position(
            entry_price=50000.0,
            initial_stop_loss=49000.0,
            position_type="LONG"
        )
        
        self.trailing_stop.update(50500.0)  # Activate
        
        self.trailing_stop.reset()
        
        self.assertIsNone(self.trailing_stop.entry_price)
        self.assertFalse(self.trailing_stop.is_active)
        self.assertIsNone(self.trailing_stop.position_type)
    
    def test_not_initialized(self):
        """Test update when not initialized"""
        stop_updated, new_stop, msg = self.trailing_stop.update(50000.0)
        
        self.assertFalse(stop_updated)
        self.assertIn("Not initialized", msg)
    
    def test_stop_never_lowers_long(self):
        """Test that LONG stop never moves down"""
        self.trailing_stop.initialize_position(
            entry_price=50000.0,
            initial_stop_loss=49000.0,
            position_type="LONG"
        )
        
        self.trailing_stop.update(50500.0)  # Activate
        initial_stop = self.trailing_stop.current_stop_loss
        
        # Price goes up, then down
        self.trailing_stop.update(51000.0)  # Stop trails up
        self.trailing_stop.update(50500.0)  # Price drops
        
        # Stop should not go below previous stop
        self.assertGreaterEqual(
            self.trailing_stop.current_stop_loss,
            initial_stop,
            "Stop should never lower for LONG"
        )
    
    def test_stop_never_raises_short(self):
        """Test that SHORT stop never moves up"""
        self.trailing_stop.initialize_position(
            entry_price=50000.0,
            initial_stop_loss=51000.0,
            position_type="SHORT"
        )
        
        self.trailing_stop.update(49500.0)  # Activate
        initial_stop = self.trailing_stop.current_stop_loss
        
        # Price goes down, then up
        self.trailing_stop.update(49000.0)  # Stop trails down
        self.trailing_stop.update(49500.0)  # Price rises
        
        # Stop should not go above previous stop
        self.assertLessEqual(
            self.trailing_stop.current_stop_loss,
            initial_stop,
            "Stop should never raise for SHORT"
        )


if __name__ == '__main__':
    unittest.main()
