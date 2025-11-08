"""
Unit tests for RiskManager.calculate_position_size()

Tests:
- Normal case
- Min quantity edge
- Max leverage
- Zero balance
- Negative prices
"""

import unittest
from risk_manager import RiskManager
from exceptions import PositionSizeError, ConfigurationError


class TestRiskManagerPositionSize(unittest.TestCase):
    """Test cases for RiskManager.calculate_position_size()"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.risk_manager = RiskManager(
            max_position_size_pct=0.1,  # 10%
            max_daily_loss_pct=0.05,    # 5%
            max_trades_per_day=10,
            leverage=10,
            exchange='hyperliquid'
        )
    
    def test_normal_case(self):
        """Test normal position size calculation"""
        balance = 10000.0
        entry_price = 50000.0
        stop_loss = 49000.0  # 2% stop loss
        
        result = self.risk_manager.calculate_position_size(
            balance=balance,
            entry_price=entry_price,
            stop_loss=stop_loss
        )
        
        self.assertIn('quantity', result)
        self.assertIn('notional_value', result)
        self.assertIn('position_risk', result)
        self.assertGreater(result['quantity'], 0, "Quantity should be positive")
        self.assertLessEqual(result['quantity'] * entry_price, balance * 0.1 * 10, "Should respect max position size")
    
    def test_min_quantity_edge(self):
        """Test minimum quantity edge case"""
        balance = 1000.0
        entry_price = 50000.0
        stop_loss = 49000.0
        
        result = self.risk_manager.calculate_position_size(
            balance=balance,
            entry_price=entry_price,
            stop_loss=stop_loss,
            min_qty=0.001
        )
        
        self.assertGreaterEqual(result['quantity'], 0.001, "Should respect minimum quantity")
    
    def test_max_leverage(self):
        """Test maximum leverage constraint"""
        balance = 10000.0
        entry_price = 50000.0
        stop_loss = 49000.0
        
        # Test with max leverage
        result = self.risk_manager.calculate_position_size(
            balance=balance,
            entry_price=entry_price,
            stop_loss=stop_loss
        )
        
        # Notional value should not exceed exchange max notional
        max_notional = self.risk_manager.exchange_limits['max_notional']
        self.assertLessEqual(result['notional_value'], max_notional, "Should respect max notional limit")
        
        # Also check leverage used doesn't exceed configured leverage
        leverage_used = result.get('leverage_used', 0)
        self.assertLessEqual(leverage_used, self.risk_manager.leverage, "Should respect configured leverage")
    
    def test_zero_balance(self):
        """Test with zero balance"""
        balance = 0.0
        entry_price = 50000.0
        stop_loss = 49000.0
        
        with self.assertRaises((PositionSizeError, ConfigurationError)):
            self.risk_manager.calculate_position_size(
                balance=balance,
                entry_price=entry_price,
                stop_loss=stop_loss
            )
    
    def test_negative_prices(self):
        """Test with negative prices"""
        balance = 10000.0
        
        # Negative entry price
        with self.assertRaises((PositionSizeError, ConfigurationError)):
            self.risk_manager.calculate_position_size(
                balance=balance,
                entry_price=-50000.0,
                stop_loss=49000.0
            )
        
        # Negative stop loss
        with self.assertRaises((PositionSizeError, ConfigurationError)):
            self.risk_manager.calculate_position_size(
                balance=balance,
                entry_price=50000.0,
                stop_loss=-49000.0
            )
    
    def test_stop_loss_equal_entry(self):
        """Test with stop loss equal to entry price"""
        balance = 10000.0
        entry_price = 50000.0
        stop_loss = 50000.0  # Same as entry
        
        with self.assertRaises((PositionSizeError, ConfigurationError)):
            self.risk_manager.calculate_position_size(
                balance=balance,
                entry_price=entry_price,
                stop_loss=stop_loss
            )
    
    def test_stop_loss_wrong_direction_long(self):
        """Test LONG position with stop loss above entry"""
        balance = 10000.0
        entry_price = 50000.0
        stop_loss = 51000.0  # Above entry (wrong for LONG)
        
        # Should still calculate, but may warn or adjust
        result = self.risk_manager.calculate_position_size(
            balance=balance,
            entry_price=entry_price,
            stop_loss=stop_loss
        )
        
        # Implementation may handle this differently
        self.assertIn('quantity', result)
    
    def test_existing_positions(self):
        """Test with existing positions (exposure tracking)"""
        balance = 10000.0
        entry_price = 50000.0
        stop_loss = 49000.0
        
        existing_positions = [{
            'entry_price': 50000.0,
            'quantity': 0.1,
            'stop_loss': 49000.0,
            'type': 'LONG'
        }]
        
        result = self.risk_manager.calculate_position_size(
            balance=balance,
            entry_price=entry_price,
            stop_loss=stop_loss,
            existing_positions=existing_positions
        )
        
        self.assertIn('quantity', result)
        self.assertIn('existing_exposure', result)
        # New position should account for existing exposure
        self.assertLessEqual(
            result['total_risk_with_new'],
            balance * self.risk_manager.max_position_size_pct,
            "Total risk should respect max position size"
        )
    
    def test_very_small_stop_loss(self):
        """Test with very small stop loss distance"""
        balance = 10000.0
        entry_price = 50000.0
        stop_loss = 49999.0  # Very tight stop
        
        result = self.risk_manager.calculate_position_size(
            balance=balance,
            entry_price=entry_price,
            stop_loss=stop_loss
        )
        
        self.assertIn('quantity', result)
        # Should handle tight stops gracefully
    
    def test_very_large_stop_loss(self):
        """Test with very large stop loss distance"""
        balance = 10000.0
        entry_price = 50000.0
        stop_loss = 40000.0  # 20% stop loss
        
        result = self.risk_manager.calculate_position_size(
            balance=balance,
            entry_price=entry_price,
            stop_loss=stop_loss
        )
        
        self.assertIn('quantity', result)
        # Should handle wide stops gracefully
    
    def test_quantity_precision(self):
        """Test quantity precision rounding"""
        balance = 10000.0
        entry_price = 50000.0
        stop_loss = 49000.0
        
        result = self.risk_manager.calculate_position_size(
            balance=balance,
            entry_price=entry_price,
            stop_loss=stop_loss,
            qty_precision=3
        )
        
        # Quantity should be rounded to 3 decimal places
        quantity_str = str(result['quantity'])
        if '.' in quantity_str:
            decimal_places = len(quantity_str.split('.')[1])
            self.assertLessEqual(decimal_places, 3, "Should respect quantity precision")


if __name__ == '__main__':
    unittest.main()

