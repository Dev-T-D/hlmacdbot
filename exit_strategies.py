"""
Advanced Exit Strategies for Trading Bot

Implements partial profit taking, volatility-adaptive trailing stops,
and time-based exits to maximize profit capture.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PartialProfitTaker:
    """Take partial profits at multiple levels to lock in gains"""

    def __init__(self, levels=[0.5, 1.0, 1.5, 2.0]):
        """
        Args:
            levels: R-multiple levels to take partial profits (0.5R, 1R, 1.5R, 2R)
        """
        self.levels = sorted(levels)
        self.position_reductions = {}  # Track how much we've closed at each level

    def calculate_profit_levels(self, entry_price, stop_loss, side, initial_quantity):
        """
        Calculate specific price levels for partial profit taking

        Args:
            entry_price: Entry price of position
            stop_loss: Stop loss price
            side: 'LONG' or 'SHORT'
            initial_quantity: Initial position size

        Returns:
            List of dicts with {price, quantity_to_close, r_multiple}
        """
        # Calculate 1R (risk amount)
        risk_amount = abs(entry_price - stop_loss)

        profit_levels = []

        # Calculate quantity to close at each level
        # Strategy: Close 25% at 0.5R, 25% at 1R, 25% at 1.5R, let 25% run to 2R+
        quantities = {
            0.5: 0.25,  # 25% at 0.5R
            1.0: 0.25,  # 25% at 1R
            1.5: 0.25,  # 25% at 1.5R
            2.0: 0.25   # Final 25% at 2R (or trailing stop)
        }

        for r_multiple in self.levels:
            # Calculate target price for this R-multiple
            if side == 'LONG':
                target_price = entry_price + (risk_amount * r_multiple)
            else:  # SHORT
                target_price = entry_price - (risk_amount * r_multiple)

            # Quantity to close at this level
            close_pct = quantities.get(r_multiple, 0)
            quantity_to_close = initial_quantity * close_pct

            profit_levels.append({
                'r_multiple': r_multiple,
                'price': target_price,
                'quantity': quantity_to_close,
                'percentage': close_pct * 100
            })

        return profit_levels

    def check_profit_levels(self, position, current_price):
        """
        Check if any profit levels have been hit

        Args:
            position: Current position dict
            current_price: Current market price

        Returns:
            List of actions to take: [{level, quantity_to_close, price}]
        """
        actions = []

        entry_price = position['entry_price']
        side = position['type']
        position_id = position.get('id', 'default')

        # Get or initialize tracking for this position
        if position_id not in self.position_reductions:
            self.position_reductions[position_id] = set()

        closed_levels = self.position_reductions[position_id]

        # Get profit levels for this position
        profit_levels = self.calculate_profit_levels(
            entry_price,
            position['stop_loss'],
            side,
            position['original_quantity']
        )

        for level in profit_levels:
            r_multiple = level['r_multiple']
            target_price = level['price']

            # Skip if already closed at this level
            if r_multiple in closed_levels:
                continue

            # Check if price has reached this level
            level_hit = False
            if side == 'LONG' and current_price >= target_price:
                level_hit = True
            elif side == 'SHORT' and current_price <= target_price:
                level_hit = True

            if level_hit:
                actions.append({
                    'r_multiple': r_multiple,
                    'quantity': level['quantity'],
                    'price': current_price,
                    'percentage': level['percentage']
                })

                # Mark level as closed
                closed_levels.add(r_multiple)

        return actions

    def reset_position(self, position_id):
        """Reset tracking when position fully closed"""
        if position_id in self.position_reductions:
            del self.position_reductions[position_id]


class VolatilityAdaptiveTrailingStop:
    """Trailing stop that adapts to current market volatility"""

    def __init__(self, atr_multiplier=2.0, min_trail_distance=0.5):
        """
        Args:
            atr_multiplier: ATR multiplier for trail distance
            min_trail_distance: Minimum trail distance as % of entry
        """
        self.atr_multiplier = atr_multiplier
        self.min_trail_distance = min_trail_distance
        self.highest_price = {}  # Track highest price per position

    def calculate_trail_distance(self, df, current_price):
        """
        Calculate optimal trailing stop distance based on volatility

        Args:
            df: DataFrame with market data including ATR
            current_price: Current market price

        Returns:
            Trail distance in price units
        """
        # Get current ATR
        current_atr = df['atr_14'].iloc[-1]

        # Calculate ATR-based distance
        atr_distance = current_atr * self.atr_multiplier

        # Calculate minimum distance
        min_distance = current_price * (self.min_trail_distance / 100)

        # Use larger of the two
        trail_distance = max(atr_distance, min_distance)

        return trail_distance

    def update_trailing_stop(self, position, current_price, df):
        """
        Update trailing stop based on current price and volatility

        Args:
            position: Position dict
            current_price: Current market price
            df: Market data DataFrame

        Returns:
            New stop loss price
        """
        position_id = position.get('id', 'default')
        side = position['type']
        current_stop = position['stop_loss']

        # Track highest/lowest price
        if position_id not in self.highest_price:
            self.highest_price[position_id] = current_price

        if side == 'LONG':
            # Update highest price
            if current_price > self.highest_price[position_id]:
                self.highest_price[position_id] = current_price

            # Calculate new stop based on highest price
            trail_distance = self.calculate_trail_distance(df, current_price)
            new_stop = self.highest_price[position_id] - trail_distance

            # Only move stop up, never down
            if new_stop > current_stop:
                return new_stop

        else:  # SHORT
            # Update lowest price
            if current_price < self.highest_price[position_id]:
                self.highest_price[position_id] = current_price

            # Calculate new stop based on lowest price
            trail_distance = self.calculate_trail_distance(df, current_price)
            new_stop = self.highest_price[position_id] + trail_distance

            # Only move stop down, never up
            if new_stop < current_stop:
                return new_stop

        # No change
        return current_stop

    def reset_position(self, position_id):
        """Reset tracking when position closed"""
        if position_id in self.highest_price:
            del self.highest_price[position_id]


class TimeBasedExit:
    """Exit positions after maximum holding time or at specific times"""

    def __init__(self, max_hold_hours=24, avoid_funding_exits=True):
        """
        Args:
            max_hold_hours: Maximum time to hold position (hours)
            avoid_funding_exits: Avoid exiting near funding times
        """
        self.max_hold_hours = max_hold_hours
        self.avoid_funding_exits = avoid_funding_exits

        # Hyperliquid funding times (UTC): 00:00, 08:00, 16:00
        self.funding_hours = [0, 8, 16]
        self.funding_window_minutes = 30  # Avoid 30min before/after funding

    def should_exit_by_time(self, position, current_time):
        """
        Check if position should be exited based on time

        Args:
            position: Position dict with entry_time
            current_time: Current datetime

        Returns:
            (should_exit: bool, reason: str)
        """
        entry_time = position['entry_time']
        holding_time = (current_time - entry_time).total_seconds() / 3600  # Hours

        # CHECK 1: Maximum holding time exceeded
        if holding_time >= self.max_hold_hours:
            return True, f"Max holding time exceeded ({holding_time:.1f}h)"

        # CHECK 2: End of trading day (if applicable)
        # For crypto 24/7, skip this check

        # CHECK 3: Approaching funding time with small profit
        if self.avoid_funding_exits:
            minutes_to_funding = self._minutes_to_next_funding(current_time)

            # If within funding window and position is profitable, hold through funding
            unrealized_pnl = position.get('unrealized_pnl', 0)

            if minutes_to_funding < self.funding_window_minutes and unrealized_pnl > 0:
                # Hold through funding to collect payment
                return False, "Holding through funding (profitable position)"

        return False, "Time-based exit not triggered"

    def _minutes_to_next_funding(self, current_time):
        """Calculate minutes until next funding time"""
        current_hour = current_time.hour
        current_minute = current_time.minute

        # Find next funding hour
        next_funding_hour = None
        for funding_hour in self.funding_hours:
            if funding_hour > current_hour:
                next_funding_hour = funding_hour
                break

        if next_funding_hour is None:
            next_funding_hour = self.funding_hours[0] + 24  # Next day

        # Calculate minutes
        hours_diff = next_funding_hour - current_hour
        minutes_to_funding = (hours_diff * 60) - current_minute

        return minutes_to_funding

    def get_optimal_exit_time(self, position, current_time):
        """
        Suggest optimal exit time considering funding and holding period

        Returns:
            datetime of optimal exit time
        """
        # Exit after funding if position is old
        minutes_to_funding = self._minutes_to_next_funding(current_time)

        if minutes_to_funding < 60:  # Less than 1 hour to funding
            # Wait until after funding
            next_funding = current_time + timedelta(minutes=minutes_to_funding + 5)
            return next_funding

        # Otherwise exit ASAP
        return current_time
