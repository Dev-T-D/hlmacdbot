"""
Order Flow Analyzer for Market Microstructure Analysis

This module provides sophisticated market microstructure analysis for the Hyperliquid trading bot,
including order book imbalance detection, trade flow analysis, volume profile construction,
and liquidity analysis to identify high-probability trading opportunities.

Key Features:
- Order book imbalance detection with dynamic thresholds
- Real-time trade flow classification and momentum analysis
- Volume profile analysis with POC and value areas
- Liquidity heatmap and support/resistance detection
- Spoofing detection and hidden liquidity identification
- Execution quality monitoring and market impact assessment
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from input_sanitizer import InputSanitizer

logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    """Represents a snapshot of the order book at a specific timestamp."""
    timestamp: datetime
    bids: List[Tuple[float, float]]  # [(price, quantity), ...]
    asks: List[Tuple[float, float]]  # [(price, quantity), ...]
    mid_price: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.bids and self.asks:
            best_bid = max(p for p, _ in self.bids)
            best_ask = min(p for p, _ in self.asks)
            self.mid_price = (best_bid + best_ask) / 2
            self.spread = best_ask - best_bid
            self.spread_bps = (self.spread / self.mid_price) * 10000  # Basis points


@dataclass
class TradeFlowMetrics:
    """Real-time trade flow analysis metrics."""
    timestamp: datetime
    net_flow_1m: float = 0.0
    net_flow_5m: float = 0.0
    net_flow_15m: float = 0.0
    flow_momentum: float = 0.0
    large_trades_count: int = 0
    aggression_ratio: float = 0.0
    buy_pressure: float = 0.0
    sell_pressure: float = 0.0


@dataclass
class VolumeProfile:
    """Volume profile analysis with POC and value areas."""
    timestamp: datetime
    price_levels: Dict[float, float] = field(default_factory=dict)
    poc: float = 0.0  # Point of Control
    vah: float = 0.0  # Value Area High
    val: float = 0.0  # Value Area Low
    total_volume: float = 0.0
    value_area_volume: float = 0.0


@dataclass
class LiquidityHeatmap:
    """Liquidity concentration analysis across price levels."""
    timestamp: datetime
    bid_liquidity: Dict[float, float] = field(default_factory=dict)
    ask_liquidity: Dict[float, float] = field(default_factory=dict)
    total_bid_volume: float = 0.0
    total_ask_volume: float = 0.0
    imbalance_ratio: float = 0.0
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)


@dataclass
class ExecutionQuality:
    """Order execution quality metrics."""
    order_id: str
    symbol: str
    side: str
    quantity: float
    target_price: float
    executed_price: float
    slippage_bps: float
    market_impact_bps: float
    execution_time_ms: int
    arrival_price: float
    realized_spread: float
    implementation_shortfall: float


class OrderFlowAnalyzer:
    """
    Comprehensive market microstructure analyzer for order flow insights.

    This class provides real-time analysis of:
    - Order book imbalance and depth
    - Trade flow classification and momentum
    - Volume profile construction
    - Liquidity concentration analysis
    - Spoofing detection
    - Hidden liquidity identification

    All analysis is performed with input validation and error handling
    for production reliability.
    """

    def __init__(self, symbol: str, config: Optional[Dict] = None):
        """
        Initialize the order flow analyzer.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            config: Configuration dictionary with analysis parameters
        """
        self.symbol = InputSanitizer.sanitize_symbol(symbol)
        self.config = config or self._get_default_config()

        # Order book analysis
        self.current_orderbook: Optional[OrderBookSnapshot] = None
        self.orderbook_history: deque = deque(maxlen=1000)  # Keep last 1000 snapshots

        # Trade flow analysis
        self.trade_history: deque = deque(maxlen=10000)  # Keep last 10k trades
        self.trade_flow_metrics: Optional[TradeFlowMetrics] = None
        self.rolling_trade_flow: Dict[str, deque] = {
            '1m': deque(maxlen=60),   # 1 minute rolling
            '5m': deque(maxlen=300),  # 5 minute rolling
            '15m': deque(maxlen=900), # 15 minute rolling
        }

        # Volume profile
        self.volume_profile: Optional[VolumeProfile] = None
        self.price_volume_map: Dict[float, float] = defaultdict(float)
        self.session_start_time: Optional[datetime] = None

        # Liquidity analysis
        self.liquidity_heatmap: Optional[LiquidityHeatmap] = None

        # Execution quality tracking
        self.execution_history: List[ExecutionQuality] = []
        self.max_execution_history = 1000

        # Analysis parameters
        self.imbalance_threshold = self.config.get('imbalance_threshold', 0.3)
        self.depth_levels = self.config.get('depth_levels', 10)
        self.large_trade_multiplier = self.config.get('large_trade_multiplier', 2.0)
        self.volume_profile_bins = self.config.get('volume_profile_bins', 50)

        logger.info(f"OrderFlowAnalyzer initialized for {self.symbol}")

    def _get_default_config(self) -> Dict:
        """Get default configuration parameters."""
        return {
            'imbalance_threshold': 0.3,
            'depth_levels': 10,
            'large_trade_multiplier': 2.0,
            'volume_profile_bins': 50,
            'min_order_size': 0.001,
            'max_spread_bps': 50,  # Maximum spread for trading
            'value_area_percentage': 0.7,  # 70% of volume in value area
        }

    # ==========================================
    # ORDER BOOK ANALYSIS
    # ==========================================

    def update_orderbook(self, bids: List[List[Union[str, float]]],
                        asks: List[List[Union[str, float]]]) -> OrderBookSnapshot:
        """
        Update order book data and perform imbalance analysis.

        Args:
            bids: List of [price, quantity] pairs for bid side
            asks: List of [price, quantity] pairs for ask side

        Returns:
            OrderBookSnapshot with calculated metrics
        """
        try:
            # Convert and validate data
            bids_clean = [(float(price), float(qty)) for price, qty in bids[:self.depth_levels]]
            asks_clean = [(float(price), float(qty)) for price, qty in asks[:self.depth_levels]]

            # Create snapshot
            snapshot = OrderBookSnapshot(
                timestamp=datetime.now(),
                bids=bids_clean,
                asks=asks_clean
            )

            # Store in history
            self.orderbook_history.append(snapshot)
            self.current_orderbook = snapshot

            # Update liquidity heatmap
            self._update_liquidity_heatmap(snapshot)

            logger.debug(f"Updated orderbook for {self.symbol}: mid={snapshot.mid_price:.2f}, spread={snapshot.spread_bps:.1f}bps")
            return snapshot

        except (ValueError, TypeError) as e:
            logger.error(f"Error updating orderbook: {e}")
            raise

    def calculate_orderbook_imbalance(self, depth: int = 10) -> Dict[str, float]:
        """
        Calculate order book imbalance metrics at multiple depths.

        Args:
            depth: Number of price levels to analyze

        Returns:
            Dictionary with imbalance metrics
        """
        if not self.current_orderbook:
            return {'imbalance': 0.0, 'bid_volume': 0.0, 'ask_volume': 0.0}

        try:
            # Get volumes at specified depth
            bid_volume = sum(qty for _, qty in self.current_orderbook.bids[:depth])
            ask_volume = sum(qty for _, qty in self.current_orderbook.asks[:depth])
            total_volume = bid_volume + ask_volume

            if total_volume == 0:
                return {'imbalance': 0.0, 'bid_volume': 0.0, 'ask_volume': 0.0}

            # Calculate imbalance: (bid - ask) / (bid + ask)
            imbalance = (bid_volume - ask_volume) / total_volume

            return {
                'imbalance': imbalance,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'total_volume': total_volume,
                'depth': depth,
                'mid_price': self.current_orderbook.mid_price,
                'spread_bps': self.current_orderbook.spread_bps
            }

        except Exception as e:
            logger.error(f"Error calculating orderbook imbalance: {e}")
            return {'imbalance': 0.0, 'bid_volume': 0.0, 'ask_volume': 0.0}

    def get_dynamic_imbalance_threshold(self, volatility: float = None) -> float:
        """
        Calculate dynamic imbalance threshold based on market volatility.

        Args:
            volatility: Current volatility measure (ATR ratio or similar)

        Returns:
            Dynamic threshold value
        """
        base_threshold = self.imbalance_threshold

        if volatility is None:
            return base_threshold

        # Increase threshold in high volatility (require stronger imbalance)
        if volatility > 3.0:  # Very high volatility
            return base_threshold * 1.5
        elif volatility > 2.0:  # High volatility
            return base_threshold * 1.2
        elif volatility < 0.5:  # Low volatility
            return base_threshold * 0.8  # Can be more lenient

        return base_threshold

    def should_trade_based_on_imbalance(self, direction: str,
                                       imbalance_data: Dict[str, float],
                                       volatility: float = None) -> Tuple[bool, str]:
        """
        Determine if trading should proceed based on order book imbalance.

        Args:
            direction: 'LONG' or 'SHORT'
            imbalance_data: Imbalance metrics from calculate_orderbook_imbalance
            volatility: Current volatility measure

        Returns:
            Tuple of (should_trade, reason)
        """
        imbalance = imbalance_data.get('imbalance', 0.0)
        spread_bps = imbalance_data.get('spread_bps', 0.0)

        # Check spread first - don't trade if spread is too wide
        if spread_bps > self.config.get('max_spread_bps', 50):
            return False, f"Spread too wide ({spread_bps:.1f}bps > {self.config['max_spread_bps']}bps)"

        # Get dynamic threshold
        threshold = self.get_dynamic_imbalance_threshold(volatility)

        if direction == 'LONG':
            if imbalance > threshold:
                return True, f"Strong bid imbalance ({imbalance:.3f} > {threshold:.3f})"
            else:
                return False, f"Weak bid imbalance ({imbalance:.3f} < {threshold:.3f})"
        elif direction == 'SHORT':
            if imbalance < -threshold:
                return True, f"Strong ask imbalance ({imbalance:.3f} < {-threshold:.3f})"
            else:
                return False, f"Weak ask imbalance ({imbalance:.3f} > {-threshold:.3f})"

        return False, f"Invalid direction: {direction}"

    # ==========================================
    # TRADE FLOW ANALYSIS
    # ==========================================

    def process_trade(self, trade_data: Dict) -> None:
        """
        Process a real-time trade and update flow metrics.

        Args:
            trade_data: Trade data dictionary with price, quantity, side info
        """
        try:
            # Extract trade information
            price = float(trade_data.get('price', 0))
            quantity = float(trade_data.get('quantity', 0))
            side = trade_data.get('side', 'unknown')  # 'buy' or 'sell'
            timestamp = datetime.fromtimestamp(trade_data.get('timestamp', time.time()))

            if price <= 0 or quantity <= 0:
                return

            # Store trade
            trade_record = {
                'timestamp': timestamp,
                'price': price,
                'quantity': quantity,
                'side': side,
                'value': price * quantity
            }
            self.trade_history.append(trade_record)

            # Update rolling trade flow
            self._update_trade_flow(trade_record)

            # Check for large trades
            self._check_large_trade(trade_record)

        except (ValueError, KeyError) as e:
            logger.debug(f"Error processing trade data: {e}")

    def _update_trade_flow(self, trade: Dict) -> None:
        """Update rolling trade flow metrics."""
        try:
            # Classify trade direction
            is_buy_pressure = trade['side'].lower() in ['buy', 'bid']
            flow_value = trade['value'] if is_buy_pressure else -trade['value']

            # Update rolling windows
            current_time = time.time()

            for window, deque_obj in self.rolling_trade_flow.items():
                # Convert window name to seconds (e.g., '1m' -> 60)
                window_seconds = self._parse_time_window(window)
                deque_obj.append((current_time, flow_value))

                # Remove old entries
                cutoff_time = current_time - window_seconds
                while deque_obj and deque_obj[0][0] < cutoff_time:
                    deque_obj.popleft()

            # Calculate current metrics
            self._calculate_trade_flow_metrics()

        except Exception as e:
            logger.debug(f"Error updating trade flow: {e}")

    def _calculate_trade_flow_metrics(self) -> None:
        """Calculate current trade flow metrics."""
        try:
            current_time = time.time()

            # Calculate net flow for each window
            metrics = {}
            for window, deque_obj in self.rolling_trade_flow.items():
                if len(deque_obj) < 2:
                    metrics[f'net_flow_{window}'] = 0.0
                    continue

                # Calculate net flow
                net_flow = sum(flow for _, flow in deque_obj)
                metrics[f'net_flow_{window}'] = net_flow

                # Calculate flow momentum (rate of change)
                if len(deque_obj) >= 10:
                    recent_flow = sum(flow for _, flow in list(deque_obj)[-10:])
                    older_flow = sum(flow for _, flow in list(deque_obj)[:-10])
                    if older_flow != 0:
                        momentum = (recent_flow - older_flow) / abs(older_flow)
                        metrics[f'flow_momentum_{window}'] = momentum

            # Calculate buy/sell pressure
            recent_trades = list(self.trade_history)[-100:]  # Last 100 trades
            buy_volume = sum(t['value'] for t in recent_trades if t['side'].lower() in ['buy', 'bid'])
            sell_volume = sum(t['value'] for t in recent_trades if t['side'].lower() in ['sell', 'ask'])

            total_volume = buy_volume + sell_volume
            if total_volume > 0:
                metrics['buy_pressure'] = buy_volume / total_volume
                metrics['sell_pressure'] = sell_volume / total_volume
                metrics['aggression_ratio'] = buy_volume / sell_volume if sell_volume > 0 else float('inf')

            # Update metrics object
            self.trade_flow_metrics = TradeFlowMetrics(
                timestamp=datetime.now(),
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            )

        except Exception as e:
            logger.debug(f"Error calculating trade flow metrics: {e}")

    def _check_large_trade(self, trade: Dict) -> None:
        """Check if trade qualifies as a large trade."""
        try:
            if not self.trade_history:
                return

            # Calculate average trade size from recent history
            recent_trades = list(self.trade_history)[-1000:]  # Last 1000 trades
            avg_size = np.mean([t['value'] for t in recent_trades])

            if trade['value'] > avg_size * self.large_trade_multiplier:
                logger.info(f"Large trade detected: ${trade['value']:,.2f} "
                          f"({trade['value']/avg_size:.1f}x average)")

                # Update metrics
                if self.trade_flow_metrics:
                    self.trade_flow_metrics.large_trades_count += 1

        except Exception as e:
            logger.debug(f"Error checking large trade: {e}")

    def get_trade_flow_signal(self, direction: str) -> Tuple[bool, str]:
        """
        Get trade flow confirmation signal for a trade direction.

        Args:
            direction: 'LONG' or 'SHORT'

        Returns:
            Tuple of (confirmed, reason)
        """
        if not self.trade_flow_metrics:
            return True, "No trade flow data available"

        # Check 5-minute net flow
        net_flow_5m = getattr(self.trade_flow_metrics, 'net_flow_5m', 0)

        if direction == 'LONG':
            if net_flow_5m > 0:
                return True, f"Positive 5m trade flow (${net_flow_5m:,.2f})"
            else:
                return False, f"Negative 5m trade flow (${net_flow_5m:,.2f})"
        elif direction == 'SHORT':
            if net_flow_5m < 0:
                return True, f"Negative 5m trade flow (${net_flow_5m:,.2f})"
            else:
                return False, f"Positive 5m trade flow (${net_flow_5m:,.2f})"

        return True, "Neutral trade flow"

    # ==========================================
    # VOLUME PROFILE ANALYSIS
    # ==========================================

    def update_volume_profile(self, price: float, volume: float) -> None:
        """
        Update volume profile with new price-volume data.

        Args:
            price: Trade price
            volume: Trade volume
        """
        try:
            # Round price to create bins (e.g., to nearest $10 for BTC)
            price_bin = self._get_price_bin(price)
            self.price_volume_map[price_bin] += volume

            # Recalculate profile periodically
            if len(self.price_volume_map) % 100 == 0:  # Every 100 updates
                self._calculate_volume_profile()

        except Exception as e:
            logger.debug(f"Error updating volume profile: {e}")

    def _calculate_volume_profile(self) -> None:
        """Calculate volume profile metrics."""
        try:
            if not self.price_volume_map:
                return

            # Convert to sorted list
            price_volume_list = sorted(self.price_volume_map.items())
            prices, volumes = zip(*price_volume_list)

            total_volume = sum(volumes)

            # Find Point of Control (price with highest volume)
            poc_index = np.argmax(volumes)
            poc = prices[poc_index]

            # Calculate Value Area (70% of volume around POC)
            target_volume = total_volume * self.config.get('value_area_percentage', 0.7)

            # Find price range containing target volume
            cumsum = np.cumsum(volumes)
            total_cumsum = cumsum[-1]

            # Find upper and lower bounds for value area
            vah_price = None
            val_price = None

            for i, cum_vol in enumerate(cumsum):
                if cum_vol >= (total_cumsum - target_volume) / 2:  # Lower bound
                    if val_price is None:
                        val_price = prices[i]
                if cum_vol >= (total_cumsum + target_volume) / 2:  # Upper bound
                    vah_price = prices[i]
                    break

            # Update volume profile
            self.volume_profile = VolumeProfile(
                timestamp=datetime.now(),
                price_levels=dict(zip(prices, volumes)),
                poc=poc,
                vah=vah_price or poc,
                val=val_price or poc,
                total_volume=total_volume,
                value_area_volume=target_volume
            )

        except Exception as e:
            logger.debug(f"Error calculating volume profile: {e}")

    def _get_price_bin(self, price: float) -> float:
        """Get price bin for volume profile (round to appropriate level)."""
        # For crypto, round to 2 decimal places for fine granularity
        return round(price, 2)

    def get_volume_profile_signal(self, direction: str, entry_price: float) -> Tuple[bool, str]:
        """
        Get volume profile confirmation for entry.

        Args:
            direction: 'LONG' or 'SHORT'
            entry_price: Proposed entry price

        Returns:
            Tuple of (confirmed, reason)
        """
        if not self.volume_profile:
            return True, "No volume profile available"

        poc = self.volume_profile.poc
        vah = self.volume_profile.vah
        val = self.volume_profile.val

        if direction == 'LONG':
            # Prefer entries near VAL (value area low - discount area)
            if val <= entry_price <= vah:
                return True, f"Entry in value area (VAL: ${val:.2f}, VAH: ${vah:.2f})"
            elif entry_price < val * 0.995:  # 0.5% below VAL
                return True, f"Entry below VAL (good discount: ${entry_price:.2f} < ${val:.2f})"
            else:
                return False, f"Entry above VAH (premium area: ${entry_price:.2f} > ${vah:.2f})"

        elif direction == 'SHORT':
            # Prefer entries near VAH (value area high - premium area)
            if val <= entry_price <= vah:
                return True, f"Entry in value area (VAL: ${val:.2f}, VAH: ${vah:.2f})"
            elif entry_price > vah * 1.005:  # 0.5% above VAH
                return True, f"Entry above VAH (good premium: ${entry_price:.2f} > ${vah:.2f})"
            else:
                return False, f"Entry below VAL (discount area: ${entry_price:.2f} < ${val:.2f})"

        return True, "Volume profile neutral"

    # ==========================================
    # LIQUIDITY ANALYSIS
    # ==========================================

    def _update_liquidity_heatmap(self, snapshot: OrderBookSnapshot) -> None:
        """Update liquidity heatmap from order book snapshot."""
        try:
            bid_liquidity = {price: qty for price, qty in snapshot.bids[:self.depth_levels]}
            ask_liquidity = {price: qty for price, qty in snapshot.asks[:self.depth_levels]}

            total_bid_volume = sum(bid_liquidity.values())
            total_ask_volume = sum(ask_liquidity.values())

            # Calculate imbalance
            total_volume = total_bid_volume + total_ask_volume
            imbalance_ratio = 0.0
            if total_volume > 0:
                imbalance_ratio = (total_bid_volume - total_ask_volume) / total_volume

            # Identify support/resistance levels (large resting orders)
            support_levels = []
            resistance_levels = []

            # Support levels: large bids below mid price
            for price, qty in bid_liquidity.items():
                if qty > total_bid_volume * 0.1:  # Top 10% of bid volume
                    support_levels.append(price)

            # Resistance levels: large asks above mid price
            for price, qty in ask_liquidity.items():
                if qty > total_ask_volume * 0.1:  # Top 10% of ask volume
                    resistance_levels.append(price)

            self.liquidity_heatmap = LiquidityHeatmap(
                timestamp=snapshot.timestamp,
                bid_liquidity=bid_liquidity,
                ask_liquidity=ask_liquidity,
                total_bid_volume=total_bid_volume,
                total_ask_volume=total_ask_volume,
                imbalance_ratio=imbalance_ratio,
                support_levels=sorted(support_levels, reverse=True),  # High to low
                resistance_levels=sorted(resistance_levels)  # Low to high
            )

        except Exception as e:
            logger.debug(f"Error updating liquidity heatmap: {e}")

    def get_liquidity_based_sl(self, entry_price: float, direction: str, risk_pct: float) -> float:
        """
        Calculate stop-loss level based on liquidity analysis.

        Args:
            entry_price: Entry price
            direction: 'LONG' or 'SHORT'
            risk_pct: Risk percentage

        Returns:
            Recommended stop-loss price
        """
        if not self.liquidity_heatmap:
            # Fallback to percentage-based stop
            if direction == 'LONG':
                return entry_price * (1 - risk_pct)
            else:
                return entry_price * (1 + risk_pct)

        try:
            if direction == 'LONG':
                # Place stop below major support levels
                potential_stops = [price for price in self.liquidity_heatmap.support_levels
                                 if price < entry_price * (1 - risk_pct * 0.5)]  # Within 50% of normal risk

                if potential_stops:
                    # Choose the highest support level (closest to entry)
                    stop_level = max(potential_stops)
                    # Add small buffer below support
                    return stop_level * 0.9995
                else:
                    # Fallback to percentage
                    return entry_price * (1 - risk_pct)

            else:  # SHORT
                # Place stop above major resistance levels
                potential_stops = [price for price in self.liquidity_heatmap.resistance_levels
                                 if price > entry_price * (1 + risk_pct * 0.5)]  # Within 50% of normal risk

                if potential_stops:
                    # Choose the lowest resistance level (closest to entry)
                    stop_level = min(potential_stops)
                    # Add small buffer above resistance
                    return stop_level * 1.0005
                else:
                    # Fallback to percentage
                    return entry_price * (1 + risk_pct)

        except Exception as e:
            logger.debug(f"Error calculating liquidity-based SL: {e}")
            # Fallback
            if direction == 'LONG':
                return entry_price * (1 - risk_pct)
            else:
                return entry_price * (1 + risk_pct)

    # ==========================================
    # SPOOFING AND MANIPULATION DETECTION
    # ==========================================

    def detect_spoofing(self, orderbook_history: List[OrderBookSnapshot] = None) -> List[Dict]:
        """
        Detect potential spoofing patterns in order book.

        Spoofing indicators:
        - Large orders that disappear quickly
        - Sudden order book imbalances that reverse
        - Orders placed far from mid price

        Args:
            orderbook_history: List of recent order book snapshots

        Returns:
            List of detected spoofing events
        """
        if orderbook_history is None:
            orderbook_history = list(self.orderbook_history)

        if len(orderbook_history) < 5:
            return []

        spoofing_events = []

        try:
            # Check for disappearing large orders
            for i in range(2, len(orderbook_history)):
                current = orderbook_history[i]
                previous = orderbook_history[i-1]

                # Check bids
                current_bid_prices = {price for price, _ in current.bids}
                prev_bid_prices = {price for price, _ in previous.bids}

                disappeared_bids = prev_bid_prices - current_bid_prices

                for bid_price in disappeared_bids:
                    # Find the quantity that disappeared
                    prev_qty = next((qty for price, qty in previous.bids if price == bid_price), 0)

                    if prev_qty > current.total_bid_volume * 0.2:  # Large order
                        spoofing_events.append({
                            'type': 'disappearing_bid',
                            'price': bid_price,
                            'quantity': prev_qty,
                            'timestamp': current.timestamp,
                            'severity': 'high' if prev_qty > current.total_bid_volume * 0.5 else 'medium'
                        })

                # Check asks (similar logic)
                current_ask_prices = {price for price, _ in current.asks}
                prev_ask_prices = {price for price, _ in previous.asks}

                disappeared_asks = prev_ask_prices - current_ask_prices

                for ask_price in disappeared_asks:
                    prev_qty = next((qty for price, qty in previous.asks if price == ask_price), 0)

                    if prev_qty > current.total_ask_volume * 0.2:
                        spoofing_events.append({
                            'type': 'disappearing_ask',
                            'price': ask_price,
                            'quantity': prev_qty,
                            'timestamp': current.timestamp,
                            'severity': 'high' if prev_qty > current.total_ask_volume * 0.5 else 'medium'
                        })

        except Exception as e:
            logger.debug(f"Error detecting spoofing: {e}")

        return spoofing_events

    def find_hidden_liquidity(self) -> Dict[str, List]:
        """
        Identify potential hidden liquidity patterns.

        Looks for:
        - Iceberg orders (consistent small orders at same level)
        - Hidden orders behind displayed liquidity

        Returns:
            Dictionary with different types of hidden liquidity patterns
        """
        patterns = {
            'iceberg_orders': [],
            'hidden_resistance': [],
            'hidden_support': []
        }

        try:
            if not self.orderbook_history:
                return patterns

            # Analyze recent order book history
            recent_books = list(self.orderbook_history)[-10:]  # Last 10 snapshots

            # Look for consistent small orders (potential iceberg)
            price_counts = defaultdict(int)
            for book in recent_books:
                for price, qty in book.bids[:5]:  # Top 5 bids
                    price_counts[f"bid_{price}"] += 1
                for price, qty in book.asks[:5]:  # Top 5 asks
                    price_counts[f"ask_{price}"] += 1

            # Find prices that appear consistently but with small sizes
            for price_key, count in price_counts.items():
                if count >= 7:  # Appears in 70% of snapshots
                    side, price = price_key.split('_', 1)
                    price = float(price)

                    # Check average size at this price
                    sizes = []
                    for book in recent_books:
                        if side == 'bid':
                            sizes.extend([qty for p, qty in book.bids if p == price])
                        else:
                            sizes.extend([qty for p, qty in book.asks if p == price])

                    if sizes:
                        avg_size = np.mean(sizes)
                        # If consistently small orders at same level
                        if avg_size < self.config.get('min_order_size', 0.001) * 10:
                            patterns['iceberg_orders'].append({
                                'price': price,
                                'side': side,
                                'avg_size': avg_size,
                                'frequency': count / len(recent_books)
                            })

        except Exception as e:
            logger.debug(f"Error finding hidden liquidity: {e}")

        return patterns

    # ==========================================
    # EXECUTION QUALITY MONITORING
    # ==========================================

    def record_execution(self, order_id: str, symbol: str, side: str, quantity: float,
                        target_price: float, executed_price: float, execution_time_ms: int,
                        arrival_price: float) -> ExecutionQuality:
        """
        Record order execution for quality analysis.

        Args:
            order_id: Order identifier
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Executed quantity
            target_price: Target execution price
            executed_price: Actual execution price
            execution_time_ms: Time to execute in milliseconds
            arrival_price: Market price when order was placed

        Returns:
            ExecutionQuality object with calculated metrics
        """
        try:
            # Calculate slippage in basis points
            slippage = (executed_price - target_price) / target_price * 10000
            if side.upper() == 'SELL':
                slippage = -slippage  # Reverse for sells

            # Market impact (difference from arrival price)
            market_impact = abs(executed_price - arrival_price) / arrival_price * 10000

            # Realized spread (for market orders)
            realized_spread = abs(executed_price - arrival_price) / arrival_price * 10000

            # Implementation shortfall
            if side.upper() == 'BUY':
                implementation_shortfall = (executed_price - target_price) / target_price * 10000
            else:
                implementation_shortfall = (target_price - executed_price) / target_price * 10000

            execution_quality = ExecutionQuality(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                target_price=target_price,
                executed_price=executed_price,
                slippage_bps=slippage,
                market_impact_bps=market_impact,
                execution_time_ms=execution_time_ms,
                arrival_price=arrival_price,
                realized_spread=realized_spread,
                implementation_shortfall=implementation_shortfall
            )

            # Store execution record
            self.execution_history.append(execution_quality)

            # Maintain history size
            if len(self.execution_history) > self.max_execution_history:
                self.execution_history = self.execution_history[-self.max_execution_history:]

            logger.debug(f"Recorded execution for {order_id}: slippage={slippage:.1f}bps, "
                        f"impact={market_impact:.1f}bps")

            return execution_quality

        except Exception as e:
            logger.error(f"Error recording execution: {e}")
            raise

    def get_execution_quality_metrics(self, lookback_period: int = 100) -> Dict[str, float]:
        """
        Calculate execution quality metrics over recent orders.

        Args:
            lookback_period: Number of recent executions to analyze

        Returns:
            Dictionary with execution quality metrics
        """
        if not self.execution_history:
            return {}

        try:
            # Get recent executions
            recent_executions = self.execution_history[-lookback_period:]

            # Calculate metrics
            slippage_bps = [ex.slippage_bps for ex in recent_executions]
            market_impact_bps = [ex.market_impact_bps for ex in recent_executions]
            execution_times = [ex.execution_time_ms for ex in recent_executions]

            return {
                'avg_slippage_bps': np.mean(slippage_bps),
                'median_slippage_bps': np.median(slippage_bps),
                'max_slippage_bps': max(slippage_bps),
                'avg_market_impact_bps': np.mean(market_impact_bps),
                'avg_execution_time_ms': np.mean(execution_times),
                'execution_count': len(recent_executions),
                'implementation_shortfall_avg': np.mean([ex.implementation_shortfall for ex in recent_executions])
            }

        except Exception as e:
            logger.error(f"Error calculating execution quality metrics: {e}")
            return {}

    # ==========================================
    # UTILITY METHODS
    # ==========================================

    def _parse_time_window(self, window: str) -> int:
        """Parse time window string to seconds."""
        window = window.lower()
        if window == '1m':
            return 60
        elif window == '5m':
            return 300
        elif window == '15m':
            return 900
        elif window == '1h':
            return 3600
        else:
            return 300  # Default to 5 minutes

    def get_comprehensive_signal(self, direction: str, entry_price: float = None,
                               volatility: float = None) -> Dict[str, Union[bool, str]]:
        """
        Get comprehensive trading signal based on all microstructure analysis.

        Args:
            direction: 'LONG' or 'SHORT'
            entry_price: Proposed entry price
            volatility: Current volatility measure

        Returns:
            Dictionary with signal components and overall recommendation
        """
        result = {
            'overall_signal': False,
            'orderbook_confirmed': False,
            'trade_flow_confirmed': False,
            'volume_profile_confirmed': False,
            'reasons': [],
            'confidence_score': 0.0
        }

        try:
            confidence_score = 0.0
            reasons = []

            # 1. Order book imbalance check
            imbalance_data = self.calculate_orderbook_imbalance()
            imbalance_ok, imbalance_reason = self.should_trade_based_on_imbalance(
                direction, imbalance_data, volatility
            )
            result['orderbook_confirmed'] = imbalance_ok
            reasons.append(f"Orderbook: {imbalance_reason}")
            if imbalance_ok:
                confidence_score += 0.4

            # 2. Trade flow confirmation
            flow_ok, flow_reason = self.get_trade_flow_signal(direction)
            result['trade_flow_confirmed'] = flow_ok
            reasons.append(f"Trade Flow: {flow_reason}")
            if flow_ok:
                confidence_score += 0.3

            # 3. Volume profile confirmation
            if entry_price:
                volume_ok, volume_reason = self.get_volume_profile_signal(direction, entry_price)
                result['volume_profile_confirmed'] = volume_ok
                reasons.append(f"Volume Profile: {volume_reason}")
                if volume_ok:
                    confidence_score += 0.3

            # Overall signal (require at least 2 confirmations)
            confirmations = sum([result['orderbook_confirmed'],
                               result['trade_flow_confirmed'],
                               result['volume_profile_confirmed']])

            result['overall_signal'] = confirmations >= 2
            result['reasons'] = reasons
            result['confidence_score'] = confidence_score
            result['confirmations'] = confirmations

        except Exception as e:
            logger.error(f"Error generating comprehensive signal: {e}")
            result['reasons'].append(f"Error: {e}")

        return result

    def reset(self) -> None:
        """Reset all analyzer state."""
        self.current_orderbook = None
        self.orderbook_history.clear()
        self.trade_history.clear()
        self.trade_flow_metrics = None
        self.volume_profile = None
        self.liquidity_heatmap = None
        self.execution_history.clear()
        self.price_volume_map.clear()

        for deque_obj in self.rolling_trade_flow.values():
            deque_obj.clear()

        logger.info(f"OrderFlowAnalyzer reset for {self.symbol}")
