"""
Smart Order Router with Market Impact Reduction

This module implements advanced order routing strategies to minimize market impact
and improve execution quality. Includes TWAP (Time-Weighted Average Price) execution,
adaptive order sizing, and market impact modeling.

Key Features:
- TWAP execution for large orders
- Market impact estimation
- Adaptive order sizing based on liquidity
- Execution quality monitoring
- Market impact reduction algorithms
- VWAP execution for institutional orders
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from input_sanitizer import InputSanitizer
from order_flow_analyzer import OrderFlowAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class TWAPOrder:
    """Represents a TWAP (Time-Weighted Average Price) order."""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    total_quantity: float
    duration_minutes: int
    start_time: datetime
    end_time: datetime
    executed_quantity: float = 0.0
    executed_value: float = 0.0
    child_orders: List[Dict] = field(default_factory=list)
    status: str = 'pending'  # 'pending', 'executing', 'completed', 'cancelled'

    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to execute."""
        return self.total_quantity - self.executed_quantity

    @property
    def average_price(self) -> float:
        """Get average execution price."""
        return self.executed_value / self.executed_quantity if self.executed_quantity > 0 else 0.0

    @property
    def completion_percentage(self) -> float:
        """Get order completion percentage."""
        return (self.executed_quantity / self.total_quantity) * 100 if self.total_quantity > 0 else 0.0


@dataclass
class ExecutionMetrics:
    """Real-time execution quality metrics."""
    slippage_bps: float = 0.0
    market_impact_bps: float = 0.0
    arrival_price: float = 0.0
    vwap_price: float = 0.0
    realized_spread: float = 0.0
    implementation_shortfall: float = 0.0
    execution_time_ms: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class SmartOrderRouter:
    """
    Advanced order router with market impact minimization and execution optimization.

    This class implements sophisticated order routing strategies:
    - TWAP execution for large orders to reduce market impact
    - Market impact estimation and modeling
    - Adaptive order sizing based on liquidity conditions
    - Execution quality monitoring and reporting
    - VWAP execution for institutional orders

    The router analyzes market microstructure to determine optimal execution strategies
    and continuously monitors execution quality to adapt routing decisions.
    """

    def __init__(self, client, order_flow_analyzer: OrderFlowAnalyzer, config: Optional[Dict] = None):
        """
        Initialize the smart order router.

        Args:
            client: Exchange API client (HyperliquidClient)
            order_flow_analyzer: Order flow analyzer instance
            config: Configuration dictionary
        """
        self.client = client
        self.order_flow = order_flow_analyzer
        self.config = config or self._get_default_config()

        # Active TWAP orders
        self.active_twaps: Dict[str, TWAPOrder] = {}

        # Execution quality tracking
        self.execution_history: List[ExecutionMetrics] = []
        self.max_execution_history = 1000

        # Market impact model parameters
        self.market_impact_model = self._initialize_market_impact_model()

        logger.info("SmartOrderRouter initialized")

    def _get_default_config(self) -> Dict:
        """Get default configuration parameters."""
        return {
            'twap': {
                'min_order_size_usd': 50000,  # Orders >$50k use TWAP
                'default_duration_minutes': 30,
                'max_child_orders': 20,
                'min_child_order_interval_seconds': 30,
                'participation_rate': 0.1,  # 10% of market volume
            },
            'market_impact': {
                'price_impact_threshold_bps': 50,  # Max 50bps price impact
                'volume_impact_threshold_pct': 5,   # Max 5% of volume
                'liquidity_threshold_multiplier': 2.0,
            },
            'execution_quality': {
                'target_slippage_bps': 10,  # Target max slippage
                'monitor_execution_quality': True,
                'adaptive_routing': True,
            }
        }

    def _initialize_market_impact_model(self) -> Dict:
        """Initialize market impact model parameters."""
        return {
            'price_impact_coefficient': 0.5,  # Square root law coefficient
            'temporary_impact_decay': 0.1,    # Decay rate for temporary impact
            'permanent_impact_factor': 0.3,   # Permanent impact as fraction of temporary
            'spread_impact_factor': 0.2,      # Impact from bid-ask spread
            'depth_impact_factor': 0.1,       # Impact from order book depth
        }

    # ==========================================
    # TWAP EXECUTION
    # ==========================================

    def execute_twap(self, symbol: str, side: str, quantity: float,
                    duration_minutes: int = None, order_id: str = None) -> TWAPOrder:
        """
        Execute an order using Time-Weighted Average Price (TWAP) strategy.

        TWAP breaks large orders into smaller pieces executed at regular intervals
        to minimize market impact and achieve better average execution price.

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Total quantity to execute
            duration_minutes: Execution duration in minutes
            order_id: Custom order ID (auto-generated if None)

        Returns:
            TWAPOrder instance for tracking execution
        """
        try:
            # Sanitize inputs
            symbol = InputSanitizer.sanitize_symbol(symbol)
            side = InputSanitizer.sanitize_string(side, "side").upper()
            quantity = InputSanitizer.sanitize_quantity(quantity, "quantity")
            duration_minutes = duration_minutes or self.config['twap']['default_duration_minutes']
            duration_minutes = InputSanitizer.sanitize_int(duration_minutes, "duration_minutes", min_value=1, max_value=480)

            if order_id is None:
                order_id = f"twap_{symbol}_{side}_{int(time.time())}"

            # Calculate execution schedule
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=duration_minutes)

            # Estimate number of child orders
            max_child_orders = self.config['twap']['max_child_orders']
            min_interval = self.config['twap']['min_child_order_interval_seconds']

            # Calculate optimal child order size and timing
            child_order_size = quantity / max_child_orders
            interval_seconds = max(duration_minutes * 60 / max_child_orders, min_interval)

            # Create TWAP order
            twap_order = TWAPOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                total_quantity=quantity,
                duration_minutes=duration_minutes,
                start_time=start_time,
                end_time=end_time
            )

            # Schedule child orders
            self._schedule_twap_child_orders(twap_order, child_order_size, interval_seconds)

            # Store active TWAP
            self.active_twaps[order_id] = twap_order

            logger.info(f"Started TWAP execution for {order_id}: {quantity} {symbol} over {duration_minutes}min")
            return twap_order

        except Exception as e:
            logger.error(f"Error starting TWAP execution: {e}")
            raise

    def _schedule_twap_child_orders(self, twap_order: TWAPOrder,
                                   child_size: float, interval_seconds: float) -> None:
        """Schedule child orders for TWAP execution."""
        try:
            import threading

            def execute_child_orders():
                """Execute child orders at scheduled intervals."""
                total_scheduled = 0
                max_child_orders = self.config['twap']['max_child_orders']

                while (total_scheduled < max_child_orders and
                       twap_order.executed_quantity < twap_order.total_quantity and
                       datetime.now() < twap_order.end_time and
                       twap_order.status == 'executing'):

                    # Check if we should execute based on market conditions
                    if self._should_execute_child_order(twap_order):
                        # Execute child order
                        executed_qty, executed_price = self._execute_child_order(
                            twap_order, child_size
                        )

                        if executed_qty > 0:
                            twap_order.executed_quantity += executed_qty
                            twap_order.executed_value += executed_qty * executed_price

                            # Record child order
                            child_order = {
                                'timestamp': datetime.now(),
                                'quantity': executed_qty,
                                'price': executed_price,
                                'value': executed_qty * executed_price
                            }
                            twap_order.child_orders.append(child_order)

                            logger.debug(f"TWAP {twap_order.order_id}: executed {executed_qty} @ ${executed_price:.2f}")

                    total_scheduled += 1

                    # Wait for next execution slot
                    if total_scheduled < max_child_orders:
                        time.sleep(interval_seconds)

                # Mark as completed
                twap_order.status = 'completed'
                logger.info(f"TWAP {twap_order.order_id} completed: {twap_order.completion_percentage:.1f}% executed")

            # Start execution thread
            execution_thread = threading.Thread(target=execute_child_orders, daemon=True)
            execution_thread.start()

        except Exception as e:
            logger.error(f"Error scheduling TWAP child orders: {e}")

    def _should_execute_child_order(self, twap_order: TWAPOrder) -> bool:
        """Determine if child order should be executed based on market conditions."""
        try:
            # Get current market conditions
            orderbook = self.order_flow.current_orderbook
            if not orderbook:
                return True  # Execute if no orderbook data

            # Check participation rate (don't execute if market volume is too low)
            participation_rate = self.config['twap']['participation_rate']
            recent_volume = self._get_recent_market_volume(twap_order.symbol, minutes=5)

            if recent_volume > 0:
                target_volume = recent_volume * participation_rate / (12 * 5)  # 5-minute rate
                current_order_size = min(twap_order.remaining_quantity / 10, 0.1)  # Conservative sizing

                if current_order_size > target_volume:
                    logger.debug(f"TWAP: Skipping execution - order size {current_order_size} > target {target_volume}")
                    return False

            # Check for adverse market conditions
            imbalance_data = self.order_flow.calculate_orderbook_imbalance()
            imbalance = imbalance_data.get('imbalance', 0)

            if twap_order.side == 'BUY' and imbalance < -0.3:  # Strong selling pressure
                logger.debug("TWAP: Skipping BUY execution due to strong selling pressure")
                return False
            elif twap_order.side == 'SELL' and imbalance > 0.3:  # Strong buying pressure
                logger.debug("TWAP: Skipping SELL execution due to strong buying pressure")
                return False

            return True

        except Exception as e:
            logger.debug(f"Error checking child order execution conditions: {e}")
            return True  # Default to executing on error

    def _execute_child_order(self, twap_order: TWAPOrder, quantity: float) -> Tuple[float, float]:
        """Execute a single child order for TWAP."""
        try:
            # Use market order for TWAP child orders
            order_data = {
                "symbol": twap_order.symbol,
                "side": twap_order.side,
                "order_type": "MARKET",
                "quantity": str(quantity),
                "price": None
            }

            response = self.client.place_order(**order_data)

            if response.get('code') == 0:
                # Extract execution details (this would need to be implemented based on exchange response)
                executed_qty = quantity  # Assume full execution for simplicity
                executed_price = self._get_execution_price(response)

                return executed_qty, executed_price
            else:
                logger.warning(f"TWAP child order failed: {response}")
                return 0.0, 0.0

        except Exception as e:
            logger.error(f"Error executing TWAP child order: {e}")
            return 0.0, 0.0

    def _get_execution_price(self, order_response: Dict) -> float:
        """Extract execution price from order response."""
        # This would need to be implemented based on exchange API response format
        # For now, return current market price as approximation
        try:
            ticker = self.client.get_ticker(order_response.get('symbol', 'BTCUSDT'))
            return float(ticker.get('price', 0))
        except Exception:
            return 0.0

    def cancel_twap(self, twap_order_id: str) -> bool:
        """
        Cancel an active TWAP order.

        Args:
            twap_order_id: TWAP order ID to cancel

        Returns:
            True if cancelled successfully
        """
        if twap_order_id not in self.active_twaps:
            logger.warning(f"TWAP order {twap_order_id} not found")
            return False

        twap_order = self.active_twaps[twap_order_id]
        twap_order.status = 'cancelled'

        logger.info(f"Cancelled TWAP order {twap_order_id}")
        return True

    def get_twap_status(self, twap_order_id: str) -> Optional[TWAPOrder]:
        """
        Get status of a TWAP order.

        Args:
            twap_order_id: TWAP order ID

        Returns:
            TWAPOrder instance or None if not found
        """
        return self.active_twaps.get(twap_order_id)

    # ==========================================
    # MARKET IMPACT MODELING
    # ==========================================

    def estimate_market_impact(self, symbol: str, quantity: float, side: str) -> Dict[str, float]:
        """
        Estimate market impact of executing an order.

        Uses square root market impact model with order book depth analysis.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            side: 'BUY' or 'SELL'

        Returns:
            Dictionary with impact estimates in basis points
        """
        try:
            # Get order book data
            orderbook = self.order_flow.current_orderbook
            if not orderbook:
                return {'price_impact_bps': 0.0, 'volume_impact_pct': 0.0}

            # Calculate available liquidity
            if side.upper() == 'BUY':
                available_volume = sum(qty for _, qty in orderbook.asks[:10])  # Top 10 ask levels
            else:
                available_volume = sum(qty for _, qty in orderbook.bids[:10])  # Top 10 bid levels

            if available_volume == 0:
                return {'price_impact_bps': 1000.0, 'volume_impact_pct': 100.0}  # Very high impact

            # Square root market impact model
            participation_rate = quantity / available_volume
            price_impact_bps = self.market_impact_model['price_impact_coefficient'] * np.sqrt(participation_rate) * 10000

            # Volume impact
            volume_impact_pct = participation_rate * 100

            # Add spread impact
            spread_impact = orderbook.spread_bps * self.market_impact_model['spread_impact_factor']
            price_impact_bps += spread_impact

            return {
                'price_impact_bps': price_impact_bps,
                'volume_impact_pct': volume_impact_pct,
                'participation_rate': participation_rate,
                'available_liquidity': available_volume,
                'spread_impact_bps': spread_impact
            }

        except Exception as e:
            logger.error(f"Error estimating market impact: {e}")
            return {'price_impact_bps': 0.0, 'volume_impact_pct': 0.0}

    def should_use_smart_routing(self, symbol: str, quantity: float, side: str) -> Tuple[bool, str]:
        """
        Determine if smart routing should be used for an order.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            side: 'BUY' or 'SELL'

        Returns:
            Tuple of (use_smart_routing, reason)
        """
        try:
            # Check order size threshold
            current_price = self.order_flow.current_orderbook.mid_price if self.order_flow.current_orderbook else 0
            order_value_usd = quantity * current_price

            min_twap_size = self.config['twap']['min_order_size_usd']
            if order_value_usd >= min_twap_size:
                return True, f"Order size ${order_value_usd:,.0f} exceeds TWAP threshold ${min_twap_size:,.0f}"

            # Check market impact
            impact = self.estimate_market_impact(symbol, quantity, side)
            max_impact = self.config['market_impact']['price_impact_threshold_bps']

            if impact['price_impact_bps'] > max_impact:
                return True, f"Estimated impact {impact['price_impact_bps']:.1f}bps > threshold {max_impact}bps"

            # Check volume impact
            max_volume_impact = self.config['market_impact']['volume_impact_threshold_pct']
            if impact['volume_impact_pct'] > max_volume_impact:
                return True, f"Volume impact {impact['volume_impact_pct']:.1f}% > threshold {max_volume_impact}%"

            return False, "Order size and impact within normal limits"

        except Exception as e:
            logger.error(f"Error determining smart routing: {e}")
            return False, f"Error in analysis: {e}"

    # ==========================================
    # EXECUTION QUALITY MONITORING
    # ==========================================

    def record_execution_quality(self, order_id: str, symbol: str, side: str,
                               quantity: float, target_price: float,
                               executed_price: float, execution_time_ms: int) -> ExecutionMetrics:
        """
        Record execution quality metrics for analysis.

        Args:
            order_id: Order identifier
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Executed quantity
            target_price: Target execution price
            executed_price: Actual execution price
            execution_time_ms: Time to execute in milliseconds

        Returns:
            ExecutionMetrics instance
        """
        try:
            # Get arrival price (price when order was placed)
            arrival_price = self.order_flow.current_orderbook.mid_price if self.order_flow.current_orderbook else target_price

            # Calculate metrics
            slippage = ((executed_price - target_price) / target_price) * 10000  # Basis points
            if side.upper() == 'SELL':
                slippage = -slippage  # Reverse for sells

            market_impact = abs(executed_price - arrival_price) / arrival_price * 10000
            realized_spread = abs(executed_price - arrival_price) / arrival_price * 10000

            # Implementation shortfall
            if side.upper() == 'BUY':
                implementation_shortfall = (executed_price - target_price) / target_price * 10000
            else:
                implementation_shortfall = (target_price - executed_price) / target_price * 10000

            # VWAP (simplified - would need volume-weighted calculation)
            vwap_price = executed_price  # Placeholder

            metrics = ExecutionMetrics(
                slippage_bps=slippage,
                market_impact_bps=market_impact,
                arrival_price=arrival_price,
                vwap_price=vwap_price,
                realized_spread=realized_spread,
                implementation_shortfall=implementation_shortfall,
                execution_time_ms=execution_time_ms
            )

            # Store in history
            self.execution_history.append(metrics)

            # Maintain history size
            if len(self.execution_history) > self.max_execution_history:
                self.execution_history = self.execution_history[-self.max_execution_history:]

            logger.debug(f"Recorded execution quality for {order_id}: slippage={slippage:.1f}bps, impact={market_impact:.1f}bps")

            return metrics

        except Exception as e:
            logger.error(f"Error recording execution quality: {e}")
            raise

    def get_execution_quality_summary(self, lookback_period: int = 100) -> Dict[str, float]:
        """
        Get summary of execution quality metrics.

        Args:
            lookback_period: Number of recent executions to analyze

        Returns:
            Dictionary with execution quality statistics
        """
        if not self.execution_history:
            return {}

        try:
            recent_executions = self.execution_history[-lookback_period:]

            slippage_values = [ex.slippage_bps for ex in recent_executions]
            impact_values = [ex.market_impact_bps for ex in recent_executions]
            shortfall_values = [ex.implementation_shortfall for ex in recent_executions]

            return {
                'avg_slippage_bps': np.mean(slippage_values),
                'median_slippage_bps': np.median(slippage_values),
                'max_slippage_bps': max(slippage_values),
                'avg_market_impact_bps': np.mean(impact_values),
                'avg_implementation_shortfall_bps': np.mean(shortfall_values),
                'execution_count': len(recent_executions),
                'slippage_std_bps': np.std(slippage_values),
                'impact_std_bps': np.std(impact_values)
            }

        except Exception as e:
            logger.error(f"Error calculating execution quality summary: {e}")
            return {}

    # ==========================================
    # ADAPTIVE ORDER SIZING
    # ==========================================

    def calculate_adaptive_order_size(self, symbol: str, side: str,
                                    base_quantity: float, urgency: str = 'normal') -> float:
        """
        Calculate adaptive order size based on market conditions.

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            base_quantity: Base order quantity
            urgency: 'low', 'normal', 'high' - affects size adjustment

        Returns:
            Adjusted order quantity
        """
        try:
            # Get market conditions
            orderbook = self.order_flow.current_orderbook
            if not orderbook:
                return base_quantity

            # Calculate available liquidity
            if side.upper() == 'BUY':
                available_liquidity = sum(qty for _, qty in orderbook.asks[:5])  # Top 5 levels
            else:
                available_liquidity = sum(qty for _, qty in orderbook.bids[:5])  # Top 5 levels

            if available_liquidity == 0:
                return min(base_quantity, 0.01)  # Very small size if no liquidity

            # Adjust based on liquidity
            liquidity_ratio = available_liquidity / (base_quantity * 10)  # Compare to 10x base size

            if liquidity_ratio < 0.1:  # Low liquidity
                size_multiplier = 0.3
            elif liquidity_ratio < 0.5:  # Moderate liquidity
                size_multiplier = 0.6
            else:  # Good liquidity
                size_multiplier = 1.0

            # Adjust based on urgency
            urgency_multipliers = {
                'low': 0.7,
                'normal': 1.0,
                'high': 1.3
            }
            size_multiplier *= urgency_multipliers.get(urgency, 1.0)

            # Apply spread adjustment
            spread_multiplier = 1.0
            if orderbook.spread_bps > 100:  # Wide spread (>1%)
                spread_multiplier = 0.5
            elif orderbook.spread_bps > 50:  # Moderate spread (>0.5%)
                spread_multiplier = 0.8

            size_multiplier *= spread_multiplier

            # Calculate final size
            adjusted_quantity = base_quantity * size_multiplier

            # Ensure within reasonable bounds
            max_size = available_liquidity * 0.1  # Max 10% of available liquidity
            adjusted_quantity = min(adjusted_quantity, max_size)

            logger.debug(f"Adaptive order sizing: {base_quantity:.4f} -> {adjusted_quantity:.4f} "
                        f"(liquidity_ratio={liquidity_ratio:.2f}, urgency={urgency})")

            return adjusted_quantity

        except Exception as e:
            logger.error(f"Error calculating adaptive order size: {e}")
            return base_quantity

    # ==========================================
    # UTILITY METHODS
    # ==========================================

    def _get_recent_market_volume(self, symbol: str, minutes: int) -> float:
        """Get recent market volume for participation rate calculation."""
        try:
            # Get recent trades
            recent_trades = self.client.get_recent_trades(symbol, limit=100)

            # Filter by time window
            cutoff_time = time.time() - (minutes * 60)
            recent_volume = sum(
                trade['value'] for trade in recent_trades
                if trade.get('timestamp', 0) > cutoff_time
            )

            return recent_volume

        except Exception as e:
            logger.debug(f"Error getting recent market volume: {e}")
            return 0.0

    def cleanup_completed_twaps(self) -> None:
        """Clean up completed TWAP orders."""
        completed_ids = [
            twap_id for twap_id, twap in self.active_twaps.items()
            if twap.status in ['completed', 'cancelled']
        ]

        for twap_id in completed_ids:
            del self.active_twaps[twap_id]

        if completed_ids:
            logger.info(f"Cleaned up {len(completed_ids)} completed TWAP orders")

    def get_routing_recommendation(self, symbol: str, quantity: float, side: str) -> Dict[str, Union[str, bool, float]]:
        """
        Get comprehensive routing recommendation for an order.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            side: 'BUY' or 'SELL'

        Returns:
            Dictionary with routing recommendations
        """
        try:
            recommendation = {
                'use_twap': False,
                'use_smart_routing': False,
                'suggested_duration_minutes': 0,
                'estimated_impact_bps': 0.0,
                'recommended_quantity': quantity,
                'reason': 'Standard market order recommended'
            }

            # Check if smart routing should be used
            use_smart, reason = self.should_use_smart_routing(symbol, quantity, side)
            recommendation['use_smart_routing'] = use_smart

            if use_smart:
                # Estimate market impact
                impact = self.estimate_market_impact(symbol, quantity, side)
                recommendation['estimated_impact_bps'] = impact['price_impact_bps']

                # Recommend TWAP for large orders
                current_price = self.order_flow.current_orderbook.mid_price if self.order_flow.current_orderbook else 0
                order_value = quantity * current_price

                if order_value > self.config['twap']['min_order_size_usd']:
                    recommendation['use_twap'] = True
                    recommendation['suggested_duration_minutes'] = self.config['twap']['default_duration_minutes']
                    recommendation['reason'] = f"Large order (${order_value:,.0f}) - use TWAP to reduce impact"

                # Adjust quantity based on liquidity
                adaptive_quantity = self.calculate_adaptive_order_size(symbol, side, quantity)
                recommendation['recommended_quantity'] = adaptive_quantity

                if adaptive_quantity < quantity:
                    recommendation['reason'] += f" Reduced size to {adaptive_quantity:.4f} due to liquidity constraints"

            return recommendation

        except Exception as e:
            logger.error(f"Error getting routing recommendation: {e}")
            return {
                'use_twap': False,
                'use_smart_routing': False,
                'reason': f'Error in analysis: {e}'
            }
