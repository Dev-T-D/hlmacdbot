"""
Order Status Tracking and Retry System

Tracks order status, handles retries for failed orders, and manages partial fills.
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"  # Order placed, waiting for confirmation
    SUBMITTED = "submitted"  # Order submitted to exchange
    PARTIAL = "partial"  # Partially filled
    FILLED = "filled"  # Fully filled
    FAILED = "failed"  # Order failed
    CANCELLED = "cancelled"  # Order cancelled
    EXPIRED = "expired"  # Order expired


class OrderTracker:
    """
    Tracks order status and handles retries
    
    Monitors orders, checks their status, retries failed orders,
    and handles partial fills.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0, 
                 status_check_interval: float = 5.0):
        """
        Initialize order tracker
        
        Args:
            max_retries: Maximum number of retry attempts for failed orders
            retry_delay: Delay between retry attempts (seconds)
            status_check_interval: Interval between status checks (seconds)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.status_check_interval = status_check_interval
        
        # Track active orders
        self.active_orders: Dict[str, Dict] = {}  # order_id -> order_info
        self.order_history: List[Dict] = []  # All orders (for history)
        self.max_history = 100  # Keep last 100 orders
        
    def track_order(self, order_id: Optional[str], order_info: Dict) -> str:
        """
        Track a new order
        
        Args:
            order_id: Order ID from exchange (may be None if order failed)
            order_info: Order details (symbol, side, quantity, price, etc.)
            
        Returns:
            Internal tracking ID
        """
        # Generate internal tracking ID if order_id is None
        if order_id is None:
            order_id = f"pending_{int(time.time() * 1000)}"
        
        tracking_id = order_id
        
        order_record = {
            'order_id': order_id,
            'tracking_id': tracking_id,
            'symbol': order_info.get('symbol'),
            'side': order_info.get('side'),
            'trade_side': order_info.get('trade_side', 'OPEN'),
            'order_type': order_info.get('order_type', 'MARKET'),
            'quantity': order_info.get('quantity'),
            'price': order_info.get('price'),
            'status': OrderStatus.SUBMITTED.value if order_id else OrderStatus.FAILED.value,
            'filled_quantity': 0.0,
            'filled_price': 0.0,
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
            'retry_count': 0,
            'error_message': None,
            'original_order_info': order_info
        }
        
        self.active_orders[tracking_id] = order_record
        self.order_history.append(order_record.copy())
        
        # Keep history size manageable
        if len(self.order_history) > self.max_history:
            self.order_history = self.order_history[-self.max_history:]
        
        logger.info(f"📋 Tracking order: {tracking_id} ({order_record['status']})")
        
        return tracking_id
    
    def update_order_status(self, tracking_id: str, status: OrderStatus, 
                          filled_quantity: Optional[float] = None,
                          filled_price: Optional[float] = None,
                          error_message: Optional[str] = None) -> None:
        """
        Update order status
        
        Args:
            tracking_id: Internal tracking ID
            status: New status
            filled_quantity: Filled quantity (for partial/filled orders)
            filled_price: Average fill price
            error_message: Error message if failed
        """
        if tracking_id not in self.active_orders:
            logger.warning(f"Order {tracking_id} not found in active orders")
            return
        
        order = self.active_orders[tracking_id]
        old_status = order['status']
        order['status'] = status.value
        order['updated_at'] = datetime.now(timezone.utc)
        
        if filled_quantity is not None:
            order['filled_quantity'] = filled_quantity
        if filled_price is not None:
            order['filled_price'] = filled_price
        if error_message:
            order['error_message'] = error_message
        
        logger.info(f"📊 Order {tracking_id} status: {old_status} → {status.value}")
        
        # Remove from active orders if completed/failed/cancelled
        if status in [OrderStatus.FILLED, OrderStatus.FAILED, OrderStatus.CANCELLED, OrderStatus.EXPIRED]:
            logger.info(f"✅ Order {tracking_id} completed (status: {status.value})")
            # Keep in history but remove from active tracking
            del self.active_orders[tracking_id]
    
    def check_order_status(self, client, tracking_id: str) -> Optional[OrderStatus]:
        """
        Check order status with exchange
        
        Args:
            client: Exchange client instance
            tracking_id: Internal tracking ID
            
        Returns:
            Current order status or None if order not found
        """
        if tracking_id not in self.active_orders:
            return None
        
        order = self.active_orders[tracking_id]
        order_id = order['order_id']
        symbol = order['symbol']
        
        try:
            # Check if order is still open
            open_orders = client.get_open_orders(symbol=symbol)
            
            # Find our order in open orders
            order_found = False
            for open_order in open_orders:
                if str(open_order.get('orderId')) == str(order_id):
                    order_found = True
                    # Check if partially filled
                    open_qty = float(open_order.get('quantity', 0))
                    original_qty = float(order.get('quantity', 0))
                    
                    if open_qty < original_qty:
                        # Partially filled
                        filled_qty = original_qty - open_qty
                        self.update_order_status(
                            tracking_id, 
                            OrderStatus.PARTIAL,
                            filled_quantity=filled_qty
                        )
                        return OrderStatus.PARTIAL
                    else:
                        # Still open, not filled
                        return OrderStatus.SUBMITTED
            
            # Order not in open orders - check if it was filled
            if not order_found:
                # Order is not open - check position to see if it was filled
                # This is a heuristic - if we have a position, assume order filled
                try:
                    position = client.get_position(symbol)
                    if position:
                        # Position exists - order likely filled
                        self.update_order_status(
                            tracking_id,
                            OrderStatus.FILLED,
                            filled_quantity=float(order.get('quantity', 0))
                        )
                        return OrderStatus.FILLED
                    else:
                        # No position - order may have been cancelled or expired
                        self.update_order_status(
                            tracking_id,
                            OrderStatus.CANCELLED,
                            error_message="Order not found in open orders and no position"
                        )
                        return OrderStatus.CANCELLED
                except Exception as e:
                    logger.warning(f"Could not verify order fill status: {e}")
                    # Assume cancelled if we can't verify
                    self.update_order_status(
                        tracking_id,
                        OrderStatus.CANCELLED,
                        error_message=f"Could not verify status: {e}"
                    )
                    return OrderStatus.CANCELLED
            
            return OrderStatus.SUBMITTED
            
        except Exception as e:
            logger.error(f"Error checking order status for {tracking_id}: {e}")
            return None
    
    def should_retry(self, tracking_id: str) -> bool:
        """
        Check if order should be retried
        
        Args:
            tracking_id: Internal tracking ID
            
        Returns:
            True if order should be retried
        """
        if tracking_id not in self.active_orders:
            return False
        
        order = self.active_orders[tracking_id]
        
        # Only retry failed orders
        if order['status'] != OrderStatus.FAILED.value:
            return False
        
        # Check retry count
        if order['retry_count'] >= self.max_retries:
            logger.warning(f"Order {tracking_id} exceeded max retries ({self.max_retries})")
            return False
        
        return True
    
    def mark_for_retry(self, tracking_id: str) -> Dict:
        """
        Mark order for retry and return order info
        
        Args:
            tracking_id: Internal tracking ID
            
        Returns:
            Order information for retry
        """
        if tracking_id not in self.active_orders:
            raise ValueError(f"Order {tracking_id} not found")
        
        order = self.active_orders[tracking_id]
        order['retry_count'] += 1
        order['status'] = OrderStatus.PENDING.value
        order['updated_at'] = datetime.now(timezone.utc)
        
        logger.info(f"🔄 Retrying order {tracking_id} (attempt {order['retry_count']}/{self.max_retries})")
        
        return order['original_order_info']
    
    def get_order(self, tracking_id: str) -> Optional[Dict]:
        """Get order information"""
        return self.active_orders.get(tracking_id)
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get all active orders
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of active orders
        """
        orders = list(self.active_orders.values())
        
        if symbol:
            orders = [o for o in orders if o.get('symbol') == symbol]
        
        return orders
    
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """
        Get order history
        
        Args:
            symbol: Optional symbol filter
            limit: Maximum number of orders to return
            
        Returns:
            List of historical orders
        """
        history = self.order_history.copy()
        
        if symbol:
            history = [o for o in history if o.get('symbol') == symbol]
        
        # Sort by created_at (newest first)
        history.sort(key=lambda x: x.get('created_at', datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
        
        return history[:limit]
    
    def cleanup_old_orders(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed orders from history
        
        Args:
            max_age_hours: Maximum age in hours for orders to keep
            
        Returns:
            Number of orders removed
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        initial_count = len(self.order_history)
        
        self.order_history = [
            o for o in self.order_history
            if o.get('created_at', datetime.now(timezone.utc)) > cutoff_time
        ]
        
        removed = initial_count - len(self.order_history)
        if removed > 0:
            logger.info(f"🧹 Cleaned up {removed} old orders from history")
        
        return removed

