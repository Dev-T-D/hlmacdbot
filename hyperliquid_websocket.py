"""
Hyperliquid WebSocket Client for Real-Time Data Streaming

Provides WebSocket-based real-time market data updates for Hyperliquid exchange.
Supports trades, orderbook, and other market data channels.

Official Hyperliquid WebSocket API Documentation:
https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket
"""

import asyncio
import json
import logging
import threading
import time
from queue import Queue
from typing import Callable, Dict, Optional

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, WebSocketException
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None
    ConnectionClosed = Exception
    WebSocketException = Exception

logger = logging.getLogger(__name__)


class HyperliquidWebSocketClient:
    """WebSocket client for real-time Hyperliquid data."""

    def __init__(
        self,
        testnet: bool = True,
        reconnect_interval: float = 5.0,
        max_reconnect_attempts: int = 10,
    ):
        """
        Initialize WebSocket client.

        Args:
            testnet: Use testnet or mainnet
            reconnect_interval: Seconds between reconnection attempts
            max_reconnect_attempts: Maximum reconnection attempts before giving up
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError(
                "websockets library not installed. Install with: pip install websockets"
            )

        self.ws_url = (
            "wss://api.hyperliquid-testnet.xyz/ws"
            if testnet
            else "wss://api.hyperliquid.xyz/ws"
        )
        self.testnet = testnet
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts

        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.subscriptions: Dict[str, Dict] = {}
        self.callbacks: Dict[str, Callable] = {}
        self.running = False
        self.connected = False
        self.reconnect_attempts = 0
        self.message_queue: Queue = Queue()

        # Threading
        self.ws_thread: Optional[threading.Thread] = None
        self.ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()

        logger.info(f"WebSocket client initialized for {'testnet' if testnet else 'mainnet'}")

    async def _connect(self) -> bool:
        """
        Establish WebSocket connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to WebSocket: {self.ws_url}")
            self.websocket = await websockets.connect(
                self.ws_url,
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=10,  # Wait 10 seconds for pong
                close_timeout=10,
            )
            self.connected = True
            self.reconnect_attempts = 0
            logger.info("WebSocket connected successfully")
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.connected = False
            return False

    async def _disconnect(self):
        """Close WebSocket connection."""
        self.connected = False
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("WebSocket disconnected")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            finally:
                self.websocket = None

    async def _send_message(self, message: Dict):
        """Send message to WebSocket."""
        if self.websocket and self.connected:
            try:
                await self.websocket.send(json.dumps(message))
                logger.debug(f"Sent WebSocket message: {message.get('method', 'unknown')}")
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                self.connected = False
                raise

    async def subscribe_trades(self, coin: str, callback: Callable):
        """
        Subscribe to trade updates for a coin.

        Args:
            coin: Coin symbol (e.g., "BTC", "ETH", "SOL")
            callback: Function to call when trade data is received
        """
        subscription = {
            "method": "subscribe",
            "subscription": {
                "type": "trades",
                "coin": coin,
            },
        }
        await self._send_message(subscription)
        subscription_key = f"trades_{coin}"
        self.subscriptions[subscription_key] = subscription
        self.callbacks[subscription_key] = callback
        logger.info(f"Subscribed to trades for {coin}")

    async def subscribe_orderbook(self, coin: str, callback: Callable):
        """
        Subscribe to order book updates.

        Args:
            coin: Coin symbol (e.g., "BTC", "ETH", "SOL")
            callback: Function to call when orderbook data is received
        """
        subscription = {
            "method": "subscribe",
            "subscription": {
                "type": "orderbook",
                "coin": coin,
            },
        }
        await self._send_message(subscription)
        subscription_key = f"orderbook_{coin}"
        self.subscriptions[subscription_key] = subscription
        self.callbacks[subscription_key] = callback
        logger.info(f"Subscribed to orderbook for {coin}")

    async def _listen(self):
        """Listen for incoming messages."""
        while self.running and not self._stop_event.is_set():
            try:
                if not self.websocket or not self.connected:
                    await asyncio.sleep(1)
                    continue

                # Set timeout for receiving messages
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(), timeout=30.0
                    )
                    data = json.loads(message)
                    await self._handle_message(data)
                except asyncio.TimeoutError:
                    # Timeout is normal - continue listening
                    continue

            except ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.connected = False
                break
            except WebSocketException as e:
                logger.error(f"WebSocket error: {e}")
                self.connected = False
                break
            except Exception as e:
                logger.error(f"Error in WebSocket listener: {e}")
                self.connected = False
                break

    async def _handle_message(self, data: Dict):
        """
        Handle incoming WebSocket message.

        Args:
            data: Parsed JSON message from WebSocket
        """
        try:
            channel = data.get("channel")
            coin = data.get("coin", "")

            if channel:
                # Try exact match first
                callback_key = f"{channel}_{coin}"
                if callback_key in self.callbacks:
                    callback = self.callbacks[callback_key]
                    try:
                        # Put message in queue for thread-safe processing
                        self.message_queue.put((callback, data))
                    except Exception as e:
                        logger.error(f"Error queuing WebSocket message: {e}")
                else:
                    logger.debug(f"No callback registered for {callback_key}")
            else:
                logger.debug(f"Received message without channel: {data}")

        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    async def _reconnect(self) -> bool:
        """
        Attempt to reconnect to WebSocket.

        Returns:
            True if reconnection successful, False otherwise
        """
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(
                f"Max reconnection attempts ({self.max_reconnect_attempts}) reached"
            )
            return False

        self.reconnect_attempts += 1
        wait_time = min(
            self.reconnect_interval * (2 ** (self.reconnect_attempts - 1)), 60.0
        )  # Exponential backoff, max 60s
        logger.info(
            f"Reconnecting in {wait_time:.1f}s "
            f"(attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})..."
        )
        await asyncio.sleep(wait_time)

        if await self._connect():
            # Resubscribe to all subscriptions
            for subscription_key, subscription in self.subscriptions.items():
                try:
                    await self._send_message(subscription)
                    logger.debug(f"Resubscribed to {subscription_key}")
                except Exception as e:
                    logger.error(f"Error resubscribing to {subscription_key}: {e}")
            return True
        return False

    async def _run(self):
        """Main WebSocket event loop."""
        logger.info("Starting WebSocket event loop")
        self.running = True

        while self.running and not self._stop_event.is_set():
            try:
                # Connect
                if not self.connected:
                    if not await self._connect():
                        if not await self._reconnect():
                            logger.error("Failed to connect/reconnect, stopping")
                            break
                        continue

                # Listen for messages
                await self._listen()

                # If we get here, connection was lost
                if self.running and not self._stop_event.is_set():
                    logger.warning("Connection lost, attempting reconnect...")
                    if not await self._reconnect():
                        break

            except Exception as e:
                logger.error(f"Error in WebSocket event loop: {e}")
                if self.running and not self._stop_event.is_set():
                    await asyncio.sleep(self.reconnect_interval)
                    if not await self._reconnect():
                        break

        logger.info("WebSocket event loop stopped")
        self.running = False
        self.connected = False

    def start(self):
        """Start WebSocket client in background thread."""
        if self.ws_thread and self.ws_thread.is_alive():
            logger.warning("WebSocket client already running")
            return

        def run_ws():
            """Run WebSocket in separate event loop."""
            self.ws_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.ws_loop)
            try:
                self.ws_loop.run_until_complete(self._run())
            except Exception as e:
                logger.error(f"WebSocket thread error: {e}")
            finally:
                self.ws_loop.close()

        self._stop_event.clear()
        self.ws_thread = threading.Thread(target=run_ws, daemon=True, name="WebSocketThread")
        self.ws_thread.start()
        logger.info("WebSocket client started in background thread")

    def stop(self):
        """Stop WebSocket client."""
        logger.info("Stopping WebSocket client...")
        self.running = False
        self._stop_event.set()

        if self.ws_loop and self.ws_loop.is_running():
            # Schedule disconnect in event loop
            asyncio.run_coroutine_threadsafe(self._disconnect(), self.ws_loop)

        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=5.0)
            if self.ws_thread.is_alive():
                logger.warning("WebSocket thread did not stop gracefully")

        logger.info("WebSocket client stopped")

    def process_messages(self):
        """
        Process queued WebSocket messages.

        Call this from the main thread to process messages thread-safely.
        """
        processed = 0
        while not self.message_queue.empty():
            try:
                callback, data = self.message_queue.get_nowait()
                callback(data)
                processed += 1
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")

        if processed > 0:
            logger.debug(f"Processed {processed} WebSocket messages")

    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.connected and self.websocket is not None

    def get_subscription_count(self) -> int:
        """Get number of active subscriptions."""
        return len(self.subscriptions)

