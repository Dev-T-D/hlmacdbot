"""
Hyperliquid WebSocket Client for real-time data streaming
"""

import asyncio
import json
import websockets
import logging
from typing import Callable, Dict, Optional
import time

logger = logging.getLogger(__name__)

# Check if websockets is available
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets library not available - using REST fallback")

class HyperliquidWebSocketClient:
    """
    WebSocket client for Hyperliquid real-time data

    Connects to Hyperliquid's WebSocket API for real-time trade data.
    """

    def __init__(self, testnet: bool = False):
        self.testnet = testnet
        self.ws_url = "wss://api.hyperliquid.xyz/ws" if not testnet else "wss://api.hyperliquid-testnet.xyz/ws"
        self.websocket = None
        self.connected = False
        self.subscriptions = {}
        self.callbacks = {}

    async def start(self):
        """Start the WebSocket connection"""
        try:
            logger.info(f"🔌 Connecting to {self.ws_url}")
            self.websocket = await websockets.connect(self.ws_url)
            self.connected = True
            logger.info("✅ WebSocket connected")

            # Start message handler
            asyncio.create_task(self._handle_messages())

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.connected = False

    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.connected and self.websocket is not None

    async def subscribe_trades(self, coin: str, callback: Callable):
        """
        Subscribe to trade updates for a coin

        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')
            callback: Function to call with trade data
        """
        if not self.is_connected():
            logger.warning("WebSocket not connected")
            return

        subscription_id = f"trades_{coin}"

        # Subscribe message
        subscribe_msg = {
            "method": "subscribe",
            "subscription": {
                "type": "trades",
                "coin": coin
            }
        }

        try:
            await self.websocket.send(json.dumps(subscribe_msg))
            self.callbacks[subscription_id] = callback
            logger.info(f"📡 Subscribed to {coin} trades")

        except Exception as e:
            logger.error(f"Failed to subscribe to {coin} trades: {e}")

    async def _handle_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)

                    # Handle trade messages
                    if "channel" in data and data["channel"] == "trades":
                        await self._handle_trade_message(data)

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON message: {message}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"WebSocket message handling error: {e}")
            self.connected = False

    async def _handle_trade_message(self, data: Dict):
        """Handle incoming trade message"""
        try:
            trades = data.get("data", [])

            for trade in trades:
                # Call all registered callbacks
                for callback_id, callback in self.callbacks.items():
                    if callback_id.startswith("trades_"):
                        try:
                            await callback(trade)
                        except Exception as e:
                            logger.error(f"Error in trade callback: {e}")

        except Exception as e:
            logger.error(f"Error handling trade message: {e}")

    async def close(self):
        """Close the WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("🔌 WebSocket closed")