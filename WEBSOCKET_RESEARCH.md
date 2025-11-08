# WebSocket Streaming Research for Real-Time Data

**Date:** November 8, 2025  
**Status:** 🔍 RESEARCH COMPLETE  
**Priority:** Future Enhancement

## Executive Summary

Research into WebSocket streaming for Hyperliquid API to replace current polling mechanism. WebSocket support would enable real-time market data updates, faster signal detection, and lower latency trading decisions.

## Current Implementation

### Polling-Based Approach

**Current Method:**
- Uses REST API polling via `get_klines()` and `get_ticker()`
- Polls every `check_interval` seconds (default: 300s = 5 minutes)
- Fetches candle data incrementally when cache is valid
- Uses smart caching to minimize API calls

**Current Flow:**
```
Trading Cycle → Check Cache → Fetch New Candles → Process → Sleep(check_interval) → Repeat
```

**Limitations:**
- **Latency**: Up to `check_interval` seconds delay before detecting new signals
- **API Load**: Still requires periodic REST API calls
- **Missed Opportunities**: Signals may occur between polling intervals
- **Resource Usage**: Constant polling consumes network and CPU resources

## Hyperliquid WebSocket API Research

### Official Documentation

**Hyperliquid API Documentation:**
- Main Docs: https://hyperliquid.gitbook.io/hyperliquid-docs
- API Reference: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api

### WebSocket Endpoints

Based on research and Hyperliquid's architecture:

**Mainnet WebSocket:**
- `wss://api.hyperliquid.xyz/ws`

**Testnet WebSocket:**
- `wss://api.hyperliquid-testnet.xyz/ws`

### Available WebSocket Channels

Hyperliquid WebSocket API supports:

1. **Market Data Streams:**
   - **Trades**: Real-time executed trades for specific coins
   - **Order Book**: Real-time bid/ask updates
   - **Candles**: OHLCV data for specific intervals (if supported)
   - **Ticker**: Real-time price and volume updates

2. **User Data Streams:**
   - Position updates (via user data subscription)
   - Order status updates
   - Account balance updates
   - Fill notifications

3. **Subscription Model:**
   - Subscribe to specific coins/channels using `method: "subscribe"`
   - Each subscription requires a unique identifier
   - Unsubscribe when no longer needed
   - Connection health maintained via ping/pong

### WebSocket Message Format

**Subscription Message (Trades Example):**
```json
{
  "method": "subscribe",
  "subscription": {
    "type": "trades",
    "coin": "BTC"
  }
}
```

**Subscription Message (Other Types):**
```json
{
  "method": "subscribe",
  "subscription": {
    "type": "orderbook",  // or "candle", "ticker", etc.
    "coin": "BTC"
  }
}
```

**Incoming Data Message:**
```json
{
  "channel": "trades",
  "data": {
    "coin": "BTC",
    "side": "B",  // B = Buy, A = Ask
    "px": "50000.0",
    "sz": "0.1",
    "time": 1699123456789,
    "hash": "0x..."
  }
}
```

**Order Execution via WebSocket:**
```json
{
  "method": "post",
  "id": 1,  // Unique identifier for request tracking
  "request": {
    "type": "action",
    "payload": {
      "action": {
        "type": "order",
        "orders": [
          {
            "a": 0,  // Asset index (0 = BTC)
            "b": true,  // is_buy
            "p": "50000",  // Price
            "s": "0.1",  // Size
            "r": false,  // reduce_only
            "t": {
              "limit": {
                "tif": "Gtc"
              }
            }
          }
        ],
        "grouping": "na"
      },
      "nonce": 1713825891591,
      "signature": {
        "r": "...",
        "s": "...",
        "v": "..."
      },
      "vaultAddress": "0x..."  // Optional
    }
  }
}
```

### Rate Limits

Hyperliquid WebSocket API has specific rate limits:

- **Maximum 100 WebSocket connections** per user/IP
- **Maximum 1,000 WebSocket subscriptions** across all connections
- **Maximum 2,000 messages sent per minute** across all WebSocket connections
- **Maximum 100 simultaneous inflight post messages** (order requests)

**Important**: These limits apply across all WebSocket connections, so careful connection management is required.

## Benefits of WebSocket Implementation

### 1. Real-Time Signal Detection
- **Current**: Up to 5-minute delay (with 300s check_interval)
- **WebSocket**: < 1 second latency for new candles
- **Impact**: Catch entry/exit signals immediately

### 2. Reduced API Load
- **Current**: REST API call every check_interval
- **WebSocket**: Single persistent connection, push-based updates
- **Impact**: Lower server load, fewer rate limit concerns

### 3. Better Resource Efficiency
- **Current**: Constant polling consumes CPU/network
- **WebSocket**: Event-driven, only processes when data arrives
- **Impact**: Lower CPU usage, more efficient

### 4. Improved Trading Performance
- **Faster Entry**: Detect signals as they form
- **Better Exit Timing**: Real-time price updates for stop-loss/take-profit
- **Reduced Slippage**: Faster order execution

## Implementation Approach

### Architecture Design

```
┌─────────────────┐
│  Trading Bot    │
│   (Main Loop)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ WebSocket Client│ ◄─── WebSocket Connection
│   (New Class)   │     (wss://api.hyperliquid.xyz/ws)
└────────┬────────┘
         │
         ├──► Market Data Handler
         │    - Candle updates
         │    - Ticker updates
         │    - Order book updates
         │
         └──► User Data Handler
              - Position updates
              - Order status
              - Balance updates
```

### Proposed Implementation

**1. New WebSocket Client Class** (`hyperliquid_websocket.py`):
```python
import asyncio
import json
import websockets
from typing import Dict, Callable, Optional

class HyperliquidWebSocketClient:
    """WebSocket client for real-time Hyperliquid data."""
    
    def __init__(self, testnet: bool = True):
        self.ws_url = (
            "wss://api.hyperliquid-testnet.xyz/ws" 
            if testnet 
            else "wss://api.hyperliquid.xyz/ws"
        )
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.subscriptions: Dict[str, Dict] = {}
        self.callbacks: Dict[str, Callable] = {}
        self.running = False
        self.message_id = 0
    
    async def connect(self):
        """Establish WebSocket connection."""
        try:
            self.websocket = await websockets.connect(self.ws_url)
            self.running = True
            logger.info(f"WebSocket connected to {self.ws_url}")
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    async def subscribe_trades(self, coin: str, callback: Callable):
        """Subscribe to trade updates for a coin.
        
        Args:
            coin: Coin symbol (e.g., "BTC", "ETH", "SOL")
            callback: Function to call when trade data is received
        """
        subscription = {
            "method": "subscribe",
            "subscription": {
                "type": "trades",
                "coin": coin
            }
        }
        await self._send_message(subscription)
        self.subscriptions[f"trades_{coin}"] = subscription
        self.callbacks[f"trades_{coin}"] = callback
    
    async def subscribe_orderbook(self, coin: str, callback: Callable):
        """Subscribe to order book updates."""
        subscription = {
            "method": "subscribe",
            "subscription": {
                "type": "orderbook",
                "coin": coin
            }
        }
        await self._send_message(subscription)
        self.subscriptions[f"orderbook_{coin}"] = subscription
        self.callbacks[f"orderbook_{coin}"] = callback
    
    async def _send_message(self, message: Dict):
        """Send message to WebSocket."""
        if self.websocket:
            await self.websocket.send(json.dumps(message))
    
    async def listen(self):
        """Listen for incoming messages."""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self._handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.running = False
        except Exception as e:
            logger.error(f"Error in WebSocket listener: {e}")
            self.running = False
    
    async def _handle_message(self, data: Dict):
        """Handle incoming WebSocket message."""
        channel = data.get("channel")
        if channel:
            callback_key = f"{channel}_{data.get('coin', '')}"
            if callback_key in self.callbacks:
                self.callbacks[callback_key](data)
    
    async def disconnect(self):
        """Close WebSocket connection."""
        self.running = False
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket disconnected")
```

**2. Integration with Trading Bot**:
```python
import asyncio
import threading

class TradingBot:
    def __init__(self, ...):
        # Existing REST client for order placement
        self.client = HyperliquidClient(...)
        
        # New WebSocket client for real-time data
        self.ws_client = HyperliquidWebSocketClient(testnet=self.testnet)
        self.ws_thread = None
        self.ws_loop = None
        
    async def setup_websocket_async(self):
        """Initialize WebSocket subscriptions (async)."""
        await self.ws_client.connect()
        
        # Get coin symbol (e.g., "BTC" from "BTCUSDT")
        coin = self.symbol.replace("USDT", "")
        
        # Subscribe to trades for real-time price updates
        await self.ws_client.subscribe_trades(
            coin=coin,
            callback=self._on_trade_update
        )
        
        # Subscribe to orderbook for depth data
        await self.ws_client.subscribe_orderbook(
            coin=coin,
            callback=self._on_orderbook_update
        )
        
        # Start listening in background
        asyncio.create_task(self.ws_client.listen())
    
    def setup_websocket(self):
        """Initialize WebSocket subscriptions (synchronous wrapper)."""
        def run_ws():
            self.ws_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.ws_loop)
            self.ws_loop.run_until_complete(self.setup_websocket_async())
            self.ws_loop.run_forever()
        
        self.ws_thread = threading.Thread(target=run_ws, daemon=True)
        self.ws_thread.start()
    
    def _on_trade_update(self, trade_data: Dict):
        """Handle trade updates from WebSocket."""
        # Extract price from trade data
        price = float(trade_data.get("px", 0))
        coin = trade_data.get("coin", "")
        
        # Update current price for stop-loss/take-profit checks
        if coin == self.symbol.replace("USDT", ""):
            self.current_price = price
            logger.debug(f"Real-time price update: ${price:.2f}")
            
            # Check if we need to update trailing stop or exit conditions
            if self.current_position:
                self._check_exit_conditions_realtime(price)
    
    def _on_orderbook_update(self, orderbook_data: Dict):
        """Handle orderbook updates from WebSocket."""
        # Can be used for advanced order placement strategies
        # (e.g., placing orders at best bid/ask)
        pass
    
    def _check_exit_conditions_realtime(self, current_price: float):
        """Check exit conditions with real-time price."""
        # Quick check for stop-loss/take-profit hits
        if self.current_position:
            stop_loss = self.current_position.get("stop_loss")
            take_profit = self.current_position.get("take_profit")
            
            if stop_loss and current_price <= stop_loss:
                logger.warning(f"Stop-loss hit at ${current_price:.2f}")
                self.close_position("Stop Loss Hit", current_price)
            elif take_profit and current_price >= take_profit:
                logger.info(f"Take-profit hit at ${current_price:.2f}")
                self.close_position("Take Profit Hit", current_price)
```

**3. Hybrid Approach** (Recommended):
- Use WebSocket for market data (candles, ticker)
- Keep REST API for order placement and account queries
- Fallback to REST polling if WebSocket disconnects

## Technical Considerations

### 1. Connection Management
- **Reconnection Logic**: Auto-reconnect on disconnect with exponential backoff
- **Heartbeat**: WebSocket ping/pong frames to keep connection alive
- **Error Handling**: Graceful fallback to REST API polling
- **Connection Limits**: Monitor connection count (max 100 per user/IP)
- **Thread Safety**: Use thread-safe queues for message passing between WebSocket thread and main thread

### 2. Data Synchronization
- **Initial Data Load**: Use REST API for historical data
- **WebSocket Updates**: Append new candles as they arrive
- **Cache Management**: Update cache in real-time

### 3. Threading/Async
- **Recommended**: Async/await with asyncio in separate thread
  - WebSocket runs in background thread with its own event loop
  - Main trading bot thread remains synchronous
  - Use thread-safe queues for communication
- **Alternative**: Full async/await migration (larger refactor)

### 4. Message Queue
- Use `queue.Queue` for thread-safe message passing
- Buffer incoming WebSocket messages
- Process in trading bot's main thread
- Prevent race conditions with locks
- Handle message ordering (trades may arrive out of order)

### 5. Rate Limit Management
- Monitor subscription count (max 1,000)
- Track message send rate (max 2,000/min)
- Implement backpressure if limits approached
- Use connection pooling if multiple symbols needed

## Dependencies Required

**Python WebSocket Libraries:**
- `websockets` (asyncio-based) - **Recommended**
- `websocket-client` (synchronous) - Alternative
- `python-socketio` (Socket.IO client) - Not needed for Hyperliquid

**Installation:**
```bash
pip install websockets  # Async/await support (recommended)
# OR
pip install websocket-client  # Synchronous alternative
```

**Additional Dependencies:**
- `asyncio` (built-in) - For async WebSocket handling
- `threading` (built-in) - For running WebSocket in background thread
- `queue` (built-in) - For thread-safe message passing

## Implementation Effort Estimate

### Phase 1: Basic WebSocket Client (1-2 days)
- WebSocket connection management
- Basic subscription/unsubscription
- Message parsing and callbacks

### Phase 2: Integration (1-2 days)
- Integrate with TradingBot
- Update market data cache from WebSocket
- Fallback to REST API

### Phase 3: Testing & Optimization (1 day)
- Test reconnection logic
- Performance testing
- Error handling refinement

**Total Estimated Effort:** 3-5 days (matches TODO.md estimate)

## Risks and Mitigation

### Risk 1: WebSocket Connection Instability
- **Mitigation**: Robust reconnection logic with exponential backoff, automatic fallback to REST API polling
- **Monitoring**: Track connection uptime, reconnection frequency

### Risk 2: Message Ordering
- **Mitigation**: Use timestamps for ordering, sequence numbers if available, handle out-of-order messages gracefully
- **Note**: Trade messages may arrive out of order - use timestamps to sort

### Risk 3: Data Consistency
- **Mitigation**: Periodic REST API sync for validation, compare WebSocket data with REST snapshots
- **Validation**: Cross-check critical data (prices, positions) between WebSocket and REST

### Risk 4: Increased Complexity
- **Mitigation**: Clean abstraction, comprehensive tests, gradual rollout
- **Testing**: Unit tests for WebSocket client, integration tests with mock server

### Risk 5: Rate Limit Exceeded
- **Mitigation**: Monitor subscription count, message rate, implement throttling
- **Prevention**: Limit subscriptions to essential channels only, batch operations when possible

## Comparison: Polling vs WebSocket

| Aspect | Current (Polling) | WebSocket |
|--------|------------------|-----------|
| **Latency** | Up to check_interval (300s) | < 1 second |
| **API Calls** | Every check_interval | Single connection |
| **CPU Usage** | Constant polling | Event-driven |
| **Complexity** | Low | Medium-High |
| **Reliability** | High (REST is stateless) | Medium (connection management) |
| **Resource Usage** | Network per poll | Persistent connection |

## Recommendations

### Short Term (Current)
- ✅ **Keep current polling approach** - It's working well with smart caching
- ✅ **Optimize check_interval** - Balance between latency and API load
- ✅ **Monitor API usage** - Ensure rate limits aren't hit

### Medium Term (Future Enhancement)
- 🔄 **Implement WebSocket for ticker updates** - Real-time price for TP/SL
- 🔄 **Hybrid approach** - WebSocket for data, REST for orders
- 🔄 **Gradual migration** - Test WebSocket alongside REST

### Long Term (Full Migration)
- 🎯 **Full WebSocket implementation** - All market data via WebSocket
- 🎯 **Real-time order updates** - WebSocket for order status
- 🎯 **Optimized latency** - Sub-second signal detection

## Next Steps

1. **Verify WebSocket API**: Test connection to Hyperliquid WebSocket endpoint
2. **Prototype**: Build minimal WebSocket client to test subscription
3. **Design Integration**: Plan how WebSocket integrates with existing code
4. **Implement Incrementally**: Start with ticker updates, then candles
5. **Test Thoroughly**: Ensure reliability and fallback mechanisms

## References

- [Hyperliquid WebSocket API Documentation](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket)
- [Hyperliquid Rate Limits](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits)
- [Hyperliquid Python SDK](https://github.com/hyperliquid-dex/hyperliquid-python-sdk) - May have WebSocket examples
- [WebSockets Library Documentation](https://websockets.readthedocs.io/)
- [WebSocket Client Library](https://websocket-client.readthedocs.io/)
- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)

## Conclusion

WebSocket streaming would provide significant benefits for real-time trading, but requires careful implementation to maintain reliability. The current polling approach with smart caching is efficient and reliable. WebSocket implementation should be considered as a future enhancement when:

1. Lower latency becomes critical
2. Trading frequency increases
3. Real-time price updates are needed for advanced strategies
4. Development resources are available for proper implementation and testing

**Recommendation**: Keep current polling approach for now, plan WebSocket as Phase 2 enhancement.

