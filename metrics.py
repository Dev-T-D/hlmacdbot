"""
Prometheus Metrics Collection for Trading Bot

Comprehensive metrics collection for monitoring trading bot performance,
health, and trading activity. Exports metrics on /metrics endpoint.

Metrics Categories:
- Trading metrics: trades, win rate, P&L, positions
- Performance metrics: API latency, order execution time
- System metrics: CPU, memory, websocket status
- Risk metrics: drawdown, daily loss, position sizes
- Strategy metrics: signals, entry conditions

"""

import time
import logging
import psutil
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info,
        generate_latest, CollectorRegistry, start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes for when prometheus is not available
    class DummyMetric:
        def __init__(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self

    Counter = Gauge = Histogram = Summary = Info = DummyMetric
    CollectorRegistry = object
    generate_latest = lambda: b"prometheus_client not installed"
    start_http_server = lambda *args: None


logger = logging.getLogger(__name__)


class TradingBotMetrics:
    """
    Comprehensive metrics collection for trading bot monitoring.

    Uses Prometheus client to expose metrics for monitoring and alerting.
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize trading bot metrics.

        Args:
            registry: Custom Prometheus registry (optional)
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning("prometheus_client not installed. Metrics will be no-op.")
            return

        self.registry = registry or CollectorRegistry()
        self._lock = Lock()

        # Info metric for bot metadata
        self.bot_info = Info(
            'trading_bot_info',
            'Trading bot information',
            ['version', 'exchange', 'symbol', 'strategy'],
            registry=self.registry
        )

        # TRADING METRICS
        # Counters for cumulative trading activity
        self.trades_total = Counter(
            'trading_bot_trades_total',
            'Total number of trades executed',
            ['symbol', 'side', 'outcome'],
            registry=self.registry
        )

        self.signals_generated = Counter(
            'trading_bot_signals_generated_total',
            'Total number of trading signals generated',
            ['symbol', 'signal_type', 'reason'],
            registry=self.registry
        )

        # Gauges for current state
        self.current_positions = Gauge(
            'trading_bot_current_positions',
            'Current number of open positions',
            ['symbol'],
            registry=self.registry
        )

        self.account_balance = Gauge(
            'trading_bot_account_balance',
            'Current account balance',
            ['currency'],
            registry=self.registry
        )

        self.unrealized_pnl = Gauge(
            'trading_bot_unrealized_pnl',
            'Current unrealized P&L',
            ['symbol', 'position_type'],
            registry=self.registry
        )

        self.win_rate = Gauge(
            'trading_bot_win_rate_ratio',
            'Current win rate (0.0 to 1.0)',
            ['symbol', 'timeframe'],
            registry=self.registry
        )

        # PERFORMANCE METRICS
        # Histograms for latency distributions
        self.api_request_duration = Histogram(
            'trading_bot_api_request_duration_seconds',
            'API request duration in seconds',
            ['endpoint', 'method', 'status'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
            registry=self.registry
        )

        self.order_execution_duration = Histogram(
            'trading_bot_order_execution_duration_seconds',
            'Order execution duration from placement to fill',
            ['symbol', 'side', 'order_type'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
            registry=self.registry
        )

        self.strategy_calculation_duration = Histogram(
            'trading_bot_strategy_calculation_duration_seconds',
            'Time spent calculating trading strategy',
            ['symbol', 'calculation_type'],
            buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0),
            registry=self.registry
        )

        # SYSTEM METRICS
        self.websocket_connected = Gauge(
            'trading_bot_websocket_connected',
            'WebSocket connection status (1=connected, 0=disconnected)',
            ['url'],
            registry=self.registry
        )

        self.memory_usage_bytes = Gauge(
            'trading_bot_memory_usage_bytes',
            'Current memory usage in bytes',
            registry=self.registry
        )

        self.cpu_usage_percent = Gauge(
            'trading_bot_cpu_usage_percent',
            'Current CPU usage percentage',
            registry=self.registry
        )

        self.disk_usage_bytes = Gauge(
            'trading_bot_disk_usage_bytes',
            'Disk usage in bytes',
            ['path'],
            registry=self.registry
        )

        # RISK METRICS
        self.current_drawdown_percent = Gauge(
            'trading_bot_current_drawdown_percent',
            'Current drawdown percentage',
            registry=self.registry
        )

        self.daily_loss_percent = Gauge(
            'trading_bot_daily_loss_percent',
            'Daily loss percentage',
            registry=self.registry
        )

        self.max_position_size_percent = Gauge(
            'trading_bot_max_position_size_percent',
            'Maximum position size as percentage of account',
            ['symbol'],
            registry=self.registry
        )

        self.risk_limit_utilization = Gauge(
            'trading_bot_risk_limit_utilization_ratio',
            'Risk limit utilization (0.0 to 1.0)',
            ['limit_type'],
            registry=self.registry
        )

        # STRATEGY METRICS
        self.entry_conditions_met = Counter(
            'trading_bot_entry_conditions_met_total',
            'Number of times entry conditions were met',
            ['symbol', 'condition_type'],
            registry=self.registry
        )

        self.exit_conditions_met = Counter(
            'trading_bot_exit_conditions_met_total',
            'Number of times exit conditions were met',
            ['symbol', 'exit_reason'],
            registry=self.registry
        )

        self.trailing_stop_updates = Counter(
            'trading_bot_trailing_stop_updates_total',
            'Number of trailing stop updates',
            ['symbol', 'direction'],
            registry=self.registry
        )

        # ERROR AND HEALTH METRICS
        self.api_errors_total = Counter(
            'trading_bot_api_errors_total',
            'Total number of API errors',
            ['endpoint', 'error_type'],
            registry=self.registry
        )

        self.websocket_reconnects_total = Counter(
            'trading_bot_websocket_reconnects_total',
            'Total number of WebSocket reconnections',
            registry=self.registry
        )

        self.circuit_breaker_trips = Counter(
            'trading_bot_circuit_breaker_trips_total',
            'Total number of circuit breaker trips',
            ['reason'],
            registry=self.registry
        )

        # PERFORMANCE SUMMARY METRICS
        self.total_pnl = Gauge(
            'trading_bot_total_pnl',
            'Total P&L since bot start',
            ['currency'],
            registry=self.registry
        )

        self.uptime_seconds = Gauge(
            'trading_bot_uptime_seconds',
            'Bot uptime in seconds',
            registry=self.registry
        )

        self.last_trade_timestamp = Gauge(
            'trading_bot_last_trade_timestamp',
            'Timestamp of last trade',
            registry=self.registry
        )

        self.last_signal_timestamp = Gauge(
            'trading_bot_last_signal_timestamp',
            'Timestamp of last trading signal',
            registry=self.registry
        )

        # Initialize bot start time
        self._bot_start_time = time.time()
        self.uptime_seconds.set(self._bot_start_time)

        logger.info("Trading bot metrics initialized")

    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        pnl: float = 0,
        outcome: str = "unknown"
    ):
        """Record a completed trade."""
        if not PROMETHEUS_AVAILABLE:
            return

        with self._lock:
            self.trades_total.labels(symbol=symbol, side=side, outcome=outcome).inc()
            self.last_trade_timestamp.set(time.time())
            self.total_pnl.inc(pnl)

            # Update win rate (simplified calculation)
            # In production, you'd track this more accurately
            if outcome in ['win', 'profit']:
                # This is a simplified win rate - you'd want more sophisticated tracking
                pass

    def record_signal(self, symbol: str, signal_type: str, reason: str = ""):
        """Record a trading signal generation."""
        if not PROMETHEUS_AVAILABLE:
            return

        with self._lock:
            self.signals_generated.labels(
                symbol=symbol,
                signal_type=signal_type,
                reason=reason
            ).inc()
            self.last_signal_timestamp.set(time.time())

    def update_position_metrics(self, symbol: str, position_count: int, unrealized_pnl: float = 0):
        """Update position-related metrics."""
        if not PROMETHEUS_AVAILABLE:
            return

        with self._lock:
            self.current_positions.labels(symbol=symbol).set(position_count)
            if position_count > 0:
                self.unrealized_pnl.labels(symbol=symbol, position_type='active').set(unrealized_pnl)

    def update_account_metrics(self, balance: float, currency: str = "USDC"):
        """Update account balance metrics."""
        if not PROMETHEUS_AVAILABLE:
            return

        with self._lock:
            self.account_balance.labels(currency=currency).set(balance)

    def record_api_request(self, endpoint: str, method: str, duration: float, status: str = "success"):
        """Record API request metrics."""
        if not PROMETHEUS_AVAILABLE:
            return

        with self._lock:
            self.api_request_duration.labels(
                endpoint=endpoint,
                method=method,
                status=status
            ).observe(duration)

            if status != "success":
                self.api_errors_total.labels(endpoint=endpoint, error_type=status).inc()

    def record_order_execution(self, symbol: str, side: str, order_type: str, duration: float):
        """Record order execution time."""
        if not PROMETHEUS_AVAILABLE:
            return

        with self._lock:
            self.order_execution_duration.labels(
                symbol=symbol,
                side=side,
                order_type=order_type
            ).observe(duration)

    def record_strategy_calculation(self, symbol: str, calculation_type: str, duration: float):
        """Record strategy calculation time."""
        if not PROMETHEUS_AVAILABLE:
            return

        with self._lock:
            self.strategy_calculation_duration.labels(
                symbol=symbol,
                calculation_type=calculation_type
            ).observe(duration)

    def update_system_metrics(self):
        """Update system resource metrics."""
        if not PROMETHEUS_AVAILABLE:
            return

        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent(interval=0.1)

            with self._lock:
                self.memory_usage_bytes.set(memory_info.rss)
                self.cpu_usage_percent.set(cpu_percent)

                # Disk usage for logs directory
                try:
                    import os
                    logs_path = "logs"
                    if os.path.exists(logs_path):
                        disk_usage = psutil.disk_usage(logs_path)
                        self.disk_usage_bytes.labels(path=logs_path).set(disk_usage.used)
                except Exception:
                    pass  # Ignore disk usage errors

        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")

    def update_risk_metrics(
        self,
        current_drawdown: float = 0,
        daily_loss_pct: float = 0,
        max_position_pct: float = 0
    ):
        """Update risk-related metrics."""
        if not PROMETHEUS_AVAILABLE:
            return

        with self._lock:
            self.current_drawdown_percent.set(current_drawdown)
            self.daily_loss_percent.set(daily_loss_pct)
            self.max_position_size_percent.set(max_position_pct)

    def update_websocket_status(self, url: str, connected: bool):
        """Update WebSocket connection status."""
        if not PROMETHEUS_AVAILABLE:
            return

        with self._lock:
            self.websocket_connected.labels(url=url).set(1 if connected else 0)

    def record_entry_condition(self, symbol: str, condition_type: str):
        """Record when entry conditions are met."""
        if not PROMETHEUS_AVAILABLE:
            return

        with self._lock:
            self.entry_conditions_met.labels(symbol=symbol, condition_type=condition_type).inc()

    def record_exit_condition(self, symbol: str, exit_reason: str):
        """Record when exit conditions are met."""
        if not PROMETHEUS_AVAILABLE:
            return

        with self._lock:
            self.exit_conditions_met.labels(symbol=symbol, exit_reason=exit_reason).inc()

    def record_trailing_stop_update(self, symbol: str, direction: str):
        """Record trailing stop updates."""
        if not PROMETHEUS_AVAILABLE:
            return

        with self._lock:
            self.trailing_stop_updates.labels(symbol=symbol, direction=direction).inc()

    def record_websocket_reconnect(self):
        """Record WebSocket reconnection."""
        if not PROMETHEUS_AVAILABLE:
            return

        with self._lock:
            self.websocket_reconnects_total.inc()

    def record_circuit_breaker_trip(self, reason: str):
        """Record circuit breaker activation."""
        if not PROMETHEUS_AVAILABLE:
            return

        with self._lock:
            self.circuit_breaker_trips.labels(reason=reason).inc()

    def set_bot_info(self, version: str = "1.0.0", exchange: str = "hyperliquid",
                    symbol: str = "BTC-USDC", strategy: str = "MACD"):
        """Set bot information metadata."""
        if not PROMETHEUS_AVAILABLE:
            return

        self.bot_info.labels(
            version=version,
            exchange=exchange,
            symbol=symbol,
            strategy=strategy
        )

    def get_metrics_text(self) -> bytes:
        """Get metrics in Prometheus text format."""
        if not PROMETHEUS_AVAILABLE:
            return b"prometheus_client not installed"

        # Update uptime
        self.uptime_seconds.set(time.time() - self._bot_start_time)

        return generate_latest(self.registry)

    def start_metrics_server(self, port: int = 8000):
        """Start Prometheus metrics HTTP server."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Cannot start metrics server - prometheus_client not installed")
            return

        try:
            start_http_server(port, registry=self.registry)
            logger.info(f"Metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")


# Global metrics instance
_metrics_instance: Optional[TradingBotMetrics] = None


def get_metrics() -> TradingBotMetrics:
    """Get global metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = TradingBotMetrics()
    return _metrics_instance


def initialize_metrics(
    port: int = 8000,
    start_server: bool = True,
    version: str = "1.0.0",
    exchange: str = "hyperliquid",
    symbol: str = "BTC-USDC",
    strategy: str = "MACD"
) -> TradingBotMetrics:
    """Initialize global metrics instance."""
    global _metrics_instance
    _metrics_instance = TradingBotMetrics()

    # Set bot info
    _metrics_instance.set_bot_info(version, exchange, symbol, strategy)

    # Start metrics server
    if start_server:
        _metrics_instance.start_metrics_server(port)

    return _metrics_instance
