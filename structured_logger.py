"""
Structured JSON Logging for Trading Bot

Provides comprehensive, structured logging with correlation IDs,
context information, and proper log levels for monitoring and debugging.

Features:
- JSON formatted logs for easy parsing
- Correlation IDs for request tracing
- Rich context information (trading data, system state)
- Proper log levels and filtering
- Performance-optimized logging
- Integration with metrics collection
"""

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union, List
from contextvars import ContextVar
from pathlib import Path


# Context variables for correlation IDs and context
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
request_context_var: ContextVar[Optional[Dict[str, Any]]] = ContextVar('request_context', default=None)


class StructuredJSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if available
        correlation_id = correlation_id_var.get()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id

        # Add request context if available
        request_context = request_context_var.get()
        if request_context:
            log_entry["context"] = request_context

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                    'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                    'thread', 'threadName', 'processName', 'process', 'message'
                }:
                    # Sanitize values to ensure JSON serializable
                    try:
                        json.dumps(value)
                        log_entry[f"extra_{key}"] = value
                    except (TypeError, ValueError):
                        log_entry[f"extra_{key}"] = str(value)

        # Add thread and process info
        log_entry["thread_id"] = record.thread
        log_entry["thread_name"] = record.threadName
        log_entry["process_id"] = record.process

        return json.dumps(log_entry, separators=(',', ':'))


class CorrelationFilter(logging.Filter):
    """Filter to add correlation ID and context to all log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation and context information to log record."""
        correlation_id = correlation_id_var.get()
        if correlation_id:
            record.correlation_id = correlation_id

        request_context = request_context_var.get()
        if request_context:
            record.context = request_context

        return True


class TradingLogger:
    """
    Structured logger for trading bot with context awareness.

    Provides convenient methods for logging trading-related events
    with rich context information.
    """

    def __init__(self, name: str = "trading_bot"):
        self.logger = logging.getLogger(name)
        self._local = threading.local()

    def debug(self, message: str, *args, **kwargs):
        """Log debug message."""
        return self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Log info message."""
        return self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log warning message."""
        return self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log error message."""
        return self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """Log critical message."""
        return self.logger.critical(message, *args, **kwargs)

    def set_correlation_id(self, correlation_id: Optional[str] = None) -> str:
        """Set correlation ID for current context."""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        correlation_id_var.set(correlation_id)
        self._local.correlation_id = correlation_id
        return correlation_id

    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID."""
        return correlation_id_var.get()

    def set_context(self, **context: Any) -> None:
        """Set logging context for current operation."""
        current_context = request_context_var.get() or {}
        current_context.update(context)
        request_context_var.set(current_context)
        self._local.context = current_context

    def clear_context(self) -> None:
        """Clear logging context."""
        request_context_var.set(None)
        if hasattr(self._local, 'context'):
            delattr(self._local, 'context')

    def log_trade_entry(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "LIMIT",
        strategy: str = "MACD",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        indicators: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log trade entry with full context."""
        self.set_context(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
            strategy=strategy,
            stop_loss=stop_loss,
            take_profit=take_profit,
            indicators=indicators or {},
            event_type="trade_entry"
        )

        risk_reward = None
        if stop_loss and take_profit and price:
            risk = abs(price - stop_loss)
            reward = abs(take_profit - price)
            risk_reward = reward / risk if risk > 0 else 0

        self.logger.info(
            f"TRADE ENTRY: {side} {quantity} {symbol} @ ${price:.2f} "
            f"(RR: {risk_reward:.2f})",
            extra={
                "trade_entry": {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "order_type": order_type,
                    "strategy": strategy,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "risk_reward_ratio": risk_reward,
                    "indicators": indicators
                }
            }
        )

        self.clear_context()

    def log_trade_exit(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        exit_reason: str,
        duration_minutes: Optional[float] = None
    ) -> None:
        """Log trade exit with P&L information."""
        pnl_percentage = (pnl / (entry_price * quantity)) * 100 if entry_price and quantity else 0

        self.set_context(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_percentage=pnl_percentage,
            exit_reason=exit_reason,
            duration_minutes=duration_minutes,
            event_type="trade_exit"
        )

        outcome = "PROFIT" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"

        self.logger.info(
            f"TRADE EXIT: {outcome} {side} {quantity} {symbol} "
            f"Entry: ${entry_price:.2f} Exit: ${exit_price:.2f} "
            f"P&L: ${pnl:.2f} ({pnl_percentage:.2f}%) - {exit_reason}",
            extra={
                "trade_exit": {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_percentage": pnl_percentage,
                    "exit_reason": exit_reason,
                    "duration_minutes": duration_minutes,
                    "outcome": outcome
                }
            }
        )

        self.clear_context()

    def log_signal_generated(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        indicators: Dict[str, Any],
        reason: str = ""
    ) -> None:
        """Log trading signal generation."""
        self.set_context(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            indicators=indicators,
            reason=reason,
            event_type="signal_generated"
        )

        self.logger.info(
            f"SIGNAL: {signal_type} {symbol} (confidence: {confidence:.2f}) - {reason}",
            extra={
                "signal": {
                    "symbol": symbol,
                    "type": signal_type,
                    "confidence": confidence,
                    "indicators": indicators,
                    "reason": reason
                }
            }
        )

        self.clear_context()

    def log_anomaly_detected(self, anomaly_score: Any) -> None:
        """Log detected anomaly."""
        self.set_context(
            anomaly_metric=anomaly_score.metric_name,
            anomaly_value=anomaly_score.value,
            anomaly_score=anomaly_score.score,
            anomaly_threshold=anomaly_score.threshold,
            event_type="anomaly_detected"
        )

        score_str = f"{anomaly_score.score:.2f}" if isinstance(anomaly_score.score, (int, float)) else str(anomaly_score.score)
        threshold_str = f"{anomaly_score.threshold:.2f}" if isinstance(anomaly_score.threshold, (int, float)) else str(anomaly_score.threshold)
        value_str = f"{anomaly_score.value:.2f}" if isinstance(anomaly_score.value, (int, float)) else str(anomaly_score.value)

        self.logger.warning(
            f"ANOMALY DETECTED: {anomaly_score.metric_name} = {value_str} "
            f"(score: {score_str}, threshold: {threshold_str})",
            extra={
                "anomaly": {
                    "metric": anomaly_score.metric_name,
                    "value": anomaly_score.value,
                    "score": anomaly_score.score,
                    "threshold": anomaly_score.threshold,
                    "method": anomaly_score.method,
                    "context": anomaly_score.context
                }
            }
        )

        self.clear_context()

    def log_risk_warning(
        self,
        warning_type: str,
        symbol: str,
        current_value: float,
        threshold: float,
        message: str
    ) -> None:
        """Log risk management warnings."""
        severity = "WARNING" if current_value < threshold * 1.5 else "CRITICAL"

        self.set_context(
            symbol=symbol,
            warning_type=warning_type,
            current_value=current_value,
            threshold=threshold,
            severity=severity,
            event_type="risk_warning"
        )

        log_method = self.logger.warning if severity == "WARNING" else self.logger.critical

        log_method(
            f"RISK {severity}: {warning_type} - {message} "
            f"(Current: {current_value:.2f}, Threshold: {threshold:.2f})",
            extra={
                "risk_warning": {
                    "type": warning_type,
                    "symbol": symbol,
                    "current_value": current_value,
                    "threshold": threshold,
                    "severity": severity,
                    "message": message
                }
            }
        )

        self.clear_context()

    def log_api_call(
        self,
        endpoint: str,
        method: str,
        duration: float,
        status_code: int,
        error: Optional[str] = None
    ) -> None:
        """Log API call with performance metrics."""
        success = status_code >= 200 and status_code < 300
        level = "INFO" if success else "WARNING" if status_code >= 400 else "ERROR"

        self.set_context(
            endpoint=endpoint,
            method=method,
            duration=duration,
            status_code=status_code,
            success=success,
            error=error,
            event_type="api_call"
        )

        status_msg = f"HTTP {status_code}"
        if error:
            status_msg += f" - {error}"

        self.logger.info(
            f"API CALL: {method} {endpoint} - {status_msg} ({duration:.3f}s)",
            extra={
                "api_call": {
                    "endpoint": endpoint,
                    "method": method,
                    "duration": duration,
                    "status_code": status_code,
                    "success": success,
                    "error": error
                }
            }
        )

        self.clear_context()

    def log_websocket_event(
        self,
        event_type: str,
        url: str,
        connected: bool,
        reconnect_count: int = 0,
        error: Optional[str] = None
    ) -> None:
        """Log WebSocket connection events."""
        self.set_context(
            event_type=event_type,
            url=url,
            connected=connected,
            reconnect_count=reconnect_count,
            error=error,
            websocket_event=True
        )

        if event_type == "connected":
            self.logger.info(f"WEBSOCKET: Connected to {url}")
        elif event_type == "disconnected":
            self.logger.warning(f"WEBSOCKET: Disconnected from {url} (reconnects: {reconnect_count})")
        elif event_type == "error":
            self.logger.error(f"WEBSOCKET: Error on {url} - {error}")
        elif event_type == "reconnecting":
            self.logger.warning(f"WEBSOCKET: Reconnecting to {url} (attempt {reconnect_count})")

        self.clear_context()

    def log_system_health(
        self,
        cpu_percent: float,
        memory_percent: float,
        disk_percent: float,
        websocket_connected: bool
    ) -> None:
        """Log system health metrics."""
        health_status = "HEALTHY"
        if cpu_percent > 80 or memory_percent > 80 or disk_percent > 90 or not websocket_connected:
            health_status = "WARNING"
        if cpu_percent > 95 or memory_percent > 95 or not websocket_connected:
            health_status = "CRITICAL"

        self.set_context(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            websocket_connected=websocket_connected,
            health_status=health_status,
            event_type="system_health"
        )

        if health_status == "HEALTHY":
            self.logger.debug(f"SYSTEM HEALTH: CPU {cpu_percent:.1f}%, MEM {memory_percent:.1f}%, DISK {disk_percent:.1f}%, WS {'✓' if websocket_connected else '✗'}")
        elif health_status == "WARNING":
            self.logger.warning(f"SYSTEM HEALTH WARNING: CPU {cpu_percent:.1f}%, MEM {memory_percent:.1f}%, DISK {disk_percent:.1f}%, WS {'✓' if websocket_connected else '✗'}")
        else:
            self.logger.critical(f"SYSTEM HEALTH CRITICAL: CPU {cpu_percent:.1f}%, MEM {memory_percent:.1f}%, DISK {disk_percent:.1f}%, WS {'✗' if not websocket_connected else '✓'}")

        self.clear_context()


def setup_structured_logging(
    log_level: str = "INFO",
    log_file: str = "logs/trading_bot.log",
    error_log_file: str = "logs/trading_bot_errors.log",
    console_level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Set up structured JSON logging for the trading bot.

    Args:
        log_level: Overall log level
        log_file: Main log file path
        error_log_file: Error-only log file path
        console_level: Console log level
        max_bytes: Max log file size
        backup_count: Number of backup files to keep
    """
    from logging.handlers import RotatingFileHandler

    # Create logs directory
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create JSON formatter
    json_formatter = StructuredJSONFormatter()

    # Create correlation filter
    correlation_filter = CorrelationFilter()

    # File handler for all logs (JSON format)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(json_formatter)
    file_handler.addFilter(correlation_filter)
    root_logger.addHandler(file_handler)

    # Error-only file handler
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=max_bytes // 2,  # Smaller for errors
        backupCount=backup_count,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(json_formatter)
    error_handler.addFilter(correlation_filter)
    root_logger.addHandler(error_handler)

    # Console handler (human-readable)
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Log setup completion
    root_logger.info("Structured JSON logging initialized", extra={
        "logging_setup": {
            "log_level": log_level,
            "log_file": log_file,
            "error_log_file": error_log_file,
            "console_level": console_level,
            "max_bytes": max_bytes,
            "backup_count": backup_count
        }
    })


# Global logger instance
_trading_logger: Optional[TradingLogger] = None


def get_trading_logger() -> TradingLogger:
    """Get global trading logger instance."""
    global _trading_logger
    if _trading_logger is None:
        _trading_logger = TradingLogger()
    return _trading_logger


def initialize_trading_logger(name: str = "trading_bot") -> TradingLogger:
    """Initialize global trading logger."""
    global _trading_logger
    _trading_logger = TradingLogger(name)
    return _trading_logger


# Context manager for correlation IDs
class CorrelationContext:
    """Context manager for setting correlation ID."""

    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.previous_id = None
        self.previous_context = None

    def __enter__(self):
        self.previous_id = correlation_id_var.get()
        self.previous_context = request_context_var.get()
        correlation_id_var.set(self.correlation_id)
        return self.correlation_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        correlation_id_var.set(self.previous_id)
        request_context_var.set(self.previous_context)


# Convenience functions
def with_correlation_id(correlation_id: Optional[str] = None):
    """Decorator to set correlation ID for function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with CorrelationContext(correlation_id):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def log_trading_context(**context):
    """Context manager to set trading context for logs."""
    return _TradingContext(context)


class _TradingContext:
    """Context manager for trading context."""

    def __init__(self, context: Dict[str, Any]):
        self.context = context
        self.previous_context = None

    def __enter__(self):
        self.previous_context = request_context_var.get()
        request_context_var.set(self.context)

    def __exit__(self, exc_type, exc_val, exc_tb):
        request_context_var.set(self.previous_context)
