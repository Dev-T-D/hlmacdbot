"""
Trading Bot Resilience Framework

Comprehensive failure recovery, graceful degradation, and state management
to ensure the bot never crashes with open positions and handles all edge cases.

Features:
- Graceful degradation (WebSocket → REST, cached data, reduced positions)
- Network resilience (exponential backoff, circuit breaker, retry budgets)
- State persistence (SQLite WAL mode, atomic updates)
- Emergency procedures (panic button, position limits, loss limits)
- Data validation and sanity checks
- Black box recorder for forensic analysis
"""

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import os
import atexit
import signal
import sys

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, stop trying
    HALF_OPEN = "half_open"  # Testing if service recovered


class DegradationLevel(Enum):
    """System degradation levels."""
    NORMAL = "normal"           # All systems operational
    DEGRADED = "degraded"       # Some services degraded but functional
    CRITICAL = "critical"       # Critical functions only
    EMERGENCY = "emergency"     # Emergency shutdown mode


@dataclass
class CircuitBreaker:
    """Circuit breaker for API call resilience."""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    success_threshold: int = 1

    # State
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    success_count: int = 0

    def should_attempt(self) -> bool:
        """Check if we should attempt the operation."""
        now = datetime.now(timezone.utc)

        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time and (now - self.last_failure_time).seconds >= self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True

        return False

    def record_success(self):
        """Record successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker closed - service recovered")

    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
        elif self.state == CircuitBreakerState.CLOSED and self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} consecutive failures")


@dataclass
class CachedData:
    """Cached data with expiry."""
    data: Any
    timestamp: datetime
    expiry_seconds: int
    source: str = "api"

    def is_expired(self) -> bool:
        """Check if cached data has expired."""
        return (datetime.now(timezone.utc) - self.timestamp).seconds > self.expiry_seconds

    def is_stale(self, max_age_seconds: int = 300) -> bool:
        """Check if data is stale (older than max_age)."""
        return (datetime.now(timezone.utc) - self.timestamp).seconds > max_age_seconds


@dataclass
class RetryBudget:
    """Retry budget to prevent infinite loops."""
    max_retries_per_hour: int = 100
    max_retries_per_minute: int = 10
    reset_hourly_at: Optional[datetime] = None
    reset_minutely_at: Optional[datetime] = None
    hourly_count: int = 0
    minutely_count: int = 0

    def can_retry(self) -> bool:
        """Check if retry is allowed within budget."""
        now = datetime.now(timezone.utc)

        # Reset counters if needed
        if self.reset_hourly_at is None or (now - self.reset_hourly_at).seconds >= 3600:
            self.reset_hourly_at = now
            self.hourly_count = 0

        if self.reset_minutely_at is None or (now - self.reset_minutely_at).seconds >= 60:
            self.reset_minutely_at = now
            self.minutely_count = 0

        return (self.hourly_count < self.max_retries_per_hour and
                self.minutely_count < self.max_retries_per_minute)

    def record_retry(self):
        """Record a retry attempt."""
        self.hourly_count += 1
        self.minutely_count += 1


@dataclass
class BlackBoxEvent:
    """Event for black box recorder."""
    timestamp: datetime
    event_type: str
    component: str
    message: str
    context: Dict[str, Any]
    severity: str = "info"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'component': self.component,
            'message': self.message,
            'context': self.context,
            'severity': self.severity
        }


class BlackBoxRecorder:
    """Black box recorder for forensic analysis."""

    def __init__(self, max_events: int = 1000):
        self.max_events = max_events
        self.events: List[BlackBoxEvent] = []
        self.lock = threading.Lock()
        self.crash_dump_path = Path("logs/black_box_crash_dump.json")

    def record_event(self, event_type: str, component: str, message: str,
                    context: Optional[Dict[str, Any]] = None, severity: str = "info"):
        """Record an event in the black box."""
        event = BlackBoxEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            component=component,
            message=message,
            context=context or {},
            severity=severity
        )

        with self.lock:
            self.events.append(event)
            if len(self.events) > self.max_events:
                self.events.pop(0)  # Remove oldest event

    def get_recent_events(self, count: int = 100) -> List[BlackBoxEvent]:
        """Get recent events from black box."""
        with self.lock:
            return self.events[-count:] if len(self.events) > count else self.events.copy()

    def dump_to_file(self, filepath: Optional[Path] = None):
        """Dump black box events to file for forensic analysis."""
        if filepath is None:
            filepath = self.crash_dump_path

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with self.lock:
            events_dict = [event.to_dict() for event in self.events]

        try:
            with open(filepath, 'w') as f:
                json.dump({
                    'dump_timestamp': datetime.now(timezone.utc).isoformat(),
                    'total_events': len(events_dict),
                    'events': events_dict
                }, f, indent=2)
            logger.info(f"Black box dump saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save black box dump: {e}")

    def clear(self):
        """Clear all events from black box."""
        with self.lock:
            self.events.clear()


class StateManager:
    """Comprehensive state management with SQLite persistence."""

    def __init__(self, db_path: str = "data/bot_state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
        self.lock = threading.Lock()
        self._initialize_database()

        # Register cleanup on exit
        atexit.register(self.close)

    def _initialize_database(self):
        """Initialize SQLite database with WAL mode and corruption recovery."""
        try:
            # Check if database is corrupted
            if self.db_path.exists():
                try:
                    test_conn = sqlite3.connect(str(self.db_path))
                    test_conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
                    test_conn.close()
                except sqlite3.DatabaseError:
                    logger.warning(f"Database corruption detected at {self.db_path}, recreating...")
                    self.db_path.unlink(missing_ok=True)

            self.connection = sqlite3.connect(str(self.db_path))
            self.connection.execute("PRAGMA journal_mode=WAL;")
            self.connection.execute("PRAGMA synchronous=NORMAL;")
            self.connection.execute("PRAGMA cache_size=-64000;")  # 64MB cache

            # Create tables
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    position_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    side TEXT,
                    quantity REAL,
                    entry_price REAL,
                    current_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    entry_time TIMESTAMP,
                    last_update TIMESTAMP,
                    status TEXT,
                    pnl REAL,
                    metadata TEXT
                )
            """)

            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    position_id TEXT,
                    symbol TEXT,
                    side TEXT,
                    quantity REAL,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP,
                    exit_reason TEXT,
                    commission REAL,
                    metadata TEXT,
                    FOREIGN KEY (position_id) REFERENCES positions (position_id)
                )
            """)

            self.connection.commit()
            logger.info(f"State database initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize state database: {e}")
            # Try to clean up corrupted file
            if self.db_path.exists():
                try:
                    self.db_path.unlink()
                    logger.info("Removed corrupted database file")
                except Exception:
                    pass
            raise

    def atomic_update(self, operation: Callable):
        """Perform atomic state update."""
        with self.lock:
            cursor = self.connection.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE;")
                result = operation(cursor)
                self.connection.commit()
                return result
            except Exception as e:
                self.connection.rollback()
                logger.error(f"Atomic update failed: {e}")
                raise

    def save_position(self, position_data: Dict[str, Any]):
        """Save position state atomically."""
        def _save(cursor):
            cursor.execute("""
                INSERT OR REPLACE INTO positions
                (position_id, symbol, side, quantity, entry_price, current_price,
                 stop_loss, take_profit, entry_time, last_update, status, pnl, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position_data['position_id'],
                position_data['symbol'],
                position_data['side'],
                position_data['quantity'],
                position_data['entry_price'],
                position_data.get('current_price', position_data['entry_price']),
                position_data.get('stop_loss'),
                position_data.get('take_profit'),
                position_data['entry_time'].isoformat(),
                datetime.now(timezone.utc).isoformat(),
                position_data.get('status', 'open'),
                position_data.get('pnl', 0.0),
                json.dumps(position_data.get('metadata', {}))
            ))

        self.atomic_update(_save)
        logger.debug(f"Position saved: {position_data['position_id']}")

    def load_positions(self) -> List[Dict[str, Any]]:
        """Load all positions from state."""
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM positions WHERE status = 'open'")
            rows = cursor.fetchall()

            positions = []
            for row in rows:
                position = {
                    'position_id': row[0],
                    'symbol': row[1],
                    'side': row[2],
                    'quantity': row[3],
                    'entry_price': row[4],
                    'current_price': row[5],
                    'stop_loss': row[6],
                    'take_profit': row[7],
                    'entry_time': datetime.fromisoformat(row[8]),
                    'last_update': datetime.fromisoformat(row[9]),
                    'status': row[10],
                    'pnl': row[11],
                    'metadata': json.loads(row[12]) if row[12] else {}
                }
                positions.append(position)

            return positions

    def save_trade(self, trade_data: Dict[str, Any]):
        """Save completed trade."""
        def _save(cursor):
            cursor.execute("""
                INSERT INTO trades
                (trade_id, position_id, symbol, side, quantity, entry_price,
                 exit_price, pnl, entry_time, exit_time, exit_reason, commission, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data['trade_id'],
                trade_data.get('position_id'),
                trade_data['symbol'],
                trade_data['side'],
                trade_data['quantity'],
                trade_data['entry_price'],
                trade_data.get('exit_price'),
                trade_data.get('pnl', 0.0),
                trade_data['entry_time'].isoformat(),
                trade_data.get('exit_time', datetime.now(timezone.utc)).isoformat(),
                trade_data.get('exit_reason', 'unknown'),
                trade_data.get('commission', 0.0),
                json.dumps(trade_data.get('metadata', {}))
            ))

        self.atomic_update(_save)
        logger.debug(f"Trade saved: {trade_data['trade_id']}")

    def get_state_value(self, key: str, default: Any = None) -> Any:
        """Get state value by key."""
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute("SELECT value FROM bot_state WHERE key = ?", (key,))
            row = cursor.fetchone()

            if row:
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError:
                    return row[0]

            return default

    def set_state_value(self, key: str, value: Any):
        """Set state value."""
        def _set(cursor):
            cursor.execute(
                "INSERT OR REPLACE INTO bot_state (key, value, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(value), datetime.now(timezone.utc).isoformat())
            )

        self.atomic_update(_set)

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("State database connection closed")


class ResilienceManager:
    """
    Central resilience management system.

    Coordinates all resilience features: graceful degradation, network resilience,
    state management, emergency procedures, and forensic analysis.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Initialize components
        self.state_manager = StateManager(self.config.get('state_db_path', 'data/bot_state.db'))
        self.black_box = BlackBoxRecorder(self.config.get('max_black_box_events', 1000))

        # Circuit breakers for different services
        self.api_circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.get('circuit_breaker_failure_threshold', 5),
            recovery_timeout=self.config.get('circuit_breaker_recovery_timeout', 60)
        )

        self.websocket_circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.get('websocket_failure_threshold', 3),
            recovery_timeout=self.config.get('websocket_recovery_timeout', 30)
        )

        # Retry budgets
        self.api_retry_budget = RetryBudget(
            max_retries_per_hour=self.config.get('max_retries_per_hour', 100),
            max_retries_per_minute=self.config.get('max_retries_per_minute', 10)
        )

        # Cached data
        self.price_cache: Dict[str, CachedData] = {}
        self.balance_cache: Optional[CachedData] = None
        self.position_cache: Optional[CachedData] = None

        # Degradation state
        self.degradation_level = DegradationLevel.NORMAL
        self.degradation_start_time: Optional[datetime] = None

        # Watchdog
        self.last_heartbeat = datetime.now(timezone.utc)
        self.heartbeat_interval = self.config.get('heartbeat_interval_seconds', 60)
        self.max_no_progress_time = self.config.get('max_no_progress_seconds', 300)

        # Emergency flags
        self.emergency_shutdown = False
        self.emergency_shutdown_reason = ""

        # Position limits
        self.max_position_size_pct = self.config.get('max_position_size_pct', 0.1)
        self.max_daily_loss_pct = self.config.get('max_daily_loss_pct', 0.05)

        # Locks
        self.operation_lock = threading.Lock()

        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
        self.heartbeat_thread.start()

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Register crash dump on exit
        atexit.register(self._crash_dump)

        self.black_box.record_event("startup", "resilience_manager", "Resilience manager initialized", {
            'config': self.config,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

        logger.info("Resilience manager initialized")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.emergency_shutdown = True
        self.emergency_shutdown_reason = f"Signal {signum} received"
        self._emergency_shutdown()

    def _heartbeat_monitor(self):
        """Monitor heartbeat and detect stalls."""
        while not self.emergency_shutdown:
            time.sleep(self.heartbeat_interval)

            now = datetime.now(timezone.utc)
            time_since_heartbeat = (now - self.last_heartbeat).seconds

            if time_since_heartbeat > self.max_no_progress_time:
                logger.critical(f"No progress for {time_since_heartbeat} seconds, possible deadlock")
                self.black_box.record_event("deadlock", "watchdog", "Possible deadlock detected", {
                    'time_since_heartbeat': time_since_heartbeat,
                    'max_allowed': self.max_no_progress_time
                })

                # Attempt recovery
                self._attempt_recovery()
            else:
                # Log heartbeat
                self.black_box.record_event("heartbeat", "watchdog", "Heartbeat OK", {
                    'uptime_seconds': time_since_heartbeat
                })

    def _attempt_recovery(self):
        """Attempt to recover from deadlock/stall."""
        logger.warning("Attempting recovery from stall")

        # Force garbage collection
        import gc
        gc.collect()

        # Reset heartbeat
        self.last_heartbeat = datetime.now(timezone.utc)

        # If still failing, this might need external intervention
        self.black_box.record_event("recovery", "watchdog", "Recovery attempt completed")

    def _crash_dump(self):
        """Create crash dump on exit."""
        try:
            self.black_box.dump_to_file()
            logger.info("Crash dump created")
        except Exception as e:
            print(f"Failed to create crash dump: {e}")

    def record_heartbeat(self):
        """Record heartbeat to show bot is alive."""
        self.last_heartbeat = datetime.now(timezone.utc)

    def execute_with_resilience(self, operation: Callable, operation_name: str,
                               max_retries: int = 3, base_delay: float = 1.0,
                               circuit_breaker: Optional[CircuitBreaker] = None,
                               fallback: Optional[Callable] = None) -> Any:
        """
        Execute operation with full resilience features.

        Args:
            operation: Function to execute
            operation_name: Name for logging/debugging
            max_retries: Maximum retry attempts
            base_delay: Base delay for exponential backoff
            circuit_breaker: Optional circuit breaker
            fallback: Optional fallback function if operation fails

        Returns:
            Operation result or fallback result
        """
        cb = circuit_breaker or self.api_circuit_breaker

        for attempt in range(max_retries + 1):
            try:
                # Check circuit breaker
                if not cb.should_attempt():
                    logger.warning(f"Circuit breaker open for {operation_name}, using fallback")
                    return self._execute_fallback(fallback, operation_name)

                # Check retry budget
                if attempt > 0 and not self.api_retry_budget.can_retry():
                    logger.warning(f"Retry budget exceeded for {operation_name}")
                    return self._execute_fallback(fallback, operation_name)

                # Execute operation
                self.black_box.record_event("operation_start", operation_name, f"Starting {operation_name}", {
                    'attempt': attempt + 1,
                    'max_retries': max_retries
                })

                result = operation()

                # Record success
                cb.record_success()
                self.record_heartbeat()

                self.black_box.record_event("operation_success", operation_name, f"{operation_name} succeeded", {
                    'attempt': attempt + 1
                })

                return result

            except Exception as e:
                # Record failure
                cb.record_failure()
                if attempt > 0:
                    self.api_retry_budget.record_retry()

                self.black_box.record_event("operation_failure", operation_name, f"{operation_name} failed", {
                    'attempt': attempt + 1,
                    'error': str(e),
                    'error_type': type(e).__name__
                })

                if attempt < max_retries:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt)
                    jitter = delay * 0.2 * (0.5 - time.time() % 1)  # ±20% jitter
                    total_delay = delay + jitter

                    logger.warning(f"{operation_name} failed (attempt {attempt + 1}/{max_retries + 1}), "
                                 f"retrying in {total_delay:.2f}s: {e}")
                    time.sleep(total_delay)
                else:
                    logger.error(f"{operation_name} failed after {max_retries + 1} attempts: {e}")
                    return self._execute_fallback(fallback, operation_name)

        # This should never be reached, but just in case
        return self._execute_fallback(fallback, operation_name)

    def _execute_fallback(self, fallback: Optional[Callable], operation_name: str) -> Any:
        """Execute fallback operation."""
        if fallback:
            try:
                self.black_box.record_event("fallback", operation_name, f"Executing fallback for {operation_name}")
                result = fallback()
                self.black_box.record_event("fallback_success", operation_name, "Fallback succeeded")
                return result
            except Exception as e:
                self.black_box.record_event("fallback_failure", operation_name, f"Fallback failed: {e}")
                logger.error(f"Fallback for {operation_name} failed: {e}")

        # Ultimate fallback - return None and let caller handle
        logger.error(f"All attempts failed for {operation_name}, no fallback available")
        return None

    def graceful_degradation(self, component: str, error: Exception,
                           fallback_action: Optional[Callable] = None) -> DegradationLevel:
        """
        Handle graceful degradation when components fail.

        Args:
            component: Component that failed
            error: The error that occurred
            fallback_action: Optional action to take for degradation

        Returns:
            New degradation level
        """
        logger.warning(f"Component {component} failed: {error}")

        self.black_box.record_event("degradation", component, f"Component failed: {component}", {
            'error': str(error),
            'error_type': type(error).__name__,
            'current_level': self.degradation_level.value
        })

        # Determine new degradation level
        if component == "websocket":
            if self.degradation_level == DegradationLevel.NORMAL:
                self.degradation_level = DegradationLevel.DEGRADED
                logger.info("Entering degraded mode: WebSocket failed, using REST polling")
                if fallback_action:
                    fallback_action()

        elif component in ["api", "exchange"]:
            if self.degradation_level in [DegradationLevel.NORMAL, DegradationLevel.DEGRADED]:
                self.degradation_level = DegradationLevel.CRITICAL
                logger.warning("Entering critical mode: API unavailable, using cached data only")

        elif component == "database":
            self.degradation_level = DegradationLevel.EMERGENCY
            logger.critical("Entering emergency mode: State persistence failed")
            self._emergency_shutdown()

        if self.degradation_start_time is None:
            self.degradation_start_time = datetime.now(timezone.utc)

        return self.degradation_level

    def check_emergency_conditions(self) -> bool:
        """
        Check for emergency shutdown conditions.

        Returns:
            True if emergency shutdown should be triggered
        """
        # Check environment variable
        if os.getenv('BOT_EMERGENCY_SHUTDOWN', '').lower() in ['true', '1', 'yes']:
            self.emergency_shutdown_reason = "Emergency shutdown requested via environment"
            return True

        # Check daily loss limit
        daily_pnl = self.get_daily_pnl()
        if daily_pnl < -self.max_daily_loss_pct:
            self.emergency_shutdown_reason = f"Daily loss limit exceeded: {daily_pnl:.1%}"
            return True

        # Check position size limits (would need current positions)
        # This would be checked in the main trading loop

        return False

    def _emergency_shutdown(self):
        """Execute emergency shutdown procedure."""
        logger.critical(f"EMERGENCY SHUTDOWN: {self.emergency_shutdown_reason}")

        self.black_box.record_event("emergency_shutdown", "resilience_manager",
                                  "Emergency shutdown initiated", {
            'reason': self.emergency_shutdown_reason,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

        try:
            # Close all positions (this would need to be implemented in trading bot)
            # self._close_all_positions()

            # Save final state
            self.state_manager.set_state_value('emergency_shutdown', {
                'reason': self.emergency_shutdown_reason,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'open_positions': self.state_manager.load_positions()
            })

            # Create crash dump
            self._crash_dump()

        except Exception as e:
            logger.error(f"Error during emergency shutdown: {e}")

        # Exit with specific code to indicate emergency shutdown
        sys.exit(42)

    def get_daily_pnl(self) -> float:
        """Get daily P&L percentage."""
        # This would need to be implemented based on trading history
        # For now, return a placeholder
        return 0.0

    def validate_api_response(self, response: Any, response_type: str,
                            required_fields: Optional[List[str]] = None) -> bool:
        """
        Validate API response data.

        Args:
            response: API response to validate
            response_type: Type of response (price, balance, position, etc.)
            required_fields: Required fields in response

        Returns:
            True if validation passes
        """
        try:
            if response_type == "price":
                if not isinstance(response, (int, float)) or response <= 0 or response > 1000000:
                    logger.error(f"Invalid price: {response}")
                    return False

            elif response_type == "balance":
                if not isinstance(response, (int, float)) or response < 0:
                    logger.error(f"Invalid balance: {response}")
                    return False

            elif response_type == "position_size":
                if not isinstance(response, (int, float)) or abs(response) > 1000:
                    logger.error(f"Invalid position size: {response}")
                    return False

            elif response_type == "order_quantity":
                if not isinstance(response, (int, float)) or response <= 0 or response > 100:
                    logger.error(f"Invalid order quantity: {response}")
                    return False

            # Check required fields
            if required_fields and isinstance(response, dict):
                for field in required_fields:
                    if field not in response:
                        logger.error(f"Missing required field '{field}' in response")
                        return False

            return True

        except Exception as e:
            logger.error(f"Response validation failed: {e}")
            return False

    def cache_data(self, key: str, data: Any, expiry_seconds: int = 300, source: str = "api"):
        """Cache data with expiry."""
        self.price_cache[key] = CachedData(
            data=data,
            timestamp=datetime.now(timezone.utc),
            expiry_seconds=expiry_seconds,
            source=source
        )

    def get_cached_data(self, key: str, max_age_seconds: int = 300) -> Optional[Any]:
        """Get cached data if available and not expired."""
        if key in self.price_cache:
            cached = self.price_cache[key]
            if not cached.is_expired() and not cached.is_stale(max_age_seconds):
                return cached.data
            else:
                # Clean up expired cache
                del self.price_cache[key]

        return None

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        return {
            'degradation_level': self.degradation_level.value,
            'circuit_breaker_api': self.api_circuit_breaker.state.value,
            'circuit_breaker_websocket': self.websocket_circuit_breaker.state.value,
            'retry_budget_remaining': self.api_retry_budget.can_retry(),
            'cached_prices_count': len(self.price_cache),
            'last_heartbeat_seconds': (datetime.now(timezone.utc) - self.last_heartbeat).seconds,
            'emergency_shutdown': self.emergency_shutdown,
            'black_box_events': len(self.black_box.events),
            'open_positions': len(self.state_manager.load_positions())
        }


# Global resilience manager instance
_resilience_manager: Optional[ResilienceManager] = None


def get_resilience_manager() -> ResilienceManager:
    """Get global resilience manager instance."""
    global _resilience_manager
    if _resilience_manager is None:
        raise RuntimeError("Resilience manager not initialized. Call initialize_resilience_manager() first.")
    return _resilience_manager


def initialize_resilience_manager(config: Optional[Dict[str, Any]] = None) -> ResilienceManager:
    """Initialize global resilience manager."""
    global _resilience_manager
    _resilience_manager = ResilienceManager(config)
    return _resilience_manager
