"""
High-Performance Database Manager for Trading Bot State

Uses SQLite with WAL mode for atomic writes and concurrent reads.
Optimized for high-frequency trading with connection pooling and batch operations.

Features:
- SQLite WAL mode for better concurrency
- Connection pooling for multiple concurrent operations
- Batch write operations for performance
- Automatic schema management and migrations
- Thread-safe operations
- Performance metrics and monitoring

Tables:
- positions: Current position state
- trades: Trade history with performance metrics
- daily_stats: Daily P&L and statistics
- config_backup: Configuration backup
- performance_metrics: Bot performance data

"""

import sqlite3
import threading
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
from datetime import datetime, timezone
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """High-performance SQLite database manager with WAL mode and connection pooling."""

    def __init__(self, db_path: str = "data/trading_bot.db", max_connections: int = 5):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
            max_connections: Maximum number of concurrent connections
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.max_connections = max_connections
        self._connection_pool = []
        self._pool_lock = threading.Lock()
        self._local = threading.local()  # Thread-local storage for connections

        # Performance metrics
        self._operation_times = {}
        self._query_count = 0
        self._write_count = 0

        # Initialize database
        self._initialize_database()

        logger.info(f"Database manager initialized with {max_connections} max connections")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool (thread-local)."""
        if not hasattr(self._local, 'connection'):
            with self._pool_lock:
                if len(self._connection_pool) < self.max_connections:
                    # Create new connection
                    conn = sqlite3.connect(
                        str(self.db_path),
                        check_same_thread=False,  # Allow multi-threaded access
                        timeout=30.0  # Connection timeout
                    )
                    conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for concurrency
                    conn.execute("PRAGMA synchronous=NORMAL")  # Balance performance/safety
                    conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
                    conn.execute("PRAGMA temp_store=MEMORY")  # Temp tables in memory
                    conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory map
                    conn.row_factory = sqlite3.Row  # Dict-like row access

                    self._connection_pool.append(conn)
                    logger.debug("Created new database connection")
                else:
                    # Reuse existing connection
                    conn = self._connection_pool[0]

            self._local.connection = conn

        return self._local.connection

    def _initialize_database(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            # Create tables
            conn.executescript("""
                -- Positions table
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    position_type TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    size REAL NOT NULL,
                    order_id TEXT,
                    timestamp TEXT NOT NULL,
                    status TEXT DEFAULT 'open',
                    pnl REAL DEFAULT 0,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    updated_at REAL DEFAULT (strftime('%s', 'now'))
                );

                -- Trades table
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    order_type TEXT DEFAULT 'LIMIT',
                    strategy TEXT DEFAULT 'MACD',
                    pnl REAL DEFAULT 0,
                    reason TEXT,
                    indicators TEXT,  -- JSON string
                    timestamp TEXT NOT NULL,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                );

                -- Daily statistics table
                CREATE TABLE IF NOT EXISTS daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    starting_balance REAL NOT NULL,
                    ending_balance REAL NOT NULL,
                    pnl REAL NOT NULL,
                    trade_count INTEGER DEFAULT 0,
                    win_count INTEGER DEFAULT 0,
                    loss_count INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                );

                -- Configuration backup table
                CREATE TABLE IF NOT EXISTS config_backup (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                );

                -- Performance metrics table
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp REAL DEFAULT (strftime('%s', 'now'))
                );

                -- Create indexes for performance
                CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
                CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
                CREATE INDEX IF NOT EXISTS idx_daily_stats_date ON daily_stats(date);
                CREATE INDEX IF NOT EXISTS idx_performance_metrics_name ON performance_metrics(metric_name);
                CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(timestamp);
            """)

            conn.commit()
            logger.info("Database schema initialized")

    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e

    def save_position(self, position_data: Dict[str, Any]) -> int:
        """Save position to database."""
        start_time = time.time()

        with self.transaction() as conn:
            cursor = conn.execute("""
                INSERT INTO positions
                (symbol, position_type, entry_price, stop_loss, take_profit, size, order_id, timestamp, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position_data.get('symbol'),
                position_data.get('type'),
                position_data.get('entry_price'),
                position_data.get('stop_loss'),
                position_data.get('take_profit'),
                position_data.get('size'),
                position_data.get('order_id'),
                position_data.get('timestamp'),
                'open'
            ))

            position_id = cursor.lastrowid

        self._write_count += 1
        self._record_operation_time("save_position", time.time() - start_time)
        logger.debug(f"Saved position {position_id} to database")
        return position_id

    def update_position(self, position_id: int, updates: Dict[str, Any]) -> bool:
        """Update position in database."""
        start_time = time.time()

        set_clause = ", ".join(f"{key} = ?" for key in updates.keys())
        values = list(updates.values()) + [position_id]

        with self.transaction() as conn:
            cursor = conn.execute(f"""
                UPDATE positions
                SET {set_clause}, updated_at = strftime('%s', 'now')
                WHERE id = ?
            """, values)

            success = cursor.rowcount > 0

        if success:
            self._write_count += 1
            self._record_operation_time("update_position", time.time() - start_time)
            logger.debug(f"Updated position {position_id}")

        return success

    def close_position(self, position_id: int, pnl: float) -> bool:
        """Mark position as closed."""
        return self.update_position(position_id, {
            'status': 'closed',
            'pnl': pnl
        })

    def get_current_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current open position for symbol."""
        start_time = time.time()

        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM positions
            WHERE symbol = ? AND status = 'open'
            ORDER BY created_at DESC
            LIMIT 1
        """, (symbol,))

        row = cursor.fetchone()

        self._query_count += 1
        self._record_operation_time("get_current_position", time.time() - start_time)

        if row:
            return dict(row)
        return None

    def save_trade(self, trade_data: Dict[str, Any]) -> int:
        """Save trade to database."""
        start_time = time.time()

        with self.transaction() as conn:
            cursor = conn.execute("""
                INSERT INTO trades
                (symbol, side, quantity, price, order_type, strategy, pnl, reason, indicators, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data.get('symbol'),
                trade_data.get('side'),
                trade_data.get('quantity'),
                trade_data.get('price'),
                trade_data.get('order_type', 'LIMIT'),
                trade_data.get('strategy', 'MACD'),
                trade_data.get('pnl', 0),
                trade_data.get('reason'),
                json.dumps(trade_data.get('indicators', {})),
                trade_data.get('timestamp')
            ))

            trade_id = cursor.lastrowid

        self._write_count += 1
        self._record_operation_time("save_trade", time.time() - start_time)
        logger.debug(f"Saved trade {trade_id} to database")
        return trade_id

    def save_daily_stats(self, date: str, stats: Dict[str, Any]) -> int:
        """Save daily statistics."""
        start_time = time.time()

        with self.transaction() as conn:
            cursor = conn.execute("""
                INSERT OR REPLACE INTO daily_stats
                (date, starting_balance, ending_balance, pnl, trade_count, win_count, loss_count, win_rate, max_drawdown)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date,
                stats.get('starting_balance', 0),
                stats.get('ending_balance', 0),
                stats.get('pnl', 0),
                stats.get('trade_count', 0),
                stats.get('win_count', 0),
                stats.get('loss_count', 0),
                stats.get('win_rate', 0),
                stats.get('max_drawdown', 0)
            ))

            stats_id = cursor.lastrowid

        self._write_count += 1
        self._record_operation_time("save_daily_stats", time.time() - start_time)
        logger.debug(f"Saved daily stats for {date}")
        return stats_id

    def get_daily_stats(self, date: str) -> Optional[Dict[str, Any]]:
        """Get daily statistics for date."""
        start_time = time.time()

        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM daily_stats WHERE date = ?
        """, (date,))

        row = cursor.fetchone()

        self._query_count += 1
        self._record_operation_time("get_daily_stats", time.time() - start_time)

        if row:
            return dict(row)
        return None

    def save_config_backup(self, config: Dict[str, Any]) -> int:
        """Save configuration backup."""
        start_time = time.time()

        with self.transaction() as conn:
            cursor = conn.execute("""
                INSERT INTO config_backup (config_json, timestamp)
                VALUES (?, ?)
            """, (
                json.dumps(config),
                datetime.now(timezone.utc).isoformat()
            ))

            backup_id = cursor.lastrowid

        self._write_count += 1
        self._record_operation_time("save_config_backup", time.time() - start_time)
        logger.debug(f"Saved config backup {backup_id}")
        return backup_id

    def record_performance_metric(self, name: str, value: float) -> None:
        """Record performance metric."""
        start_time = time.time()

        with self.transaction() as conn:
            conn.execute("""
                INSERT INTO performance_metrics (metric_name, metric_value)
                VALUES (?, ?)
            """, (name, value))

        self._write_count += 1
        self._record_operation_time("record_performance_metric", time.time() - start_time)

    def get_recent_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades."""
        start_time = time.time()

        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM trades
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        trades = [dict(row) for row in cursor.fetchall()]

        self._query_count += 1
        self._record_operation_time("get_recent_trades", time.time() - start_time)

        return trades

    def get_trade_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get trade statistics for recent period."""
        start_time = time.time()

        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                AVG(pnl) as avg_pnl,
                SUM(pnl) as total_pnl,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade
            FROM trades
            WHERE created_at >= strftime('%s', 'now', '-{} days')
        """.format(days))

        row = cursor.fetchone()

        self._query_count += 1
        self._record_operation_time("get_trade_statistics", time.time() - start_time)

        if row:
            stats = dict(row)
            stats['win_rate'] = (stats['winning_trades'] / stats['total_trades']) * 100 if stats['total_trades'] > 0 else 0
            return stats

        return {}

    def batch_operations(self, operations: List[Tuple[str, Dict[str, Any]]]) -> List[Any]:
        """Execute multiple operations in a single transaction."""
        start_time = time.time()
        results = []

        with self.transaction() as conn:
            for operation, data in operations:
                if operation == "save_trade":
                    cursor = conn.execute("""
                        INSERT INTO trades
                        (symbol, side, quantity, price, order_type, strategy, pnl, reason, indicators, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        data.get('symbol'),
                        data.get('side'),
                        data.get('quantity'),
                        data.get('price'),
                        data.get('order_type', 'LIMIT'),
                        data.get('strategy', 'MACD'),
                        data.get('pnl', 0),
                        data.get('reason'),
                        json.dumps(data.get('indicators', {})),
                        data.get('timestamp')
                    ))
                    results.append(cursor.lastrowid)

                elif operation == "save_position":
                    cursor = conn.execute("""
                        INSERT INTO positions
                        (symbol, position_type, entry_price, stop_loss, take_profit, size, order_id, timestamp, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        data.get('symbol'),
                        data.get('type'),
                        data.get('entry_price'),
                        data.get('stop_loss'),
                        data.get('take_profit'),
                        data.get('size'),
                        data.get('order_id'),
                        data.get('timestamp'),
                        'open'
                    ))
                    results.append(cursor.lastrowid)

                elif operation == "update_position":
                    position_id = data.pop('id')
                    set_clause = ", ".join(f"{key} = ?" for key in data.keys())
                    values = list(data.values()) + [position_id]

                    conn.execute(f"""
                        UPDATE positions
                        SET {set_clause}, updated_at = strftime('%s', 'now')
                        WHERE id = ?
                    """, values)
                    results.append(True)

        self._write_count += len(operations)
        self._record_operation_time("batch_operations", time.time() - start_time)
        logger.debug(f"Executed {len(operations)} batch operations")
        return results

    def vacuum_database(self) -> None:
        """Optimize database by rebuilding and reclaiming space."""
        start_time = time.time()

        conn = self._get_connection()
        conn.execute("VACUUM")
        conn.commit()

        self._record_operation_time("vacuum_database", time.time() - start_time)
        logger.info("Database vacuum completed")

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and performance metrics."""
        conn = self._get_connection()

        # Get table sizes
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)

        tables = cursor.fetchall()
        table_stats = {}

        for table in tables:
            table_name = table[0]
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            table_stats[table_name] = count

        # Database file size
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "database_size_bytes": db_size,
            "database_size_mb": db_size / (1024 * 1024),
            "table_counts": table_stats,
            "queries_executed": self._query_count,
            "writes_executed": self._write_count,
            "connection_pool_size": len(self._connection_pool),
            "max_connections": self.max_connections,
            "operation_times": self._operation_times.copy(),
        }

    def _record_operation_time(self, operation: str, duration: float) -> None:
        """Record operation timing for metrics."""
        if operation not in self._operation_times:
            self._operation_times[operation] = []

        # Keep only last 100 measurements
        if len(self._operation_times[operation]) >= 100:
            self._operation_times[operation] = self._operation_times[operation][-99:]

        self._operation_times[operation].append(duration)

    def close(self) -> None:
        """Close all database connections."""
        with self._pool_lock:
            for conn in self._connection_pool:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(f"Error closing database connection: {e}")

            self._connection_pool.clear()
            logger.info("Database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance."""
    return _db_manager


def initialize_database_manager(db_path: str = "data/trading_bot.db", max_connections: int = 5) -> DatabaseManager:
    """Initialize global database manager."""
    global _db_manager
    _db_manager = DatabaseManager(db_path, max_connections)
    return _db_manager


def shutdown_database_manager() -> None:
    """Shutdown global database manager."""
    global _db_manager
    if _db_manager:
        _db_manager.close()
        _db_manager = None
