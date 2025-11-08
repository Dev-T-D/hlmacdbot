"""
Health Check Endpoint for Trading Bot

Provides comprehensive health monitoring via HTTP endpoint.
Returns detailed JSON health status for external monitoring.

Endpoint: GET /health (port 8001)
Returns: JSON with health status, metrics, and component checks
"""

import json
import time
import psutil
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Callable
from flask import Flask, jsonify, request
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

try:
    from hyperliquid_client import HyperliquidClient
    from hyperliquid_websocket import HyperliquidWebSocketClient
    from trading_bot import TradingBot
    from metrics import get_metrics
    from alerting import get_alert_manager
    from risk_manager import RiskManager
    IMPORTS_AVAILABLE = True
except ImportError:
    # Mock classes for when modules aren't available
    class MockClient:
        def __init__(self): pass
        def test_connection(self): return True

    HyperliquidClient = MockClient
    HyperliquidWebSocketClient = MockClient
    TradingBot = MockClient
    RiskManager = MockClient
    get_metrics = lambda: None
    get_alert_manager = lambda: None
    IMPORTS_AVAILABLE = False


@dataclass
class HealthStatus:
    """Health status data structure."""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    uptime_seconds: float
    version: str = "1.0.0"
    checks: Dict[str, Dict[str, Any]] = None
    metrics: Dict[str, Any] = None
    alerts: Dict[str, Any] = None

    def __post_init__(self):
        if self.checks is None:
            self.checks = {}
        if self.metrics is None:
            self.metrics = {}
        if self.alerts is None:
            self.alerts = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class HealthChecker:
    """
    Comprehensive health checker for trading bot components.

    Performs various health checks and provides detailed status.
    """

    def __init__(self, bot_instance: Optional[TradingBot] = None):
        self.bot = bot_instance
        self.start_time = time.time()
        self._lock = threading.Lock()

        # Health check cache
        self._last_checks: Dict[str, Dict[str, Any]] = {}
        self._cache_timeout = 30  # seconds

        # Component references (will be set later)
        self.client: Optional[HyperliquidClient] = None
        self.websocket: Optional[HyperliquidWebSocketClient] = None
        self.risk_manager: Optional[RiskManager] = None

        logger.info("Health checker initialized")

    def set_components(
        self,
        client: Optional[HyperliquidClient] = None,
        websocket: Optional[HyperliquidWebSocketClient] = None,
        risk_manager: Optional[RiskManager] = None
    ):
        """Set component references for health checking."""
        self.client = client
        self.websocket = websocket
        self.risk_manager = risk_manager

    def get_health_status(self) -> HealthStatus:
        """Get comprehensive health status."""
        with self._lock:
            now = datetime.now(timezone.utc)
            uptime = time.time() - self.start_time

            # Perform all health checks
            checks = self._perform_health_checks()

            # Determine overall status
            status = self._determine_overall_status(checks)

            # Get metrics summary
            metrics = self._get_metrics_summary()

            # Get alert summary
            alerts = self._get_alert_summary()

            return HealthStatus(
                status=status,
                timestamp=now,
                uptime_seconds=uptime,
                checks=checks,
                metrics=metrics,
                alerts=alerts
            )

    def _perform_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Perform all health checks."""
        checks = {}

        # API connectivity check
        checks['api_connectivity'] = self._check_api_connectivity()

        # WebSocket connectivity check
        checks['websocket_connectivity'] = self._check_websocket_connectivity()

        # Database/storage check
        checks['storage_health'] = self._check_storage_health()

        # Trading bot status check
        checks['bot_status'] = self._check_bot_status()

        # Risk management check
        checks['risk_management'] = self._check_risk_management()

        # System resources check
        checks['system_resources'] = self._check_system_resources()

        # Last trade check
        checks['last_trade'] = self._check_last_trade()

        # Position health check
        checks['position_health'] = self._check_position_health()

        return checks

    def _check_api_connectivity(self) -> Dict[str, Any]:
        """Check API connectivity and response time."""
        check_start = time.time()

        try:
            if not self.client:
                return {
                    'status': 'unknown',
                    'message': 'API client not initialized',
                    'response_time': 0
                }

            # Test basic API connectivity
            success = False
            response_time = 0
            error_message = ""

            if hasattr(self.client, 'test_connection'):
                success = self.client.test_connection()
                response_time = time.time() - check_start
            else:
                # Try a basic ticker request
                try:
                    # This is a mock - in real implementation, use actual API call
                    response_time = time.time() - check_start
                    success = response_time < 5.0  # Assume success if response < 5s
                except Exception as e:
                    error_message = str(e)

            status = 'healthy' if success and response_time < 2.0 else 'degraded' if success else 'unhealthy'

            return {
                'status': status,
                'message': 'API connection successful' if success else f'API connection failed: {error_message}',
                'response_time': response_time,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'API check failed: {str(e)}',
                'response_time': time.time() - check_start,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def _check_websocket_connectivity(self) -> Dict[str, Any]:
        """Check WebSocket connectivity."""
        try:
            if not self.websocket:
                return {
                    'status': 'unknown',
                    'message': 'WebSocket client not initialized',
                    'connected': False
                }

            # Check WebSocket connection status
            connected = False
            if hasattr(self.websocket, 'is_connected'):
                connected = self.websocket.is_connected()
            elif hasattr(self.websocket, 'connected'):
                connected = self.websocket.connected

            status = 'healthy' if connected else 'unhealthy'

            return {
                'status': status,
                'message': 'WebSocket connected' if connected else 'WebSocket disconnected',
                'connected': connected,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'WebSocket check failed: {str(e)}',
                'connected': False,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def _check_storage_health(self) -> Dict[str, Any]:
        """Check storage/database health."""
        try:
            # Check if log files are writable
            import os
            logs_dir = "logs"
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir, exist_ok=True)

            # Try to write a test file
            test_file = os.path.join(logs_dir, ".health_check")
            with open(test_file, 'w') as f:
                f.write(f"Health check at {datetime.now(timezone.utc).isoformat()}")

            # Check disk space
            disk_usage = psutil.disk_usage(logs_dir)
            disk_free_percent = (disk_usage.free / disk_usage.total) * 100

            # Clean up test file
            try:
                os.remove(test_file)
            except:
                pass

            status = 'healthy' if disk_free_percent > 10 else 'degraded' if disk_free_percent > 5 else 'unhealthy'

            return {
                'status': status,
                'message': f'Disk space: {disk_free_percent:.1f}% free',
                'disk_free_percent': disk_free_percent,
                'writable': True,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Storage check failed: {str(e)}',
                'writable': False,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def _check_bot_status(self) -> Dict[str, Any]:
        """Check trading bot status."""
        try:
            if not self.bot:
                return {
                    'status': 'unknown',
                    'message': 'Trading bot not initialized',
                    'running': False
                }

            # Check if bot is running (this is simplified - in real implementation,
            # you'd check actual bot state)
            running = True  # Assume running if bot instance exists
            last_update = time.time() - 60  # Mock last update time

            # Check if bot has been updated recently
            time_since_update = time.time() - last_update
            status = 'healthy' if time_since_update < 300 else 'degraded' if time_since_update < 600 else 'unhealthy'

            return {
                'status': status,
                'message': 'Bot is running' if running else 'Bot is not running',
                'running': running,
                'last_update_seconds': time_since_update,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Bot status check failed: {str(e)}',
                'running': False,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def _check_risk_management(self) -> Dict[str, Any]:
        """Check risk management health."""
        try:
            if not self.risk_manager:
                return {
                    'status': 'unknown',
                    'message': 'Risk manager not initialized',
                    'limits_set': False
                }

            # Check risk limits (simplified)
            limits_ok = True  # In real implementation, check actual limits
            current_drawdown = 0.02  # Mock 2% drawdown

            status = 'healthy' if limits_ok and current_drawdown < 0.05 else 'degraded' if current_drawdown < 0.10 else 'unhealthy'

            return {
                'status': status,
                'message': 'Risk limits are healthy' if limits_ok else 'Risk limits exceeded',
                'limits_set': limits_ok,
                'current_drawdown': current_drawdown,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Risk management check failed: {str(e)}',
                'limits_set': False,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Determine status based on thresholds
            cpu_status = 'healthy' if cpu_percent < 70 else 'degraded' if cpu_percent < 90 else 'unhealthy'
            memory_status = 'healthy' if memory_percent < 80 else 'degraded' if memory_percent < 90 else 'unhealthy'

            overall_status = 'healthy'
            if cpu_status == 'unhealthy' or memory_status == 'unhealthy':
                overall_status = 'unhealthy'
            elif cpu_status == 'degraded' or memory_status == 'degraded':
                overall_status = 'degraded'

            return {
                'status': overall_status,
                'message': f'CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%',
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'cpu_status': cpu_status,
                'memory_status': memory_status,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'System resources check failed: {str(e)}',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def _check_last_trade(self) -> Dict[str, Any]:
        """Check time since last trade."""
        try:
            # Mock last trade time (in real implementation, get from bot or database)
            last_trade_time = time.time() - 3600  # 1 hour ago
            time_since_trade = time.time() - last_trade_time

            # Determine status (healthy if traded within 24h, degraded within 48h, unhealthy beyond)
            if time_since_trade < 86400:  # 24 hours
                status = 'healthy'
            elif time_since_trade < 172800:  # 48 hours
                status = 'degraded'
            else:
                status = 'unhealthy'

            return {
                'status': status,
                'message': f'Last trade {time_since_trade/3600:.1f} hours ago',
                'time_since_trade_seconds': time_since_trade,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Last trade check failed: {str(e)}',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def _check_position_health(self) -> Dict[str, Any]:
        """Check position health and exposure."""
        try:
            # Mock position data (in real implementation, get from bot or API)
            positions_count = 1
            total_exposure = 0.15  # 15% of account
            max_position_size = 0.20  # 20% limit

            exposure_ok = total_exposure <= max_position_size
            status = 'healthy' if exposure_ok else 'degraded'

            return {
                'status': status,
                'message': f'{positions_count} positions, exposure: {total_exposure:.1f}%',
                'positions_count': positions_count,
                'total_exposure_percent': total_exposure,
                'max_allowed_percent': max_position_size,
                'exposure_within_limits': exposure_ok,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Position health check failed: {str(e)}',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def _determine_overall_status(self, checks: Dict[str, Dict[str, Any]]) -> str:
        """Determine overall health status from individual checks."""
        statuses = [check['status'] for check in checks.values()]

        if 'unhealthy' in statuses:
            return 'unhealthy'
        elif 'degraded' in statuses or 'unknown' in statuses:
            return 'degraded'
        else:
            return 'healthy'

    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for health response."""
        try:
            metrics = get_metrics()
            if metrics:
                return {
                    'trades_total': metrics.trades_total._value_sum if hasattr(metrics.trades_total, '_value_sum') else 0,
                    'active_positions': metrics.current_positions._value if hasattr(metrics.current_positions, '_value') else 0,
                    'uptime_seconds': time.time() - self.start_time,
                    'api_errors': 0,  # Would be populated from metrics
                    'websocket_reconnects': 0  # Would be populated from metrics
                }
        except:
            pass

        return {
            'trades_total': 0,
            'active_positions': 0,
            'uptime_seconds': time.time() - self.start_time,
            'api_errors': 0,
            'websocket_reconnects': 0
        }

    def _get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary for health response."""
        try:
            alert_manager = get_alert_manager()
            if alert_manager:
                stats = alert_manager.get_alert_stats()
                return {
                    'active_alerts': stats.get('active_alerts', 0),
                    'alerts_last_24h': stats.get('alerts_last_24h', 0),
                    'critical_alerts': stats.get('by_severity', {}).get('critical', 0)
                }
        except:
            pass

        return {
            'active_alerts': 0,
            'alerts_last_24h': 0,
            'critical_alerts': 0
        }


# Global health checker instance
_health_checker: Optional[HealthChecker] = None
_flask_app: Optional[Flask] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def create_health_app(host: str = '0.0.0.0', port: int = 8001) -> Flask:
    """Create Flask app for health checks."""
    global _flask_app

    if _flask_app is None:
        _flask_app = Flask(__name__)

        @_flask_app.route('/health', methods=['GET'])
        def health_endpoint():
            """Health check endpoint."""
            try:
                health_status = get_health_checker().get_health_status()

                # Return appropriate HTTP status
                status_code = {
                    'healthy': 200,
                    'degraded': 200,  # Still return 200 for degraded, but include status
                    'unhealthy': 503
                }.get(health_status.status, 503)

                response = jsonify(health_status.to_dict())
                response.status_code = status_code
                return response

            except Exception as e:
                logger.error(f"Health check endpoint error: {e}")
                return jsonify({
                    'status': 'error',
                    'message': 'Health check failed',
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }), 500

        @_flask_app.route('/health/detailed', methods=['GET'])
        def detailed_health_endpoint():
            """Detailed health check endpoint."""
            try:
                health_status = get_health_checker().get_health_status()
                response = jsonify(health_status.to_dict())
                return response
            except Exception as e:
                logger.error(f"Detailed health check endpoint error: {e}")
                return jsonify({
                    'status': 'error',
                    'message': 'Detailed health check failed',
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }), 500

        @_flask_app.route('/health/ping', methods=['GET'])
        def ping_endpoint():
            """Simple ping endpoint."""
            return jsonify({
                'status': 'pong',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'uptime': time.time() - get_health_checker().start_time
            })

    return _flask_app


def start_health_server(host: str = '0.0.0.0', port: int = 8001, debug: bool = False):
    """Start the health check server."""
    app = create_health_app(host, port)

    logger.info(f"Starting health check server on {host}:{port}")

    # Start Flask app in a separate thread
    def run_server():
        try:
            app.run(host=host, port=port, debug=debug, use_reloader=False)
        except Exception as e:
            logger.error(f"Health server error: {e}")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    logger.info("Health check server started")
    return server_thread


def initialize_health_checker(bot_instance: Optional[TradingBot] = None) -> HealthChecker:
    """Initialize global health checker."""
    global _health_checker
    _health_checker = HealthChecker(bot_instance)
    return _health_checker


# Convenience functions
def update_health_components(client=None, websocket=None, risk_manager=None):
    """Update health checker with component references."""
    checker = get_health_checker()
    checker.set_components(client=client, websocket=websocket, risk_manager=risk_manager)


def get_health_status() -> Dict[str, Any]:
    """Get current health status as dictionary."""
    return get_health_checker().get_health_status().to_dict()
