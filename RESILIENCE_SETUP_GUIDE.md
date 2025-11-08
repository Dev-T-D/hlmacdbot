# Trading Bot Resilience Setup Guide

## Overview

This guide provides comprehensive instructions for implementing and maintaining bulletproof reliability in your trading bot. The resilience system ensures the bot can survive failures, recover automatically, and maintain position integrity under all circumstances.

## Architecture Components

### Core Resilience System (`resilience.py`)
- **Circuit Breakers**: Prevent cascade failures during API outages
- **Retry Logic**: Exponential backoff with jitter and budget limits
- **State Management**: SQLite WAL mode for atomic state persistence
- **Graceful Degradation**: Automatic fallback to degraded modes
- **Emergency Procedures**: Controlled shutdown with position protection

### Watchdog System (`watchdog.py`)
- **Process Monitoring**: Independent process health checking
- **Automatic Restart**: Recovery from crashes and hangs
- **Emergency Actions**: Position closure when recovery fails
- **Health Reporting**: Comprehensive system status

### Forensic Analysis
- **Black Box Recorder**: Event logging for incident analysis
- **Crash Dumps**: Automatic state preservation on failures
- **Performance Metrics**: System health tracking

## Prerequisites

### System Requirements
- **Python**: 3.11+ (asyncio support required)
- **Database**: SQLite 3.35+ (WAL mode support)
- **Memory**: 2GB minimum, 4GB recommended
- **Storage**: 10GB for logs and state data
- **Network**: Stable internet connection with failover

### Required Packages
```bash
pip install psutil requests
```

### System Configuration
```bash
# Enable core dumps for forensic analysis
echo "core.%e.%p.%t" > /proc/sys/kernel/core_pattern

# Increase file descriptor limits
ulimit -n 65536

# Configure systemd for automatic restarts
systemctl set-property user-.slice MemoryLimit=2G
```

## Installation & Configuration

### 1. Initialize Resilience System

```python
from resilience import initialize_resilience_manager, get_resilience_manager
from watchdog import BotWatchdog, WatchdogConfig

# Initialize resilience manager
resilience_config = {
    'state_db_path': 'data/bot_state.db',
    'circuit_breaker_failure_threshold': 5,
    'circuit_breaker_recovery_timeout': 60,
    'max_retries_per_hour': 100,
    'max_no_progress_seconds': 300,
    'heartbeat_interval_seconds': 60
}

resilience_manager = initialize_resilience_manager(resilience_config)

# Initialize watchdog
watchdog_config = WatchdogConfig({
    'bot_process_name': 'trading_bot.py',
    'bot_command': ['python', 'trading_bot.py'],
    'health_check_interval': 30,
    'max_restart_attempts': 5,
    'restart_backoff_seconds': 60,
    'emergency_mode_timeout': 300
})

watchdog = BotWatchdog(watchdog_config.__dict__)
```

### 2. Integrate into Trading Bot

```python
from resilience import get_resilience_manager

class ResilientTradingBot:
    def __init__(self):
        self.resilience = get_resilience_manager()
        self.resilience.set_components(
            client=self.exchange_client,
            websocket=self.websocket_client,
            risk_manager=self.risk_manager
        )

    def execute_trade_with_resilience(self, trade_params):
        """Execute trade with full resilience features."""

        def _place_order():
            return self.exchange_client.place_order(trade_params)

        def _fallback_place_order():
            # Reduced size, market order fallback
            fallback_params = trade_params.copy()
            fallback_params['quantity'] *= 0.5  # Half size
            fallback_params['type'] = 'MARKET'
            return self.exchange_client.place_order(fallback_params)

        result = self.resilience.execute_with_resilience(
            _place_order,
            "place_order",
            max_retries=3,
            base_delay=1.0,
            fallback=_fallback_place_order
        )

        # Record successful trade
        if result:
            self.resilience.state_manager.save_trade({
                'trade_id': result['order_id'],
                'position_id': trade_params.get('position_id'),
                'symbol': trade_params['symbol'],
                'side': trade_params['side'],
                'quantity': trade_params['quantity'],
                'entry_price': result['price'],
                'pnl': 0.0,  # Will be updated on close
                'entry_time': datetime.now(timezone.utc),
                'commission': result.get('commission', 0.0)
            })

        return result

    def heartbeat(self):
        """Send heartbeat to indicate bot is alive."""
        self.resilience.record_heartbeat()
        self.resilience.black_box.record_event(
            "heartbeat", "trading_bot",
            f"Bot alive - {len(self.resilience.state_manager.load_positions())} positions"
        )
```

### 3. State Persistence Setup

```python
# Initialize state management
state_manager = resilience_manager.state_manager

# Save critical state after operations
def save_critical_state():
    """Save all critical bot state."""
    try:
        # Save positions
        for position in self.positions:
            state_manager.save_position({
                'position_id': position.id,
                'symbol': position.symbol,
                'side': position.side,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'entry_time': position.entry_time,
                'status': 'open'
            })

        # Save configuration
        state_manager.set_state_value('bot_config', self.config)
        state_manager.set_state_value('last_update', datetime.now(timezone.utc).isoformat())

    except Exception as e:
        logger.error(f"Failed to save critical state: {e}")
        resilience_manager.black_box.record_event(
            "error", "state_management", f"State save failed: {e}"
        )
```

### 4. Recovery Logic Implementation

```python
def recover_state_on_startup():
    """Recover bot state on startup."""
    try:
        positions = state_manager.load_positions()

        # Reconcile with exchange
        exchange_positions = exchange_client.get_positions()

        # Compare and reconcile
        for saved_pos in positions:
            exchange_pos = next(
                (p for p in exchange_positions if p['symbol'] == saved_pos['symbol']),
                None
            )

            if exchange_pos:
                # Verify consistency
                qty_diff = abs(saved_pos['quantity'] - exchange_pos['quantity'])
                if qty_diff > 0.001:  # Allow small differences
                    logger.warning(f"Position quantity mismatch for {saved_pos['symbol']}: "
                                 f"saved={saved_pos['quantity']}, exchange={exchange_pos['quantity']}")

                    # Prefer exchange as source of truth
                    saved_pos['quantity'] = exchange_pos['quantity']
                    state_manager.save_position(saved_pos)

                # Recalculate stop loss if missing
                if not saved_pos.get('stop_loss') and saved_pos['entry_price']:
                    # Recalculate based on risk management rules
                    stop_loss = calculate_stop_loss(saved_pos)
                    saved_pos['stop_loss'] = stop_loss
                    state_manager.save_position(saved_pos)
                    exchange_client.set_stop_loss(saved_pos['position_id'], stop_loss)

            else:
                logger.error(f"Saved position {saved_pos['position_id']} not found on exchange")
                # Handle missing position (emergency close if needed)

        logger.info(f"State recovery completed: {len(positions)} positions recovered")

    except Exception as e:
        logger.error(f"State recovery failed: {e}")
        # Continue with empty state rather than crashing
```

## Monitoring & Alerting

### Health Check Endpoint

```python
from flask import Flask, jsonify
import psutil

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Comprehensive health check endpoint."""
    resilience = get_resilience_manager()

    health_data = {
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'uptime_seconds': resilience.uptime_seconds,
        'degradation_level': resilience.degradation_level.value,
        'open_positions': len(resilience.state_manager.load_positions()),
        'circuit_breakers': {
            'api': resilience.api_circuit_breaker.state.value,
            'websocket': resilience.websocket_circuit_breaker.state.value
        },
        'system_resources': {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        },
        'last_heartbeat': resilience.last_heartbeat.isoformat(),
        'black_box_events': len(resilience.black_box.events)
    }

    # Determine overall health
    if (health_data['system_resources']['memory_percent'] > 90 or
        health_data['degradation_level'] != 'normal' or
        health_data['circuit_breakers']['api'] == 'open'):
        health_data['status'] = 'degraded'

    return jsonify(health_data), 200 if health_data['status'] == 'healthy' else 503
```

### Grafana Dashboards

Create dashboards for resilience metrics:

1. **Circuit Breaker Status**
   ```
   Metric: trading_bot_circuit_breaker_state{breaker="api"}
   ```

2. **State Recovery Time**
   ```
   Metric: trading_bot_state_recovery_duration_seconds
   ```

3. **Black Box Events**
   ```
   Metric: trading_bot_black_box_events_total
   ```

4. **Degradation Level**
   ```
   Metric: trading_bot_degradation_level
   ```

## Emergency Procedures

### Environment-Based Emergency Shutdown

```bash
# Trigger emergency shutdown
export BOT_EMERGENCY_SHUTDOWN=true

# Or via file
echo "true" > /tmp/bot_emergency_shutdown
```

### Position Limit Enforcement

```python
def enforce_position_limits():
    """Enforce position size and leverage limits."""
    positions = state_manager.load_positions()
    total_exposure = sum(abs(p['quantity'] * p['current_price']) for p in positions)

    max_exposure = self.config.get('max_total_exposure', 10000)  # $10k max

    if total_exposure > max_exposure:
        logger.critical(f"Position limit exceeded: ${total_exposure:.2f} > ${max_exposure:.2f}")

        # Reduce positions proportionally
        reduction_factor = max_exposure / total_exposure
        for position in positions:
            new_quantity = position['quantity'] * reduction_factor
            # Close partial position
            close_position_partial(position['position_id'], position['quantity'] - new_quantity)
```

### Daily Loss Limit

```python
def check_daily_loss_limit():
    """Check and enforce daily loss limit."""
    today = datetime.now(timezone.utc).date()
    daily_pnl = calculate_daily_pnl(today)

    max_daily_loss = self.config.get('max_daily_loss_pct', 0.05) * initial_balance

    if daily_pnl < -max_daily_loss:
        logger.critical(f"Daily loss limit reached: {daily_pnl:.2f} < {-max_daily_loss:.2f}")
        initiate_emergency_shutdown("Daily loss limit exceeded")
```

## Testing & Validation

### Automated Recovery Testing

```python
from test_recovery_scenarios import run_recovery_tests

# Run comprehensive recovery tests
if __name__ == "__main__":
    success = run_recovery_tests()
    if not success:
        print("❌ Recovery tests failed - do not deploy!")
        exit(1)
    else:
        print("✅ All recovery tests passed!")
```

### Chaos Engineering

```python
def simulate_failures():
    """Simulate various failure scenarios."""

    # Network failure
    simulate_network_outage(duration_seconds=30)

    # API timeout
    simulate_api_timeout(endpoint="place_order", duration_seconds=60)

    # Database corruption
    simulate_database_corruption()

    # Memory pressure
    simulate_memory_pressure(target_usage_pct=95)

    # Process crash
    simulate_process_crash(signal=signal.SIGKILL)
```

### Load Testing

```python
def stress_test_resilience():
    """Test resilience under load."""

    # High-frequency trading simulation
    for i in range(1000):
        # Simulate rapid order placement
        place_order_with_resilience(test_order_params)

        # Random failures
        if random.random() < 0.1:  # 10% failure rate
            simulate_api_failure()

        time.sleep(0.1)  # 10 orders/second

    # Check system stability
    health = get_resilience_manager().get_system_health()
    assert health['degradation_level'] == 'normal'
    assert health['circuit_breakers']['api'] != 'open'
```

## Deployment & Operations

### Systemd Service Configuration

```ini
[Unit]
Description=Trading Bot with Resilience
After=network.target
Wants=network.target

[Service]
Type=simple
User=trading
Group=trading
WorkingDirectory=/opt/trading-bot
ExecStart=/usr/bin/python3 trading_bot.py
ExecStartPost=/usr/bin/python3 watchdog.py --daemon
Restart=always
RestartSec=10

# Resource limits
MemoryLimit=2G
CPUQuota=200%

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/trading-bot/data /opt/trading-bot/logs
ProtectHome=true

[Install]
WantedBy=multi-user.target
```

### Docker Container Setup

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create trading user
RUN useradd -m -s /bin/bash trading

# Set up application
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set proper permissions
RUN chown -R trading:trading /app
USER trading

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "trading_bot.py"]
```

### Production Monitoring

```python
def setup_production_monitoring():
    """Set up comprehensive production monitoring."""

    # Prometheus metrics
    from metrics import initialize_metrics
    metrics = initialize_metrics(port=8000)

    # Health check server
    from health_check import start_health_server
    health_server = start_health_server(port=8001)

    # Alert manager
    from alerting import initialize_alert_manager
    alerts = initialize_alert_manager({
        'channels': {
            'email': {'enabled': True, 'smtp_server': 'smtp.gmail.com', ...},
            'telegram': {'enabled': True, 'bot_token': os.getenv('TELEGRAM_TOKEN')},
            'pagerduty': {'enabled': True, 'routing_key': os.getenv('PAGERDUTY_KEY')}
        }
    })

    # Log all critical events
    logging.getLogger().addHandler(create_resilience_log_handler())
```

## Maintenance Procedures

### Daily Checks
- [ ] Review alert history
- [ ] Check system resource usage
- [ ] Verify backup integrity
- [ ] Monitor position reconciliation

### Weekly Maintenance
- [ ] Run recovery scenario tests
- [ ] Review black box event logs
- [ ] Update resilience configurations
- [ ] Test emergency shutdown procedures

### Monthly Reviews
- [ ] Analyze failure patterns
- [ ] Update incident response procedures
- [ ] Review and optimize circuit breaker settings
- [ ] Test full system recovery

## Troubleshooting

### Common Issues

**Circuit Breaker Stuck Open**
```python
# Manual reset
resilience = get_resilience_manager()
resilience.api_circuit_breaker.state = CircuitBreakerState.CLOSED
resilience.api_circuit_breaker.failure_count = 0
```

**State Database Corruption**
```python
# Recreate database
state_manager.close()
os.remove('data/bot_state.db')
# Restart will create new database
```

**Watchdog Not Restarting**
```bash
# Check watchdog logs
tail -f logs/watchdog.log

# Manual restart
pkill -f watchdog.py
python watchdog.py --daemon
```

**Memory Leaks**
```python
# Force garbage collection
import gc
gc.collect()

# Check for object references
import objgraph
objgraph.show_most_common_types(limit=20)
```

## Performance Optimization

### Circuit Breaker Tuning
```python
# Conservative settings for stable environments
circuit_breaker = CircuitBreaker(
    failure_threshold=10,  # More tolerant
    recovery_timeout=120   # Longer recovery time
)

# Aggressive settings for high-reliability requirements
circuit_breaker = CircuitBreaker(
    failure_threshold=3,   # Quick to open
    recovery_timeout=30    # Fast recovery attempts
)
```

### Database Optimization
```python
# Connection pooling for high-frequency operations
import sqlite3

class PooledStateManager(StateManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._connection_pool = queue.Queue(maxsize=10)

    def get_connection(self):
        try:
            return self._connection_pool.get_nowait()
        except queue.Empty:
            return sqlite3.connect(self.db_path)
```

### Memory Management
```python
# Limit black box size in memory-constrained environments
black_box = BlackBoxRecorder(max_events=500)  # Reduced from 1000

# Periodic cleanup
def cleanup_old_data():
    """Clean up old cached data and logs."""
    # Remove expired cache entries
    resilience.price_cache = {
        k: v for k, v in resilience.price_cache.items()
        if not v.is_expired()
    }

    # Rotate log files
    rotate_logs_if_needed()
```

## Security Considerations

### State Data Protection
```python
# Encrypt sensitive state data
from cryptography.fernet import Fernet

class EncryptedStateManager(StateManager):
    def __init__(self, *args, encryption_key=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cipher = Fernet(encryption_key) if encryption_key else None

    def _encrypt_value(self, value):
        if self.cipher and isinstance(value, str):
            return self.cipher.encrypt(value.encode()).decode()
        return value

    def _decrypt_value(self, value):
        if self.cipher and isinstance(value, str):
            return self.cipher.decrypt(value.encode()).decode()
        return value
```

### Access Control
```python
# Restrict access to resilience endpoints
from flask_httpauth import HTTPBasicAuth

auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    return username == os.getenv('HEALTH_USERNAME') and \
           password == os.getenv('HEALTH_PASSWORD')

@app.route('/health')
@auth.login_required
def health_check():
    return get_resilience_manager().get_system_health()
```

## Conclusion

The resilience system transforms your trading bot from a fragile application into a bulletproof trading platform. Key benefits:

- **Zero-downtime operation** through automatic recovery
- **Position protection** even during catastrophic failures
- **Comprehensive monitoring** for proactive issue detection
- **Forensic analysis** for rapid incident resolution
- **Scalable architecture** for high-frequency trading

**Implementation Checklist:**
- [ ] Resilience system integrated
- [ ] Watchdog configured and tested
- [ ] State recovery validated
- [ ] Emergency procedures documented
- [ ] Monitoring dashboards created
- [ ] Recovery tests passing
- [ ] Incident response procedures established

Your trading bot is now ready for 24/7 operation with enterprise-grade reliability! 🛡️
