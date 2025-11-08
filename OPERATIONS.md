# Operations Guide

This comprehensive operations guide covers daily operations, performance monitoring, alert response, maintenance tasks, debugging procedures, and performance tuning for the Hyperliquid MACD trading bot.

## 📋 Table of Contents

- [Daily Operations Checklist](#-daily-operations-checklist)
- [Performance Monitoring](#-performance-monitoring)
- [Alert Response Runbook](#-alert-response-runbook)
- [Common Maintenance Tasks](#-common-maintenance-tasks)
- [Debugging Guide](#-debugging-guide)
- [Performance Tuning](#-performance-tuning)
- [Health Checks](#-health-checks)
- [Backup and Recovery](#-backup-and-recovery)

## 📅 Daily Operations Checklist

### Morning Pre-Market Check (8:00 AM)

```bash
#!/bin/bash
# daily_pre_market_check.sh

echo "=== DAILY PRE-MARKET CHECK ==="
echo "Time: $(date)"

# 1. System Health Check
echo "1. System Health..."
curl -s http://localhost:8001/health | jq '.status'
if [ $? -ne 0 ]; then
    echo "❌ Health check failed"
    exit 1
fi

# 2. Service Status
echo "2. Service Status..."
sudo systemctl status trading-bot --no-pager -l
sudo systemctl status trading-bot-watchdog --no-pager -l

# 3. Resource Usage
echo "3. Resource Usage..."
echo "CPU: $(uptime | awk '{print $NF}')"
echo "Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
echo "Disk: $(df -h / | tail -1 | awk '{print $5}')"

# 4. Network Connectivity
echo "4. Network Connectivity..."
ping -c 3 api.hyperliquid.xyz > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ API connectivity OK"
else
    echo "❌ API connectivity failed"
fi

# 5. Configuration Validation
echo "5. Configuration Validation..."
cd /home/trading/trading-bot
source venv/bin/activate
python3.11 -c "
import json
config = json.load(open('config/config.json'))
print('✅ Config loaded successfully')
print(f'   Testnet: {config.get(\"testnet\", True)}')
print(f'   Dry run: {config.get(\"dry_run\", True)}')
"

# 6. Recent Logs Check
echo "6. Recent Logs..."
echo "Errors in last hour:"
grep -c "ERROR\|CRITICAL" logs/trading_bot.log | tail -1

echo "Warnings in last hour:"
grep -c "WARNING" logs/trading_bot.log | tail -1

echo "✅ Pre-market check complete"
```

### Market Hours Monitoring (9:30 AM - 4:00 PM)

#### Real-time Monitoring Commands

```bash
# Watch live metrics
watch -n 30 'curl -s http://localhost:8000 | jq ".data.result[] | select(.metric.__name__==\"trading_pnl_total\") | .value[1]"'

# Monitor position changes
tail -f logs/trading_bot.log | grep -E "(BUY|SELL|position|stop_loss)"

# Check API latency
curl -s http://localhost:8000 | jq '.data.result[] | select(.metric.__name__=="api_request_duration_seconds") | .value[1] | tonumber | .*1000 | floor'

# Monitor system resources
htop -p $(pgrep -f trading_bot.py)
```

#### Key Metrics to Monitor

| Metric | Normal Range | Warning | Critical |
|--------|--------------|---------|----------|
| **Win Rate** | 55-65% | <50% | <40% |
| **API Latency** | <500ms | >1s | >5s |
| **Memory Usage** | <70% | >80% | >90% |
| **CPU Usage** | <50% | >70% | >90% |
| **Daily P&L** | N/A | >-3% | >-5% |
| **Position Size** | <5% balance | >10% | >15% |

### End-of-Day Review (4:00 PM)

```bash
#!/bin/bash
# end_of_day_review.sh

echo "=== END OF DAY REVIEW ==="
echo "Date: $(date)"

# 1. Trading Performance
echo "1. Trading Performance Summary"
cd /home/trading/trading-bot
source venv/bin/activate

python3.11 -c "
from trade_analytics import TradeAnalytics
analytics = TradeAnalytics('data/bot_state.db')

today = datetime.now().date()
summary = analytics.get_daily_summary(today)

print(f'Trades: {summary[\"total_trades\"]}')
print(f'Win Rate: {summary[\"win_rate\"]:.1%}')
print(f'P&L: ${summary[\"total_pnl\"]:.2f}')
print(f'Best Trade: ${summary[\"best_trade\"]:.2f}')
print(f'Worst Trade: ${summary[\"worst_trade\"]:.2f}')
"

# 2. System Performance
echo "2. System Performance"
echo "Uptime: $(uptime -p)"
echo "Load Average: $(uptime | awk '{print $NF}')"
echo "Memory Peak: $(grep 'VmPeak' /proc/\$(pgrep -f trading_bot.py)/status 2>/dev/null | awk '{print \$2 \$3}')"

# 3. Error Analysis
echo "3. Error Analysis"
echo "Total errors today: $(grep -c 'ERROR\|CRITICAL' logs/trading_bot.log)"
echo "Most common errors:"
grep 'ERROR\|CRITICAL' logs/trading_bot.log | awk '{print \$4}' | sort | uniq -c | sort -nr | head -5

# 4. API Usage
echo "4. API Usage"
echo "Total API calls today: $(grep -c 'API call' logs/trading_bot.log)"
echo "Failed API calls: $(grep -c 'API.*failed\|API.*error' logs/trading_bot.log)"

# 5. Backup Verification
echo "5. Backup Verification"
if [ -f "backups/trading_bot_$(date +%Y%m%d)*.tar.gz" ]; then
    echo "✅ Daily backup exists"
else
    echo "❌ Daily backup missing"
fi

# 6. Generate Daily Report
echo "6. Generating Daily Report..."
python3.11 -c "
from trade_analytics import TradeAnalytics
analytics = TradeAnalytics('data/bot_state.db')
analytics.generate_daily_report(datetime.now().date(), 'reports/')
print('✅ Daily report generated')
"

echo "✅ End of day review complete"
```

### Weekly Maintenance (Saturday 10:00 AM)

```bash
#!/bin/bash
# weekly_maintenance.sh

echo "=== WEEKLY MAINTENANCE ==="
echo "Date: $(date)"

# 1. System Updates
echo "1. System Updates..."
sudo apt update && sudo apt upgrade -y

# 2. Log Rotation
echo "2. Log Rotation..."
sudo logrotate -f /etc/logrotate.d/trading-bot

# 3. Database Optimization
echo "3. Database Maintenance..."
cd /home/trading/trading-bot
source venv/bin/activate

python3.11 -c "
import sqlite3
conn = sqlite3.connect('data/bot_state.db')
conn.execute('VACUUM')
conn.execute('ANALYZE')
conn.close()
print('✅ Database optimized')
"

# 4. Performance Benchmark
echo "4. Performance Benchmark..."
python3.11 performance_benchmark.py --quick

# 5. Security Audit
echo "5. Security Audit..."
python3.11 security_test_suite.py --quick

# 6. Backup Integrity Check
echo "6. Backup Integrity Check..."
find backups/ -name "*.tar.gz" -mtime -7 -exec tar -tzf {} \; > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ All backups are valid"
else
    echo "❌ Corrupted backups found"
fi

# 7. Configuration Review
echo "7. Configuration Review..."
git diff HEAD~7 config/config.json || echo "No config changes this week"

echo "✅ Weekly maintenance complete"
```

## 📊 Performance Monitoring

### Real-time Metrics Dashboard

#### Trading Performance Metrics

```bash
# Current P&L
curl -s http://localhost:8000 | jq '.data.result[] | select(.metric.__name__=="trading_pnl_total") | .value[1]'

# Win Rate (last 20 trades)
curl -s http://localhost:8000 | jq '.data.result[] | select(.metric.__name__=="trading_win_rate") | .value[1]'

# Active Positions
curl -s http://localhost:8000 | jq '.data.result[] | select(.metric.__name__=="trading_positions_active") | .value[1]'

# Sharpe Ratio
curl -s http://localhost:8000 | jq '.data.result[] | select(.metric.__name__=="trading_sharpe_ratio") | .value[1]'
```

#### System Performance Metrics

```bash
# Memory Usage
curl -s http://localhost:8000 | jq '.data.result[] | select(.metric.__name__=="process_resident_memory_bytes") | .value[1] | . / 1024 / 1024 | floor'

# CPU Usage
ps aux --no-headers -o pcpu -C python3.11 | awk '{sum+=$1} END {print sum "%"}'

# Disk I/O
iostat -d 1 1 | grep sda | awk '{print "Read: " $3 " kB/s, Write: " $4 " kB/s"}'

# Network I/O
sar -n DEV 1 1 | grep -E "eth0|ens" | tail -1 | awk '{print "RX: " $5 " kb/s, TX: " $6 " kb/s"}'
```

#### API Performance Metrics

```bash
# API Latency Distribution
curl -s http://localhost:8000 | jq '.data.result[] | select(.metric.__name__=="api_request_duration_seconds" and .metric.quantile=="0.95") | .value[1]'

# API Error Rate
curl -s http://localhost:8000 | jq '.data.result[] | select(.metric.__name__=="api_requests_total" and .metric.status=="error") | .value[1]'

# WebSocket Connection Status
curl -s http://localhost:8001/health | jq '.websocket.connected'

# Rate Limit Status
curl -s http://localhost:8000 | jq '.data.result[] | select(.metric.__name__=="rate_limiter_requests_total") | .value[1]'
```

### Custom Monitoring Queries

#### Trading Performance Analysis

```python
from trade_analytics import TradeAnalytics
from datetime import datetime, timedelta

analytics = TradeAnalytics('data/bot_state.db')

# Last 24 hours performance
end_date = datetime.now()
start_date = end_date - timedelta(days=1)

performance = analytics.get_performance_metrics(start_date, end_date)
print(f"24h P&L: ${performance['total_pnl']:.2f}")
print(f"Win Rate: {performance['win_rate']:.1%}")
print(f"Profit Factor: {performance['profit_factor']:.2f}")
print(f"Max Drawdown: {performance['max_drawdown']:.1%}")
```

#### Risk Metrics Monitoring

```python
from risk_manager import RiskManager

risk_manager = RiskManager(config)

# Current risk exposure
current_balance = client.get_account_balance()
positions = client.get_positions()

risk_metrics = risk_manager.calculate_portfolio_risk(current_balance, positions)
print(f"Current Exposure: {risk_metrics['total_exposure_pct']:.1%}")
print(f"Max Position Size: {risk_metrics['max_position_pct']:.1%}")
print(f"Daily Loss Limit: {risk_metrics['daily_loss_limit']:.1%}")
print(f"VaR 95%: ${risk_metrics['value_at_risk']:.2f}")
```

#### Strategy Effectiveness

```python
from macd_strategy_enhanced import EnhancedMACDStrategy

strategy = EnhancedMACDStrategy(config)

# Strategy performance by market regime
regime_performance = strategy.analyze_regime_performance()
for regime, metrics in regime_performance.items():
    print(f"{regime}: Win Rate {metrics['win_rate']:.1%}, Profit Factor {metrics['profit_factor']:.2f}")
```

## 🚨 Alert Response Runbook

### Critical Alerts (Immediate Action Required)

#### 1. Bot Crashed/Stops Trading

**Alert Pattern:**
```
CRITICAL - Bot process not responding
CRITICAL - Trading bot exited unexpectedly
```

**Response Steps:**
```bash
# 1. Check service status
sudo systemctl status trading-bot

# 2. Check logs for crash reason
sudo journalctl -u trading-bot -n 50 --no-pager

# 3. Attempt automatic restart
sudo systemctl restart trading-bot

# 4. If restart fails, check system resources
free -h
df -h
ps aux --sort=-%mem | head -10

# 5. Manual recovery if needed
cd /home/trading/trading-bot
source venv/bin/activate
python3.11 scripts/recovery.sh

# 6. Notify team and document incident
```

**Escalation:**
- If bot doesn't restart within 5 minutes: Wake up on-call engineer
- If positions are open: Consider manual position closure

#### 2. WebSocket Disconnected >2 Minutes

**Alert Pattern:**
```
CRITICAL - WebSocket disconnected for 120+ seconds
CRITICAL - Real-time data feed lost
```

**Response Steps:**
```bash
# 1. Check network connectivity
ping api.hyperliquid.xyz

# 2. Restart WebSocket connection
sudo systemctl restart trading-bot

# 3. Verify reconnection
curl -s http://localhost:8001/health | jq '.websocket'

# 4. Check for API rate limiting
grep "rate limit" logs/trading_bot.log | tail -5

# 5. If persistent, switch to REST polling
# Edit config to disable WebSocket temporarily
sed -i 's/"websocket": {"enabled": true}/"websocket": {"enabled": false}/' config/config.json
sudo systemctl restart trading-bot
```

#### 3. Stop-Loss Triggered on Large Position

**Alert Pattern:**
```
CRITICAL - Stop-loss triggered on position >$X
CRITICAL - Large loss realized
```

**Response Steps:**
```bash
# 1. Verify position closure
curl -s http://localhost:8001/health | jq '.positions[] | select(.symbol=="BTCUSDT")'

# 2. Check P&L impact
python3.11 -c "
from trade_analytics import TradeAnalytics
analytics = TradeAnalytics('data/bot_state.db')
today_pnl = analytics.get_daily_pnl()
print(f'Today P&L: ${today_pnl:.2f}')
"

# 3. Assess strategy health
grep "win rate" logs/trading_bot.log | tail -1

# 4. Consider trading pause if losses are large
if [ $(echo "$today_pnl < -500" | bc -l) -eq 1 ]; then
    echo "BOT_EMERGENCY_SHUTDOWN=true" >> ~/.trading_env
    sudo systemctl restart trading-bot
fi
```

#### 4. Daily Loss Limit Reached

**Alert Pattern:**
```
CRITICAL - Daily loss limit exceeded: -$X (Y% of balance)
```

**Response Steps:**
```bash
# 1. Confirm loss limit breach
python3.11 -c "
from risk_manager import RiskManager
risk = RiskManager(config)
daily_loss = risk.get_daily_loss()
print(f'Daily Loss: ${daily_loss:.2f}')
"

# 2. Stop all trading immediately
echo "BOT_EMERGENCY_SHUTDOWN=true" >> ~/.trading_env
sudo systemctl restart trading-bot

# 3. Close any remaining positions manually if needed
# Check positions
curl -s http://localhost:8001/health | jq '.positions'

# 4. Document incident
echo "$(date): Daily loss limit exceeded - Trading stopped" >> logs/incidents.log

# 5. Review strategy settings for next day
```

#### 5. API Authentication Failed

**Alert Pattern:**
```
CRITICAL - API authentication failed
CRITICAL - Invalid API credentials
```

**Response Steps:**
```bash
# 1. Check API key validity
cd /home/trading/trading-bot
source venv/bin/activate
python3.11 test_hyperliquid_connection.py

# 2. Verify key format
python3.11 -c "
from secure_key_storage import SecureKeyStorage
storage = SecureKeyStorage()
key = storage.get_private_key()
print(f'Key format valid: {key.startswith(\"0x\") and len(key) == 66}')
"

# 3. Check for key rotation needs
grep "key.*expir" logs/trading_bot.log | tail -5

# 4. If keys are invalid, emergency shutdown
echo "BOT_EMERGENCY_SHUTDOWN=true" >> ~/.trading_env
sudo systemctl restart trading-bot

# 5. Alert security team immediately
```

### Warning Alerts (Review Within 30 Minutes)

#### 1. API Latency >5 Seconds

**Response Steps:**
```bash
# 1. Check current latency
curl -s http://localhost:8000 | jq '.data.result[] | select(.metric.__name__=="api_request_duration_seconds" and .metric.quantile=="0.95") | .value[1]'

# 2. Test API connectivity
time curl -s https://api.hyperliquid.xyz/info > /dev/null

# 3. Check system load
uptime
free -h

# 4. Restart if latency persists
if [ $(curl -s http://localhost:8000 | jq '.data.result[] | select(.metric.__name__=="api_request_duration_seconds" and .metric.quantile=="0.95") | .value[1] | . > 5') ]; then
    sudo systemctl restart trading-bot
fi
```

#### 2. Memory Usage >80%

**Response Steps:**
```bash
# 1. Check memory usage
free -h
ps aux --sort=-%mem | head -10

# 2. Clear system cache if needed
echo 3 > /proc/sys/vm/drop_caches

# 3. Restart service if memory >90%
memory_pct=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
if [ $memory_pct -gt 90 ]; then
    sudo systemctl restart trading-bot
fi

# 4. Check for memory leaks
grep "memory" logs/trading_bot.log | tail -10
```

#### 3. Win Rate Drops Below 40%

**Response Steps:**
```bash
# 1. Verify win rate
curl -s http://localhost:8000 | jq '.data.result[] | select(.metric.__name__=="trading_win_rate") | .value[1]'

# 2. Check market conditions
python3.11 -c "
from macd_strategy_enhanced import EnhancedMACDStrategy
strategy = EnhancedMACDStrategy(config)
regime = strategy.get_market_condition()
print(f'Current market regime: {regime}')
"

# 3. Consider strategy adjustment
if [ "$(curl -s http://localhost:8000 | jq '.data.result[] | select(.metric.__name__=="trading_win_rate") | .value[1] | . < 0.4')" == "true" ]; then
    # Reduce position sizes
    sed -i 's/"max_position_size_pct": 0.05/"max_position_size_pct": 0.02/' config/config.json
    sudo systemctl restart trading-bot
fi
```

### Info Alerts (Daily Digest)

#### Daily Trading Summary

```python
from trade_analytics import TradeAnalytics

analytics = TradeAnalytics('data/bot_state.db')

# Generate daily summary
summary = analytics.get_daily_summary(datetime.now().date())

alert_message = f"""
📊 Daily Trading Summary

Trades: {summary['total_trades']}
Win Rate: {summary['win_rate']:.1%}
P&L: ${summary['total_pnl']:.2f}
Best Trade: ${summary['best_trade']:.2f}
Worst Trade: ${summary['worst_trade']:.2f}

Market Regimes:
{chr(10).join(f"- {regime}: {count} trades" for regime, count in summary['regime_distribution'].items())}
"""

alerts.send_email("Daily Trading Summary", alert_message)
```

## 🔧 Common Maintenance Tasks

### Log Management

#### Log Rotation
```bash
# Force log rotation
sudo logrotate -f /etc/logrotate.d/trading-bot

# Compress old logs
find logs/ -name "*.log.*" -mtime +30 -exec gzip {} \;

# Clean old compressed logs (keep 90 days)
find logs/ -name "*.gz" -mtime +90 -delete
```

#### Log Analysis
```bash
# Most common errors
grep "ERROR\|CRITICAL" logs/trading_bot.log | \
    awk '{print $4}' | \
    sort | \
    uniq -c | \
    sort -nr | \
    head -10

# API call patterns
grep "API call" logs/trading_bot.log | \
    awk '{print $3}' | \
    sort | \
    uniq -c | \
    sort -nr

# Trading frequency
grep "Trade executed" logs/trading_bot.log | \
    awk '{print $1}' | \
    cut -d'T' -f1 | \
    sort | \
    uniq -c
```

### Database Maintenance

#### Vacuum and Analyze
```python
import sqlite3

def optimize_database(db_path: str):
    """Optimize SQLite database."""
    conn = sqlite3.connect(db_path)

    # Vacuum to reclaim space
    conn.execute("VACUUM")

    # Analyze for query optimization
    conn.execute("ANALYZE")

    # Check integrity
    result = conn.execute("PRAGMA integrity_check").fetchone()
    if result[0] != "ok":
        raise Exception(f"Database integrity check failed: {result[0]}")

    conn.close()
    print("✅ Database optimized")

# Run optimization
optimize_database("data/bot_state.db")
```

#### Backup Database
```bash
# Create database backup
sqlite3 data/bot_state.db ".backup 'backups/bot_state_$(date +%Y%m%d_%H%M%S).db'"

# Verify backup
sqlite3 backups/bot_state_latest.db "SELECT COUNT(*) FROM trades;" > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Database backup successful"
else
    echo "❌ Database backup failed"
fi
```

### Configuration Management

#### Configuration Validation
```python
import json
import jsonschema

def validate_config(config_path: str) -> bool:
    """Validate configuration file."""
    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Define schema
    schema = {
        "type": "object",
        "required": ["exchange", "trading", "strategy", "risk"],
        "properties": {
            "exchange": {"type": "string", "enum": ["hyperliquid"]},
            "testnet": {"type": "boolean"},
            "trading": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "timeframe": {"type": "string"},
                    "check_interval": {"type": "number", "minimum": 60}
                }
            }
        }
    }

    try:
        jsonschema.validate(config, schema)
        return True
    except jsonschema.ValidationError as e:
        print(f"Configuration validation failed: {e}")
        return False

# Validate current config
if validate_config("config/config.json"):
    print("✅ Configuration is valid")
else:
    print("❌ Configuration has errors")
```

#### Configuration Backup
```bash
# Backup current configuration
cp config/config.json config/config.json.backup.$(date +%s)

# Track configuration changes
git add config/config.json
git commit -m "Configuration update $(date)"
```

### Security Updates

#### Update Dependencies
```bash
cd /home/trading/trading-bot

# Activate virtual environment
source venv/bin/activate

# Update pip
pip install --upgrade pip

# Update dependencies (cautiously)
pip install --upgrade -r requirements.txt

# Test after update
python3.11 -c "import trading_bot; print('✅ Imports successful')"

# Restart service
sudo systemctl restart trading-bot

deactivate
```

#### Security Scanning
```python
import subprocess

def run_security_scan():
    """Run security scans on codebase."""
    results = {}

    # Bandit security linting
    try:
        result = subprocess.run(
            ["bandit", "-r", ".", "--exclude", "./venv,./backups"],
            capture_output=True, text=True, timeout=300
        )
        results['bandit'] = {
            'returncode': result.returncode,
            'output': result.stdout,
            'errors': result.stderr
        }
    except subprocess.TimeoutExpired:
        results['bandit'] = {'error': 'Scan timeout'}

    # Check for sensitive data in logs
    try:
        result = subprocess.run(
            ["grep", "-r", "0x[a-fA-F0-9]\{64\}", "logs/"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            results['sensitive_logs'] = result.stdout
        else:
            results['sensitive_logs'] = None
    except Exception as e:
        results['sensitive_logs'] = str(e)

    return results

# Run security scan
scan_results = run_security_scan()
if scan_results['sensitive_logs']:
    print("🚨 SENSITIVE DATA FOUND IN LOGS")
    print(scan_results['sensitive_logs'])
```

## 🐛 Debugging Guide

### Common Issues and Solutions

#### Bot Not Starting

**Symptoms:**
- Service shows as failed
- No trading activity
- Logs show import errors

**Debug Steps:**
```bash
# Check service status
sudo systemctl status trading-bot

# Check logs
sudo journalctl -u trading-bot -n 20

# Test manual startup
cd /home/trading/trading-bot
source venv/bin/activate
python3.11 trading_bot.py --dry-run --log-level DEBUG

# Check dependencies
python3.11 -c "import pandas, numpy, websockets; print('✅ Dependencies OK')"
```

#### Trading Stops Unexpectedly

**Symptoms:**
- No new trades for extended period
- Bot appears healthy but inactive

**Debug Steps:**
```bash
# Check strategy signals
python3.11 -c "
from macd_strategy_enhanced import EnhancedMACDStrategy
strategy = EnhancedMACDStrategy(config)
# Test signal generation
print('Strategy loaded successfully')
"

# Check market data
curl -s http://localhost:8001/health | jq '.market_data'

# Review recent logs
grep "strategy\|signal" logs/trading_bot.log | tail -10

# Check risk limits
python3.11 -c "
from risk_manager import RiskManager
risk = RiskManager(config)
limits = risk.get_current_limits()
print(f'Risk limits: {limits}')
"
```

#### High Memory Usage

**Symptoms:**
- System slow or unresponsive
- Memory usage >80%

**Debug Steps:**
```bash
# Check memory usage
ps aux --sort=-%mem | head -10

# Profile memory usage
python3.11 -c "
import psutil
import os
process = psutil.Process(os.getpid())
mem_info = process.memory_info()
print(f'Memory usage: {mem_info.rss / 1024 / 1024:.1f} MB')
"

# Check for memory leaks
python3.11 -m tracemalloc trading_bot.py --dry-run --profile-duration 60

# Restart with memory monitoring
sudo systemctl restart trading-bot
watch -n 10 'ps aux --no-headers -o pmem -C python3.11'
```

#### API Connection Issues

**Symptoms:**
- API errors in logs
- Failed trades
- WebSocket disconnections

**Debug Steps:**
```bash
# Test basic connectivity
ping api.hyperliquid.xyz

# Test API endpoint
curl -v https://api.hyperliquid.xyz/info

# Test WebSocket
python3.11 -c "
import asyncio
import websockets

async def test_ws():
    try:
        async with websockets.connect('wss://api.hyperliquid.xyz/ws') as ws:
            print('✅ WebSocket connection successful')
    except Exception as e:
        print(f'❌ WebSocket connection failed: {e}')

asyncio.run(test_ws())
"

# Check rate limits
grep "rate limit" logs/trading_bot.log | tail -5

# Test authentication
python3.11 test_hyperliquid_connection.py
```

### Advanced Debugging

#### Performance Profiling

```python
import cProfile
import pstats
from io import StringIO

def profile_function(func, *args, **kwargs):
    """Profile a function's performance."""
    pr = cProfile.Profile()
    pr.enable()

    result = func(*args, **kwargs)

    pr.disable()
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    return result

# Profile strategy calculation
from macd_strategy_enhanced import EnhancedMACDStrategy
strategy = EnhancedMACDStrategy(config)

profile_function(strategy.calculate_indicators, df)
```

#### Memory Debugging

```python
import tracemalloc
import gc

def trace_memory_usage():
    """Trace memory usage over time."""
    tracemalloc.start()

    # Take initial snapshot
    snapshot1 = tracemalloc.take_snapshot()

    # Run some operations
    for i in range(100):
        df = client.get_klines('BTCUSDT', '5m', 100)
        indicators = strategy.calculate_indicators(df)

    # Take second snapshot
    snapshot2 = tracemalloc.take_snapshot()

    # Compare snapshots
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    print("Top memory allocations:")
    for stat in top_stats[:10]:
        print(stat)

    tracemalloc.stop()

trace_memory_usage()
```

#### Thread Debugging

```python
import threading
import sys

def dump_threads():
    """Dump information about all threads."""
    for thread in threading.enumerate():
        print(f"Thread: {thread.name} (ID: {thread.ident})")
        print(f"  Alive: {thread.is_alive()}")
        print(f"  Daemon: {thread.daemon}")

        # Get stack trace
        frame = sys._current_frames().get(thread.ident)
        if frame:
            print("  Stack trace:")
            for line in traceback.format_stack(frame):
                print(f"    {line.strip()}")

dump_threads()
```

## ⚡ Performance Tuning

### System-Level Optimization

#### Kernel Parameters
```bash
# Network optimization
sudo sysctl -w net.core.somaxconn=65536
sudo sysctl -w net.ipv4.tcp_tw_reuse=1
sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535"

# Memory management
sudo sysctl -w vm.swappiness=10
sudo sysctl -w vm.dirty_ratio=60
sudo sysctl -w vm.dirty_background_ratio=2

# Make permanent
echo "net.core.somaxconn=65536" >> /etc/sysctl.conf
echo "vm.swappiness=10" >> /etc/sysctl.conf
```

#### CPU Affinity
```bash
# Pin trading bot to specific CPU cores
taskset -c 0-3 -p $(pgrep -f trading_bot.py)

# Make permanent in systemd service
sudo tee -a /etc/systemd/system/trading-bot.service > /dev/null <<EOF
CPUAffinity=0-3
EOF
sudo systemctl daemon-reload
```

### Application-Level Optimization

#### Async/Await Optimization
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class OptimizedTradingBot:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.loop = asyncio.get_event_loop()

    async def parallel_api_calls(self):
        """Execute multiple API calls in parallel."""
        tasks = [
            self.loop.run_in_executor(self.executor, client.get_ticker, symbol)
            for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def optimized_trading_loop(self):
        """Optimized main trading loop."""
        while True:
            # Parallel data fetching
            market_data = await self.parallel_api_calls()

            # Parallel indicator calculation
            indicator_tasks = [
                self.loop.run_in_executor(
                    self.executor,
                    strategy.calculate_indicators,
                    data
                )
                for data in market_data
            ]

            indicators = await asyncio.gather(*indicator_tasks)

            # Process signals
            await self.process_signals(indicators)

            await asyncio.sleep(1)  # 1 second interval
```

#### Memory Optimization

```python
import gc
from weakref import WeakValueDictionary

class MemoryOptimizedStrategy:
    def __init__(self):
        self.cache = WeakValueDictionary()
        self.gc_threshold = 1000

    def calculate_indicators_cached(self, df):
        """Cache indicator calculations."""
        # Create cache key
        cache_key = hash(df.values.tobytes())

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Calculate indicators
        result = self.calculate_indicators(df)

        # Cache result
        self.cache[cache_key] = result

        # Periodic garbage collection
        if len(self.cache) > self.gc_threshold:
            gc.collect()

        return result

    def optimize_dataframe_usage(self, df):
        """Optimize DataFrame memory usage."""
        # Use appropriate dtypes
        df = df.copy()

        # Convert to float32 for memory savings
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].astype(np.float32)

        # Remove unused columns
        keep_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[keep_columns]

        return df
```

### Database Optimization

#### Connection Pooling
```python
import sqlite3
from concurrent.futures import ThreadPoolExecutor

class OptimizedDatabaseManager:
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self.executor = ThreadPoolExecutor(max_workers=pool_size)
        self.connections = []

    def get_connection(self):
        """Get connection from pool."""
        if not self.connections:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None  # Autocommit mode
            )
            # Enable WAL mode
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self.connections.append(conn)

        return self.connections[0]  # Simple round-robin

    def execute_query_async(self, query: str, params: tuple = None):
        """Execute query asynchronously."""
        def _execute():
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            result = cursor.fetchall()
            cursor.close()
            return result

        return self.executor.submit(_execute)
```

### Monitoring Optimization

#### Efficient Metrics Collection
```python
from collections import deque
import time

class EfficientMetricsCollector:
    def __init__(self, max_samples: int = 1000):
        self.samples = deque(maxlen=max_samples)
        self.last_collection = 0
        self.collection_interval = 1.0  # 1 second

    def record_sample(self, name: str, value: float, labels: dict = None):
        """Record metric sample efficiently."""
        current_time = time.time()

        # Throttle collection
        if current_time - self.last_collection < self.collection_interval:
            return

        self.samples.append({
            'name': name,
            'value': value,
            'labels': labels or {},
            'timestamp': current_time
        })

        self.last_collection = current_time

    def get_aggregated_metrics(self):
        """Get aggregated metrics efficiently."""
        if not self.samples:
            return {}

        # Group by name and labels
        aggregated = {}
        for sample in self.samples:
            key = (sample['name'], frozenset(sample['labels'].items()))

            if key not in aggregated:
                aggregated[key] = {
                    'name': sample['name'],
                    'labels': sample['labels'],
                    'samples': []
                }

            aggregated[key]['samples'].append(sample['value'])

        # Calculate aggregations
        result = {}
        for key, data in aggregated.items():
            samples = data['samples']
            result[data['name']] = {
                'count': len(samples),
                'sum': sum(samples),
                'avg': sum(samples) / len(samples),
                'min': min(samples),
                'max': max(samples),
                'labels': data['labels']
            }

        return result
```

## 🏥 Health Checks

### Automated Health Checks

#### Comprehensive Health Check Script
```bash
#!/bin/bash
# comprehensive_health_check.sh

echo "=== COMPREHENSIVE HEALTH CHECK ==="
echo "Timestamp: $(date)"

# Configuration
HEALTH_LOG="logs/health_check_$(date +%Y%m%d).log"
WARNING_THRESHOLD=80
CRITICAL_THRESHOLD=90

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$HEALTH_LOG"
}

# 1. Service Health
log "1. Checking service status..."
if sudo systemctl is-active --quiet trading-bot; then
    log "✅ Trading bot service is running"
else
    log "❌ Trading bot service is not running"
    exit 1
fi

if sudo systemctl is-active --quiet trading-bot-watchdog; then
    log "✅ Watchdog service is running"
else
    log "❌ Watchdog service is not running"
    exit 1
fi

# 2. Application Health
log "2. Checking application health..."
HEALTH_RESPONSE=$(curl -s -w "%{http_code}" http://localhost:8001/health)
HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -c 3)
HEALTH_BODY=$(echo "$HEALTH_RESPONSE" | head -n -1)

if [ "$HTTP_CODE" -eq 200 ]; then
    log "✅ Health endpoint responding (HTTP $HTTP_CODE)"

    # Parse health status
    STATUS=$(echo "$HEALTH_BODY" | jq -r '.status')
    if [ "$STATUS" = "healthy" ]; then
        log "✅ Application reports healthy status"
    else
        log "⚠️ Application reports: $STATUS"
    fi
else
    log "❌ Health endpoint not responding (HTTP $HTTP_CODE)"
    exit 1
fi

# 3. System Resources
log "3. Checking system resources..."

# CPU Usage
CPU_USAGE=$(uptime | awk '{print $NF}' | sed 's/,//')
CPU_NUM=$(echo "$CPU_USAGE" | sed 's/\..*//')
if [ "$CPU_NUM" -gt "$CRITICAL_THRESHOLD" ]; then
    log "❌ CRITICAL: CPU usage at ${CPU_USAGE}%"
    exit 1
elif [ "$CPU_NUM" -gt "$WARNING_THRESHOLD" ]; then
    log "⚠️ WARNING: CPU usage at ${CPU_USAGE}%"
else
    log "✅ CPU usage at ${CPU_USAGE}%"
fi

# Memory Usage
MEM_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
if [ "$MEM_USAGE" -gt "$CRITICAL_THRESHOLD" ]; then
    log "❌ CRITICAL: Memory usage at ${MEM_USAGE}%"
    exit 1
elif [ "$MEM_USAGE" -gt "$WARNING_THRESHOLD" ]; then
    log "⚠️ WARNING: Memory usage at ${MEM_USAGE}%"
else
    log "✅ Memory usage at ${MEM_USAGE}%"
fi

# Disk Usage
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt "$CRITICAL_THRESHOLD" ]; then
    log "❌ CRITICAL: Disk usage at ${DISK_USAGE}%"
    exit 1
elif [ "$DISK_USAGE" -gt "$WARNING_THRESHOLD" ]; then
    log "⚠️ WARNING: Disk usage at ${DISK_USAGE}%"
else
    log "✅ Disk usage at ${DISK_USAGE}%"
fi

# 4. Network Connectivity
log "4. Checking network connectivity..."
if ping -c 3 -W 5 api.hyperliquid.xyz > /dev/null 2>&1; then
    log "✅ API connectivity OK"
else
    log "❌ API connectivity failed"
fi

# 5. Database Health
log "5. Checking database health..."
cd /home/trading/trading-bot
source venv/bin/activate

DB_CHECK=$(python3.11 -c "
import sqlite3
try:
    conn = sqlite3.connect('data/bot_state.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM trades')
    count = cursor.fetchone()[0]
    cursor.execute('PRAGMA integrity_check')
    integrity = cursor.fetchone()[0]
    conn.close()

    if integrity == 'ok':
        print(f'✅ Database healthy: {count} trades')
    else:
        print(f'❌ Database integrity check failed: {integrity}')
        exit(1)
except Exception as e:
    print(f'❌ Database check failed: {e}')
    exit(1)
" 2>&1)

if echo "$DB_CHECK" | grep -q "❌"; then
    log "$DB_CHECK"
    exit 1
else
    log "$DB_CHECK"
fi

# 6. Recent Activity Check
log "6. Checking recent activity..."
RECENT_TRADES=$(grep -c "Trade executed" logs/trading_bot.log)
RECENT_ERRORS=$(grep -c "ERROR\|CRITICAL" logs/trading_bot.log)

log "Recent trades: $RECENT_TRADES"
log "Recent errors: $RECENT_ERRORS"

if [ "$RECENT_ERRORS" -gt 10 ]; then
    log "⚠️ WARNING: High error count ($RECENT_ERRORS)"
fi

# 7. Backup Status
log "7. Checking backup status..."
if [ -f "backups/trading_bot_$(date +%Y%m%d)*.tar.gz" ]; then
    log "✅ Daily backup exists"
else
    log "❌ Daily backup missing"
fi

log "=== HEALTH CHECK COMPLETE ==="
echo "Results logged to: $HEALTH_LOG"
```

### Health Check Integration

#### SystemD Timer for Regular Checks
```bash
# Create health check timer
sudo tee /etc/systemd/system/trading-bot-health.timer > /dev/null <<EOF
[Unit]
Description=Run comprehensive health checks every 15 minutes

[Timer]
OnBootSec=5min
OnUnitActiveSec=15min
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Create health check service
sudo tee /etc/systemd/system/trading-bot-health.service > /dev/null <<EOF
[Unit]
Description=Trading Bot Comprehensive Health Check

[Service]
Type=oneshot
User=trading
WorkingDirectory=/home/trading/trading-bot
ExecStart=/home/trading/trading-bot/scripts/comprehensive_health_check.sh
EOF

# Enable health check timer
sudo systemctl daemon-reload
sudo systemctl enable trading-bot-health.timer
sudo systemctl start trading-bot-health.timer
```

## 💾 Backup and Recovery

### Automated Backup System

#### Daily Backup Script
```bash
#!/bin/bash
# daily_backup.sh

BACKUP_DIR="/home/trading/trading-bot/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="trading_bot_${TIMESTAMP}"

echo "Starting daily backup: $BACKUP_NAME"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Stop trading for consistent backup
sudo systemctl stop trading-bot

# Create database backup
sqlite3 data/bot_state.db ".backup '${BACKUP_DIR}/${BACKUP_NAME}.db'"

# Create full system backup
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" \
    --exclude='*.log' \
    --exclude='__pycache__' \
    --exclude='backups' \
    --exclude='*.pyc' \
    --exclude='venv' \
    /home/trading/trading-bot/

# Verify backup integrity
if tar -tzf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" > /dev/null 2>&1; then
    echo "✅ Backup created successfully: ${BACKUP_NAME}"

    # Update latest backup symlink
    ln -sf "${BACKUP_NAME}.tar.gz" "${BACKUP_DIR}/latest.tar.gz"
    ln -sf "${BACKUP_NAME}.db" "${BACKUP_DIR}/latest.db"

    # Clean old backups (keep 30 days)
    find "$BACKUP_DIR" -name "trading_bot_*.tar.gz" -mtime +30 -delete
    find "$BACKUP_DIR" -name "trading_bot_*.db" -mtime +30 -delete

    echo "✅ Old backups cleaned up"
else
    echo "❌ Backup verification failed"
    exit 1
fi

# Restart trading
sudo systemctl start trading-bot

echo "Backup completed successfully"
```

#### Recovery Procedures

#### Full System Recovery
```bash
#!/bin/bash
# disaster_recovery.sh

RECOVERY_BACKUP="$1"
if [ -z "$RECOVERY_BACKUP" ]; then
    echo "Usage: $0 <backup-file>"
    exit 1
fi

echo "Starting disaster recovery from: $RECOVERY_BACKUP"

# Stop all services
sudo systemctl stop trading-bot
sudo systemctl stop trading-bot-watchdog

# Create recovery directory
RECOVERY_DIR="/tmp/trading_recovery_$(date +%s)"
mkdir -p "$RECOVERY_DIR"

# Extract backup
if [[ "$RECOVERY_BACKUP" == *.tar.gz ]]; then
    tar -xzf "$RECOVERY_BACKUP" -C "$RECOVERY_DIR"
elif [[ "$RECOVERY_BACKUP" == *.db ]]; then
    cp "$RECOVERY_BACKUP" "$RECOVERY_DIR/data/bot_state.db"
else
    echo "Unsupported backup format"
    exit 1
fi

# Validate backup
if [ ! -f "$RECOVERY_DIR/home/trading/trading-bot/config/config.json" ]; then
    echo "❌ Invalid backup - config.json not found"
    rm -rf "$RECOVERY_DIR"
    exit 1
fi

# Backup current state
if [ -d "/home/trading/trading-bot" ]; then
    mv /home/trading/trading-bot "/home/trading/trading-bot.backup.$(date +%s)"
fi

# Restore from backup
cp -r "$RECOVERY_DIR/home/trading/trading-bot" /home/trading/

# Restore permissions
chown -R trading:trading /home/trading/trading-bot
chmod 600 /home/trading/trading-bot/config/*.json

# Test recovery
cd /home/trading/trading-bot
source venv/bin/activate
python3.11 -c "import trading_bot; print('✅ Recovery successful')"

# Clean up
rm -rf "$RECOVERY_DIR"

# Start services
sudo systemctl start trading-bot-watchdog
sudo systemctl start trading-bot

echo "✅ Disaster recovery completed"
```

#### Point-in-Time Recovery
```python
def point_in_time_recovery(target_timestamp: datetime):
    """Recover to specific point in time."""
    # Find appropriate backup
    backup_files = list_backups()
    suitable_backup = find_backup_before_timestamp(backup_files, target_timestamp)

    if not suitable_backup:
        raise ValueError("No suitable backup found")

    # Restore backup
    restore_backup(suitable_backup)

    # Replay transactions from logs
    replay_transactions_from_logs(suitable_backup.timestamp, target_timestamp)

    logger.info(f"Point-in-time recovery completed for {target_timestamp}")
```

This comprehensive operations guide provides all the tools and procedures needed to maintain, monitor, and troubleshoot the Hyperliquid MACD trading bot in production. Regular use of these procedures will ensure reliable, high-performance operation.
