# Trading Bot Monitoring Setup Guide

This guide provides comprehensive instructions for setting up monitoring, alerting, and observability for the Hyperliquid trading bot.

## Architecture Overview

The monitoring system consists of:

- **Metrics Collection**: Prometheus client for real-time metrics
- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Multi-Channel Alerting**: Email, Telegram, Discord, PagerDuty
- **Health Checks**: HTTP endpoint for external monitoring
- **Trade Analytics**: Performance analysis and reporting
- **Anomaly Detection**: Statistical detection of unusual patterns
- **Grafana Dashboards**: Real-time visualization

## Prerequisites

### System Requirements

- Python 3.11+
- 2GB RAM minimum (4GB recommended)
- 10GB disk space for logs and data
- Network access to external services

### Required Packages

```bash
pip install prometheus_client flask python-telegram-bot discord-webhook reportlab matplotlib pandas numpy scikit-learn
```

### External Services

- **Grafana** (optional): For dashboards (localhost:3000)
- **Prometheus** (optional): For metrics storage (localhost:9090)
- **SMTP Server**: For email alerts
- **Telegram Bot**: For instant messaging alerts
- **Discord Webhook**: For team channel alerts
- **PagerDuty**: For critical incident management

## Configuration

### 1. Update config/config.json

Add monitoring configuration:

```json
{
  "monitoring": {
    "enabled": true,
    "metrics_port": 8000,
    "health_port": 8001,
    "structured_logging": true,
    "alerts_enabled": true
  },
  "alerting": {
    "channels": {
      "email": {
        "enabled": true,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_username": "your-email@gmail.com",
        "smtp_password": "your-app-password",
        "from_email": "trading-bot@yourdomain.com",
        "to_emails": ["alerts@yourdomain.com"]
      },
      "telegram": {
        "enabled": true,
        "bot_token": "your-bot-token",
        "chat_ids": ["123456789"]
      },
      "discord": {
        "enabled": true,
        "webhook_url": "https://discord.com/api/webhooks/..."
      },
      "pagerduty": {
        "enabled": false,
        "routing_key": "your-routing-key"
      }
    },
    "dedup_window_minutes": 5,
    "max_notifications_per_alert": 3
  },
  "logging": {
    "structured": true,
    "correlation_ids": true,
    "log_level": "INFO",
    "max_file_size": "10MB",
    "backup_count": 5
  },
  "analytics": {
    "enabled": true,
    "report_directory": "reports",
    "auto_generate_reports": true,
    "report_frequency": "daily"
  },
  "anomaly_detection": {
    "enabled": true,
    "statistical_window": 100,
    "z_threshold": 3.0,
    "ml_enabled": true,
    "contamination": 0.1
  }
}
```

### 2. Set Environment Variables

```bash
# Telegram Bot Setup
export TELEGRAM_BOT_TOKEN="your-bot-token"
export TELEGRAM_CHAT_IDS="123456789,987654321"

# Discord Webhook
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."

# PagerDuty
export PAGERDUTY_ROUTING_KEY="your-routing-key"

# Email Configuration
export SMTP_USERNAME="your-email@gmail.com"
export SMTP_PASSWORD="your-app-password"
```

## Setup Steps

### Step 1: Initialize Monitoring Components

In your trading bot startup code:

```python
from metrics import initialize_metrics, get_metrics
from structured_logger import setup_structured_logging
from alerting import initialize_alert_manager
from health_check import initialize_health_checker, start_health_server
from anomaly_detection import initialize_anomaly_detector

# Initialize structured logging
setup_structured_logging(
    log_level="INFO",
    log_file="logs/trading_bot.log",
    error_log_file="logs/trading_bot_errors.log"
)

# Initialize metrics collection
metrics = initialize_metrics(port=8000, start_server=True)

# Initialize alerting system
alert_manager = initialize_alert_manager(config=your_config)

# Initialize health checker
health_checker = initialize_health_checker(bot_instance=your_bot)

# Start health server
health_server = start_health_server(port=8001)

# Initialize anomaly detection
anomaly_detector = initialize_anomaly_detector(config=your_config)
```

### Step 2: Integrate Monitoring into Trading Logic

Add monitoring calls throughout your trading code:

```python
from metrics import get_metrics
from structured_logger import get_trading_logger
from anomaly_detection import check_trading_anomalies

logger = get_trading_logger()
metrics = get_metrics()

# Before trade execution
correlation_id = logger.set_correlation_id()

try:
    # Record API call
    start_time = time.time()
    result = api_call()
    duration = time.time() - start_time

    metrics.record_api_request("place_order", "POST", duration)

    # Check for anomalies
    anomalies = check_trading_anomalies(
        trades_per_minute=current_trades_per_minute,
        api_latency_seconds=duration
    )

    if anomalies.has_anomalies():
        for anomaly in anomalies.get_all_anomalies():
            logger.log_anomaly_detected(anomaly)

    # Log successful trade
    logger.log_trade_entry(
        symbol=trade.symbol,
        side=trade.side,
        quantity=trade.quantity,
        price=trade.price,
        strategy=trade.strategy
    )

except Exception as e:
    metrics.record_api_error("place_order", str(type(e).__name__))
    logger.log_api_error("place_order", str(e))
    raise
```

### Step 3: Set Up Grafana Dashboards

1. **Install Grafana**:
   ```bash
   # Using Docker
   docker run -d -p 3000:3000 --name grafana grafana/grafana

   # Or download from https://grafana.com/get/
   ```

2. **Configure Prometheus Data Source**:
   - Open Grafana at http://localhost:3000 (admin/admin)
   - Add Prometheus data source: http://localhost:9090
   - Test connection

3. **Import Dashboards**:
   ```bash
   # Generate dashboard JSON files
   python grafana_dashboards.py

   # Import via API or UI
   ./import_dashboards.sh
   ```

### Step 4: Set Up Prometheus (Optional)

1. **Install Prometheus**:
   ```bash
   # Using Docker
   docker run -d -p 9090:9090 -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
   ```

2. **Configure prometheus.yml**:
   ```yaml
   global:
     scrape_interval: 15s

   scrape_configs:
     - job_name: 'trading_bot'
       static_configs:
         - targets: ['localhost:8000']  # Metrics endpoint
   ```

### Step 5: Set Up Alert Channels

#### Telegram Bot Setup
1. Message @BotFather on Telegram
2. Create new bot: `/newbot`
3. Get bot token
4. Add bot to your channel/group
5. Get chat ID using `@userinfobot` or API call

#### Discord Webhook Setup
1. Go to Server Settings → Integrations → Webhooks
2. Create new webhook
3. Copy webhook URL

#### Email Setup
1. Use Gmail App Passwords or SMTP service
2. Configure SMTP settings in config

#### PagerDuty Setup
1. Create service integration
2. Get routing key
3. Configure in alerting system

### Step 6: Test Monitoring System

Run the test script to verify everything works:

```bash
python test_monitoring_system.py
```

This will:
- Test metrics collection
- Verify health endpoints
- Send test alerts
- Generate sample reports
- Check anomaly detection

## Monitoring Dashboard Overview

### Trading Performance Dashboard
- Real-time P&L graph
- Win rate trends
- Trading activity heatmap
- Risk gauge cluster
- Trade outcome distribution

### System Health Dashboard
- CPU/Memory usage
- API latency graphs
- WebSocket connection status
- Error rate monitoring
- System resource alerts

### Risk Management Dashboard
- Drawdown monitoring
- Daily loss limits
- Position size controls
- VaR calculations
- Circuit breaker status

### Strategy Analytics Dashboard
- Strategy performance comparison
- Signal generation rates
- Best/worst trading hours
- Indicator performance
- Strategy consistency scores

### API Performance Dashboard
- Request rate monitoring
- Latency percentiles
- Error rate tracking
- Success rate by endpoint
- WebSocket reconnection monitoring

## Alert Configuration

### Critical Alerts (Immediate Action Required)
- Bot crashed/stopped
- WebSocket disconnected >2 minutes
- Position hit stop-loss
- Daily loss limit reached (100%)
- API authentication failed
- Insufficient balance

### Warning Alerts (Review within 30 minutes)
- API latency >5 seconds
- Memory usage >80%
- CPU usage >90%
- Win rate drops below 40%
- Daily loss limit >80%
- Unusual market conditions

### Info Alerts (Daily Digest)
- Daily trading summary
- New balance high
- Strategy performance reports
- System health summaries

## Maintenance Tasks

### Daily
- Review alert history
- Check system resource usage
- Verify backup integrity
- Review trading performance

### Weekly
- Analyze strategy performance
- Review risk metrics
- Update alert thresholds
- Clean old log files

### Monthly
- Generate comprehensive reports
- Review alert effectiveness
- Update monitoring configurations
- Archive old data

## Troubleshooting

### Common Issues

**Metrics not appearing in Grafana**
- Check Prometheus targets status
- Verify metrics endpoint is accessible
- Check metric names match dashboard queries

**Alerts not sending**
- Verify channel configurations
- Check API keys/tokens
- Review SMTP settings
- Check network connectivity

**High memory usage**
- Reduce log retention
- Increase metrics collection intervals
- Review anomaly detection window sizes

**False positive alerts**
- Adjust anomaly detection thresholds
- Review alert deduplication settings
- Fine-tune alert conditions

### Performance Tuning

**Reduce CPU usage**:
- Increase metrics collection intervals
- Disable ML-based anomaly detection
- Reduce log verbosity

**Reduce memory usage**:
- Decrease data retention windows
- Use file-based logging instead of memory buffers
- Reduce concurrent alert processing

**Improve alert responsiveness**:
- Reduce deduplication windows
- Increase alert processing threads
- Use local alert channels first

## Security Considerations

- Store API keys and tokens securely
- Use HTTPS for external communications
- Regularly rotate alert channel credentials
- Monitor for alert system abuse
- Implement rate limiting on alert endpoints
- Use VPN for remote monitoring access

## Backup and Recovery

- Backup configuration files regularly
- Archive alert history and metrics
- Document alert runbook procedures
- Test alert system failover
- Maintain offline alert capabilities

## Support and Resources

- Check logs in `logs/` directory
- Review metrics at `http://localhost:8000`
- Access health status at `http://localhost:8001/health`
- View Grafana dashboards at `http://localhost:3000`

For issues, check:
1. System logs for errors
2. Health endpoint for component status
3. Alert history for recent issues
4. Grafana dashboards for visual indicators
