# Hyperliquid MACD Trading Bot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

A production-ready, enterprise-grade cryptocurrency trading bot for the Hyperliquid exchange featuring advanced MACD strategy, comprehensive risk management, and bulletproof resilience.

## 🚀 Features

### Core Trading Features
- **Advanced MACD Strategy**: Multi-timeframe analysis with 12+ entry filters
- **Real-time Execution**: WebSocket streaming with REST API fallback
- **Risk Management**: Stop-loss, take-profit, trailing stops, position sizing
- **Multi-Asset Support**: BTC, ETH, SOL, and other major cryptocurrencies

### Enterprise Reliability
- **Bulletproof Resilience**: Circuit breakers, automatic recovery, state persistence
- **Comprehensive Monitoring**: Prometheus metrics, Grafana dashboards, alerting
- **Security First**: Encrypted key storage, audit logging, secure API calls
- **Production Ready**: 24/7 operation with enterprise-grade reliability

### Advanced Analytics
- **Performance Analytics**: Sharpe ratio, drawdown analysis, trade analytics
- **Strategy Optimization**: Walk-forward testing, Monte Carlo simulation
- **Anomaly Detection**: Statistical analysis of trading patterns
- **Backtesting Suite**: Historical testing with transaction cost modeling

## 📋 Table of Contents

- [Quick Start](#-quick-start-5-minutes)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Monitoring](#-monitoring)
- [Troubleshooting](#-troubleshooting)
- [FAQ](#-faq)
- [Contributing](#-contributing)
- [License](#-license)

## ⚡ Quick Start (5 Minutes)

### 1. Clone and Install
```bash
git clone https://github.com/yourusername/hyperliquid-macd-bot.git
cd hyperliquid-macd-bot
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
# Set up your Hyperliquid credentials
python manage_credentials.py --setup

# Or manually create config/config.json
cp config/config.example.json config/config.json
# Edit config.json with your API keys
```

### 3. Run the Bot
```bash
# Test connection first
python test_hyperliquid_connection.py

# Run in dry-run mode (recommended)
python trading_bot.py --dry-run

# Run live (with caution!)
python trading_bot.py
```

### 4. Monitor Performance
```bash
# View metrics
curl http://localhost:8000

# View health status
curl http://localhost:8001/health

# Start Grafana dashboards
python grafana_dashboards.py
```

## 📋 Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ / macOS 12+ / Windows 10+
- **Python**: 3.11 or higher
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 10GB for logs and data
- **Network**: Stable internet connection

### Python Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
- `pandas` - Data analysis and manipulation
- `numpy` - Numerical computations
- `websockets` - Real-time WebSocket connections
- `prometheus_client` - Metrics collection
- `flask` - Health check endpoints

### Hyperliquid Account
1. Create account at [Hyperliquid](https://hyperliquid.xyz)
2. Generate API credentials (private key + wallet address)
3. Testnet recommended for initial testing
4. Fund account with USDC for live trading

## 🔧 Installation

### Option 1: Standard Installation

```bash
# Clone repository
git clone https://github.com/yourusername/hyperliquid-macd-bot.git
cd hyperliquid-macd-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Option 2: Docker Installation

```bash
# Build Docker image
docker build -t hyperliquid-bot .

# Run container
docker run -d \
  --name hyperliquid-bot \
  -p 8000:8000 \
  -p 8001:8001 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  hyperliquid-bot
```

### Option 3: Production Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete production setup including:
- Ubuntu server configuration
- SystemD service setup
- Security hardening
- Monitoring integration

## ⚙️ Configuration

### Basic Configuration

Create `config/config.json`:

```json
{
  "exchange": "hyperliquid",
  "private_key": "0x...",
  "wallet_address": "0x...",
  "testnet": true,

  "trading": {
    "symbol": "BTCUSDT",
    "timeframe": "5m",
    "check_interval": 300,
    "dry_run": true
  },

  "strategy": {
    "enhanced_strategy": true,
    "fast_length": 12,
    "slow_length": 26,
    "signal_length": 9,
    "risk_reward_ratio": 2.0,
    "require_volume_confirmation": true,
    "use_atr_filter": true,
    "use_market_regime_filter": true
  },

  "risk": {
    "leverage": 5,
    "max_position_size_pct": 0.05,
    "max_daily_loss_pct": 0.03,
    "trailing_stop": {
      "enabled": true,
      "trail_percent": 2.0
    }
  },

  "monitoring": {
    "enabled": true,
    "metrics_port": 8000,
    "health_port": 8001,
    "structured_logging": true
  }
}
```

### Advanced Configuration

For production use, enable all resilience features:

```json
{
  "resilience": {
    "enabled": true,
    "state_db_path": "data/bot_state.db",
    "circuit_breaker_failure_threshold": 5,
    "max_retries_per_hour": 100,
    "heartbeat_interval_seconds": 60
  },

  "alerting": {
    "channels": {
      "telegram": {
        "enabled": true,
        "bot_token": "your-bot-token",
        "chat_ids": ["123456789"]
      },
      "email": {
        "enabled": true,
        "smtp_server": "smtp.gmail.com",
        "to_emails": ["alerts@yourdomain.com"]
      }
    }
  },

  "security": {
    "encrypted_keys": true,
    "audit_logging": true,
    "max_login_attempts": 3
  }
}
```

### Environment Variables

Override configuration with environment variables:

```bash
export HYPERLIQUID_PRIVATE_KEY="0x..."
export HYPERLIQUID_WALLET_ADDRESS="0x..."
export BOT_DRY_RUN="true"
export BOT_EMERGENCY_SHUTDOWN="false"
```

## 🎯 Usage

### Command Line Options

```bash
python trading_bot.py [OPTIONS]

Options:
  --dry-run          Run in simulation mode (recommended for testing)
  --symbol SYMBOL    Trading symbol (default: BTCUSDT)
  --config FILE      Configuration file path
  --log-level LEVEL  Logging level (DEBUG, INFO, WARNING, ERROR)
  --help            Show help message
```

### Running Modes

#### Dry Run Mode (Recommended for Testing)
```bash
python trading_bot.py --dry-run
```
- Simulates all trades without real money
- Full logging and metrics collection
- Safe for strategy testing and validation

#### Live Trading Mode
```bash
python trading_bot.py
```
- Executes real trades on Hyperliquid
- Requires funded account
- Start with small position sizes

#### Backtesting Mode
```bash
python run_backtest.py --strategy enhanced --symbol BTCUSDT --start-date 2024-01-01
```
- Test strategy on historical data
- Generate performance reports
- Optimize parameters

### Monitoring Commands

```bash
# View real-time metrics
curl http://localhost:8000

# Check bot health
curl http://localhost:8001/health

# View recent logs
tail -f logs/trading_bot.log

# Start Grafana dashboards
python grafana_dashboards.py

# Run health checks
python test_hyperliquid_connection.py
```

## 📊 Monitoring

### Built-in Monitoring
- **Prometheus Metrics**: Real-time performance metrics on port 8000
- **Health Endpoints**: System health checks on port 8001
- **Grafana Dashboards**: Pre-built visualization templates
- **Alert System**: Multi-channel notifications (Telegram, Email, Discord)

### Key Metrics to Monitor
- **Trading Performance**: Win rate, profit factor, Sharpe ratio
- **System Health**: CPU usage, memory usage, API latency
- **Risk Metrics**: Drawdown, position size, daily P&L
- **Resilience Status**: Circuit breaker state, retry counts

### Alert Types
- **Critical**: Bot crashes, API authentication fails, position losses
- **Warning**: High latency, memory usage, strategy performance drops
- **Info**: Daily summaries, new high balances, strategy signals

## 🔧 Troubleshooting

### Common Issues

#### "Connection failed" Errors
```bash
# Test network connectivity
ping api.hyperliquid.xyz

# Test API access
python test_hyperliquid_connection.py

# Check proxy/firewall settings
curl -v https://api.hyperliquid.xyz/info
```

#### High Memory Usage
```bash
# Check memory usage
ps aux --sort=-%mem | head -10

# Clear caches and restart
rm -rf __pycache__/
python trading_bot.py
```

#### WebSocket Disconnection
```bash
# Check WebSocket connectivity
python -c "
import websockets
async def test():
    async with websockets.connect('wss://api.hyperliquid.xyz/ws') as ws:
        print('WebSocket connected')
asyncio.run(test())
"
```

#### Strategy Not Trading
```bash
# Check market conditions
curl http://localhost:8001/health | jq '.market_condition'

# Review strategy logs
grep "strategy" logs/trading_bot.log | tail -20

# Test strategy manually
python -c "
from macd_strategy_enhanced import EnhancedMACDStrategy
strategy = EnhancedMACDStrategy()
print('Strategy initialized successfully')
"
```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export LOG_LEVEL=DEBUG
```

### Emergency Shutdown

If the bot behaves unexpectedly:

```bash
# Graceful shutdown
export BOT_EMERGENCY_SHUTDOWN=true

# Force kill (last resort)
pkill -f trading_bot.py
```

## ❓ FAQ

### General Questions

**Q: Is this bot profitable?**
A: Past performance doesn't guarantee future results. This bot implements a proven MACD strategy but requires proper risk management and market conditions.

**Q: Can I run multiple instances?**
A: Yes, but configure different symbols or use different accounts to avoid conflicts.

**Q: What exchanges are supported?**
A: Currently supports Hyperliquid. The architecture is designed for easy extension to other exchanges.

**Q: How much capital do I need?**
A: Minimum $100-500 for testing. Production trading requires $1000+ depending on position sizing strategy.

### Technical Questions

**Q: Can I modify the strategy?**
A: Yes, the strategy is fully configurable. See [STRATEGY.md](STRATEGY.md) for customization options.

**Q: How do I add new indicators?**
A: Extend the `EnhancedMACDStrategy` class and add indicator calculations in the `calculate_indicators` method.

**Q: What happens if the bot crashes?**
A: The resilience system automatically recovers state and continues trading. See [ARCHITECTURE.md](ARCHITECTURE.md) for details.

**Q: How secure is the bot?**
A: Implements enterprise security practices including encrypted key storage, audit logging, and secure API calls. See [SECURITY.md](SECURITY.md).

### Performance Questions

**Q: What's the expected win rate?**
A: The enhanced strategy targets 55-65% win rate with 2:1 risk-reward ratio, depending on market conditions.

**Q: How much drawdown should I expect?**
A: Maximum drawdown is typically 10-15% with proper position sizing. The bot includes automatic risk controls.

**Q: Can I trade 24/7?**
A: Yes, the bot is designed for continuous operation with built-in resilience and monitoring.

### Support Questions

**Q: Where can I get help?**
A: Check the documentation first, then create an issue on GitHub with detailed logs.

**Q: Can I contribute to the project?**
A: Yes! See [Contributing](#contributing) section below.

**Q: Is commercial support available?**
A: Contact the maintainers for enterprise support options.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

### Development Setup
```bash
git clone https://github.com/yourusername/hyperliquid-macd-bot.git
cd hyperliquid-macd-bot
pip install -r requirements-dev.txt
pre-commit install
```

### Testing
```bash
# Run all tests
python run_tests.py

# Run specific test suite
python test_strategy_enhanced.py
python test_recovery_scenarios.py
python test_monitoring_system.py
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all functions
- Add comprehensive docstrings
- Write tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.

### Risk Warnings
- Never trade with money you cannot afford to lose
- Start with small position sizes
- Test strategies thoroughly before live trading
- Monitor positions closely
- Have emergency stop procedures ready

### No Financial Advice
This software does not constitute financial advice. The authors are not responsible for any financial losses incurred through the use of this software.

---

## 📚 Additional Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and component overview
- **[STRATEGY.md](STRATEGY.md)** - Detailed strategy explanation and optimization
- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment guide
- **[SECURITY.md](SECURITY.md)** - Security practices and procedures
- **[OPERATIONS.md](OPERATIONS.md)** - Daily operations and maintenance
- **[MONITORING_SETUP_GUIDE.md](MONITORING_SETUP_GUIDE.md)** - Monitoring and alerting setup
- **[STRATEGY_ENHANCEMENTS_GUIDE.md](STRATEGY_ENHANCEMENTS_GUIDE.md)** - Advanced strategy features
- **[RESILIENCE_SETUP_GUIDE.md](RESILIENCE_SETUP_GUIDE.md)** - Resilience and recovery setup

---

**Happy Trading! 🚀📈**

For questions or support, please create an issue on GitHub.