# MACD Futures Trading Bot

A cryptocurrency futures trading bot featuring MACD Overlay strategy with advanced risk management and trailing stop-loss functionality. **Now supports both Hyperliquid (decentralized) and Bitunix (centralized) exchanges.**

## 🌟 Features

✅ **Dual Exchange Support**: Hyperliquid (DEX) and Bitunix (CEX)  
✅ **MACD Strategy**: Proven technical analysis for entry/exit signals  
✅ **Flexible Leverage**: Up to 50x on Hyperliquid, 125x on Bitunix  
✅ **Advanced Risk Management**: Position sizing, daily loss limits, trade limits  
✅ **Trailing Stop-Loss**: Automatically lock in profits as trades move favorably  
✅ **Dry Run Mode**: Test strategies without risking real funds  
✅ **Comprehensive Logging**: Track all decisions and trades  
✅ **Wallet-Based Authentication**: Secure, non-custodial trading on Hyperliquid  

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Choose Your Exchange

#### **Option A: Hyperliquid (Recommended)**

Decentralized, non-custodial futures trading:

```bash
# Copy example config
cp config/config.example.json config/config.json

# Edit config.json and set:
{
  "exchange": "hyperliquid",
  "private_key": "0xYOUR_PRIVATE_KEY",
  "wallet_address": "0xYOUR_WALLET_ADDRESS",
  "testnet": true
}
```

**Get Started:**
- 📖 **Setup Guide**: See `HYPERLIQUID_SETUP.md`
- 🧪 **Test Connection**: `python3 test_hyperliquid_connection.py`

#### **Option B: Bitunix (Legacy)**

Centralized exchange with API key authentication:

```bash
# Edit config.json and set:
{
  "exchange": "bitunix",
  "api_key": "YOUR_API_KEY",
  "secret_key": "YOUR_SECRET_KEY",
  "testnet": false
}
```

**Get Started:**
- 🧪 **Test Connection**: `python3 test_connection.py`

### 3. Validate Configuration

```bash
python3 config/config_validator.py
```

### 4. Test Connection

```bash
# For Hyperliquid:
python3 test_hyperliquid_connection.py

# For Bitunix:
python3 test_connection.py
```

### 5. Run Bot (Dry Run Mode)

```bash
python3 trading_bot.py
```

Monitor logs:
```bash
tail -f logs/bot.log
```

## 📋 Configuration

### Exchange Settings

```json
{
  "exchange": "hyperliquid",  // or "bitunix"
  
  // For Hyperliquid:
  "private_key": "0x...",
  "wallet_address": "0x...",
  
  // For Bitunix:
  "api_key": "...",
  "secret_key": "...",
  
  "testnet": true
}
```

### Trading Settings

```json
"trading": {
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "check_interval": 300,
  "dry_run": true
}
```

### Strategy Settings (MACD)

```json
"strategy": {
  "fast_length": 12,
  "slow_length": 26,
  "signal_length": 9,
  "risk_reward_ratio": 2.0
}
```

### Risk Management

```json
"risk": {
  "leverage": 10,
  "max_position_size_pct": 0.1,
  "max_daily_loss_pct": 0.05,
  "max_trades_per_day": 10,
  "trailing_stop": {
    "enabled": true,
    "trail_percent": 2.0,
    "activation_percent": 1.0,
    "update_threshold_percent": 0.5
  }
}
```

## 🎯 Trailing Stop-Loss

Intelligent trailing stop-loss automatically adjusts as the market moves in your favor:

- 🔒 **Lock in profits** automatically
- 📈 **Let winners run** while protecting gains
- 🤖 **Works 24/7** without manual intervention
- ⚙️ **Fully configurable** to match your strategy

**Learn More:**
- 📖 **Detailed Guide**: `TRAILING_STOP_GUIDE.md`
- 🧪 **Test Suite**: `python3 test_trailing_stop.py`

## 📁 Project Structure

```
macd-trading-bot/
├── trading_bot.py              # Main bot orchestration
├── hyperliquid_client.py       # Hyperliquid API client
├── bitunix_client.py           # Bitunix API client
├── macd_strategy.py            # MACD strategy logic
├── risk_manager.py             # Risk management + trailing stop
├── config/
│   ├── config.json             # Your configuration (git-ignored)
│   ├── config.example.json     # Template configuration
│   ├── config_validator.py     # Configuration validator
│   └── README_CONFIG.md        # Configuration guide
├── test_hyperliquid_connection.py  # Hyperliquid connection test
├── test_connection.py          # Bitunix connection test
├── test_trailing_stop.py       # Trailing stop test suite
├── HYPERLIQUID_SETUP.md        # Hyperliquid setup guide
├── TRAILING_STOP_GUIDE.md      # Trailing stop documentation
├── TERMINAL_SETUP_GUIDE.md     # Terminal setup guide
├── logs/                       # Bot activity logs
└── .gitignore                  # Protects credentials
```

## 🔒 Security Features

### For Hyperliquid:
- ✅ **Wallet-based authentication** - Private keys never leave your machine
- ✅ **Agent wallet support** - Use dedicated trading wallet (not your main wallet)
- ✅ **Non-custodial** - You maintain full control of funds
- ✅ **Testnet available** - Test with fake USDC first

### For Bitunix:
- ✅ **API key authentication** - Restricted permissions
- ✅ **Never hardcode credentials** - Use config files or env vars
- ✅ **Git protection** - .gitignore prevents credential commits

### General:
- ✅ **Validate all order parameters** before execution
- ✅ **Enforce risk limits strictly** - Daily loss, position size
- ✅ **Comprehensive error handling** - Graceful failure recovery
- ✅ **Dry run mode** - Test without risking real funds
- ✅ **Trailing stop safety** - Never moves against you

## 🧪 Testing

### Configuration Validation
```bash
python3 config/config_validator.py
```

### Connection Tests
```bash
# Hyperliquid
python3 test_hyperliquid_connection.py

# Bitunix
python3 test_connection.py
```

### Trailing Stop Test
```bash
python3 test_trailing_stop.py
```

### Dry Run Trading
Set `"dry_run": true` in config.json:
```bash
python3 trading_bot.py
```

## 📊 Live Trading

⚠️ **Warning**: Only use live trading after thorough testing!

### Pre-Flight Checklist:

1. ✅ **Tested extensively in dry run mode** (24+ hours)
2. ✅ **Validated configuration** (no errors)
3. ✅ **Tested connection successfully**
4. ✅ **Reviewed risk limits** (leverage, position size, daily loss)
5. ✅ **Started with testnet** (Hyperliquid testnet or Bitunix testnet if available)
6. ✅ **Understand the strategy** (MACD, trailing stops)
7. ✅ **Monitoring setup** (log watching, alerts)

### Go Live:

1. Set `"testnet": false` in config.json
2. Set `"dry_run": false` in config.json
3. **Start with very small positions** (1-5% of capital)
4. Run: `python3 trading_bot.py`
5. **Monitor closely** during first 24 hours

## 📝 Logs

All activity is logged to:
- **Console output** - Real-time updates
- **logs/bot.log** - Persistent file logging

Monitor for:
- Trade signals and entries
- Position management
- Trailing stop adjustments
- Risk limit checks
- Errors and warnings

```bash
# Watch logs in real-time
tail -f logs/bot.log

# Search logs
grep "ENTRY SIGNAL" logs/bot.log
grep "ERROR" logs/bot.log
```

## 📦 Requirements

- **Python**: 3.11+
- **Dependencies**:
  - pandas
  - numpy
  - requests
  - eth-account (for Hyperliquid)
  - web3 (for Hyperliquid)
  - eth-utils (for Hyperliquid)

- **Exchange Account**:
  - Hyperliquid wallet with USDC, OR
  - Bitunix account with API access

### Optional: Official Hyperliquid SDK

Hyperliquid provides an [official Python SDK](https://github.com/hyperliquid-dex/hyperliquid-python-sdk) (⭐ 1.2k stars):

```bash
pip install hyperliquid-python-sdk
```

**Note**: This bot uses a custom implementation (`hyperliquid_client.py`) that maintains  
perfect interface compatibility with the Bitunix client, enabling seamless exchange  
switching. The official SDK is available for reference or advanced features.

## 🔐 Security Best Practices

### Hyperliquid:
1. **Use agent wallets** - Create dedicated API wallet (not your main wallet)
2. **Secure private keys** - Never share, never commit to git
3. **Testnet first** - Test with fake USDC before using real funds
4. **Environment variables** - Use env vars for production deployments
5. **Regular audits** - Review bot activity and positions regularly

### Bitunix:
1. **Never commit** `config.json` with real API keys to git
2. **Use environment variables** for production
3. **Restrict API permissions** to trading only (no withdrawals)
4. **Whitelist IPs** if possible
5. **Monitor logs** regularly for unusual activity

### General:
1. **Start small** - Test with minimal position sizes
2. **Scale gradually** - Increase size only after consistent results
3. **Monitor actively** - Especially during first weeks
4. **Keep updated** - Review and update bot regularly
5. **Understand risks** - Crypto trading is highly volatile

## 🆘 Support & Documentation

### Setup Guides:
- 📖 **Hyperliquid Setup**: `HYPERLIQUID_SETUP.md`
- 📖 **Terminal Setup**: `TERMINAL_SETUP_GUIDE.md`
- 📖 **Configuration**: `config/README_CONFIG.md`

### Feature Guides:
- 📖 **Trailing Stops**: `TRAILING_STOP_GUIDE.md`
- 📖 **Migration Status**: `MIGRATION_STATUS.md`

### Technical Docs:
- 📖 **Hyperliquid Client**: `HYPERLIQUID_CLIENT_NOTES.md`
- 📖 **Phase Summaries**: `PHASE*_SUMMARY.md`

### External Resources:
- 🌐 **Hyperliquid Docs**: https://hyperliquid.gitbook.io/hyperliquid-docs
- 🌐 **Hyperliquid Python SDK**: https://github.com/hyperliquid-dex/hyperliquid-python-sdk
- 🌐 **Hyperliquid API Reference**: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api
- 🌐 **Bitunix API Docs**: https://openapidoc.bitunix.com

## 🔄 Switching Exchanges

To switch between Hyperliquid and Bitunix, simply change one field in config.json:

```json
{
  "exchange": "hyperliquid"  // or "bitunix"
}
```

No code changes needed! The bot automatically detects and uses the correct client.

## 🆕 What's New

### v2.0 - Hyperliquid Support
- ✅ Added Hyperliquid DEX support
- ✅ Wallet-based authentication
- ✅ Non-custodial trading
- ✅ Maintained 100% strategy compatibility
- ✅ Comprehensive configuration validation
- ✅ Enhanced documentation

### v1.0 - Trailing Stop-Loss
- ✅ Intelligent trailing stop-loss
- ✅ Automatic profit protection
- ✅ Configurable activation and trail distance
- ✅ Works with both exchanges

## ⚠️ Disclaimer

**This bot is for educational purposes. Cryptocurrency trading carries significant risk. Never trade with money you cannot afford to lose. Past performance does not guarantee future results. Use at your own risk.**

**The developers are not responsible for any financial losses. Trading cryptocurrencies involves substantial risk of loss. You should carefully consider whether trading is appropriate for you in light of your experience, objectives, financial resources, and other relevant circumstances.**

## 📄 License

This project is provided as-is for personal use and learning.

---

## 🚀 Getting Started

**New to the bot?** Follow these steps:

1. **Read** `HYPERLIQUID_SETUP.md` or `TERMINAL_SETUP_GUIDE.md`
2. **Install** dependencies: `pip install -r requirements.txt`
3. **Configure** your settings: `cp config/config.example.json config/config.json`
4. **Validate** config: `python3 config/config_validator.py`
5. **Test** connection: `python3 test_hyperliquid_connection.py`
6. **Run** in dry-run: `python3 trading_bot.py`
7. **Monitor** logs: `tail -f logs/bot.log`

**Happy Trading! 🚀**

*Remember: Risk management is more important than perfect entries.*
