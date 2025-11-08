# Configuration Guide

## Quick Setup

### 1. Create Configuration File

```bash
cd config
cp config.example.json config.json
```

### 2. For Hyperliquid (Recommended)

Edit `config.json`:

```json
{
  "exchange": "hyperliquid",
  "private_key": "0xYOUR_PRIVATE_KEY_HERE",
  "wallet_address": "0xYOUR_WALLET_ADDRESS_HERE",
  "testnet": true,
  ...
}
```

**Get Your Credentials:**
1. Visit https://app.hyperliquid-testnet.xyz (testnet) or https://app.hyperliquid.xyz (mainnet)
2. Connect your wallet
3. Go to API section
4. Create an "Agent Wallet" (API wallet)
5. Copy the private key and wallet address

### 3. For Bitunix (Legacy)

Edit `config.json`:

```json
{
  "exchange": "bitunix",
  "api_key": "YOUR_API_KEY",
  "secret_key": "YOUR_SECRET_KEY",
  "testnet": false,
  ...
}
```

## Environment Variables (Production Recommended)

Instead of storing credentials in `config.json`, use environment variables:

### Linux/Mac:

```bash
export HYPERLIQUID_PRIVATE_KEY="0x..."
export HYPERLIQUID_WALLET_ADDRESS="0x..."
```

### Windows (PowerShell):

```powershell
$env:HYPERLIQUID_PRIVATE_KEY="0x..."
$env:HYPERLIQUID_WALLET_ADDRESS="0x..."
```

### Docker/Docker Compose:

```yaml
environment:
  - HYPERLIQUID_PRIVATE_KEY=0x...
  - HYPERLIQUID_WALLET_ADDRESS=0x...
```

### .env File (Alternative):

Create `.env` file in project root:

```bash
HYPERLIQUID_PRIVATE_KEY=0x...
HYPERLIQUID_WALLET_ADDRESS=0x...
```

**Note**: `.env` files are ignored by git automatically

## Configuration Parameters

### Exchange Settings

| Parameter | Type | Description |
|-----------|------|-------------|
| `exchange` | string | "hyperliquid" or "bitunix" |
| `private_key` | string | Ethereum private key (Hyperliquid only) |
| `wallet_address` | string | Ethereum wallet address (Hyperliquid only) |
| `api_key` | string | API key (Bitunix only) |
| `secret_key` | string | Secret key (Bitunix only) |
| `testnet` | boolean | Use testnet (true) or mainnet (false) |

### Trading Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | string | "BTCUSDT" | Trading pair |
| `timeframe` | string | "1h" | Candle interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w) |
| `check_interval` | number | 300 | Seconds between checks (300 = 5 min) |
| `dry_run` | boolean | true | Simulate trading without real orders |

### Strategy Settings (MACD)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fast_length` | number | 12 | Fast EMA period |
| `slow_length` | number | 26 | Slow EMA period |
| `signal_length` | number | 9 | Signal line period |
| `risk_reward_ratio` | number | 2.0 | Target R:R ratio |

### Risk Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `leverage` | number | 10 | Trading leverage (1-50 for HL, 1-125 for Bitunix) |
| `max_position_size_pct` | number | 0.1 | Max position as % of equity (0.1 = 10%) |
| `max_daily_loss_pct` | number | 0.05 | Max daily loss as % of equity (0.05 = 5%) |
| `max_trades_per_day` | number | 10 | Maximum trades per day |

### Trailing Stop Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | true | Enable trailing stop-loss |
| `trail_percent` | number | 2.0 | Distance to trail behind best price (%) |
| `activation_percent` | number | 1.0 | Profit threshold to activate trailing (%) |
| `update_threshold_percent` | number | 0.5 | Minimum movement to update stop (%) |

## Validation

Validate your configuration before running the bot:

```bash
python config/config_validator.py
```

Or validate a specific file:

```bash
python config/config_validator.py config/my_config.json
```

### Common Validation Errors:

**"Invalid private_key format"**
- Ensure key starts with `0x` and is 66 characters total
- Format: `0x` + 64 hex characters

**"Invalid wallet_address format"**
- Must start with `0x` and be 42 characters total
- Use checksummed address for best results

**"fast_length must be less than slow_length"**
- MACD requires fast EMA < slow EMA
- Typical: fast=12, slow=26

## Security Best Practices

### ✅ DO:
- ✅ Use `.env` files or environment variables for production
- ✅ Set proper file permissions: `chmod 600 config/config.json`
- ✅ Start with `testnet=true` for development
- ✅ Use `dry_run=true` to test strategies
- ✅ Keep private keys secure and never share them
- ✅ Use agent wallets (not your main wallet) for Hyperliquid

### ❌ DON'T:
- ❌ Commit `config.json` with real credentials to git
- ❌ Share your private keys or API keys
- ❌ Use mainnet without thorough testing
- ❌ Start with high leverage (use 5-10x max initially)
- ❌ Store credentials in plain text on shared systems

## Migration from Bitunix to Hyperliquid

### Step 1: Backup Current Config

```bash
cp config/config.json config/config.bitunix.backup.json
```

### Step 2: Update Exchange Type

Change in `config.json`:

```json
{
  "exchange": "hyperliquid",  // Changed from "bitunix"
  ...
}
```

### Step 3: Replace Credentials

Remove:
```json
"api_key": "...",
"secret_key": "...",
```

Add:
```json
"private_key": "0x...",
"wallet_address": "0x...",
"testnet": true,
```

### Step 4: Validate

```bash
python config/config_validator.py
```

### Step 5: Test

Run bot in dry-run mode first:

```json
"dry_run": true
```

## Troubleshooting

### "Configuration file not found"
- Ensure `config/config.json` exists
- Copy from `config.example.json` if needed

### "Invalid JSON"
- Check for missing commas
- Remove trailing commas
- Validate JSON syntax at https://jsonlint.com

### "Placeholder credentials detected"
- Replace example values with your actual credentials
- Private key should not be all zeros

### "Permission denied"
- On Linux/Mac: `chmod 600 config/config.json`
- Ensure you have read access to the file

### Environment variables not loaded
- Ensure variables are exported in current shell
- Restart terminal/IDE after setting variables
- Check variable names match exactly (case-sensitive)

## Example Configurations

### Conservative (Recommended for Beginners)

```json
{
  "exchange": "hyperliquid",
  "testnet": true,
  "trading": {
    "dry_run": true,
    "check_interval": 300
  },
  "risk": {
    "leverage": 5,
    "max_position_size_pct": 0.05,
    "max_daily_loss_pct": 0.02,
    "max_trades_per_day": 5
  }
}
```

### Moderate (Experienced Traders)

```json
{
  "exchange": "hyperliquid",
  "testnet": false,
  "trading": {
    "dry_run": false,
    "check_interval": 300
  },
  "risk": {
    "leverage": 10,
    "max_position_size_pct": 0.1,
    "max_daily_loss_pct": 0.05,
    "max_trades_per_day": 10
  }
}
```

### Aggressive (High Risk)

```json
{
  "exchange": "hyperliquid",
  "testnet": false,
  "trading": {
    "dry_run": false,
    "check_interval": 180
  },
  "risk": {
    "leverage": 20,
    "max_position_size_pct": 0.15,
    "max_daily_loss_pct": 0.08,
    "max_trades_per_day": 20
  }
}
```

⚠️ **Warning**: Aggressive settings significantly increase risk!

## Support

For issues with configuration:

1. Run validator: `python config/config_validator.py`
2. Check logs: `tail -f logs/bot.log`
3. Review documentation: `README.md`
4. Test connection: `python test_connection.py`

---

**Remember**: Always test with `dry_run=true` and `testnet=true` before risking real funds!

