# Credentials Setup Guide

## Your Hyperliquid Credentials

✅ **Verified**: Wallet address matches private key

- **Private Key**: `0x0a4f7d6f4f17aac74bcb3f50eaa1528e97678b48c7c380bd45d7e484b1ae3028`
- **Wallet Address**: `0xd36cd831e00cf42c505a9d705475f74fc8c492ad`
- **Network**: Testnet (recommended for testing)

## Setup Options

### Option 1: Environment Variables (Recommended)

**Quick Setup:**
```bash
export HYPERLIQUID_PRIVATE_KEY=0x0a4f7d6f4f17aac74bcb3f50eaa1528e97678b48c7c380bd45d7e484b1ae3028
export HYPERLIQUID_WALLET_ADDRESS=0xd36cd831e00cf42c505a9d705475f74fc8c492ad
export HYPERLIQUID_TESTNET=true
```

**Persistent Setup (using .env file):**
```bash
# Create .env file (already created for you)
source .env

# Or run the setup script
./setup_credentials.sh
```

### Option 2: System Keyring (Most Secure)

Install keyring for secure credential storage:
```bash
pip install keyring
python manage_credentials.py set hyperliquid_private_key 0x0a4f7d6f4f17aac74bcb3f50eaa1528e97678b48c7c380bd45d7e484b1ae3028
python manage_credentials.py set hyperliquid_wallet_address 0xd36cd831e00cf42c505a9d705475f74fc8c492ad
```

### Option 3: Config File (Less Secure - Testing Only)

⚠️ **Warning**: Only use for testing. Never commit config.json with real credentials!

Edit `config/config.json`:
```json
{
  "exchange": "hyperliquid",
  "private_key": "0x0a4f7d6f4f17aac74bcb3f50eaa1528e97678b48c7c380bd45d7e484b1ae3028",
  "wallet_address": "0xd36cd831e00cf42c505a9d705475f74fc8c492ad",
  "testnet": true
}
```

## Priority Order

The bot loads credentials in this order:
1. **System Keyring** (if keyring installed) - Most secure
2. **Environment Variables** - Good for production
3. **Config File** - Convenient but less secure

## Security Best Practices

✅ **DO:**
- Use environment variables for production
- Use system keyring when available
- Keep `.env` file in `.gitignore`
- Use testnet for development/testing
- Start with `dry_run: true` in config

❌ **DON'T:**
- Commit `.env` or `config.json` with real credentials
- Share private keys
- Use mainnet until thoroughly tested
- Store credentials in code

## Testing Your Setup

```bash
# Test credential loading
python manage_credentials.py test

# Test Hyperliquid connection
python test_hyperliquid_connection.py

# Run bot in dry-run mode
python trading_bot.py
```

## Switching to Mainnet

When ready for mainnet:
1. Change `testnet: false` in config
2. Set `HYPERLIQUID_TESTNET=false` in environment
3. Ensure you have sufficient funds
4. Start with small positions
5. Monitor closely

## Troubleshooting

**Credentials not found:**
- Check environment variables: `echo $HYPERLIQUID_PRIVATE_KEY`
- Verify config.json has correct values
- Run `python manage_credentials.py test`

**Connection errors:**
- Verify testnet/mainnet setting matches your network
- Check wallet has sufficient balance
- Ensure private key and wallet address match
