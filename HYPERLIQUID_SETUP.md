# Hyperliquid Setup Guide

Complete guide to setting up and running the MACD trading bot on Hyperliquid's decentralized perpetual futures exchange.

═══════════════════════════════════════════════════════════════

## 📋 Table of Contents

1. [What is Hyperliquid?](#what-is-hyperliquid)
2. [Prerequisites](#prerequisites)
3. [Getting Started](#getting-started)
4. [Creating Your Wallet](#creating-your-wallet)
5. [Getting Testnet Tokens](#getting-testnet-tokens)
6. [Creating an Agent Wallet](#creating-an-agent-wallet)
7. [Exporting Private Keys Safely](#exporting-private-keys-safely)
8. [Bot Configuration](#bot-configuration)
9. [Security Best Practices](#security-best-practices)
10. [Testnet Testing Guide](#testnet-testing-guide)
11. [Going Live Checklist](#going-live-checklist)
12. [Troubleshooting](#troubleshooting)

═══════════════════════════════════════════════════════════════

## What is Hyperliquid?

**Hyperliquid** is a decentralized perpetual futures exchange built on its own Layer 1 blockchain:

### Key Features:
- 🔓 **Non-custodial**: You maintain full control of your funds
- ⚡ **High performance**: 200,000+ orders per second
- 💰 **Low fees**: Competitive maker/taker fees
- 🔐 **Secure**: Wallet-based authentication (no API keys)
- 🧪 **Testnet available**: Practice with fake USDC

### Advantages for Bot Trading:
- ✅ No KYC required
- ✅ Wallet-based authentication (more secure than API keys)
- ✅ Full on-chain transparency
- ✅ Great testnet for development
- ✅ Up to 50x leverage

═══════════════════════════════════════════════════════════════

## Prerequisites

### Required:
1. **Ethereum wallet** (MetaMask, Rainbow, etc.)
2. **USDC on Arbitrum** (for mainnet) or test USDC (for testnet)
3. **Python 3.11+** installed
4. **Basic terminal knowledge**

### Recommended:
- Familiarity with cryptocurrency wallets
- Understanding of perpetual futures trading
- Some Python/coding experience (helpful but not required)

═══════════════════════════════════════════════════════════════

## Getting Started

### Step 1: Install Bot Dependencies

```bash
# Navigate to bot directory
cd /home/ink/bitunix-macd-bot

# Install Python packages
pip install -r requirements.txt

# Verify installation
python3 -c "import eth_account; print('✅ Dependencies installed')"
```

### Step 2: Choose Your Environment

#### **Testnet (Recommended for beginners)**
- ✅ Free test USDC
- ✅ No real money at risk
- ✅ Same features as mainnet
- ✅ Perfect for learning

#### **Mainnet (For experienced traders)**
- ⚠️ Real money at risk
- ⚠️ Requires actual USDC
- ⚠️ Test on testnet first!

═══════════════════════════════════════════════════════════════

## Creating Your Wallet

### Option A: Using MetaMask (Recommended)

1. **Install MetaMask**:
   - Visit https://metamask.io
   - Install browser extension
   - Follow setup wizard

2. **Create New Wallet**:
   - Click "Create a Wallet"
   - Set strong password
   - **CRITICAL**: Write down seed phrase on paper
   - Store seed phrase securely (NOT digitally)

3. **Add Arbitrum Network**:
   - Open MetaMask
   - Click network dropdown
   - Select "Add Network"
   - Choose "Arbitrum One"

### Option B: Using Rainbow Wallet

1. **Download Rainbow**:
   - iOS: App Store
   - Android: Google Play
   - Visit https://rainbow.me

2. **Create Wallet**:
   - Open app
   - "Create a new wallet"
   - **Backup seed phrase securely**

3. **Connect to Arbitrum**:
   - Rainbow auto-detects networks
   - Switch to Arbitrum when needed

### Option C: Hardware Wallet (Most Secure)

Use Ledger or Trezor for maximum security:
- Follow manufacturer's setup guide
- Never expose seed phrase digitally
- Use with MetaMask integration

═══════════════════════════════════════════════════════════════

## Getting Testnet Tokens

### For Hyperliquid Testnet:

#### Step 1: Access Testnet
```
Visit: https://app.hyperliquid-testnet.xyz
```

#### Step 2: Connect Wallet
1. Click "Connect Wallet"
2. Choose your wallet (MetaMask, etc.)
3. Approve connection
4. Wallet should show connected

#### Step 3: Get Test USDC
1. Look for "Faucet" or "Get Test USDC"
2. Click to request tokens
3. Wait for transaction confirmation
4. Check balance (should show test USDC)

**Troubleshooting:**
- If no faucet visible, join Hyperliquid Discord
- Ask for testnet tokens in #testnet channel
- Provide your wallet address
- Usually receive tokens within hours

#### Step 4: Verify Balance
```bash
# After connecting bot:
python3 test_hyperliquid_connection.py

# Should show your test USDC balance
```

═══════════════════════════════════════════════════════════════

## Creating an Agent Wallet

**Agent wallets** are sub-wallets authorized by your main account for API/bot trading. They can trade but **cannot withdraw funds** - this adds security!

### Why Use Agent Wallets?

✅ **Security**: Even if compromised, attacker can't steal funds  
✅ **Isolation**: Separate bot trading from manual trading  
✅ **Revocable**: Can disable anytime from main account  
✅ **Best Practice**: Industry standard for automated trading  

### Step-by-Step Creation:

#### 1. Access Hyperliquid
```
Testnet: https://app.hyperliquid-testnet.xyz
Mainnet: https://app.hyperliquid.xyz
```

#### 2. Connect Main Wallet
- Click "Connect Wallet"
- Approve connection in MetaMask

#### 3. Navigate to API Settings
- Look for "API" or "Settings" menu
- Find "Agent Wallets" or "API Keys" section

#### 4. Create New Agent Wallet
1. Click "Create Agent Wallet" or "Add API Wallet"
2. System generates new wallet address
3. **IMPORTANT**: Save the private key shown (one-time only!)
4. Confirm creation
5. Authorize agent wallet from main account

#### 5. Verify Agent Wallet
- Should see agent wallet address listed
- Status should show "Active" or "Authorized"
- Can revoke access anytime

### Manual Agent Wallet Creation:

If Hyperliquid doesn't have built-in agent wallet creation:

```python
# Run this Python script to generate a wallet:
from eth_account import Account
import secrets

# Generate new wallet
private_key = "0x" + secrets.token_hex(32)
account = Account.from_key(private_key)

print("═══════════════════════════════════════")
print("AGENT WALLET GENERATED")
print("═══════════════════════════════════════")
print(f"Address: {account.address}")
print(f"Private Key: {private_key}")
print("═══════════════════════════════════════")
print("⚠️  SAVE THESE SECURELY!")
print("⚠️  NEVER SHARE WITH ANYONE!")
print("═══════════════════════════════════════")

# Save to secure location
# THEN authorize this address in Hyperliquid UI
```

**After generating**:
1. Save private key securely
2. Go to Hyperliquid UI
3. Find "Approve Agent" option
4. Enter the agent wallet address
5. Confirm authorization transaction

═══════════════════════════════════════════════════════════════

## Exporting Private Keys Safely

⚠️ **CRITICAL SECURITY SECTION** ⚠️

### Exporting from MetaMask:

#### Step 1: Access Account
1. Open MetaMask
2. Click account icon (top right)
3. Select account to export

#### Step 2: Export Private Key
1. Click three dots (⋮)
2. Select "Account Details"
3. Click "Export Private Key"
4. Enter MetaMask password
5. **Private key is displayed**

#### Step 3: Save Securely
```bash
# Create secure config file:
cd /home/ink/bitunix-macd-bot/config
nano config.json

# Add (replace with your values):
{
  "exchange": "hyperliquid",
  "private_key": "0x...",  # Paste here
  "wallet_address": "0x...",  # Your agent wallet address
  "testnet": true
}

# Save and exit (Ctrl+X, Y, Enter)

# Secure the file:
chmod 600 config.json

# Verify it's protected:
ls -l config.json
# Should show: -rw------- (only you can read/write)
```

### Security Best Practices:

#### ✅ DO:
- ✅ Use agent wallets (not main wallet)
- ✅ Store private keys in encrypted files
- ✅ Use `chmod 600` on config files
- ✅ Keep backups in secure locations (encrypted USB, password manager)
- ✅ Use environment variables for production
- ✅ Test on testnet first
- ✅ Rotate keys regularly

#### ❌ DON'T:
- ❌ Never commit private keys to git
- ❌ Never email private keys
- ❌ Never share private keys in Discord/Telegram
- ❌ Never store in Google Docs/Dropbox unencrypted
- ❌ Never screenshot private keys
- ❌ Never use main wallet for bot trading
- ❌ Never skip testnet testing

### Using Environment Variables (Production):

```bash
# Create .env file (git-ignored):
nano .env

# Add:
HYPERLIQUID_PRIVATE_KEY=0x...
HYPERLIQUID_WALLET_ADDRESS=0x...

# Secure it:
chmod 600 .env

# Bot will auto-load from env vars
```

═══════════════════════════════════════════════════════════════

## Bot Configuration

### Step 1: Copy Example Config

```bash
cd /home/ink/bitunix-macd-bot
cp config/config.example.json config/config.json
```

### Step 2: Edit Configuration

```bash
nano config/config.json
```

### Step 3: Hyperliquid Settings

```json
{
  "exchange": "hyperliquid",
  
  "private_key": "0xYOUR_AGENT_WALLET_PRIVATE_KEY",
  "wallet_address": "0xYOUR_AGENT_WALLET_ADDRESS",
  "testnet": true,
  
  "trading": {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "check_interval": 300,
    "dry_run": true
  },
  
  "strategy": {
    "fast_length": 12,
    "slow_length": 26,
    "signal_length": 9,
    "risk_reward_ratio": 2.0
  },
  
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
}
```

### Step 4: Validate Configuration

```bash
python3 config/config_validator.py
```

Expected output:
```
✅ Configuration is valid!
```

═══════════════════════════════════════════════════════════════

## Security Best Practices

### File Security:

```bash
# Secure config file (Unix/Linux/Mac):
chmod 600 config/config.json

# Verify .gitignore is working:
git status
# config.json should NOT appear

# Check file permissions:
ls -la config/
```

### Key Management:

1. **Separate Wallets**:
   - Main wallet: Store funds
   - Agent wallet: Bot trading only
   - Never use same wallet for both

2. **Regular Audits**:
   ```bash
   # Check bot activity:
   tail -100 logs/bot.log
   
   # Check positions on Hyperliquid UI
   # Verify trades match logs
   ```

3. **Set Limits**:
   - Start with low leverage (5-10x)
   - Small position sizes (5-10% of capital)
   - Strict daily loss limits

4. **Monitor Actively**:
   - First 24 hours: Check every hour
   - First week: Check every 4-6 hours
   - After that: Check daily minimum

### Network Security:

- ✅ Use secure WiFi (not public)
- ✅ VPN recommended
- ✅ Keep system updated
- ✅ Antivirus/firewall active
- ✅ 2FA on all accounts

═══════════════════════════════════════════════════════════════

## Testnet Testing Guide

### Phase 1: Connection Testing (Day 1)

```bash
# Test 1: Validate config
python3 config/config_validator.py

# Test 2: Connection test
python3 test_hyperliquid_connection.py

# Test 3: Check balance
# Should show test USDC balance
```

### Phase 2: Dry Run Testing (Days 2-3)

```bash
# Ensure settings:
{
  "testnet": true,
  "dry_run": true  # No real orders even on testnet
}

# Run bot:
python3 trading_bot.py

# Watch logs:
tail -f logs/bot.log

# Look for:
# - ✅ MACD calculations
# - ✅ Signal detection
# - ✅ Position sizing
# - ✅ Risk checks
# - ✅ Trailing stop updates
```

### Phase 3: Testnet Live Trading (Days 4-7)

```bash
# Update config:
{
  "testnet": true,
  "dry_run": false  # Real orders on testnet
}

# Start with small leverage:
"leverage": 5

# Run bot:
python3 trading_bot.py

# Monitor closely:
# - Check Hyperliquid UI for actual orders
# - Verify positions match logs
# - Test trailing stops work
# - Confirm risk limits enforced
```

### Phase 4: Analysis (Day 8)

Review results:
- Total trades executed
- Win rate
- Average profit/loss
- Risk limits respected?
- Any errors or issues?
- Strategy performance

### Testing Checklist:

- [ ] Configuration validated
- [ ] Connection test passed
- [ ] Balance shows correctly
- [ ] Dry run mode works
- [ ] Entry signals detected
- [ ] Orders placed successfully
- [ ] Positions tracked correctly
- [ ] Trailing stops update
- [ ] Take profit hit
- [ ] Stop loss hit
- [ ] Daily loss limit enforced
- [ ] Trade limit enforced
- [ ] No unexpected errors
- [ ] Logs are clear

═══════════════════════════════════════════════════════════════

## Going Live Checklist

⚠️ **Only proceed after successful testnet testing!**

### Pre-Flight Checklist:

#### Configuration:
- [ ] Tested on testnet for 7+ days
- [ ] All features verified working
- [ ] Configuration validated (no errors)
- [ ] Using agent wallet (not main wallet)
- [ ] Private keys secured properly
- [ ] Leverage set appropriately (5-10x recommended)
- [ ] Position size limits conservative (5-10%)
- [ ] Daily loss limits set (2-5%)
- [ ] Trailing stops enabled and tested

#### Security:
- [ ] Config file permissions set (`chmod 600`)
- [ ] .gitignore working (config.json not in git)
- [ ] Backups of config stored securely
- [ ] 2FA enabled on all accounts
- [ ] Using secure network
- [ ] VPN active (recommended)

#### Risk Management:
- [ ] Understand strategy fully
- [ ] Comfortable with maximum loss
- [ ] Start amount decided (1-5% of total capital)
- [ ] Emergency stop plan ready
- [ ] Monitoring schedule planned

#### Technical:
- [ ] Server/computer stable
- [ ] Internet connection reliable
- [ ] System updates applied
- [ ] Logs directory writable
- [ ] Enough disk space
- [ ] Process monitoring setup (optional)

### Going Live Steps:

#### Step 1: Update Configuration

```json
{
  "exchange": "hyperliquid",
  "testnet": false,  // ← CHANGE THIS
  "trading": {
    "dry_run": false  // ← AND THIS
  },
  "risk": {
    "leverage": 5,  // Start conservative
    "max_position_size_pct": 0.05  // 5% of capital
  }
}
```

#### Step 2: Final Validation

```bash
python3 config/config_validator.py
python3 test_hyperliquid_connection.py
```

#### Step 3: Start Bot

```bash
# In a tmux/screen session (recommended):
tmux new -s trading-bot
python3 trading_bot.py

# Or directly:
python3 trading_bot.py

# Monitor logs:
tail -f logs/bot.log
```

#### Step 4: First 24 Hours

- ✅ Check logs every hour
- ✅ Verify trades on Hyperliquid UI
- ✅ Monitor position sizes
- ✅ Watch trailing stops
- ✅ Confirm risk limits work

#### Step 5: First Week

- ✅ Check logs every 4-6 hours
- ✅ Review daily P&L
- ✅ Adjust parameters if needed
- ✅ Document any issues

#### Step 6: Scaling Up

After 1-2 weeks of successful operation:
- Gradually increase position size
- Consider increasing leverage (cautiously)
- Monitor performance metrics
- Adjust strategy parameters based on results

═══════════════════════════════════════════════════════════════

## Troubleshooting

### Common Issues:

#### "Configuration validation failed"
```bash
# Check syntax:
python3 -m json.tool config/config.json

# Verify required fields:
python3 config/config_validator.py
```

#### "Invalid private key format"
```
Error: Must be 0x followed by 64 hex characters

Fix:
- Ensure private key starts with 0x
- Check length (should be 66 characters total)
- No spaces or special characters
```

#### "Connection test failed"
```bash
# Check internet connection
ping api.hyperliquid.xyz

# Verify testnet vs mainnet setting matches your wallet
# Testnet: Must have test USDC
# Mainnet: Must have real USDC

# Check wallet address is correct
# Check private key is for the agent wallet
```

#### "Low balance warning"
```
Need test USDC on testnet:
1. Visit https://app.hyperliquid-testnet.xyz
2. Use faucet or ask in Discord
3. Wait for tokens
4. Try again
```

#### "Trailing stop not working"
```json
// Verify enabled in config:
"trailing_stop": {
  "enabled": true,  // Must be true
  "trail_percent": 2.0
}

// Check logs for activation:
grep "Trailing stop ACTIVATED" logs/bot.log
```

#### "Orders not executing"
```
Check:
1. dry_run is false (if you want real orders)
2. Sufficient balance
3. Leverage is set
4. Risk limits not hit
5. Check Hyperliquid UI for orders
```

### Getting Help:

1. **Check Logs**:
   ```bash
   tail -100 logs/bot.log
   grep ERROR logs/bot.log
   ```

2. **Review Documentation**:
   - README.md
   - HYPERLIQUID_CLIENT_NOTES.md
   - config/README_CONFIG.md

3. **Community Support**:
   - Hyperliquid Discord
   - GitHub Issues (if public repo)

═══════════════════════════════════════════════════════════════

## Additional Resources

### Official Hyperliquid Documentation:
- 🌐 **Main Docs**: https://hyperliquid.gitbook.io/hyperliquid-docs
- 🌐 **API Reference**: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api
- 🌐 **Exchange Endpoint**: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/exchange-endpoint
- 🌐 **Info Endpoint**: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint
- 🐍 **Official Python SDK**: https://github.com/hyperliquid-dex/hyperliquid-python-sdk (⭐ 1.2k)

### Hyperliquid Applications:
- 🌐 **Mainnet**: https://app.hyperliquid.xyz
- 🧪 **Testnet**: https://app.hyperliquid-testnet.xyz

### This Bot's Documentation:
- 📖 **Configuration Guide**: `config/README_CONFIG.md`
- 📖 **Trailing Stops**: `TRAILING_STOP_GUIDE.md`
- 📖 **Terminal Setup**: `TERMINAL_SETUP_GUIDE.md`
- 📖 **Client Implementation**: `HYPERLIQUID_CLIENT_NOTES.md`

### Security Resources:
- 🔐 **MetaMask Security**: https://metamask.io/security/
- 🔐 **Wallet Best Practices**: Research hardware wallets
- 🔐 **Hyperliquid Security**: https://hyperliquid.gitbook.io/hyperliquid-docs/risks

═══════════════════════════════════════════════════════════════

## Summary

**You've learned:**
- ✅ What Hyperliquid is and why use it
- ✅ How to create and secure wallets
- ✅ How to get testnet tokens
- ✅ How to create agent wallets
- ✅ How to safely export private keys
- ✅ How to configure the bot
- ✅ Security best practices
- ✅ Complete testing workflow
- ✅ Going live checklist
- ✅ Troubleshooting common issues

**Next Steps:**
1. Create your agent wallet
2. Get testnet tokens
3. Configure the bot
4. Run connection test
5. Start testing!

═══════════════════════════════════════════════════════════════

**Good luck with your trading bot! 🚀**

*Remember: Start on testnet, test thoroughly, and never risk more than you can afford to lose.*

