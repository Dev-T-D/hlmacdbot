# Terminal Setup Guide

Complete terminal/command-line setup guide for running the MACD trading bot with Hyperliquid or Bitunix.

═══════════════════════════════════════════════════════════════

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Installing Dependencies](#installing-dependencies)
4. [Configuration Setup](#configuration-setup)
5. [Testing the Connection](#testing-the-connection)
6. [Running the Bot](#running-the-bot)
7. [Monitoring and Logs](#monitoring-and-logs)
8. [Common Terminal Commands](#common-terminal-commands)
9. [Troubleshooting](#troubleshooting)
10. [Production Deployment](#production-deployment)

═══════════════════════════════════════════════════════════════

## Prerequisites

### System Requirements:
- **Operating System**: Linux, macOS, or Windows (with WSL)
- **Python**: 3.11 or higher
- **Terminal**: bash, zsh, or compatible shell
- **Internet**: Stable connection required

### Check Your Python Version:

```bash
# Check Python version
python3 --version
# Should output: Python 3.11.x or higher

# If not installed, install Python 3.11+:
# Ubuntu/Debian:
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# macOS (with Homebrew):
brew install python@3.11

# Verify pip is available:
pip3 --version
```

═══════════════════════════════════════════════════════════════

## Initial Setup

### Step 1: Navigate to Bot Directory

```bash
# Change to your bot directory:
cd /home/ink/bitunix-macd-bot

# Verify you're in the correct directory:
pwd
# Should output: /home/ink/bitunix-macd-bot

# List files to confirm:
ls -la
# Should see: trading_bot.py, config/, logs/, etc.
```

### Step 2: Create Required Directories

```bash
# Create logs directory if it doesn't exist:
mkdir -p logs

# Create config directory if it doesn't exist:
mkdir -p config

# Verify directories:
ls -la
```

### Step 3: Set Up Python Virtual Environment (Recommended)

```bash
# Create virtual environment:
python3 -m venv venv

# Activate virtual environment:
# On Linux/macOS:
source venv/bin/activate

# On Windows (WSL):
source venv/Scripts/activate

# Your prompt should change to show (venv)
# Example: (venv) user@machine:~/bitunix-macd-bot$

# To deactivate later:
deactivate
```

═══════════════════════════════════════════════════════════════

## Installing Dependencies

### Step 1: Upgrade pip

```bash
# Upgrade pip to latest version:
pip install --upgrade pip

# Verify pip version:
pip --version
```

### Step 2: Install Bot Dependencies

```bash
# Install all dependencies from requirements.txt:
pip install -r requirements.txt

# This installs:
# - pandas (data manipulation)
# - numpy (numerical computing)
# - requests (HTTP requests)
# - python-dateutil (date handling)
# - eth-account (Hyperliquid wallet signing)
# - web3 (Ethereum utilities)
# - eth-utils (Ethereum helper functions)

# Wait for installation to complete...
```

### Step 3: Verify Installation

```bash
# Test imports:
python3 << EOF
import pandas as pd
import numpy as np
import requests
from eth_account import Account
import web3
print("✅ All dependencies installed successfully!")
EOF
```

### Troubleshooting Installation:

If you encounter errors:

```bash
# Error: "No module named 'pip'"
python3 -m ensurepip --upgrade

# Error: Permission denied
pip install --user -r requirements.txt

# Error: "externally-managed-environment"
# Use virtual environment (recommended):
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

═══════════════════════════════════════════════════════════════

## Configuration Setup

### Step 1: Copy Example Configuration

```bash
# Copy example config to create your config:
cp config/config.example.json config/config.json

# Verify it was created:
ls -l config/config.json
```

### Step 2: Edit Configuration

Choose your preferred editor:

#### **Option A: nano (Beginner-friendly)**

```bash
# Open with nano:
nano config/config.json

# Make your changes
# Save: Ctrl+X, then Y, then Enter
```

#### **Option B: vim (Advanced)**

```bash
# Open with vim:
vim config/config.json

# Press 'i' to enter insert mode
# Make your changes
# Press Esc, then type :wq, then Enter to save and quit
```

#### **Option C: GUI Editor**

```bash
# Open with system default editor:
# Linux:
xdg-open config/config.json

# macOS:
open config/config.json

# Edit in the GUI editor, save, and close
```

### Step 3: Configure for Hyperliquid

```bash
# Edit config.json with these settings:
nano config/config.json
```

```json
{
  "exchange": "hyperliquid",
  
  "private_key": "0xYOUR_PRIVATE_KEY_HERE",
  "wallet_address": "0xYOUR_WALLET_ADDRESS_HERE",
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

### Step 4: Secure Configuration File

```bash
# Set strict permissions (only you can read/write):
chmod 600 config/config.json

# Verify permissions:
ls -l config/config.json
# Should show: -rw------- 1 user user ... config.json

# Test the file is readable:
head -n 5 config/config.json
```

### Step 5: Validate Configuration

```bash
# Run configuration validator:
python3 config/config_validator.py

# Expected output:
# ✅ Configuration is valid!

# If errors, fix them and rerun validator
```

═══════════════════════════════════════════════════════════════

## Testing the Connection

### For Hyperliquid:

```bash
# Run connection test:
python3 test_hyperliquid_connection.py

# Watch for:
# ✅ Configuration loaded
# ✅ Client initialized
# ✅ Ticker data retrieved
# ✅ Account info retrieved
# ✅ All 6 tests passing

# If tests fail, check:
# 1. Private key is correct
# 2. Wallet address matches
# 3. Testnet setting matches your wallet
# 4. You have test USDC (for testnet)
```

### For Bitunix:

```bash
# Run connection test:
python3 test_connection.py

# Watch for successful connection messages
```

### Test Output Examples:

**Success:**
```
======================================================================
HYPERLIQUID CONNECTION TEST
======================================================================

✅ Configuration loaded
   Exchange: hyperliquid
   Testnet: True

📡 Initializing Hyperliquid client...
✅ Client initialized for wallet: 0x1234...

----------------------------------------------------------------------
TEST 1: Get Ticker Data
----------------------------------------------------------------------
✅ Ticker data retrieved successfully
   Mark Price: $45,231.50

... (all tests pass)

======================================================================
✅ CONNECTION TEST COMPLETED SUCCESSFULLY
======================================================================
```

**Failure:**
```
❌ TEST 1: Get Ticker Data FAILED
Error: Connection timeout

Fix:
1. Check internet connection
2. Verify API endpoints
3. Check firewall settings
```

═══════════════════════════════════════════════════════════════

## Running the Bot

### Step 1: Start in Dry-Run Mode (Recommended)

```bash
# Ensure config.json has:
# "dry_run": true

# Run bot:
python3 trading_bot.py

# You should see:
# ============================================================
# HYPERLIQUID MACD FUTURES TRADING BOT INITIALIZED
# ============================================================
# Exchange: Hyperliquid
# Symbol: BTCUSDT
# Dry Run Mode: True
# ...
```

### Step 2: Monitor Bot Operation

Open a second terminal for log monitoring:

```bash
# Terminal 1: Run bot
python3 trading_bot.py

# Terminal 2: Watch logs
tail -f logs/bot.log

# You'll see real-time updates:
# - Market data fetches
# - MACD calculations
# - Signal detection
# - Position management
# - Trailing stop updates
```

### Step 3: Stop the Bot Gracefully

```bash
# In the terminal running the bot:
# Press: Ctrl+C

# Bot will:
# 1. Catch the interrupt signal
# 2. Close any open positions (if configured)
# 3. Log shutdown message
# 4. Exit cleanly
```

### Running in Background:

#### Option A: Using nohup

```bash
# Run bot in background:
nohup python3 trading_bot.py > output.log 2>&1 &

# Check if running:
ps aux | grep trading_bot.py

# View logs:
tail -f logs/bot.log

# Stop bot:
pkill -f trading_bot.py
```

#### Option B: Using screen

```bash
# Install screen if not available:
sudo apt install screen  # Ubuntu/Debian
brew install screen      # macOS

# Create new screen session:
screen -S trading-bot

# Run bot:
python3 trading_bot.py

# Detach from screen:
# Press: Ctrl+A, then D

# Reattach to screen:
screen -r trading-bot

# List all screens:
screen -ls

# Kill screen session:
screen -X -S trading-bot quit
```

#### Option C: Using tmux (Recommended)

```bash
# Install tmux if not available:
sudo apt install tmux  # Ubuntu/Debian
brew install tmux      # macOS

# Create new tmux session:
tmux new -s trading-bot

# Run bot:
python3 trading_bot.py

# Detach from tmux:
# Press: Ctrl+B, then D

# Reattach to tmux:
tmux attach -t trading-bot

# List all sessions:
tmux ls

# Kill session:
tmux kill-session -t trading-bot
```

═══════════════════════════════════════════════════════════════

## Monitoring and Logs

### Real-Time Log Monitoring:

```bash
# Watch logs continuously:
tail -f logs/bot.log

# Watch last 100 lines:
tail -n 100 logs/bot.log

# Watch and follow with colors (if supported):
tail -f logs/bot.log | ccze -A

# Stop watching:
# Press: Ctrl+C
```

### Log Analysis:

```bash
# Count total lines:
wc -l logs/bot.log

# Search for errors:
grep "ERROR" logs/bot.log

# Search for trade signals:
grep "ENTRY SIGNAL" logs/bot.log

# Search for position closures:
grep "CLOSING" logs/bot.log

# Count occurrences:
grep -c "ENTRY SIGNAL" logs/bot.log

# Show last 50 errors:
grep "ERROR" logs/bot.log | tail -50

# Filter by date:
grep "2025-11-07" logs/bot.log

# Search multiple terms:
grep -E "ERROR|WARNING" logs/bot.log
```

### Log Rotation:

```bash
# Archive old logs:
cp logs/bot.log logs/bot.$(date +%Y%m%d).log
> logs/bot.log  # Clear current log

# Compress old logs:
gzip logs/bot.20251107.log

# Delete old logs (older than 30 days):
find logs/ -name "*.log.gz" -mtime +30 -delete
```

### System Resource Monitoring:

```bash
# Check bot memory usage:
ps aux | grep trading_bot.py

# Watch CPU and memory:
top -p $(pgrep -f trading_bot.py)

# Continuous monitoring:
watch -n 5 'ps aux | grep trading_bot.py'
```

═══════════════════════════════════════════════════════════════

## Common Terminal Commands

### File Operations:

```bash
# View file contents:
cat config/config.json              # Show entire file
head -20 logs/bot.log               # Show first 20 lines
tail -50 logs/bot.log               # Show last 50 lines
less logs/bot.log                   # Paginated view (q to quit)

# Edit files:
nano filename                       # Simple editor
vim filename                        # Advanced editor

# Copy files:
cp config/config.json config/backup.json

# Move/rename files:
mv old_name.log new_name.log

# Delete files:
rm filename                         # Delete file
rm -rf directory/                   # Delete directory (careful!)

# Create directories:
mkdir new_directory
mkdir -p path/to/nested/directory

# Change permissions:
chmod 600 config/config.json        # Read/write owner only
chmod 755 script.sh                 # Executable script

# Check disk space:
df -h                               # Disk usage
du -sh logs/                        # Directory size
```

### Process Management:

```bash
# Find process ID:
pgrep -f trading_bot.py
ps aux | grep trading_bot.py

# Kill process:
kill PID                            # Graceful stop
kill -9 PID                         # Force kill
pkill -f trading_bot.py             # Kill by name

# Check if running:
ps aux | grep trading_bot.py | grep -v grep
```

### Network Commands:

```bash
# Test connectivity:
ping api.hyperliquid.xyz

# Check DNS resolution:
nslookup api.hyperliquid.xyz

# Test port:
telnet api.hyperliquid.xyz 443

# Check network usage:
nethogs                             # Per-process bandwidth
iftop                               # Network traffic
```

### Git Commands (if using version control):

```bash
# Check status:
git status

# Verify config.json is ignored:
git status
# Should NOT show config.json

# Pull latest changes:
git pull

# View changes:
git diff

# Discard changes:
git checkout -- filename
```

═══════════════════════════════════════════════════════════════

## Troubleshooting

### Common Issues and Solutions:

#### "Command not found"

```bash
# Issue: python3: command not found
# Solution:
which python3                       # Check if installed
python --version                    # Try without '3'
sudo apt install python3           # Install if missing (Ubuntu)

# Issue: pip: command not found
# Solution:
python3 -m pip --version           # Try this instead
sudo apt install python3-pip       # Install if missing
```

#### "Permission denied"

```bash
# Issue: Permission denied when running script
# Solution 1: Add execute permission:
chmod +x script.py

# Solution 2: Run with python explicitly:
python3 script.py

# Issue: Permission denied config.json
# Solution: Fix ownership:
sudo chown $USER:$USER config/config.json
chmod 600 config/config.json
```

#### "Module not found"

```bash
# Issue: ModuleNotFoundError: No module named 'pandas'
# Solution:
pip install pandas                 # Install specific module
pip install -r requirements.txt    # Install all dependencies

# If using virtual environment:
source venv/bin/activate           # Activate first
pip install -r requirements.txt    # Then install
```

#### "Port already in use"

```bash
# Issue: Port 8080 already in use
# Solution: Find and kill process:
lsof -i :8080                      # Find process using port
kill $(lsof -t -i:8080)           # Kill that process
```

#### "Configuration error"

```bash
# Issue: Invalid JSON in config file
# Solution: Validate JSON:
python3 -m json.tool config/config.json

# If errors, common fixes:
# - Remove trailing commas
# - Check quotes are matched
# - Verify brackets are balanced
# - Use online JSON validator
```

#### "Connection timeout"

```bash
# Issue: Connection to API timeout
# Solutions:
# 1. Check internet:
ping 8.8.8.8

# 2. Check DNS:
ping api.hyperliquid.xyz

# 3. Check firewall:
sudo ufw status

# 4. Try different network

# 5. Check if VPN interfering
```

═══════════════════════════════════════════════════════════════

## Production Deployment

### Setting Up a Production Environment:

#### Step 1: Use Environment Variables

```bash
# Create .env file:
nano .env

# Add credentials:
HYPERLIQUID_PRIVATE_KEY=0x...
HYPERLIQUID_WALLET_ADDRESS=0x...

# Secure it:
chmod 600 .env

# Bot will auto-load these
```

#### Step 2: Set Up Systemd Service (Linux)

```bash
# Create service file:
sudo nano /etc/systemd/system/trading-bot.service
```

```ini
[Unit]
Description=MACD Trading Bot
After=network.target

[Service]
Type=simple
User=yourusername
WorkingDirectory=/home/ink/bitunix-macd-bot
ExecStart=/home/ink/bitunix-macd-bot/venv/bin/python3 trading_bot.py
Restart=always
RestartSec=10
Environment=PATH=/home/ink/bitunix-macd-bot/venv/bin

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service:
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot

# Check status:
sudo systemctl status trading-bot

# View logs:
sudo journalctl -u trading-bot -f

# Stop/restart:
sudo systemctl stop trading-bot
sudo systemctl restart trading-bot
```

#### Step 3: Set Up Log Rotation

```bash
# Create logrotate config:
sudo nano /etc/logrotate.d/trading-bot
```

```
/home/ink/bitunix-macd-bot/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 yourusername yourusername
    sharedscripts
    postrotate
        pkill -HUP -f trading_bot.py
    endscript
}
```

#### Step 4: Set Up Monitoring

```bash
# Install monitoring tools:
sudo apt install htop iotop nethogs

# Create monitoring script:
nano monitor.sh
```

```bash
#!/bin/bash
while true; do
    clear
    echo "=== Trading Bot Monitor ==="
    echo "Time: $(date)"
    echo
    echo "=== Process Status ==="
    ps aux | grep trading_bot.py | grep -v grep || echo "Bot not running!"
    echo
    echo "=== Last 10 Log Lines ==="
    tail -10 logs/bot.log
    echo
    sleep 10
done
```

```bash
# Make executable:
chmod +x monitor.sh

# Run:
./monitor.sh
```

═══════════════════════════════════════════════════════════════

## Quick Reference

### Essential Commands:

```bash
# Activate environment:
source venv/bin/activate

# Validate config:
python3 config/config_validator.py

# Test connection:
python3 test_hyperliquid_connection.py

# Run bot:
python3 trading_bot.py

# Watch logs:
tail -f logs/bot.log

# Stop bot:
Ctrl+C (in bot terminal) or pkill -f trading_bot.py

# Check if running:
ps aux | grep trading_bot.py

# View recent errors:
grep "ERROR" logs/bot.log | tail -20
```

═══════════════════════════════════════════════════════════════

## Summary

You've learned:
- ✅ How to set up Python environment
- ✅ How to install dependencies
- ✅ How to configure the bot via terminal
- ✅ How to test connections
- ✅ How to run and monitor the bot
- ✅ Common terminal commands
- ✅ Troubleshooting techniques
- ✅ Production deployment options

**Next Steps:**
1. Complete configuration
2. Run connection tests
3. Start bot in dry-run mode
4. Monitor logs
5. Proceed to live trading when ready

═══════════════════════════════════════════════════════════════

**Happy trading! 🚀**

*For more help, see: README.md, HYPERLIQUID_SETUP.md*

