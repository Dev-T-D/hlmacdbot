# Production Deployment Guide

This guide provides comprehensive instructions for deploying the Hyperliquid MACD trading bot to production environments, including security hardening, monitoring setup, and operational procedures.

## 📋 Table of Contents

- [Prerequisites](#-prerequisites)
- [Ubuntu Server Setup](#-ubuntu-server-setup)
- [Security Hardening](#-security-hardening)
- [Application Deployment](#-application-deployment)
- [SystemD Service Setup](#-systemd-service-setup)
- [Monitoring Integration](#-monitoring-integration)
- [Backup and Disaster Recovery](#-backup-and-disaster-recovery)
- [Scaling Considerations](#-scaling-considerations)
- [Troubleshooting](#-troubleshooting)

## 📋 Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **CPU** | 2 cores | 4+ cores | For parallel processing and monitoring |
| **RAM** | 4GB | 8GB+ | For caching and concurrent operations |
| **Storage** | 20GB SSD | 50GB+ SSD | For logs, data, and backups |
| **Network** | 10 Mbps | 100 Mbps+ | Low latency essential for trading |

### Software Requirements

#### Operating System
- **Ubuntu 20.04 LTS** or **22.04 LTS** (recommended)
- **Kernel**: 5.4+ with real-time capabilities (optional)

#### System Packages
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    htop \
    iotop \
    sysstat \
    logrotate \
    fail2ban \
    ufw \
    unattended-upgrades
```

#### Python Environment
```bash
# Install Python 3.11 if not available
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Verify installation
python3.11 --version
```

## 🖥️ Ubuntu Server Setup

### 1. Initial Server Configuration

#### Create Trading User
```bash
# Create dedicated user for trading bot
sudo useradd -m -s /bin/bash trading
sudo usermod -aG sudo trading

# Set password
sudo passwd trading

# Switch to trading user
su - trading
```

#### Directory Structure
```bash
# Create application directory structure
mkdir -p ~/trading-bot
cd ~/trading-bot

# Create subdirectories
mkdir -p config logs data backups scripts monitoring

# Set proper permissions
chmod 700 ~/trading-bot
chmod 755 ~/trading-bot/logs
chmod 755 ~/trading-bot/monitoring
```

#### Environment Variables
```bash
# Create environment file
cat > ~/.trading_env << 'EOF'
# Trading Bot Environment Variables
export HYPERLIQUID_PRIVATE_KEY="your-private-key-here"
export HYPERLIQUID_WALLET_ADDRESS="your-wallet-address-here"
export BOT_ENVIRONMENT="production"
export BOT_DRY_RUN="false"
export LOG_LEVEL="INFO"

# Database
export BOT_STATE_DB_PATH="/home/trading/trading-bot/data/bot_state.db"

# Monitoring
export PROMETHEUS_PORT="8000"
export HEALTH_CHECK_PORT="8001"

# Security
export BOT_EMERGENCY_SHUTDOWN="false"
export AUDIT_LOG_PATH="/home/trading/trading-bot/logs/audit.log"
EOF

# Secure the environment file
chmod 600 ~/.trading_env
```

### 2. Firewall Configuration

```bash
# Enable UFW
sudo ufw enable

# Allow SSH (change port if customized)
sudo ufw allow ssh
sudo ufw allow 22

# Allow monitoring ports
sudo ufw allow 8000/tcp  # Prometheus metrics
sudo ufw allow 8001/tcp  # Health checks

# Allow Grafana (if installed locally)
sudo ufw allow 3000/tcp

# Deny all other incoming traffic
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Check status
sudo ufw status verbose
```

### 3. SSH Security

#### SSH Key Authentication
```bash
# Generate SSH key pair on your local machine
ssh-keygen -t ed25519 -C "trading-server"

# Copy public key to server
ssh-copy-id trading@your-server-ip

# Disable password authentication
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

# Restart SSH service
sudo systemctl restart ssh

# Test SSH key authentication
ssh trading@your-server-ip
```

#### SSH Configuration Hardening
```bash
# Backup original config
sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup

# Apply security settings
sudo tee -a /etc/ssh/sshd_config > /dev/null <<EOF
# Security hardening
Port 22
PermitRootLogin no
PermitEmptyPasswords no
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2

# Disable unused authentication methods
ChallengeResponseAuthentication no
KerberosAuthentication no
GSSAPIAuthentication no
EOF

# Restart SSH
sudo systemctl restart ssh
```

## 🔒 Security Hardening

### 1. System Security

#### Automatic Updates
```bash
# Configure unattended upgrades
sudo dpkg-reconfigure --priority=low unattended-upgrades

# Edit configuration
sudo nano /etc/apt/apt.conf.d/50unattended-upgrades

# Ensure these lines are present:
# Unattended-Upgrade::Allowed-Origins {
#     "${distro_id}:${distro_codename}-security";
#     "${distro_id}:${distro_codename}-updates";
# };

# Enable automatic updates
sudo systemctl enable unattended-upgrades
sudo systemctl start unattended-upgrades
```

#### Fail2Ban Setup
```bash
# Configure fail2ban for SSH protection
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local

# Edit jail.local
sudo nano /etc/fail2ban/jail.local

# Ensure SSH jail is enabled:
# [sshd]
# enabled = true
# port = ssh
# filter = sshd
# logpath = /var/log/auth.log
# maxretry = 3
# bantime = 3600

# Restart fail2ban
sudo systemctl enable fail2ban
sudo systemctl restart fail2ban
```

#### System Monitoring
```bash
# Install monitoring tools
sudo apt install -y htop iotop sysstat ncdu

# Enable sysstat data collection
sudo sed -i 's/ENABLED="false"/ENABLED="true"/' /etc/default/sysstat
sudo systemctl enable sysstat
sudo systemctl start sysstat
```

### 2. Application Security

#### Private Key Management
```bash
# Generate secure encryption key for key storage
python3.11 -c "
import secrets
key = secrets.token_bytes(32)
print('Encryption key:', key.hex())
" > encryption_key.txt

# Store securely
chmod 600 encryption_key.txt
mv encryption_key.txt ~/trading-bot/config/
```

#### File Permissions
```bash
# Set proper permissions for application files
cd ~/trading-bot

# Config files - readable by owner only
chmod 600 config/*

# Log files - writable by application
chmod 755 logs/
chmod 644 logs/*.log 2>/dev/null || true

# Data directory - application access only
chmod 700 data/

# Scripts - executable by owner
chmod 700 scripts/*.sh 2>/dev/null || true
find scripts/ -name "*.sh" -exec chmod +x {} \;
```

#### Log Rotation
```bash
# Create logrotate configuration
sudo tee /etc/logrotate.d/trading-bot > /dev/null <<EOF
/home/trading/trading-bot/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 trading trading
    postrotate
        systemctl reload trading-bot 2>/dev/null || true
    endscript
}
EOF

# Test logrotate configuration
sudo logrotate -d /etc/logrotate.d/trading-bot
```

## 🚀 Application Deployment

### 1. Code Deployment

#### Clone Repository
```bash
cd ~/trading-bot

# Clone the repository
git clone https://github.com/yourusername/hyperliquid-macd-bot.git .

# Or if using SSH
git clone git@github.com:yourusername/hyperliquid-macd-bot.git .
```

#### Virtual Environment Setup
```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install additional production dependencies
pip install gunicorn prometheus_client

# Deactivate virtual environment
deactivate
```

#### Configuration Setup
```bash
# Copy configuration template
cp config/config.example.json config/config.json

# Edit production configuration
nano config/config.json
```

Production configuration example:
```json
{
  "exchange": "hyperliquid",
  "testnet": false,
  "private_key": "0x...",
  "wallet_address": "0x...",

  "trading": {
    "symbol": "BTCUSDT",
    "timeframe": "5m",
    "check_interval": 300,
    "dry_run": false
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

  "resilience": {
    "enabled": true,
    "state_db_path": "/home/trading/trading-bot/data/bot_state.db",
    "circuit_breaker_failure_threshold": 5,
    "max_retries_per_hour": 100,
    "heartbeat_interval_seconds": 60
  },

  "monitoring": {
    "enabled": true,
    "metrics_port": 8000,
    "health_port": 8001,
    "structured_logging": true
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
        "smtp_port": 587,
        "username": "alerts@yourdomain.com",
        "to_emails": ["admin@yourdomain.com"]
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

### 2. Pre-deployment Testing

#### Health Checks
```bash
cd ~/trading-bot
source venv/bin/activate

# Test imports
python3.11 -c "import trading_bot; print('✅ Imports successful')"

# Test configuration validation
python3.11 test_config_validation.py

# Test Hyperliquid connection
python3.11 test_hyperliquid_connection.py

# Test strategy calculations
python3.11 -c "
from macd_strategy_enhanced import EnhancedMACDStrategy
strategy = EnhancedMACDStrategy({})
print('✅ Strategy initialization successful')
"

deactivate
```

#### Dry Run Testing
```bash
# Test in dry-run mode
export BOT_DRY_RUN=true
source venv/bin/activate

timeout 300 python3.11 trading_bot.py --dry-run --log-level DEBUG

deactivate
```

## ⚙️ SystemD Service Setup

### 1. Create SystemD Service

```bash
# Create systemd service file
sudo tee /etc/systemd/system/trading-bot.service > /dev/null <<EOF
[Unit]
Description=Hyperliquid MACD Trading Bot
After=network.target
Wants=network.target

[Service]
Type=simple
User=trading
Group=trading
EnvironmentFile=/home/trading/.trading_env
WorkingDirectory=/home/trading/trading-bot
ExecStart=/home/trading/trading-bot/venv/bin/python3.11 trading_bot.py
ExecReload=/bin/kill -HUP \$MAINPID

# Restart on failure
Restart=always
RestartSec=10

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/home/trading/trading-bot
ProtectHome=true

# Resource limits
MemoryLimit=1G
CPUQuota=200%

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=trading-bot

[Install]
WantedBy=multi-user.target
EOF
```

### 2. SystemD Configuration

#### Service Management
```bash
# Reload systemd daemon
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable trading-bot

# Start service
sudo systemctl start trading-bot

# Check service status
sudo systemctl status trading-bot

# View service logs
sudo journalctl -u trading-bot -f

# Restart service
sudo systemctl restart trading-bot

# Stop service
sudo systemctl stop trading-bot
```

#### Watchdog Configuration
```bash
# Create watchdog service
sudo tee /etc/systemd/system/trading-bot-watchdog.service > /dev/null <<EOF
[Unit]
Description=Trading Bot Watchdog
After=trading-bot.service
Requires=trading-bot.service

[Service]
Type=simple
User=trading
Group=trading
EnvironmentFile=/home/trading/.trading_env
WorkingDirectory=/home/trading/trading-bot
ExecStart=/home/trading/trading-bot/venv/bin/python3.11 watchdog.py
Restart=always
RestartSec=30

# Security
NoNewPrivileges=true
PrivateTmp=true

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=trading-bot-watchdog

[Install]
WantedBy=multi-user.target
EOF

# Enable and start watchdog
sudo systemctl daemon-reload
sudo systemctl enable trading-bot-watchdog
sudo systemctl start trading-bot-watchdog
```

### 3. Service Monitoring

#### SystemD Timers for Health Checks
```bash
# Create health check timer
sudo tee /etc/systemd/system/trading-bot-health.timer > /dev/null <<EOF
[Unit]
Description=Run trading bot health checks every 5 minutes

[Timer]
OnBootSec=5min
OnUnitActiveSec=5min
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Create health check service
sudo tee /etc/systemd/system/trading-bot-health.service > /dev/null <<EOF
[Unit]
Description=Trading Bot Health Check

[Service]
Type=oneshot
User=trading
WorkingDirectory=/home/trading/trading-bot
ExecStart=/home/trading/trading-bot/scripts/health_check.sh
EOF

# Enable health check timer
sudo systemctl daemon-reload
sudo systemctl enable trading-bot-health.timer
sudo systemctl start trading-bot-health.timer
```

## 📊 Monitoring Integration

### 1. Prometheus Setup

#### Install Prometheus
```bash
# Download and install Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.40.0/prometheus-2.40.0.linux-amd64.tar.gz
tar xvf prometheus-2.40.0.linux-amd64.tar.gz
sudo mv prometheus-2.40.0.linux-amd64 /opt/prometheus

# Create prometheus user
sudo useradd -M -r -s /bin/false prometheus

# Set permissions
sudo chown -R prometheus:prometheus /opt/prometheus

# Create systemd service
sudo tee /etc/systemd/system/prometheus.service > /dev/null <<EOF
[Unit]
Description=Prometheus
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/opt/prometheus/prometheus \
  --config.file=/opt/prometheus/prometheus.yml \
  --storage.tsdb.path=/opt/prometheus/data \
  --web.console.templates=/opt/prometheus/consoles \
  --web.console.libraries=/opt/prometheus/console_libraries

[Install]
WantedBy=multi-user.target
EOF
```

#### Prometheus Configuration
```bash
# Create Prometheus configuration
sudo tee /opt/prometheus/prometheus.yml > /dev/null <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'trading-bot'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 5s

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
EOF

# Start Prometheus
sudo systemctl daemon-reload
sudo systemctl enable prometheus
sudo systemctl start prometheus
```

### 2. Grafana Setup

#### Install Grafana
```bash
# Add Grafana repository
sudo apt-get install -y apt-transport-https
sudo apt-get install -y software-properties-common wget
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -

echo "deb https://packages.grafana.com/oss/deb stable main" | sudo tee -a /etc/apt/sources.list.d/grafana.list

# Install Grafana
sudo apt-get update
sudo apt-get install grafana

# Start Grafana
sudo systemctl daemon-reload
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```

#### Grafana Configuration
```bash
# Access Grafana at http://your-server:3000
# Default credentials: admin/admin

# Add Prometheus data source
# Configuration -> Data Sources -> Add data source -> Prometheus
# URL: http://localhost:9090

# Import trading bot dashboards
python3.11 grafana_dashboards.py
```

### 3. Node Exporter Setup

#### Install Node Exporter
```bash
# Download and install Node Exporter
wget https://github.com/prometheus/node_exporter/releases/download/v1.5.0/node_exporter-1.5.0.linux-amd64.tar.gz
tar xvf node_exporter-1.5.0.linux-amd64.tar.gz
sudo mv node_exporter-1.5.0.linux-amd64 /opt/node_exporter

# Create node_exporter user
sudo useradd -M -r -s /bin/false node_exporter
sudo chown -R node_exporter:node_exporter /opt/node_exporter

# Create systemd service
sudo tee /etc/systemd/system/node_exporter.service > /dev/null <<EOF
[Unit]
Description=Node Exporter
After=network.target

[Service]
User=node_exporter
Group=node_exporter
Type=simple
ExecStart=/opt/node_exporter/node_exporter

[Install]
WantedBy=multi-user.target
EOF

# Start Node Exporter
sudo systemctl daemon-reload
sudo systemctl enable node_exporter
sudo systemctl start node_exporter
```

## 💾 Backup and Disaster Recovery

### 1. Backup Strategy

#### Automated Backup Script
```bash
# Create backup script
tee ~/trading-bot/scripts/backup.sh > /dev/null <<'EOF'
#!/bin/bash

# Trading Bot Backup Script
BACKUP_DIR="/home/trading/trading-bot/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="trading_bot_${TIMESTAMP}"

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Create backup archive
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" \
    --exclude='*.log' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='backups' \
    /home/trading/trading-bot/

# Keep only last 7 daily backups
find "${BACKUP_DIR}" -name "trading_bot_*.tar.gz" -mtime +7 -delete

# Log backup completion
echo "$(date): Backup completed - ${BACKUP_NAME}" >> /home/trading/trading-bot/logs/backup.log

# Optional: Upload to remote storage
# rclone copy "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" remote:backups/
EOF

# Make executable
chmod +x ~/trading-bot/scripts/backup.sh
```

#### SystemD Timer for Backups
```bash
# Create backup timer
sudo tee /etc/systemd/system/trading-bot-backup.timer > /dev/null <<EOF
[Unit]
Description=Run trading bot backup daily

[Timer]
OnCalendar=daily
Persistent=true
RandomizedDelaySec=3600

[Install]
WantedBy=timers.target
EOF

# Create backup service
sudo tee /etc/systemd/system/trading-bot-backup.service > /dev/null <<EOF
[Unit]
Description=Trading Bot Backup

[Service]
Type=oneshot
User=trading
ExecStart=/home/trading/trading-bot/scripts/backup.sh
EOF

# Enable backup timer
sudo systemctl daemon-reload
sudo systemctl enable trading-bot-backup.timer
sudo systemctl start trading-bot-backup.timer
```

### 2. Disaster Recovery

#### Recovery Script
```bash
# Create recovery script
tee ~/trading-bot/scripts/recovery.sh > /dev/null <<'EOF'
#!/bin/bash

# Trading Bot Disaster Recovery Script
BACKUP_DIR="/home/trading/trading-bot/backups"
RECOVERY_DIR="/tmp/trading_bot_recovery"

echo "Starting disaster recovery..."

# Stop services
sudo systemctl stop trading-bot
sudo systemctl stop trading-bot-watchdog

# Create recovery directory
mkdir -p "${RECOVERY_DIR}"

# Find latest backup
LATEST_BACKUP=$(find "${BACKUP_DIR}" -name "trading_bot_*.tar.gz" | sort | tail -1)

if [ -z "${LATEST_BACKUP}" ]; then
    echo "ERROR: No backup found!"
    exit 1
fi

echo "Using backup: ${LATEST_BACKUP}"

# Extract backup
tar -xzf "${LATEST_BACKUP}" -C "${RECOVERY_DIR}"

# Validate backup integrity
if [ ! -f "${RECOVERY_DIR}/home/trading/trading-bot/config/config.json" ]; then
    echo "ERROR: Invalid backup - config.json not found"
    exit 1
fi

# Backup current state (if exists)
if [ -d "/home/trading/trading-bot" ]; then
    mv /home/trading/trading-bot /home/trading/trading-bot.backup.$(date +%s)
fi

# Restore from backup
mv "${RECOVERY_DIR}/home/trading/trading-bot" /home/trading/

# Restore permissions
chown -R trading:trading /home/trading/trading-bot
chmod 600 /home/trading/trading-bot/config/config.json

# Clean up
rm -rf "${RECOVERY_DIR}"

# Test recovery
cd /home/trading/trading-bot
source venv/bin/activate
python3.11 -c "import trading_bot; print('Recovery successful')"

# Restart services
sudo systemctl start trading-bot-watchdog
sudo systemctl start trading-bot

echo "Disaster recovery completed successfully"
EOF

# Make executable
chmod +x ~/trading-bot/scripts/recovery.sh
```

#### Recovery Testing
```bash
# Test recovery procedure
sudo systemctl stop trading-bot
~/trading-bot/scripts/backup.sh
~/trading-bot/scripts/recovery.sh
```

## 📈 Scaling Considerations

### Horizontal Scaling

#### Multiple Trading Instances
```bash
# Create multiple instances for different symbols
for symbol in BTCUSDT ETHUSDT; do
    # Create instance directory
    mkdir -p ~/trading-bot-${symbol}

    # Copy codebase
    cp -r ~/trading-bot/* ~/trading-bot-${symbol}/

    # Modify configuration
    sed -i "s/BTCUSDT/${symbol}/g" ~/trading-bot-${symbol}/config/config.json

    # Create systemd service
    sudo tee /etc/systemd/system/trading-bot-${symbol}.service > /dev/null <<EOF
[Unit]
Description=Trading Bot ${symbol}
After=network.target

[Service]
Type=simple
User=trading
EnvironmentFile=/home/trading/.trading_env
WorkingDirectory=/home/trading/trading-bot-${symbol}
ExecStart=/home/trading/trading-bot-${symbol}/venv/bin/python3.11 trading_bot.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

    # Enable and start service
    sudo systemctl daemon-reload
    sudo systemctl enable trading-bot-${symbol}
    sudo systemctl start trading-bot-${symbol}
done
```

### Vertical Scaling

#### Performance Optimization
```bash
# Increase system limits
echo "trading soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "trading hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize kernel parameters
sudo tee -a /etc/sysctl.conf > /dev/null <<EOF
# Network optimization
net.core.somaxconn = 65536
net.ipv4.tcp_tw_reuse = 1
net.ipv4.ip_local_port_range = 1024 65535

# Memory management
vm.swappiness = 10
vm.dirty_ratio = 60
vm.dirty_background_ratio = 2
EOF

sudo sysctl -p
```

### Load Balancing

#### API Call Distribution
```python
# Implement load balancing in client
class LoadBalancedHyperliquidClient:
    def __init__(self, configs):
        self.clients = [HyperliquidClient(config) for config in configs]
        self.current_client = 0

    def get_ticker(self, symbol):
        client = self.clients[self.current_client]
        self.current_client = (self.current_client + 1) % len(self.clients)
        return client.get_ticker(symbol)
```

## 🔧 Troubleshooting

### Common Deployment Issues

#### Service Won't Start
```bash
# Check service status
sudo systemctl status trading-bot

# View detailed logs
sudo journalctl -u trading-bot -n 50

# Check configuration
cd ~/trading-bot
source venv/bin/activate
python3.11 -c "import json; print(json.load(open('config/config.json')))"

# Test basic functionality
python3.11 -c "from hyperliquid_client import HyperliquidClient; print('Import successful')"
```

#### High Memory Usage
```bash
# Monitor memory usage
htop

# Check for memory leaks
ps aux --sort=-%mem | head -10

# Restart service if needed
sudo systemctl restart trading-bot

# Check logs for memory issues
grep -i memory ~/trading-bot/logs/trading_bot.log | tail -20
```

#### Network Connectivity Issues
```bash
# Test basic connectivity
ping api.hyperliquid.xyz

# Test API endpoints
curl -v https://api.hyperliquid.xyz/info

# Check firewall rules
sudo ufw status

# Test WebSocket connection
python3.11 -c "
import asyncio
import websockets
async def test():
    try:
        async with websockets.connect('wss://api.hyperliquid.xyz/ws') as ws:
            print('WebSocket connection successful')
    except Exception as e:
        print(f'WebSocket connection failed: {e}')
asyncio.run(test())
"
```

#### Database Issues
```bash
# Check database file
ls -la ~/trading-bot/data/bot_state.db

# Verify database integrity
cd ~/trading-bot
source venv/bin/activate
python3.11 -c "
import sqlite3
conn = sqlite3.connect('data/bot_state.db')
cursor = conn.cursor()
cursor.execute('SELECT name FROM sqlite_master WHERE type=\"table\"')
tables = cursor.fetchall()
print('Database tables:', tables)
conn.close()
"

# Rebuild database if corrupted
rm ~/trading-bot/data/bot_state.db
sudo systemctl restart trading-bot
```

### Emergency Procedures

#### Emergency Shutdown
```bash
# Immediate shutdown
echo "BOT_EMERGENCY_SHUTDOWN=true" >> ~/.trading_env
sudo systemctl restart trading-bot

# Or manually
sudo systemctl stop trading-bot
sudo systemctl stop trading-bot-watchdog

# Force kill if needed
pkill -f trading_bot.py
pkill -f watchdog.py
```

#### Recovery from Crash
```bash
# Check what crashed
sudo journalctl -u trading-bot --since "1 hour ago" | tail -50

# Restart services
sudo systemctl start trading-bot-watchdog
sudo systemctl start trading-bot

# Check if recovery worked
curl http://localhost:8001/health
```

### Performance Monitoring

#### System Performance
```bash
# CPU usage
top -b -n1 | head -20

# Disk I/O
iotop -b -n1

# Network connections
netstat -tuln | grep :800

# System load
uptime
cat /proc/loadavg
```

#### Application Performance
```bash
# Check metrics
curl http://localhost:8000 | jq '.data.result[0].value'

# Profile application
cd ~/trading-bot
source venv/bin/activate
python3.11 -m cProfile -s time trading_bot.py --dry-run --profile-duration 60
```

### Log Analysis

#### Common Log Patterns
```bash
# View recent errors
grep -i error ~/trading-bot/logs/trading_bot.log | tail -10

# Check for API failures
grep -i "api.*fail" ~/trading-bot/logs/trading_bot.log | tail -10

# Monitor position changes
grep -i "position" ~/trading-bot/logs/trading_bot.log | tail -10

# Check strategy signals
grep -i "signal" ~/trading-bot/logs/trading_bot.log | tail -10
```

#### Log Rotation Issues
```bash
# Check logrotate status
sudo logrotate -d /etc/logrotate.d/trading-bot

# Force log rotation
sudo logrotate -f /etc/logrotate.d/trading-bot

# Check rotated logs
ls -la ~/trading-bot/logs/
```

This comprehensive deployment guide provides everything needed to deploy the trading bot to production with enterprise-grade reliability, security, and monitoring. Follow each section carefully and test thoroughly before going live with real funds.
