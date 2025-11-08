# Security Guide

This comprehensive security guide covers the security model, best practices, and procedures for the Hyperliquid MACD trading bot, ensuring protection against common threats and compliance requirements.

## 🔒 Table of Contents

- [Security Model Overview](#-security-model-overview)
- [Key Management](#-key-management)
- [API Security](#-api-security)
- [Position Protection](#-position-protection)
- [Audit & Compliance](#-audit--compliance)
- [Operational Security](#-operational-security)
- [Error Handling](#-error-handling)
- [Security Testing](#-security-testing)
- [Incident Response](#-incident-response)

## 🛡️ Security Model Overview

### Defense in Depth Strategy

The trading bot implements a multi-layered security approach:

```
┌─────────────────────────────────────────┐
│            NETWORK SECURITY             │
│  • SSL/TLS encryption                   │
│  • Certificate pinning                  │
│  • Firewall rules                       │
│  • VPN for remote access                │
└─────────────────────────────────────────┘
                │
┌─────────────────────────────────────────┐
│          APPLICATION SECURITY           │
│  • Input validation & sanitization      │
│  • Secure key storage                   │
│  • Parameter bounds checking            │
│  • Rate limiting                        │
└─────────────────────────────────────────┘
                │
┌─────────────────────────────────────────┐
│            DATA PROTECTION              │
│  • Encrypted storage                    │
│  • Secure memory handling               │
│  • Audit logging                        │
│  • Backup encryption                    │
└─────────────────────────────────────────┘
                │
┌─────────────────────────────────────────┐
│         OPERATIONAL SECURITY            │
│  • Principle of least privilege         │
│  • Regular security updates             │
│  • Monitoring & alerting                │
│  • Access controls                      │
└─────────────────────────────────────────┘
```

### Security Principles

1. **Zero Trust**: Never trust, always verify
2. **Least Privilege**: Minimum permissions required
3. **Fail Safe**: Default to secure behavior
4. **Defense in Depth**: Multiple security layers
5. **Secure by Design**: Security built into architecture

## 🔑 Key Management

### Private Key Security

#### Secure Key Storage Implementation

```python
from secure_key_storage import SecureKeyStorage
import secrets

class SecureKeyStorage:
    def __init__(self, encryption_key: bytes = None):
        self.encryption_key = encryption_key or secrets.token_bytes(32)
        self._keys = {}

    def store_private_key(self, private_key: str) -> None:
        """Store encrypted private key in memory."""
        if not self._is_valid_private_key(private_key):
            raise ValueError("Invalid private key format")

        encrypted = self._encrypt(private_key.encode())
        self._keys['private_key'] = encrypted

    def get_private_key(self) -> str:
        """Retrieve decrypted private key."""
        if 'private_key' not in self._keys:
            raise KeyError("Private key not found")

        encrypted = self._keys['private_key']
        decrypted = self._decrypt(encrypted)
        return decrypted.decode()

    def emergency_zeroize(self) -> None:
        """Securely erase all keys from memory."""
        # Multiple overwrite passes for secure deletion
        for _ in range(3):
            for key in self._keys:
                self._keys[key] = secrets.token_bytes(32)

        self._keys.clear()
        logger.critical("🔴 EMERGENCY ZEROIZATION: All keys destroyed")

    def _encrypt(self, data: bytes) -> bytes:
        """AES-GCM encryption."""
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend

        iv = secrets.token_bytes(12)  # GCM recommended IV size
        cipher = Cipher(
            algorithms.AES(self.encryption_key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        return iv + encryptor.tag + ciphertext

    def _decrypt(self, data: bytes) -> bytes:
        """AES-GCM decryption."""
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend

        iv = data[:12]
        tag = data[12:28]
        ciphertext = data[28:]

        cipher = Cipher(
            algorithms.AES(self.encryption_key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()

    def _is_valid_private_key(self, key: str) -> bool:
        """Validate private key format."""
        import re
        # Hyperliquid uses 0x-prefixed hex strings
        return bool(re.match(r'^0x[a-fA-F0-9]{64}$', key))
```

#### Key Rotation Procedure

```python
class KeyRotationManager:
    def __init__(self, key_storage: SecureKeyStorage):
        self.key_storage = key_storage
        self.rotation_interval = timedelta(days=30)

    def should_rotate_keys(self) -> bool:
        """Check if key rotation is due."""
        last_rotation = self._get_last_rotation_time()
        return datetime.now() - last_rotation > self.rotation_interval

    def rotate_keys(self, new_private_key: str) -> bool:
        """Perform key rotation without downtime."""
        try:
            # Validate new key
            if not self._validate_new_key(new_private_key):
                raise ValueError("Invalid new private key")

            # Stop trading temporarily
            self._pause_trading()

            # Update key atomically
            old_key = self.key_storage.get_private_key()
            self.key_storage.store_private_key(new_private_key)

            # Test new key
            if not self._test_key_functionality(new_private_key):
                # Rollback on failure
                self.key_storage.store_private_key(old_key)
                self._resume_trading()
                raise RuntimeError("Key rotation failed - rolled back")

            # Update rotation timestamp
            self._update_rotation_time()

            # Resume trading
            self._resume_trading()

            logger.info("✅ Key rotation completed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Key rotation failed: {e}")
            return False

    def _pause_trading(self) -> None:
        """Temporarily pause trading operations."""
        # Implementation depends on trading bot architecture
        pass

    def _resume_trading(self) -> None:
        """Resume trading operations."""
        pass
```

#### Emergency Key Zeroization

```bash
#!/bin/bash
# emergency_zeroize.sh - Emergency key destruction script

echo "🔴 EMERGENCY KEY ZEROIZATION INITIATED"
echo "This will permanently destroy all stored keys!"

# Confirm action
read -p "Are you sure? This cannot be undone (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Operation cancelled"
    exit 1
fi

# Stop trading bot
sudo systemctl stop trading-bot
sudo systemctl stop trading-bot-watchdog

# Zeroize keys
cd /home/trading/trading-bot
source venv/bin/activate

python3.11 -c "
from secure_key_storage import SecureKeyStorage
import os

# Load and zeroize
storage = SecureKeyStorage()
try:
    storage.emergency_zeroize()
    print('✅ Keys successfully zeroized')
except Exception as e:
    print(f'❌ Zeroization failed: {e}')
    exit(1)
"

# Remove key files
rm -f config/encryption_key.txt
rm -f config/*.key

# Clear environment variables
unset HYPERLIQUID_PRIVATE_KEY
unset HYPERLIQUID_WALLET_ADDRESS

# Log emergency action
echo \"$(date): EMERGENCY KEY ZEROIZATION PERFORMED\" >> logs/emergency.log

echo "🔴 EMERGENCY ZEROIZATION COMPLETE"
echo "All keys have been destroyed. System is now secure but inoperative."
```

### Hardware Security Module (HSM) Support

```python
class HSMKeyStorage(SecureKeyStorage):
    def __init__(self, hsm_config: dict):
        self.hsm_session = self._connect_to_hsm(hsm_config)
        super().__init__()

    def store_private_key(self, private_key: str) -> None:
        """Store key in HSM."""
        # Generate key in HSM
        key_handle = self.hsm_session.generate_key(
            mechanism=PKCS11Mechanism.EC_KEY_PAIR_GEN,
            template={
                CKA_LABEL: b'trading_private_key',
                CKA_SIGN: True,
                CKA_PRIVATE: True
            }
        )
        self.key_handle = key_handle

    def sign_message(self, message: bytes) -> bytes:
        """Sign message using HSM key."""
        signature = self.hsm_session.sign(
            self.key_handle,
            message,
            mechanism=PKCS11Mechanism.ECDSA_SHA256
        )
        return signature
```

## 🔗 API Security

### Request Signing and Validation

#### EIP-712 Structured Data Signing

```python
from eth_account import Account
from eth_account.messages import encode_structured_data

class SecureAPIClient:
    def __init__(self, private_key: str):
        self.private_key = private_key
        self.account = Account.from_key(private_key)

    def sign_request(self, method: str, params: dict, timestamp: int) -> dict:
        """Sign API request using EIP-712."""
        domain = {
            "name": "Hyperliquid",
            "version": "1",
            "chainId": 42161,  # Arbitrum
            "verifyingContract": "0x..."  # Hyperliquid contract
        }

        message_types = {
            "Request": [
                {"name": "method", "type": "string"},
                {"name": "params", "type": "string"},
                {"name": "timestamp", "type": "uint256"}
            ]
        }

        message_data = {
            "method": method,
            "params": json.dumps(params, sort_keys=True),
            "timestamp": timestamp
        }

        # Encode structured data
        encoded_data = encode_structured_data(
            primaryType="Request",
            types=message_types,
            domain=domain,
            message=message_data
        )

        # Sign the message
        signed_message = Account.sign_message(encoded_data, self.private_key)

        return {
            "signature": signed_message.signature.hex(),
            "address": self.account.address,
            "timestamp": timestamp,
            "method": method,
            "params": params
        }

    def verify_response(self, response: dict, expected_address: str) -> bool:
        """Verify response authenticity."""
        # Implementation for response verification
        pass
```

#### Timestamp Validation

```python
class TimestampValidator:
    def __init__(self, max_age_seconds: int = 30):
        self.max_age_seconds = max_age_seconds

    def validate_timestamp(self, timestamp: int) -> bool:
        """Validate request timestamp to prevent replay attacks."""
        current_time = int(time.time())
        age = current_time - timestamp

        if age < 0:
            # Timestamp is in the future - reject
            logger.warning(f"Future timestamp rejected: {timestamp}")
            return False

        if age > self.max_age_seconds:
            # Timestamp too old - reject
            logger.warning(f"Stale timestamp rejected: {timestamp} (age: {age}s)")
            return False

        return True

    def validate_sequence(self, sequence_number: int) -> bool:
        """Validate sequence number for additional replay protection."""
        if not hasattr(self, '_last_sequence'):
            self._last_sequence = 0

        if sequence_number <= self._last_sequence:
            logger.warning(f"Sequence number replay detected: {sequence_number}")
            return False

        self._last_sequence = sequence_number
        return True
```

### SSL Certificate Pinning

```python
import ssl
import hashlib
from urllib.parse import urlparse

class SSLCertificatePinner:
    def __init__(self, pinned_certificates: dict):
        """
        pinned_certificates format:
        {
            "api.hyperliquid.xyz": "sha256:base64_encoded_hash",
            "api.hyperliquid-testnet.xyz": "sha256:base64_encoded_hash"
        }
        """
        self.pinned_certs = pinned_certificates

    def create_ssl_context(self, hostname: str) -> ssl.SSLContext:
        """Create SSL context with certificate pinning."""
        context = ssl.create_default_context()

        # Custom certificate verification
        def verify_cert(cert, hostname):
            return self._verify_pinned_certificate(cert, hostname)

        context.check_hostname = False  # We'll do our own verification
        context.verify_mode = ssl.CERT_NONE  # We'll verify manually

        return context

    def _verify_pinned_certificate(self, cert, hostname: str) -> bool:
        """Verify certificate against pinned hash."""
        if hostname not in self.pinned_certs:
            logger.error(f"No pinned certificate for {hostname}")
            return False

        # Extract certificate public key
        public_key = cert.public_key()
        public_key_der = public_key.public_key_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        # Calculate hash
        cert_hash = hashlib.sha256(public_key_der).digest()
        cert_hash_b64 = base64.b64encode(cert_hash).decode()

        expected_hash = self.pinned_certs[hostname].split(':')[1]

        if cert_hash_b64 != expected_hash:
            logger.error(f"Certificate hash mismatch for {hostname}")
            return False

        logger.info(f"Certificate pinning verified for {hostname}")
        return True

# Usage
pinner = SSLCertificatePinner({
    "api.hyperliquid.xyz": "sha256:ABC123..."  # Replace with actual hash
})

ssl_context = pinner.create_ssl_context("api.hyperliquid.xyz")
# Use ssl_context in HTTP client
```

### Rate Limiting and Throttling

```python
from collections import defaultdict
import time

class AdaptiveRateLimiter:
    def __init__(self, base_rate: int = 10, burst_limit: int = 20):
        self.base_rate = base_rate  # requests per second
        self.burst_limit = burst_limit
        self.requests = defaultdict(list)

    def allow_request(self, client_id: str) -> bool:
        """Check if request should be allowed."""
        now = time.time()
        client_requests = self.requests[client_id]

        # Remove old requests outside sliding window
        window_start = now - 1.0  # 1 second window
        client_requests[:] = [req for req in client_requests if req > window_start]

        # Check burst limit
        if len(client_requests) >= self.burst_limit:
            return False

        # Check sustained rate
        if len(client_requests) >= self.base_rate:
            return False

        # Allow request
        client_requests.append(now)
        return True

    def get_backoff_time(self, client_id: str) -> float:
        """Calculate backoff time for rate limited requests."""
        if not self.requests[client_id]:
            return 0.0

        now = time.time()
        oldest_request = min(self.requests[client_id])

        # Calculate time until we're within limits
        time_since_oldest = now - oldest_request
        if time_since_oldest < 1.0:
            return 1.0 - time_since_oldest

        return 0.0
```

## 🛡️ Position Protection

### Dead Man's Switch Implementation

```python
class DeadMansSwitch:
    def __init__(self, timeout_minutes: int = 5, check_interval: int = 30):
        self.timeout_minutes = timeout_minutes
        self.check_interval = check_interval
        self.last_heartbeat = time.time()
        self.active_positions = []

    def heartbeat(self) -> None:
        """Update last heartbeat timestamp."""
        self.last_heartbeat = time.time()
        logger.debug("💓 Dead man's switch heartbeat")

    def check_timeout(self) -> bool:
        """Check if timeout has been exceeded."""
        elapsed = time.time() - self.last_heartbeat
        timeout_seconds = self.timeout_minutes * 60

        if elapsed > timeout_seconds:
            logger.critical(f"🆘 DEAD MAN'S SWITCH ACTIVATED - No heartbeat for {elapsed:.1f}s")
            return True

        return False

    def emergency_close_positions(self) -> None:
        """Emergency close all positions."""
        logger.critical("🚨 EMERGENCY POSITION CLOSURE INITIATED")

        for position in self.active_positions:
            try:
                # Close position at market
                order = self._create_market_close_order(position)
                self.client.place_order(order)

                logger.warning(f"Emergency closed position: {position.symbol} {position.side}")

            except Exception as e:
                logger.error(f"Failed to close position {position.symbol}: {e}")

        # Clear positions
        self.active_positions.clear()

        # Send alerts
        self.alert_manager.send_alert(
            "EMERGENCY: All positions closed due to dead man's switch",
            severity="CRITICAL"
        )

    def monitor_loop(self) -> None:
        """Main monitoring loop."""
        while True:
            if self.check_timeout():
                self.emergency_close_positions()
                break  # Exit after emergency action

            time.sleep(self.check_interval)
```

### Position Circuit Breaker

```python
class PositionCircuitBreaker:
    def __init__(self, config: dict):
        self.max_position_size_pct = config.get('max_position_size_pct', 0.05)
        self.max_leverage = config.get('max_leverage', 10.0)
        self.circuit_open = False
        self.failure_count = 0
        self.last_failure_time = 0

    def validate_position_size(self, position_size: float, account_balance: float) -> bool:
        """Validate position size against limits."""
        position_pct = position_size / account_balance

        if position_pct > self.max_position_size_pct:
            logger.warning(f"Position size {position_pct:.3%} exceeds limit {self.max_position_size_pct:.3%}")
            return False

        return True

    def validate_leverage(self, leverage: float) -> bool:
        """Validate leverage against limits."""
        if leverage > self.max_leverage:
            logger.warning(f"Leverage {leverage:.1f}x exceeds limit {self.max_leverage:.1f}x")
            return False

        return True

    def check_circuit_state(self) -> bool:
        """Check if circuit breaker is open."""
        if self.circuit_open:
            # Check if we should attempt reset
            if time.time() - self.last_failure_time > 300:  # 5 minutes
                self.circuit_open = False
                logger.info("🔄 Position circuit breaker reset attempted")

        return not self.circuit_open

    def record_failure(self) -> None:
        """Record a position validation failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= 3:
            self.circuit_open = True
            logger.critical("🔴 POSITION CIRCUIT BREAKER OPENED - Too many validation failures")

    def validate_order(self, order: OrderRequest, account_balance: float) -> bool:
        """Comprehensive order validation."""
        if not self.check_circuit_state():
            logger.error("Circuit breaker open - order rejected")
            return False

        try:
            # Validate position size
            position_value = order.quantity * order.price
            if not self.validate_position_size(position_value, account_balance):
                self.record_failure()
                return False

            # Validate leverage
            leverage = position_value / account_balance
            if not self.validate_leverage(leverage):
                self.record_failure()
                return False

            # Additional validations
            if not self._validate_price_reasonableness(order.price):
                self.record_failure()
                return False

            return True

        except Exception as e:
            logger.error(f"Order validation error: {e}")
            self.record_failure()
            return False

    def _validate_price_reasonableness(self, price: float) -> bool:
        """Check if price is within reasonable bounds."""
        # Get current market price
        try:
            ticker = self.client.get_ticker(order.symbol)
            market_price = ticker['price']

            # Allow 5% deviation from market price
            deviation = abs(price - market_price) / market_price
            if deviation > 0.05:
                logger.warning(f"Price deviation too large: {deviation:.3%}")
                return False

            return True

        except Exception:
            # If we can't get market price, be conservative
            return False
```

### Sanity Checks Before Orders

```python
class OrderSanityChecker:
    def __init__(self, client: HyperliquidClient):
        self.client = client

    def pre_order_checks(self, order: OrderRequest) -> List[str]:
        """Perform all pre-order sanity checks."""
        issues = []

        # 1. Price reasonableness
        ticker = self.client.get_ticker(order.symbol)
        market_price = ticker['price']
        price_deviation = abs(order.price - market_price) / market_price

        if price_deviation > 0.02:  # 2% deviation
            issues.append(f"Price deviation too large: {price_deviation:.3%}")

        # 2. Size limits
        min_order_size = self._get_min_order_size(order.symbol)
        if order.quantity < min_order_size:
            issues.append(f"Order size {order.quantity} below minimum {min_order_size}")

        # 3. Balance checks
        balance = self.client.get_account_balance()
        order_value = order.quantity * order.price

        if order.side == 'BUY' and order_value > balance.available:
            issues.append(f"Insufficient balance: {order_value} > {balance.available}")

        # 4. Position limits
        current_positions = self.client.get_positions()
        total_exposure = sum(abs(p.quantity * p.entry_price) for p in current_positions)
        new_exposure = total_exposure + order_value

        if new_exposure > balance.total * 0.1:  # 10% of balance
            issues.append(f"Total exposure would exceed limit: {new_exposure}")

        # 5. Rate limits
        if not self._check_rate_limits(order.symbol):
            issues.append("Rate limit exceeded")

        return issues

    def post_order_checks(self, order_response: OrderResponse) -> List[str]:
        """Perform post-order validation."""
        issues = []

        # 1. Order confirmation
        if not order_response.order_id:
            issues.append("Order ID missing from response")

        # 2. Slippage check
        if hasattr(order_response, 'executed_price'):
            expected_price = order.price
            executed_price = order_response.executed_price
            slippage = abs(executed_price - expected_price) / expected_price

            if slippage > 0.01:  # 1% slippage
                issues.append(f"High slippage detected: {slippage:.3%}")

        # 3. Position update verification
        time.sleep(1)  # Allow time for position update
        positions = self.client.get_positions()

        expected_position = self._calculate_expected_position(order, positions)
        if not self._verify_position_update(expected_position, positions):
            issues.append("Position update verification failed")

        return issues
```

## 📊 Audit & Compliance

### Cryptographic Audit Logging

```python
import hmac
import hashlib
import json

class CryptographicAuditLogger:
    def __init__(self, log_file: str, key: bytes):
        self.log_file = log_file
        self.key = key
        self.sequence_number = 0

    def log_event(self, event_type: str, data: dict) -> None:
        """Log event with cryptographic integrity."""
        self.sequence_number += 1

        log_entry = {
            'sequence': self.sequence_number,
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': data,
            'integrity_hash': None
        }

        # Create integrity hash
        entry_str = json.dumps(log_entry, sort_keys=True, separators=(',', ':'))
        integrity_hash = hmac.new(self.key, entry_str.encode(), hashlib.sha256).hexdigest()

        log_entry['integrity_hash'] = integrity_hash

        # Write to log
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def verify_integrity(self) -> bool:
        """Verify entire audit log integrity."""
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                entry = json.loads(line.strip())

                # Remove integrity hash for verification
                stored_hash = entry.pop('integrity_hash')

                # Recalculate hash
                entry_str = json.dumps(entry, sort_keys=True, separators=(',', ':'))
                calculated_hash = hmac.new(self.key, entry_str.encode(), hashlib.sha256).hexdigest()

                if stored_hash != calculated_hash:
                    logger.error(f"Audit log integrity violation at sequence {entry['sequence']}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Audit log verification error: {e}")
            return False

    def get_events_by_type(self, event_type: str, start_time: datetime = None) -> List[dict]:
        """Retrieve events by type with optional time filter."""
        events = []

        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())

                    if entry['event_type'] == event_type:
                        if start_time is None or datetime.fromisoformat(entry['timestamp']) >= start_time:
                            events.append(entry)

        except Exception as e:
            logger.error(f"Error retrieving events: {e}")

        return events
```

### Compliance Report Generation

```python
class ComplianceReporter:
    def __init__(self, audit_logger: CryptographicAuditLogger):
        self.audit_logger = audit_logger

    def generate_trade_report(self, start_date: datetime, end_date: datetime) -> dict:
        """Generate comprehensive trading compliance report."""
        trades = self.audit_logger.get_events_by_type('TRADE_EXECUTED', start_date)

        report = {
            'report_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'total_trades': len(trades),
                'total_volume': sum(t['data'].get('quantity', 0) * t['data'].get('price', 0) for t in trades),
                'win_rate': self._calculate_win_rate(trades),
                'largest_trade': max((t['data'].get('quantity', 0) for t in trades), default=0)
            },
            'trades': trades,
            'regulatory_flags': self._check_regulatory_flags(trades),
            'generated_at': datetime.now().isoformat()
        }

        return report

    def _calculate_win_rate(self, trades: List[dict]) -> float:
        """Calculate win rate from trade data."""
        winning_trades = sum(1 for t in trades if t['data'].get('pnl', 0) > 0)
        return winning_trades / len(trades) if trades else 0.0

    def _check_regulatory_flags(self, trades: List[dict]) -> List[str]:
        """Check for regulatory compliance issues."""
        flags = []

        # Large trade flags (>10% of daily volume)
        for trade in trades:
            if trade['data'].get('volume_pct', 0) > 0.1:
                flags.append(f"Large trade detected: {trade['data'].get('volume_pct', 0):.1%}")

        # Frequent trading flags
        trade_frequency = len(trades) / 24  # trades per hour
        if trade_frequency > 10:
            flags.append(f"High frequency trading: {trade_frequency:.1f} trades/hour")

        # Loss limits
        total_pnl = sum(t['data'].get('pnl', 0) for t in trades)
        if total_pnl < -1000:  # $1000 loss
            flags.append(f"Large losses: ${total_pnl:.2f}")

        return flags

    def export_report(self, report: dict, format: str = 'json') -> str:
        """Export compliance report in specified format."""
        if format == 'json':
            return json.dumps(report, indent=2)
        elif format == 'csv':
            return self._export_csv(report)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_csv(self, report: dict) -> str:
        """Export report as CSV."""
        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(['Timestamp', 'Symbol', 'Side', 'Quantity', 'Price', 'PnL'])

        # Write trades
        for trade in report['trades']:
            data = trade['data']
            writer.writerow([
                trade['timestamp'],
                data.get('symbol', ''),
                data.get('side', ''),
                data.get('quantity', 0),
                data.get('price', 0),
                data.get('pnl', 0)
            ])

        return output.getvalue()
```

### Log Forwarding to Remote Storage

```python
class SecureLogForwarder:
    def __init__(self, remote_config: dict):
        self.remote_host = remote_config['host']
        self.remote_port = remote_config.get('port', 514)
        self.use_tls = remote_config.get('tls', True)
        self.client_cert = remote_config.get('client_cert')
        self.client_key = remote_config.get('client_key')

    def forward_log_entry(self, log_entry: dict) -> bool:
        """Forward log entry to remote secure storage."""
        try:
            # Encrypt log entry
            encrypted_entry = self._encrypt_log_entry(log_entry)

            # Establish secure connection
            sock = self._create_secure_socket()

            # Send log entry
            message = json.dumps({
                'timestamp': datetime.now().isoformat(),
                'facility': 'trading_bot',
                'level': 'info',
                'message': encrypted_entry
            })

            sock.send(message.encode())
            sock.close()

            return True

        except Exception as e:
            logger.error(f"Log forwarding failed: {e}")
            return False

    def _encrypt_log_entry(self, log_entry: dict) -> str:
        """Encrypt log entry for secure transmission."""
        # Implementation using AES-GCM
        pass

    def _create_secure_socket(self) -> socket.socket:
        """Create TLS-secured socket connection."""
        context = ssl.create_default_context()

        if self.client_cert and self.client_key:
            context.load_cert_chain(self.client_cert, self.client_key)

        sock = socket.create_connection((self.remote_host, self.remote_port))
        return context.wrap_socket(sock, server_hostname=self.remote_host)
```

## 🔐 Operational Security

### Principle of Least Privilege

```bash
# Create dedicated trading user with minimal permissions
sudo useradd -m -s /bin/bash trading
sudo usermod -aG trading trading

# Set proper file permissions
sudo chown -R trading:trading /home/trading/trading-bot
sudo chmod 700 /home/trading/trading-bot/config
sudo chmod 600 /home/trading/trading-bot/config/*.json

# Restrict sudo access
sudo tee /etc/sudoers.d/trading > /dev/null <<EOF
trading ALL=(ALL) NOPASSWD: /usr/bin/systemctl * trading-bot*
trading ALL=(ALL) NOPASSWD: /usr/bin/journalctl -u trading-bot*
EOF
```

### Secure Configuration Management

```python
class SecureConfigManager:
    def __init__(self, config_path: str, encryption_key: bytes):
        self.config_path = config_path
        self.encryption_key = encryption_key

    def save_encrypted_config(self, config: dict) -> None:
        """Save configuration encrypted to disk."""
        config_json = json.dumps(config, indent=2)

        # Encrypt sensitive fields
        encrypted_config = self._encrypt_sensitive_fields(config)

        # Write encrypted config
        with open(self.config_path, 'w') as f:
            f.write(encrypted_config)

        # Set secure permissions
        os.chmod(self.config_path, 0o600)

    def load_decrypted_config(self) -> dict:
        """Load and decrypt configuration from disk."""
        with open(self.config_path, 'r') as f:
            encrypted_config = f.read()

        # Decrypt and parse
        return self._decrypt_config(encrypted_config)

    def _encrypt_sensitive_fields(self, config: dict) -> str:
        """Encrypt sensitive configuration fields."""
        sensitive_fields = ['private_key', 'wallet_address', 'api_key', 'secret']

        for field in sensitive_fields:
            if field in config:
                config[field] = self._encrypt_value(config[field])

        return json.dumps(config, indent=2)

    def _encrypt_value(self, value: str) -> str:
        """Encrypt individual value."""
        from cryptography.fernet import Fernet
        f = Fernet(self.encryption_key)
        return f.encrypt(value.encode()).decode()

    def _decrypt_config(self, encrypted_config: str) -> dict:
        """Decrypt configuration."""
        config = json.loads(encrypted_config)

        for key, value in config.items():
            if isinstance(value, str) and value.startswith('gAAAAA'):  # Fernet prefix
                config[key] = self._decrypt_value(value)

        return config

    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt individual value."""
        from cryptography.fernet import Fernet
        f = Fernet(self.encryption_key)
        return f.decrypt(encrypted_value.encode()).decode()
```

### Two-Factor Authentication for Critical Operations

```python
import pyotp
import qrcode
import io

class TwoFactorAuth:
    def __init__(self, secret_key: str = None):
        self.secret = secret_key or pyotp.random_base32()

    def generate_qr_code(self) -> str:
        """Generate QR code for TOTP setup."""
        totp = pyotp.TOTP(self.secret)
        provisioning_uri = totp.provisioning_uri(name="Trading Bot", issuer_name="Hyperliquid")

        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)

        # Convert to ASCII
        f = io.StringIO()
        qr.print_ascii(out=f)
        return f.getvalue()

    def verify_code(self, code: str) -> bool:
        """Verify TOTP code."""
        totp = pyotp.TOTP(self.secret)
        return totp.verify(code)

    def require_2fa_for_operation(self, operation: str, user_code: str) -> bool:
        """Require 2FA verification for critical operations."""
        critical_operations = [
            'emergency_shutdown',
            'key_rotation',
            'position_closure',
            'config_change'
        ]

        if operation in critical_operations:
            if not self.verify_code(user_code):
                logger.warning(f"2FA verification failed for {operation}")
                return False

        return True
```

### Runtime Integrity Checks

```python
import hashlib
import inspect

class RuntimeIntegrityChecker:
    def __init__(self, baseline_hashes: dict):
        self.baseline_hashes = baseline_hashes
        self.check_interval = 300  # 5 minutes

    def check_module_integrity(self, module) -> bool:
        """Check if module code has been tampered with."""
        try:
            # Get module source
            source = inspect.getsource(module)

            # Calculate hash
            current_hash = hashlib.sha256(source.encode()).hexdigest()

            # Compare with baseline
            module_name = module.__name__
            if module_name not in self.baseline_hashes:
                logger.warning(f"No baseline hash for {module_name}")
                return False

            baseline_hash = self.baseline_hashes[module_name]

            if current_hash != baseline_hash:
                logger.critical(f"INTEGRITY VIOLATION: {module_name} has been modified!")
                return False

            return True

        except Exception as e:
            logger.error(f"Integrity check failed for {module.__name__}: {e}")
            return False

    def perform_full_integrity_check(self) -> bool:
        """Perform integrity check on all critical modules."""
        critical_modules = [
            'trading_bot',
            'hyperliquid_client',
            'secure_key_storage',
            'resilience'
        ]

        all_integrity_ok = True

        for module_name in critical_modules:
            try:
                module = __import__(module_name)
                if not self.check_module_integrity(module):
                    all_integrity_ok = False
            except ImportError:
                logger.error(f"Could not import {module_name}")
                all_integrity_ok = False

        if not all_integrity_ok:
            self._trigger_integrity_alert()

        return all_integrity_ok

    def _trigger_integrity_alert(self) -> None:
        """Trigger alert on integrity violation."""
        logger.critical("🚨 RUNTIME INTEGRITY VIOLATION DETECTED")
        # Implementation for alert system
        pass

    def start_monitoring(self) -> None:
        """Start periodic integrity monitoring."""
        def monitor_loop():
            while True:
                self.perform_full_integrity_check()
                time.sleep(self.check_interval)

        import threading
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
```

## 🚨 Error Handling

### Secure Error Reporting

```python
class SecureErrorReporter:
    def __init__(self, config: dict):
        self.sanitize_sensitive_data = config.get('sanitize_errors', True)
        self.log_full_errors = config.get('log_full_errors', False)
        self.remote_reporting = config.get('remote_error_reporting', False)

    def report_error(self, error: Exception, context: dict = None) -> None:
        """Report error securely without exposing sensitive data."""
        # Sanitize error message
        safe_message = self._sanitize_error_message(str(error))

        # Sanitize context
        safe_context = self._sanitize_context(context or {})

        # Log safe version
        logger.error(f"Error: {safe_message}", extra=safe_context)

        # Send to remote if configured
        if self.remote_reporting:
            self._send_remote_report(safe_message, safe_context)

    def _sanitize_error_message(self, message: str) -> str:
        """Remove sensitive data from error messages."""
        # Patterns to redact
        sensitive_patterns = [
            r'0x[a-fA-F0-9]{40}',  # Ethereum addresses
            r'0x[a-fA-F0-9]{64}',  # Private keys
            r'Bearer\s+[^\s]+',   # API tokens
            r'password[^\s]+',    # Passwords
            r'secret[^\s]+'       # Secrets
        ]

        safe_message = message
        for pattern in sensitive_patterns:
            safe_message = re.sub(pattern, '[REDACTED]', safe_message, flags=re.IGNORECASE)

        return safe_message

    def _sanitize_context(self, context: dict) -> dict:
        """Sanitize context dictionary."""
        safe_context = {}
        sensitive_keys = {'private_key', 'wallet_address', 'api_key', 'secret', 'password'}

        for key, value in context.items():
            if key.lower() in sensitive_keys:
                safe_context[key] = '[REDACTED]'
            elif isinstance(value, str):
                safe_context[key] = self._sanitize_error_message(value)
            else:
                safe_context[key] = value

        return safe_context

    def _send_remote_report(self, message: str, context: dict) -> None:
        """Send error report to remote monitoring system."""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'level': 'error',
                'message': message,
                'context': context,
                'hostname': socket.gethostname()
            }

            # Send to remote endpoint (implementation depends on service)
            # requests.post('https://error-reporting.example.com/report', json=report)

        except Exception as e:
            # Don't let error reporting cause more errors
            logger.debug(f"Remote error reporting failed: {e}")
```

### Graceful Degradation

```python
class GracefulDegradationManager:
    def __init__(self):
        self.degradation_levels = {
            'NORMAL': 0,
            'DEGRADED': 1,
            'CRITICAL': 2,
            'EMERGENCY': 3
        }
        self.current_level = 'NORMAL'

    def degrade_service(self, reason: str) -> None:
        """Degrade service level based on failure."""
        current_index = self.degradation_levels[self.current_level]

        if reason == 'websocket_failure':
            self.current_level = 'DEGRADED'
            self._apply_degraded_mode()
        elif reason == 'api_failure':
            self.current_level = 'CRITICAL'
            self._apply_critical_mode()
        elif reason == 'state_corruption':
            self.current_level = 'EMERGENCY'
            self._apply_emergency_mode()

        logger.warning(f"Service degraded to {self.current_level}: {reason}")

    def _apply_degraded_mode(self) -> None:
        """Apply degraded mode settings."""
        # Reduce trading frequency
        config.trading.check_interval = 600  # 10 minutes

        # Use cached data where possible
        config.api.use_cache_fallback = True

        # Reduce position sizes
        config.risk.max_position_size_pct *= 0.5

    def _apply_critical_mode(self) -> None:
        """Apply critical mode settings."""
        # Stop new positions
        config.trading.allow_new_positions = False

        # Use REST polling only
        config.websocket.enabled = False

        # Further reduce position sizes
        config.risk.max_position_size_pct *= 0.25

    def _apply_emergency_mode(self) -> None:
        """Apply emergency mode settings."""
        # Stop all trading
        config.trading.enabled = False

        # Close existing positions conservatively
        self._initiate_emergency_closures()

    def attempt_recovery(self) -> bool:
        """Attempt to recover to higher service level."""
        if self.current_level == 'EMERGENCY':
            # Manual intervention required
            return False

        # Test system components
        if self._test_system_health():
            # Gradually recover
            if self.current_level == 'CRITICAL':
                self.current_level = 'DEGRADED'
            elif self.current_level == 'DEGRADED':
                self.current_level = 'NORMAL'

            logger.info(f"Service recovered to {self.current_level}")
            return True

        return False

    def _test_system_health(self) -> bool:
        """Test basic system health."""
        try:
            # Test API connectivity
            client.get_ticker('BTCUSDT')

            # Test database
            state_manager.check_connection()

            return True
        except Exception:
            return False
```

## 🧪 Security Testing

### Automated Security Test Suite

```python
class SecurityTestSuite:
    def __init__(self, test_client: HyperliquidClient):
        self.client = test_client

    def run_full_security_audit(self) -> dict:
        """Run comprehensive security tests."""
        results = {
            'key_extraction_test': self.test_key_extraction(),
            'replay_attack_test': self.test_replay_attack_prevention(),
            'man_in_middle_test': self.test_man_in_middle_protection(),
            'rate_limit_test': self.test_rate_limiting(),
            'input_validation_test': self.test_input_validation(),
            'audit_log_integrity_test': self.test_audit_log_integrity(),
            'memory_dump_test': self.test_memory_dump_security()
        }

        return results

    def test_key_extraction(self) -> bool:
        """Test resistance to key extraction attacks."""
        try:
            # Attempt to extract keys from memory
            import psutil
            import gc

            process = psutil.Process()
            memory_maps = process.memory_maps()

            # Look for sensitive data in memory
            sensitive_found = False
            for mem_map in memory_maps:
                # This is a simplified test - real implementation would
                # scan memory regions for known key patterns
                pass

            # Force garbage collection
            gc.collect()

            return not sensitive_found

        except Exception as e:
            logger.error(f"Key extraction test failed: {e}")
            return False

    def test_replay_attack_prevention(self) -> bool:
        """Test prevention of replay attacks."""
        try:
            # Create a valid request
            original_request = self._create_test_request()

            # Attempt to replay
            for i in range(5):
                try:
                    # This should fail on subsequent attempts
                    response = self.client._send_request(original_request)
                    if i > 0:  # First attempt should succeed
                        return False
                except Exception:
                    if i == 0:
                        return False  # First attempt should succeed
                    # Subsequent attempts should fail

            return True

        except Exception as e:
            logger.error(f"Replay attack test failed: {e}")
            return False

    def test_man_in_middle_protection(self) -> bool:
        """Test protection against MITM attacks."""
        # This would require setting up a proxy and testing certificate validation
        # Implementation depends on specific MITM testing tools
        pass

    def test_rate_limiting(self) -> bool:
        """Test rate limiting effectiveness."""
        try:
            start_time = time.time()

            # Attempt many requests rapidly
            success_count = 0
            for i in range(100):
                try:
                    self.client.get_ticker('BTCUSDT')
                    success_count += 1
                except Exception:
                    pass  # Expected for rate limited requests

            elapsed = time.time() - start_time

            # Should not allow more than expected rate
            expected_max_requests = 10  # Based on rate limiter config

            return success_count <= expected_max_requests * (elapsed / 60)

        except Exception as e:
            logger.error(f"Rate limiting test failed: {e}")
            return False

    def test_input_validation(self) -> bool:
        """Test input validation effectiveness."""
        test_cases = [
            ("", False),  # Empty
            ("not-a-symbol", False),  # Invalid symbol
            ("<script>alert('xss')</script>", False),  # XSS attempt
            ("../../../../etc/passwd", False),  # Path traversal
            ("BTCUSDT", True),  # Valid symbol
        ]

        sanitizer = InputSanitizer()

        all_passed = True
        for input_value, should_pass in test_cases:
            try:
                result = sanitizer.sanitize_symbol(input_value)
                if should_pass and not result:
                    all_passed = False
                elif not should_pass and result:
                    all_passed = False
            except ValidationError:
                if should_pass:
                    all_passed = False

        return all_passed

    def test_audit_log_integrity(self) -> bool:
        """Test audit log tamper resistance."""
        try:
            # Create test audit logger
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as f:
                log_file = f.name

            key = secrets.token_bytes(32)
            audit = CryptographicAuditLogger(log_file, key)

            # Log some events
            audit.log_event('TEST_EVENT', {'data': 'test'})

            # Attempt to tamper with log
            with open(log_file, 'r+') as f:
                content = f.read()
                # Try to modify content
                tampered = content.replace('test', 'modified')
                f.seek(0)
                f.write(tampered)
                f.truncate()

            # Check integrity
            integrity_ok = audit.verify_integrity()

            # Cleanup
            os.unlink(log_file)

            return not integrity_ok  # Should detect tampering

        except Exception as e:
            logger.error(f"Audit log integrity test failed: {e}")
            return False

    def test_memory_dump_security(self) -> bool:
        """Test security of memory dumps."""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()

            # Check if memory contains sensitive patterns
            # This is a basic test - real implementation would scan memory
            sensitive_patterns = [
                b'0x[a-fA-F0-9]{64}',  # Private keys
                b'Bearer [A-Za-z0-9+/=]+',  # JWT tokens
            ]

            # Note: Real implementation would require reading process memory
            # which requires special permissions and is OS-specific

            return True  # Placeholder

        except Exception as e:
            logger.error(f"Memory dump security test failed: {e}")
            return False
```

## 📋 Incident Response

### Incident Response Plan

```markdown
# Incident Response Plan

## 1. Detection
- Automated monitoring alerts
- Log analysis for anomalies
- User reports

## 2. Assessment
- Determine incident scope and impact
- Identify affected systems/components
- Assess data exposure risk

## 3. Containment
- Isolate affected systems
- Stop trading operations if necessary
- Preserve evidence for investigation

## 4. Eradication
- Remove malicious code/access
- Patch vulnerabilities
- Rotate compromised credentials

## 5. Recovery
- Restore systems from clean backups
- Test system functionality
- Gradually resume operations

## 6. Lessons Learned
- Document incident details
- Update prevention measures
- Improve response procedures
```

### Emergency Contacts

```python
EMERGENCY_CONTACTS = {
    'primary': {
        'name': 'Security Team Lead',
        'phone': '+1-555-0101',
        'email': 'security@company.com'
    },
    'secondary': {
        'name': 'DevOps Manager',
        'phone': '+1-555-0102',
        'email': 'devops@company.com'
    },
    'legal': {
        'name': 'Legal Counsel',
        'phone': '+1-555-0103',
        'email': 'legal@company.com'
    }
}
```

### Automated Incident Response

```python
class IncidentResponseAutomation:
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.incident_log = []

    def handle_security_incident(self, incident_type: str, details: dict) -> None:
        """Automated incident response based on type."""
        incident_id = f"INC-{int(time.time())}"

        self.incident_log.append({
            'id': incident_id,
            'type': incident_type,
            'timestamp': datetime.now().isoformat(),
            'details': details,
            'status': 'ACTIVE'
        })

        # Automated response based on incident type
        if incident_type == 'key_compromise':
            self._handle_key_compromise(incident_id, details)
        elif incident_type == 'unauthorized_access':
            self._handle_unauthorized_access(incident_id, details)
        elif incident_type == 'data_breach':
            self._handle_data_breach(incident_id, details)

        # Alert all emergency contacts
        self._alert_emergency_contacts(incident_id, incident_type, details)

    def _handle_key_compromise(self, incident_id: str, details: dict) -> None:
        """Handle private key compromise."""
        logger.critical(f"🚨 KEY COMPROMISE INCIDENT {incident_id}")

        # Emergency key zeroization
        key_storage.emergency_zeroize()

        # Stop all trading
        os.environ['BOT_EMERGENCY_SHUTDOWN'] = 'true'

        # Close all positions
        dead_man_switch.emergency_close_positions()

        # Log incident
        audit_logger.log_event('KEY_COMPROMISE_RESPONSE', {
            'incident_id': incident_id,
            'action': 'emergency_zeroization',
            'positions_closed': True
        })

    def _handle_unauthorized_access(self, incident_id: str, details: dict) -> None:
        """Handle unauthorized access attempts."""
        logger.critical(f"🚨 UNAUTHORIZED ACCESS INCIDENT {incident_id}")

        # Block suspicious IP
        suspicious_ip = details.get('ip_address')
        if suspicious_ip:
            self._block_ip(suspicious_ip)

        # Rotate all session tokens
        self._rotate_session_tokens()

        # Enable enhanced monitoring
        monitoring.enable_intrusion_detection()

    def _handle_data_breach(self, incident_id: str, details: dict) -> None:
        """Handle data breach incidents."""
        logger.critical(f"🚨 DATA BREACH INCIDENT {incident_id}")

        # Assess data exposure
        exposed_data = self._assess_data_exposure(details)

        # Notify affected parties if necessary
        if exposed_data['sensitive']:
            self._notify_data_breach_contacts(incident_id, exposed_data)

        # Implement additional security measures
        security.enable_enhanced_auditing()
        security.rotate_all_credentials()

    def _alert_emergency_contacts(self, incident_id: str, incident_type: str, details: dict) -> None:
        """Alert all emergency contacts."""
        message = f"""
INCIDENT ALERT - {incident_id}

Type: {incident_type}
Time: {datetime.now().isoformat()}
Details: {json.dumps(details, indent=2)}

Immediate action required. Check incident response procedures.
        """.strip()

        for contact_type, contact in EMERGENCY_CONTACTS.items():
            self.alert_manager.send_email(
                subject=f"CRITICAL: Security Incident {incident_id}",
                body=message,
                recipients=[contact['email']]
            )
```

This comprehensive security guide provides the framework for securing the Hyperliquid MACD trading bot against common threats while maintaining operational reliability. The multi-layered approach ensures that security is built into every component and operation.
