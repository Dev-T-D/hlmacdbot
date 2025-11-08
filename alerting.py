"""
Multi-Channel Alerting System for Trading Bot

Provides comprehensive alerting across multiple channels:
- Email (SMTP)
- Telegram bot
- Discord webhook
- PagerDuty (for critical alerts)

Features:
- Alert deduplication to prevent spam
- Escalation policies
- Alert history and status tracking
- Configurable alert thresholds
- Integration with metrics and logging
"""

import json
import smtplib
import threading
import time
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging

try:
    import requests
except ImportError:
    requests = None

try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    Bot = None
    TelegramError = Exception
    TELEGRAM_AVAILABLE = False

try:
    from discord_webhook import DiscordWebhook, DiscordEmbed
    DISCORD_AVAILABLE = True
except ImportError:
    DiscordWebhook = None
    DiscordEmbed = None
    DISCORD_AVAILABLE = False


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    source: str
    timestamp: datetime
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None
    status: AlertStatus = AlertStatus.ACTIVE
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    last_notification: Optional[datetime] = None
    notification_count: int = 0

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        if self.acknowledged_at:
            data['acknowledged_at'] = self.acknowledged_at.isoformat()
        if self.last_notification:
            data['last_notification'] = self.last_notification.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create alert from dictionary."""
        data_copy = data.copy()
        data_copy['severity'] = AlertSeverity(data_copy['severity'])
        data_copy['status'] = AlertStatus(data_copy['status'])
        data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])
        if 'resolved_at' in data_copy and data_copy['resolved_at']:
            data_copy['resolved_at'] = datetime.fromisoformat(data_copy['resolved_at'])
        if 'acknowledged_at' in data_copy and data_copy['acknowledged_at']:
            data_copy['acknowledged_at'] = datetime.fromisoformat(data_copy['acknowledged_at'])
        if 'last_notification' in data_copy and data_copy['last_notification']:
            data_copy['last_notification'] = datetime.fromisoformat(data_copy['last_notification'])
        return cls(**data_copy)


class AlertChannel:
    """Base class for alert channels."""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    def send_alert(self, alert: Alert) -> bool:
        """Send alert through this channel. Returns success status."""
        raise NotImplementedError

    def test_connection(self) -> bool:
        """Test connection to alert channel. Returns success status."""
        raise NotImplementedError


class EmailChannel(AlertChannel):
    """Email alert channel using SMTP."""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int = 587,
        smtp_username: str = "",
        smtp_password: str = "",
        from_email: str = "trading-bot@example.com",
        to_emails: List[str] = None,
        enabled: bool = True,
        use_tls: bool = True
    ):
        super().__init__("email", enabled)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.from_email = from_email
        self.to_emails = to_emails or []
        self.use_tls = use_tls

    def send_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        if not self.enabled or not self.to_emails:
            return True

        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"

            # Add body
            body = f"""
Trading Bot Alert
=================

Severity: {alert.severity.value.upper()}
Title: {alert.title}
Message: {alert.message}
Source: {alert.source}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

Tags: {json.dumps(alert.tags, indent=2)}
Metadata: {json.dumps(alert.metadata, indent=2)}

Status: {alert.status.value}
            """

            msg.attach(MIMEText(body, 'plain'))

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()

            if self.smtp_username and self.smtp_password:
                server.login(self.smtp_username, self.smtp_password)

            server.sendmail(self.from_email, self.to_emails, msg.as_string())
            server.quit()

            logger.info(f"Email alert sent to {len(self.to_emails)} recipients")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def test_connection(self) -> bool:
        """Test SMTP connection."""
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()

            if self.smtp_username and self.smtp_password:
                server.login(self.smtp_username, self.smtp_password)

            server.quit()
            return True
        except Exception as e:
            logger.error(f"SMTP connection test failed: {e}")
            return False


class TelegramChannel(AlertChannel):
    """Telegram bot alert channel."""

    def __init__(self, bot_token: str, chat_ids: List[str] = None, enabled: bool = True):
        super().__init__("telegram", enabled)
        if not TELEGRAM_AVAILABLE:
            logger.warning("Telegram bot library not available. Install python-telegram-bot")
            self.enabled = False
            return

        self.bot_token = bot_token
        self.chat_ids = chat_ids or []
        self.bot = Bot(token=bot_token) if bot_token else None

    def send_alert(self, alert: Alert) -> bool:
        """Send alert via Telegram."""
        if not self.enabled or not self.bot or not self.chat_ids:
            return True

        try:
            emoji = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.ERROR: "❌",
                AlertSeverity.CRITICAL: "🚨"
            }.get(alert.severity, "📢")

            message = f"""{emoji} **{alert.severity.value.upper()} ALERT**

**{alert.title}**
{alert.message}

*Source:* {alert.source}
*Time:* {alert.timestamp.strftime('%H:%M:%S UTC')}

*Tags:* {', '.join(f'{k}={v}' for k, v in alert.tags.items())}
"""

            success_count = 0
            for chat_id in self.chat_ids:
                try:
                    self.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode='Markdown',
                        disable_web_page_preview=True
                    )
                    success_count += 1
                except TelegramError as e:
                    logger.error(f"Failed to send Telegram alert to {chat_id}: {e}")

            logger.info(f"Telegram alert sent to {success_count}/{len(self.chat_ids)} chats")
            return success_count > 0

        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False

    def test_connection(self) -> bool:
        """Test Telegram bot connection."""
        if not self.bot:
            return False

        try:
            # Try to get bot info
            self.bot.get_me()
            return True
        except Exception as e:
            logger.error(f"Telegram bot connection test failed: {e}")
            return False


class DiscordChannel(AlertChannel):
    """Discord webhook alert channel."""

    def __init__(self, webhook_url: str, enabled: bool = True):
        super().__init__("discord", enabled)
        if not DISCORD_AVAILABLE:
            logger.warning("Discord webhook library not available. Install discord-webhook")
            self.enabled = False
            return

        self.webhook_url = webhook_url
        self.webhook = DiscordWebhook(url=webhook_url) if webhook_url else None

    def send_alert(self, alert: Alert) -> bool:
        """Send alert via Discord webhook."""
        if not self.enabled or not self.webhook:
            return True

        try:
            # Create embed
            embed = DiscordEmbed(
                title=f"{alert.severity.value.upper()} ALERT: {alert.title}",
                description=alert.message,
                color={
                    AlertSeverity.INFO: 0x3498db,
                    AlertSeverity.WARNING: 0xf39c12,
                    AlertSeverity.ERROR: 0xe74c3c,
                    AlertSeverity.CRITICAL: 0x992d22
                }.get(alert.severity, 0x95a5a6)
            )

            embed.add_embed_field(name="Source", value=alert.source, inline=True)
            embed.add_embed_field(name="Time", value=alert.timestamp.strftime('%H:%M:%S UTC'), inline=True)
            embed.add_embed_field(name="Status", value=alert.status.value, inline=True)

            if alert.tags:
                embed.add_embed_field(
                    name="Tags",
                    value=', '.join(f'{k}: {v}' for k, v in alert.tags.items()),
                    inline=False
                )

            if alert.metadata:
                embed.add_embed_field(
                    name="Metadata",
                    value=json.dumps(alert.metadata, indent=2),
                    inline=False
                )

            embed.set_footer(text=f"Alert ID: {alert.alert_id}")

            self.webhook.add_embed(embed)
            response = self.webhook.execute()

            success = response.status_code == 200
            if success:
                logger.info("Discord alert sent successfully")
            else:
                logger.error(f"Discord alert failed with status {response.status_code}")

            return success

        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False

    def test_connection(self) -> bool:
        """Test Discord webhook connection."""
        if not self.webhook:
            return False

        try:
            # Try to send a test message
            test_webhook = DiscordWebhook(url=self.webhook_url, content="Test connection")
            response = test_webhook.execute()
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Discord webhook connection test failed: {e}")
            return False


class PagerDutyChannel(AlertChannel):
    """PagerDuty alert channel for critical incidents."""

    def __init__(self, routing_key: str, enabled: bool = True):
        super().__init__("pagerduty", enabled)
        self.routing_key = routing_key
        self.api_url = "https://events.pagerduty.com/v2/enqueue"

    def send_alert(self, alert: Alert) -> bool:
        """Send alert to PagerDuty."""
        if not self.enabled or not requests:
            return True

        try:
            # Map severity to PagerDuty
            severity = {
                AlertSeverity.INFO: "info",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "error",
                AlertSeverity.CRITICAL: "critical"
            }.get(alert.severity, "critical")

            # Create PagerDuty event
            event = {
                "routing_key": self.routing_key,
                "event_action": "trigger" if alert.status == AlertStatus.ACTIVE else "resolve",
                "dedup_key": alert.alert_id,
                "payload": {
                    "summary": alert.title,
                    "source": alert.source,
                    "severity": severity,
                    "timestamp": alert.timestamp.isoformat(),
                    "component": "trading_bot",
                    "group": "trading_system",
                    "class": "alert",
                    "custom_details": {
                        "message": alert.message,
                        "tags": alert.tags,
                        "metadata": alert.metadata
                    }
                }
            }

            response = requests.post(
                self.api_url,
                json=event,
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            success = response.status_code == 202
            if success:
                logger.info("PagerDuty alert sent successfully")
            else:
                logger.error(f"PagerDuty alert failed with status {response.status_code}: {response.text}")

            return success

        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False

    def test_connection(self) -> bool:
        """Test PagerDuty connection with a test event."""
        if not requests:
            return False

        try:
            # Send a test resolve event (which should be ignored but test connectivity)
            event = {
                "routing_key": self.routing_key,
                "event_action": "resolve",
                "dedup_key": "test_connection",
                "payload": {
                    "summary": "Connection test",
                    "source": "trading_bot",
                    "severity": "info"
                }
            }

            response = requests.post(
                self.api_url,
                json=event,
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            return response.status_code in [200, 202, 400]  # 400 means invalid key, but connection works
        except Exception as e:
            logger.error(f"PagerDuty connection test failed: {e}")
            return False


class AlertManager:
    """
    Central alert management system.

    Handles alert creation, routing, deduplication, and escalation.
    """

    def __init__(self, config: Dict[str, Any] = None, storage_file: str = "data/alerts.json"):
        self.config = config or {}
        self.storage_file = Path(storage_file)
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)

        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.max_history = self.config.get('max_history', 1000)

        # Channels
        self.channels: Dict[str, AlertChannel] = {}
        self._setup_channels()

        # Alert rules
        self.alert_rules = self._load_alert_rules()

        # Deduplication
        self.dedup_window = timedelta(minutes=self.config.get('dedup_window_minutes', 5))
        self.max_notifications_per_alert = self.config.get('max_notifications_per_alert', 5)

        # Background thread for alert processing
        self._running = False
        self._thread = None
        self._alert_queue: List[Alert] = []
        self._queue_lock = threading.Lock()

        logger.info("Alert manager initialized")

    def _setup_channels(self):
        """Set up alert channels from configuration."""
        channels_config = self.config.get('channels', {})

        # Email channel
        if channels_config.get('email', {}).get('enabled', False):
            email_config = channels_config['email']
            self.channels['email'] = EmailChannel(
                smtp_server=email_config['smtp_server'],
                smtp_port=email_config.get('smtp_port', 587),
                smtp_username=email_config.get('smtp_username', ''),
                smtp_password=email_config.get('smtp_password', ''),
                from_email=email_config.get('from_email', 'trading-bot@example.com'),
                to_emails=email_config.get('to_emails', []),
                use_tls=email_config.get('use_tls', True)
            )

        # Telegram channel
        if channels_config.get('telegram', {}).get('enabled', False):
            telegram_config = channels_config['telegram']
            self.channels['telegram'] = TelegramChannel(
                bot_token=telegram_config['bot_token'],
                chat_ids=telegram_config.get('chat_ids', [])
            )

        # Discord channel
        if channels_config.get('discord', {}).get('enabled', False):
            discord_config = channels_config['discord']
            self.channels['discord'] = DiscordChannel(
                webhook_url=discord_config['webhook_url']
            )

        # PagerDuty channel
        if channels_config.get('pagerduty', {}).get('enabled', False):
            pd_config = channels_config['pagerduty']
            self.channels['pagerduty'] = PagerDutyChannel(
                routing_key=pd_config['routing_key']
            )

    def _load_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined alert rules."""
        return {
            # Critical alerts - immediate action required
            "bot_crashed": {
                "severity": AlertSeverity.CRITICAL,
                "channels": ["email", "telegram", "discord", "pagerduty"],
                "dedup_window": 1,  # 1 minute
                "max_notifications": 3
            },
            "websocket_disconnected": {
                "severity": AlertSeverity.CRITICAL,
                "channels": ["telegram", "discord"],
                "dedup_window": 2,  # 2 minutes
                "max_notifications": 5
            },
            "position_stop_loss_hit": {
                "severity": AlertSeverity.CRITICAL,
                "channels": ["telegram", "discord", "pagerduty"],
                "dedup_window": 0,  # No deduplication
                "max_notifications": 1
            },
            "daily_loss_limit_reached": {
                "severity": AlertSeverity.CRITICAL,
                "channels": ["email", "telegram", "discord", "pagerduty"],
                "dedup_window": 60,  # 1 hour
                "max_notifications": 2
            },
            "api_authentication_failed": {
                "severity": AlertSeverity.CRITICAL,
                "channels": ["email", "telegram", "discord", "pagerduty"],
                "dedup_window": 5,  # 5 minutes
                "max_notifications": 3
            },
            "insufficient_balance": {
                "severity": AlertSeverity.CRITICAL,
                "channels": ["email", "telegram", "discord", "pagerduty"],
                "dedup_window": 15,  # 15 minutes
                "max_notifications": 3
            },

            # Warning alerts - review within 30 minutes
            "api_latency_high": {
                "severity": AlertSeverity.WARNING,
                "channels": ["telegram", "discord"],
                "dedup_window": 30,  # 30 minutes
                "max_notifications": 2
            },
            "memory_usage_high": {
                "severity": AlertSeverity.WARNING,
                "channels": ["telegram"],
                "dedup_window": 15,  # 15 minutes
                "max_notifications": 3
            },
            "win_rate_low": {
                "severity": AlertSeverity.WARNING,
                "channels": ["telegram", "discord"],
                "dedup_window": 60,  # 1 hour
                "max_notifications": 1
            },
            "daily_loss_limit_approaching": {
                "severity": AlertSeverity.WARNING,
                "channels": ["telegram"],
                "dedup_window": 30,  # 30 minutes
                "max_notifications": 2
            },
            "unusual_market_conditions": {
                "severity": AlertSeverity.WARNING,
                "channels": ["telegram", "discord"],
                "dedup_window": 10,  # 10 minutes
                "max_notifications": 2
            },

            # Info alerts - daily digest
            "daily_trading_summary": {
                "severity": AlertSeverity.INFO,
                "channels": ["email"],
                "dedup_window": 1440,  # 24 hours
                "max_notifications": 1
            },
            "new_balance_high": {
                "severity": AlertSeverity.INFO,
                "channels": ["telegram"],
                "dedup_window": 1440,  # 24 hours
                "max_notifications": 1
            },
            "strategy_signal_generated": {
                "severity": AlertSeverity.INFO,
                "channels": ["telegram"],
                "dedup_window": 60,  # 1 hour
                "max_notifications": 5
            }
        }

    def send_alert(
        self,
        alert_type: str,
        title: str,
        message: str,
        source: str = "trading_bot",
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        severity: Optional[AlertSeverity] = None,
        force: bool = False
    ) -> Optional[str]:
        """
        Send an alert with deduplication and routing.

        Args:
            alert_type: Type of alert (matches alert_rules keys)
            title: Alert title
            message: Alert message
            source: Alert source
            tags: Additional tags
            metadata: Additional metadata
            force: Force sending even if deduplicated

        Returns:
            Alert ID if alert was sent/queued, None if deduplicated
        """
        # Get alert rule
        rule = self.alert_rules.get(alert_type, {
            "severity": AlertSeverity.WARNING,
            "channels": ["telegram"],
            "dedup_window": 5,
            "max_notifications": 3
        })

        # Override severity if specified
        if severity:
            rule["severity"] = severity

        alert_id = f"{alert_type}_{int(time.time())}"

        # Check for deduplication
        if not force and self._should_deduplicate(alert_type, rule):
            logger.debug(f"Alert {alert_type} deduplicated")
            return None

        # Create alert
        alert = Alert(
            alert_id=alert_id,
            title=title,
            message=message,
            severity=rule["severity"],
            source=source,
            timestamp=datetime.now(timezone.utc),
            tags=tags or {},
            metadata=metadata or {}
        )

        # Store active alert
        self.active_alerts[alert_id] = alert

        # Queue for processing
        with self._queue_lock:
            self._alert_queue.append(alert)

        logger.info(f"Alert queued: {alert_type} - {title}")
        return alert_id

    def _should_deduplicate(self, alert_type: str, rule: Dict[str, Any]) -> bool:
        """Check if alert should be deduplicated."""
        dedup_window = timedelta(minutes=rule.get("dedup_window", 5))
        cutoff_time = datetime.now(timezone.utc) - dedup_window

        # Check recent alerts of same type
        for alert in self.active_alerts.values():
            if (alert.title == f"{alert_type}_" or alert.title.startswith(f"{alert_type}:")) and \
               alert.timestamp > cutoff_time and \
               alert.notification_count >= rule.get("max_notifications", 3):
                return True

        return False

    def resolve_alert(self, alert_id: str, resolution_message: str = "") -> bool:
        """Resolve an active alert."""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now(timezone.utc)
        alert.metadata['resolution'] = resolution_message

        # Move to history
        self.alert_history.append(alert)
        del self.active_alerts[alert_id]

        # Send resolution notification
        self._send_resolution_notification(alert)

        # Trim history
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]

        # Save state
        self._save_alerts()

        logger.info(f"Alert resolved: {alert_id}")
        return True

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now(timezone.utc)

        # Save state
        self._save_alerts()

        logger.info(f"Alert acknowledged: {alert_id}")
        return True

    def _send_resolution_notification(self, alert: Alert):
        """Send resolution notification through channels."""
        rule = self.alert_rules.get(alert.title.split('_')[0], {})
        channels = rule.get("channels", ["telegram"])

        resolution_message = f"✅ RESOLVED: {alert.title}"

        for channel_name in channels:
            if channel_name in self.channels:
                channel = self.channels[channel_name]
                # Create resolution alert
                resolution_alert = Alert(
                    alert_id=f"{alert.alert_id}_resolved",
                    title=f"RESOLVED: {alert.title}",
                    message=f"Alert has been resolved. Original: {alert.message}",
                    severity=AlertSeverity.INFO,
                    source=alert.source,
                    timestamp=datetime.now(timezone.utc),
                    tags=alert.tags,
                    metadata={**alert.metadata, "original_alert_id": alert.alert_id}
                )
                channel.send_alert(resolution_alert)

    def _process_alert_queue(self):
        """Process queued alerts (runs in background thread)."""
        while self._running:
            alerts_to_process = []

            # Get alerts from queue
            with self._queue_lock:
                if self._alert_queue:
                    alerts_to_process = self._alert_queue.copy()
                    self._alert_queue.clear()

            # Process alerts
            for alert in alerts_to_process:
                self._process_alert(alert)

            # Save state periodically
            self._save_alerts()

            # Sleep
            time.sleep(1)

    def _process_alert(self, alert: Alert):
        """Process a single alert."""
        # Determine channels based on alert type
        alert_type = alert.title.split('_')[0] if '_' in alert.title else alert.title
        rule = self.alert_rules.get(alert_type, {})
        channels = rule.get("channels", ["telegram"])

        success_count = 0

        # Send through each channel
        for channel_name in channels:
            if channel_name in self.channels:
                channel = self.channels[channel_name]
                if channel.send_alert(alert):
                    success_count += 1
                else:
                    logger.error(f"Failed to send alert through {channel_name}")

        # Update alert metadata
        alert.notification_count += 1
        alert.last_notification = datetime.now(timezone.utc)

        logger.info(f"Alert processed: {alert.alert_id} - sent through {success_count}/{len(channels)} channels")

    def start(self):
        """Start the alert manager."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._process_alert_queue, daemon=True)
        self._thread.start()

        # Load existing alerts
        self._load_alerts()

        logger.info("Alert manager started")

    def stop(self):
        """Stop the alert manager."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

        # Save final state
        self._save_alerts()

        logger.info("Alert manager stopped")

    def _load_alerts(self):
        """Load alerts from storage."""
        try:
            if self.storage_file.exists():
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)

                # Load active alerts
                for alert_data in data.get('active_alerts', []):
                    alert = Alert.from_dict(alert_data)
                    self.active_alerts[alert.alert_id] = alert

                # Load history
                for alert_data in data.get('alert_history', []):
                    alert = Alert.from_dict(alert_data)
                    self.alert_history.append(alert)

                logger.info(f"Loaded {len(self.active_alerts)} active alerts and {len(self.alert_history)} historical alerts")

        except Exception as e:
            logger.error(f"Failed to load alerts: {e}")

    def _save_alerts(self):
        """Save alerts to storage."""
        try:
            data = {
                'active_alerts': [alert.to_dict() for alert in self.active_alerts.values()],
                'alert_history': [alert.to_dict() for alert in self.alert_history[-self.max_history:]]
            }

            with open(self.storage_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")

    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        now = datetime.now(timezone.utc)
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)

        # Count alerts by severity and time period
        stats = {
            'active_alerts': len(self.active_alerts),
            'total_history': len(self.alert_history),
            'alerts_last_24h': len([a for a in self.alert_history if a.timestamp > last_24h]),
            'alerts_last_7d': len([a for a in self.alert_history if a.timestamp > last_7d]),
            'by_severity': {},
            'by_type': {}
        }

        # Count all alerts by severity
        all_alerts = list(self.active_alerts.values()) + self.alert_history
        for alert in all_alerts:
            severity = alert.severity.value
            alert_type = alert.title.split('_')[0]

            stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1
            stats['by_type'][alert_type] = stats['by_type'].get(alert_type, 0) + 1

        return stats

    def test_channels(self) -> Dict[str, bool]:
        """Test all configured channels."""
        results = {}
        test_alert = Alert(
            alert_id="test_connection",
            title="Connection Test",
            message="This is a test alert to verify channel connectivity.",
            severity=AlertSeverity.INFO,
            source="alert_manager",
            timestamp=datetime.now(timezone.utc)
        )

        for name, channel in self.channels.items():
            results[name] = channel.test_connection()

        return results


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        raise RuntimeError("Alert manager not initialized. Call initialize_alert_manager() first.")
    return _alert_manager


def initialize_alert_manager(config: Dict[str, Any] = None) -> AlertManager:
    """Initialize global alert manager."""
    global _alert_manager
    _alert_manager = AlertManager(config)
    _alert_manager.start()
    return _alert_manager


# Convenience functions for common alerts
def alert_bot_crashed(error_message: str):
    """Alert that the bot has crashed."""
    return get_alert_manager().send_alert(
        "bot_crashed",
        "Bot Crashed",
        f"The trading bot has stopped unexpectedly: {error_message}",
        tags={"component": "bot", "type": "crash"}
    )


def alert_websocket_disconnected(duration_seconds: int):
    """Alert that WebSocket has been disconnected."""
    return get_alert_manager().send_alert(
        "websocket_disconnected",
        "WebSocket Disconnected",
        f"WebSocket has been disconnected for {duration_seconds} seconds",
        tags={"component": "websocket", "duration": str(duration_seconds)}
    )


def alert_position_stop_loss_hit(symbol: str, loss_amount: float):
    """Alert that a position hit stop loss."""
    return get_alert_manager().send_alert(
        "position_stop_loss_hit",
        f"Stop Loss Hit: {symbol}",
        f"Position in {symbol} hit stop loss with loss of ${loss_amount:.2f}",
        tags={"symbol": symbol, "type": "stop_loss"},
        metadata={"loss_amount": loss_amount}
    )


def alert_daily_loss_limit_reached(loss_percentage: float):
    """Alert that daily loss limit has been reached."""
    return get_alert_manager().send_alert(
        "daily_loss_limit_reached",
        "Daily Loss Limit Reached",
        f"Daily loss limit reached at {loss_percentage:.2f}%",
        tags={"type": "risk_limit"},
        metadata={"loss_percentage": loss_percentage}
    )


def alert_api_latency_high(endpoint: str, latency_seconds: float, threshold: float):
    """Alert about high API latency."""
    return get_alert_manager().send_alert(
        "api_latency_high",
        f"High API Latency: {endpoint}",
        f"API latency for {endpoint} is {latency_seconds:.2f}s (threshold: {threshold:.2f}s)",
        tags={"endpoint": endpoint, "type": "performance"},
        metadata={"latency": latency_seconds, "threshold": threshold}
    )


def alert_memory_usage_high(usage_percent: float):
    """Alert about high memory usage."""
    return get_alert_manager().send_alert(
        "memory_usage_high",
        "High Memory Usage",
        f"Memory usage is at {usage_percent:.1f}%",
        tags={"component": "system", "type": "resource"},
        metadata={"memory_percent": usage_percent}
    )


def alert_daily_summary(trades_count: int, pnl: float, win_rate: float):
    """Send daily trading summary."""
    return get_alert_manager().send_alert(
        "daily_trading_summary",
        "Daily Trading Summary",
        f"Completed {trades_count} trades with P&L: ${pnl:.2f}, Win Rate: {win_rate:.1f}%",
        tags={"type": "summary", "period": "daily"},
        metadata={"trades": trades_count, "pnl": pnl, "win_rate": win_rate}
    )
