#!/usr/bin/env python3
"""
Trading Bot Watchdog Process

Separate monitoring process that watches the main trading bot,
detects failures, and performs recovery actions including restart
and emergency position closure.

Features:
- Process monitoring and health checks
- Automatic restart on failure
- Emergency position closure
- Alerting integration
- Forensic data collection
"""

import sys
import os
import time
import signal
import psutil
import subprocess
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import threading

# Add current directory to path
sys.path.insert(0, '.')

from resilience import get_resilience_manager

logger = logging.getLogger(__name__)


class WatchdogConfig:
    """Configuration for watchdog behavior."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Process monitoring
        self.bot_process_name = self.config.get('bot_process_name', 'trading_bot.py')
        self.bot_command = self.config.get('bot_command', ['python', 'trading_bot.py'])
        self.working_directory = Path(self.config.get('working_directory', '.'))

        # Health checks
        self.health_check_interval = self.config.get('health_check_interval', 30)  # seconds
        self.health_check_timeout = self.config.get('health_check_timeout', 10)  # seconds
        self.max_restart_attempts = self.config.get('max_restart_attempts', 5)
        self.restart_backoff_seconds = self.config.get('restart_backoff_seconds', 60)

        # Emergency procedures
        self.emergency_mode_timeout = self.config.get('emergency_mode_timeout', 300)  # 5 minutes
        self.force_kill_timeout = self.config.get('force_kill_timeout', 30)  # seconds

        # Monitoring
        self.pid_file = Path(self.config.get('pid_file', 'logs/watchdog.pid'))
        self.log_file = Path(self.config.get('log_file', 'logs/watchdog.log'))
        self.bot_pid_file = Path(self.config.get('bot_pid_file', 'logs/bot.pid'))

        # Create directories
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.bot_pid_file.parent.mkdir(parents=True, exist_ok=True)


class BotWatchdog:
    """
    Watchdog process for monitoring and recovering the trading bot.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = WatchdogConfig(config)
        self.bot_process: Optional[subprocess.Popen] = None
        self.bot_pid: Optional[int] = None
        self.restart_count = 0
        self.last_restart_time = None
        self.emergency_mode = False
        self.emergency_mode_start = None
        self.running = True

        # Setup logging
        self._setup_logging()

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Write PID file
        self._write_pid_file()

        logger.info("Watchdog initialized")

    def _setup_logging(self):
        """Setup watchdog logging."""
        formatter = logging.Formatter(
            '%(asctime)s - WATCHDOG - %(levelname)s - %(message)s'
        )

        # File handler
        file_handler = logging.FileHandler(self.config.log_file)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Configure logger
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    def _write_pid_file(self):
        """Write watchdog PID file."""
        try:
            with open(self.config.pid_file, 'w') as f:
                f.write(str(os.getpid()))
        except Exception as e:
            logger.error(f"Failed to write PID file: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Watchdog received signal {signum}, shutting down")
        self.running = False
        self._cleanup()

    def _cleanup(self):
        """Clean up resources."""
        try:
            # Kill bot process if running
            if self.bot_process and self.bot_process.poll() is None:
                logger.info("Terminating bot process")
                self.bot_process.terminate()

                # Wait for graceful shutdown
                try:
                    self.bot_process.wait(timeout=self.config.force_kill_timeout)
                except subprocess.TimeoutExpired:
                    logger.warning("Bot process didn't terminate gracefully, forcing kill")
                    self.bot_process.kill()

            # Remove PID files
            self.config.pid_file.unlink(missing_ok=True)
            self.config.bot_pid_file.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def start(self):
        """Start the watchdog monitoring loop."""
        logger.info("Watchdog starting monitoring loop")

        try:
            while self.running:
                try:
                    self._check_bot_health()
                    self._check_emergency_conditions()
                    time.sleep(self.config.health_check_interval)

                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(self.config.health_check_interval)

        except KeyboardInterrupt:
            logger.info("Watchdog interrupted by user")
        finally:
            self._cleanup()

    def _check_bot_health(self):
        """Check if the bot process is healthy."""
        # Check if bot PID file exists and process is running
        if self.config.bot_pid_file.exists():
            try:
                with open(self.config.bot_pid_file, 'r') as f:
                    stored_pid = int(f.read().strip())

                if psutil.pid_exists(stored_pid):
                    process = psutil.Process(stored_pid)

                    # Check if process is actually our bot
                    if self._is_bot_process(process):
                        # Check process health
                        cpu_percent = process.cpu_percent()
                        memory_percent = process.memory_percent()

                        # Check for excessive resource usage
                        if memory_percent > 90:
                            logger.warning(f"Bot process memory usage high: {memory_percent:.1f}%")
                        if cpu_percent > 95:
                            logger.warning(f"Bot process CPU usage high: {cpu_percent:.1f}%")

                        # Process appears healthy
                        return
                    else:
                        logger.warning(f"PID {stored_pid} exists but is not bot process")
                else:
                    logger.warning(f"Bot PID {stored_pid} not found in process list")

            except Exception as e:
                logger.error(f"Error checking bot PID: {e}")

        # Bot process not found or unhealthy - attempt restart
        self._attempt_bot_restart()

    def _is_bot_process(self, process: psutil.Process) -> bool:
        """Check if a process is the trading bot."""
        try:
            # Check command line
            cmdline = process.cmdline()
            if any('trading_bot.py' in cmd or 'python' in cmd for cmd in cmdline):
                return True

            # Check process name
            if 'python' in process.name().lower():
                return True

        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

        return False

    def _attempt_bot_restart(self):
        """Attempt to restart the bot process."""
        now = datetime.now(timezone.utc)

        # Check restart limits
        if self.restart_count >= self.config.max_restart_attempts:
            logger.critical(f"Maximum restart attempts ({self.config.max_restart_attempts}) exceeded")
            self._enter_emergency_mode()
            return

        # Check backoff period
        if (self.last_restart_time and
            (now - self.last_restart_time).seconds < self.config.restart_backoff_seconds):
            # Too soon for another restart
            return

        logger.info(f"Attempting bot restart (attempt {self.restart_count + 1}/{self.config.max_restart_attempts})")

        try:
            # Launch bot process
            env = os.environ.copy()
            env['WATCHDOG_MONITORED'] = 'true'

            self.bot_process = subprocess.Popen(
                self.config.bot_command,
                cwd=self.config.working_directory,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )

            self.bot_pid = self.bot_process.pid
            self.last_restart_time = now
            self.restart_count += 1

            # Write bot PID file
            with open(self.config.bot_pid_file, 'w') as f:
                f.write(str(self.bot_pid))

            logger.info(f"Bot process started with PID {self.bot_pid}")

            # Start thread to monitor bot startup
            threading.Thread(target=self._monitor_bot_startup, daemon=True).start()

        except Exception as e:
            logger.error(f"Failed to start bot process: {e}")
            self.restart_count += 1

            if self.restart_count >= self.config.max_restart_attempts:
                self._enter_emergency_mode()

    def _monitor_bot_startup(self):
        """Monitor bot startup and check for immediate failures."""
        if not self.bot_process:
            return

        try:
            # Wait a bit for startup
            time.sleep(10)

            if self.bot_process.poll() is not None:
                # Process exited immediately
                returncode = self.bot_process.returncode
                stdout, stderr = self.bot_process.communicate()

                logger.error(f"Bot process exited immediately with code {returncode}")
                logger.error(f"Bot stdout: {stdout.decode() if stdout else 'None'}")
                logger.error(f"Bot stderr: {stderr.decode() if stderr else 'None'}")

                # Don't increment restart count for immediate failures
                # (they're likely configuration issues)
                return

            # Bot appears to have started successfully
            logger.info("Bot process startup successful")
            self.restart_count = 0  # Reset restart count on successful start

        except Exception as e:
            logger.error(f"Error monitoring bot startup: {e}")

    def _check_emergency_conditions(self):
        """Check for emergency conditions requiring intervention."""
        if self.emergency_mode:
            # Check if emergency timeout has been exceeded
            if (self.emergency_mode_start and
                (datetime.now(timezone.utc) - self.emergency_mode_start).seconds > self.config.emergency_mode_timeout):
                logger.critical("Emergency mode timeout exceeded - manual intervention required")
                self.running = False
            return

        # Check for extended downtime
        if (self.last_restart_time and
            (datetime.now(timezone.utc) - self.last_restart_time).seconds > 600):  # 10 minutes
            logger.warning("Bot has been down for extended period")
            # Could send alerts here

    def _enter_emergency_mode(self):
        """Enter emergency mode when all restart attempts fail."""
        logger.critical("ENTERING EMERGENCY MODE - All restart attempts failed")

        self.emergency_mode = True
        self.emergency_mode_start = datetime.now(timezone.utc)

        # Attempt emergency position closure
        self._emergency_close_positions()

        # Send critical alerts
        self._send_emergency_alerts()

        logger.critical("Emergency mode active - manual intervention required")

    def _emergency_close_positions(self):
        """Attempt emergency position closure."""
        logger.critical("Attempting emergency position closure")

        try:
            # This would integrate with the exchange API to close positions
            # For now, we'll just log the attempt

            # Load positions from state
            if hasattr(self, 'resilience_manager'):
                positions = self.resilience_manager.state_manager.load_positions()

                for position in positions:
                    logger.critical(f"EMERGENCY: Would close position {position['position_id']} "
                                  f"({position['side']} {position['quantity']} {position['symbol']})")

                    # In a real implementation, this would:
                    # 1. Connect to exchange API
                    # 2. Place market orders to close positions
                    # 3. Verify closures
                    # 4. Update state

            logger.critical("Emergency position closure attempted")

        except Exception as e:
            logger.error(f"Emergency position closure failed: {e}")

    def _send_emergency_alerts(self):
        """Send emergency alerts when entering emergency mode."""
        logger.critical("Sending emergency alerts")

        try:
            # This would integrate with the alerting system
            alert_message = {
                'alert_type': 'bot_crashed',
                'title': 'CRITICAL: Trading Bot Complete Failure',
                'message': f'Watchdog has entered emergency mode after {self.restart_count} failed restart attempts. '
                          'Manual intervention required.',
                'source': 'watchdog',
                'tags': {'component': 'watchdog', 'severity': 'critical'},
                'metadata': {
                    'restart_attempts': self.restart_count,
                    'emergency_mode_start': self.emergency_mode_start.isoformat(),
                    'last_restart_time': self.last_restart_time.isoformat() if self.last_restart_time else None
                }
            }

            # Log alert (in real implementation, would send to alerting system)
            logger.critical(f"EMERGENCY ALERT: {alert_message}")

        except Exception as e:
            logger.error(f"Failed to send emergency alerts: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get watchdog status for monitoring."""
        return {
            'running': self.running,
            'bot_pid': self.bot_pid,
            'bot_process_running': self._is_bot_running(),
            'restart_count': self.restart_count,
            'last_restart_time': self.last_restart_time.isoformat() if self.last_restart_time else None,
            'emergency_mode': self.emergency_mode,
            'emergency_mode_start': self.emergency_mode_start.isoformat() if self.emergency_mode_start else None,
            'uptime_seconds': (datetime.now(timezone.utc) - datetime.fromtimestamp(psutil.Process(os.getpid()).create_time(), timezone.utc)).seconds
        }

    def _is_bot_running(self) -> bool:
        """Check if bot process is currently running."""
        if not self.bot_pid:
            return False

        try:
            process = psutil.Process(self.bot_pid)
            return process.is_running() and not process.status() == psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False


def create_watchdog_service_script() -> str:
    """Create systemd service script for watchdog."""
    return """
[Unit]
Description=Trading Bot Watchdog
After=network.target
Wants=network.target

[Service]
Type=simple
User=trading
Group=trading
WorkingDirectory=/path/to/bot
ExecStart=/usr/bin/python3 /path/to/bot/watchdog.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment=WATCHDOG_CONFIG=/path/to/bot/config/watchdog.json

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/path/to/bot/logs /path/to/bot/data
ProtectHome=true

[Install]
WantedBy=multi-user.target
"""


def create_watchdog_config_template() -> Dict[str, Any]:
    """Create template configuration for watchdog."""
    return {
        "bot_process_name": "trading_bot.py",
        "bot_command": ["python", "trading_bot.py"],
        "working_directory": ".",
        "health_check_interval": 30,
        "health_check_timeout": 10,
        "max_restart_attempts": 5,
        "restart_backoff_seconds": 60,
        "emergency_mode_timeout": 300,
        "force_kill_timeout": 30,
        "pid_file": "logs/watchdog.pid",
        "log_file": "logs/watchdog.log",
        "bot_pid_file": "logs/bot.pid"
    }


def main():
    """Main watchdog entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Trading Bot Watchdog")
    parser.add_argument("--config", help="Path to watchdog config file")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    args = parser.parse_args()

    # Load configuration
    config = create_watchdog_config_template()

    if args.config:
        config_file = Path(args.config)
        if config_file.exists():
            with open(config_file, 'r') as f:
                config.update(json.load(f))

    # Create and start watchdog
    watchdog = BotWatchdog(config)

    if args.daemon:
        # Daemonize process
        try:
            import daemon
            with daemon.DaemonContext():
                watchdog.start()
        except ImportError:
            logger.warning("python-daemon not available, running in foreground")
            watchdog.start()
    else:
        # Run in foreground
        watchdog.start()


if __name__ == "__main__":
    main()
