"""
Health Check Endpoint for Trading Bot

Provides HTTP endpoint to monitor bot status, last cycle time, errors, etc.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Optional
from flask import Flask, jsonify

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Health monitoring server for trading bot
    
    Provides HTTP endpoint to check bot status, last cycle time, errors, etc.
    Runs in a separate thread to avoid blocking main bot loop.
    """
    
    def __init__(self, bot_instance, port: int = 8080, host: str = '127.0.0.1'):
        """
        Initialize health monitor
        
        Args:
            bot_instance: TradingBot instance to monitor
            port: HTTP server port (default: 8080)
            host: HTTP server host (default: 127.0.0.1 for localhost only)
        """
        self.bot = bot_instance
        self.port = port
        self.host = host
        self.app = Flask(__name__)
        self.server_thread: Optional[threading.Thread] = None
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint - returns bot status"""
            try:
                status = self._get_bot_status()
                return jsonify(status), 200
            except Exception as e:
                logger.error(f"Error generating health status: {e}")
                return jsonify({
                    'status': 'error',
                    'error': str(e)
                }), 500
        
        @self.app.route('/', methods=['GET'])
        def root():
            """Root endpoint - redirects to health"""
            return jsonify({
                'message': 'Trading Bot Health Monitor',
                'endpoints': {
                    '/health': 'Get bot health status'
                }
            }), 200
    
    def _get_bot_status(self) -> Dict:
        """
        Get comprehensive bot status
        
        Returns:
            Dict with bot status information
        """
        now = datetime.now(timezone.utc)
        
        # Calculate uptime
        uptime_seconds = 0
        uptime_str = "Unknown"
        if hasattr(self.bot, 'start_time') and self.bot.start_time:
            uptime_seconds = (now - self.bot.start_time).total_seconds()
            hours = int(uptime_seconds // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            seconds = int(uptime_seconds % 60)
            uptime_str = f"{hours}h {minutes}m {seconds}s"
        
        # Calculate time since last cycle
        last_cycle_ago = None
        last_cycle_ago_str = "Never"
        if hasattr(self.bot, 'last_cycle_time') and self.bot.last_cycle_time:
            last_cycle_ago = (now - self.bot.last_cycle_time).total_seconds()
            if last_cycle_ago < 60:
                last_cycle_ago_str = f"{int(last_cycle_ago)}s ago"
            elif last_cycle_ago < 3600:
                last_cycle_ago_str = f"{int(last_cycle_ago // 60)}m ago"
            else:
                last_cycle_ago_str = f"{int(last_cycle_ago // 3600)}h ago"
        
        # Determine bot health status
        health_status = "healthy"
        if last_cycle_ago is not None:
            # Check if bot is stuck (last cycle was more than 2x check_interval ago)
            check_interval = getattr(self.bot, 'check_interval', 60)
            if last_cycle_ago > (check_interval * 2):
                health_status = "stuck"
            elif last_cycle_ago > check_interval:
                health_status = "delayed"
        
        # Get position status
        position_status = None
        if self.bot.current_position:
            position_status = {
                'type': self.bot.current_position.get('type', 'UNKNOWN'),
                'entry_price': self.bot.current_position.get('entry_price', 0),
                'quantity': self.bot.current_position.get('quantity', 0),
                'stop_loss': self.bot.current_position.get('stop_loss', 0),
                'take_profit': self.bot.current_position.get('take_profit', 0),
                'entry_time': self.bot.current_position.get('entry_time', '').isoformat() if isinstance(
                    self.bot.current_position.get('entry_time'), datetime) else str(
                    self.bot.current_position.get('entry_time', ''))
            }
        
        # Get trailing stop status
        trailing_stop_status = None
        if self.bot.trailing_stop_enabled and self.bot.trailing_stop:
            trailing_stop_status = self.bot.trailing_stop.get_status()
            # Convert datetime to ISO string if present
            if trailing_stop_status.get('last_update') and isinstance(trailing_stop_status['last_update'], datetime):
                trailing_stop_status['last_update'] = trailing_stop_status['last_update'].isoformat()
        
        # Get market data cache status
        cache_status = {
            'cached': self.bot.market_data_cache is not None,
            'cache_age_seconds': None,
            'cache_size': len(self.bot.market_data_cache) if self.bot.market_data_cache is not None else 0
        }
        if self.bot.cache_timestamp:
            cache_age = (now - self.bot.cache_timestamp).total_seconds()
            cache_status['cache_age_seconds'] = int(cache_age)
        
        # Get recent errors (last 10)
        recent_errors = []
        if hasattr(self.bot, 'recent_errors') and self.bot.recent_errors:
            recent_errors = self.bot.recent_errors[-10:]  # Last 10 errors
        
        # Get risk manager stats
        risk_stats = {
            'daily_pnl': getattr(self.bot.risk_manager, 'daily_pnl', 0),
            'daily_trades': getattr(self.bot.risk_manager, 'daily_trades', 0),
            'max_position_size_pct': self.bot.risk_manager.max_position_size_pct * 100,
            'max_daily_loss_pct': self.bot.risk_manager.max_daily_loss_pct * 100,
            'leverage': self.bot.risk_manager.leverage
        }
        
        # Compile status
        status = {
            'status': health_status,
            'timestamp': now.isoformat(),
            'uptime': {
                'seconds': int(uptime_seconds),
                'formatted': uptime_str
            },
            'bot': {
                'exchange': self.bot.exchange_name,
                'symbol': self.bot.symbol,
                'timeframe': self.bot.timeframe,
                'check_interval': self.bot.check_interval,
                'dry_run': self.bot.dry_run,
                'leverage': getattr(self.bot, 'config', {}).get('risk', {}).get('leverage', 10)
            },
            'cycle': {
                'last_cycle_time': self.bot.last_cycle_time.isoformat() if hasattr(self.bot, 'last_cycle_time') and self.bot.last_cycle_time else None,
                'last_cycle_ago': last_cycle_ago_str,
                'last_cycle_ago_seconds': int(last_cycle_ago) if last_cycle_ago is not None else None,
                'cycle_count': getattr(self.bot, 'cycle_count', 0)
            },
            'position': position_status,
            'trailing_stop': trailing_stop_status,
            'market_data_cache': cache_status,
            'risk_manager': risk_stats,
            'recent_errors': recent_errors
        }
        
        return status
    
    def start(self):
        """Start health monitor server in background thread"""
        if self.server_thread and self.server_thread.is_alive():
            logger.warning("Health monitor already running")
            return
        
        def run_server():
            try:
                logger.info(f"🏥 Health monitor starting on http://{self.host}:{self.port}")
                # Disable Flask logging to reduce noise
                log = logging.getLogger('werkzeug')
                log.setLevel(logging.WARNING)
                self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
            except Exception as e:
                logger.error(f"Health monitor server error: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        logger.info(f"✅ Health monitor started (thread: {self.server_thread.name})")
    
    def stop(self):
        """Stop health monitor server"""
        # Flask doesn't have a clean shutdown, but since it's a daemon thread,
        # it will be terminated when main process exits
        logger.info("Health monitor stopping...")
    
    @property
    def config(self):
        """Access bot config"""
        return self.bot.config if hasattr(self.bot, 'config') else {}

