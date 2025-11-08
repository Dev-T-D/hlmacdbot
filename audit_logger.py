"""
Audit Logger for Trading Bot

Provides tamper-protected audit logging for all trading activities.
Each log entry includes a cryptographic hash to detect tampering.

Features:
- Separate audit log file (logs/audit.log)
- Cryptographic hash verification (SHA-256)
- Structured JSON format for easy parsing
- Automatic log rotation
- Tamper detection on log file read
"""

import json
import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Tamper-protected audit logger for trading activities.
    
    Each log entry includes:
    - Timestamp (UTC)
    - Event type
    - Event data
    - Previous entry hash (chain)
    - Current entry hash (for verification)
    """
    
    def __init__(self, log_file: str = "logs/audit.log"):
        """
        Initialize audit logger.
        
        Args:
            log_file: Path to audit log file
        """
        self.log_file = log_file
        self.log_dir = os.path.dirname(log_file)
        
        # Ensure log directory exists
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        
        # Last entry hash (for chaining)
        self._last_hash: Optional[str] = None
        
        # Initialize log file if it doesn't exist
        if not os.path.exists(self.log_file):
            self._initialize_log_file()
        else:
            # Load last hash from existing log
            self._load_last_hash()
        
        logger.info(f"Audit logger initialized: {self.log_file}")
    
    def _initialize_log_file(self) -> None:
        """Initialize audit log file with header."""
        header = {
            "type": "AUDIT_LOG_HEADER",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            "description": "Trading Bot Audit Log - Tamper Protected",
            "hash": None,
            "previous_hash": None
        }
        
        header_hash = self._calculate_hash(header)
        header["hash"] = header_hash
        
        with open(self.log_file, 'w') as f:
            f.write(json.dumps(header, separators=(',', ':')) + '\n')
        
        self._last_hash = header_hash
        logger.debug(f"Initialized audit log file: {self.log_file}")
    
    def _load_last_hash(self) -> None:
        """Load the last entry hash from the log file."""
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    # Read last line
                    last_line = lines[-1].strip()
                    if last_line:
                        entry = json.loads(last_line)
                        self._last_hash = entry.get('hash')
        except Exception as e:
            logger.warning(f"Failed to load last hash from audit log: {e}")
            self._last_hash = None
    
    def _calculate_hash(self, data: Dict) -> str:
        """
        Calculate SHA-256 hash of log entry.
        
        Args:
            data: Log entry dictionary (without hash fields)
            
        Returns:
            Hexadecimal hash string
        """
        # Create a copy without hash fields for hashing
        hash_data = {k: v for k, v in data.items() if k not in ['hash', 'previous_hash']}
        
        # Convert to JSON string (sorted keys for consistency)
        json_str = json.dumps(hash_data, separators=(',', ':'), sort_keys=True)
        
        # Calculate SHA-256 hash
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    
    def _write_entry(self, entry: Dict) -> None:
        """
        Write audit log entry with tamper protection.
        
        Args:
            entry: Log entry dictionary
        """
        # Add timestamp if not present
        if 'timestamp' not in entry:
            entry['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        # Add previous hash (chain)
        entry['previous_hash'] = self._last_hash
        
        # Calculate hash
        entry_hash = self._calculate_hash(entry)
        entry['hash'] = entry_hash
        
        # Write to file (append mode)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry, separators=(',', ':')) + '\n')
        
        # Update last hash
        self._last_hash = entry_hash
        
        logger.debug(f"Audit log entry written: {entry.get('type', 'UNKNOWN')}")
    
    def log_trade_entry(self, 
                       symbol: str,
                       position_type: str,
                       entry_price: float,
                       quantity: float,
                       stop_loss: float,
                       take_profit: float,
                       order_id: Optional[str] = None,
                       balance: Optional[float] = None,
                       leverage: Optional[int] = None,
                       dry_run: bool = False) -> None:
        """
        Log trade entry event.
        
        Args:
            symbol: Trading symbol
            position_type: "LONG" or "SHORT"
            entry_price: Entry price
            quantity: Position quantity
            stop_loss: Stop loss price
            take_profit: Take profit price
            order_id: Exchange order ID (if available)
            balance: Account balance at entry
            leverage: Leverage used
            dry_run: Whether this is a dry run
        """
        entry = {
            "type": "TRADE_ENTRY",
            "symbol": symbol,
            "position_type": position_type,
            "entry_price": float(entry_price),
            "quantity": float(quantity),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "order_id": order_id,
            "balance": float(balance) if balance is not None else None,
            "leverage": leverage,
            "dry_run": dry_run
        }
        
        self._write_entry(entry)
    
    def log_trade_exit(self,
                      symbol: str,
                      position_type: str,
                      entry_price: float,
                      exit_price: float,
                      quantity: float,
                      pnl: float,
                      pnl_pct: float,
                      reason: str,
                      order_id: Optional[str] = None,
                      leverage: Optional[int] = None,
                      margin_used: Optional[float] = None,
                      dry_run: bool = False) -> None:
        """
        Log trade exit event.
        
        Args:
            symbol: Trading symbol
            position_type: "LONG" or "SHORT"
            entry_price: Original entry price
            exit_price: Exit price
            quantity: Position quantity
            pnl: Profit/Loss amount
            pnl_pct: Profit/Loss percentage
            reason: Exit reason
            order_id: Exchange order ID (if available)
            leverage: Leverage used
            margin_used: Margin used for position
            dry_run: Whether this is a dry run
        """
        entry = {
            "type": "TRADE_EXIT",
            "symbol": symbol,
            "position_type": position_type,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "quantity": float(quantity),
            "pnl": float(pnl),
            "pnl_pct": float(pnl_pct),
            "reason": reason,
            "order_id": order_id,
            "leverage": leverage,
            "margin_used": float(margin_used) if margin_used is not None else None,
            "dry_run": dry_run
        }
        
        self._write_entry(entry)
    
    def log_order_status_change(self,
                               order_id: str,
                               symbol: str,
                               status: str,
                               previous_status: Optional[str] = None,
                               details: Optional[Dict] = None) -> None:
        """
        Log order status change event.
        
        Args:
            order_id: Order ID
            symbol: Trading symbol
            status: New status
            previous_status: Previous status (if known)
            details: Additional details
        """
        entry = {
            "type": "ORDER_STATUS_CHANGE",
            "order_id": order_id,
            "symbol": symbol,
            "status": status,
            "previous_status": previous_status,
            "details": details or {}
        }
        
        self._write_entry(entry)
    
    def log_position_update(self,
                           symbol: str,
                           position_type: str,
                           quantity: float,
                           entry_price: float,
                           mark_price: float,
                           unrealized_pnl: float,
                           reason: str) -> None:
        """
        Log position update event.
        
        Args:
            symbol: Trading symbol
            position_type: "LONG" or "SHORT"
            quantity: Position quantity
            entry_price: Entry price
            mark_price: Current mark price
            unrealized_pnl: Unrealized P&L
            reason: Update reason
        """
        entry = {
            "type": "POSITION_UPDATE",
            "symbol": symbol,
            "position_type": position_type,
            "quantity": float(quantity),
            "entry_price": float(entry_price),
            "mark_price": float(mark_price),
            "unrealized_pnl": float(unrealized_pnl),
            "reason": reason
        }
        
        self._write_entry(entry)
    
    def log_stop_loss_update(self,
                            symbol: str,
                            old_stop_loss: float,
                            new_stop_loss: float,
                            reason: str) -> None:
        """
        Log stop loss update event.
        
        Args:
            symbol: Trading symbol
            old_stop_loss: Previous stop loss price
            new_stop_loss: New stop loss price
            reason: Update reason
        """
        entry = {
            "type": "STOP_LOSS_UPDATE",
            "symbol": symbol,
            "old_stop_loss": float(old_stop_loss),
            "new_stop_loss": float(new_stop_loss),
            "reason": reason
        }
        
        self._write_entry(entry)
    
    def log_config_change(self,
                         change_type: str,
                         field: str,
                         old_value: Any,
                         new_value: Any,
                         reason: str) -> None:
        """
        Log configuration change event.
        
        Args:
            change_type: Type of change
            field: Configuration field name
            old_value: Previous value
            new_value: New value
            reason: Change reason
        """
        entry = {
            "type": "CONFIG_CHANGE",
            "change_type": change_type,
            "field": field,
            "old_value": str(old_value) if old_value is not None else None,
            "new_value": str(new_value) if new_value is not None else None,
            "reason": reason
        }
        
        self._write_entry(entry)
    
    def verify_log_integrity(self) -> tuple[bool, List[str]]:
        """
        Verify integrity of audit log file.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not os.path.exists(self.log_file):
            errors.append("Audit log file does not exist")
            return False, errors
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                errors.append("Audit log file is empty")
                return False, errors
            
            previous_hash = None
            
            for line_num, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    
                    # Skip header
                    if entry.get('type') == 'AUDIT_LOG_HEADER':
                        previous_hash = entry.get('hash')
                        continue
                    
                    # Verify previous hash chain
                    entry_previous_hash = entry.get('previous_hash')
                    if previous_hash != entry_previous_hash:
                        errors.append(
                            f"Line {line_num}: Hash chain broken. "
                            f"Expected: {previous_hash}, Got: {entry_previous_hash}"
                        )
                    
                    # Verify entry hash
                    stored_hash = entry.get('hash')
                    if stored_hash:
                        # Calculate hash without hash fields
                        calculated_hash = self._calculate_hash(entry)
                        if calculated_hash != stored_hash:
                            errors.append(
                                f"Line {line_num}: Entry hash mismatch. "
                                f"Expected: {calculated_hash}, Got: {stored_hash}"
                            )
                    
                    previous_hash = stored_hash
                    
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON: {e}")
                except Exception as e:
                    errors.append(f"Line {line_num}: Error: {e}")
            
            is_valid = len(errors) == 0
            return is_valid, errors
            
        except Exception as e:
            errors.append(f"Failed to read audit log file: {e}")
            return False, errors
    
    def get_recent_entries(self, limit: int = 100) -> List[Dict]:
        """
        Get recent audit log entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of log entries (most recent first)
        """
        entries = []
        
        if not os.path.exists(self.log_file):
            return entries
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            # Read from end (most recent first)
            for line in reversed(lines[-limit:]):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    # Skip header
                    if entry.get('type') != 'AUDIT_LOG_HEADER':
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
            
            return entries
            
        except Exception as e:
            logger.error(f"Failed to read audit log entries: {e}")
            return []

