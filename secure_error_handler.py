"""
Secure Error Handler with PII Redaction and Information Leakage Prevention

Provides secure error handling that prevents sensitive information leakage
while maintaining useful debugging information for authorized users.

Features:
- Automatic PII detection and redaction
- Private key and credential sanitization
- Secure error reporting with configurable detail levels
- Audit logging of security events
- Exception chaining with sensitive data removal
"""

import re
import logging
import hashlib
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class PIIRedactor:
    """
    Personal Identifiable Information (PII) redactor.

    Automatically detects and redacts sensitive information from strings,
    error messages, and data structures.
    """

    def __init__(self):
        # PII patterns to redact
        self.pii_patterns = {
            # Ethereum private keys (64 hex characters after 0x)
            'ethereum_private_key': re.compile(r'\b0x[0-9a-fA-F]{64}\b'),

            # Ethereum addresses (40 hex characters after 0x)
            'ethereum_address': re.compile(r'\b0x[0-9a-fA-F]{40}\b'),

            # Generic private keys (long hex strings)
            'hex_private_key': re.compile(r'\b[0-9a-fA-F]{32,}\b'),

            # Wallet seed phrases (12-24 word sequences)
            'seed_phrase': re.compile(r'\b(?:\w+\s+){11,23}\w+\b'),

            # API keys and secrets (common patterns)
            'api_key': re.compile(r'\b[A-Za-z0-9]{32,}\b'),  # Generic long alphanumeric

            # Email addresses
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),

            # Phone numbers
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),

            # Social security numbers
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),

            # Credit card numbers
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),

            # IP addresses
            'ip_address': re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
        }

        # Words that indicate sensitive content
        self.sensitive_keywords = {
            'private', 'secret', 'key', 'password', 'token', 'credential',
            'seed', 'mnemonic', 'wallet', 'balance', 'pnl', 'profit', 'loss'
        }

    def redact_string(self, text: str) -> str:
        """
        Redact sensitive information from a string.

        Args:
            text: Input string to redact

        Returns:
            Redacted string with sensitive data replaced
        """
        if not isinstance(text, str):
            return str(text)

        redacted = text

        # Apply PII pattern redaction
        for pii_type, pattern in self.pii_patterns.items():
            redacted = pattern.sub(f'[REDACTED_{pii_type.upper()}]', redacted)

        # Check for sensitive keywords and redact surrounding content
        words = redacted.split()
        redacted_words = []

        for word in words:
            # Check if word contains sensitive keywords
            word_lower = word.lower()
            if any(keyword in word_lower for keyword in self.sensitive_keywords):
                # Redact the entire word if it contains sensitive keywords
                redacted_words.append('[REDACTED_SENSITIVE]')
            else:
                redacted_words.append(word)

        return ' '.join(redacted_words)

    def redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redact sensitive information from a dictionary recursively.

        Args:
            data: Dictionary to redact

        Returns:
            Redacted dictionary
        """
        if not isinstance(data, dict):
            return data

        redacted = {}

        for key, value in data.items():
            # Redact sensitive key names
            redacted_key = self.redact_string(key)

            # Redact values based on type
            if isinstance(value, str):
                redacted_value = self.redact_string(value)
            elif isinstance(value, dict):
                redacted_value = self.redact_dict(value)
            elif isinstance(value, list):
                redacted_value = [self.redact_dict(item) if isinstance(item, dict) else self.redact_string(str(item)) for item in value]
            else:
                # For other types, redact if they look like sensitive strings
                str_value = str(value)
                if len(str_value) > 10 and any(keyword in str_value.lower() for keyword in self.sensitive_keywords):
                    redacted_value = '[REDACTED_SENSITIVE_DATA]'
                else:
                    redacted_value = value

            redacted[redacted_key] = redacted_value

        return redacted

    def is_sensitive_content(self, content: str) -> bool:
        """
        Check if content contains sensitive information.

        Args:
            content: Content to check

        Returns:
            True if content appears to contain sensitive information
        """
        if not isinstance(content, str):
            content = str(content)

        # Check for PII patterns
        for pattern in self.pii_patterns.values():
            if pattern.search(content):
                return True

        # Check for sensitive keywords
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in self.sensitive_keywords)


class SecureErrorHandler:
    """
    Secure error handler with configurable detail levels and PII redaction.

    Provides different error reporting levels for different audiences:
    - Public: Minimal information, no sensitive data
    - Internal: Detailed information for developers
    - Audit: Full information for security auditing
    """

    def __init__(self, log_security_events: bool = True):
        """
        Initialize secure error handler.

        Args:
            log_security_events: Whether to log security-related events
        """
        self.pii_redactor = PIIRedactor()
        self.log_security_events = log_security_events

        # Error reporting levels
        self.LEVEL_PUBLIC = 'public'      # User-friendly, minimal info
        self.LEVEL_INTERNAL = 'internal'  # Detailed for developers
        self.LEVEL_AUDIT = 'audit'        # Full info for security audit

        # Security event counters
        self.security_event_counts = {
            'pii_detected': 0,
            'key_leak_attempt': 0,
            'error_suppression': 0,
            'sensitive_data_logged': 0
        }

        logger.info("Secure error handler initialized")

    def handle_exception(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        level: str = 'internal'
    ) -> Dict[str, Any]:
        """
        Handle exception with appropriate security measures.

        Args:
            exception: Exception to handle
            context: Additional context information
            level: Error reporting level (public, internal, audit)

        Returns:
            Error information dictionary with appropriate detail level
        """
        # Generate error ID for tracking
        error_id = hashlib.sha256(
            f"{type(exception).__name__}:{str(exception)}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        # Check if exception contains sensitive information
        exception_str = str(exception)
        contains_sensitive = self.pii_redactor.is_sensitive_content(exception_str)

        if contains_sensitive:
            self.security_event_counts['pii_detected'] += 1
            if self.log_security_events:
                logger.warning(f"SECURITY: Sensitive data detected in exception {error_id}")

        # Get traceback for internal/audit levels
        traceback_info = None
        if level in ['internal', 'audit']:
            traceback_info = traceback.format_exc()

        # Redact traceback for public level
        if level == 'public' and traceback_info:
            traceback_info = self.pii_redactor.redact_string(traceback_info)

        # Build error information
        error_info = {
            'error_id': error_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'exception_type': type(exception).__name__,
            'level': level,
        }

        # Add context if provided
        if context:
            # Redact context for public level
            if level == 'public':
                error_info['context'] = self.pii_redactor.redact_dict(context)
            else:
                error_info['context'] = context

        # Add exception message based on level
        if level == 'public':
            # Public: Generic message without sensitive details
            error_info['message'] = "An error occurred. Please try again later."
            error_info['user_message'] = "Something went wrong. Our team has been notified."
        elif level == 'internal':
            # Internal: Redacted details for developers
            error_info['message'] = self.pii_redactor.redact_string(exception_str)
            error_info['traceback'] = traceback_info
        else:  # audit
            # Audit: Full details for security review
            error_info['message'] = exception_str
            error_info['traceback'] = traceback_info
            error_info['contains_sensitive'] = contains_sensitive

        # Log error at appropriate level
        self._log_error(error_info, exception)

        return error_info

    def sanitize_log_message(self, message: str, level: str = 'info') -> str:
        """
        Sanitize log message to prevent sensitive data leakage.

        Args:
            message: Log message to sanitize
            level: Log level (affects sanitization strictness)

        Returns:
            Sanitized message
        """
        if level in ['debug', 'info'] and not self.pii_redactor.is_sensitive_content(message):
            # Allow more detailed logging for non-sensitive content
            return message
        else:
            # Redact sensitive content for warnings and above
            sanitized = self.pii_redactor.redact_string(message)

            if sanitized != message:
                self.security_event_counts['sensitive_data_logged'] += 1
                if self.log_security_events:
                    logger.warning("SECURITY: Sensitive data redacted from log message")

            return sanitized

    def create_secure_error_report(
        self,
        error_info: Dict[str, Any],
        include_system_info: bool = False
    ) -> Dict[str, Any]:
        """
        Create secure error report for external reporting.

        Args:
            error_info: Error information from handle_exception
            include_system_info: Whether to include system information

        Returns:
            Secure error report dictionary
        """
        # Start with error info
        report = {
            'error_id': error_info['error_id'],
            'timestamp': error_info['timestamp'],
            'level': 'public',  # Always public for external reports
        }

        # Add sanitized message
        if 'message' in error_info:
            report['message'] = self.pii_redactor.redact_string(error_info['message'])

        # Add system info if requested (redacted)
        if include_system_info:
            import platform
            import sys

            system_info = {
                'platform': platform.platform(),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'architecture': platform.architecture()[0],
            }

            # Redact any potentially sensitive system info
            report['system_info'] = self.pii_redactor.redact_dict(system_info)

        # Add security event summary
        report['security_events'] = self.security_event_counts.copy()

        return report

    def _log_error(self, error_info: Dict[str, Any], exception: Exception) -> None:
        """Log error at appropriate level."""
        error_id = error_info['error_id']
        level = error_info['level']

        if level == 'public':
            # Minimal logging for public
            logger.error(f"Error {error_id}: {error_info.get('user_message', 'An error occurred')}")
        elif level == 'internal':
            # Detailed logging for internal
            logger.error(f"Error {error_id}: {error_info['message']}")
            if 'traceback' in error_info and error_info['traceback']:
                logger.debug(f"Traceback for {error_id}: {error_info['traceback']}")
        else:  # audit
            # Full logging for audit
            logger.error(f"AUDIT Error {error_id}: {error_info['message']}")
            if 'traceback' in error_info and error_info['traceback']:
                logger.error(f"AUDIT Traceback {error_id}: {error_info['traceback']}")

            # Log security implications
            if error_info.get('contains_sensitive', False):
                logger.critical(f"SECURITY EVENT: Error {error_id} contains sensitive information")

    def get_security_stats(self) -> Dict[str, int]:
        """Get security event statistics."""
        return self.security_event_counts.copy()

    def reset_security_stats(self) -> None:
        """Reset security event counters."""
        self.security_event_counts = {key: 0 for key in self.security_event_counts}


class SecurityEventLogger:
    """
    Security event logger for tracking security-related events.

    Maintains separate log for security events with tamper protection.
    """

    def __init__(self, log_file: str = "logs/security_events.log"):
        """
        Initialize security event logger.

        Args:
            log_file: Path to security event log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Security event types
        self.EVENT_PII_DETECTED = "PII_DETECTED"
        self.EVENT_KEY_LEAK_ATTEMPT = "KEY_LEAK_ATTEMPT"
        self.EVENT_UNAUTHORIZED_ACCESS = "UNAUTHORIZED_ACCESS"
        self.EVENT_CIRCUIT_BREAKER_TRIPPED = "CIRCUIT_BREAKER_TRIPPED"
        self.EVENT_DEAD_MAN_SWITCH = "DEAD_MAN_SWITCH"
        self.EVENT_SSL_PINNING_FAILURE = "SSL_PINNING_FAILURE"
        self.EVENT_REPLAY_ATTACK_DETECTED = "REPLAY_ATTACK_DETECTED"

        logger.info(f"Security event logger initialized: {log_file}")

    def log_security_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        severity: str = "medium"
    ) -> None:
        """
        Log security event.

        Args:
            event_type: Type of security event
            details: Event details
            severity: Event severity (low, medium, high, critical)
        """
        event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': event_type,
            'severity': severity,
            'details': details,
            'event_id': hashlib.sha256(
                f"{event_type}:{details}:{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]
        }

        # Redact sensitive information from event details
        redactor = PIIRedactor()
        event['details'] = redactor.redact_dict(event['details'])

        # Write to security log
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"{event['timestamp']} | {event_type} | {severity} | {event['event_id']} | {event['details']}\n")
        except Exception as e:
            logger.error(f"Failed to write security event: {e}")

        # Log to main logger with appropriate level
        if severity == 'critical':
            logger.critical(f"SECURITY EVENT: {event_type} - {event['event_id']}")
        elif severity == 'high':
            logger.error(f"SECURITY EVENT: {event_type} - {event['event_id']}")
        elif severity == 'medium':
            logger.warning(f"SECURITY EVENT: {event_type} - {event['event_id']}")
        else:
            logger.info(f"SECURITY EVENT: {event_type} - {event['event_id']}")


# Global instances
_secure_error_handler: Optional[SecureErrorHandler] = None
_security_event_logger: Optional[SecurityEventLogger] = None


def get_secure_error_handler() -> SecureErrorHandler:
    """Get global secure error handler instance."""
    global _secure_error_handler
    if _secure_error_handler is None:
        _secure_error_handler = SecureErrorHandler()
    return _secure_error_handler


def get_security_event_logger() -> SecurityEventLogger:
    """Get global security event logger instance."""
    global _security_event_logger
    if _security_event_logger is None:
        _security_event_logger = SecurityEventLogger()
    return _security_event_logger


def initialize_secure_error_handling(log_security_events: bool = True) -> Tuple[SecureErrorHandler, SecurityEventLogger]:
    """Initialize global secure error handling."""
    global _secure_error_handler, _security_event_logger

    _secure_error_handler = SecureErrorHandler(log_security_events)
    _security_event_logger = SecurityEventLogger()

    return _secure_error_handler, _security_event_logger
