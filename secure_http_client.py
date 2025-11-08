"""
Secure HTTP Client with SSL Pinning and Request Signing

Provides hardened HTTP client functionality with:
- SSL certificate pinning to prevent MITM attacks
- Request signing with replay prevention
- Timestamp validation (30-second window)
- Comprehensive request/response validation
- Rate limiting and connection pooling
"""

import asyncio
import hashlib
import hmac
import json
import logging
import ssl
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone
from urllib.parse import urlparse

import aiohttp
from aiohttp import ClientTimeout, ClientError
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.exceptions import InvalidSignature

from exceptions import (
    ExchangeError,
    ExchangeAPIError,
    ExchangeNetworkError,
    ExchangeTimeoutError,
    ExchangeInvalidResponseError
)

logger = logging.getLogger(__name__)


class SSLFingerprintValidator:
    """
    SSL certificate fingerprint validator for certificate pinning.

    Prevents man-in-the-middle attacks by validating server certificates
    against known good fingerprints.
    """

    def __init__(self, pinned_fingerprints: Dict[str, List[str]]):
        """
        Initialize SSL fingerprint validator.

        Args:
            pinned_fingerprints: Dict mapping hostnames to lists of valid SHA-256 fingerprints
        """
        self.pinned_fingerprints = pinned_fingerprints
        self._ssl_context = None

    def create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with certificate pinning."""
        if self._ssl_context is None:
            self._ssl_context = ssl.create_default_context()
            self._ssl_context.check_hostname = True
            self._ssl_context.verify_mode = ssl.CERT_REQUIRED

        return self._ssl_context

    def validate_certificate(self, hostname: str, cert_der: bytes) -> bool:
        """
        Validate certificate against pinned fingerprints.

        Args:
            hostname: Server hostname
            cert_der: Certificate in DER format

        Returns:
            True if certificate is valid and pinned
        """
        if hostname not in self.pinned_fingerprints:
            logger.warning(f"No pinned fingerprints for hostname: {hostname}")
            return False

        try:
            # Load certificate
            cert = x509.load_der_x509_certificate(cert_der)

            # Calculate SHA-256 fingerprint
            fingerprint = cert.fingerprint(hashes.SHA256()).hex()

            # Check against pinned fingerprints
            valid_fingerprints = self.pinned_fingerprints[hostname]
            is_valid = fingerprint in valid_fingerprints

            if is_valid:
                logger.debug(f"SSL certificate validated for {hostname}: {fingerprint}")
            else:
                logger.critical(f"SSL certificate pinning failed for {hostname}!")
                logger.critical(f"Received fingerprint: {fingerprint}")
                logger.critical(f"Expected fingerprints: {valid_fingerprints}")
                return False

            return True

        except Exception as e:
            logger.error(f"SSL certificate validation error for {hostname}: {e}")
            return False


class RequestSigner:
    """
    Request signer with replay prevention.

    Signs requests with HMAC and includes sequence numbers
    to prevent replay attacks.
    """

    def __init__(self, secret_key: bytes):
        """
        Initialize request signer.

        Args:
            secret_key: Secret key for HMAC signing (must be kept secure)
        """
        self.secret_key = secret_key
        self.sequence_number = 0
        self.server_timestamp_offset = 0  # For clock synchronization

    def sign_request(self, method: str, url: str, body: Optional[str] = None,
                    headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Sign HTTP request with replay prevention.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            body: Request body (JSON string)
            headers: Request headers

        Returns:
            Dictionary of security headers to add to request
        """
        # Generate timestamp (with offset for clock sync)
        timestamp = int(time.time() * 1000) + self.server_timestamp_offset

        # Increment sequence number
        self.sequence_number += 1

        # Create signature payload
        payload_parts = [
            method.upper(),
            url,
            body or "",
            str(timestamp),
            str(self.sequence_number)
        ]

        payload = "\n".join(payload_parts)

        # Create HMAC signature
        signature = hmac.new(
            self.secret_key,
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        # Return security headers
        return {
            'X-Request-Timestamp': str(timestamp),
            'X-Request-Sequence': str(self.sequence_number),
            'X-Request-Signature': signature,
            'X-Request-Algorithm': 'HMAC-SHA256'
        }

    def validate_response_signature(self, response_body: str,
                                  response_headers: Dict[str, str]) -> bool:
        """
        Validate response signature if server supports signed responses.

        Args:
            response_body: Response body
            response_headers: Response headers

        Returns:
            True if signature is valid or not present (optional feature)
        """
        # Check if server provides response signature
        signature = response_headers.get('X-Response-Signature')
        if not signature:
            return True  # Optional feature, not required

        try:
            # Recreate payload for signature verification
            timestamp = response_headers.get('X-Response-Timestamp')
            sequence = response_headers.get('X-Response-Sequence')

            if not timestamp or not sequence:
                return False

            payload = f"{response_body}\n{timestamp}\n{sequence}"

            # Verify HMAC signature
            expected_signature = hmac.new(
                self.secret_key,
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            is_valid = hmac.compare_digest(signature, expected_signature)

            if not is_valid:
                logger.warning("Response signature validation failed")

            return is_valid

        except Exception as e:
            logger.error(f"Response signature validation error: {e}")
            return False

    def synchronize_clock(self, server_timestamp: int) -> None:
        """
        Synchronize local clock with server timestamp.

        Args:
            server_timestamp: Server timestamp in milliseconds
        """
        local_timestamp = int(time.time() * 1000)
        self.server_timestamp_offset = server_timestamp - local_timestamp

        logger.debug(f"Clock synchronized with offset: {self.server_timestamp_offset}ms")


class SecureHTTPClient:
    """
    Secure HTTP client with SSL pinning, request signing, and replay prevention.

    Provides hardened HTTP communication for trading APIs with comprehensive
    security features and validation.
    """

    def __init__(
        self,
        ssl_validator: Optional[SSLFingerprintValidator] = None,
        request_signer: Optional[RequestSigner] = None,
        timeout: int = 30,
        max_concurrent_requests: int = 10,
        rate_limit_per_second: float = 5.0
    ):
        """
        Initialize secure HTTP client.

        Args:
            ssl_validator: SSL certificate validator for pinning
            request_signer: Request signer for authentication
            timeout: Request timeout in seconds
            max_concurrent_requests: Maximum concurrent requests
            rate_limit_per_second: Rate limit for requests per second
        """
        self.ssl_validator = ssl_validator
        self.request_signer = request_signer
        self.timeout = timeout
        self.max_concurrent_requests = max_concurrent_requests
        self.rate_limit_per_second = rate_limit_per_second

        # Request tracking for replay prevention
        self._request_history: Dict[str, float] = {}
        self._history_ttl = 300  # 5 minutes

        # Rate limiting
        self._last_request_time = 0.0
        self._request_semaphore = asyncio.Semaphore(max_concurrent_requests)

        # Connection pooling
        self._connector = aiohttp.TCPConnector(
            limit=max_concurrent_requests,
            limit_per_host=max_concurrent_requests // 2,
            ttl_dns_cache=300,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
        )

        # SSL context for certificate pinning
        if ssl_validator:
            self._connector.ssl = ssl_validator.create_ssl_context()

        logger.info("Secure HTTP client initialized")

    async def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
        json_data: Optional[Dict] = None,
        params: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True
    ) -> Tuple[int, Dict[str, str], str]:
        """
        Make secure HTTP request with all security features enabled.

        Args:
            method: HTTP method
            url: Request URL
            headers: Request headers
            data: Raw request data
            json_data: JSON data to send
            params: URL parameters
            verify_ssl: Whether to verify SSL certificates

        Returns:
            Tuple of (status_code, response_headers, response_body)
        """
        async with self._request_semaphore:
            # Apply rate limiting
            await self._apply_rate_limit()

            # Prepare request data
            request_headers = headers or {}
            request_body = None

            if json_data is not None:
                request_body = json.dumps(json_data, separators=(',', ':'))
                request_headers['Content-Type'] = 'application/json'
            elif data is not None:
                request_body = str(data) if not isinstance(data, str) else data

            # Add security headers if signer is available
            if self.request_signer:
                security_headers = self.request_signer.sign_request(
                    method, url, request_body, request_headers
                )
                request_headers.update(security_headers)

            # Check for replay attacks (local tracking)
            request_id = self._generate_request_id(method, url, request_body)
            if self._is_replay_attack(request_id):
                raise ExchangeAPIError("Request replay detected")

            # Track request
            self._track_request(request_id)

            # Make request
            timeout_config = aiohttp.ClientTimeout(total=self.timeout)

            try:
                async with aiohttp.ClientSession(
                    connector=self._connector,
                    timeout=timeout_config
                ) as session:
                    async with session.request(
                        method=method,
                        url=url,
                        headers=request_headers,
                        data=request_body,
                        params=params,
                        verify_ssl=verify_ssl
                    ) as response:

                        # Validate SSL certificate if pinning is enabled
                        if self.ssl_validator and verify_ssl:
                            await self._validate_ssl_certificate(url, response)

                        # Read response
                        response_body = await response.text()
                        response_headers = dict(response.headers)

                        # Validate response signature if available
                        if self.request_signer:
                            if not self.request_signer.validate_response_signature(
                                response_body, response_headers
                            ):
                                logger.warning("Response signature validation failed")

                        # Validate timestamp if present
                        self._validate_response_timestamp(response_headers)

                        return response.status, response_headers, response_body

            except asyncio.TimeoutError:
                raise ExchangeTimeoutError(f"Request timeout after {self.timeout}s")
            except ClientError as e:
                raise ExchangeNetworkError(f"Network error: {e}")
            except Exception as e:
                raise ExchangeError(f"HTTP request failed: {e}")

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting to requests."""
        current_time = time.time()

        if self.rate_limit_per_second > 0:
            time_since_last_request = current_time - self._last_request_time
            min_interval = 1.0 / self.rate_limit_per_second

            if time_since_last_request < min_interval:
                sleep_time = min_interval - time_since_last_request
                await asyncio.sleep(sleep_time)

        self._last_request_time = time.time()

    async def _validate_ssl_certificate(self, url: str, response) -> None:
        """Validate SSL certificate against pinned fingerprints."""
        if not self.ssl_validator:
            return

        try:
            hostname = urlparse(url).hostname
            if hostname:
                # Get peer certificate
                peer_cert = response.connection.transport.get_extra_info('peercert')
                if peer_cert:
                    # Convert to DER format
                    cert_der = ssl.PEM_cert_to_DER_cert(peer_cert)

                    if not self.ssl_validator.validate_certificate(hostname, cert_der):
                        raise ExchangeAPIError("SSL certificate validation failed")

        except Exception as e:
            logger.error(f"SSL certificate validation error: {e}")
            raise ExchangeAPIError("SSL certificate validation failed")

    def _generate_request_id(self, method: str, url: str, body: Optional[str]) -> str:
        """Generate unique request ID for replay prevention."""
        payload = f"{method}:{url}:{body or ''}"
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def _is_replay_attack(self, request_id: str) -> bool:
        """Check if request is a replay attack."""
        current_time = time.time()

        # Clean old entries
        cutoff_time = current_time - self._history_ttl
        self._request_history = {
            req_id: timestamp
            for req_id, timestamp in self._request_history.items()
            if timestamp > cutoff_time
        }

        # Check if request ID exists and is recent
        if request_id in self._request_history:
            return True

        return False

    def _track_request(self, request_id: str) -> None:
        """Track request for replay prevention."""
        self._request_history[request_id] = time.time()

    def _validate_response_timestamp(self, headers: Dict[str, str]) -> None:
        """Validate response timestamp to prevent replay attacks."""
        timestamp_str = headers.get('X-Response-Timestamp')
        if not timestamp_str:
            return  # Optional validation

        try:
            server_timestamp = int(timestamp_str)
            current_timestamp = int(time.time() * 1000)

            # Allow 30-second window for timestamp validation
            time_diff = abs(current_timestamp - server_timestamp)

            if time_diff > 30000:  # 30 seconds in milliseconds
                logger.warning(f"Response timestamp validation failed: diff={time_diff}ms")
                raise ExchangeAPIError("Response timestamp outside valid window")

        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid response timestamp: {e}")

    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        if self._connector:
            await self._connector.close()
            logger.debug("Secure HTTP client closed")


# Pre-configured SSL fingerprints for major exchanges (example - update with real fingerprints)
HYPERLIQUID_SSL_FINGERPRINTS = {
    "api.hyperliquid.xyz": [
        # Add real SHA-256 fingerprints here
        # Example: "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
    ],
    "api.hyperliquid-testnet.xyz": [
        # Add real SHA-256 fingerprints here
    ]
}


def create_secure_hyperliquid_client(
    private_key: str,
    enable_ssl_pinning: bool = False,
    enable_request_signing: bool = False
) -> SecureHTTPClient:
    """
    Create a secure HTTP client configured for Hyperliquid API.

    Args:
        private_key: Private key for request signing
        enable_ssl_pinning: Enable SSL certificate pinning
        enable_request_signing: Enable request signing

    Returns:
        Configured secure HTTP client
    """
    # SSL validator
    ssl_validator = None
    if enable_ssl_pinning:
        ssl_validator = SSLFingerprintValidator(HYPERLIQUID_SSL_FINGERPRINTS)
        logger.info("SSL certificate pinning enabled")

    # Request signer
    request_signer = None
    if enable_request_signing:
        # Use private key hash as signing key (in production, use separate signing key)
        signing_key = hashlib.sha256(private_key.encode('utf-8')).digest()
        request_signer = RequestSigner(signing_key)
        logger.info("Request signing enabled")

    # Create client
    client = SecureHTTPClient(
        ssl_validator=ssl_validator,
        request_signer=request_signer,
        timeout=30,
        max_concurrent_requests=10,
        rate_limit_per_second=5.0
    )

    return client
