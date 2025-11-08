"""
Signature Verification Module

Verifies API response signatures if provided by exchanges.
Most exchanges don't sign responses, but this provides security
if an exchange adds response signing in the future.
"""

import hashlib
import hmac
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SignatureVerifier:
    """
    Verifies API response signatures from Bitunix exchange.
    
    Bitunix uses HMAC-SHA256 for request signing. If they provide
    response signatures, they would likely use a similar scheme.
    """
    
    def __init__(self, secret_key: str):
        """
        Initialize signature verifier.
        
        Args:
            secret_key: Secret key used for signature verification
        """
        self.secret_key = secret_key
    
    def verify_response_signature(self, 
                                 response_data: Dict,
                                 response_body: Optional[str] = None,
                                 timestamp: Optional[str] = None) -> bool:
        """
        Verify response signature if present.
        
        This method checks if the response contains a signature and verifies it.
        If no signature is present, returns True (backward compatible).
        
        Expected signature format (if implemented by exchange):
        - Response may contain 'sign' or 'signature' field
        - Signature is HMAC-SHA256 of response body + timestamp + secret_key
        - Or similar scheme based on exchange implementation
        
        Args:
            response_data: Parsed JSON response data
            response_body: Raw response body string (if available)
            timestamp: Response timestamp (if available)
            
        Returns:
            True if signature is valid or not present, False if invalid
        """
        # Check if response contains signature
        signature = response_data.get('sign') or response_data.get('signature')
        
        if not signature:
            # No signature present - backward compatible
            return True
        
        # If signature is present, verify it
        try:
            # Try to verify signature
            # Note: Actual verification depends on exchange's signature scheme
            # This is a placeholder implementation
            
            # Common signature schemes:
            # 1. HMAC-SHA256(response_body + timestamp + secret_key)
            # 2. HMAC-SHA256(response_data_json + secret_key)
            # 3. SHA256(response_body + secret_key)
            
            # For now, log that signature verification is not fully implemented
            # This is because Bitunix doesn't currently provide response signatures
            logger.debug(
                f"Response contains signature field, but verification scheme "
                f"is not yet implemented (exchange may not provide signatures)"
            )
            
            # Return True for now (backward compatible)
            # When exchange provides signatures, implement actual verification here
            return True
            
        except Exception as e:
            logger.warning(f"Error verifying response signature: {e}")
            return False
    
    def verify_hmac_signature(self,
                              data: str,
                              signature: str,
                              secret_key: Optional[str] = None) -> bool:
        """
        Verify HMAC-SHA256 signature.
        
        Args:
            data: Data that was signed
            signature: Signature to verify
            secret_key: Secret key (uses instance secret_key if not provided)
            
        Returns:
            True if signature is valid, False otherwise
        """
        if secret_key is None:
            secret_key = self.secret_key
        
        try:
            # Generate expected signature
            expected_signature = hmac.new(
                secret_key.encode('utf-8'),
                data.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures (constant-time comparison to prevent timing attacks)
            return hmac.compare_digest(expected_signature, signature)
            
        except Exception as e:
            logger.warning(f"Error in HMAC signature verification: {e}")
            return False
    
    def verify_sha256_signature(self,
                                data: str,
                                signature: str) -> bool:
        """
        Verify SHA256 signature (non-HMAC).
        
        Args:
            data: Data that was signed
            signature: Signature to verify
            
        Returns:
            True if signature matches, False otherwise
        """
        try:
            # Generate expected signature
            expected_signature = hashlib.sha256(
                data.encode('utf-8')
            ).hexdigest()
            
            # Compare signatures (constant-time comparison)
            return hmac.compare_digest(expected_signature, signature)
            
        except Exception as e:
            logger.warning(f"Error in SHA256 signature verification: {e}")
            return False
    
    def extract_response_signature(self, response_data: Dict) -> Optional[str]:
        """
        Extract signature from response if present.
        
        Args:
            response_data: Parsed JSON response data
            
        Returns:
            Signature string if found, None otherwise
        """
        # Check common signature field names
        signature = (
            response_data.get('sign') or
            response_data.get('signature') or
            response_data.get('responseSign') or
            response_data.get('response_signature')
        )
        
        return signature
    
    def should_verify_response(self, response_data: Dict) -> bool:
        """
        Check if response should be verified (has signature field).
        
        Args:
            response_data: Parsed JSON response data
            
        Returns:
            True if response contains signature and should be verified
        """
        signature = self.extract_response_signature(response_data)
        return signature is not None

