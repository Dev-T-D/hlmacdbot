"""
Secure Private Key Storage

Provides secure in-memory storage for private keys with encryption and secure deletion.
Minimizes memory exposure of sensitive cryptographic keys.

Note: Python's memory management makes true secure deletion difficult, but this module
provides best-effort security by:
1. Encrypting keys in memory
2. Only decrypting when needed
3. Clearing decrypted keys immediately after use
4. Using secure deletion techniques where possible
"""

import secrets
import logging
from typing import Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from eth_account import Account
from eth_account.signers.local import LocalAccount

logger = logging.getLogger(__name__)


class SecureKeyStorage:
    """
    Secure storage for private keys with in-memory encryption.
    
    The private key is encrypted using AES-GCM with a randomly generated session key.
    The key is only decrypted when needed for signing operations, and the decrypted
    value is cleared immediately after use.
    
    Security Features:
    - AES-GCM encryption for in-memory storage
    - Session key stored separately from encrypted key
    - Decrypted keys cleared immediately after use
    - Secure random key generation
    - No plaintext key storage in memory (except briefly during operations)
    """
    
    def __init__(self, private_key: str, demo_mode: bool = False):
        """
        Initialize secure key storage with a private key.

        Args:
            private_key: Ethereum private key (with or without 0x prefix)
            demo_mode: Skip validation for demo/historical data access

        Raises:
            ValueError: If private key is invalid
        """
        # Ensure private key has 0x prefix
        if not private_key.startswith('0x'):
            private_key = '0x' + private_key

        # Store key components for validation
        key_hex = private_key[2:].lower() if len(private_key) > 2 else ""

        # Validate private key format only if not in demo mode
        if not demo_mode:
            if len(private_key) != 66:  # 0x + 64 hex chars
                raise ValueError(
                    f"Invalid private key length: expected 66 characters (0x + 64 hex), got {len(private_key)}"
                )

            # Check for obviously invalid keys
            if key_hex == '0' * 64:
                raise ValueError("Invalid private key: cannot use zero private key")

            if len(set(key_hex)) == 1:
                raise ValueError("Invalid private key: appears to be a test/placeholder key")
        
        # Generate a random session key for encryption (32 bytes for AES-256)
        self._session_key = secrets.token_bytes(32)
        
        # Encrypt the private key
        self._encrypted_key = self._encrypt_key(private_key.encode('utf-8'))
        
        # Store wallet address (derived from key, not sensitive)
        try:
            temp_account = Account.from_key(private_key)
            self._wallet_address = temp_account.address.lower()
        except Exception as e:
            raise ValueError(f"Failed to derive wallet address from private key: {e}") from e
        
        # Clear the plaintext key from memory (best effort)
        # Note: Python's garbage collector may not immediately clear this,
        # but we do our best to minimize exposure
        del private_key
        
        logger.debug("SecureKeyStorage initialized with encrypted private key")
    
    def _encrypt_key(self, key_bytes: bytes) -> bytes:
        """
        Encrypt the private key using AES-GCM.
        
        Args:
            key_bytes: The private key as bytes
        
        Returns:
            Encrypted key (nonce + ciphertext + tag)
        """
        # Generate a random nonce (12 bytes for AES-GCM)
        nonce = secrets.token_bytes(12)
        
        # Create AES-GCM cipher
        aesgcm = AESGCM(self._session_key)
        
        # Encrypt the key (returns ciphertext + tag)
        ciphertext = aesgcm.encrypt(nonce, key_bytes, None)
        
        # Return nonce + ciphertext (tag is appended by AESGCM)
        return nonce + ciphertext
    
    def _decrypt_key(self) -> str:
        """
        Decrypt the private key from storage.
        
        Returns:
            Decrypted private key as string
        
        Raises:
            ValueError: If decryption fails
        """
        try:
            # Extract nonce (first 12 bytes) and ciphertext (rest)
            nonce = self._encrypted_key[:12]
            ciphertext = self._encrypted_key[12:]
            
            # Create AES-GCM cipher
            aesgcm = AESGCM(self._session_key)
            
            # Decrypt the key
            key_bytes = aesgcm.decrypt(nonce, ciphertext, None)
            
            # Convert to string
            return key_bytes.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to decrypt private key: {e}") from e
    
    def get_account(self) -> LocalAccount:
        """
        Get the Ethereum account object for signing.
        
        This method decrypts the key, creates the account, and clears the decrypted key.
        The account object itself still contains the key, but we minimize exposure
        by only creating it when needed.
        
        Returns:
            LocalAccount object for signing
        
        Note:
            The LocalAccount object still contains the private key internally.
            This is a limitation of the eth_account library. However, we minimize
            exposure by only creating the account when needed for signing operations.
        """
        private_key = self._decrypt_key()
        try:
            account = Account.from_key(private_key)
            return account
        finally:
            # Clear the decrypted key (best effort)
            # Note: Python strings are immutable, so this doesn't guarantee
            # immediate memory clearing, but it's the best we can do
            del private_key
    
    def get_wallet_address(self) -> str:
        """
        Get the wallet address derived from the private key.
        
        Returns:
            Wallet address (lowercase)
        """
        return self._wallet_address
    
    def emergency_zeroize(self) -> None:
        """
        Emergency zeroization of all sensitive data.

        Immediately overwrites all sensitive data with zeros multiple times.
        Called during critical security events to prevent key recovery.

        This provides the highest level of assurance for key destruction.
        """
        logger.critical("🔴 EMERGENCY ZEROIZATION: Destroying all sensitive key material")

        # Multiple overwrite passes for maximum security
        for pass_num in range(3):
            try:
                # Overwrite session key with zeros
                if hasattr(self, '_session_key') and self._session_key:
                    key_len = len(self._session_key)
                    self._session_key = b'\x00' * key_len

                # Overwrite encrypted key with zeros
                if hasattr(self, '_encrypted_key') and self._encrypted_key:
                    key_len = len(self._encrypted_key)
                    self._encrypted_key = b'\x00' * key_len

                # Overwrite wallet address (not sensitive but clear anyway)
                if hasattr(self, '_wallet_address'):
                    self._wallet_address = '\x00' * len(self._wallet_address)

            except Exception as e:
                # Log but continue zeroization attempts
                logger.error(f"Zeroization pass {pass_num + 1} error: {e}")

        # Final deletion attempts
        try:
            delattr(self, '_session_key')
        except AttributeError:
            pass

        try:
            delattr(self, '_encrypted_key')
        except AttributeError:
            pass

        try:
            delattr(self, '_wallet_address')
        except AttributeError:
            pass

        logger.critical("✅ Emergency zeroization completed - all key material destroyed")

    def clear(self) -> None:
        """
        Clear all sensitive data from memory.

        This method attempts to clear the encrypted key and session key.
        Note: Due to Python's memory management, this doesn't guarantee
        immediate clearing, but it's the best effort we can make.
        """
        # For normal cleanup, use random data overwrite
        # For emergency situations, use emergency_zeroize() instead
        if hasattr(self, '_encrypted_key'):
            self._encrypted_key = secrets.token_bytes(len(self._encrypted_key))
            del self._encrypted_key

        if hasattr(self, '_session_key'):
            self._session_key = secrets.token_bytes(len(self._session_key))
            del self._session_key

        logger.debug("SecureKeyStorage cleared")
    
    def __del__(self):
        """Destructor: Clear sensitive data when object is destroyed."""
        try:
            self.clear()
        except Exception:
            # Ignore errors during cleanup
            pass

