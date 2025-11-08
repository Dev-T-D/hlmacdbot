"""
Secure Credential Storage Manager

Provides secure credential storage using the system keyring with fallbacks
to environment variables and config files. Supports both Hyperliquid and Bitunix exchanges.

Priority order:
1. System keyring (most secure)
2. Environment variables
3. Config file (with warnings)
"""

import os
import logging
from typing import Optional, Dict
from exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Try to import keyring, but make it optional
try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    logger.warning(
        "keyring module not available. Install with: pip install keyring\n"
        "Falling back to environment variables and config file."
    )


class CredentialManager:
    """
    Manages secure credential storage and retrieval.
    
    Uses system keyring (Windows Credential Manager, macOS Keychain, Linux Secret Service)
    with fallbacks to environment variables and config files.
    """
    
    # Service name for keyring (identifies this application)
    KEYRING_SERVICE = "bitunix-macd-bot"
    
    # Credential keys
    HYPERLIQUID_PRIVATE_KEY = "hyperliquid_private_key"
    HYPERLIQUID_WALLET_ADDRESS = "hyperliquid_wallet_address"
    
    def __init__(self, use_keyring: bool = True):
        """
        Initialize credential manager.
        
        Args:
            use_keyring: If True, attempt to use keyring (if available)
        """
        self.use_keyring = use_keyring and KEYRING_AVAILABLE
        
        if self.use_keyring:
            logger.debug("Credential manager initialized with keyring support")
        else:
            if use_keyring and not KEYRING_AVAILABLE:
                logger.info(
                    "Keyring not available. Using environment variables and config file only. "
                    "Install keyring for secure credential storage: pip install keyring"
                )
            else:
                logger.debug("Credential manager initialized without keyring")
    
    def get_credential(self, key: str, env_var: Optional[str] = None, 
                      config_value: Optional[str] = None) -> Optional[str]:
        """
        Get credential from secure storage with fallback chain.
        
        Priority:
        1. System keyring
        2. Environment variable
        3. Config file value
        
        Args:
            key: Credential key name (for keyring)
            env_var: Environment variable name (optional)
            config_value: Value from config file (optional)
            
        Returns:
            Credential value or None if not found
        """
        # Try keyring first (most secure)
        if self.use_keyring:
            try:
                credential = keyring.get_password(self.KEYRING_SERVICE, key)
                if credential:
                    logger.debug(f"✅ Loaded {key} from system keyring")
                    return credential
            except Exception as e:
                logger.warning(
                    f"Failed to retrieve {key} from keyring: {e}. "
                    f"Falling back to environment variables/config file."
                )
        
        # Fallback to environment variable
        if env_var:
            credential = os.getenv(env_var)
            if credential:
                logger.debug(f"✅ Loaded {key} from environment variable {env_var}")
                return credential
        
        # Fallback to config file (with warning)
        if config_value:
            logger.warning(
                f"⚠️  SECURITY WARNING: {key} found in config file. "
                f"Consider using secure storage:\n"
                f"  - System keyring: python -m credential_manager set {key} <value>\n"
                f"  - Environment variable: export {env_var}=<value>"
            )
            return config_value
        
        return None
    
    def set_credential(self, key: str, value: str) -> bool:
        """
        Store credential in system keyring.
        
        Args:
            key: Credential key name
            value: Credential value to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.use_keyring:
            raise ConfigurationError(
                "Keyring not available. Install with: pip install keyring"
            )
        
        try:
            keyring.set_password(self.KEYRING_SERVICE, key, value)
            logger.info(f"✅ Stored {key} in system keyring")
            return True
        except Exception as e:
            logger.error(f"Failed to store {key} in keyring: {e}")
            raise ConfigurationError(
                f"Failed to store credential in keyring: {e}"
            ) from e
    
    def delete_credential(self, key: str) -> bool:
        """
        Delete credential from system keyring.
        
        Args:
            key: Credential key name
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if not self.use_keyring:
            raise ConfigurationError(
                "Keyring not available. Install with: pip install keyring"
            )
        
        try:
            keyring.delete_password(self.KEYRING_SERVICE, key)
            logger.info(f"✅ Deleted {key} from system keyring")
            return True
        except keyring.errors.PasswordDeleteError:
            logger.warning(f"Credential {key} not found in keyring")
            return False
        except Exception as e:
            logger.error(f"Failed to delete {key} from keyring: {e}")
            raise ConfigurationError(
                f"Failed to delete credential from keyring: {e}"
            ) from e
    
    def has_credential(self, key: str) -> bool:
        """
        Check if credential exists in keyring.
        
        Args:
            key: Credential key name
            
        Returns:
            True if credential exists, False otherwise
        """
        if not self.use_keyring:
            return False
        
        try:
            credential = keyring.get_password(self.KEYRING_SERVICE, key)
            return credential is not None
        except Exception:
            return False
    
    def get_hyperliquid_credentials(self, config: Dict) -> Dict[str, Optional[str]]:
        """
        Get Hyperliquid credentials with fallback chain.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dict with 'private_key' and 'wallet_address'
        """
        private_key = self.get_credential(
            self.HYPERLIQUID_PRIVATE_KEY,
            env_var='HYPERLIQUID_PRIVATE_KEY',
            config_value=config.get('private_key')
        )
        
        wallet_address = self.get_credential(
            self.HYPERLIQUID_WALLET_ADDRESS,
            env_var='HYPERLIQUID_WALLET_ADDRESS',
            config_value=config.get('wallet_address')
        )
        
        return {
            'private_key': private_key,
            'wallet_address': wallet_address
        }
    
    def list_credentials(self) -> Dict[str, bool]:
        """
        List all stored credentials (without values).
        
        Returns:
            Dict mapping credential keys to existence status
        """
        credentials = {
            self.HYPERLIQUID_PRIVATE_KEY: self.has_credential(self.HYPERLIQUID_PRIVATE_KEY),
            self.HYPERLIQUID_WALLET_ADDRESS: self.has_credential(self.HYPERLIQUID_WALLET_ADDRESS),
        }
        return credentials

