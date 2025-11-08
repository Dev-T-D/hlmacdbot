#!/usr/bin/env python3
"""
Credential Management CLI Tool

Provides command-line interface for managing credentials securely.

Usage:
    python manage_credentials.py set <key> <value>
    python manage_credentials.py get <key>
    python manage_credentials.py delete <key>
    python manage_credentials.py list
    python manage_credentials.py test

Examples:
    python manage_credentials.py set hyperliquid_private_key 0x...
    python manage_credentials.py get hyperliquid_private_key
    python manage_credentials.py delete hyperliquid_private_key
    python manage_credentials.py list
"""

import sys
import getpass
import argparse
import logging
from credential_manager import CredentialManager, KEYRING_AVAILABLE

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Manage trading bot credentials securely',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s set hyperliquid_private_key 0x...
  %(prog)s get hyperliquid_private_key
  %(prog)s delete hyperliquid_private_key
  %(prog)s list
  %(prog)s test

Credential Keys:
  - hyperliquid_private_key
  - hyperliquid_wallet_address
  - bitunix_api_key
  - bitunix_secret_key
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Set command
    set_parser = subparsers.add_parser('set', help='Store a credential')
    set_parser.add_argument('key', help='Credential key name')
    set_parser.add_argument('value', nargs='?', help='Credential value (prompted if not provided)')
    
    # Get command
    get_parser = subparsers.add_parser('get', help='Retrieve a credential')
    get_parser.add_argument('key', help='Credential key name')
    get_parser.add_argument('--show', action='store_true', help='Show credential value (default: masked)')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a credential')
    delete_parser.add_argument('key', help='Credential key name')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all stored credentials')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test credential retrieval')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Check keyring availability
    if not KEYRING_AVAILABLE:
        logger.error(
            "Keyring module not available!\n"
            "Install with: pip install keyring\n"
            "Falling back to environment variables and config file."
        )
        if args.command in ['set', 'delete']:
            sys.exit(1)
    
    manager = CredentialManager(use_keyring=True)
    
    try:
        if args.command == 'set':
            key = args.key
            value = args.value
            
            if not value:
                # Prompt for value securely
                value = getpass.getpass(f"Enter value for {key}: ")
                if not value:
                    logger.error("No value provided")
                    sys.exit(1)
            
            # Validate key
            valid_keys = [
                CredentialManager.HYPERLIQUID_PRIVATE_KEY,
                CredentialManager.HYPERLIQUID_WALLET_ADDRESS,
                CredentialManager.BITUNIX_API_KEY,
                CredentialManager.BITUNIX_SECRET_KEY,
            ]
            
            if key not in valid_keys:
                logger.warning(
                    f"Unknown key: {key}\n"
                    f"Valid keys: {', '.join(valid_keys)}"
                )
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    sys.exit(1)
            
            manager.set_credential(key, value)
            logger.info(f"✅ Successfully stored {key}")
            
        elif args.command == 'get':
            key = args.key
            credential = manager.get_credential(key, env_var=None, config_value=None)
            
            if credential:
                if args.show:
                    logger.info(f"{key}: {credential}")
                else:
                    # Mask sensitive values
                    if len(credential) > 8:
                        masked = credential[:4] + '*' * (len(credential) - 8) + credential[-4:]
                    else:
                        masked = '*' * len(credential)
                    logger.info(f"{key}: {masked}")
            else:
                logger.warning(f"Credential {key} not found")
                logger.info("Check environment variables or config file")
                sys.exit(1)
                
        elif args.command == 'delete':
            key = args.key
            if manager.delete_credential(key):
                logger.info(f"✅ Successfully deleted {key}")
            else:
                logger.warning(f"Credential {key} not found in keyring")
                
        elif args.command == 'list':
            credentials = manager.list_credentials()
            
            print("\n📋 Stored Credentials:")
            print("=" * 60)
            
            for key, exists in credentials.items():
                status = "✅ Stored" if exists else "❌ Not stored"
                print(f"{key:30} {status}")
            
            print("=" * 60)
            print("\nNote: Credentials may also be available via:")
            print("  - Environment variables")
            print("  - Config file (config/config.json)")
            
        elif args.command == 'test':
            print("\n🧪 Testing Credential Retrieval:")
            print("=" * 60)
            
            # Test Hyperliquid credentials
            test_config = {
                'private_key': None,
                'wallet_address': None,
            }
            
            hyperliquid_creds = manager.get_hyperliquid_credentials(test_config)
            print(f"\nHyperliquid Private Key: {'✅ Found' if hyperliquid_creds['private_key'] else '❌ Not found'}")
            print(f"Hyperliquid Wallet Address: {'✅ Found' if hyperliquid_creds['wallet_address'] else '❌ Not found'}")
            
            # Test Bitunix credentials
            test_config = {
                'api_key': None,
                'secret_key': None,
            }
            
            bitunix_creds = manager.get_bitunix_credentials(test_config)
            print(f"\nBitunix API Key: {'✅ Found' if bitunix_creds['api_key'] else '❌ Not found'}")
            print(f"Bitunix Secret Key: {'✅ Found' if bitunix_creds['secret_key'] else '❌ Not found'}")
            
            print("\n" + "=" * 60)
            print("\nPriority order:")
            print("  1. System keyring (most secure)")
            print("  2. Environment variables")
            print("  3. Config file (with warnings)")
            
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

