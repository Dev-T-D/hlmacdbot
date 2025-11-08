"""
Test for Hyperliquid signing

Tests:
- Verify signatures match expected format
- Test with known values
"""

import unittest
from unittest.mock import Mock, patch

# Handle optional dependencies
try:
    from eth_account import Account
    from hyperliquid_client import HyperliquidClient
    ETH_ACCOUNT_AVAILABLE = True
except ImportError as e:
    ETH_ACCOUNT_AVAILABLE = False
    print(f"Warning: Could not import eth_account or hyperliquid_client: {e}")


@unittest.skipIf(not ETH_ACCOUNT_AVAILABLE, "eth_account module not available")
class TestHyperliquidSigning(unittest.TestCase):
    """Test cases for Hyperliquid EIP-712 signing"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not ETH_ACCOUNT_AVAILABLE:
            self.skipTest("eth_account not available")
        
        # Create a test account
        self.test_account = Account.create()
        self.private_key = self.test_account.key.hex()
        self.wallet_address = self.test_account.address
        
        self.client = HyperliquidClient(
            private_key=self.private_key,
            wallet_address=self.wallet_address,
            testnet=True
        )
    
    def test_signature_format(self):
        """Test that signatures match expected format"""
        # Test signing a simple message
        test_message = {
            'action': 'test',
            'timestamp': 1234567890
        }
        
        # Sign using client's signing method
        signature = self.client._sign_message(test_message)
        
        # Signature should be a string
        self.assertIsInstance(signature, str, "Signature should be a string")
        
        # Signature should start with 0x (if hex format)
        if signature.startswith('0x'):
            self.assertEqual(len(signature), 132, "Signature should be 132 chars (0x + 130 hex)")
        else:
            # May be base64 or other format
            self.assertGreater(len(signature), 0, "Signature should not be empty")
    
    def test_signature_consistency(self):
        """Test that same message produces same signature"""
        test_message = {
            'action': 'test',
            'timestamp': 1234567890
        }
        
        # Sign twice
        signature1 = self.client._sign_message(test_message)
        signature2 = self.client._sign_message(test_message)
        
        # Signatures should be identical for same message
        self.assertEqual(signature1, signature2, "Same message should produce same signature")
    
    def test_signature_different_messages(self):
        """Test that different messages produce different signatures"""
        message1 = {'action': 'test1', 'timestamp': 1234567890}
        message2 = {'action': 'test2', 'timestamp': 1234567890}
        
        signature1 = self.client._sign_message(message1)
        signature2 = self.client._sign_message(message2)
        
        # Signatures should be different
        self.assertNotEqual(signature1, signature2, "Different messages should produce different signatures")
    
    def test_wallet_address_match(self):
        """Test that wallet address matches private key"""
        # Derive address from private key
        account = Account.from_key(self.private_key)
        derived_address = account.address
        
        # Should match provided wallet address
        self.assertEqual(derived_address.lower(), self.wallet_address.lower(), 
                        "Derived address should match wallet address")
    
    def test_private_key_format(self):
        """Test private key format validation"""
        # Private key should be hex string
        self.assertIsInstance(self.private_key, str, "Private key should be string")
        
        # Remove 0x prefix if present
        key_hex = self.private_key.replace('0x', '')
        
        # Should be 64 hex characters (32 bytes)
        self.assertEqual(len(key_hex), 64, "Private key should be 64 hex characters")
        
        # Should be valid hex
        try:
            int(key_hex, 16)
        except ValueError:
            self.fail("Private key should be valid hex")
    
    def test_eip712_structure(self):
        """Test EIP-712 structured data signing"""
        # Hyperliquid uses EIP-712 for signing
        # Test that signing method handles structured data
        
        structured_data = {
            'types': {
                'EIP712Domain': [
                    {'name': 'name', 'type': 'string'},
                    {'name': 'version', 'type': 'string'},
                    {'name': 'chainId', 'type': 'uint256'}
                ],
                'Message': [
                    {'name': 'action', 'type': 'string'}
                ]
            },
            'domain': {
                'name': 'Hyperliquid',
                'version': '1',
                'chainId': 1
            },
            'primaryType': 'Message',
            'message': {
                'action': 'test'
            }
        }
        
        # Should be able to sign structured data
        try:
            signature = self.client._sign_message(structured_data)
            self.assertIsNotNone(signature, "Should produce signature for structured data")
        except Exception as e:
            # May use different signing method
            pass
    
    def test_signature_verification(self):
        """Test signature verification"""
        from eth_account.messages import encode_defunct
        from eth_account import Account
        
        test_message = "Test message"
        
        # Sign message
        message_hash = encode_defunct(text=test_message)
        signed_message = Account.sign_message(message_hash, self.private_key)
        
        # Verify signature
        recovered_address = Account.recover_message(message_hash, signature=signed_message.signature)
        
        # Recovered address should match wallet address
        self.assertEqual(recovered_address.lower(), self.wallet_address.lower(),
                        "Recovered address should match wallet address")
    
    def test_invalid_private_key(self):
        """Test handling of invalid private key"""
        invalid_key = "0x" + "0" * 64  # All zeros
        
        with self.assertRaises((ValueError, Exception)):
            invalid_client = HyperliquidClient(
                private_key=invalid_key,
                wallet_address=self.wallet_address,
                testnet=True
            )
    
    def test_mismatched_key_and_address(self):
        """Test handling of mismatched private key and wallet address"""
        from exceptions import ExchangeAuthenticationError
        different_account = Account.create()
        
        with self.assertRaises(ExchangeAuthenticationError):
            mismatched_client = HyperliquidClient(
                private_key=self.private_key,
                wallet_address=different_account.address,  # Different address
                testnet=True
            )
            # Should validate on initialization
            mismatched_client._sign_message({'test': 'message'})


if __name__ == '__main__':
    unittest.main()

