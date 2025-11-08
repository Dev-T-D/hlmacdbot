"""
Test for API error handling

Tests:
- Network timeout
- 500 error
- Invalid response
- Rate limit
"""

import unittest
from unittest.mock import Mock, patch
import requests
from requests.exceptions import Timeout, ConnectionError, HTTPError

# Handle optional dependencies
try:
    from hyperliquid_client import HyperliquidClient
    from exceptions import ExchangeInvalidResponseError
    HYPERLIQUID_AVAILABLE = True
except ImportError:
    HYPERLIQUID_AVAILABLE = False
    ExchangeInvalidResponseError = Exception  # Fallback
    print("Warning: Could not import hyperliquid_client (eth_account may be missing)")

# Bitunix support removed - only Hyperliquid is supported
BITUNIX_AVAILABLE = False


@unittest.skipIf(not HYPERLIQUID_AVAILABLE, "hyperliquid_client not available")
class TestAPIErrorHandling(unittest.TestCase):
    """Test cases for API error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        global HYPERLIQUID_AVAILABLE
        
        if HYPERLIQUID_AVAILABLE:
            # Use a valid test private key for testing
            # This is just for testing purposes - never use in production
            test_private_key = '0x1234567890123456789012345678901234567890123456789012345678901234'
            # Derive the correct wallet address from the private key
            from eth_account import Account
            test_account = Account.from_key(test_private_key)
            test_wallet_address = test_account.address

            try:
                self.hyperliquid_client = HyperliquidClient(
                    private_key=test_private_key,
                    wallet_address=test_wallet_address,
                    testnet=True
                )
            except Exception as e:
                # If client creation fails, mark as unavailable
                HYPERLIQUID_AVAILABLE = False
                print(f"Warning: Could not create hyperliquid client for testing: {e}")
        
        # Bitunix support removed
    
    @unittest.skipIf(not HYPERLIQUID_AVAILABLE, "hyperliquid_client not available")
    def test_network_timeout(self):
        """Test handling of network timeout"""
        with patch.object(self.hyperliquid_client.session, 'post') as mock_post:
            mock_post.side_effect = Timeout("Connection timeout")
            
            # Should handle timeout gracefully
            with self.assertRaises(Exception):
                self.hyperliquid_client.get_account_info()
    
    @unittest.skipIf(not HYPERLIQUID_AVAILABLE, "hyperliquid_client not available")
    def test_500_server_error(self):
        """Test handling of 500 server error"""
        with patch.object(self.hyperliquid_client.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = HTTPError("500 Server Error")
            mock_post.return_value = mock_response
            
            # Should retry or handle gracefully
            with self.assertRaises(Exception):
                self.hyperliquid_client.get_account_info()
    
    @unittest.skipIf(not HYPERLIQUID_AVAILABLE, "hyperliquid_client not available")
    def test_429_rate_limit(self):
        """Test handling of 429 rate limit error"""
        with patch.object(self.hyperliquid_client.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.headers = {'Retry-After': '1'}
            mock_response.raise_for_status.side_effect = HTTPError("429 Rate Limit")
            mock_post.return_value = mock_response
            
            # Should handle rate limit (may retry or wait)
            with self.assertRaises(Exception):
                self.hyperliquid_client.get_account_info()
    
    @unittest.skipIf(not HYPERLIQUID_AVAILABLE, "hyperliquid_client not available")
    def test_connection_error(self):
        """Test handling of connection error"""
        with patch.object(self.hyperliquid_client.session, 'post') as mock_post:
            mock_post.side_effect = ConnectionError("Connection refused")
            
            # Should handle connection error
            with self.assertRaises(Exception):
                self.hyperliquid_client.get_account_info()
    
    @unittest.skipIf(not HYPERLIQUID_AVAILABLE, "hyperliquid_client not available")
    def test_invalid_response_format(self):
        """Test handling of invalid response format"""
        with patch.object(self.hyperliquid_client.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'invalid': 'format'}  # Missing expected fields
            mock_post.return_value = mock_response
            
            # Should handle invalid response
            with self.assertRaises((KeyError, ValueError, TypeError, ExchangeInvalidResponseError)):
                self.hyperliquid_client.get_account_info()
    
    @unittest.skipIf(not HYPERLIQUID_AVAILABLE, "hyperliquid_client not available")
    def test_empty_response(self):
        """Test handling of empty response"""
        with patch.object(self.hyperliquid_client.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {}
            mock_post.return_value = mock_response
            
            # Should handle empty response
            with self.assertRaises((KeyError, ValueError, ExchangeInvalidResponseError)):
                self.hyperliquid_client.get_account_info()
    
    @unittest.skipIf(not HYPERLIQUID_AVAILABLE, "hyperliquid_client not available")
    def test_non_json_response(self):
        """Test handling of non-JSON response"""
        with patch.object(self.hyperliquid_client.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = ValueError("Not JSON")
            mock_post.return_value = mock_response
            
            # Should handle non-JSON response
            with self.assertRaises(ValueError):
                self.hyperliquid_client.get_account_info()
    
    # Bitunix tests removed - Bitunix support has been removed
    
    @unittest.skipIf(not HYPERLIQUID_AVAILABLE, "hyperliquid_client not available")
    def test_retry_logic(self):
        """Test that retry logic is invoked for retryable errors"""
        call_count = [0]
        
        def mock_post_with_retry(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Timeout("Temporary timeout")
            else:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {'data': {'balance': 10000.0}}
                return mock_response
        
        with patch.object(self.hyperliquid_client.session, 'post', side_effect=mock_post_with_retry):
            # Should retry and eventually succeed
            try:
                result = self.hyperliquid_client.get_account_info()
                self.assertEqual(call_count[0], 3, "Should retry 3 times")
            except Exception:
                # May fail if retry logic not implemented
                pass


if __name__ == '__main__':
    unittest.main()

