"""
Test Hyperliquid Connection

Verifies Hyperliquid API credentials and connectivity
Tests all essential API methods before running the bot

"""

import json
import logging
import os
import sys
from credential_manager import CredentialManager
from hyperliquid_client import HyperliquidClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_hyperliquid_connection(config_path: str = "config/config.json"):
    """
    Test Hyperliquid connection and API methods
    
    Args:
        config_path: Path to configuration file
    """
    print("\n" + "=" * 70)
    print("HYPERLIQUID CONNECTION TEST")
    print("=" * 70)
    
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Verify exchange is set to hyperliquid
        exchange = config.get('exchange', '').lower()
        if exchange != 'hyperliquid':
            print(f"\n❌ ERROR: Config exchange is '{exchange}', should be 'hyperliquid'")
            print("   Update config.json: \"exchange\": \"hyperliquid\"")
            return False
        
        print(f"\n✅ Configuration loaded")
        print(f"   Exchange: {exchange}")
        print(f"   Testnet: {config.get('testnet', True)}")
        print(f"   Symbol: {config['trading']['symbol']}")
        
        # Load credentials using credential manager (supports env vars, keyring, config)
        print("\n🔐 Loading credentials...")
        credential_manager = CredentialManager(use_keyring=True)
        creds = credential_manager.get_hyperliquid_credentials(config)
        
        if not creds['private_key'] or not creds['wallet_address']:
            print("\n❌ ERROR: Could not load credentials")
            print("   Please set credentials using:")
            print("   - Environment variables: HYPERLIQUID_PRIVATE_KEY, HYPERLIQUID_WALLET_ADDRESS")
            print("   - System keyring: python manage_credentials.py set hyperliquid_private_key <key>")
            print("   - Config file: Update config.json (less secure)")
            return False
        
        # Initialize client
        print("\n📡 Initializing Hyperliquid client...")
        client = HyperliquidClient(
            private_key=creds['private_key'],
            wallet_address=creds['wallet_address'],
            testnet=config.get('testnet', True)
        )
        print(f"✅ Client initialized for wallet: {creds['wallet_address']}")
        
        # Test 1: Get Ticker
        print("\n" + "-" * 70)
        print("TEST 1: Get Ticker Data")
        print("-" * 70)
        try:
            symbol = config['trading']['symbol']
            ticker = client.get_ticker(symbol)
            if ticker and 'markPrice' in ticker:
                print(f"✅ Ticker data retrieved successfully")
                print(f"   Symbol: {ticker.get('symbol', symbol)}")
                print(f"   Mark Price: ${float(ticker['markPrice']):,.2f}")
                print(f"   Last Price: ${float(ticker.get('lastPrice', ticker['markPrice'])):,.2f}")
            else:
                print(f"⚠️  Warning: Ticker data incomplete")
                print(f"   Response: {ticker}")
        except Exception as e:
            print(f"❌ Ticker test failed: {e}")
            return False
        
        # Test 2: Get Account Info
        print("\n" + "-" * 70)
        print("TEST 2: Get Account Information")
        print("-" * 70)
        try:
            account_info = client.get_account_info()
            if account_info and 'balance' in account_info:
                balance = float(account_info['balance'])
                print(f"✅ Account info retrieved successfully")
                print(f"   Balance: ${balance:,.2f} USDT")
                print(f"   Account Value: ${float(account_info.get('accountValue', balance)):,.2f}")
                
                if balance < 10:
                    print(f"   ⚠️  Warning: Low balance. Consider adding funds for testing")
            else:
                print(f"⚠️  Warning: Account info incomplete")
                print(f"   Response: {account_info}")
        except Exception as e:
            print(f"❌ Account info test failed: {e}")
            return False
        
        # Test 3: Get Position
        print("\n" + "-" * 70)
        print("TEST 3: Get Current Position")
        print("-" * 70)
        try:
            symbol = config['trading']['symbol']
            position = client.get_position(symbol)
            if position:
                print(f"✅ Existing position found:")
                print(f"   Symbol: {position.get('symbol', symbol)}")
                print(f"   Side: {position.get('side', 'N/A')}")
                print(f"   Size: {position.get('holdAmount', '0')}")
                print(f"   Entry Price: ${float(position.get('entryPrice', 0)):,.2f}")
            else:
                print(f"✅ No open position for {symbol}")
        except Exception as e:
            print(f"❌ Position test failed: {e}")
            return False
        
        # Test 4: Get Klines
        print("\n" + "-" * 70)
        print("TEST 4: Get Candlestick Data")
        print("-" * 70)
        try:
            symbol = config['trading']['symbol']
            timeframe = config['trading']['timeframe']
            klines = client.get_klines(symbol, timeframe, limit=10)
            
            if klines and len(klines) > 0:
                print(f"✅ Kline data retrieved successfully")
                print(f"   Symbol: {symbol}")
                print(f"   Timeframe: {timeframe}")
                print(f"   Candles received: {len(klines)}")
                
                if len(klines) >= 10:
                    latest = klines[-1]
                    print(f"   Latest candle:")
                    print(f"     Open:  ${float(latest[1]):,.2f}")
                    print(f"     High:  ${float(latest[2]):,.2f}")
                    print(f"     Low:   ${float(latest[3]):,.2f}")
                    print(f"     Close: ${float(latest[4]):,.2f}")
            else:
                print(f"⚠️  Warning: No kline data received")
        except Exception as e:
            print(f"❌ Kline test failed: {e}")
            return False
        
        # Test 5: Set Leverage
        print("\n" + "-" * 70)
        print("TEST 5: Set Leverage")
        print("-" * 70)
        try:
            symbol = config['trading']['symbol']
            leverage = config['risk']['leverage']
            
            if config.get('testnet') and config['trading'].get('dry_run'):
                print(f"✅ Leverage setting (test mode):")
                print(f"   Symbol: {symbol}")
                print(f"   Target Leverage: {leverage}x")
                print(f"   Note: Not actually setting leverage in test mode")
            else:
                result = client.set_leverage(symbol, leverage)
                if result.get('code') == 0:
                    print(f"✅ Leverage set successfully")
                    print(f"   Symbol: {symbol}")
                    print(f"   Leverage: {leverage}x")
                else:
                    print(f"⚠️  Leverage response: {result}")
        except Exception as e:
            print(f"⚠️  Leverage test: {e}")
            print(f"   (This is optional, bot will continue)")
        
        # Test 6: Get Open Orders
        print("\n" + "-" * 70)
        print("TEST 6: Get Open Orders")
        print("-" * 70)
        try:
            symbol = config['trading']['symbol']
            orders = client.get_open_orders(symbol)
            
            if orders and len(orders) > 0:
                print(f"✅ Open orders found: {len(orders)}")
                for i, order in enumerate(orders[:3], 1):  # Show first 3
                    print(f"   Order {i}:")
                    print(f"     ID: {order.get('orderId', 'N/A')}")
                    print(f"     Side: {order.get('side', 'N/A')}")
                    print(f"     Price: ${float(order.get('price', 0)):,.2f}")
                    print(f"     Quantity: {order.get('quantity', 'N/A')}")
            else:
                print(f"✅ No open orders for {symbol}")
        except Exception as e:
            print(f"❌ Open orders test failed: {e}")
            return False
        
        # Summary
        print("\n" + "=" * 70)
        print("✅ CONNECTION TEST COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\n📋 Summary:")
        print("   ✓ Configuration valid")
        print("   ✓ Client initialized")
        print("   ✓ API connectivity confirmed")
        print("   ✓ Ticker data working")
        print("   ✓ Account info working")
        print("   ✓ Position query working")
        print("   ✓ Kline data working")
        print("   ✓ Open orders working")
        
        print("\n🎯 Next Steps:")
        print("   1. Review the bot configuration (config/config.json)")
        print("   2. Ensure dry_run=true for initial testing")
        print("   3. Run: python trading_bot.py")
        print("   4. Monitor logs/bot.log for activity")
        
        print("\n⚠️  Important Reminders:")
        print("   • Start with testnet=true")
        print("   • Use dry_run=true initially")
        print("   • Monitor the bot closely")
        print("   • Test with small positions first")
        
        print()
        return True
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: Configuration file not found: {config_path}")
        print("   Copy config.example.json to config.json and update credentials")
        return False
    except KeyError as e:
        print(f"\n❌ ERROR: Missing configuration key: {e}")
        print("   Check config.json against config.example.json")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.json"
    
    success = test_hyperliquid_connection(config_path)
    
    if success:
        print("✅ All tests passed! Ready to run trading bot.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please fix issues before running bot.")
        sys.exit(1)

