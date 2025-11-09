#!/usr/bin/env python3
"""
Test script for data collection system
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.hyperliquid_collector import HyperliquidDataCollector
from data_collection.data_manager import DataManager

async def test_collector():
    """Test the data collector"""
    print("🧪 Testing Hyperliquid Data Collector...")

    # Create collector with minimal config for testing
    collector = HyperliquidDataCollector(
        symbols=['BTC'],  # Just test with BTC
        timeframes=['1h'],  # Just 1h timeframe
        data_dir='data/test'
    )

    print("✅ Collector initialized")

    # Test getting data (should fail gracefully since no data collected yet)
    df = collector.get_latest_data('BTC', '1h', periods=10)
    print(f"📊 Latest data shape: {df.shape}")

    print("✅ Collector test completed")

def test_data_manager():
    """Test the data manager"""
    print("🧪 Testing Data Manager...")

    dm = DataManager(db_path='data/test_trading.db')

    # Test getting data (should be empty)
    df = dm.get_candles('BTC', '1h')
    print(f"📊 Database candles: {len(df)} rows")

    print("✅ Data manager test completed")

def main():
    """Run all tests"""
    print("🚀 Starting Data Collection System Tests")
    print("=" * 50)

    # Test data manager (sync)
    test_data_manager()

    # Test collector (async)
    asyncio.run(test_collector())

    print("=" * 50)
    print("✅ All tests completed successfully!")
    print("\n📝 Next steps:")
    print("1. Start data collection: ./scripts/start_data_collection.sh")
    print("2. Monitor logs: tail -f logs/data_collector.log")
    print("3. Check collected data in data/live/ directory")

if __name__ == "__main__":
    main()
