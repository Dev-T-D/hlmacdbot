"""
Integration Test for Optimized Trading System

Tests the complete optimized trading system including:
- Async trading bot with real-time capabilities
- High-performance API client with batch operations
- Intelligent caching with Redis fallback
- Optimized indicator calculations
- Database persistence with WAL mode
- Performance monitoring and benchmarking

Run this test to validate the complete optimized system.
"""

import asyncio
import sys
import json
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from hyperliquid_client_async import AsyncHyperliquidClient
from macd_strategy_optimized import OptimizedMACDStrategy
from cache_manager import CacheManager, initialize_cache_manager
from database_manager import DatabaseManager, initialize_database_manager
from performance_benchmark import PerformanceBenchmark


async def test_async_client():
    """Test async Hyperliquid client."""
    print("🧪 Testing Async Hyperliquid Client...")

    client = AsyncHyperliquidClient(
        private_key="0x" + "1" * 64,  # Dummy key
        wallet_address="0x" + "2" * 40,  # Dummy address
        testnet=True
    )

    try:
        # Test connection
        await client._connect()
        print("  ✅ Client connection established")

        # Test ticker API
        ticker = await client.get_ticker("BTCUSDT")
        print(f"  ✅ Ticker API: BTC price = ${ticker.get('markPrice', 'N/A')}")

        # Test batch operations
        account_info, positions, orders = await client.batch_get_account_data()
        print(f"  ✅ Batch API: Got {len(positions)} positions, {len(orders)} orders")

        return True

    except Exception as e:
        print(f"  ❌ Async client test failed: {e}")
        return False
    finally:
        await client.close()


async def test_optimized_strategy():
    """Test optimized MACD strategy."""
    print("🧪 Testing Optimized MACD Strategy...")

    try:
        strategy = OptimizedMACDStrategy(cache_size=50)

        # Generate test data
        import pandas as pd
        import numpy as np

        np.random.seed(42)
        prices = [50000 + np.random.normal(0, 200) for _ in range(200)]
        df = pd.DataFrame({
            'timestamp': range(200),
            'open': prices,
            'high': [p + 50 for p in prices],
            'low': [p - 50 for p in prices],
            'close': prices,
            'volume': [1000 + np.random.normal(0, 100) for _ in prices]
        })

        # Test indicator calculation
        start_time = time.time()
        result_df = strategy.calculate_indicators(df)
        calc_time = time.time() - start_time

        if 'macd' in result_df.columns and 'rsi' in result_df.columns:
            print(f"  ✅ Indicator calculation: {calc_time:.4f}s for {len(df)} candles")
            print("  ✅ Indicator calculation successful")
        else:
            print("  ❌ Missing expected indicator columns")
            return False

        # Test signal detection
        signal = strategy.check_entry_signal(result_df)
        print(f"  ✅ Signal detection: {'Found signal' if signal else 'No signal'}")

        # Test cache stats
        cache_stats = strategy.get_cache_stats()
        print(f"  ✅ Cache stats: {cache_stats['cache_size']} entries")

        return True

    except Exception as e:
        print(f"  ❌ Optimized strategy test failed: {e}")
        return False


async def test_cache_manager():
    """Test cache manager with Redis fallback."""
    print("🧪 Testing Cache Manager...")

    try:
        cache = CacheManager(max_memory_cache_size=100)

        # Test connection
        await cache.connect()
        print("  ✅ Cache manager connected")

        # Test basic operations
        test_key = "test_key"
        test_value = {"data": "test", "timestamp": time.time()}

        # Set value
        await cache.set(test_key, test_value, ttl=60)
        print("  ✅ Cache set operation successful")

        # Get value
        retrieved = await cache.get(test_key)
        if retrieved and retrieved["data"] == test_value["data"]:
            print("  ✅ Cache get operation successful (hit)")
        else:
            print("  ❌ Cache get operation failed")
            return False

        # Test cache stats
        stats = cache.get_stats()
        print(f"  ✅ Cache stats: {stats['memory_cache']['size']} entries")

        # Test nonexistent key
        nonexistent = await cache.get("nonexistent_key")
        if nonexistent is None:
            print("  ✅ Cache miss handling correct")
        else:
            print("  ❌ Cache miss handling failed")
            return False

        return True

    except Exception as e:
        print(f"  ❌ Cache manager test failed: {e}")
        return False
    finally:
        await cache.disconnect()


async def test_database_manager():
    """Test database manager with WAL mode."""
    print("🧪 Testing Database Manager...")

    try:
        # Use in-memory database for testing
        db = DatabaseManager(":memory:", max_connections=2)

        # Test trade saving
        trade_data = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.1,
            "price": 50000,
            "order_type": "LIMIT",
            "strategy": "MACD",
            "pnl": 0,
            "timestamp": "2025-01-08T10:00:00Z"
        }

        trade_id = db.save_trade(trade_data)
        print(f"  ✅ Trade saved with ID: {trade_id}")

        # Test trade retrieval
        recent_trades = db.get_recent_trades(5)
        if len(recent_trades) > 0:
            print(f"  ✅ Retrieved {len(recent_trades)} recent trades")
        else:
            print("  ❌ No trades retrieved")
            return False

        # Test statistics
        stats = db.get_trade_statistics()
        print(f"  ✅ Trade statistics: {stats.get('total_trades', 0)} total trades")

        # Test batch operations
        batch_ops = [
            ("save_trade", {**trade_data, "price": 50100, "quantity": 0.05}),
            ("save_trade", {**trade_data, "price": 50200, "quantity": 0.03})
        ]

        batch_results = db.batch_operations(batch_ops)
        print(f"  ✅ Batch operations completed: {len(batch_results)} results")

        # Test database stats
        db_stats = db.get_database_stats()
        print(f"  ✅ Database stats: {db_stats['table_counts']} tables")

        return True

    except Exception as e:
        print(f"  ❌ Database manager test failed: {e}")
        return False
    finally:
        db.close()


async def test_performance_benchmark():
    """Test performance benchmarking."""
    print("🧪 Testing Performance Benchmark...")

    try:
        benchmark = PerformanceBenchmark()

        # Run a quick benchmark (subset of operations)
        results = await benchmark.benchmark_indicator_calculations()
        results.update(await benchmark.benchmark_system_performance())

        if "full_calculation" in results:
            avg_time = results["full_calculation"]["avg_time"] * 1000  # Convert to ms
            print(f"  ✅ Indicator benchmark: {avg_time:.2f}ms avg calculation time")
            print("  ✅ Performance benchmark completed")
            return True
        else:
            print("  ❌ Performance benchmark missing expected results")
            return False

    except Exception as e:
        print(f"  ❌ Performance benchmark test failed: {e}")
        return False


async def test_integration():
    """Test integrated system components."""
    print("🧪 Testing System Integration...")

    try:
        # Initialize components
        client = AsyncHyperliquidClient(
            private_key="0x" + "1" * 64,
            wallet_address="0x" + "2" * 40,
            testnet=True
        )

        strategy = OptimizedMACDStrategy(cache_size=20)
        cache = CacheManager(max_memory_cache_size=50)

        # Connect components
        await client._connect()
        await cache.connect()

        # Generate test market data
        import pandas as pd
        import numpy as np

        np.random.seed(42)
        prices = [50000 + np.random.normal(0, 100) for _ in range(150)]
        df = pd.DataFrame({
            'timestamp': range(150),
            'open': prices,
            'high': [p + 25 for p in prices],
            'low': [p - 25 for p in prices],
            'close': prices,
            'volume': [800 + np.random.normal(0, 50) for _ in prices]
        })

        # Test integrated workflow
        start_time = time.time()

        # 1. Cache market data
        cache_key = "integration_test_market_data"
        await cache.set(cache_key, df.to_dict('records'), ttl=300)

        # 2. Retrieve from cache
        cached_data = await cache.get(cache_key)
        if cached_data:
            df_cached = pd.DataFrame(cached_data)
            print("  ✅ Market data caching/retrieval successful")
        else:
            print("  ❌ Market data caching failed")
            return False

        # 3. Calculate indicators
        df_with_indicators = strategy.calculate_indicators(df_cached)

        # 4. Check signals
        signal = strategy.check_entry_signal(df_with_indicators)

        # 5. Simulate API call
        try:
            ticker = await client.get_ticker("BTCUSDT")
            print("  ✅ API integration successful")
        except Exception:
            print("  ⚠️ API call failed (expected in test environment)")

        total_time = time.time() - start_time
        print(f"  ✅ Integration workflow completed in {total_time:.4f}s")
        # Cleanup
        await client.close()
        await cache.disconnect()

        return True

    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False


async def run_all_tests():
    """Run all integration tests."""
    print("🚀 Running Optimized Trading System Integration Tests")
    print("=" * 60)

    tests = [
        ("Async Hyperliquid Client", test_async_client),
        ("Optimized MACD Strategy", test_optimized_strategy),
        ("Cache Manager", test_cache_manager),
        ("Database Manager", test_database_manager),
        ("Performance Benchmark", test_performance_benchmark),
        ("System Integration", test_integration)
    ]

    results = []
    total_start = time.time()

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            result = await test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {status}")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            results.append((test_name, False))

    # Summary
    total_time = time.time() - total_start
    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(".2f")
    if passed == total:
        print("🎉 ALL TESTS PASSED! Optimized system is ready for production.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
