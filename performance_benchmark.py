"""
Performance Benchmark Suite for Trading Bot

Comprehensive benchmarking of trading bot performance across multiple dimensions:
- API call latency and throughput
- Indicator calculation performance
- Database operations speed
- Cache hit rates and performance
- Memory usage and CPU utilization
- End-to-end trading cycle performance

Provides before/after comparisons and performance regression detection.

"""

import asyncio
import time
import statistics
import psutil
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import json
from pathlib import Path

# Import bot components for benchmarking
from hyperliquid_client_async import AsyncHyperliquidClient
from macd_strategy_optimized import OptimizedMACDStrategy
from cache_manager import CacheManager
from database_manager import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    iterations: int
    total_time: float
    avg_time: float
    median_time: float
    min_time: float
    max_time: float
    throughput: float
    memory_usage_mb: float
    cpu_percent: float
    timestamp: datetime


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""

    def __init__(self, results_dir: str = "benchmarks"):
        """
        Initialize performance benchmark suite.

        Args:
            results_dir: Directory to save benchmark results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Benchmark results
        self.results: List[BenchmarkResult] = []

        # System monitoring
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        logger.info("Performance benchmark suite initialized")

    async def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        logger.info("Starting full benchmark suite...")

        results = {}

        # API Performance Benchmarks
        results["api_benchmarks"] = await self.benchmark_api_performance()

        # Indicator Calculation Benchmarks
        results["indicator_benchmarks"] = await self.benchmark_indicator_calculations()

        # Cache Performance Benchmarks
        results["cache_benchmarks"] = await self.benchmark_cache_performance()

        # Database Performance Benchmarks
        results["database_benchmarks"] = await self.benchmark_database_performance()

        # Memory and CPU Benchmarks
        results["system_benchmarks"] = await self.benchmark_system_performance()

        # End-to-End Trading Cycle
        results["trading_cycle_benchmarks"] = await self.benchmark_trading_cycle()

        # Save results
        self.save_results(results)

        logger.info("Full benchmark suite completed")
        return results

    async def benchmark_api_performance(self) -> Dict[str, Any]:
        """Benchmark API call performance."""
        logger.info("Benchmarking API performance...")

        results = {}

        # Create test client
        client = AsyncHyperliquidClient(
            private_key="0x" + "1" * 64,  # Dummy key for testing
            wallet_address="0x" + "2" * 40,  # Dummy address
            testnet=True
        )

        try:
            await client._connect()

            # Benchmark ticker API
            ticker_times = []
            for i in range(50):
                start_time = time.time()
                try:
                    await client.get_ticker("BTCUSDT")
                    ticker_times.append(time.time() - start_time)
                except Exception:
                    pass  # Skip failed requests

            if ticker_times:
                results["ticker_api"] = self._calculate_stats("ticker_api", ticker_times)

            # Benchmark account info API
            account_times = []
            for i in range(20):  # Fewer iterations for account API
                start_time = time.time()
                try:
                    await client.get_account_info()
                    account_times.append(time.time() - start_time)
                except Exception:
                    pass

            if account_times:
                results["account_api"] = self._calculate_stats("account_api", account_times)

            # Benchmark batch operations
            batch_times = []
            for i in range(10):
                start_time = time.time()
                try:
                    await client.batch_get_account_data()
                    batch_times.append(time.time() - start_time)
                except Exception:
                    pass

            if batch_times:
                results["batch_api"] = self._calculate_stats("batch_api", batch_times)

        finally:
            await client.close()

        return results

    async def benchmark_indicator_calculations(self) -> Dict[str, Any]:
        """Benchmark indicator calculation performance."""
        logger.info("Benchmarking indicator calculations...")

        results = {}

        # Create test strategy
        strategy = OptimizedMACDStrategy(cache_size=100)

        # Generate test data (1000 candles)
        import numpy as np
        import pandas as pd

        np.random.seed(42)  # For reproducible results

        # Generate realistic price data
        base_price = 50000
        prices = []
        for i in range(1000):
            price = base_price + np.random.normal(0, 500)
            price = max(price, 100)  # Ensure positive prices
            prices.append(price)

        volumes = np.random.normal(1000, 200, 1000)
        volumes = np.maximum(volumes, 10)  # Ensure positive volumes

        df = pd.DataFrame({
            'timestamp': range(1000),
            'open': prices,
            'high': [p + abs(np.random.normal(0, 100)) for p in prices],
            'low': [p - abs(np.random.normal(0, 100)) for p in prices],
            'close': prices,
            'volume': volumes
        })

        # Benchmark full indicator calculation
        calc_times = []
        for i in range(20):
            start_time = time.time()
            result_df = strategy.calculate_indicators(df.copy())
            calc_times.append(time.time() - start_time)

        results["full_calculation"] = self._calculate_stats("full_calculation", calc_times)

        # Benchmark incremental updates
        incremental_times = []
        for i in range(20):
            # Simulate adding 10 new candles
            new_candles = df.tail(10).copy()
            start_time = time.time()
            result_df = strategy.calculate_indicators_incremental(df, new_candles)
            incremental_times.append(time.time() - start_time)

        results["incremental_calculation"] = self._calculate_stats("incremental_calculation", incremental_times)

        return results

    async def benchmark_cache_performance(self) -> Dict[str, Any]:
        """Benchmark cache performance."""
        logger.info("Benchmarking cache performance...")

        results = {}

        # Create test cache
        cache = CacheManager(max_memory_cache_size=1000)

        # Benchmark cache operations
        set_times = []
        get_times = []

        # Test data
        test_data = [{"key": f"test_{i}", "value": {"data": "x" * 100}} for i in range(100)]

        # Benchmark cache sets
        for item in test_data:
            start_time = time.time()
            await cache.set(item["key"], item["value"], ttl=300)
            set_times.append(time.time() - start_time)

        results["cache_set"] = self._calculate_stats("cache_set", set_times)

        # Benchmark cache gets (should be hits)
        for item in test_data:
            start_time = time.time()
            result = await cache.get(item["key"])
            get_times.append(time.time() - start_time)

        results["cache_get_hit"] = self._calculate_stats("cache_get_hit", get_times)

        # Benchmark cache gets (misses)
        miss_times = []
        for i in range(100):
            start_time = time.time()
            result = await cache.get(f"nonexistent_{i}")
            miss_times.append(time.time() - start_time)

        results["cache_get_miss"] = self._calculate_stats("cache_get_miss", miss_times)

        # Get cache stats
        cache_stats = cache.get_stats()
        results["cache_stats"] = cache_stats

        await cache.disconnect()

        return results

    async def benchmark_database_performance(self) -> Dict[str, Any]:
        """Benchmark database performance."""
        logger.info("Benchmarking database performance...")

        results = {}

        # Create test database
        db_path = ":memory:"  # Use in-memory database for testing
        db = DatabaseManager(db_path, max_connections=3)

        # Benchmark individual operations
        trade_data = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.1,
            "price": 50000,
            "order_type": "LIMIT",
            "strategy": "MACD",
            "pnl": 0,
            "timestamp": datetime.now().isoformat()
        }

        # Benchmark single trade saves
        save_times = []
        for i in range(100):
            test_trade = trade_data.copy()
            test_trade["price"] = 50000 + i
            start_time = time.time()
            db.save_trade(test_trade)
            save_times.append(time.time() - start_time)

        results["single_trade_save"] = self._calculate_stats("single_trade_save", save_times)

        # Benchmark batch operations
        batch_operations = []
        for i in range(50):
            batch_operations.append(("save_trade", {
                **trade_data,
                "price": 50000 + i,
                "quantity": 0.1 + (i * 0.01)
            }))

        start_time = time.time()
        db.batch_operations(batch_operations)
        batch_time = time.time() - start_time

        results["batch_trade_save"] = self._calculate_stats("batch_trade_save", [batch_time / 50])  # Per operation

        # Benchmark queries
        query_times = []
        for i in range(50):
            start_time = time.time()
            trades = db.get_recent_trades(10)
            query_times.append(time.time() - start_time)

        results["trade_query"] = self._calculate_stats("trade_query", query_times)

        # Get database stats
        db_stats = db.get_database_stats()
        results["database_stats"] = db_stats

        db.close()

        return results

    async def benchmark_system_performance(self) -> Dict[str, Any]:
        """Benchmark system resource usage."""
        logger.info("Benchmarking system performance...")

        results = {}

        # Memory usage
        memory_info = self.process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024

        # CPU usage (over 1 second)
        cpu_percent = self.process.cpu_percent(interval=1.0)

        results["memory_usage_mb"] = memory_usage_mb
        results["cpu_percent"] = cpu_percent
        results["memory_delta_mb"] = memory_usage_mb - self.initial_memory

        return results

    async def benchmark_trading_cycle(self) -> Dict[str, Any]:
        """Benchmark end-to-end trading cycle performance."""
        logger.info("Benchmarking end-to-end trading cycle...")

        results = {}

        # Create mock trading components
        client = AsyncHyperliquidClient(
            private_key="0x" + "1" * 64,
            wallet_address="0x" + "2" * 40,
            testnet=True
        )

        strategy = OptimizedMACDStrategy()
        cache = CacheManager()

        try:
            # Warm up components
            await client._connect()
            await cache.connect()

            # Generate test market data
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

            # Benchmark complete trading cycle simulation
            cycle_times = []
            for i in range(10):
                start_time = time.time()

                # Simulate trading cycle operations
                # 1. Get market data (with caching)
                cache_key = f"market_data_BTCUSDT_15m_100"
                cached_data = await cache.get(cache_key)
                if not cached_data:
                    await cache.set(cache_key, df.to_dict('records'), 300)

                # 2. Calculate indicators
                indicators_df = strategy.calculate_indicators(df)

                # 3. Check signals
                signal = strategy.check_entry_signal(indicators_df)

                # 4. Simulate API calls
                try:
                    await client.get_ticker("BTCUSDT")
                except Exception:
                    pass  # Ignore API errors in benchmark

                cycle_times.append(time.time() - start_time)

            results["full_trading_cycle"] = self._calculate_stats("full_trading_cycle", cycle_times)

        finally:
            await client.close()
            await cache.disconnect()

        return results

    def _calculate_stats(self, name: str, times: List[float]) -> BenchmarkResult:
        """Calculate statistics for a benchmark."""
        if not times:
            return BenchmarkResult(
                name=name,
                iterations=0,
                total_time=0,
                avg_time=0,
                median_time=0,
                min_time=0,
                max_time=0,
                throughput=0,
                memory_usage_mb=self.process.memory_info().rss / 1024 / 1024,
                cpu_percent=self.process.cpu_percent(),
                timestamp=datetime.now()
            )

        total_time = sum(times)
        avg_time = statistics.mean(times)
        median_time = statistics.median(times)
        min_time = min(times)
        max_time = max(times)
        throughput = len(times) / total_time if total_time > 0 else 0

        result = BenchmarkResult(
            name=name,
            iterations=len(times),
            total_time=total_time,
            avg_time=avg_time,
            median_time=median_time,
            min_time=min_time,
            max_time=max_time,
            throughput=throughput,
            memory_usage_mb=self.process.memory_info().rss / 1024 / 1024,
            cpu_percent=self.process.cpu_percent(),
            timestamp=datetime.now()
        )

        self.results.append(result)
        return result

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"

        output = {
            "timestamp": timestamp,
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
            },
            "results": results,
            "detailed_results": [result.__dict__ for result in self.results]
        }

        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Benchmark results saved to {filepath}")

    def compare_results(self, baseline_file: str, current_file: str) -> Dict[str, Any]:
        """Compare benchmark results with baseline."""
        # Load baseline results
        with open(self.results_dir / baseline_file, 'r') as f:
            baseline = json.load(f)

        # Load current results
        with open(self.results_dir / current_file, 'r') as f:
            current = json.load(f)

        # Compare key metrics
        comparison = {}

        def compare_metric(baseline_val: float, current_val: float, name: str) -> Dict[str, Any]:
            if baseline_val == 0:
                return {"error": "Baseline value is zero"}

            improvement = ((baseline_val - current_val) / baseline_val) * 100
            return {
                "baseline": baseline_val,
                "current": current_val,
                "improvement_percent": improvement,
                "status": "improvement" if improvement > 0 else "regression" if improvement < -5 else "stable"
            }

        # Compare key performance metrics
        for category in ["api_benchmarks", "indicator_benchmarks", "cache_benchmarks", "database_benchmarks"]:
            if category in baseline["results"] and category in current["results"]:
                comparison[category] = {}
                for metric_name, baseline_result in baseline["results"][category].items():
                    if metric_name in current["results"][category]:
                        current_result = current["results"][category][metric_name]
                        if isinstance(baseline_result, dict) and isinstance(current_result, dict):
                            comparison[category][metric_name] = compare_metric(
                                baseline_result.get("avg_time", 0),
                                current_result.get("avg_time", 0),
                                f"{category}.{metric_name}"
                            )

        return comparison


async def run_benchmarks():
    """Run benchmark suite and display results."""
    benchmark = PerformanceBenchmark()

    print("🚀 Running Performance Benchmark Suite...")
    print("=" * 60)

    results = await benchmark.run_full_benchmark_suite()

        # Display key results
    print("\n📊 Key Performance Metrics:")
    print("-" * 40)

    for category, metrics in results.items():
        if isinstance(metrics, dict):
            print(f"\n{category.replace('_', ' ').title()}:")
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and "avg_time" in metric_data:
                    avg_time = metric_data["avg_time"] * 1000  # Convert to ms
                    throughput = metric_data.get("throughput", 0)
                    print(f"  {metric_name}: {avg_time:.2f}ms avg, {throughput:.1f} req/s")
    print(f"\n✅ Benchmark suite completed. Results saved to {benchmark.results_dir}")


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
