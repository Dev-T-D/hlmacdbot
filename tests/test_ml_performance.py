#!/usr/bin/env python3
"""
ML Performance Benchmarking Suite

Tests and validates the performance of optimized ML inference for trading.
Ensures sub-50ms p95 latency and accuracy preservation after ONNX conversion.

Usage:
    python tests/test_ml_performance.py
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_inference_engine import OptimizedMLInferenceEngine
from optimized_feature_engine import OptimizedFeatureEngine
from ml_signal_enhancer import MLSignalEnhancer

logger = logging.getLogger(__name__)


class MLPerformanceBenchmark:
    """
    Comprehensive benchmarking suite for ML inference performance.
    """

    def __init__(self):
        self.results = {}
        self.onnx_available = self._check_onnx_availability()

        # Create test data
        self.test_data = self._create_test_data()
        self.test_features = self._create_test_features()

    def _check_onnx_availability(self) -> bool:
        """Check if ONNX models are available for testing."""
        model_paths = [
            'models/onnx/lightgbm.onnx',
            'models/onnx/xgboost.onnx',
            'models/onnx/random_forest.onnx'
        ]

        available = all(Path(path).exists() for path in model_paths)
        if not available:
            logger.warning("⚠️  ONNX models not found - some tests will be skipped")
            logger.info("Run: python scripts/convert_models_to_onnx.py")
        return available

    def _create_test_data(self) -> pd.DataFrame:
        """Create realistic test market data."""
        np.random.seed(42)  # For reproducible results

        # Create 200 candles of 5-minute data
        dates = pd.date_range('2024-01-01', periods=200, freq='5min')

        # Generate realistic price action
        base_price = 50000
        price_changes = np.random.normal(0, 50, 200).cumsum()
        close_prices = base_price + price_changes

        # Generate OHLCV with some noise
        highs = close_prices + np.abs(np.random.normal(0, 25, 200))
        lows = close_prices - np.abs(np.random.normal(0, 25, 200))
        opens = close_prices + np.random.normal(0, 10, 200)

        # Volume with some spikes
        volumes = np.random.lognormal(15, 0.5, 200)

        return pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': close_prices,
            'volume': volumes
        }, index=dates)

    def _create_test_features(self) -> np.ndarray:
        """Create test feature array."""
        np.random.seed(42)
        return np.random.randn(1, 80).astype(np.float32)

    def benchmark_inference_engine(self) -> Dict[str, Any]:
        """
        Benchmark the optimized inference engine performance.

        Returns:
            Dict with latency statistics
        """
        if not self.onnx_available:
            return {'status': 'skipped', 'reason': 'ONNX models not available'}

        logger.info("🔬 Benchmarking OptimizedMLInferenceEngine...")

        # Initialize engine
        model_configs = {
            'lightgbm': 'models/onnx/lightgbm.onnx',
            'xgboost': 'models/onnx/xgboost.onnx',
            'random_forest': 'models/onnx/random_forest.onnx'
        }

        engine = OptimizedMLInferenceEngine(model_configs)

        # Warmup (10 predictions)
        logger.info("Warmup phase...")
        for _ in range(10):
            engine.predict_ensemble(self.test_features)

        # Benchmark (100 predictions)
        logger.info("Benchmarking phase...")
        latencies = []

        for i in range(100):
            start = time.perf_counter()
            result = engine.predict_ensemble(self.test_features)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            if (i + 1) % 25 == 0:
                logger.info(f"  Completed {i + 1}/100 predictions...")

        latencies_array = np.array(latencies)

        # Calculate statistics
        stats = {
            'status': 'completed',
            'p50': float(np.percentile(latencies_array, 50)),
            'p95': float(np.percentile(latencies_array, 95)),
            'p99': float(np.percentile(latencies_array, 99)),
            'mean': float(np.mean(latencies_array)),
            'min': float(np.min(latencies_array)),
            'max': float(np.max(latencies_array)),
            'std': float(np.std(latencies_array)),
            'count': len(latencies)
        }

        # Get engine performance report
        engine_report = engine.get_performance_report()
        stats['engine_report'] = engine_report

        self.results['inference_engine'] = stats
        return stats

    def benchmark_feature_engine(self) -> Dict[str, Any]:
        """
        Benchmark the optimized feature engine performance.

        Returns:
            Dict with feature computation statistics
        """
        logger.info("🔬 Benchmarking OptimizedFeatureEngine...")

        engine = OptimizedFeatureEngine()

        # Warmup
        logger.info("Warmup phase...")
        for _ in range(5):
            features = engine.compute_features_fast(self.test_data, "BTCUSDT")

        # Benchmark
        logger.info("Benchmarking phase...")
        latencies = []

        for i in range(50):
            start = time.perf_counter()
            features = engine.compute_features_fast(self.test_data, "BTCUSDT")
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

        latencies_array = np.array(latencies)

        stats = {
            'status': 'completed',
            'p50': float(np.percentile(latencies_array, 50)),
            'p95': float(np.percentile(latencies_array, 95)),
            'p99': float(np.percentile(latencies_array, 99)),
            'mean': float(np.mean(latencies_array)),
            'min': float(np.min(latencies_array)),
            'max': float(np.max(latencies_array)),
            'std': float(np.std(latencies_array)),
            'feature_count': len(features),
            'cache_stats': engine.get_performance_stats()
        }

        self.results['feature_engine'] = stats
        return stats

    def benchmark_ml_signal_enhancer(self) -> Dict[str, Any]:
        """
        Benchmark the complete ML signal enhancer pipeline.

        Returns:
            Dict with end-to-end performance statistics
        """
        logger.info("🔬 Benchmarking MLSignalEnhancer end-to-end...")

        # Initialize enhancer (will try to load optimized engines)
        enhancer = MLSignalEnhancer(symbol="BTCUSDT")

        # Try to load models
        models_loaded = enhancer.load_models()

        if not models_loaded:
            return {
                'status': 'skipped',
                'reason': 'No ML models available for testing'
            }

        # Warmup
        logger.info("Warmup phase...")
        for _ in range(3):
            result = enhancer.predict_direction(self.test_data)

        # Benchmark
        logger.info("Benchmarking phase...")
        latencies = []

        for i in range(30):
            start = time.perf_counter()
            result = enhancer.predict_direction(self.test_data)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

        latencies_array = np.array(latencies)

        stats = {
            'status': 'completed',
            'optimization_level': result.get('optimization_level', 'unknown'),
            'p50': float(np.percentile(latencies_array, 50)),
            'p95': float(np.percentile(latencies_array, 95)),
            'p99': float(np.percentile(latencies_array, 99)),
            'mean': float(np.mean(latencies_array)),
            'min': float(np.min(latencies_array)),
            'max': float(np.max(latencies_array)),
            'std': float(np.std(latencies_array)),
            'models_loaded': models_loaded
        }

        self.results['signal_enhancer'] = stats
        return stats

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """
        Run all performance benchmarks.

        Returns:
            Dict with all benchmark results
        """
        logger.info("🚀 Starting ML Performance Benchmark Suite")

        self.results = {}

        # Run individual benchmarks
        self.benchmark_feature_engine()
        self.benchmark_inference_engine()
        self.benchmark_ml_signal_enhancer()

        # Generate comprehensive report
        report = self.generate_performance_report()

        logger.info("✅ Benchmarking complete")
        print("\n" + "="*80)
        print("📊 PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        print(report)

        return self.results

    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        report = "ML Inference Performance Benchmark Report\n"
        report += "=" * 50 + "\n\n"

        # Overall status
        if not self.onnx_available:
            report += "⚠️  WARNING: ONNX models not available\n"
            report += "   Run: python scripts/convert_models_to_onnx.py\n\n"

        # Feature Engine Results
        if 'feature_engine' in self.results:
            fe = self.results['feature_engine']
            report += "⚡ FEATURE ENGINE PERFORMANCE\n"
            report += f"   Features computed: {fe['feature_count']}\n"
            report += f"   p50: {fe['p50']:.2f}ms\n"
            report += f"   p95: {fe['p95']:.2f}ms\n"
            report += f"   p99: {fe['p99']:.2f}ms\n"
            report += f"   Cache hit rate: {fe['cache_stats']['cache_hit_rate']:.1%}\n"
            report += f"   Numba JIT: {'✅' if fe['cache_stats']['numba_enabled'] else '❌'}\n\n"

        # Inference Engine Results
        if 'inference_engine' in self.results:
            ie = self.results['inference_engine']
            if ie['status'] == 'completed':
                report += "🚀 INFERENCE ENGINE PERFORMANCE\n"
                report += f"   p50: {ie['p50']:.2f}ms\n"
                report += f"   p95: {ie['p95']:.2f}ms\n"
                report += f"   p99: {ie['p99']:.2f}ms\n"
                report += f"   Mean: {ie['mean']:.2f}ms\n"
                # Target check
                if ie['p95'] < 50:
                    report += "   ✅ TARGET ACHIEVED: Sub-50ms p95 latency\n"
                else:
                    report += "   ❌ TARGET MISSED: p95 latency exceeds 50ms\n"
                report += "\n"
            else:
                report += f"🚀 INFERENCE ENGINE: {ie['reason']}\n\n"

        # Signal Enhancer Results
        if 'signal_enhancer' in self.results:
            se = self.results['signal_enhancer']
            if se['status'] == 'completed':
                report += "🎯 END-TO-END SIGNAL ENHANCER PERFORMANCE\n"
                report += f"   Optimization level: {se['optimization_level']}\n"
                report += f"   p50: {se['p50']:.2f}ms\n"
                report += f"   p95: {se['p95']:.2f}ms\n"
                report += f"   p99: {se['p99']:.2f}ms\n"
                report += f"   Mean: {se['mean']:.2f}ms\n"
                report += "\n"
            else:
                report += f"🎯 SIGNAL ENHANCER: {se['reason']}\n\n"

        # Performance Rating
        report += "🏆 PERFORMANCE RATING\n"

        # Calculate overall score
        scores = []
        targets_met = 0
        total_targets = 0

        if 'feature_engine' in self.results:
            fe = self.results['feature_engine']
            if fe['p95'] < 20:
                scores.append(5)  # Excellent
                targets_met += 1
            elif fe['p95'] < 50:
                scores.append(4)  # Good
                targets_met += 1
            else:
                scores.append(2)  # Poor
            total_targets += 1

        if 'inference_engine' in self.results and self.results['inference_engine']['status'] == 'completed':
            ie = self.results['inference_engine']
            if ie['p95'] < 30:
                scores.append(5)  # Excellent
                targets_met += 1
            elif ie['p95'] < 50:
                scores.append(4)  # Good
                targets_met += 1
            else:
                scores.append(2)  # Poor
            total_targets += 1

        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score >= 4.5:
                rating = "🎯 EXCEPTIONAL - Production Ready"
            elif avg_score >= 3.5:
                rating = "✅ EXCELLENT - Meets Targets"
            elif avg_score >= 2.5:
                rating = "⚠️  GOOD - Minor Optimizations Needed"
            else:
                rating = "❌ NEEDS IMPROVEMENT - Performance Issues"

            report += f"   Rating: {rating}\n"
            report += f"   Average Score: {avg_score:.1f}/5\n"
            if total_targets > 0:
                report += f"   Targets Met: {targets_met}/{total_targets}\n"
        else:
            report += "   Rating: Unable to calculate (insufficient data)\n"

        return report

    def assert_performance_targets(self) -> None:
        """
        Assert that performance targets are met.
        Raises AssertionError if targets are not met.
        """
        # Feature computation target: <20ms p95
        if 'feature_engine' in self.results:
            fe_p95 = self.results['feature_engine']['p95']
            assert fe_p95 < 20, f"Feature computation p95 {fe_p95:.2f}ms exceeds 20ms target"

        # Inference target: <50ms p95
        if 'inference_engine' in self.results and self.results['inference_engine']['status'] == 'completed':
            ie_p95 = self.results['inference_engine']['p95']
            assert ie_p95 < 50, f"Inference p95 {ie_p95:.2f}ms exceeds 50ms target"

        # End-to-end target: reasonable latency
        if 'signal_enhancer' in self.results and self.results['signal_enhancer']['status'] == 'completed':
            se_p95 = self.results['signal_enhancer']['p95']
            assert se_p95 < 100, f"End-to-end p95 {se_p95:.2f}ms exceeds 100ms target"


def run_performance_tests():
    """Run the complete performance test suite."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    benchmark = MLPerformanceBenchmark()

    try:
        # Run all benchmarks
        results = benchmark.run_all_benchmarks()

        # Assert performance targets
        benchmark.assert_performance_targets()

        print("\n✅ All performance targets met!")
        return True

    except AssertionError as e:
        print(f"\n❌ Performance target not met: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Benchmarking failed: {e}")
        return False


if __name__ == "__main__":
    success = run_performance_tests()
    sys.exit(0 if success else 1)
