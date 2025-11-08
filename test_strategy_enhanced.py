#!/usr/bin/env python3
"""
Test Enhanced MACD Strategy

Comprehensive testing of the enhanced MACD strategy with all advanced features.
Tests individual components and integration scenarios.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json

# Add current directory to path
sys.path.insert(0, '.')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test_data(num_candles: int = 1000) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)  # For reproducible results

    # Generate timestamps
    start_time = datetime(2024, 1, 1)
    timestamps = [start_time + timedelta(minutes=i*5) for i in range(num_candles)]

    # Generate price data with trend and noise
    prices = [50000.0]  # Starting price
    for i in range(1, num_candles):
        # Add trend component
        trend = 0.0001 * np.sin(i / 50)  # Slow trend cycle

        # Add noise
        noise = np.random.normal(0, 0.005)  # 0.5% volatility

        # Generate OHLC
        prev_close = prices[-1]
        open_price = prev_close
        change = (trend + noise) * prev_close
        close_price = prev_close + change

        # Add some gap/overlap
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.002)) * prev_close
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.002)) * prev_close

        prices.append(close_price)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': [p for p in prices],  # Simplified - using close as open
        'high': [p + abs(np.random.normal(0, 0.003)) * p for p in prices],
        'low': [p - abs(np.random.normal(0, 0.003)) * p for p in prices],
        'close': prices,
        'volume': [int(np.random.lognormal(10, 1)) for _ in range(num_candles)]
    })

    df.set_index('timestamp', inplace=True)

    # Ensure OHLC relationships
    for idx in df.index:
        df.loc[idx, 'high'] = max(df.loc[idx, ['open', 'high', 'low', 'close']])
        df.loc[idx, 'low'] = min(df.loc[idx, ['open', 'high', 'low', 'close']])

    return df


def test_basic_functionality():
    """Test basic enhanced strategy functionality."""
    print("🧪 Testing Basic Enhanced Strategy Functionality...")

    try:
        from macd_strategy_enhanced import EnhancedMACDStrategy, MarketRegime

        # Create strategy with default enhanced parameters
        strategy = EnhancedMACDStrategy()

        # Generate test data
        df = generate_test_data(200)

        # Test indicator calculation
        df_with_indicators = strategy.calculate_indicators(df)

        # Check that indicators were added
        expected_indicators = ['macd', 'signal', 'histogram', 'rsi', 'atr', 'bb_middle', 'adx']
        for indicator in expected_indicators:
            assert indicator in df_with_indicators.columns, f"Missing indicator: {indicator}"

        print("✅ Basic functionality test passed")
        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_multi_timeframe_analysis():
    """Test multi-timeframe confirmation logic."""
    print("🧪 Testing Multi-Timeframe Analysis...")

    try:
        from macd_strategy_enhanced import EnhancedMACDStrategy

        strategy = EnhancedMACDStrategy(
            higher_timeframe_multiplier=12,
            require_higher_tf_alignment=True
        )

        df = generate_test_data(500)

        # Test market condition analysis
        market_condition = strategy.get_market_condition(df)
        assert hasattr(market_condition, 'regime')
        assert hasattr(market_condition, 'adx_value')

        # Test multi-timeframe alignment check
        alignment_ok, reason = strategy.check_multi_timeframe_alignment(df)
        assert isinstance(alignment_ok, bool)
        assert isinstance(reason, str)

        print("✅ Multi-timeframe analysis test passed")
        return True

    except Exception as e:
        print(f"❌ Multi-timeframe analysis test failed: {e}")
        return False


def test_volume_analysis():
    """Test volume confirmation and surge detection."""
    print("🧪 Testing Volume Analysis...")

    try:
        from macd_strategy_enhanced import EnhancedMACDStrategy

        strategy = EnhancedMACDStrategy(
            require_volume_confirmation=True,
            volume_period=20,
            min_volume_multiplier=1.2,
            volume_surge_threshold=3.0
        )

        df = generate_test_data(100)

        # Test volume conditions
        volume_ok, reason = strategy.check_volume_conditions(df)
        assert isinstance(volume_ok, bool)
        assert isinstance(reason, str)

        print("✅ Volume analysis test passed")
        return True

    except Exception as e:
        print(f"❌ Volume analysis test failed: {e}")
        return False


def test_volatility_filters():
    """Test ATR and Bollinger Band volatility filters."""
    print("🧪 Testing Volatility Filters...")

    try:
        from macd_strategy_enhanced import EnhancedMACDStrategy

        strategy = EnhancedMACDStrategy(
            use_atr_filter=True,
            atr_period=14,
            max_volatility_multiplier=3.0,
            use_bollinger_filter=True,
            bollinger_period=20,
            bollinger_std=2.0
        )

        df = generate_test_data(100)

        # Test volatility conditions
        volatility_ok, reason = strategy.check_volatility_conditions(df)
        assert isinstance(volatility_ok, bool)
        assert isinstance(reason, str)

        print("✅ Volatility filters test passed")
        return True

    except Exception as e:
        print(f"❌ Volatility filters test failed: {e}")
        return False


def test_market_regime_detection():
    """Test ADX-based market regime classification."""
    print("🧪 Testing Market Regime Detection...")

    try:
        from macd_strategy_enhanced import EnhancedMACDStrategy, MarketRegime

        strategy = EnhancedMACDStrategy(
            use_market_regime_filter=True,
            adx_period=14,
            adx_trending_threshold=25.0,
            adx_ranging_threshold=20.0
        )

        df = generate_test_data(100)

        # Test market condition detection
        market_condition = strategy.get_market_condition(df)

        assert isinstance(market_condition.regime, MarketRegime)
        assert isinstance(market_condition.adx_value, (int, float))
        assert market_condition.adx_value >= 0

        print(f"   Detected regime: {market_condition.regime.value}")
        print("✅ Market regime detection test passed")
        return True

    except Exception as e:
        print(f"❌ Market regime detection test failed: {e}")
        return False


def test_additional_filters():
    """Test RSI divergence, support/resistance, and other filters."""
    print("🧪 Testing Additional Entry Filters...")

    try:
        from macd_strategy_enhanced import EnhancedMACDStrategy

        strategy = EnhancedMACDStrategy(
            use_rsi_divergence=True,
            rsi_period=14,
            use_support_resistance=True,
            sr_lookback_periods=50,
            use_fibonacci_levels=True,
            use_round_number_filter=True,
            round_number_tolerance=0.001,
            use_time_filter=True,
            trading_hours_start=0,
            trading_hours_end=23
        )

        df = generate_test_data(200)

        # Test additional filters
        filters_ok, reason = strategy.check_additional_filters(df, 'LONG')
        assert isinstance(filters_ok, bool)
        assert isinstance(reason, str)

        print("✅ Additional filters test passed")
        return True

    except Exception as e:
        print(f"❌ Additional filters test failed: {e}")
        return False


def test_entry_signal_generation():
    """Test complete entry signal generation with all filters."""
    print("🧪 Testing Entry Signal Generation...")

    try:
        from macd_strategy_enhanced import EnhancedMACDStrategy

        strategy = EnhancedMACDStrategy()

        df = generate_test_data(300)

        # Test entry signal generation
        signal = strategy.check_entry_signal(df)

        # Signal can be None (no signal) or a dict with signal details
        if signal is not None:
            required_keys = ['signal', 'price', 'stop_loss', 'take_profit', 'position_size_pct']
            for key in required_keys:
                assert key in signal, f"Missing required signal key: {key}"

            assert signal['signal'] in ['LONG', 'SHORT']
            assert signal['price'] > 0
            assert signal['position_size_pct'] > 0

            print(f"   Generated signal: {signal['signal']} at ${signal['price']:.2f}")
        else:
            print("   No signal generated (expected with random data)")

        print("✅ Entry signal generation test passed")
        return True

    except Exception as e:
        print(f"❌ Entry signal generation test failed: {e}")
        return False


def test_adaptive_parameters():
    """Test adaptive parameter adjustment."""
    print("🧪 Testing Adaptive Parameters...")

    try:
        from macd_strategy_enhanced import EnhancedMACDStrategy

        strategy = EnhancedMACDStrategy(use_adaptive_parameters=True)

        # Test initial state
        assert not strategy._is_trading_paused()

        # Simulate poor performance
        for i in range(25):  # More than performance_window
            strategy.update_performance({'pnl': -10})  # Losing trade

        # Should trigger pause
        assert strategy._is_trading_paused()

        # Test strategy config
        config = strategy.get_strategy_config()
        assert isinstance(config, dict)
        assert 'fast_length' in config

        print("✅ Adaptive parameters test passed")
        return True

    except Exception as e:
        print(f"❌ Adaptive parameters test failed: {e}")
        return False


def test_backtesting_enhanced():
    """Test enhanced backtesting framework."""
    print("🧪 Testing Enhanced Backtesting...")

    try:
        from backtesting_enhanced import EnhancedBacktester

        strategy = EnhancedMACDStrategy()
        backtester = EnhancedBacktester()

        df = generate_test_data(500)

        # Run backtest
        result = backtester.run_backtest(df, strategy)

        # Check result structure
        assert hasattr(result, 'total_return')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'max_drawdown')
        assert hasattr(result, 'win_rate')
        assert hasattr(result, 'total_trades')
        assert hasattr(result, 'trades')
        assert hasattr(result, 'equity_curve')

        # Basic sanity checks
        assert isinstance(result.total_trades, int)
        assert result.total_trades >= 0
        assert isinstance(result.equity_curve, list)
        assert len(result.equity_curve) > 0

        print(f"   Backtest result: {result.total_trades} trades, {result.win_rate:.1%} win rate")
        print("✅ Enhanced backtesting test passed")
        return True

    except Exception as e:
        print(f"❌ Enhanced backtesting test failed: {e}")
        return False


def test_walk_forward_optimization():
    """Test walk-forward optimization."""
    print("🧪 Testing Walk-Forward Optimization...")

    try:
        from backtesting_enhanced import EnhancedBacktester
        from macd_strategy_enhanced import EnhancedMACDStrategy

        backtester = EnhancedBacktester()
        df = generate_test_data(1000)  # Need more data for walk-forward

        # Simple parameter ranges for testing
        parameter_ranges = {
            'fast_length': [8, 12, 16],
            'slow_length': [21, 26, 31]
        }

        # Run walk-forward optimization
        wf_result = backtester.walk_forward_optimization(
            df, parameter_ranges,
            in_sample_periods=2,  # Shorter for testing
            out_of_sample_periods=1,
            step_months=1
        )

        # Check results
        assert hasattr(wf_result, 'in_sample_results')
        assert hasattr(wf_result, 'out_of_sample_results')
        assert hasattr(wf_result, 'parameter_sets')
        assert hasattr(wf_result, 'overall_performance')
        assert hasattr(wf_result, 'robustness_score')

        print(f"   Walk-forward completed with robustness score: {wf_result.robustness_score:.2f}")
        print("✅ Walk-forward optimization test passed")
        return True

    except Exception as e:
        print(f"❌ Walk-forward optimization test failed: {e}")
        return False


def test_strategy_comparison():
    """Test strategy comparison functionality."""
    print("🧪 Testing Strategy Comparison...")

    try:
        from backtesting_enhanced import compare_strategies
        from macd_strategy_enhanced import EnhancedMACDStrategy

        df = generate_test_data(300)

        # Create different strategy variants
        strategies = {
            'conservative': EnhancedMACDStrategy(
                fast_length=8, slow_length=21, signal_length=5,
                base_position_size_pct=0.03
            ),
            'aggressive': EnhancedMACDStrategy(
                fast_length=16, slow_length=31, signal_length=13,
                base_position_size_pct=0.08
            ),
            'balanced': EnhancedMACDStrategy()  # Default parameters
        }

        # Compare strategies
        results = compare_strategies(df, strategies)

        # Check results
        assert len(results) == len(strategies)
        for name in strategies.keys():
            assert name in results
            assert hasattr(results[name], 'total_return')

        # Find best performing strategy
        best_strategy = max(results.keys(), key=lambda k: results[k].total_return)

        print(f"   Best performing strategy: {best_strategy}")
        print("✅ Strategy comparison test passed")
        return True

    except Exception as e:
        print(f"❌ Strategy comparison test failed: {e}")
        return False


def test_performance_report_generation():
    """Test performance report generation."""
    print("🧪 Testing Performance Report Generation...")

    try:
        from backtesting_enhanced import generate_performance_report
        from backtesting_enhanced import EnhancedBacktester
        from macd_strategy_enhanced import EnhancedMACDStrategy

        df = generate_test_data(200)

        # Create and run strategies
        strategies = {
            'strategy_a': EnhancedMACDStrategy(fast_length=12),
            'strategy_b': EnhancedMACDStrategy(fast_length=16)
        }

        backtester = EnhancedBacktester()
        results = {}

        for name, strategy in strategies.items():
            results[name] = backtester.run_backtest(df, strategy)

        # Generate report
        report_file = "test_performance_report.html"
        generate_performance_report(results, report_file)

        # Check that file was created
        import os
        assert os.path.exists(report_file), "Performance report file not created"

        # Clean up
        os.remove(report_file)

        print("✅ Performance report generation test passed")
        return True

    except Exception as e:
        print(f"❌ Performance report generation test failed: {e}")
        return False


def run_all_tests():
    """Run all enhanced strategy tests."""
    print("🚀 Starting Enhanced MACD Strategy Tests")
    print("=" * 60)

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Multi-Timeframe Analysis", test_multi_timeframe_analysis),
        ("Volume Analysis", test_volume_analysis),
        ("Volatility Filters", test_volatility_filters),
        ("Market Regime Detection", test_market_regime_detection),
        ("Additional Filters", test_additional_filters),
        ("Entry Signal Generation", test_entry_signal_generation),
        ("Adaptive Parameters", test_adaptive_parameters),
        ("Enhanced Backtesting", test_backtesting_enhanced),
        ("Walk-Forward Optimization", test_walk_forward_optimization),
        ("Strategy Comparison", test_strategy_comparison),
        ("Performance Report Generation", test_performance_report_generation),
    ]

    results = {}
    total_time = 0

    for test_name, test_func in tests:
        import time
        start_time = time.time()

        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
        except Exception as e:
            print(f"❌ {test_name} - Unexpected error: {e}")
            results[test_name] = False
            status = "❌ FAILED"

        elapsed = time.time() - start_time
        total_time += elapsed

        print(f"{status} {test_name} ({elapsed:.2f}s)")
        print("-" * 50)

    # Summary
    print("=" * 60)
    print("📊 Test Results Summary")
    print(f"⏱️  Total time: {total_time:.2f} seconds")

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")

    # Detailed results
    print("\n📋 Detailed Results:")
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}")

    if passed == total:
        print("\n🎉 All enhanced strategy tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
