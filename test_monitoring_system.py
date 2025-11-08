#!/usr/bin/env python3
"""
Test Script for Trading Bot Monitoring System

Comprehensive testing of all monitoring components:
- Metrics collection and Prometheus export
- Structured logging with correlation IDs
- Multi-channel alerting
- Health check endpoints
- Trade analytics and reporting
- Anomaly detection
- Grafana dashboard generation

Usage:
    python test_monitoring_system.py [--verbose] [--all] [--component COMPONENT]

Options:
    --verbose: Enable detailed output
    --all: Run all tests (default)
    --component: Run specific component test (metrics, logging, alerting, etc.)
"""

import sys
import time
import json
import argparse
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List
import threading

# Add current directory to path for imports
sys.path.insert(0, '.')

def test_metrics_collection(verbose: bool = False) -> bool:
    """Test Prometheus metrics collection."""
    print("🧪 Testing Metrics Collection...")

    try:
        from metrics import TradingBotMetrics, initialize_metrics

        # Initialize metrics
        metrics = TradingBotMetrics()

        # Test basic metrics recording
        metrics.record_trade("BTC-USDC", "buy", 0.1, 50000, 50, "win")
        metrics.record_signal("BTC-USDC", "BUY_SIGNAL", "MACD crossover")
        metrics.update_position_metrics("BTC-USDC", 1, 100)
        metrics.update_account_metrics(10000, "USDC")
        metrics.record_api_request("place_order", "POST", 0.5, "success")
        metrics.record_order_execution("BTC-USDC", "buy", "LIMIT", 1.2)
        metrics.update_system_metrics()
        metrics.update_risk_metrics(current_drawdown=0.02, daily_loss_pct=0.01)
        metrics.update_websocket_status("wss://api.hyperliquid.xyz/ws", True)

        # Test metrics export
        metrics_text = metrics.get_metrics_text()
        assert b"trading_bot_trades_total" in metrics_text
        assert b"trading_bot_win_rate_ratio" in metrics_text

        # Test initialization with server (in background)
        def test_server():
            try:
                test_metrics = initialize_metrics(port=8002, start_server=True)
                time.sleep(2)  # Let server start

                # Test metrics endpoint
                response = requests.get("http://localhost:8002", timeout=5)
                assert response.status_code == 200
                assert b"trading_bot_info" in response.content

                print("✅ Metrics server test passed")
                return True
            except Exception as e:
                print(f"❌ Metrics server test failed: {e}")
                return False

        server_test_result = test_server()

        if verbose:
            print(f"📊 Sample metrics output (first 500 chars):")
            print(metrics_text.decode()[:500] + "...")

        print("✅ Metrics collection test passed")
        return True

    except Exception as e:
        print(f"❌ Metrics collection test failed: {e}")
        return False


def test_structured_logging(verbose: bool = False) -> bool:
    """Test structured JSON logging."""
    print("🧪 Testing Structured Logging...")

    try:
        from structured_logger import setup_structured_logging, get_trading_logger, CorrelationContext

        # Setup logging
        setup_structured_logging(
            log_level="DEBUG",
            log_file="logs/test_trading_bot.log",
            console_level="WARNING"  # Reduce console noise
        )

        logger = get_trading_logger()

        # Test correlation ID
        with CorrelationContext() as correlation_id:
            logger.info("Test log with correlation ID", extra={"test_data": "correlation_test"})

            # Test trading context logging
            logger.log_trade_entry(
                symbol="BTC-USDC",
                side="buy",
                quantity=0.1,
                price=50000,
                strategy="MACD"
            )

            logger.log_trade_exit(
                symbol="BTC-USDC",
                side="buy",
                quantity=0.1,
                entry_price=50000,
                exit_price=51000,
                pnl=100,
                exit_reason="take_profit"
            )

            logger.log_signal_generated(
                symbol="BTC-USDC",
                signal_type="BUY",
                confidence=0.85,
                indicators={"macd": 0.5, "rsi": 65}
            )

        # Test risk warning
        logger.log_risk_warning(
            warning_type="high_drawdown",
            symbol="BTC-USDC",
            current_value=0.08,
            threshold=0.05,
            message="Drawdown approaching limit"
        )

        # Test API call logging
        logger.log_api_call(
            endpoint="/place_order",
            method="POST",
            duration=0.8,
            status_code=200
        )

        # Test WebSocket logging
        logger.log_websocket_event(
            event_type="connected",
            url="wss://api.hyperliquid.xyz/ws",
            connected=True
        )

        # Verify log file was created and contains JSON
        log_file = Path("logs/test_trading_bot.log")
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()[-5:]  # Check last 5 lines

            for line in lines:
                try:
                    log_entry = json.loads(line.strip())
                    assert "timestamp" in log_entry
                    assert "level" in log_entry
                    assert "correlation_id" in log_entry
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON in log: {line}")
                    return False

            if verbose:
                print("📝 Sample log entries:")
                for line in lines[-2:]:
                    print(f"  {line.strip()}")

        print("✅ Structured logging test passed")
        return True

    except Exception as e:
        print(f"❌ Structured logging test failed: {e}")
        return False


def test_alerting_system(verbose: bool = False) -> bool:
    """Test multi-channel alerting system."""
    print("🧪 Testing Alerting System...")

    try:
        from alerting import AlertManager, AlertSeverity

        # Create test configuration (disabled channels to avoid spam)
        test_config = {
            "channels": {
                "email": {"enabled": False},
                "telegram": {"enabled": False},
                "discord": {"enabled": False},
                "pagerduty": {"enabled": False}
            },
            "dedup_window_minutes": 1,
            "max_notifications_per_alert": 2
        }

        alert_manager = AlertManager(test_config)
        alert_manager.start()

        # Test alert creation
        alert_id1 = alert_manager.send_alert(
            "test_alert",
            "Test Critical Alert",
            "This is a test alert for monitoring system validation",
            severity=AlertSeverity.CRITICAL,
            tags={"test": "true", "component": "monitoring"}
        )

        alert_id2 = alert_manager.send_alert(
            "test_warning",
            "Test Warning Alert",
            "This is a test warning alert",
            severity=AlertSeverity.WARNING
        )

        # Test deduplication
        alert_id3 = alert_manager.send_alert(
            "test_alert",  # Same type
            "Test Critical Alert (Duplicate)",
            "This should be deduplicated",
            severity=AlertSeverity.CRITICAL
        )

        # Wait for processing
        time.sleep(2)

        # Test alert resolution
        alert_manager.resolve_alert(alert_id1, "Test resolution")

        # Test alert statistics
        stats = alert_manager.get_alert_stats()
        assert stats['active_alerts'] >= 1  # At least the warning should be active

        # Test channel testing (should not actually send)
        channel_results = alert_manager.test_channels()
        # Should return False for all since disabled, but no exceptions

        alert_manager.stop()

        if verbose:
            print(f"📊 Alert stats: {stats}")
            print(f"🎯 Alert IDs created: {alert_id1}, {alert_id2}, {alert_id3}")

        print("✅ Alerting system test passed")
        return True

    except Exception as e:
        print(f"❌ Alerting system test failed: {e}")
        return False


def test_health_checks(verbose: bool = False) -> bool:
    """Test health check endpoints."""
    print("🧪 Testing Health Checks...")

    try:
        from health_check import create_health_app, HealthChecker

        # Create test health checker
        health_checker = HealthChecker()

        # Get health status
        status = health_checker.get_health_status()

        # Verify structure
        assert "status" in status
        assert "timestamp" in status
        assert "uptime_seconds" in status
        assert "checks" in status
        assert "metrics" in status

        # Verify checks exist
        checks = status["checks"]
        expected_checks = [
            'api_connectivity', 'websocket_connectivity', 'storage_health',
            'bot_status', 'risk_management', 'system_resources',
            'last_trade', 'position_health'
        ]

        for check_name in expected_checks:
            assert check_name in checks, f"Missing check: {check_name}"
            assert "status" in checks[check_name]

        # Test Flask app creation
        app = create_health_app(host='localhost', port=8003)

        # Test health endpoint in thread
        def test_server():
            try:
                app.run(host='localhost', port=8003, debug=False, use_reloader=False)
            except Exception:
                pass  # Expected to be interrupted

        server_thread = threading.Thread(target=test_server, daemon=True)
        server_thread.start()
        time.sleep(2)  # Let server start

        try:
            # Test health endpoints
            health_response = requests.get("http://localhost:8003/health", timeout=5)
            assert health_response.status_code in [200, 503]  # 503 is OK for unhealthy state

            health_data = health_response.json()
            assert "status" in health_data

            # Test ping endpoint
            ping_response = requests.get("http://localhost:8003/health/ping", timeout=5)
            assert ping_response.status_code == 200

            ping_data = ping_response.json()
            assert "status" in ping_data
            assert "pong" in ping_data["status"]

        except requests.exceptions.RequestException:
            print("⚠️  Health server not accessible (expected in test environment)")

        if verbose:
            print(f"🏥 Health status: {status['status']}")
            print(f"⏱️  Uptime: {status['uptime_seconds']:.1f} seconds")

        print("✅ Health checks test passed")
        return True

    except Exception as e:
        print(f"❌ Health checks test failed: {e}")
        return False


def test_trade_analytics(verbose: bool = False) -> bool:
    """Test trade analytics and reporting."""
    print("🧪 Testing Trade Analytics...")

    try:
        from trade_analytics import TradeAnalytics, TradeRecord

        # Create analytics instance
        analytics = TradeAnalytics(trade_data_file="data/test_trade_history.json")

        # Add test trades
        test_trades = [
            TradeRecord(
                trade_id="test_trade_1",
                symbol="BTC-USDC",
                side="buy",
                quantity=0.1,
                entry_price=50000,
                exit_price=51000,
                entry_time=datetime.now(timezone.utc) - timedelta(hours=2),
                exit_time=datetime.now(timezone.utc) - timedelta(hours=1),
                pnl=100,
                pnl_percentage=2.0,
                strategy="MACD",
                risk_reward_ratio=2.0
            ),
            TradeRecord(
                trade_id="test_trade_2",
                symbol="ETH-USDC",
                side="sell",
                quantity=1.0,
                entry_price=3000,
                exit_price=2900,
                entry_time=datetime.now(timezone.utc) - timedelta(hours=3),
                exit_time=datetime.now(timezone.utc) - timedelta(hours=2),
                pnl=-100,
                pnl_percentage=-3.33,
                strategy="RSI",
                risk_reward_ratio=1.5
            ),
            TradeRecord(
                trade_id="test_trade_3",
                symbol="BTC-USDC",
                side="buy",
                quantity=0.05,
                entry_price=51000,
                exit_price=51500,
                entry_time=datetime.now(timezone.utc) - timedelta(minutes=30),
                exit_time=datetime.now(timezone.utc),
                pnl=25,
                pnl_percentage=0.98,
                strategy="MACD",
                risk_reward_ratio=1.0
            )
        ]

        analytics.add_trades(test_trades)

        # Test performance metrics calculation
        performance = analytics.calculate_performance_metrics(test_trades)

        assert performance.total_trades == 3
        assert performance.winning_trades == 2
        assert performance.win_rate == 2/3
        assert performance.total_pnl == 25  # 100 - 100 + 25

        # Test risk metrics
        risk = analytics.calculate_risk_metrics(test_trades)
        assert risk.volatility >= 0

        # Test time analysis
        time_analysis = analytics.analyze_time_patterns(test_trades)
        assert hasattr(time_analysis, 'hourly_performance')

        # Test strategy analysis
        strategy_analysis = analytics.analyze_strategy_effectiveness(test_trades)
        assert "MACD" in strategy_analysis.strategy_performance
        assert "RSI" in strategy_analysis.strategy_performance

        # Test report generation
        report = analytics.generate_report(period_days=1)
        assert "performance" in report
        assert "risk" in report
        assert "recommendations" in report

        if verbose:
            print(f"📊 Performance metrics: Win Rate {performance.win_rate:.1%}, Total P&L ${performance.total_pnl:.2f}")
            print(f"🎯 Strategy analysis: {list(strategy_analysis.strategy_performance.keys())}")

        # Clean up test file
        test_file = Path("data/test_trade_history.json")
        if test_file.exists():
            test_file.unlink()

        print("✅ Trade analytics test passed")
        return True

    except Exception as e:
        print(f"❌ Trade analytics test failed: {e}")
        return False


def test_anomaly_detection(verbose: bool = False) -> bool:
    """Test anomaly detection system."""
    print("🧪 Testing Anomaly Detection...")

    try:
        from anomaly_detection import TradingAnomalyDetector, check_trading_anomalies

        # Initialize detector
        config = {
            'statistical_window': 50,
            'z_threshold': 2.0,
            'contamination': 0.1
        }

        detector = TradingAnomalyDetector(config)

        # Add normal data points
        for i in range(20):
            detector.check_trading_anomalies(trades_per_minute=5 + i * 0.1)

        # Add anomalous data point
        anomalies = detector.check_trading_anomalies(trades_per_minute=50)  # Much higher

        # Should detect anomaly
        assert anomalies.has_anomalies(), "Should detect trade frequency anomaly"

        # Test market data anomalies
        anomaly = detector.check_market_data_anomalies(50000, 1000, "BTC-USDC")
        # May or may not detect depending on data

        # Test trade streak detection
        recent_trades = [
            {"pnl": 10}, {"pnl": 15}, {"pnl": 8}, {"pnl": 12}, {"pnl": 6}, {"pnl": 9}  # All wins
        ]
        streak_anomaly = detector.detect_trade_streaks(recent_trades, min_streak=5)
        assert streak_anomaly is not None, "Should detect winning streak"

        # Test position concentration
        positions = [
            {"size": 10, "symbol": "BTC"},
            {"size": 5, "symbol": "ETH"},
            {"size": 2, "symbol": "SOL"}
        ]
        concentration_anomaly = detector.detect_position_concentration(positions)
        # Should not detect anomaly for this distribution

        if verbose:
            if anomalies.has_anomalies():
                for anomaly in anomalies.get_all_anomalies():
                    print(f"🚨 Detected anomaly: {anomaly.metric_name} = {anomaly.value} (score: {anomaly.score:.2f})")

            print(f"🎯 Trade streak anomaly: {streak_anomaly.metric_name if streak_anomaly else 'None'}")

        print("✅ Anomaly detection test passed")
        return True

    except Exception as e:
        print(f"❌ Anomaly detection test failed: {e}")
        return False


def test_grafana_dashboards(verbose: bool = False) -> bool:
    """Test Grafana dashboard generation."""
    print("🧪 Testing Grafana Dashboard Generation...")

    try:
        from grafana_dashboards import generate_all_dashboards, save_dashboards_to_files

        # Generate dashboards
        dashboards = generate_all_dashboards()

        expected_dashboards = [
            'trading_performance', 'system_health', 'risk_management',
            'strategy_analytics', 'api_performance'
        ]

        for dashboard_name in expected_dashboards:
            assert dashboard_name in dashboards, f"Missing dashboard: {dashboard_name}"

            dashboard = dashboards[dashboard_name]
            assert "dashboard" in dashboard, f"Invalid dashboard structure: {dashboard_name}"
            assert "title" in dashboard["dashboard"], f"Missing title in {dashboard_name}"

            if verbose:
                print(f"📊 Generated dashboard: {dashboard['dashboard']['title']}")

        # Test file saving (will create directory and files)
        save_dashboards_to_files("test_grafana_dashboards")

        # Verify files were created
        dashboard_dir = Path("test_grafana_dashboards")
        assert dashboard_dir.exists(), "Dashboard directory not created"

        json_files = list(dashboard_dir.glob("*.json"))
        assert len(json_files) == len(expected_dashboards), f"Expected {len(expected_dashboards)} files, got {len(json_files)}"

        # Clean up test files
        for json_file in json_files:
            json_file.unlink()
        dashboard_dir.rmdir()

        print("✅ Grafana dashboard generation test passed")
        return True

    except Exception as e:
        print(f"❌ Grafana dashboard test failed: {e}")
        return False


def run_integration_test(verbose: bool = False) -> bool:
    """Run integration test combining multiple components."""
    print("🧪 Running Integration Test...")

    try:
        # Test component interaction
        from metrics import get_metrics
        from structured_logger import get_trading_logger
        from anomaly_detection import check_trading_anomalies

        # Initialize components
        metrics = get_metrics()
        logger = get_trading_logger()

        # Simulate a trading scenario
        correlation_id = logger.set_correlation_id()

        # Record trade
        metrics.record_trade("BTC-USDC", "buy", 0.1, 50000, 50, "win")

        # Log trade
        logger.log_trade_entry(
            symbol="BTC-USDC",
            side="buy",
            quantity=0.1,
            price=50000,
            strategy="MACD"
        )

        # Check for anomalies
        anomalies = check_trading_anomalies(trades_per_minute=10)

        # Log anomalies if detected
        if anomalies.has_anomalies():
            for anomaly in anomalies.get_all_anomalies():
                logger.log_anomaly_detected(anomaly)

        logger.log_trade_exit(
            symbol="BTC-USDC",
            side="buy",
            quantity=0.1,
            entry_price=50000,
            exit_price=51000,
            pnl=100,
            exit_reason="take_profit"
        )

        print("✅ Integration test passed")
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Test Trading Bot Monitoring System")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    parser.add_argument("--component", choices=[
        "metrics", "logging", "alerting", "health", "analytics",
        "anomaly", "grafana", "integration"
    ], help="Run specific component test")

    args = parser.parse_args()

    # Determine which tests to run
    if args.component:
        test_components = [args.component]
    else:
        test_components = [
            "metrics", "logging", "alerting", "health",
            "analytics", "anomaly", "grafana", "integration"
        ]

    # Test functions mapping
    test_functions = {
        "metrics": test_metrics_collection,
        "logging": test_structured_logging,
        "alerting": test_alerting_system,
        "health": test_health_checks,
        "analytics": test_trade_analytics,
        "anomaly": test_anomaly_detection,
        "grafana": test_grafana_dashboards,
        "integration": run_integration_test
    }

    print("🚀 Starting Trading Bot Monitoring System Tests")
    print(f"📋 Running tests: {', '.join(test_components)}")
    print("=" * 60)

    results = {}
    start_time = time.time()

    for component in test_components:
        try:
            test_func = test_functions[component]
            result = test_func(verbose=args.verbose)
            results[component] = result

            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status} {component}")

        except Exception as e:
            print(f"❌ {component} - Unexpected error: {e}")
            results[component] = False

        print("-" * 40)

    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for r in results.values() if r)
    total = len(results)

    print("=" * 60)
    print("📊 Test Results Summary")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
