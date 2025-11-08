#!/usr/bin/env python3
"""
Test Runner Script

Run all unit tests or specific test suites.

Usage:
    python3 run_tests.py                    # Run all tests
    python3 run_tests.py --unit              # Run unit tests only
    python3 run_tests.py --integration       # Run integration tests only
    python3 run_tests.py test_macd_strategy  # Run specific test file
"""

import unittest
import sys
import argparse


def run_all_tests():
    """Run all test suites"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Discover all test files
    test_files = [
        'test_macd_strategy',
        'test_risk_manager',
        'test_trailing_stop',
        'test_integration',
        'test_api_error_handling',
        'test_position_sync',
        'test_daily_reset',
        'test_config_validation',
        'test_performance',
        'test_hyperliquid_signing'
    ]
    
    for test_file in test_files:
        try:
            tests = loader.loadTestsFromName(test_file)
            suite.addTests(tests)
            print(f"✅ Loaded {test_file}")
        except Exception as e:
            print(f"⚠️  Failed to load {test_file}: {e}")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_unit_tests():
    """Run unit tests only"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    unit_test_files = [
        'test_macd_strategy',
        'test_risk_manager',
        'test_trailing_stop',
        'test_daily_reset',
        'test_config_validation',
        'test_performance',
        'test_hyperliquid_signing'
    ]
    
    for test_file in unit_test_files:
        try:
            tests = loader.loadTestsFromName(test_file)
            suite.addTests(tests)
            print(f"✅ Loaded {test_file}")
        except Exception as e:
            print(f"⚠️  Failed to load {test_file}: {e}")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_integration_tests():
    """Run integration tests only"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    integration_test_files = [
        'test_integration',
        'test_api_error_handling',
        'test_position_sync'
    ]
    
    for test_file in integration_test_files:
        try:
            tests = loader.loadTestsFromName(test_file)
            suite.addTests(tests)
            print(f"✅ Loaded {test_file}")
        except Exception as e:
            print(f"⚠️  Failed to load {test_file}: {e}")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_specific_test(test_name):
    """Run a specific test file"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_name)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description='Run trading bot tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('test_name', nargs='?', help='Run specific test file')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("TRADING BOT TEST SUITE")
    print("=" * 70)
    print()
    
    if args.test_name:
        success = run_specific_test(args.test_name)
    elif args.unit:
        success = run_unit_tests()
    elif args.integration:
        success = run_integration_tests()
    else:
        success = run_all_tests()
    
    print()
    print("=" * 70)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 70)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

