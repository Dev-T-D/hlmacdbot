#!/usr/bin/env python3
"""
Strategy Optimization Script

Iteratively tests different strategy configurations to find profitable settings.
"""

import json
import subprocess
import sys
from typing import Dict, List, Tuple
import pandas as pd

def run_backtest(config: Dict, test_name: str) -> Dict:
    """Run backtest with given config and return results."""
    # Save config temporarily
    config_path = "config/config_optimize.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run backtest
    cmd = [
        "python3", "run_backtest.py",
        "--data", "data/BNB_30m_5000.csv",
        "--config", config_path,
        "--initial-balance", "10000",
        "--leverage", "10",
        "--slippage", "0.001",
        "--export-trades", f"data/optimize_{test_name}_trades.csv",
        "--export-equity", f"data/optimize_{test_name}_equity.csv"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        # Parse results from output
        output = result.stdout + result.stderr
        results = {}
        
        # Extract key metrics
        for line in output.split('\n'):
            if 'Win Rate:' in line and '%' in line:
                try:
                    results['win_rate'] = float(line.split(':')[1].strip().replace('%', ''))
                except:
                    pass
            elif 'Total Return:' in line and '%' in line:
                try:
                    results['total_return'] = float(line.split('%')[0].split()[-1].replace('(', '').replace(')', ''))
                except:
                    pass
            elif 'Profit Factor:' in line:
                try:
                    results['profit_factor'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Total Trades:' in line:
                try:
                    results['total_trades'] = int(line.split(':')[1].strip())
                except:
                    pass
        
        # Also read from CSV if available
        try:
            trades_df = pd.read_csv(f"data/optimize_{test_name}_trades.csv")
            if len(trades_df) > 0:
                results['total_trades'] = len(trades_df)
                results['win_rate'] = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100
                results['total_pnl'] = trades_df['pnl'].sum()
                results['profit_factor'] = abs(
                    trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                    trades_df[trades_df['pnl'] < 0]['pnl'].sum()
                ) if trades_df[trades_df['pnl'] < 0]['pnl'].sum() != 0 else 0
        except:
            pass
        
        return results
    except Exception as e:
        print(f"Error running backtest: {e}")
        return {}

def load_base_config() -> Dict:
    """Load base configuration."""
    with open("config/config.json", 'r') as f:
        return json.load(f)

def test_configurations():
    """Test different strategy configurations."""
    base_config = load_base_config()
    
    # Test configurations
    test_configs = [
        {
            "name": "SHORT_only",
            "description": "Only SHORT trades (LONG disabled)",
            "config": {
                **base_config,
                "strategy": {
                    **base_config["strategy"],
                    "disable_long_trades": True,
                    "strict_long_conditions": True,
                    "require_volume_confirmation": True,
                }
            }
        },
        {
            "name": "SHORT_only_stricter",
            "description": "SHORT only with stricter filters",
            "config": {
                **base_config,
                "strategy": {
                    **base_config["strategy"],
                    "disable_long_trades": True,
                    "min_histogram_strength": 0.5,
                    "min_trend_strength": 0.0005,
                    "require_volume_confirmation": True,
                    "strict_long_conditions": True,
                }
            }
        },
        {
            "name": "SHORT_only_very_strict",
            "description": "SHORT only with very strict filters",
            "config": {
                **base_config,
                "strategy": {
                    **base_config["strategy"],
                    "disable_long_trades": True,
                    "min_histogram_strength": 1.0,
                    "min_trend_strength": 0.001,
                    "require_volume_confirmation": True,
                    "strict_long_conditions": True,
                    "rsi_overbought": 75.0,
                }
            }
        },
        {
            "name": "SHORT_only_relaxed",
            "description": "SHORT only with relaxed filters",
            "config": {
                **base_config,
                "strategy": {
                    **base_config["strategy"],
                    "disable_long_trades": True,
                    "min_histogram_strength": 0.0,
                    "min_trend_strength": 0.0,
                    "require_volume_confirmation": False,
                    "strict_long_conditions": False,
                }
            }
        },
    ]
    
    results = []
    
    print("=" * 70)
    print("STRATEGY OPTIMIZATION - FINDING PROFITABLE CONFIGURATION")
    print("=" * 70)
    print()
    
    for i, test in enumerate(test_configs, 1):
        print(f"[{i}/{len(test_configs)}] Testing: {test['name']}")
        print(f"  Description: {test['description']}")
        
        result = run_backtest(test['config'], test['name'])
        result['name'] = test['name']
        result['description'] = test['description']
        results.append(result)
        
        if result:
            print(f"  Results:")
            print(f"    Total Trades: {result.get('total_trades', 'N/A')}")
            print(f"    Win Rate: {result.get('win_rate', 'N/A'):.2f}%")
            print(f"    Total Return: {result.get('total_return', 'N/A'):.2f}%")
            print(f"    Profit Factor: {result.get('profit_factor', 'N/A'):.2f}")
            print(f"    Total P&L: ${result.get('total_pnl', 'N/A'):,.2f}")
            print()
        else:
            print(f"  Failed to get results")
            print()
    
    # Find best configuration
    print("=" * 70)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 70)
    print()
    
    # Sort by total return (descending)
    profitable_results = [r for r in results if r.get('total_return', -999) > 0]
    
    if profitable_results:
        profitable_results.sort(key=lambda x: x.get('total_return', 0), reverse=True)
        print("✅ PROFITABLE CONFIGURATIONS FOUND:")
        print()
        for i, result in enumerate(profitable_results, 1):
            print(f"{i}. {result['name']}")
            print(f"   Description: {result['description']}")
            print(f"   Win Rate: {result.get('win_rate', 0):.2f}%")
            print(f"   Total Return: {result.get('total_return', 0):.2f}%")
            print(f"   Profit Factor: {result.get('profit_factor', 0):.2f}")
            print(f"   Total Trades: {result.get('total_trades', 0)}")
            print()
        
        best = profitable_results[0]
        print("=" * 70)
        print(f"🏆 BEST CONFIGURATION: {best['name']}")
        print("=" * 70)
        print(f"Description: {best['description']}")
        print(f"Win Rate: {best.get('win_rate', 0):.2f}%")
        print(f"Total Return: {best.get('total_return', 0):.2f}%")
        print(f"Profit Factor: {best.get('profit_factor', 0):.2f}")
        print(f"Total Trades: {best.get('total_trades', 0)}")
        print()
        print("Update config.json with these settings to use the best configuration.")
    else:
        print("❌ No profitable configurations found yet.")
        print("   Consider:")
        print("   - Adjusting risk/reward ratio")
        print("   - Testing different timeframes")
        print("   - Testing different symbols")
        print("   - Further tightening filters")
    
    return results

if __name__ == "__main__":
    test_configurations()

