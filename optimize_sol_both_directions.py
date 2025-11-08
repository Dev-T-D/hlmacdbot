#!/usr/bin/env python3
"""
Optimize SOL strategy for both LONG and SHORT trades
Tests multiple parameter combinations to find profitable configuration
"""

import json
import subprocess
import sys
from typing import Dict, List

def update_config(params: Dict) -> None:
    """Update config.json with new parameters"""
    config = json.load(open('config/config.json'))
    
    for key, value in params.items():
        if key in config['strategy']:
            config['strategy'][key] = value
    
    json.dump(config, open('config/config.json', 'w'), indent=2)

def run_backtest() -> Dict:
    """Run backtest and parse results"""
    try:
        result = subprocess.run(
            ['python3', 'run_backtest.py',
             '--data', 'data/SOL_15m.csv',
             '--config', 'config/config.json',
             '--initial-balance', '10000',
             '--leverage', '10',
             '--slippage', '0.001'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        output = result.stdout + result.stderr
        
        # Parse results
        results = {}
        for line in output.split('\n'):
            if 'Total Trades:' in line:
                results['total_trades'] = int(line.split()[-1])
            elif 'Win Rate:' in line:
                results['win_rate'] = float(line.split()[-1].replace('%', ''))
            elif 'Total Return:' in line and '%' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if '%' in part:
                        results['total_return_pct'] = float(part.replace('%', '').replace('(', '').replace(')', ''))
                        break
            elif 'Profit Factor:' in line:
                results['profit_factor'] = float(line.split()[-1])
            elif 'Sharpe Ratio:' in line:
                results['sharpe_ratio'] = float(line.split()[-1])
            elif 'Max Drawdown:' in line and '%' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if '%' in part:
                        results['max_drawdown_pct'] = float(part.replace('%', '').replace('(', '').replace(')', ''))
                        break
        
        return results
    except Exception as e:
        print(f"Error running backtest: {e}")
        return {}

def test_configuration(name: str, params: Dict) -> Dict:
    """Test a configuration and return results"""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"Parameters: {params}")
    print('='*70)
    
    update_config(params)
    results = run_backtest()
    
    if results:
        print(f"Results: {results}")
        return {
            'name': name,
            'params': params,
            **results
        }
    return None

def main():
    """Main optimization loop"""
    
    # Test configurations
    test_configs = [
        {
            'name': 'Balanced - Moderate Filters',
            'params': {
                'disable_long_trades': False,
                'strict_long_conditions': False,
                'min_histogram_strength': 0.0,
                'min_trend_strength': 0.0,
                'require_volume_confirmation': False,
                'rsi_oversold': 30.0,
                'rsi_overbought': 70.0,
                'risk_reward_ratio': 2.5
            }
        },
        {
            'name': 'Relaxed - More Trades',
            'params': {
                'disable_long_trades': False,
                'strict_long_conditions': False,
                'min_histogram_strength': 0.0,
                'min_trend_strength': 0.0,
                'require_volume_confirmation': False,
                'rsi_oversold': 40.0,
                'rsi_overbought': 60.0,
                'risk_reward_ratio': 2.0
            }
        },
        {
            'name': 'Strict Filters - Quality Trades',
            'params': {
                'disable_long_trades': False,
                'strict_long_conditions': True,
                'min_histogram_strength': 0.5,
                'min_trend_strength': 0.0005,
                'require_volume_confirmation': True,
                'rsi_oversold': 35.0,
                'rsi_overbought': 65.0,
                'risk_reward_ratio': 3.0
            }
        },
        {
            'name': 'Volume Confirmed',
            'params': {
                'disable_long_trades': False,
                'strict_long_conditions': False,
                'min_histogram_strength': 0.2,
                'min_trend_strength': 0.0002,
                'require_volume_confirmation': True,
                'rsi_oversold': 35.0,
                'rsi_overbought': 65.0,
                'risk_reward_ratio': 2.5
            }
        },
        {
            'name': 'High R:R - Fewer Trades',
            'params': {
                'disable_long_trades': False,
                'strict_long_conditions': False,
                'min_histogram_strength': 0.3,
                'min_trend_strength': 0.0003,
                'require_volume_confirmation': False,
                'rsi_oversold': 30.0,
                'rsi_overbought': 70.0,
                'risk_reward_ratio': 3.5
            }
        }
    ]
    
    results = []
    
    for config in test_configs:
        result = test_configuration(config['name'], config['params'])
        if result:
            results.append(result)
    
    # Find best configuration
    print(f"\n{'='*70}")
    print("OPTIMIZATION RESULTS")
    print('='*70)
    
    # Sort by total return
    results.sort(key=lambda x: x.get('total_return_pct', -999), reverse=True)
    
    print("\nTop Configurations (sorted by Total Return %):")
    print("-" * 70)
    
    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. {result['name']}")
        print(f"   Total Return: {result.get('total_return_pct', 0):+.2f}%")
        print(f"   Total Trades: {result.get('total_trades', 0)}")
        print(f"   Win Rate: {result.get('win_rate', 0):.2f}%")
        print(f"   Profit Factor: {result.get('profit_factor', 0):.2f}")
        print(f"   Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
        print(f"   Max Drawdown: {result.get('max_drawdown_pct', 0):.2f}%")
        print(f"   Parameters: {result['params']}")
    
    # Find profitable configs
    profitable = [r for r in results if r.get('total_return_pct', 0) > 0]
    
    if profitable:
        best = profitable[0]
        print(f"\n{'='*70}")
        print(f"✅ BEST PROFITABLE CONFIGURATION: {best['name']}")
        print('='*70)
        print(f"Total Return: {best.get('total_return_pct', 0):+.2f}%")
        print(f"Total Trades: {best.get('total_trades', 0)}")
        print(f"Win Rate: {best.get('win_rate', 0):.2f}%")
        print(f"Profit Factor: {best.get('profit_factor', 0):.2f}")
        print(f"\nApplying best configuration to config.json...")
        update_config(best['params'])
        print("✅ Configuration updated!")
    else:
        print("\n⚠️  No profitable configuration found. Best configuration:")
        best = results[0]
        print(f"   {best['name']}: {best.get('total_return_pct', 0):+.2f}%")
        update_config(best['params'])
        print("✅ Best configuration applied (may need further optimization)")

if __name__ == '__main__':
    main()

