#!/usr/bin/env python3
"""
Strategy Improvement Script
Tests various improvements to reduce stop loss hits and increase profitability
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
        elif key in config['risk']:
            config['risk'][key] = value
        elif key.startswith('trailing_'):
            if 'trailing_stop' not in config['risk']:
                config['risk']['trailing_stop'] = {}
            config['risk']['trailing_stop'][key.replace('trailing_', '')] = value
    
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
                for part in parts:
                    if '%' in part and '(' in part:
                        results['total_return_pct'] = float(part.split('(')[1].replace('%', '').replace(')', ''))
                        break
            elif 'Profit Factor:' in line:
                results['profit_factor'] = float(line.split()[-1])
            elif 'Max Drawdown:' in line and '%' in line:
                parts = line.split()
                for part in parts:
                    if '%' in part and '(' in part:
                        results['max_drawdown_pct'] = float(part.split('(')[1].replace('%', '').replace(')', ''))
                        break
        
        return results
    except Exception as e:
        print(f"Error: {e}")
        return {}

def test_improvement(name: str, params: Dict) -> Dict:
    """Test an improvement"""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"Changes: {params}")
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
    """Test various improvements"""
    
    # Current baseline
    baseline = {
        'disable_long_trades': False,
        'strict_long_conditions': False,
        'min_histogram_strength': 0.15,
        'min_trend_strength': 0.00015,
        'require_volume_confirmation': False,
        'rsi_oversold': 35.0,
        'rsi_overbought': 65.0,
        'risk_reward_ratio': 2.5
    }
    
    improvements = [
        {
            'name': 'Baseline (Current)',
            'params': baseline
        },
        {
            'name': 'Wider Stops - Higher R:R',
            'params': {
                **baseline,
                'risk_reward_ratio': 3.0  # Wider stops, higher TP
            }
        },
        {
            'name': 'Stricter Entry Filters',
            'params': {
                **baseline,
                'min_histogram_strength': 0.3,
                'min_trend_strength': 0.0003,
                'require_volume_confirmation': True
            }
        },
        {
            'name': 'Better RSI Levels',
            'params': {
                **baseline,
                'rsi_oversold': 30.0,
                'rsi_overbought': 70.0,
                'strict_long_conditions': True
            }
        },
        {
            'name': 'Combined: Stricter + Wider Stops',
            'params': {
                **baseline,
                'min_histogram_strength': 0.25,
                'min_trend_strength': 0.00025,
                'require_volume_confirmation': True,
                'risk_reward_ratio': 3.0,
                'rsi_oversold': 30.0,
                'rsi_overbought': 70.0
            }
        },
        {
            'name': 'Tighter Trailing Stop',
            'params': {
                **baseline,
                'trailing_trail_percent': 1.5,  # Tighter trail
                'trailing_activation_percent': 0.8  # Activate earlier
            }
        }
    ]
    
    results = []
    
    for improvement in improvements:
        result = test_improvement(improvement['name'], improvement['params'])
        if result:
            results.append(result)
    
    # Find best
    print(f"\n{'='*70}")
    print("IMPROVEMENT RESULTS")
    print('='*70)
    
    results.sort(key=lambda x: x.get('total_return_pct', -999), reverse=True)
    
    print("\nRanked by Total Return %:")
    print("-" * 70)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['name']}")
        print(f"   Return: {result.get('total_return_pct', 0):+.2f}%")
        print(f"   Trades: {result.get('total_trades', 0)}")
        print(f"   Win Rate: {result.get('win_rate', 0):.2f}%")
        print(f"   Profit Factor: {result.get('profit_factor', 0):.2f}")
        print(f"   Max DD: {result.get('max_drawdown_pct', 0):.2f}%")
    
    # Find profitable configs
    profitable = [r for r in results if r.get('total_return_pct', 0) > 0]
    
    if profitable:
        best = profitable[0]
        print(f"\n{'='*70}")
        print(f"✅ BEST IMPROVEMENT: {best['name']}")
        print('='*70)
        print(f"Total Return: {best.get('total_return_pct', 0):+.2f}%")
        print(f"Total Trades: {best.get('total_trades', 0)}")
        print(f"Win Rate: {best.get('win_rate', 0):.2f}%")
        print(f"Profit Factor: {best.get('profit_factor', 0):.2f}")
        print(f"\nApplying best improvement...")
        update_config(best['params'])
        print("✅ Configuration updated!")
    else:
        print("\n⚠️  No improvement found. Keeping baseline.")
        update_config(baseline)

if __name__ == '__main__':
    main()

