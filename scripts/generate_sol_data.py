#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Create data directory if it doesn't exist
os.makedirs('../data', exist_ok=True)

# Generate synthetic SOL-USDC data (SOL is more volatile than BTC)
np.random.seed(123)  # Different seed for SOL
dates = pd.date_range('2024-05-01', periods=5000, freq='15min')
base_price = 150  # SOL around $150

# SOL is more volatile than BTC - higher volatility parameters
returns = np.random.normal(0.0002, 0.008, len(dates))  # Higher drift and volatility
prices = base_price * np.exp(np.cumsum(returns))

# Create OHLCV data with SOL's higher volatility
df = pd.DataFrame({
    'timestamp': dates,
    'open': prices,
    'high': prices * (1 + np.random.uniform(0, 0.010, len(dates))),  # Higher volatility
    'low': prices * (1 - np.random.uniform(0, 0.010, len(dates))),
    'close': prices + np.random.normal(0, prices * 0.002, len(dates)),  # Higher noise
    'volume': np.random.uniform(500, 2000, len(dates))  # SOL typically has good volume
})

# Ensure high >= close >= low >= 0 and open is reasonable
df['high'] = np.maximum(df[['open', 'high', 'close']].max(axis=1), df['high'])
df['low'] = np.minimum(df[['open', 'low', 'close']].min(axis=1), df['low'])
df['close'] = np.clip(df['close'], df['low'], df['high'])

# Save to CSV
df.to_csv('../data/sol_usdc_15m.csv', index=False)
print(f'Generated {len(df)} SOL-USDC 15m candles')
print(f'Date range: {df.iloc[0]["timestamp"]} to {df.iloc[-1]["timestamp"]}')
print(f'Price range: ${df["close"].min():.2f} - ${df["close"].max():.2f}')
print('Saved to data/sol_usdc_15m.csv')

# Calculate volatility comparison
volatility = df["close"].pct_change().std() * 100
print(f'SOL volatility is ~{volatility:.1f}% per 15min candle (typically higher than BTC)')
