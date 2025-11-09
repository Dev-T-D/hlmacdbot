#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Create data directory if it doesn't exist
os.makedirs('../data', exist_ok=True)

# Generate synthetic BTCUSDT data (similar to BNB data we had before)
np.random.seed(42)
dates = pd.date_range('2024-05-01', periods=5000, freq='15min')
base_price = 60000  # BTC around $60k

# Generate realistic price movements
returns = np.random.normal(0.0001, 0.005, len(dates))  # Small drift with volatility
prices = base_price * np.exp(np.cumsum(returns))

# Create OHLCV data
df = pd.DataFrame({
    'timestamp': dates,
    'open': prices,
    'high': prices * (1 + np.random.uniform(0, 0.005, len(dates))),
    'low': prices * (1 - np.random.uniform(0, 0.005, len(dates))),
    'close': prices + np.random.normal(0, prices * 0.001, len(dates)),
    'volume': np.random.uniform(100, 1000, len(dates))
})

# Ensure high >= close >= low >= 0 and open is reasonable
df['high'] = np.maximum(df[['open', 'high', 'close']].max(axis=1), df['high'])
df['low'] = np.minimum(df[['open', 'low', 'close']].min(axis=1), df['low'])
df['close'] = np.clip(df['close'], df['low'], df['high'])

# Save to CSV
df.to_csv('../data/btcusdt_15m.csv', index=False)
print(f'Generated {len(df)} BTCUSDT 15m candles')
print(f'Date range: {df.iloc[0]["timestamp"]} to {df.iloc[-1]["timestamp"]}')
print(f'Price range: ${df["close"].min():.2f} - ${df["close"].max():.2f}')
print('Saved to data/btcusdt_15m.csv')
