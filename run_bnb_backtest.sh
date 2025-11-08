#!/bin/bash
# Quick script to run backtest on BNB data

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run backtest with BNB data
python3 run_backtest.py \
    --data data/BNB_30m_5000.csv \
    --config config/config.json \
    --initial-balance 10000 \
    --leverage 10 \
    --slippage 0.001 \
    --export-trades data/bnb_backtest_trades.csv \
    --export-equity data/bnb_backtest_equity.csv

echo ""
echo "✅ Backtest complete!"
echo "📊 Check results in:"
echo "   - Trades: data/bnb_backtest_trades.csv"
echo "   - Equity curve: data/bnb_backtest_equity.csv"

