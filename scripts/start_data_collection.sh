#!/bin/bash

echo "🚀 Starting Hyperliquid Real-Time Data Collection"

# Activate virtual environment
source .venv/bin/activate

# Start data collector in background
echo "📊 Starting data collector..."
nohup python -m data_collection.hyperliquid_collector > logs/data_collector.log 2>&1 &
COLLECTOR_PID=$!

echo "   Collector PID: $COLLECTOR_PID"
echo $COLLECTOR_PID > logs/collector.pid

# Wait a moment for collector to start
sleep 5

# Start auto-retrainer in background
echo "🤖 Starting auto-retrainer..."
nohup python -m ml_training.auto_retrain > logs/auto_retrain.log 2>&1 &
RETRAIN_PID=$!

echo "   Retrainer PID: $RETRAIN_PID"
echo $RETRAIN_PID > logs/retrain.pid

echo ""
echo "✅ Services started!"
echo ""

echo "Monitor logs:"
echo "   Data Collector: tail -f logs/data_collector.log"
echo "   Auto Retrain:   tail -f logs/auto_retrain.log"
echo ""

echo "Stop services:"
echo "   kill \$(cat logs/collector.pid)"
echo "   kill \$(cat logs/retrain.pid)"
