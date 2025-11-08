# Trailing Stop-Loss Guide

## Overview

The trailing stop-loss feature automatically adjusts your stop-loss price as the market moves in your favor, helping you lock in profits while allowing your winning trades to run.

## How It Works

### Basic Concept

- **For LONG positions**: The stop-loss trails *below* the highest price reached
- **For SHORT positions**: The stop-loss trails *above* the lowest price reached
- The stop-loss **never moves against you** - it only moves to protect more profit

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trail_percent` | 2.0% | Distance the stop-loss trails behind the best price |
| `activation_percent` | 1.0% | Profit threshold required before trailing activates |
| `update_threshold_percent` | 0.5% | Minimum price movement to trigger stop-loss update |

## Configuration

### In `config/config.json`:

```json
"risk": {
  "leverage": 10,
  "max_position_size_pct": 0.1,
  "max_daily_loss_pct": 0.05,
  "max_trades_per_day": 10,
  "trailing_stop": {
    "enabled": true,
    "trail_percent": 2.0,
    "activation_percent": 1.0,
    "update_threshold_percent": 0.5
  }
}
```

### Enable/Disable

Set `"enabled": false` to disable trailing stop-loss completely.

## Example Scenarios

### LONG Position Example

```
Entry Price:     $50,000
Initial Stop:    $49,000 (2% below entry)

Price Movement:
1. $50,000 → $50,300 (0.6% profit)
   - Trailing NOT activated yet (need 1%)
   - Stop stays at $49,000

2. $50,500 (1.0% profit) ✅
   - Trailing ACTIVATES
   - Stop moves to $49,490 (2% below $50,500)

3. $51,500 (3.0% profit)
   - Stop moves to $50,470 (2% below $51,500)
   - You've locked in $470 profit!

4. $50,470 (pullback)
   - STOP HIT - Position closed
   - Final profit: $470 (~0.94%)
```

**Result**: Without trailing stop, a pullback to $49,000 would have resulted in a loss. With trailing stop, you locked in nearly 1% profit!

### SHORT Position Example

```
Entry Price:     $50,000
Initial Stop:    $51,000 (2% above entry)

Price Movement:
1. $50,000 → $49,500 (1.0% profit) ✅
   - Trailing ACTIVATES
   - Stop moves to $50,490 (2% above $49,500)

2. $48,500 (3.0% profit)
   - Stop moves to $49,470 (2% above $48,500)
   - You've locked in $530 profit!

3. $49,470 (bounce up)
   - STOP HIT - Position closed
   - Final profit: $530 (~1.06%)
```

## Benefits

### 1. **Automatic Profit Protection**
- No need to manually adjust stop-loss
- Locks in profits as price moves favorably
- Removes emotional decision-making

### 2. **Let Winners Run**
- Gives trades room to breathe
- Doesn't cap your upside potential
- Only exits on significant pullbacks

### 3. **Risk Management**
- Reduces average loss size
- Increases average win size
- Improves risk/reward ratio

### 4. **Works 24/7**
- Monitors prices continuously
- No need to watch charts constantly
- Works while you sleep

## How to Tune Parameters

### Trail Percent (Distance)

**Tighter (1.0% - 1.5%)**
- ✅ Locks in profits sooner
- ✅ Good for volatile markets
- ❌ May exit winning trades prematurely

**Standard (2.0% - 2.5%)**
- ✅ Balanced approach (recommended)
- ✅ Good for most market conditions
- ✅ Allows for normal pullbacks

**Wider (3.0% - 5.0%)**
- ✅ Lets trades run longer
- ✅ Good for trending markets
- ❌ May give back more profit on reversals

### Activation Percent

**Lower (0.5% - 1.0%)**
- ✅ Starts protecting profits earlier
- ❌ May activate too soon in choppy markets

**Higher (1.5% - 2.0%)**
- ✅ Only trails on strong moves
- ❌ May miss protecting smaller wins

### Update Threshold

**More Frequent (0.3% - 0.5%)**
- ✅ Stop-loss updates more often
- ✅ Tighter profit protection
- ❌ More API calls to exchange

**Less Frequent (0.7% - 1.0%)**
- ✅ Fewer exchange updates
- ❌ Less precise trailing

## Testing

Run the test suite to see trailing stop in action:

```bash
cd /home/ink/bitunix-macd-bot
python3 test_trailing_stop.py
```

## Monitoring

The bot logs trailing stop activity:

```
🟢 Trailing stop ACTIVATED at 1.23% profit
📈 LONG stop trailed: $49000.00 → $49490.00 (Best: $50000.00 → $50500.00)
📊 Holding LONG position | Trailing: ACTIVE (SL: $50470.00)
🛑 Trailing Stop Hit ($50470.00)
```

## Best Practices

### 1. **Backtest First**
- Test different parameters with historical data
- Find settings that work for your market/timeframe

### 2. **Start Conservative**
- Begin with wider trail_percent (2.5% - 3.0%)
- Gradually tighten as you get comfortable

### 3. **Consider Volatility**
- Crypto is volatile - don't trail too tight
- Adjust trail_percent based on average daily range

### 4. **Monitor Performance**
- Track how often trailing stop is hit
- Compare P&L with/without trailing stop
- Adjust parameters based on results

### 5. **Combine with Strategy**
- Trailing stop works with your MACD signals
- Still respects original take-profit targets
- Provides an additional exit mechanism

## Dry Run Mode

In dry run mode (recommended for testing):
- Trailing stop is tracked internally
- Stop-loss updates are logged but NOT sent to exchange
- Perfect for testing without risk

Set `"dry_run": true` in config.json

## Live Trading

When ready for live trading:
1. Set `"dry_run": false` in config.json
2. Bot will update stop-loss orders on exchange
3. Monitor logs carefully during first few trades
4. Start with small position sizes

## Troubleshooting

### Trailing Stop Not Activating
- Check activation_percent - may be too high
- Verify trades are reaching profit threshold
- Confirm trailing_stop.enabled is true

### Stop Getting Hit Too Soon
- Increase trail_percent (give more room)
- Consider market volatility
- Check if activation_percent is too low

### Stop Not Updating on Exchange
- Check API key permissions
- Verify dry_run setting
- Review logs for error messages

## Technical Details

### Files Modified

1. **config/config.json**
   - Added trailing_stop configuration section

2. **risk_manager.py**
   - New `TrailingStopLoss` class
   - Handles all trailing stop logic

3. **bitunix_client.py**
   - New `update_stop_loss()` method
   - Updates stop-loss on exchange

4. **trading_bot.py**
   - Integrated trailing stop into main loop
   - Initializes on position entry
   - Updates during position monitoring
   - Checks for stop-loss hits

### Safety Features

- ✅ Stop-loss never moves against you
- ✅ Activation threshold prevents premature trailing
- ✅ Update threshold prevents excessive API calls
- ✅ Full logging of all adjustments
- ✅ Graceful error handling
- ✅ Compatible with dry run mode

## FAQs

**Q: Will this work with my existing positions?**
A: Trailing stop initializes on new positions. Existing positions use original stop-loss.

**Q: Can I use trailing stop without the MACD strategy?**
A: Yes! The trailing stop is independent of the entry strategy.

**Q: Does it work for both LONG and SHORT?**
A: Yes, fully supports both directions with appropriate logic.

**Q: What if the exchange API fails?**
A: Bot tracks stop internally, logs error, continues monitoring.

**Q: Can I manually override the trailing stop?**
A: Not recommended - defeats the purpose. Adjust parameters instead.

## Support

For issues or questions:
1. Check logs in `logs/bot.log`
2. Run test suite: `python3 test_trailing_stop.py`
3. Review this guide
4. Test in dry_run mode first

---

**Remember**: Trailing stops are a tool, not a guarantee. Always trade responsibly and never risk more than you can afford to lose.

