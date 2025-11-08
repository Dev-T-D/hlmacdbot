# Logging Configuration Guide

## Overview

The trading bot uses a comprehensive logging setup with rotation, different log levels per handler, and separate error logging.

## Log Files

### Main Log File (`logs/bot.log`)
- **Level**: DEBUG (captures all log levels)
- **Format**: Detailed with function name and line number
- **Rotation**: 10MB per file, keeps 5 backups
- **Purpose**: Complete log of all bot activities for debugging

**Example format**:
```
2025-11-08 14:40:18 - trading_bot - INFO - setup_logging:130 - Logging configured:
```

### Error Log File (`logs/bot_errors.log`)
- **Level**: ERROR (only errors and critical messages)
- **Format**: Same detailed format as main log
- **Rotation**: 5MB per file, keeps 3 backups
- **Purpose**: Quick access to errors without filtering through debug logs

**Example format**:
```
2025-11-08 14:40:18 - trading_bot - ERROR - run_trading_cycle:2301 - Error in trading cycle: ...
```

### Console Output (stdout)
- **Level**: INFO (less verbose)
- **Format**: Simplified format (time, level, message)
- **Purpose**: Clean terminal output for monitoring

**Example format**:
```
14:40:18 - INFO - Logging configured:
```

## Log Levels

The bot uses standard Python logging levels:

- **DEBUG**: Detailed diagnostic information (file only)
- **INFO**: General informational messages (file + console)
- **WARNING**: Warning messages (file + console)
- **ERROR**: Error messages (file + console + error log)
- **CRITICAL**: Critical errors (file + console + error log)

## Log Rotation

Logs automatically rotate when they reach size limits:

- **Main log**: Rotates at 10MB, keeps 5 backups
  - `bot.log` (current)
  - `bot.log.1` (most recent backup)
  - `bot.log.2` (older backup)
  - ... up to `bot.log.5`

- **Error log**: Rotates at 5MB, keeps 3 backups
  - `bot_errors.log` (current)
  - `bot_errors.log.1` (most recent backup)
  - ... up to `bot_errors.log.3`

When a log file reaches its size limit:
1. Current log is renamed (e.g., `bot.log` → `bot.log.1`)
2. Older backups are shifted (e.g., `bot.log.1` → `bot.log.2`)
3. New log file is created
4. Oldest backup is deleted if backup count exceeded

## Configuration

The logging setup is configured in `trading_bot.py`:

```python
def setup_logging():
    """Configure logging with rotation and different levels per handler."""
    # ... configuration code ...
```

### Customization

To modify logging behavior, edit the `setup_logging()` function:

**Change log levels**:
```python
file_handler.setLevel(logging.INFO)  # Change file handler level
console_handler.setLevel(logging.WARNING)  # Change console level
```

**Change rotation size**:
```python
RotatingFileHandler(
    "logs/bot.log",
    maxBytes=20 * 1024 * 1024,  # 20MB instead of 10MB
    backupCount=10,  # Keep 10 backups instead of 5
)
```

**Change log format**:
```python
file_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",  # Simpler format
    datefmt="%Y-%m-%d %H:%M:%S",
)
```

## Usage

### Viewing Logs

**Main log** (all activities):
```bash
tail -f logs/bot.log
```

**Error log** (errors only):
```bash
tail -f logs/bot_errors.log
```

**Search logs**:
```bash
grep "ERROR" logs/bot.log
grep "order" logs/bot.log | tail -20
```

**View recent entries**:
```bash
tail -100 logs/bot.log
```

### Log Analysis

**Count errors**:
```bash
grep -c "ERROR" logs/bot.log
```

**Find specific function calls**:
```bash
grep "place_order" logs/bot.log
```

**View logs from specific time**:
```bash
grep "2025-11-08 14:" logs/bot.log
```

## Best Practices

1. **Monitor error log regularly**: Check `bot_errors.log` for issues
2. **Archive old logs**: Move rotated logs to archive storage periodically
3. **Set up log monitoring**: Use tools like `logrotate` or monitoring services
4. **Don't commit logs**: Log files are in `.gitignore`
5. **Use DEBUG sparingly**: Only enable DEBUG for troubleshooting

## Troubleshooting

### Logs not being created
- Check `logs/` directory exists and is writable
- Verify file permissions: `chmod 755 logs/`

### Logs growing too large
- Reduce `maxBytes` in `RotatingFileHandler`
- Reduce `backupCount` to keep fewer backups
- Set console handler to WARNING level to reduce file logging

### Missing log entries
- Check log level settings (DEBUG captures everything)
- Verify handlers are not filtered out
- Check disk space availability

### Console too verbose
- Increase console handler level to WARNING:
  ```python
  console_handler.setLevel(logging.WARNING)
  ```

## Integration with Other Components

The logging setup integrates with:

- **Audit Logger**: Separate audit log (`logs/audit.log`) for trading activities
- **Health Monitor**: Uses same logging system for health check logs
- **Backtester**: Uses similar logging setup in `run_backtest.py`

## See Also

- `trading_bot.py` - Main logging configuration
- `audit_logger.py` - Audit logging for trading activities
- `health_monitor.py` - Health check logging
- `.gitignore` - Log file exclusion patterns

