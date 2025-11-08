"""
Grafana Dashboard Configurations for Trading Bot

Creates JSON configurations for comprehensive Grafana dashboards
that visualize trading bot metrics, performance, and health.

Dashboards include:
- Trading Performance Overview
- System Health & Resources
- Risk Management
- API Performance
- Strategy Analytics
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path


def create_trading_performance_dashboard() -> Dict[str, Any]:
    """Create trading performance dashboard configuration."""
    return {
        "dashboard": {
            "id": None,
            "title": "Trading Bot - Performance Overview",
            "tags": ["trading", "performance", "bot"],
            "timezone": "UTC",
            "panels": [
                # Row 1: Key Performance Indicators
                {
                    "id": 1,
                    "title": "Key Performance Metrics",
                    "type": "stat",
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 0},
                    "targets": [
                        {
                            "expr": "trading_bot_total_pnl",
                            "legendFormat": "Total P&L"
                        },
                        {
                            "expr": "rate(trading_bot_trades_total[1h]) * 3600",
                            "legendFormat": "Trades per Hour"
                        },
                        {
                            "expr": "trading_bot_win_rate_ratio",
                            "legendFormat": "Win Rate"
                        },
                        {
                            "expr": "trading_bot_current_drawdown_percent",
                            "legendFormat": "Current Drawdown %"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "mappings": [],
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "red", "value": 80}
                                ]
                            }
                        }
                    }
                },

                # Row 2: P&L Over Time
                {
                    "id": 2,
                    "title": "P&L Over Time",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                    "targets": [
                        {
                            "expr": "trading_bot_total_pnl",
                            "legendFormat": "Total P&L ($)"
                        },
                        {
                            "expr": "trading_bot_unrealized_pnl",
                            "legendFormat": "Unrealized P&L ($)"
                        }
                    ],
                    "yAxes": [
                        {"format": "currencyUSD", "label": "P&L ($)"},
                        {"format": "short"}
                    ]
                },

                # Row 2: Win Rate Trend
                {
                    "id": 3,
                    "title": "Win Rate Trend",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                    "targets": [
                        {
                            "expr": "trading_bot_win_rate_ratio",
                            "legendFormat": "Win Rate"
                        }
                    ],
                    "yAxes": [
                        {"format": "percent", "max": 1, "min": 0},
                        {"format": "short"}
                    ]
                },

                # Row 3: Trading Activity
                {
                    "id": 4,
                    "title": "Trading Activity (Last 24h)",
                    "type": "bargauge",
                    "gridPos": {"h": 8, "w": 8, "x": 0, "y": 16},
                    "targets": [
                        {
                            "expr": "sum(rate(trading_bot_trades_total[24h])) by (side)",
                            "legendFormat": "{{side}} trades"
                        }
                    ]
                },

                # Row 3: Position Size
                {
                    "id": 5,
                    "title": "Current Position Size",
                    "type": "gauge",
                    "gridPos": {"h": 8, "w": 8, "x": 8, "y": 16},
                    "targets": [
                        {
                            "expr": "trading_bot_current_positions",
                            "legendFormat": "Open Positions"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "mappings": [],
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "yellow", "value": 1},
                                    {"color": "red", "value": 5}
                                ]
                            }
                        }
                    }
                },

                # Row 3: Risk Limits
                {
                    "id": 6,
                    "title": "Risk Limits",
                    "type": "table",
                    "gridPos": {"h": 8, "w": 8, "x": 16, "y": 16},
                    "targets": [
                        {
                            "expr": "trading_bot_risk_limit_utilization_ratio",
                            "legendFormat": "{{limit_type}}"
                        },
                        {
                            "expr": "trading_bot_daily_loss_percent",
                            "legendFormat": "Daily Loss %"
                        },
                        {
                            "expr": "trading_bot_max_position_size_percent",
                            "legendFormat": "Max Position %"
                        }
                    ]
                },

                # Row 4: Trade Distribution
                {
                    "id": 7,
                    "title": "Trade Outcome Distribution",
                    "type": "piechart",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 24},
                    "targets": [
                        {
                            "expr": "sum(trading_bot_trades_total) by (outcome)",
                            "legendFormat": "{{outcome}}"
                        }
                    ]
                },

                # Row 4: Strategy Signals
                {
                    "id": 8,
                    "title": "Strategy Signals (Last 1h)",
                    "type": "barchart",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24},
                    "targets": [
                        {
                            "expr": "sum(rate(trading_bot_signals_generated_total[1h])) by (signal_type)",
                            "legendFormat": "{{signal_type}}"
                        }
                    ]
                }
            ],
            "time": {"from": "now-24h", "to": "now"},
            "timepicker": {},
            "templating": {
                "list": []
            },
            "annotations": {
                "list": [
                    {
                        "name": "Trade Annotations",
                        "datasource": "prometheus",
                        "expr": "trading_bot_trades_total",
                        "titleFormat": "{{side}} {{quantity}} {{symbol}}",
                        "textFormat": "P&L: {{pnl}}"
                    }
                ]
            },
            "refresh": "30s",
            "schemaVersion": 27,
            "version": 0,
            "links": []
        }
    }


def create_system_health_dashboard() -> Dict[str, Any]:
    """Create system health and resources dashboard."""
    return {
        "dashboard": {
            "id": None,
            "title": "Trading Bot - System Health",
            "tags": ["trading", "system", "health"],
            "timezone": "UTC",
            "panels": [
                # Row 1: System Resources
                {
                    "id": 1,
                    "title": "CPU Usage",
                    "type": "gauge",
                    "gridPos": {"h": 6, "w": 8, "x": 0, "y": 0},
                    "targets": [
                        {
                            "expr": "trading_bot_cpu_usage_percent",
                            "legendFormat": "CPU %"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "mappings": [],
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "yellow", "value": 70},
                                    {"color": "red", "value": 90}
                                ]
                            }
                        }
                    }
                },

                {
                    "id": 2,
                    "title": "Memory Usage",
                    "type": "gauge",
                    "gridPos": {"h": 6, "w": 8, "x": 8, "y": 0},
                    "targets": [
                        {
                            "expr": "trading_bot_memory_usage_bytes",
                            "legendFormat": "Memory Usage"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "bytes",
                            "mappings": [],
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "yellow", "value": 8e8},  # 800MB
                                    {"color": "red", "value": 1.5e9}    # 1.5GB
                                ]
                            }
                        }
                    }
                },

                {
                    "id": 3,
                    "title": "WebSocket Status",
                    "type": "stat",
                    "gridPos": {"h": 6, "w": 8, "x": 16, "y": 0},
                    "targets": [
                        {
                            "expr": "trading_bot_websocket_connected",
                            "legendFormat": "WebSocket Connected"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "mappings": [
                                {"options": {"0": {"text": "Disconnected", "color": "red"}},
                                 "options": {"1": {"text": "Connected", "color": "green"}}}
                            ]
                        }
                    }
                },

                # Row 2: API Performance
                {
                    "id": 4,
                    "title": "API Request Latency",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 6},
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(trading_bot_api_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "95th percentile"
                        },
                        {
                            "expr": "histogram_quantile(0.50, rate(trading_bot_api_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "Median"
                        }
                    ],
                    "yAxes": [
                        {"format": "s", "label": "Latency (seconds)"},
                        {"format": "short"}
                    ]
                },

                {
                    "id": 5,
                    "title": "API Error Rate",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 6},
                    "targets": [
                        {
                            "expr": "rate(trading_bot_api_errors_total[5m])",
                            "legendFormat": "Errors per second"
                        }
                    ],
                    "yAxes": [
                        {"format": "reqps", "label": "Errors/sec"},
                        {"format": "short"}
                    ]
                },

                # Row 3: Order Execution
                {
                    "id": 6,
                    "title": "Order Execution Time",
                    "type": "heatmap",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 14},
                    "targets": [
                        {
                            "expr": "trading_bot_order_execution_duration_seconds",
                            "legendFormat": "Execution time"
                        }
                    ]
                },

                {
                    "id": 7,
                    "title": "Circuit Breaker Status",
                    "type": "table",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 14},
                    "targets": [
                        {
                            "expr": "rate(trading_bot_circuit_breaker_trips_total[1h])",
                            "legendFormat": "{{reason}}"
                        }
                    ]
                },

                # Row 4: System Logs
                {
                    "id": 8,
                    "title": "Recent System Events",
                    "type": "logs",
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 22},
                    "targets": [
                        {
                            "expr": '{job="trading_bot"} | json',
                            "legendFormat": ""
                        }
                    ],
                    "options": {
                        "showTime": True,
                        "showLabels": False,
                        "showCommonLabels": False,
                        "wrapLogMessage": True
                    }
                }
            ],
            "time": {"from": "now-1h", "to": "now"},
            "timepicker": {},
            "templating": {
                "list": []
            },
            "annotations": {
                "list": []
            },
            "refresh": "10s",
            "schemaVersion": 27,
            "version": 0,
            "links": []
        }
    }


def create_risk_management_dashboard() -> Dict[str, Any]:
    """Create risk management dashboard."""
    return {
        "dashboard": {
            "id": None,
            "title": "Trading Bot - Risk Management",
            "tags": ["trading", "risk", "management"],
            "timezone": "UTC",
            "panels": [
                # Row 1: Risk Gauges
                {
                    "id": 1,
                    "title": "Current Drawdown",
                    "type": "gauge",
                    "gridPos": {"h": 6, "w": 8, "x": 0, "y": 0},
                    "targets": [
                        {
                            "expr": "trading_bot_current_drawdown_percent",
                            "legendFormat": "Drawdown %"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent",
                            "mappings": [],
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "yellow", "value": 5},
                                    {"color": "red", "value": 10}
                                ]
                            }
                        }
                    }
                },

                {
                    "id": 2,
                    "title": "Daily Loss Limit",
                    "type": "gauge",
                    "gridPos": {"h": 6, "w": 8, "x": 8, "y": 0},
                    "targets": [
                        {
                            "expr": "trading_bot_daily_loss_percent",
                            "legendFormat": "Daily Loss %"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent",
                            "mappings": [],
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "yellow", "value": 5},
                                    {"color": "red", "value": 8}
                                ]
                            }
                        }
                    }
                },

                {
                    "id": 3,
                    "title": "Position Size Limit",
                    "type": "gauge",
                    "gridPos": {"h": 6, "w": 8, "x": 16, "y": 0},
                    "targets": [
                        {
                            "expr": "trading_bot_max_position_size_percent",
                            "legendFormat": "Position Size %"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent",
                            "mappings": [],
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "yellow", "value": 15},
                                    {"color": "red", "value": 25}
                                ]
                            }
                        }
                    }
                },

                # Row 2: Risk Over Time
                {
                    "id": 4,
                    "title": "Risk Metrics Over Time",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 6},
                    "targets": [
                        {
                            "expr": "trading_bot_current_drawdown_percent",
                            "legendFormat": "Drawdown %"
                        },
                        {
                            "expr": "trading_bot_daily_loss_percent",
                            "legendFormat": "Daily Loss %"
                        },
                        {
                            "expr": "trading_bot_risk_limit_utilization_ratio",
                            "legendFormat": "Risk Limit Utilization"
                        }
                    ],
                    "yAxes": [
                        {"format": "percent", "label": "Risk %"},
                        {"format": "short"}
                    ]
                },

                # Row 3: Stop Loss Performance
                {
                    "id": 5,
                    "title": "Trailing Stop Updates",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 14},
                    "targets": [
                        {
                            "expr": "rate(trading_bot_trailing_stop_updates_total[5m])",
                            "legendFormat": "Stop updates per minute"
                        }
                    ]
                },

                {
                    "id": 6,
                    "title": "Risk Events",
                    "type": "table",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 14},
                    "targets": [
                        {
                            "expr": "trading_bot_circuit_breaker_trips_total",
                            "legendFormat": "{{reason}}"
                        }
                    ]
                },

                # Row 4: VaR Analysis
                {
                    "id": 7,
                    "title": "Value at Risk (Historical)",
                    "type": "stat",
                    "gridPos": {"h": 6, "w": 12, "x": 0, "y": 22},
                    "targets": [
                        {
                            "expr": "trading_bot_var_95",  # Would need to be exposed from analytics
                            "legendFormat": "VaR 95%"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "currencyUSD",
                            "mappings": [],
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "red", "value": 1000}
                                ]
                            }
                        }
                    }
                },

                {
                    "id": 8,
                    "title": "Expected Shortfall",
                    "type": "stat",
                    "gridPos": {"h": 6, "w": 12, "x": 12, "y": 22},
                    "targets": [
                        {
                            "expr": "trading_bot_expected_shortfall_95",
                            "legendFormat": "ES 95%"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "currencyUSD",
                            "mappings": [],
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "red", "value": 2000}
                                ]
                            }
                        }
                    }
                }
            ],
            "time": {"from": "now-24h", "to": "now"},
            "timepicker": {},
            "templating": {
                "list": []
            },
            "annotations": {
                "list": [
                    {
                        "name": "Risk Events",
                        "datasource": "prometheus",
                        "expr": "trading_bot_circuit_breaker_trips_total > 0",
                        "titleFormat": "Circuit Breaker: {{reason}}",
                        "textFormat": "Risk limit tripped"
                    }
                ]
            },
            "refresh": "30s",
            "schemaVersion": 27,
            "version": 0,
            "links": []
        }
    }


def create_strategy_analytics_dashboard() -> Dict[str, Any]:
    """Create strategy analytics dashboard."""
    return {
        "dashboard": {
            "id": None,
            "title": "Trading Bot - Strategy Analytics",
            "tags": ["trading", "strategy", "analytics"],
            "timezone": "UTC",
            "panels": [
                # Row 1: Strategy Performance
                {
                    "id": 1,
                    "title": "Strategy Performance Comparison",
                    "type": "table",
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 0},
                    "targets": [
                        {
                            "expr": "trading_bot_strategy_pnl_total",
                            "legendFormat": "{{strategy}} P&L"
                        },
                        {
                            "expr": "trading_bot_strategy_win_rate",
                            "legendFormat": "{{strategy}} Win Rate"
                        },
                        {
                            "expr": "trading_bot_strategy_profit_factor",
                            "legendFormat": "{{strategy}} Profit Factor"
                        }
                    ]
                },

                # Row 2: Signal Generation
                {
                    "id": 2,
                    "title": "Signals Generated by Type",
                    "type": "barchart",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                    "targets": [
                        {
                            "expr": "sum(rate(trading_bot_signals_generated_total[1h])) by (signal_type)",
                            "legendFormat": "{{signal_type}}"
                        }
                    ]
                },

                {
                    "id": 3,
                    "title": "Entry Conditions Met",
                    "type": "barchart",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                    "targets": [
                        {
                            "expr": "rate(trading_bot_entry_conditions_met_total[1h])",
                            "legendFormat": "{{condition_type}}"
                        }
                    ]
                },

                # Row 3: Strategy Timing
                {
                    "id": 4,
                    "title": "Best Trading Hours",
                    "type": "heatmap",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                    "targets": [
                        {
                            "expr": "trading_bot_hourly_performance",
                            "legendFormat": "Hour {{hour}}"
                        }
                    ]
                },

                {
                    "id": 5,
                    "title": "Strategy Consistency",
                    "type": "bargauge",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
                    "targets": [
                        {
                            "expr": "trading_bot_strategy_consistency",
                            "legendFormat": "{{strategy}}"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent",
                            "mappings": [],
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "red", "value": 0},
                                    {"color": "yellow", "value": 0.5},
                                    {"color": "green", "value": 0.8}
                                ]
                            }
                        }
                    }
                },

                # Row 4: Indicator Performance
                {
                    "id": 6,
                    "title": "Indicator Calculation Time",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 24},
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(trading_bot_strategy_calculation_duration_seconds_bucket[5m]))",
                            "legendFormat": "95th percentile"
                        }
                    ],
                    "yAxes": [
                        {"format": "s", "label": "Calculation Time"},
                        {"format": "short"}
                    ]
                },

                {
                    "id": 7,
                    "title": "Strategy Sharpe Ratio",
                    "type": "barchart",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24},
                    "targets": [
                        {
                            "expr": "trading_bot_strategy_sharpe_ratio",
                            "legendFormat": "{{strategy}}"
                        }
                    ]
                }
            ],
            "time": {"from": "now-7d", "to": "now"},
            "timepicker": {},
            "templating": {
                "list": []
            },
            "annotations": {
                "list": []
            },
            "refresh": "5m",
            "schemaVersion": 27,
            "version": 0,
            "links": []
        }
    }


def create_api_performance_dashboard() -> Dict[str, Any]:
    """Create API performance dashboard."""
    return {
        "dashboard": {
            "id": None,
            "title": "Trading Bot - API Performance",
            "tags": ["trading", "api", "performance"],
            "timezone": "UTC",
            "panels": [
                # Row 1: API Health Overview
                {
                    "id": 1,
                    "title": "API Request Rate",
                    "type": "graph",
                    "gridPos": {"h": 6, "w": 12, "x": 0, "y": 0},
                    "targets": [
                        {
                            "expr": "rate(trading_bot_api_request_duration_seconds_count[5m])",
                            "legendFormat": "Requests/sec"
                        }
                    ]
                },

                {
                    "id": 2,
                    "title": "API Error Rate",
                    "type": "graph",
                    "gridPos": {"h": 6, "w": 12, "x": 12, "y": 0},
                    "targets": [
                        {
                            "expr": "rate(trading_bot_api_errors_total[5m])",
                            "legendFormat": "Errors/sec"
                        }
                    ]
                },

                # Row 2: Latency Analysis
                {
                    "id": 3,
                    "title": "API Latency Percentiles",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 6},
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.50, rate(trading_bot_api_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "50th percentile (median)"
                        },
                        {
                            "expr": "histogram_quantile(0.95, rate(trading_bot_api_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "95th percentile"
                        },
                        {
                            "expr": "histogram_quantile(0.99, rate(trading_bot_api_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "99th percentile"
                        }
                    ],
                    "yAxes": [
                        {"format": "s", "label": "Latency (seconds)"},
                        {"format": "short"}
                    ]
                },

                {
                    "id": 4,
                    "title": "API Latency by Endpoint",
                    "type": "heatmap",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 6},
                    "targets": [
                        {
                            "expr": "rate(trading_bot_api_request_duration_seconds_sum[5m]) / rate(trading_bot_api_request_duration_seconds_count[5m])",
                            "legendFormat": "{{endpoint}}"
                        }
                    ]
                },

                # Row 3: Success Rates
                {
                    "id": 5,
                    "title": "API Success Rate by Endpoint",
                    "type": "bargauge",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 14},
                    "targets": [
                        {
                            "expr": "rate(trading_bot_api_request_duration_seconds_count{status='success'}[5m]) / rate(trading_bot_api_request_duration_seconds_count[5m])",
                            "legendFormat": "{{endpoint}}"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percentunit",
                            "mappings": [],
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "red", "value": 0},
                                    {"color": "yellow", "value": 0.95},
                                    {"color": "green", "value": 0.99}
                                ]
                            }
                        }
                    }
                },

                {
                    "id": 6,
                    "title": "WebSocket Reconnections",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 14},
                    "targets": [
                        {
                            "expr": "rate(trading_bot_websocket_reconnects_total[5m])",
                            "legendFormat": "Reconnects/min"
                        }
                    ]
                },

                # Row 4: Request Distribution
                {
                    "id": 7,
                    "title": "Requests by HTTP Method",
                    "type": "piechart",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 22},
                    "targets": [
                        {
                            "expr": "sum(rate(trading_bot_api_request_duration_seconds_count[1h])) by (method)",
                            "legendFormat": "{{method}}"
                        }
                    ]
                },

                {
                    "id": 8,
                    "title": "Top Error Endpoints",
                    "type": "table",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 22},
                    "targets": [
                        {
                            "expr": "sum(rate(trading_bot_api_errors_total[1h])) by (endpoint)",
                            "legendFormat": "{{endpoint}}"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "mappings": [],
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "red", "value": 10}
                                ]
                            }
                        }
                    }
                }
            ],
            "time": {"from": "now-1h", "to": "now"},
            "timepicker": {},
            "templating": {
                "list": []
            },
            "annotations": {
                "list": [
                    {
                        "name": "API Errors",
                        "datasource": "prometheus",
                        "expr": "rate(trading_bot_api_errors_total[5m]) > 0",
                        "titleFormat": "API Error: {{endpoint}}",
                        "textFormat": "{{error_type}}"
                    }
                ]
            },
            "refresh": "30s",
            "schemaVersion": 27,
            "version": 0,
            "links": []
        }
    }


def generate_all_dashboards() -> Dict[str, Dict[str, Any]]:
    """Generate all dashboard configurations."""
    return {
        "trading_performance": create_trading_performance_dashboard(),
        "system_health": create_system_health_dashboard(),
        "risk_management": create_risk_management_dashboard(),
        "strategy_analytics": create_strategy_analytics_dashboard(),
        "api_performance": create_api_performance_dashboard()
    }


def save_dashboards_to_files(output_dir: str = "grafana_dashboards"):
    """Save all dashboard configurations to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dashboards = generate_all_dashboards()

    for name, config in dashboards.items():
        filename = output_path / f"{name}_dashboard.json"
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Saved {name} dashboard to {filename}")


def create_dashboard_import_script() -> str:
    """Create a script to import dashboards into Grafana via API."""
    return '''
#!/bin/bash
# Import Grafana dashboards via API

GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
GRAFANA_API_KEY="${GRAFANA_API_KEY:-}"

if [ -z "$GRAFANA_API_KEY" ]; then
    echo "Error: GRAFANA_API_KEY environment variable not set"
    exit 1
fi

DASHBOARD_DIR="grafana_dashboards"

for dashboard_file in "$DASHBOARD_DIR"/*_dashboard.json; do
    if [ -f "$dashboard_file" ]; then
        echo "Importing $(basename "$dashboard_file")..."

        # Import dashboard
        curl -X POST "$GRAFANA_URL/api/dashboards/db" \\
            -H "Authorization: Bearer $GRAFANA_API_KEY" \\
            -H "Content-Type: application/json" \\
            -d @"$dashboard_file"

        echo ""
    fi
done

echo "Dashboard import complete!"
'''


if __name__ == "__main__":
    # Generate and save all dashboards
    save_dashboards_to_files()

    # Create import script
    script_content = create_dashboard_import_script()
    with open("import_dashboards.sh", "w") as f:
        f.write(script_content)

    print("Grafana dashboards generated and saved to 'grafana_dashboards/' directory")
    print("Run './import_dashboards.sh' to import dashboards into Grafana")
