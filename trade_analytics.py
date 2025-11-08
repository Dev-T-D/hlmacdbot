"""
Trade Analytics and Reporting for Trading Bot

Comprehensive post-trade analysis with performance metrics,
risk analysis, and automated report generation.

Features:
- Trade performance analysis (win rate, profit factor, R-multiple)
- Risk metrics (drawdown, recovery time, Sharpe ratio)
- Time-based analysis (hourly, daily, weekly patterns)
- Strategy effectiveness analysis
- PDF/HTML report generation
- Performance benchmarking
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import io
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Individual trade record."""
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percentage: float
    commission: float = 0.0
    strategy: str = "unknown"
    risk_reward_ratio: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    duration_minutes: Optional[float] = None
    tags: Dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.duration_minutes is None and self.entry_time and self.exit_time:
            self.duration_minutes = (self.exit_time - self.entry_time).total_seconds() / 60


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    total_pnl_percentage: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    average_trade_duration: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: float = 0.0
    max_drawdown_duration: Optional[timedelta] = None
    recovery_time: Optional[timedelta] = None
    expectancy: float = 0.0
    r_multiple_distribution: Dict[str, int] = None

    def __post_init__(self):
        if self.r_multiple_distribution is None:
            self.r_multiple_distribution = {}


@dataclass
class RiskMetrics:
    """Risk analysis metrics."""
    value_at_risk_95: float
    value_at_risk_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    volatility: float
    beta: Optional[float] = None
    correlation_coefficient: Optional[float] = None
    stress_test_results: Dict[str, float] = None

    def __post_init__(self):
        if self.stress_test_results is None:
            self.stress_test_results = {}


@dataclass
class TimeAnalysis:
    """Time-based performance analysis."""
    hourly_performance: Dict[int, Dict[str, float]]
    daily_performance: Dict[str, Dict[str, float]]
    weekly_performance: Dict[int, Dict[str, float]]
    monthly_performance: Dict[int, Dict[str, float]]
    best_hour: int
    worst_hour: int
    best_day: str
    worst_day: str
    weekend_performance: Dict[str, float]


@dataclass
class StrategyAnalysis:
    """Strategy effectiveness analysis."""
    strategy_performance: Dict[str, Dict[str, Any]]
    best_strategy: str
    worst_strategy: str
    strategy_consistency: Dict[str, float]
    strategy_risk_adjusted: Dict[str, float]


@dataclass
class TradeAnalyticsReport:
    """Complete analytics report."""
    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    performance: PerformanceMetrics
    risk: RiskMetrics
    time_analysis: TimeAnalysis
    strategy_analysis: StrategyAnalysis
    trade_log: List[TradeRecord]
    summary_stats: Dict[str, Any]
    recommendations: List[str]


class TradeAnalytics:
    """
    Comprehensive trade analytics and reporting system.

    Analyzes trading performance, generates reports, and provides insights.
    """

    def __init__(self, trade_data_file: str = "data/trade_history.json"):
        self.trade_data_file = Path(trade_data_file)
        self.trade_data_file.parent.mkdir(parents=True, exist_ok=True)
        self.trades: List[TradeRecord] = []
        self._load_trade_data()

    def _load_trade_data(self):
        """Load trade data from storage."""
        try:
            if self.trade_data_file.exists():
                with open(self.trade_data_file, 'r') as f:
                    data = json.load(f)

                self.trades = []
                for trade_data in data.get('trades', []):
                    # Convert timestamps
                    trade_data['entry_time'] = datetime.fromisoformat(trade_data['entry_time'])
                    trade_data['exit_time'] = datetime.fromisoformat(trade_data['exit_time'])
                    self.trades.append(TradeRecord(**trade_data))

                logger.info(f"Loaded {len(self.trades)} trades from {self.trade_data_file}")

        except Exception as e:
            logger.error(f"Failed to load trade data: {e}")

    def _save_trade_data(self):
        """Save trade data to storage."""
        try:
            data = {
                'trades': [asdict(trade) for trade in self.trades],
                'last_updated': datetime.now(timezone.utc).isoformat()
            }

            with open(self.trade_data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save trade data: {e}")

    def add_trade(self, trade: TradeRecord):
        """Add a new trade to the analytics."""
        self.trades.append(trade)
        self._save_trade_data()
        logger.info(f"Added trade {trade.trade_id} to analytics")

    def add_trades(self, trades: List[TradeRecord]):
        """Add multiple trades to the analytics."""
        self.trades.extend(trades)
        self._save_trade_data()
        logger.info(f"Added {len(trades)} trades to analytics")

    def get_trades_in_period(self, start_date: datetime, end_date: datetime) -> List[TradeRecord]:
        """Get trades within a specific time period."""
        return [
            trade for trade in self.trades
            if start_date <= trade.entry_time <= end_date
        ]

    def calculate_performance_metrics(self, trades: List[TradeRecord]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        if not trades:
            return PerformanceMetrics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                total_pnl=0.0,
                total_pnl_percentage=0.0,
                average_win=0.0,
                average_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                average_trade_duration=0.0
            )

        # Basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl < 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # P&L calculations
        total_pnl = sum(t.pnl for t in trades)
        winning_pnls = [t.pnl for t in trades if t.pnl > 0]
        losing_pnls = [t.pnl for t in trades if t.pnl < 0]

        average_win = np.mean(winning_pnls) if winning_pnls else 0
        average_loss = np.mean(losing_pnls) if losing_pnls else 0
        largest_win = max(winning_pnls) if winning_pnls else 0
        largest_loss = min(losing_pnls) if losing_pnls else 0

        # Profit factor
        total_wins = sum(winning_pnls)
        total_losses = abs(sum(losing_pnls))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Duration analysis
        durations = [t.duration_minutes for t in trades if t.duration_minutes]
        average_trade_duration = np.mean(durations) if durations else 0

        # R-multiple distribution (simplified)
        r_multiples = []
        for trade in trades:
            if trade.stop_loss and trade.entry_price:
                risk = abs(trade.entry_price - trade.stop_loss)
                if risk > 0:
                    r_multiple = trade.pnl / risk
                    r_multiples.append(r_multiple)

        r_distribution = {}
        for r in r_multiples:
            if r >= 2:
                r_distribution['2R+'] = r_distribution.get('2R+', 0) + 1
            elif r >= 1:
                r_distribution['1-2R'] = r_distribution.get('1-2R', 0) + 1
            elif r >= 0:
                r_distribution['0-1R'] = r_distribution.get('0-1R', 0) + 1
            elif r >= -1:
                r_distribution['-1-0R'] = r_distribution.get('-1-0R', 0) + 1
            else:
                r_distribution['-1R-'] = r_distribution.get('-1R-', 0) + 1

        # Drawdown analysis
        cumulative_pnl = np.cumsum([t.pnl for t in trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = running_max - cumulative_pnl
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Sharpe ratio (simplified, assuming daily returns)
        if len(trades) > 1:
            daily_pnls = self._group_trades_by_day(trades)
            returns = [pnl for pnl in daily_pnls.values()]
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        # Expectancy
        avg_win_amount = average_win * win_rate
        avg_loss_amount = abs(average_loss) * (1 - win_rate)
        expectancy = avg_win_amount - avg_loss_amount

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_pnl=total_pnl,
            total_pnl_percentage=0.0,  # Would need account balance context
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            average_trade_duration=average_trade_duration,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            expectancy=expectancy,
            r_multiple_distribution=r_distribution
        )

    def calculate_risk_metrics(self, trades: List[TradeRecord]) -> RiskMetrics:
        """Calculate risk metrics including VaR and stress testing."""
        if not trades:
            return RiskMetrics(
                value_at_risk_95=0.0,
                value_at_risk_99=0.0,
                expected_shortfall_95=0.0,
                expected_shortfall_99=0.0,
                volatility=0.0
            )

        pnls = [t.pnl for t in trades]

        # Value at Risk (historical simulation)
        sorted_pnls = sorted(pnls)
        var_95 = np.percentile(sorted_pnls, 5)  # 5th percentile (95% confidence)
        var_99 = np.percentile(sorted_pnls, 1)  # 1st percentile (99% confidence)

        # Expected Shortfall (Conditional VaR)
        es_95 = np.mean(sorted_pnls[:int(len(sorted_pnls) * 0.05)])
        es_99 = np.mean(sorted_pnls[:int(len(sorted_pnls) * 0.01)])

        # Volatility
        volatility = np.std(pnls)

        # Stress test scenarios
        stress_results = {}
        scenarios = {
            'market_crash': -0.20,  # 20% drop
            'high_volatility': 2.0,  # 2x volatility
            'liquidity_crisis': -0.15  # 15% drop
        }

        for scenario, impact in scenarios.items():
            if scenario == 'high_volatility':
                stressed_pnls = [pnl * impact for pnl in pnls]
            else:
                stressed_pnls = [pnl * (1 + impact) for pnl in pnls]
            stress_results[scenario] = np.sum(stressed_pnls)

        return RiskMetrics(
            value_at_risk_95=abs(var_95),  # Return positive value
            value_at_risk_99=abs(var_99),
            expected_shortfall_95=abs(es_95),
            expected_shortfall_99=abs(es_99),
            volatility=volatility,
            stress_test_results=stress_results
        )

    def analyze_time_patterns(self, trades: List[TradeRecord]) -> TimeAnalysis:
        """Analyze performance patterns by time."""
        if not trades:
            return TimeAnalysis(
                hourly_performance={},
                daily_performance={},
                weekly_performance={},
                monthly_performance={},
                best_hour=0,
                worst_hour=0,
                best_day="Monday",
                worst_day="Monday",
                weekend_performance={}
            )

        # Hourly analysis
        hourly_performance = {}
        for hour in range(24):
            hour_trades = [t for t in trades if t.entry_time.hour == hour]
            if hour_trades:
                pnl = sum(t.pnl for t in hour_trades)
                win_rate = len([t for t in hour_trades if t.pnl > 0]) / len(hour_trades)
                hourly_performance[hour] = {
                    'trades': len(hour_trades),
                    'pnl': pnl,
                    'win_rate': win_rate
                }

        # Daily analysis
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_performance = {}
        for i, day in enumerate(day_names):
            day_trades = [t for t in trades if t.entry_time.weekday() == i]
            if day_trades:
                pnl = sum(t.pnl for t in day_trades)
                win_rate = len([t for t in day_trades if t.pnl > 0]) / len(day_trades)
                daily_performance[day] = {
                    'trades': len(day_trades),
                    'pnl': pnl,
                    'win_rate': win_rate
                }

        # Weekly analysis
        weekly_performance = {}
        for week in range(1, 53):
            week_trades = [t for t in trades if t.entry_time.isocalendar()[1] == week]
            if week_trades:
                pnl = sum(t.pnl for t in week_trades)
                win_rate = len([t for t in week_trades if t.pnl > 0]) / len(week_trades)
                weekly_performance[week] = {
                    'trades': len(week_trades),
                    'pnl': pnl,
                    'win_rate': win_rate
                }

        # Monthly analysis
        monthly_performance = {}
        for month in range(1, 13):
            month_trades = [t for t in trades if t.entry_time.month == month]
            if month_trades:
                pnl = sum(t.pnl for t in month_trades)
                win_rate = len([t for t in month_trades if t.pnl > 0]) / len(month_trades)
                monthly_performance[month] = {
                    'trades': len(month_trades),
                    'pnl': pnl,
                    'win_rate': win_rate
                }

        # Find best/worst periods
        best_hour = max(hourly_performance.keys(), key=lambda h: hourly_performance[h]['pnl']) if hourly_performance else 0
        worst_hour = min(hourly_performance.keys(), key=lambda h: hourly_performance[h]['pnl']) if hourly_performance else 0

        best_day = max(daily_performance.keys(), key=lambda d: daily_performance[d]['pnl']) if daily_performance else "Monday"
        worst_day = min(daily_performance.keys(), key=lambda d: daily_performance[d]['pnl']) if daily_performance else "Monday"

        # Weekend performance
        weekend_trades = [t for t in trades if t.entry_time.weekday() >= 5]
        weekend_performance = {
            'trades': len(weekend_trades),
            'pnl': sum(t.pnl for t in weekend_trades),
            'win_rate': len([t for t in weekend_trades if t.pnl > 0]) / len(weekend_trades) if weekend_trades else 0
        }

        return TimeAnalysis(
            hourly_performance=hourly_performance,
            daily_performance=daily_performance,
            weekly_performance=weekly_performance,
            monthly_performance=monthly_performance,
            best_hour=best_hour,
            worst_hour=worst_hour,
            best_day=best_day,
            worst_day=worst_day,
            weekend_performance=weekend_performance
        )

    def analyze_strategy_effectiveness(self, trades: List[TradeRecord]) -> StrategyAnalysis:
        """Analyze strategy performance."""
        if not trades:
            return StrategyAnalysis(
                strategy_performance={},
                best_strategy="unknown",
                worst_strategy="unknown",
                strategy_consistency={},
                strategy_risk_adjusted={}
            )

        # Group trades by strategy
        strategies = {}
        for trade in trades:
            strategy = trade.strategy
            if strategy not in strategies:
                strategies[strategy] = []
            strategies[strategy].append(trade)

        strategy_performance = {}
        strategy_consistency = {}
        strategy_risk_adjusted = {}

        for strategy, strategy_trades in strategies.items():
            performance = self.calculate_performance_metrics(strategy_trades)

            strategy_performance[strategy] = {
                'total_trades': performance.total_trades,
                'win_rate': performance.win_rate,
                'profit_factor': performance.profit_factor,
                'total_pnl': performance.total_pnl,
                'average_win': performance.average_win,
                'average_loss': performance.average_loss
            }

            # Consistency score (based on win rate stability)
            win_rates = []
            # Group by weeks for consistency analysis
            weekly_groups = {}
            for trade in strategy_trades:
                week = trade.entry_time.isocalendar()[1]
                if week not in weekly_groups:
                    weekly_groups[week] = []
                weekly_groups[week].append(trade)

            for week_trades in weekly_groups.values():
                if week_trades:
                    weekly_win_rate = len([t for t in week_trades if t.pnl > 0]) / len(week_trades)
                    win_rates.append(weekly_win_rate)

            consistency = 1 - np.std(win_rates) if win_rates else 0
            strategy_consistency[strategy] = consistency

            # Risk-adjusted return (PnL / max drawdown)
            risk_adjusted = performance.total_pnl / performance.max_drawdown if performance.max_drawdown > 0 else 0
            strategy_risk_adjusted[strategy] = risk_adjusted

        # Find best/worst strategies
        if strategy_performance:
            best_strategy = max(strategy_performance.keys(), key=lambda s: strategy_performance[s]['total_pnl'])
            worst_strategy = min(strategy_performance.keys(), key=lambda s: strategy_performance[s]['total_pnl'])
        else:
            best_strategy = worst_strategy = "unknown"

        return StrategyAnalysis(
            strategy_performance=strategy_performance,
            best_strategy=best_strategy,
            worst_strategy=worst_strategy,
            strategy_consistency=strategy_consistency,
            strategy_risk_adjusted=strategy_risk_adjusted
        )

    def generate_report(
        self,
        period_days: int = 30,
        report_format: str = "dict"
    ) -> Union[TradeAnalyticsReport, Dict[str, Any]]:
        """
        Generate comprehensive analytics report.

        Args:
            period_days: Number of days to analyze
            report_format: "dict" or "object"

        Returns:
            Complete analytics report
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=period_days)

        # Get trades in period
        period_trades = self.get_trades_in_period(start_date, end_date)

        # Calculate all metrics
        performance = self.calculate_performance_metrics(period_trades)
        risk = self.calculate_risk_metrics(period_trades)
        time_analysis = self.analyze_time_patterns(period_trades)
        strategy_analysis = self.analyze_strategy_effectiveness(period_trades)

        # Summary statistics
        summary_stats = {
            'total_return': performance.total_pnl,
            'annualized_return': performance.total_pnl * (365 / period_days) if period_days > 0 else 0,
            'volatility': risk.volatility,
            'sharpe_ratio': performance.sharpe_ratio or 0,
            'max_drawdown': performance.max_drawdown,
            'win_rate': performance.win_rate,
            'profit_factor': performance.profit_factor,
            'average_trade': performance.total_pnl / performance.total_trades if performance.total_trades > 0 else 0,
            'expectancy': performance.expectancy
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(performance, risk, time_analysis, strategy_analysis)

        report = TradeAnalyticsReport(
            report_id=f"analytics_{int(end_date.timestamp())}",
            generated_at=end_date,
            period_start=start_date,
            period_end=end_date,
            performance=performance,
            risk=risk,
            time_analysis=time_analysis,
            strategy_analysis=strategy_analysis,
            trade_log=period_trades,
            summary_stats=summary_stats,
            recommendations=recommendations
        )

        if report_format == "dict":
            return asdict(report)
        else:
            return report

    def _generate_recommendations(
        self,
        performance: PerformanceMetrics,
        risk: RiskMetrics,
        time_analysis: TimeAnalysis,
        strategy_analysis: StrategyAnalysis
    ) -> List[str]:
        """Generate trading recommendations based on analysis."""
        recommendations = []

        # Performance recommendations
        if performance.win_rate < 0.4:
            recommendations.append("Win rate is below 40%. Consider reviewing entry criteria and strategy parameters.")

        if performance.profit_factor < 1.5:
            recommendations.append("Profit factor is below 1.5. Focus on improving winners vs losers ratio.")

        if performance.max_drawdown > 0.20:  # 20%
            recommendations.append("Maximum drawdown exceeds 20%. Consider implementing stricter risk management.")

        # Risk recommendations
        if risk.volatility > risk.value_at_risk_95:
            recommendations.append("Portfolio volatility is high relative to VaR. Consider position sizing adjustments.")

        # Time-based recommendations
        if time_analysis.best_hour != time_analysis.worst_hour:
            recommendations.append(f"Best trading hour is {time_analysis.best_hour}:00. Consider focusing activity during this time.")

        if time_analysis.weekend_performance['pnl'] < 0:
            recommendations.append("Weekend trading is showing losses. Consider avoiding weekend positions.")

        # Strategy recommendations
        if strategy_analysis.best_strategy != strategy_analysis.worst_strategy:
            recommendations.append(f"Strategy '{strategy_analysis.best_strategy}' is performing best. Consider allocating more capital to it.")

        consistent_strategies = [s for s, c in strategy_analysis.strategy_consistency.items() if c > 0.7]
        if consistent_strategies:
            recommendations.append(f"Strategies {consistent_strategies} show high consistency. Prioritize these strategies.")

        return recommendations

    def export_report_pdf(self, report: TradeAnalyticsReport, filename: str) -> bool:
        """Export analytics report as PDF."""
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab not available. Cannot generate PDF reports.")
            return False

        try:
            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30,
            )
            story.append(Paragraph("Trading Performance Analytics Report", title_style))
            story.append(Spacer(1, 12))

            # Report metadata
            story.append(Paragraph(f"Report Period: {report.period_start.date()} to {report.period_end.date()}", styles['Normal']))
            story.append(Paragraph(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}", styles['Normal']))
            story.append(Spacer(1, 20))

            # Performance Summary
            story.append(Paragraph("Performance Summary", styles['Heading2']))
            perf_data = [
                ["Metric", "Value"],
                ["Total Trades", str(report.performance.total_trades)],
                ["Win Rate", ".1%"],
                ["Profit Factor", ".2f"],
                ["Total P&L", ".2f"],
                ["Max Drawdown", ".1%"],
                ["Sharpe Ratio", ".2f" if report.performance.sharpe_ratio else "N/A"]
            ]

            perf_table = Table(perf_data)
            perf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(perf_table)
            story.append(Spacer(1, 20))

            # Risk Metrics
            story.append(Paragraph("Risk Analysis", styles['Heading2']))
            risk_data = [
                ["Risk Metric", "Value"],
                ["Value at Risk (95%)", ".2f"],
                ["Value at Risk (99%)", ".2f"],
                ["Expected Shortfall (95%)", ".2f"],
                ["Volatility", ".2f"]
            ]

            risk_table = Table(risk_data)
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(risk_table)
            story.append(Spacer(1, 20))

            # Recommendations
            if report.recommendations:
                story.append(Paragraph("Recommendations", styles['Heading2']))
                for rec in report.recommendations:
                    story.append(Paragraph(f"• {rec}", styles['Normal']))
                    story.append(Spacer(1, 6))
                story.append(Spacer(1, 20))

            # Build PDF
            doc.build(story)
            logger.info(f"PDF report generated: {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            return False

    def export_report_html(self, report: TradeAnalyticsReport, filename: str) -> bool:
        """Export analytics report as HTML."""
        try:
            html_content = ".2f"".2f"".1%"".2f"".1%"".2f"f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trading Performance Analytics Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                    .metric-card {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                    .metric-label {{ color: #7f8c8d; font-size: 14px; }}
                    .positive {{ color: #27ae60; }}
                    .negative {{ color: #e74c3c; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .recommendations {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Trading Performance Analytics Report</h1>
                    <p>Report Period: {report.period_start.date()} to {report.period_end.date()}</p>
                    <p>Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                </div>

                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value {'positive' if report.performance.total_pnl >= 0 else 'negative'}">${report.performance.total_pnl:.2f}</div>
                        <div class="metric-label">Total P&L</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{report.performance.win_rate:.1%}</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{report.performance.profit_factor:.2f}</div>
                        <div class="metric-label">Profit Factor</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value {'negative' if report.performance.max_drawdown > 0.20 else 'positive'}">{report.performance.max_drawdown:.1%}</div>
                        <div class="metric-label">Max Drawdown</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{report.performance.total_trades}</div>
                        <div class="metric-label">Total Trades</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{report.summary_stats.get('sharpe_ratio', 0):.2f}</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                </div>

                <h2>Risk Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Value at Risk (95%)</td><td>${report.risk.value_at_risk_95:.2f}</td></tr>
                    <tr><td>Value at Risk (99%)</td><td>${report.risk.value_at_risk_99:.2f}</td></tr>
                    <tr><td>Expected Shortfall (95%)</td><td>${report.risk.expected_shortfall_95:.2f}</td></tr>
                    <tr><td>Volatility</td><td>${report.risk.volatility:.2f}</td></tr>
                </table>

                <h2>Strategy Analysis</h2>
                <table>
                    <tr><th>Strategy</th><th>Trades</th><th>Win Rate</th><th>P&L</th><th>Profit Factor</th></tr>
            """

            for strategy, metrics in report.strategy_analysis.strategy_performance.items():
                html_content += f"""
                    <tr>
                        <td>{strategy}</td>
                        <td>{metrics['total_trades']}</td>
                        <td>{metrics['win_rate']:.1%}</td>
                        <td>${metrics['total_pnl']:.2f}</td>
                        <td>{metrics['profit_factor']:.2f}</td>
                    </tr>
                """

            html_content += """
                </table>

                <h2>Time Analysis</h2>
                <table>
                    <tr><th>Period</th><th>Best</th><th>Worst</th></tr>
                    <tr><td>Hour</td><td>{report.time_analysis.best_hour}:00</td><td>{report.time_analysis.worst_hour}:00</td></tr>
                    <tr><td>Day</td><td>{report.time_analysis.best_day}</td><td>{report.time_analysis.worst_day}</td></tr>
                </table>
            """

            if report.recommendations:
                html_content += """
                <div class="recommendations">
                    <h2>Recommendations</h2>
                    <ul>
                """
                for rec in report.recommendations:
                    html_content += f"<li>{rec}</li>"
                html_content += "</ul></div>"

            html_content += """
            </body>
            </html>
            """

            with open(filename, 'w') as f:
                f.write(html_content)

            logger.info(f"HTML report generated: {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return False

    def _group_trades_by_day(self, trades: List[TradeRecord]) -> Dict[str, float]:
        """Group trades by day and sum P&L."""
        daily_pnl = {}
        for trade in trades:
            day_key = trade.entry_time.date().isoformat()
            daily_pnl[day_key] = daily_pnl.get(day_key, 0) + trade.pnl
        return daily_pnl

    # Convenience methods for common reports
    def get_daily_report(self) -> Dict[str, Any]:
        """Generate daily performance report."""
        return self.generate_report(period_days=1)

    def get_weekly_report(self) -> Dict[str, Any]:
        """Generate weekly performance report."""
        return self.generate_report(period_days=7)

    def get_monthly_report(self) -> Dict[str, Any]:
        """Generate monthly performance report."""
        return self.generate_report(period_days=30)

    def get_quarterly_report(self) -> Dict[str, Any]:
        """Generate quarterly performance report."""
        return self.generate_report(period_days=90)
