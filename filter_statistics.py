"""
Filter Statistics Tracker

Track which filters are most effective at preventing losses and improving win rates.
"""

import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class FilterStatistics:
    """Track which filters are most effective"""

    def __init__(self):
        self.filter_results = defaultdict(list)

    def record_filter_result(self, filter_name, passed, signal_direction):
        """Record filter result"""
        self.filter_results[filter_name].append({
            'passed': passed,
            'direction': signal_direction,
            'timestamp': pd.Timestamp.now()
        })

    def get_filter_efficiency(self):
        """Calculate filter efficiency (how often it rejects bad signals)"""
        stats = {}

        for filter_name, results in self.filter_results.items():
            df = pd.DataFrame(results)

            total = len(df)
            passed = df['passed'].sum()
            rejected = total - passed

            stats[filter_name] = {
                'total_signals': total,
                'passed': passed,
                'rejected': rejected,
                'rejection_rate': rejected / total if total > 0 else 0
            }

        return stats

    def analyze_filter_cascade(self):
        """Show how signals flow through filter cascade"""
        print("\n📊 Signal Filter Cascade Analysis:")
        print("=" * 60)

        # Assuming filters are in order
        filter_order = [
            'ml_confidence',
            'mtf_alignment',
            'volume_quality',
            'market_regime'
        ]

        remaining = 100  # Start with 100% of signals

        for i, filter_name in enumerate(filter_order):
            stats = self.get_filter_efficiency().get(filter_name, {})
            rejection_rate = stats.get('rejection_rate', 0)

            filtered_out = remaining * rejection_rate
            remaining -= filtered_out

            print(f"Filter {i+1} ({filter_name}):")
            print(".1f")
            print(".1f")
            print()

        print(".1f")
        print("=" * 60)

    def get_filter_performance_report(self):
        """Generate comprehensive filter performance report"""
        report = {
            'filter_efficiency': self.get_filter_efficiency(),
            'total_signals_processed': sum(len(results) for results in self.filter_results.values()),
            'filter_cascade_analysis': self._analyze_cascade_flow()
        }

        return report

    def _analyze_cascade_flow(self):
        """Analyze how signals flow through the cascade"""
        # This would require tracking signals through the entire cascade
        # For now, return basic stats
        return {
            'filters_active': list(self.filter_results.keys()),
            'signals_per_filter': {k: len(v) for k, v in self.filter_results.items()}
        }
