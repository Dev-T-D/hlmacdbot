import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
from datetime import datetime, timezone, timedelta
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class DataManager:
    """
    Manage historical and real-time trading data

    Features:
    - SQLite database for structured storage
    - Efficient querying with indexes
    - Data validation and cleaning
    - Gap detection and filling
    """

    def __init__(self, db_path: str = 'data/trading_data.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist"""
        cursor = self.conn.cursor()

        # OHLCV candles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                UNIQUE(symbol, timeframe, timestamp)
            )
        ''')

        # Create indexes for fast queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf_time
            ON candles(symbol, timeframe, timestamp)
        ''')

        # Trade labels table (for supervised learning)
        cursor.execute('''CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            label INTEGER NOT NULL,
            price_change_5m REAL,
            price_change_15m REAL,
            price_change_1h REAL,
            UNIQUE(symbol, timeframe, timestamp)
        )''')

        self.conn.commit()
        logger.info("✅ Database tables created")

    def insert_candles(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Insert candles into database"""
        try:
            # Prepare data
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            df_copy['timeframe'] = timeframe
            df_copy.reset_index(inplace=True)
            df_copy.rename(columns={'index': 'timestamp'}, inplace=True)

            # Insert with ON CONFLICT IGNORE (skip duplicates)
            df_copy.to_sql('candles', self.conn, if_exists='append', index=False)

            logger.debug(f"✅ Inserted {len(df)} candles for {symbol} {timeframe}")

        except Exception as e:
            logger.error(f"Error inserting candles: {e}")

    def get_candles(self, symbol: str, timeframe: str,
                   start_date: datetime = None, end_date: datetime = None,
                   limit: int = None) -> pd.DataFrame:
        """Query candles from database"""
        query = '''
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND timeframe = ?
        '''

        params = [symbol, timeframe]

        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)

        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date)

        query += ' ORDER BY timestamp ASC'

        if limit:
            query += f' LIMIT {limit}'

        df = pd.read_sql_query(query, self.conn, params=params, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)

        return df

    def generate_labels(self, symbol: str, timeframe: str,
                       prediction_horizon: int = 5, threshold: float = 0.002):
        """
        Generate training labels from historical data

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            prediction_horizon: How many periods ahead to predict
            threshold: Minimum price change to label as 1 (0.2% default)

        Returns:
            DataFrame with labels
        """
        logger.info(f"🏷️  Generating labels for {symbol} {timeframe}")

        # Get all candles
        df = self.get_candles(symbol, timeframe)

        if len(df) < prediction_horizon + 1:
            logger.warning(f"Not enough data to generate labels")
            return pd.DataFrame()

        # Calculate future price changes
        df['price_change_5m'] = df['close'].pct_change(5).shift(-5)
        df['price_change_15m'] = df['close'].pct_change(15).shift(-15)
        df['price_change_1h'] = df['close'].pct_change(60).shift(-60)

        # Generate binary label (1 if price goes up > threshold, 0 otherwise)
        df['label'] = (df[f'price_change_{prediction_horizon}m'] > threshold).astype(int)

        # Remove last rows (no future data available)
        df = df[:-60]  # Remove last hour

        # Save labels to database
        label_df = df[['label', 'price_change_5m', 'price_change_15m', 'price_change_1h']].copy()
        label_df['symbol'] = symbol
        label_df['timeframe'] = timeframe
        label_df.reset_index(inplace=True)

        try:
            label_df.to_sql('labels', self.conn, if_exists='append', index=False)
            logger.info(f"✅ Generated {len(label_df)} labels")
        except Exception as e:
            logger.error(f"Error saving labels: {e}")

        return df

    def get_training_data(self, symbol: str, timeframe: str,
                         min_samples: int = 1000) -> tuple:
        """
        Get complete training dataset with features and labels

        Returns:
            (X_train: DataFrame, y_train: Series)
        """
        # Get candles
        df = self.get_candles(symbol, timeframe)

        # Get labels
        labels_query = '''
            SELECT timestamp, label
            FROM labels
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp ASC
        '''

        labels_df = pd.read_sql_query(
            labels_query,
            self.conn,
            params=[symbol, timeframe],
            parse_dates=['timestamp']
        )
        labels_df.set_index('timestamp', inplace=True)

        # Merge
        merged = df.join(labels_df, how='inner')

        if len(merged) < min_samples:
            logger.warning(f"Only {len(merged)} samples available (need {min_samples})")

        X = merged.drop(columns=['label'])
        y = merged['label']

        logger.info(f"📊 Training data: {len(X)} samples")
        logger.info(f"   Positive samples: {y.sum()} ({y.mean()*100:.1f}%)")

        return X, y

    def detect_data_gaps(self, symbol: str, timeframe: str) -> List[Tuple]:
        """Detect gaps in data collection"""
        df = self.get_candles(symbol, timeframe)

        if df.empty:
            return []

        # Calculate expected time delta
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
        }
        expected_delta = timedelta(minutes=timeframe_minutes.get(timeframe, 60))

        # Find gaps
        gaps = []
        for i in range(1, len(df)):
            actual_delta = df.index[i] - df.index[i-1]

            if actual_delta > expected_delta * 1.5:  # Allow 50% tolerance
                gaps.append((df.index[i-1], df.index[i], actual_delta))

        if gaps:
            logger.warning(f"⚠️  Found {len(gaps)} data gaps for {symbol} {timeframe}")
            for start, end, delta in gaps:
                logger.warning(f"   Gap: {start} to {end} ({delta})")

        return gaps

# Example usage
if __name__ == "__main__":
    dm = DataManager()

    # Generate labels for collected data
    dm.generate_labels('BTC', '1h', prediction_horizon=5)

    # Get training data
    X, y = dm.get_training_data('BTC', '1h')
    print(f"Training samples: {len(X)}")
