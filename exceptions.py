"""
Custom Exceptions for Trading Bot

Standardized exception hierarchy for consistent error handling across the codebase.
"""

from typing import Optional, Dict


class TradingBotError(Exception):
    """Base exception for all trading bot errors"""
    pass


class ConfigurationError(TradingBotError):
    """Raised when configuration is invalid or missing"""
    pass


class ExchangeError(TradingBotError):
    """Base exception for exchange-related errors"""
    pass


class ExchangeAPIError(ExchangeError):
    """Raised when exchange API call fails"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 response: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class ExchangeAuthenticationError(ExchangeError):
    """Raised when exchange authentication fails"""
    pass


class ExchangeRateLimitError(ExchangeError):
    """Raised when exchange rate limit is exceeded"""
    pass


class ExchangeNetworkError(ExchangeError):
    """Raised when network/connection errors occur with exchange"""
    pass


class ExchangeTimeoutError(ExchangeError):
    """Raised when exchange API request times out"""
    pass


class ExchangeInvalidResponseError(ExchangeError):
    """Raised when exchange returns invalid/unexpected response format"""
    
    def __init__(self, message: str, response: Optional[Dict] = None):
        super().__init__(message)
        self.response = response


class MarketDataError(TradingBotError):
    """Raised when market data is invalid or unavailable"""
    pass


class MarketDataUnavailableError(MarketDataError):
    """Raised when market data cannot be fetched"""
    pass


class MarketDataValidationError(MarketDataError):
    """Raised when market data fails validation"""
    pass


class StrategyError(TradingBotError):
    """Raised when strategy calculation or validation fails"""
    pass


class IndicatorCalculationError(StrategyError):
    """Raised when indicator calculation produces invalid results"""
    pass


class EntrySignalError(StrategyError):
    """Raised when entry signal validation fails"""
    pass


class RiskManagementError(TradingBotError):
    """Raised when risk management checks fail"""
    pass


class PositionSizeError(RiskManagementError):
    """Raised when position size calculation or validation fails"""
    pass


class DailyLimitError(RiskManagementError):
    """Raised when daily trading limits are exceeded"""
    pass


class OrderError(TradingBotError):
    """Raised when order placement or management fails"""
    pass


class OrderValidationError(OrderError):
    """Raised when order validation fails before placement"""
    pass


class OrderExecutionError(OrderError):
    """Raised when order execution fails"""
    
    def __init__(self, message: str, order_id: Optional[str] = None,
                 exchange_response: Optional[Dict] = None):
        super().__init__(message)
        self.order_id = order_id
        self.exchange_response = exchange_response


class PositionError(TradingBotError):
    """Raised when position management fails"""
    pass


class PositionSyncError(PositionError):
    """Raised when position synchronization with exchange fails"""
    pass



