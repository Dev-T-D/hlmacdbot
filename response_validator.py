"""
API Response Validation Module

Validates API responses from exchanges to ensure they match expected schemas
and handle unexpected formats gracefully.
"""

from typing import Dict, List, Optional, Any
import logging

from exceptions import ExchangeInvalidResponseError

logger = logging.getLogger(__name__)


def validate_dict_response(response: Any, required_fields: List[str], 
                          response_name: str = "API response") -> Dict:
    """
    Validate that response is a dict with required fields.
    
    Args:
        response: Response to validate
        required_fields: List of required field names
        response_name: Name of response for error messages
        
    Returns:
        Validated dict response
        
    Raises:
        ExchangeInvalidResponseError: If response is invalid
    """
    if not isinstance(response, dict):
        raise ExchangeInvalidResponseError(
            f"Invalid {response_name}: expected dict, got {type(response).__name__}",
            response=response
        )
    
    missing_fields = [field for field in required_fields if field not in response]
    if missing_fields:
        available_fields = list(response.keys())
        raise ExchangeInvalidResponseError(
            f"Invalid {response_name}: missing required fields {missing_fields}. "
            f"Available fields: {available_fields}",
            response=response
        )
    
    return response


def validate_list_response(response: Any, min_items: int = 0,
                          response_name: str = "API response") -> List:
    """
    Validate that response is a list with minimum items.
    
    Args:
        response: Response to validate
        min_items: Minimum number of items required
        response_name: Name of response for error messages
        
    Returns:
        Validated list response
        
    Raises:
        ExchangeInvalidResponseError: If response is invalid
    """
    if not isinstance(response, list):
        raise ExchangeInvalidResponseError(
            f"Invalid {response_name}: expected list, got {type(response).__name__}",
            response=response
        )
    
    if len(response) < min_items:
        raise ExchangeInvalidResponseError(
            f"Invalid {response_name}: expected at least {min_items} items, got {len(response)}",
            response=response
        )
    
    return response


def validate_nested_dict(response: Dict, path: str, required_fields: List[str],
                        response_name: str = "API response") -> Dict:
    """
    Validate nested dict within response.
    
    Args:
        response: Response dict
        path: Dot-separated path to nested dict (e.g., "data.account")
        required_fields: List of required field names in nested dict
        response_name: Name of response for error messages
        
    Returns:
        Validated nested dict
        
    Raises:
        ExchangeInvalidResponseError: If nested dict is invalid or missing
    """
    parts = path.split('.')
    current = response
    
    for part in parts:
        if not isinstance(current, dict):
            raise ExchangeInvalidResponseError(
                f"Invalid {response_name}: path '{path}' - '{part}' is not a dict",
                response=response
            )
        
        if part not in current:
            raise ExchangeInvalidResponseError(
                f"Invalid {response_name}: missing path '{path}'",
                response=response
            )
        
        current = current[part]
    
    if not isinstance(current, dict):
        raise ExchangeInvalidResponseError(
            f"Invalid {response_name}: path '{path}' is not a dict, got {type(current).__name__}",
            response=response
        )
    
    missing_fields = [field for field in required_fields if field not in current]
    if missing_fields:
        available_fields = list(current.keys())
        raise ExchangeInvalidResponseError(
            f"Invalid {response_name}: path '{path}' missing required fields {missing_fields}. "
            f"Available fields: {available_fields}",
            response=response
        )
    
    return current


def validate_field_type(response: Dict, field: str, expected_type: type,
                       response_name: str = "API response") -> Any:
    """
    Validate that a field exists and has the correct type.
    
    Args:
        response: Response dict
        field: Field name to validate
        expected_type: Expected type (e.g., str, int, float, dict, list)
        response_name: Name of response for error messages
        
    Returns:
        Field value
        
    Raises:
        ExchangeInvalidResponseError: If field is missing or wrong type
    """
    if field not in response:
        raise ExchangeInvalidResponseError(
            f"Invalid {response_name}: missing field '{field}'",
            response=response
        )
    
    value = response[field]
    if not isinstance(value, expected_type):
        raise ExchangeInvalidResponseError(
            f"Invalid {response_name}: field '{field}' expected {expected_type.__name__}, "
            f"got {type(value).__name__}",
            response=response
        )
    
    return value


def validate_numeric_field(response: Dict, field: str, min_value: Optional[float] = None,
                          max_value: Optional[float] = None, allow_zero: bool = True,
                          response_name: str = "API response") -> float:
    """
    Validate numeric field with range checks.
    
    Args:
        response: Response dict
        field: Field name to validate
        min_value: Minimum allowed value (None = no minimum)
        max_value: Maximum allowed value (None = no maximum)
        allow_zero: Whether zero is allowed
        response_name: Name of response for error messages
        
    Returns:
        Validated numeric value
        
    Raises:
        ExchangeInvalidResponseError: If field is invalid
    """
    if field not in response:
        raise ExchangeInvalidResponseError(
            f"Invalid {response_name}: missing field '{field}'",
            response=response
        )
    
    value = response[field]
    
    # Try to convert to float if it's a string number
    if isinstance(value, str):
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ExchangeInvalidResponseError(
                f"Invalid {response_name}: field '{field}' cannot be converted to number: {value}",
                response=response
            )
    
    if not isinstance(value, (int, float)):
        raise ExchangeInvalidResponseError(
            f"Invalid {response_name}: field '{field}' expected number, got {type(value).__name__}",
            response=response
        )
    
    value = float(value)
    
    if not allow_zero and value == 0:
        raise ExchangeInvalidResponseError(
            f"Invalid {response_name}: field '{field}' cannot be zero",
            response=response
        )
    
    if min_value is not None and value < min_value:
        raise ExchangeInvalidResponseError(
            f"Invalid {response_name}: field '{field}' value {value} below minimum {min_value}",
            response=response
        )
    
    if max_value is not None and value > max_value:
        raise ExchangeInvalidResponseError(
            f"Invalid {response_name}: field '{field}' value {value} above maximum {max_value}",
            response=response
        )
    
    return value


def validate_order_response(response: Dict) -> Dict:
    """
    Validate order placement response structure.
    
    Expected format:
    {
        "code": 0 or -1,
        "msg": "Success" or error message,
        "data": {
            "orderId": "..." (optional)
        }
    }
    
    Args:
        response: Order response to validate
        
    Returns:
        Validated response dict
        
    Raises:
        ExchangeInvalidResponseError: If response is invalid
    """
    response = validate_dict_response(
        response,
        required_fields=["code", "msg"],
        response_name="order response"
    )
    
    code = validate_field_type(response, "code", int, "order response")
    
    if code not in [0, -1]:
        logger.warning(
            f"Unexpected order response code: {code}. "
            f"Expected 0 (success) or -1 (error)"
        )
    
    # Validate data field if present
    if "data" in response:
        data = response["data"]
        if not isinstance(data, dict):
            raise ExchangeInvalidResponseError(
                f"Invalid order response: 'data' field expected dict, got {type(data).__name__}",
                response=response
            )
    
    return response


def validate_account_info_response(response: Dict) -> Dict:
    """
    Validate account info response structure.
    
    Expected format:
    {
        "balance": str or float,
        "accountValue": str or float (optional),
        "totalMargin": str or float (optional),
        "availableBalance": str or float (optional)
    }
    
    Args:
        response: Account info response to validate
        
    Returns:
        Validated response dict
        
    Raises:
        ExchangeInvalidResponseError: If response is invalid
    """
    response = validate_dict_response(
        response,
        required_fields=["balance"],
        response_name="account info response"
    )
    
    # Validate balance is numeric (can be string representation)
    balance = response.get("balance")
    if balance is not None:
        try:
            float(balance)
        except (ValueError, TypeError):
            raise ExchangeInvalidResponseError(
                f"Invalid account info response: 'balance' must be numeric, got {type(balance).__name__}: {balance}",
                response=response
            )
    
    return response


def validate_position_response(response: Optional[Dict]) -> Optional[Dict]:
    """
    Validate position response structure.
    
    Expected format:
    {
        "symbol": str,
        "side": "LONG" or "SHORT",
        "holdAmount": str or float,
        "openPrice": str or float,
        "unrealizedPnl": str or float (optional),
        ...
    }
    
    Args:
        response: Position response to validate (can be None)
        
    Returns:
        Validated response dict or None
        
    Raises:
        ExchangeInvalidResponseError: If response is invalid
    """
    if response is None:
        return None
    
    response = validate_dict_response(
        response,
        required_fields=["holdAmount"],
        response_name="position response"
    )
    
    # Validate holdAmount is numeric
    hold_amount = response.get("holdAmount")
    if hold_amount is not None:
        try:
            float(hold_amount)
        except (ValueError, TypeError):
            raise ExchangeInvalidResponseError(
                f"Invalid position response: 'holdAmount' must be numeric, got {type(hold_amount).__name__}: {hold_amount}",
                response=response
            )
    
    return response


def validate_ticker_response(response: Dict) -> Dict:
    """
    Validate ticker response structure.
    
    Expected format:
    {
        "markPrice": str or float,
        "lastPrice": str or float (optional),
        "bidPrice": str or float (optional),
        "askPrice": str or float (optional),
        ...
    }
    
    Args:
        response: Ticker response to validate
        
    Returns:
        Validated response dict
        
    Raises:
        ExchangeInvalidResponseError: If response is invalid
    """
    response = validate_dict_response(
        response,
        required_fields=["markPrice"],
        response_name="ticker response"
    )
    
    # Validate markPrice is numeric
    mark_price = response.get("markPrice")
    if mark_price is not None:
        try:
            price = float(mark_price)
            if price <= 0:
                raise ExchangeInvalidResponseError(
                    f"Invalid ticker response: 'markPrice' must be positive, got {price}",
                    response=response
                )
        except (ValueError, TypeError):
            raise ExchangeInvalidResponseError(
                f"Invalid ticker response: 'markPrice' must be numeric, got {type(mark_price).__name__}: {mark_price}",
                response=response
            )
    
    return response


def validate_klines_response(response: List[Dict]) -> List[Dict]:
    """
    Validate klines/candles response structure.
    
    Expected format:
    [
        {
            "timestamp": int or str,
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float
        },
        ...
    ]
    
    Args:
        response: Klines response to validate
        
    Returns:
        Validated list of candle dicts
        
    Raises:
        ExchangeInvalidResponseError: If response is invalid
    """
    response = validate_list_response(
        response,
        min_items=0,
        response_name="klines response"
    )
    
    required_fields = ["open", "high", "low", "close", "volume"]
    
    for idx, candle in enumerate(response):
        if not isinstance(candle, dict):
            raise ExchangeInvalidResponseError(
                f"Invalid klines response: candle at index {idx} expected dict, got {type(candle).__name__}",
                response=response
            )
        
        missing_fields = [field for field in required_fields if field not in candle]
        if missing_fields:
            raise ExchangeInvalidResponseError(
                f"Invalid klines response: candle at index {idx} missing required fields {missing_fields}",
                response=response
            )
        
        # Validate price fields are numeric and positive
        for price_field in ["open", "high", "low", "close"]:
            price = candle.get(price_field)
            if price is not None:
                try:
                    price_val = float(price)
                    if price_val <= 0:
                        raise ExchangeInvalidResponseError(
                            f"Invalid klines response: candle at index {idx} field '{price_field}' "
                            f"must be positive, got {price_val}",
                            response=response
                        )
                except (ValueError, TypeError):
                    raise ExchangeInvalidResponseError(
                        f"Invalid klines response: candle at index {idx} field '{price_field}' "
                        f"must be numeric, got {type(price).__name__}: {price}",
                        response=response
                    )
        
        # Validate volume is numeric and non-negative
        volume = candle.get("volume")
        if volume is not None:
            try:
                volume_val = float(volume)
                if volume_val < 0:
                    raise ExchangeInvalidResponseError(
                        f"Invalid klines response: candle at index {idx} field 'volume' "
                        f"must be non-negative, got {volume_val}",
                        response=response
                    )
            except (ValueError, TypeError):
                raise ExchangeInvalidResponseError(
                    f"Invalid klines response: candle at index {idx} field 'volume' "
                    f"must be numeric, got {type(volume).__name__}: {volume}",
                    response=response
                )
    
    return response


def safe_get(response: Dict, *keys: str, default: Any = None) -> Any:
    """
    Safely get nested value from response dict.
    
    Args:
        response: Response dict
        *keys: Keys to traverse (e.g., "data", "orderId")
        default: Default value if key path not found
        
    Returns:
        Value at key path or default
    """
    current = response
    for key in keys:
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]
    return current

