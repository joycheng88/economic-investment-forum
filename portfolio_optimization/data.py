from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List, Dict
import pandas as pd
import numpy as np
import warnings
import logging
import os

# Suppress HTTP and network errors
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('requests').setLevel(logging.CRITICAL)


def get_risk_free_rate() -> float:
    """
    Fetch the current risk-free rate using the US 10-year Treasury yield.
    
    Returns:
    --------
    float
        Annualized risk-free rate (e.g., 0.045 for 4.5%)
        Falls back to 0.02 (2%) if fetch fails
    """
    try:
        import yfinance as yf
        # US 10-year Treasury yield is a common proxy for risk-free rate
        tnx = yf.download('^TNX', period='1d', progress=False)
        if tnx is not None and len(tnx) > 0:
            # Get the closing price as a scalar
            current_close = tnx['Close'].iloc[-1]
            if isinstance(current_close, (int, float)):
                current_rate = float(current_close) / 100  # Convert percentage to decimal
            else:
                current_rate = float(current_close.values[-1]) / 100
            return current_rate
        else:
            warnings.warn("Could not fetch Treasury yield, using default 2%")
            return 0.02
    except Exception as e:
        warnings.warn(f"Error fetching risk-free rate: {e}. Using default 2%")
        return 0.02


@dataclass
class DataConfig:
    """Configuration for data fetching and processing.
    
    Attributes:
    -----------
    start : str
        Start date in 'YYYY-MM-DD' format (default: "2022-01-01")
    end : Optional[str]
        End date in 'YYYY-MM-DD' format (default: None = today)
    interval : str
        Data interval: "1d", "1wk", "1mo" (default: "1d")
    auto_adjust : bool
        Whether to auto-adjust OHLC data for splits/dividends (default: False)
    min_non_na_ratio : float
        Minimum ratio of non-NA values required per column, 0-1 (default: 0.95)
    return_type : str
        Type of returns: "log" or "simple" (default: "log")
    fill_method : str
        Method to fill remaining NAs: "forward", "drop" (default: "forward")
    """
    start: str = "2022-01-01"
    end: Optional[str] = None
    interval: str = "1d"
    auto_adjust: bool = False
    min_non_na_ratio: float = 0.95
    return_type: str = "log"
    fill_method: str = "forward"

def validate_ticker(ticker: str, timeout: int = 5) -> Dict[str, any]:
    """
    Validate if a ticker exists and identify its type (stock, ETF, or global).
    
    Scope:
    ------
    This function uses yfinance, which supports:
    - US Stocks: AAPL, MSFT, TSLA, etc.
    - International Stocks: 0001.HK (Hong Kong), 005930.KS (South Korea), etc.
    - ETFs: SPY, QQQ, VTI, VXUS, etc.
    - Cryptocurrencies: BTC-USD (Bitcoin), ETH-USD (Ethereum), etc.
    - Bonds: BND (treasury bonds), TLT (long-term treasury), etc.
    
    Any ticker supported by Yahoo Finance can be used.
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol to validate (e.g., 'AAPL', 'SPY', '0001.HK', 'BTC-USD')
    timeout : int
        Timeout in seconds for the validation check
        
    Returns:
    --------
    dict
        {
            'valid': bool,
            'ticker': str,
            'name': str or None,
            'type': str ('Stock', 'ETF', 'Crypto', 'Unknown'),
            'exchange': str or None,
            'error': str or None
        }
    """
    try:
        import yfinance as yf
    except ImportError:
        return {
            'valid': False,
            'ticker': ticker,
            'name': None,
            'type': 'Unknown',
            'exchange': None,
            'error': 'yfinance not installed'
        }
    
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        # Try to fetch info with timeout
        info = ticker_obj.info
        
        if info and len(info) > 1:  # More than 1 key means real data (not just error)
            # Determine asset type
            asset_type = info.get('quoteType', 'Unknown')
            name = info.get('longName', ticker.upper())
            exchange = info.get('exchange', 'Unknown')
            
            # Map quote type to readable format
            type_map = {
                'EQUITY': 'Stock',
                'ETF': 'ETF',
                'INDEX': 'Index',
                'FUTURE': 'Futures',
                'CURRENCY': 'Crypto'
            }
            
            return {
                'valid': True,
                'ticker': ticker.upper(),
                'name': name,
                'type': type_map.get(asset_type, asset_type),
                'exchange': exchange,
                'error': None
            }
        else:
            # Check if there's at least some valid history
            hist = ticker_obj.history(period='1d')
            if hist is not None and len(hist) > 0:
                return {
                    'valid': True,
                    'ticker': ticker.upper(),
                    'name': ticker.upper(),
                    'type': 'Asset',
                    'exchange': 'Unknown',
                    'error': None
                }
            else:
                return {
                    'valid': False,
                    'ticker': ticker.upper(),
                    'name': None,
                    'type': 'Unknown',
                    'exchange': None,
                    'error': 'No data available for this ticker'
                }
    except Exception as e:
        return {
            'valid': False,
            'ticker': ticker.upper(),
            'name': None,
            'type': 'Unknown',
            'exchange': None,
            'error': str(e)
        }


def fetch_prices(tickers: Iterable[str], config: DataConfig) -> pd.DataFrame:
    """
    Fetch historical price data from yfinance for the given tickers.

    Parameters:
    -----------
    tickers : Iterable[str]
        An iterable of stock ticker symbols
    config : DataConfig
        Data fetching configuration

    Returns:
    --------
    pd.DataFrame
        DataFrame with index=dates, columns=tickers, values=adjusted close prices

    Raises:
    -------
    ValueError
        If tickers are empty or invalid
    ImportError
        If yfinance is not installed
    RuntimeError
        If yfinance returns no data or data is malformed
    """
    # Normalize tickers
    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if len(tickers) == 0:
        raise ValueError("Tickers list is empty after cleaning. Provide at least one valid ticker.")

    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("yfinance not installed. Install with: pip install yfinance") from e

    # Validate date format
    try:
        pd.to_datetime(config.start)
        if config.end:
            pd.to_datetime(config.end)
    except Exception as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD format. Error: {e}") from e

    # Download with error handling and retry logic for transient errors
    import time
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            raw = yf.download(
                tickers=tickers,
                start=config.start,
                end=config.end,
                interval=config.interval,
                auto_adjust=config.auto_adjust,
                progress=False,
                group_by="column",
                threads=True,
            )
            break  # Success, exit retry loop
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)  # Wait before retrying
                continue
            else:
                raise RuntimeError(f"yfinance download failed after {max_retries} attempts: {e}") from e

    if raw is None or raw.empty:
        raise RuntimeError(
            f"yfinance returned no data. Check tickers {tickers} "
            f"and date range {config.start} to {config.end}."
        )

    # Extract adjusted close prices, with fallback to close
    prices = _extract_prices_from_raw(raw, tickers)

    # Ensure chronological order and no duplicate columns
    prices = prices.sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated(keep="first")]

    return prices


def _extract_prices_from_raw(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Extract adjusted close prices from raw yfinance download data.

    Handles both single and multiple ticker cases.

    Parameters:
    -----------
    raw : pd.DataFrame
        Raw data from yfinance.download()
    tickers : list
        List of tickers requested

    Returns:
    --------
    pd.DataFrame
        Prices with index=dates, columns=tickers

    Raises:
    -------
    RuntimeError
        If Adj Close or Close data is not available
    """
    # Case 1: Multiple tickers -> MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        level_0 = raw.columns.get_level_values(0)
        
        if "Adj Close" in level_0:
            prices = raw["Adj Close"].copy()
        elif "Close" in level_0:
            prices = raw["Close"].copy()
        else:
            available = level_0.unique().tolist()
            raise RuntimeError(
                f"Data does not contain 'Adj Close' or 'Close'. "
                f"Available columns: {available}"
            )
    # Case 2: Single ticker -> non-MultiIndex columns
    else:
        if "Adj Close" in raw.columns:
            prices = raw[["Adj Close"]].copy()
            prices.columns = [tickers[0]]
        elif "Close" in raw.columns:
            prices = raw[["Close"]].copy()
            prices.columns = [tickers[0]]
        else:
            available = raw.columns.tolist()
            raise RuntimeError(
                f"Data does not contain 'Adj Close' or 'Close'. "
                f"Available columns: {available}"
            )

    return prices



def clean_prices(
    prices: pd.DataFrame,
    config: DataConfig = DataConfig(),
) -> pd.DataFrame:
    """
    Clean price data by removing rows/columns with excessive missing values and filling remaining gaps.

    Steps:
    1. Remove rows where all prices are missing (trading halts)
    2. Remove columns (tickers) with missing data ratio > threshold
    3. Fill remaining missing values using specified method

    Parameters:
    -----------
    prices : pd.DataFrame
        Price data with index=dates, columns=tickers
    config : DataConfig
        Configuration with min_non_na_ratio and fill_method

    Returns:
    --------
    pd.DataFrame
        Cleaned price data, ready for analysis

    Raises:
    -------
    ValueError
        If prices is None or empty
    RuntimeError
        If no tickers survive the quality threshold
    """
    if prices is None or prices.empty:
        raise ValueError("Price data is None or empty, cannot clean.")

    original_shape = prices.shape
    
    # Step 1: Remove rows with all missing values (complete trading halts)
    prices = prices.dropna(how="all")
    
    # Step 2: Remove tickers with too many missing values
    non_na_ratio = prices.notna().mean(axis=0)
    keep_tickers = non_na_ratio[non_na_ratio >= config.min_non_na_ratio].index.tolist()
    
    if len(keep_tickers) == 0:
        raise RuntimeError(
            f"All tickers dropped due to min_non_na_ratio={config.min_non_na_ratio}. "
            f"Tickers and their NA ratios:\n{non_na_ratio}\n"
            f"Try lowering min_non_na_ratio or check data quality."
        )
    
    dropped_tickers = set(prices.columns) - set(keep_tickers)
    if dropped_tickers:
        warnings.warn(
            f"Dropped {len(dropped_tickers)} tickers due to missing data: {dropped_tickers}",
            UserWarning
        )
    
    prices = prices[keep_tickers]
    
    # Step 3: Handle remaining missing values
    if prices.isna().any().any():
        if config.fill_method == "forward":
            prices = prices.fillna(method="ffill").fillna(method="bfill")
        elif config.fill_method == "drop":
            prices = prices.dropna(how="any")
        else:
            raise ValueError(f"Unknown fill_method: {config.fill_method}")
    
    final_shape = prices.shape
    print(f"Data cleaning: {original_shape} -> {final_shape} (removed {original_shape[0]-final_shape[0]} rows, {original_shape[1]-final_shape[1]} tickers)")
    
    return prices



def compute_returns(
    prices: pd.DataFrame,
    config: DataConfig | None = None,
) -> pd.DataFrame:
    """
    Compute returns from price data.

    Supports two methods:
    - Log returns: ln(P_t / P_{t-1})
    - Simple returns: P_t / P_{t-1} - 1

    Parameters:
    -----------
    prices : pd.DataFrame
        Price data with index=dates, columns=tickers
    config : DataConfig, optional
        Configuration specifying return_type. If None, uses default config.

    Returns:
    --------
    pd.DataFrame
        Returns data with index=dates (starting from day 2), columns=tickers
        First row (NaN) is automatically dropped.

    Raises:
    -------
    ValueError
        If prices is None/empty or return_type is invalid
    """
    if prices is None or prices.empty:
        raise ValueError("Price data is None or empty, cannot compute returns.")

    if config is None:
        config = DataConfig()

    if config.return_type == "log":
        returns = np.log(prices / prices.shift(1))
    elif config.return_type == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"return_type must be 'log' or 'simple', got: {config.return_type}")

    # Drop first row (NaN) from shift
    returns = returns.iloc[1:].copy()
    
    return returns

def load_data(
    tickers: Iterable[str],
    config: DataConfig = DataConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and process stock price data end-to-end.

    Pipeline:
    1. Fetch prices from yfinance
    2. Clean (remove low-quality tickers, fill gaps)
    3. Compute returns

    Parameters:
    -----------
    tickers : Iterable[str]
        Stock ticker symbols
    config : DataConfig
        Configuration for all processing steps

    Returns:
    --------
    prices : pd.DataFrame
        Cleaned adjusted closing prices, index=dates, columns=tickers
    returns : pd.DataFrame
        Computed returns, index=dates, columns=tickers

    Raises:
    -------
    ValueError, ImportError, RuntimeError
        Various data issues (see fetch_prices, clean_prices, compute_returns)

    Example:
    --------
    >>> tickers = ["SPY", "QQQ", "IWM"]
    >>> cfg = DataConfig(start="2023-01-01", return_type="log")
    >>> prices, returns = load_data(tickers, cfg)
    """
    prices = fetch_prices(tickers, config)
    prices_clean = clean_prices(prices, config)
    returns = compute_returns(prices_clean, config)
    
    return prices_clean, returns


def data_summary(prices: pd.DataFrame, returns: pd.DataFrame) -> None:
    """
    Print summary statistics of loaded data.

    Parameters:
    -----------
    prices : pd.DataFrame
        Price data
    returns : pd.DataFrame
        Return data
    """
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    
    print("\nPRICES:")
    print(f"  Shape: {prices.shape} (dates x tickers)")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  Tickers: {', '.join(prices.columns)}")
    print(f"  Missing values: {prices.isna().sum().sum()}")
    
    print("\nRETURNS:")
    print(f"  Shape: {returns.shape}")
    print(f"  Mean (annualized):\n{(returns.mean() * 252)}")
    print(f"  Volatility (annualized):\n{(returns.std() * np.sqrt(252))}")
    print(f"  Correlation:\n{returns.corr()}")
    
    print("\n" + "="*70)