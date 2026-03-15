"""Real-time options data fetching from yfinance with validation."""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Tuple, Optional, List, Dict
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def get_underlying_price(ticker: str, retries: int = 3) -> float:
    """Fetch current underlying asset price from yfinance."""
    for attempt in range(retries):
        try:
            data = yf.download(ticker, period='1d', progress=False)
            if data is None or len(data) == 0:
                raise ValueError(f"No data for {ticker}")
            price = float(data['Close'].iloc[-1])
            if price <= 0:
                raise ValueError(f"Invalid price {price}")
            logger.info(f"Price for {ticker}: ${price:.2f}")
            return price
        except Exception as e:
            if attempt == retries - 1:
                raise ValueError(f"Failed to fetch {ticker}: {e}")
    return None


def get_risk_free_rate(tenor: str = '3m') -> float:
    """
    Fetch current risk-free rate (US Treasury yield).
    
    Parameters:
    -----------
    tenor : str
        Tenor for risk-free rate ('3m', '6m', '1y', '10y')
        
    Returns:
    --------
    float
        Risk-free rate (annualized, as decimal)
    """
    try:
        # Fetch 10-year Treasury yield as proxy
        treasury = yf.download('^TNX', period='1d', progress=False)
        if treasury is None or len(treasury) == 0:
            logger.warning("Could not fetch Treasury yield, using default 4.5%")
            return 0.045
        
        yield_rate = float(treasury['Close'].iloc[-1]) / 100.0
        logger.info(f"Fetched risk-free rate: {yield_rate:.2%}")
        return yield_rate
        
    except Exception as e:
        logger.warning(f"Failed to fetch risk-free rate: {e}. Using default 4.5%")
        return 0.045


def get_dividend_yield(ticker: str) -> float:
    """
    Fetch dividend yield for underlying asset.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
        
    Returns:
    --------
    float
        Dividend yield (annualized, as decimal)
    """
    try:
        info = yf.Ticker(ticker).info
        dividend_yield = info.get('dividendYield', 0.0)
        
        if dividend_yield is None:
            dividend_yield = 0.0
        
        logger.info(f"Dividend yield for {ticker}: {dividend_yield:.2%}")
        return float(dividend_yield)
        
    except Exception as e:
        logger.warning(f"Failed to fetch dividend yield for {ticker}: {e}. Using 0%")
        return 0.0


def get_expiration_dates(ticker: str, num_expirations: int = 5) -> List[str]:
    """
    Fetch available option expiration dates.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    num_expirations : int
        Maximum number of expiration dates to return
        
    Returns:
    --------
    List[str]
        List of available expiration dates (YYYY-MM-DD format)
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        exp_dates = ticker_obj.options[:num_expirations]
        
        logger.info(f"Found {len(exp_dates)} expiration dates for {ticker}")
        return exp_dates
        
    except Exception as e:
        logger.error(f"Failed to fetch expiration dates for {ticker}: {e}")
        return []


def fetch_option_chain(ticker: str, expiration_date: str, 
                      min_bid: float = 0.01) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch call and put option chains for a given expiration date.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    expiration_date : str
        Expiration date (YYYY-MM-DD format)
    min_bid : float
        Minimum bid price to filter out illiquid options
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        (calls_df, puts_df) cleaned and filtered option chains
    """
    try:
        logger.info(f"Fetching option chain for {ticker} expiration {expiration_date}")
        
        ticker_obj = yf.Ticker(ticker)
        option_chain = ticker_obj.option_chain(expiration_date)
        
        calls = option_chain.calls.copy()
        puts = option_chain.puts.copy()
        
        # Clean and filter
        calls = _clean_option_chain(calls, min_bid, 'CALL')
        puts = _clean_option_chain(puts, min_bid, 'PUT')
        
        logger.info(f"Retrieved {len(calls)} calls and {len(puts)} puts for {ticker}")
        
        return calls, puts
        
    except Exception as e:
        logger.error(f"Failed to fetch option chain: {e}")
        return pd.DataFrame(), pd.DataFrame()


def _clean_option_chain(df: pd.DataFrame, min_bid: float, option_type: str) -> pd.DataFrame:
    """
    Clean and filter option chain data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw option chain data from yfinance
    min_bid : float
        Minimum bid price threshold
    option_type : str
        'CALL' or 'PUT' for logging
        
    Returns:
    --------
    pd.DataFrame
        Cleaned option chain
    """
    try:
        # Drop rows with missing critical data
        df = df.dropna(subset=['bid', 'ask', 'impliedVolatility', 'strike'])
        
        # Filter out zero/negative bids (illiquid options)
        df = df[df['bid'] >= min_bid]
        
        # Calculate mid price
        df['mid'] = (df['bid'] + df['ask']) / 2.0
        
        # Calculate bid-ask spread
        df['spread'] = df['ask'] - df['bid']
        
        # Calculate spread as percentage of mid
        df['spread_pct'] = df['spread'] / df['mid']
        
        # Filter extreme spreads (likely data quality issues)
        df = df[df['spread_pct'] < 0.5]
        
        # Validate implied volatility
        df = df[(df['impliedVolatility'] > 0) & (df['impliedVolatility'] < 2.0)]
        
        # Rename for consistency
        df = df.rename(columns={
            'impliedVolatility': 'iv',
            'lastPrice': 'lastPrice'
        })
        
        df['type'] = option_type
        
        return df.reset_index(drop=True)
        
    except Exception as e:
        logger.error(f"Error cleaning {option_type} chain: {e}")
        return pd.DataFrame()


def prepare_pricing_inputs(ticker: str, expiration_date: str, 
                          valuation_date: Optional[datetime] = None) -> Dict:
    """
    Prepare comprehensive inputs for options pricing models.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    expiration_date : str
        Option expiration date (YYYY-MM-DD)
    valuation_date : Optional[datetime]
        Valuation date (default: today)
        
    Returns:
    --------
    Dict
        Dictionary with all pricing inputs:
        {
            'S0': underlying_price,
            'K': np.array of strike prices,
            'r': risk_free_rate,
            'q': dividend_yield,
            'sigma': np.array of implied volatilities,
            'T': time_to_maturity,
            'calls': pd.DataFrame,
            'puts': pd.DataFrame,
            'metadata': {...}
        }
    """
    if valuation_date is None:
        valuation_date = datetime.now()
    
    try:
        logger.info(f"Preparing pricing inputs for {ticker}")
        
        # Fetch market data
        S0 = get_underlying_price(ticker)
        r = get_risk_free_rate()
        q = get_dividend_yield(ticker)
        calls, puts = fetch_option_chain(ticker, expiration_date)
        
        if calls.empty or puts.empty:
            raise ValueError("Failed to fetch option chains")
        
        # Calculate time to maturity
        exp_date = pd.to_datetime(expiration_date)
        T = max((exp_date - valuation_date).days / 365.0, 0.001)
        
        # Prepare returns dictionary
        result = {
            'S0': S0,
            'r': r,
            'q': q,
            'T': T,
            'calls': calls,
            'puts': puts,
            'metadata': {
                'ticker': ticker,
                'expiration_date': expiration_date,
                'valuation_date': valuation_date.strftime('%Y-%m-%d'),
                'fetch_timestamp': datetime.now().isoformat(),
                'num_calls': len(calls),
                'num_puts': len(puts)
            }
        }
        
        logger.info(f"Successfully prepared pricing inputs for {ticker}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to prepare pricing inputs: {e}")
        return {}


def fetch_historical_data(ticker: str, start_date: str, end_date: str, 
                         interval: str = '1d') -> pd.DataFrame:
    """
    Fetch historical price data for backtesting or analysis.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    interval : str
        Data interval ('1d', '1wk', '1mo')
        
    Returns:
    --------
    pd.DataFrame
        Historical OHLCV data
    """
    try:
        logger.info(f"Fetching historical data for {ticker} from {start_date} to {end_date}")
        
        data = yf.download(ticker, start=start_date, end=end_date, 
                          interval=interval, progress=False)
        
        if data is None or len(data) == 0:
            raise ValueError(f"No historical data found for {ticker}")
        
        data['Returns'] = data['Adj Close'].pct_change()
        data['Log_Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
        
        logger.info(f"Retrieved {len(data)} historical records for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to fetch historical data: {e}")
        return pd.DataFrame()


def calculate_historical_volatility(ticker: str, periods: int = 252) -> float:
    """
    Calculate historical volatility from recent data.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    periods : int
        Number of trading days to use (default: 1 year = 252 days)
        
    Returns:
    --------
    float
        Annualized historical volatility
    """
    try:
        start_date = (datetime.now() - timedelta(days=periods*1.5)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        data = fetch_historical_data(ticker, start_date, end_date)
        
        if data.empty:
            logger.warning(f"Could not calculate historical volatility for {ticker}. Using 20%")
            return 0.20
        
        log_returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
        log_returns = log_returns.dropna()
        
        hist_vol = log_returns.std() * np.sqrt(252)  # Annualized
        
        logger.info(f"Historical volatility for {ticker}: {hist_vol:.2%}")
        return hist_vol
        
    except Exception as e:
        logger.error(f"Failed to calculate historical volatility: {e}. Using 20%")
        return 0.20


if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    
    print("=" * 60)
    print("Options Data Fetching Module - Examples")
    print("=" * 60)
    
    # Get basic market data
    try:
        price = get_underlying_price(ticker)
        print(f"\n✓ Current price for {ticker}: ${price:.2f}")
    except Exception as e:
        print(f"✗ Failed to fetch price: {e}")
    
    # Get risk-free rate
    try:
        rfr = get_risk_free_rate()
        print(f"✓ Risk-free rate: {rfr:.2%}")
    except Exception as e:
        print(f"✗ Failed to fetch risk-free rate: {e}")
    
    # Get dividend yield
    try:
        div_yield = get_dividend_yield(ticker)
        print(f"✓ Dividend yield for {ticker}: {div_yield:.2%}")
    except Exception as e:
        print(f"✗ Failed to fetch dividend yield: {e}")
    
    # Get expiration dates
    try:
        exp_dates = get_expiration_dates(ticker)
        print(f"✓ Available expirations: {exp_dates[:3]}")
    except Exception as e:
        print(f"✗ Failed to fetch expiration dates: {e}")
    
    # Get historical volatility
    try:
        hist_vol = calculate_historical_volatility(ticker)
        print(f"✓ Historical volatility: {hist_vol:.2%}")
    except Exception as e:
        print(f"✗ Failed to calculate historical volatility: {e}")
    
    print("\n" + "=" * 60)
