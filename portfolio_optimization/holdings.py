"""
Risk Factor Exposure Decomposition

Analyzes portfolio exposure to key risk factors:
- Market (Beta): Systematic market risk
- Size: Large-cap vs Small-cap exposure
- Value: Value stocks vs Growth stocks (P/B ratio)
- Momentum: Trending vs Mean-reverting exposure
- Quality: High-quality vs Low-quality companies (ROE, profitability)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def fetch_stock_factors(ticker: str) -> Dict:
    """
    Fetch fundamental factors for a stock from Yahoo Finance
    
    Parameters:
    -----------
    ticker : str
        Stock ticker
        
    Returns:
    --------
    factors : dict
        Dictionary with keys: market_cap, pb_ratio, pe_ratio, roe, payout_ratio
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # Extract key metrics with fallback values
        market_cap = info.get('marketCap', np.nan)
        pb_ratio = info.get('priceToBook', np.nan)
        pe_ratio = info.get('trailingPE', np.nan)
        roe = info.get('returnOnEquity', np.nan)
        payout_ratio = info.get('payoutRatio', np.nan)
        current_price = info.get('currentPrice', np.nan)
        
        return {
            'ticker': ticker,
            'market_cap': market_cap,
            'pb_ratio': pb_ratio,
            'pe_ratio': pe_ratio,
            'roe': roe,
            'payout_ratio': payout_ratio,
            'current_price': current_price
        }
    except Exception as e:
        print(f"Error fetching factors for {ticker}: {e}")
        return {
            'ticker': ticker,
            'market_cap': np.nan,
            'pb_ratio': np.nan,
            'pe_ratio': np.nan,
            'roe': np.nan,
            'payout_ratio': np.nan,
            'current_price': np.nan
        }


def calculate_beta(ticker: str, returns_data: pd.DataFrame, 
                  spy_returns: pd.Series) -> float:
    """
    Calculate stock beta relative to S&P 500 (SPY)
    
    Parameters:
    -----------
    ticker : str
        Stock ticker
    returns_data : pd.DataFrame
        Portfolio returns DataFrame (indexed by date)
    spy_returns : pd.Series
        SPY returns (market returns)
        
    Returns:
    --------
    beta : float
        Beta coefficient (market sensitivity)
    """
    if ticker not in returns_data.columns:
        return np.nan
    
    stock_ret = returns_data[ticker].dropna()
    
    # Find common dates between stock and market returns
    common_idx = stock_ret.index.intersection(spy_returns.index)
    
    if len(common_idx) < 2:
        return np.nan
    
    # Align both series to common dates and drop NaNs
    stock_ret_aligned = stock_ret[common_idx].dropna()
    market_ret = spy_returns[common_idx].dropna()
    
    # Re-align after dropping NaNs to ensure same length
    final_idx = stock_ret_aligned.index.intersection(market_ret.index)
    stock_ret_aligned = stock_ret_aligned[final_idx].values
    market_ret = market_ret[final_idx].values
    
    if len(final_idx) < 2:
        return np.nan
    
    # Calculate covariance and variance
    covariance = np.cov(stock_ret_aligned, market_ret)[0, 1]
    market_var = np.var(market_ret)
    
    if market_var == 0:
        return np.nan
    
    beta = covariance / market_var
    return beta


def calculate_size_exposure(market_cap: float) -> float:
    """
    Calculate size exposure as log of market cap
    
    Large market cap -> positive exposure (large-cap bias)
    Small market cap -> negative exposure (small-cap bias)
    
    Parameters:
    -----------
    market_cap : float
        Market capitalization in USD
        
    Returns:
    --------
    size_exposure : float
        Log market cap (normalized)
    """
    if pd.isna(market_cap) or market_cap <= 0:
        return np.nan
    
    log_market_cap = np.log(market_cap)
    return log_market_cap


def calculate_value_exposure(pb_ratio: float) -> float:
    """
    Calculate value exposure (Value Factor)
    
    High P/B -> Growth stocks (negative value exposure)
    Low P/B -> Value stocks (positive value exposure)
    
    Use inverse: Higher value exposure = lower P/B ratio
    
    Parameters:
    -----------
    pb_ratio : float
        Price-to-Book ratio
        
    Returns:
    --------
    value_exposure : float
        Normalized value factor (lower P/B = higher exposure)
    """
    if pd.isna(pb_ratio) or pb_ratio <= 0:
        return np.nan
    
    # Use inverse (lower P/B = higher value exposure)
    # Normalize by log to reduce skewness
    value_factor = -np.log(pb_ratio)  # Negative because we invert
    return value_factor


def calculate_momentum_exposure(returns_data: pd.DataFrame, ticker: str, 
                                lookback_days: int = 126) -> float:
    """
    Calculate momentum exposure (6-month or 12-month trailing return)
    
    Positive momentum -> stock trending up
    Negative momentum -> stock trending down
    
    Parameters:
    -----------
    returns_data : pd.DataFrame
        Returns DataFrame
    ticker : str
        Stock ticker
    lookback_days : int
        Number of trading days for momentum calculation (default 126 ≈ 6 months)
        
    Returns:
    --------
    momentum : float
        Cumulative return over lookback period
    """
    if ticker not in returns_data.columns:
        return np.nan
    
    returns = returns_data[ticker].tail(lookback_days)
    
    if len(returns) < 10:
        return np.nan
    
    # Calculate cumulative return (momentum)
    momentum = (1 + returns).prod() - 1
    return momentum


def calculate_quality_score(pb_ratio: float, roe: float, 
                           pe_ratio: float, payout_ratio: float) -> float:
    """
    Calculate quality score (combination of profitability and financial health)
    
    High quality: High ROE, Low P/B, Reasonable P/E, Sustainable payout
    Low quality: Low ROE, High P/B, High P/E, Unsustainable payout
    
    Parameters:
    -----------
    pb_ratio : float
        Price-to-Book ratio
    roe : float
        Return on Equity (ROE)
    pe_ratio : float
        Price-to-Earnings ratio
    payout_ratio : float
        Dividend payout ratio
        
    Returns:
    --------
    quality_score : float
        Combined quality metric (-1 to +1 typically)
    """
    components = []
    
    # ROE component (higher ROE = higher quality)
    if not pd.isna(roe) and roe > 0:
        roe_component = np.clip(roe, 0, 0.5) / 0.5  # Max at 50% ROE
        components.append(roe_component)
    
    # P/B component (lower P/B = higher quality for given earnings)
    if not pd.isna(pb_ratio) and pb_ratio > 0:
        pb_component = 1 / (1 + pb_ratio)  # Range 0-1
        components.append(pb_component)
    
    # P/E component (lower P/E = higher quality)
    if not pd.isna(pe_ratio) and pe_ratio > 0:
        pe_component = 1 / (1 + pe_ratio / 20)  # Normalized to typical P/E of 20
        components.append(pe_component)
    
    # Payout ratio component (sustainable payout = higher quality)
    if not pd.isna(payout_ratio):
        payout_component = 1 - abs(payout_ratio - 0.3)  # Optimal around 30%
        payout_component = np.clip(payout_component, 0, 1)
        components.append(payout_component)
    
    if not components:
        return np.nan
    
    # Average components and normalize to roughly -1 to +1
    quality_score = np.mean(components) * 2 - 1
    return quality_score


def portfolio_factor_exposure(weights: pd.Series, returns_data: pd.DataFrame,
                             spy_ticker: str = 'SPY') -> Dict:
    """
    Calculate portfolio-level factor exposures
    
    Parameters:
    -----------
    weights : pd.Series
        Portfolio weights indexed by ticker
    returns_data : pd.DataFrame
        Returns DataFrame for calculating beta and momentum
    spy_ticker : str
        Ticker for market benchmark (default 'SPY')
        
    Returns:
    --------
    exposures : dict
        Portfolio factor exposures with keys:
        - beta_portfolio: Weighted average beta
        - size_portfolio: Weighted average size exposure
        - value_portfolio: Weighted average value exposure
        - momentum_portfolio: Weighted average momentum
        - quality_portfolio: Weighted average quality score
        - factor_breakdown: DataFrame with individual stock exposures
    """
    
    # Download SPY data for beta calculation
    spy_data = yf.download(spy_ticker, start=returns_data.index[0], 
                           end=returns_data.index[-1], progress=False, auto_adjust=False)
    
    # Handle both single and multi-index column structures
    if isinstance(spy_data.columns, pd.MultiIndex):
        # MultiIndex: columns are (price_type, ticker)
        spy_returns = spy_data['Adj Close', spy_ticker].pct_change()
    elif 'Adj Close' in spy_data.columns:
        spy_returns = spy_data['Adj Close'].pct_change()
    else:
        # Fallback: use first column if available
        spy_returns = spy_data.iloc[:, 0].pct_change()
    
    results = []
    
    for ticker, weight in weights.items():
        if weight < 1e-4:  # Skip near-zero positions
            continue
        
        # Get fundamental factors
        factors = fetch_stock_factors(ticker)
        
        # Calculate derived factors
        beta = calculate_beta(ticker, returns_data, spy_returns)
        size = calculate_size_exposure(factors['market_cap'])
        value = calculate_value_exposure(factors['pb_ratio'])
        momentum = calculate_momentum_exposure(returns_data, ticker)
        quality = calculate_quality_score(
            factors['pb_ratio'], 
            factors['roe'], 
            factors['pe_ratio'], 
            factors['payout_ratio']
        )
        
        results.append({
            'Ticker': ticker,
            'Weight': weight,
            'Beta': beta,
            'Size': size,
            'Value': value,
            'Momentum': momentum,
            'Quality': quality,
            'Market Cap': factors['market_cap'],
            'P/B': factors['pb_ratio'],
            'ROE': factors['roe']
        })
    
    df_breakdown = pd.DataFrame(results)
    
    if df_breakdown.empty:
        return {
            'beta_portfolio': np.nan,
            'size_portfolio': np.nan,
            'value_portfolio': np.nan,
            'momentum_portfolio': np.nan,
            'quality_portfolio': np.nan,
            'factor_breakdown': df_breakdown
        }
    
    # Compute weighted factors
    beta_portfolio = (df_breakdown['Beta'] * df_breakdown['Weight']).sum() / df_breakdown['Weight'].sum()
    size_portfolio = (df_breakdown['Size'] * df_breakdown['Weight']).sum() / df_breakdown['Weight'].sum()
    value_portfolio = (df_breakdown['Value'] * df_breakdown['Weight']).sum() / df_breakdown['Weight'].sum()
    momentum_portfolio = (df_breakdown['Momentum'] * df_breakdown['Weight']).sum() / df_breakdown['Weight'].sum()
    quality_portfolio = (df_breakdown['Quality'] * df_breakdown['Weight']).sum() / df_breakdown['Weight'].sum()
    
    return {
        'beta_portfolio': beta_portfolio,
        'size_portfolio': size_portfolio,
        'value_portfolio': value_portfolio,
        'momentum_portfolio': momentum_portfolio,
        'quality_portfolio': quality_portfolio,
        'factor_breakdown': df_breakdown
    }


def get_benchmark_exposures(benchmark_tickers: list = ['SPY'], 
                            returns_data: Optional[pd.DataFrame] = None) -> Dict:
    """
    Calculate factor exposures for benchmark portfolios
    
    Parameters:
    -----------
    benchmark_tickers : list
        List of benchmark tickers (e.g., ['SPY', 'QQQ', 'IWM'])
    returns_data : pd.DataFrame, optional
        Returns data for momentum/beta calculation
        
    Returns:
    --------
    benchmarks : dict
        Dictionary mapping benchmark ticker -> exposures dict
    """
    benchmarks = {}
    
    for ticker in benchmark_tickers:
        # Equal weight benchmark (simplistic approach)
        weights = pd.Series([1.0], index=[ticker])
        
        exposures = {
            'ticker': ticker,
            'beta': 1.0 if ticker == 'SPY' else np.nan,  # SPY vs itself = 1
            'size': np.nan,
            'value': np.nan,
            'momentum': np.nan,
            'quality': np.nan
        }
        
        # Try to fetch factors
        factors = fetch_stock_factors(ticker)
        exposures['size'] = calculate_size_exposure(factors['market_cap'])
        exposures['value'] = calculate_value_exposure(factors['pb_ratio'])
        exposures['quality'] = calculate_quality_score(
            factors['pb_ratio'],
            factors['roe'],
            factors['pe_ratio'],
            factors['payout_ratio']
        )
        
        if returns_data is not None and ticker in returns_data.columns:
            exposures['momentum'] = calculate_momentum_exposure(returns_data, ticker)
        
        benchmarks[ticker] = exposures
    
    return benchmarks
