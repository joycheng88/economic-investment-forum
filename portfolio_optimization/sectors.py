"""
Sector and Industry Classification for Stocks

Maps each stock to its sector and industry for constraint-based portfolio optimization.
"""

STOCK_SECTORS = {
    # Technology
    "MSFT": {"sector": "Technology", "industry": "Software & Services"},
    "GOOG": {"sector": "Technology", "industry": "Internet & e-Commerce"},
    "NVDA": {"sector": "Technology", "industry": "Semiconductors"},
    "META": {"sector": "Technology", "industry": "Internet & e-Commerce"},
    "KWEB": {"sector": "Technology", "industry": "Semiconductors/Internet"},  # Chinese tech ETF
    "IBKR": {"sector": "Technology", "industry": "Financial Software"},
    "SHOP": {"sector": "Technology", "industry": "Internet & e-Commerce"},
    "COIN": {"sector": "Technology", "industry": "Cryptocurrency/Blockchain"},
    "ZETA": {"sector": "Technology", "industry": "Software & Services"},
    
    # Financials
    "JPM": {"sector": "Financials", "industry": "Commercial Banking"},
    "BRK-B": {"sector": "Financials", "industry": "Diversified Financials"},
    "MKL": {"sector": "Financials", "industry": "Insurance"},
    "EFX": {"sector": "Financials", "industry": "Financial Services"},
    
    # Healthcare
    "JNJ": {"sector": "Healthcare", "industry": "Pharmaceuticals"},
    "UNH": {"sector": "Healthcare", "industry": "Health Insurance"},
    
    # Consumer
    "AMZN": {"sector": "Consumer Discretionary", "industry": "Internet & e-Commerce"},
    "UBER": {"sector": "Consumer Discretionary", "industry": "Transportation"},
    "HOOD": {"sector": "Consumer Discretionary", "industry": "Financial Services"},
    
    # Industrials
    "TDW": {"sector": "Industrials", "industry": "Shipping & Logistics"},
    "AMR": {"sector": "Industrials", "industry": "Airlines"},
    
    # Materials & Energy
    "NE": {"sector": "Energy", "industry": "Oil & Gas"},
    
    # Fixed Income
    "SHV": {"sector": "Fixed Income", "industry": "US Treasuries/Short-term Bonds"},
    
    # International/Emerging
    "TSM": {"sector": "Technology", "industry": "Semiconductors (Taiwan)"},
    "HDB": {"sector": "Financials", "industry": "Commercial Banking (China)"},
    "IBN": {"sector": "Financials", "industry": "Commercial Banking (China)"},
    "HCC": {"sector": "Energy", "industry": "Oil & Gas (China)"},
    "BN": {"sector": "Consumer Discretionary", "industry": "E-Commerce (Brazil)"},
    "VAL": {"sector": "Consumer Discretionary", "industry": "Retail (Brazil)"},
}

def get_sectors():
    """Get unique sectors"""
    return sorted(set(s["sector"] for s in STOCK_SECTORS.values()))

def get_industries():
    """Get unique industries"""
    return sorted(set(s["industry"] for s in STOCK_SECTORS.values()))

def get_stocks_by_sector(sector):
    """Get all stocks in a given sector"""
    return [ticker for ticker, info in STOCK_SECTORS.items() 
            if info["sector"] == sector]

def get_stocks_by_industry(industry):
    """Get all stocks in a given industry"""
    return [ticker for ticker, info in STOCK_SECTORS.items() 
            if info["industry"] == industry]

def get_sector_for_stock(ticker):
    """Get sector for a specific stock"""
    return STOCK_SECTORS.get(ticker, {}).get("sector", "Unknown")

def get_industry_for_stock(ticker):
    """Get industry for a specific stock"""
    return STOCK_SECTORS.get(ticker, {}).get("industry", "Unknown")

def fetch_sector_from_yfinance(ticker):
    """
    Attempt to fetch sector from Yahoo Finance data
    
    Parameters:
    -----------
    ticker : str
        Stock ticker
        
    Returns:
    --------
    sector : str or None
        Sector name if found, None otherwise
    """
    try:
        import yfinance as yf
        ticker_data = yf.Ticker(ticker)
        info = ticker_data.info
        return info.get('sector', None)
    except:
        return None

def update_custom_sector(ticker, sector, industry="Custom", custom_mapping=None):
    """
    Register a custom sector/industry mapping for a new stock
    
    Parameters:
    -----------
    ticker : str
        Stock ticker
    sector : str
        Sector name
    industry : str
        Industry name (default "Custom")
    custom_mapping : dict, optional
        Custom mapping dict to update (if None, returns new dict)
        
    Returns:
    --------
    updated_mapping : dict
        Updated sector mapping
    """
    if custom_mapping is None:
        custom_mapping = {}
    
    custom_mapping[ticker] = {
        "sector": sector,
        "industry": industry
    }
    return custom_mapping

def get_sector_allocation(weights):
    """
    Calculate sector allocation given portfolio weights
    
    Parameters:
    -----------
    weights : pd.Series
        Portfolio weights indexed by ticker
        
    Returns:
    --------
    sector_allocation : dict
        Dict mapping sector -> total weight
    """
    allocation = {}
    for ticker, weight in weights.items():
        sector = get_sector_for_stock(ticker)
        allocation[sector] = allocation.get(sector, 0) + weight
    return allocation
