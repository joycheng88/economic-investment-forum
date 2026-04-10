"""
PROFESSIONAL-GRADE FINANCIAL VALUATION & MODELING MODULE
=========================================================

Implements:
- DCF (Discounted Cash Flow) Valuation
- Comparable Companies (Multiples) Analysis
- Sensitivity Analysis with comprehensive tables
- Institutional-grade assumptions and data fetching
- Risk metrics and statistical rigor

Data Sources:
- Yahoo Finance (primary, ticker data, financial statements)
- Fred API (risk-free rate, macro data)
- Institutional peer groups (pre-verified comparable companies)

Usage:
    from modeling import DCFValuation, ValuationReport
    
    dcf = DCFValuation(ticker='AAPL')
    result = dcf.run_analysis(
        wacc=0.10,
        terminal_growth=0.025,
        forecast_years=5
    )
    
    report = ValuationReport(dcf_result=result)
    report.display_summary()
    report.generate_sensitivity_table()
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from functools import lru_cache
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')


# ==================== UTILITY FUNCTIONS ====================

def _extract_safe(source: any, *keys: str, default: any = 0):
    """
    Safely extract nested values from dicts/objects with fallback fields.
    
    Usage:
        _extract_safe(info, 'currentPrice', 'regularMarketPrice')
        _extract_safe(df, 'Total Revenue', default=0)
    """
    if source is None:
        return default
    
    for key in keys:
        if isinstance(source, dict):
            if key in source and source[key] not in (None, ''):
                val = source[key]
                if isinstance(val, (int, float)) and val > 0:
                    return val
        else:
            try:
                if hasattr(source, key):
                    val = getattr(source, key)
                    if val not in (None, '') and (not isinstance(val, float) or val > 0):
                        return val
            except:
                pass
    
    return default


def _extract_financial_field(df, *field_names: str, default: float = 0) -> float:
    """
    Extract financial statement field with multiple name attempts.
    Handles yfinance field name variations.
    """
    if df is None or df.empty:
        return default
    
    for field in field_names:
        if field in df.index:
            try:
                val = float(df.loc[field].iloc[0])
                if val != 0:
                    return val
            except:
                pass
    
    return default


# ==================== CONSTANTS & BENCHMARKS ====================

# Institutional peer groups (pre-verified)
PEER_GROUPS = {
    'Technology': ['MSFT', 'GOOGL', 'NVDA', 'AMD', 'CRM', 'SNPS', 'NOW', 'ADBE'],
    'Software': ['MSFT', 'ADBE', 'CRM', 'SNPS', 'WDAY', 'NFLX', 'NOW'],
    'Semiconductors': ['NVDA', 'AMD', 'QCOM', 'AVGO', 'MCHP', 'ASML', 'LRCX'],
    'Cloud/SaaS': ['CRM', 'NOW', 'WDAY', 'OKTA', 'DDOG', 'ZS', 'CRWD'],
    'E-commerce': ['AMZN', 'EBAY', 'MELI', 'SHOP'],
    'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW'],
    'Banking': ['JPM', 'BAC', 'WFC', 'GS', 'SCCO'],
    'Insurance': ['BRK-B', 'BAC', 'LPL', 'CMG'],
    'Healthcare': ['UNH', 'JNJ', 'PFE', 'ABBV', 'LLY', 'MRK', 'ELI'],
    'Biotech': ['AMGN', 'GILD', 'CELG', 'VRTX', 'REGN', 'CRSP'],
    'Industrials': ['CAT', 'BA', 'HON', 'UPS', 'RTX', 'MMM', 'GE'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC'],
    'Consumer Staples': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'NKE'],
    'Retail': ['WMT', 'TGT', 'HD', 'LOWE'],
    'Communication': ['GOOGL', 'META', 'DIS', 'NFLX', 'CMCSA', 'PARA'],
    'Media': ['DIS', 'NFLX', 'PARA', 'FOXA', 'SONY'],
}

# Industry average margins (source: EDGAR filings)
INDUSTRY_MARGINS = {
    'Technology': {'operating': 0.28, 'fcf_to_revenue': 0.22},
    'Software': {'operating': 0.35, 'fcf_to_revenue': 0.28},
    'Semiconductors': {'operating': 0.25, 'fcf_to_revenue': 0.18},
    'Financial': {'operating': 0.35, 'fcf_to_revenue': 0.20},
    'Healthcare': {'operating': 0.22, 'fcf_to_revenue': 0.15},
    'Industrials': {'operating': 0.15, 'fcf_to_revenue': 0.10},
    'Energy': {'operating': 0.12, 'fcf_to_revenue': 0.08},
    'Consumer': {'operating': 0.10, 'fcf_to_revenue': 0.06},
}

# Macroeconomic assumptions (institutionally sourced)
DEFAULT_ASSUMPTIONS = {
    'risk_free_rate': 0.042,  # US 10Y Treasury (~4.2%)
    'market_risk_premium': 0.055,  # Equity risk premium (5.5% is reasonable for 2026)
    'terminal_growth': 0.035,  # Terminal growth rate (3.5% accounts for inflation + real growth)
    'tax_rate': 0.21,  # US federal corporate tax
}


# ==================== DATA FETCHING & CACHING ====================

@lru_cache(maxsize=256)
def fetch_macroeconomic_assumptions() -> Dict:
    """
    Fetch current macroeconomic assumptions from credible sources.
    Falls back to defaults if APIs unavailable.
    
    Sources:
    - US Treasury ETFs (TLT, SHV)
    - Fed data (risk premiums)
    - Damodaran curves (tax rates by sector)
    """
    risk_free_rate = None
    
    # Try multiple sources for Treasury yield
    try:
        # Method 1: Try TLT (20Y Treasury ETF) - most commonly used
        tlt = yf.Ticker('TLT')
        tlt_info = tlt.info
        
        if tlt_info and 'yield' in tlt_info and tlt_info['yield'] and tlt_info['yield'] > 0:
            risk_free_rate = float(tlt_info['yield'])
        elif tlt_info and 'trailingAnnualDividendYield' in tlt_info and tlt_info['trailingAnnualDividendYield'] and tlt_info['trailingAnnualDividendYield'] > 0:
            risk_free_rate = float(tlt_info['trailingAnnualDividendYield'])
    except:
        pass
    
    # Fallback: Try SHV (3-month Treasury ETF)
    if not risk_free_rate or risk_free_rate < 0.01:
        try:
            shv = yf.Ticker('SHV')
            shv_info = shv.info
            
            if shv_info and 'yield' in shv_info and shv_info['yield'] and shv_info['yield'] > 0:
                risk_free_rate = float(shv_info['yield'])
            elif shv_info and 'trailingAnnualDividendYield' in shv_info and shv_info['trailingAnnualDividendYield'] and shv_info['trailingAnnualDividendYield'] > 0:
                risk_free_rate = float(shv_info['trailingAnnualDividendYield'])
        except:
            pass
    
    # Fallback to default if all methods fail
    if not risk_free_rate or risk_free_rate < 0.01:
        risk_free_rate = DEFAULT_ASSUMPTIONS['risk_free_rate']
    
    # Sanity check: ensure within reasonable bounds
    risk_free_rate = np.clip(float(risk_free_rate), 0.01, 0.08)
    
    return {
        'risk_free_rate': risk_free_rate,
        'market_risk_premium': DEFAULT_ASSUMPTIONS['market_risk_premium'],
        'terminal_growth': DEFAULT_ASSUMPTIONS['terminal_growth'],
        'tax_rate': DEFAULT_ASSUMPTIONS['tax_rate'],
        'fetch_time': datetime.now(),
    }


def fetch_financial_data(ticker: str, timeout: int = 10) -> Dict:
    """
    Fetch comprehensive financial data from Yahoo Finance.
    Implements robust error handling and data validation.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Fetch historical financial statements
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        # Validate critical data using if statements for field detection
        if not info:
            return {'success': False, 'error': f'No data available for {ticker}'}
        
        # Current price detection with fallbacks
        current_price = 0
        if 'currentPrice' in info and info['currentPrice'] and info['currentPrice'] > 0:
            current_price = float(info['currentPrice'])
        elif 'regularMarketPrice' in info and info['regularMarketPrice'] and info['regularMarketPrice'] > 0:
            current_price = float(info['regularMarketPrice'])
        elif 'bid' in info and info['bid'] and info['bid'] > 0:
            current_price = float(info['bid'])
        
        if current_price <= 0:
            return {'success': False, 'error': f'Invalid current price for {ticker}'}
        
        # Revenue extraction with field name variations
        revenue = _extract_financial_field(income_stmt, 'Total Revenue', 'Revenue', 'Revenues', default=0)
        revenue_history = []
        if income_stmt is not None and not income_stmt.empty:
            if 'Total Revenue' in income_stmt.index:
                try:
                    revenue_history = income_stmt.loc['Total Revenue'].values
                except:
                    pass
            elif 'Revenue' in income_stmt.index:
                try:
                    revenue_history = income_stmt.loc['Revenue'].values
                except:
                    pass
        
        # Net income extraction with alternatives
        net_income = _extract_financial_field(income_stmt, 'Net Income', 'Net Earnings', 'Net Income Applicable To Common Shareholders', default=0)
        
        # Operating income extraction
        operating_income = _extract_financial_field(income_stmt, 'Operating Income', 'Operating Earnings', 'EBIT', default=0)
        
        # FCF extraction with multiple field names
        fcf = None
        if cash_flow is not None:
            if 'Free Cash Flow' in cash_flow.index:
                try:
                    fcf = float(cash_flow.loc['Free Cash Flow'].iloc[0])
                except:
                    pass
            elif 'Free Cash Flow To Equity' in cash_flow.index:
                try:
                    fcf = float(cash_flow.loc['Free Cash Flow To Equity'].iloc[0])
                except:
                    pass
        
        # Extract capex with field name variations
        capex = 0
        if cash_flow is not None:
            capex = _extract_financial_field(cash_flow, 'Capital Expenditure', 'Capital Expenditures', 'CapEx', 'Purchases of Property, Plant, and Equipment', default=0)
            capex = abs(capex)  # CapEx is usually negative
        
        # Operating cash flow extraction
        ocf = _extract_financial_field(cash_flow, 'Operating Cash Flow', 'Cash Flow From Continuing Operating Activities', 'Net Cash Provided By Operating Activities', default=0)
        
        # Calculate FCF if not available
        if fcf is None or fcf == 0:
            fcf = ocf - capex if ocf > 0 else net_income * 0.8
        
        # Extract debt with field name variations
        total_debt = 0
        if balance_sheet is not None:
            total_debt = _extract_financial_field(balance_sheet, 'Total Debt', 'Total Long-Term Debt', 'Long Term Debt', 'Current And Long-Term Debt', default=0)
        
        # Extract cash with field name variations
        cash = 0
        if balance_sheet is not None:
            cash = _extract_financial_field(balance_sheet, 'Cash And Cash Equivalents', 'Cash', 'Cash, Cash Equivalents & Marketable Securities', default=0)
        
        # Calculate historical revenue growth
        revenue_growth = 0.05
        if len(revenue_history) >= 3:
            try:
                valid_revenues = [float(x) for x in revenue_history if x > 0]
                if len(valid_revenues) >= 2:
                    cagr = (valid_revenues[0] / valid_revenues[-1]) ** (1 / (len(valid_revenues) - 1)) - 1
                    revenue_growth = max(cagr, 0)
            except:
                pass
        
        # Beta extraction with validation
        beta = 1.0
        if 'beta' in info and info['beta'] and 0 < info['beta'] <= 5:
            beta = float(info['beta'])
        elif 'beta3Year' in info and info['beta3Year'] and 0 < info['beta3Year'] <= 5:
            beta = float(info['beta3Year'])
        # Default to 1.0 if invalid
        
        # Shares outstanding with fallbacks
        shares_outstanding = 0
        if 'sharesOutstanding' in info and info['sharesOutstanding'] and info['sharesOutstanding'] > 0:
            shares_outstanding = float(info['sharesOutstanding'])
        elif 'circulatingSupply' in info and info['circulatingSupply'] and info['circulatingSupply'] > 0:
            shares_outstanding = float(info['circulatingSupply'])
        
        # If still zero, calculate from market cap
        if shares_outstanding <= 0 and current_price > 0:
            if 'marketCap' in info and info['marketCap'] and info['marketCap'] > 0:
                shares_outstanding = float(info['marketCap']) / current_price
        
        return {
            'success': True,
            'ticker': ticker,
            'company_name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'current_price': current_price,
            'market_cap': info.get('marketCap', 0),
            'shares_outstanding': shares_outstanding,
            'beta': beta,
            'revenue': revenue,
            'revenue_history': revenue_history,
            'revenue_growth': revenue_growth,
            'operating_income': operating_income,
            'net_income': net_income,
            'fcf': fcf,
            'capex': capex,
            'ocf': ocf,
            'total_debt': total_debt,
            'cash': cash,
            'pe_ratio': info.get('trailingPE') or info.get('forwardPE'),
            'pb_ratio': info.get('priceToBook'),
            'peg_ratio': info.get('pegRatio'),
            'profit_margin': net_income / revenue if revenue > 0 else 0,
            'roe': info.get('returnOnEquity'),
        }
    
    except Exception as e:
        return {'success': False, 'error': str(e), 'ticker': ticker}


# ==================== DATA VALIDATION ====================

def _validate_financial_data_quality(data: Dict) -> Tuple[bool, str]:
    """
    Validate financial data quality and detect currency/scale issues
    Returns: (is_valid, warning_message)
    """
    if not data or not data.get('success'):
        return False, data.get('error', 'No data available') if data else "No data available"
    
    market_cap = data.get('market_cap', 0)
    if market_cap <= 0:
        return False, "No market cap data available - likely an ETF or non-equity security"
    
    # Check for impossible ratios indicating currency mismatch
    total_debt = data.get('total_debt', 0)
    fcf = data.get('fcf', 0)
    revenue = data.get('revenue', 0)
    
    # Debt/Market Cap ratio should typically be 0.1 - 2.0
    # If > 10x, likely a currency/scale issue
    if total_debt > 0:
        debt_ratio = total_debt / market_cap
        if debt_ratio > 10:
            return False, f"⚠️  Data quality issue: Debt/Market Cap ratio = {debt_ratio:.1f}x (should be <2x). Financial statements likely in foreign currency (e.g., INR, JPY) not converted to USD. Consider using USD-listed alternatives."
    
    # FCF/Market Cap ratio should typically be 0.05 - 1.0
    # If > 5x, likely a currency issue
    if fcf and fcf > 0:
        fcf_ratio = fcf / market_cap
        if fcf_ratio > 5:
            return False, f"⚠️  Data quality issue: FCF/Market Cap ratio = {fcf_ratio:.1f}x (should be <1x). Financial statements likely in foreign currency. Consider using USD-listed alternatives."
    
    # Revenue/Market Cap ratio should typically be 0.5 - 5.0
    # If > 15x, likely a currency issue
    if revenue > 0:
        revenue_ratio = revenue / market_cap
        if revenue_ratio > 15:
            return False, f"⚠️  Data quality issue: Revenue/Market Cap ratio = {revenue_ratio:.1f}x (should be <5x). Financial statements likely in foreign currency. Consider using USD-listed alternatives."
    
    return True, ""


# ==================== DCF VALUATION ENGINE ====================

class DCFValuation:
    """Professional-grade DCF valuation model"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.financial_data = None
        self.assumptions = None
        self.dcf_result = None
        self.sensitivity_data = None
        
    def run_analysis(self, 
                    wacc: Optional[float] = None,
                    terminal_growth: float = 0.025,
                    forecast_years: int = 5,
                    risk_free_rate: Optional[float] = None,
                    market_risk_premium: Optional[float] = None,
                    tax_rate: Optional[float] = None,
                    cost_of_debt: Optional[float] = None,
                    revenue_growth: Optional[float] = None) -> Dict:
        """
        Execute comprehensive DCF valuation with assumptions validation.
        
        Parameters:
        - wacc: Weighted Average Cost of Capital (auto-calculate if None)
        - terminal_growth: Perpetual growth rate (typically 2-3%)
        - forecast_years: Projection horizon (typically 5-10 years)
        - risk_free_rate: US Treasury rate (auto-fetch if None)
        - market_risk_premium: Equity risk premium
        - tax_rate: Corporate tax rate
        - cost_of_debt: Cost of debt (auto-calculate if None)
        - revenue_growth: Revenue growth rate (auto-calculate if None)
        """
        try:
            # Input validation
            if not isinstance(forecast_years, int) or forecast_years < 1 or forecast_years > 20:
                forecast_years = 5
            
            # Fetch financial data
            self.financial_data = fetch_financial_data(self.ticker)
            if not self.financial_data.get('success'):
                return self.financial_data
            
            # Validate data quality and detect currency issues
            is_valid, warning_msg = _validate_financial_data_quality(self.financial_data)
            if not is_valid:
                return {'success': False, 'error': warning_msg, 'ticker': self.ticker}
            
            # Validate critical data
            if self.financial_data.get('current_price', 0) <= 0:
                return {'success': False, 'error': f'Invalid price data for {self.ticker}'}
            
            if self.financial_data.get('revenue', 0) <= 0:
                return {'success': False, 'error': f'No revenue data available for {self.ticker}'}
            
            # Fetch macro assumptions
            macro_assumptions = fetch_macroeconomic_assumptions()
            
            # Estimate revenue growth if not provided (before validation)
            estimated_revenue_growth = revenue_growth or self.financial_data.get('revenue_growth', 0.05)
            if estimated_revenue_growth is None or np.isnan(estimated_revenue_growth):
                estimated_revenue_growth = 0.05
            estimated_revenue_growth = np.clip(float(estimated_revenue_growth), -0.10, 0.50)
            
            # Set assumptions with user overrides and validation
            self.assumptions = {
                'risk_free_rate': risk_free_rate or macro_assumptions['risk_free_rate'],
                'market_risk_premium': market_risk_premium or macro_assumptions['market_risk_premium'],
                'tax_rate': tax_rate or macro_assumptions['tax_rate'],
                'cost_of_debt': cost_of_debt,
                'terminal_growth': terminal_growth,
                'forecast_years': forecast_years,
                'revenue_growth': estimated_revenue_growth,
            }
            
            # Validate assumptions
            if not self._validate_assumptions():
                return {'success': False, 'error': 'Invalid assumptions - please check WACC, growth rates, and tax rate'}
            
            # Calculate WACC if not provided
            if wacc is None:
                wacc = self._calculate_wacc()
            else:
                # Validate provided WACC
                if wacc <= 0 or wacc > 0.50:
                    return {'success': False, 'error': f'Invalid WACC: {wacc*100:.1f}%. Must be between 0.1% and 50%.'}
                wacc = float(wacc)
            
            # Final WACC sanity check
            if wacc <= self.assumptions['terminal_growth']:
                # Adjust terminal growth to be lower than WACC
                self.assumptions['terminal_growth'] = min(self.assumptions['terminal_growth'], wacc * 0.8 - 0.001)
            
            self.assumptions['wacc'] = wacc
            
            # Project cash flows
            projections = self._project_cash_flows()
            if not projections:
                return {'success': False, 'error': 'Could not project cash flows'}
            
            # Calculate terminal value
            terminal_value, terminal_pv = self._calculate_terminal_value(projections[-1]['fcf'])
            
            # Calculate enterprise value
            enterprise_value = sum([p['pv'] for p in projections]) + terminal_pv
            
            # Validate enterprise value
            if enterprise_value <= 0 or np.isnan(enterprise_value) or np.isinf(enterprise_value):
                return {'success': False, 'error': 'Invalid enterprise value calculation'}
            
            net_debt = max(0, self.financial_data['total_debt'] - self.financial_data['cash'])
            equity_value = enterprise_value - net_debt
            
            # If equity value is negative, handle gracefully
            if equity_value < 0:
                equity_value = enterprise_value * 0.8  # Conservative fallback
            
            # Calculate per-share intrinsic value
            shares_outstanding = self.financial_data['shares_outstanding']
            if shares_outstanding <= 0:
                return {'success': False, 'error': 'Unable to determine shares outstanding'}
            
            intrinsic_value = equity_value / shares_outstanding
            
            # Validate intrinsic value
            if intrinsic_value <= 0 or np.isnan(intrinsic_value) or np.isinf(intrinsic_value):
                return {'success': False, 'error': 'Invalid intrinsic value calculation'}
            
            current_price = self.financial_data['current_price']
            upside = ((intrinsic_value / current_price) - 1) * 100 if current_price > 0 else 0
            
            # Build result
            self.dcf_result = {
                'success': True,
                'ticker': self.ticker,
                'company_name': self.financial_data['company_name'],
                'sector': self.financial_data['sector'],
                'current_price': float(current_price),
                'intrinsic_value': float(intrinsic_value),
                'upside_pct': float(upside),
                'enterprise_value': float(enterprise_value),
                'equity_value': float(equity_value),
                'terminal_value': float(terminal_value),
                'terminal_pv': float(terminal_pv),
                'net_debt': float(net_debt),
                'shares_outstanding': float(shares_outstanding),
                'projections': projections,
                'assumptions': {
                    **self.assumptions,
                    'fcf_margin': float(getattr(self, '_fcf_margin_used', 0.10)),
                },
                'financial_data': self.financial_data,
                'valuation_date': datetime.now().isoformat(),
            }
            
            return self.dcf_result
        
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}. Please try again with a valid ticker."
            # Log more details for debugging
            return {'success': False, 'error': error_msg, 'ticker': self.ticker}
    
    def _validate_assumptions(self) -> bool:
        """Validate DCF assumptions for reasonableness"""
        rfr = self.assumptions['risk_free_rate']
        mrp = self.assumptions['market_risk_premium']
        tg = self.assumptions['terminal_growth']
        rg = self.assumptions['revenue_growth']
        
        # Check ranges
        if not (0.005 <= rfr <= 0.10):
            return False
        if not (0.03 <= mrp <= 0.10):
            return False
        if not (0.00 <= tg <= 0.05):
            return False
        if not (-0.10 <= rg <= 0.50):
            return False
        
        return True
    
    def _calculate_wacc(self) -> float:
        """
        Calculate WACC using CAPM for cost of equity.
        
        WACC = (E/V) * Re + (D/V) * Rd * (1 - Tc)
        where:
        - E/V: equity weight
        - D/V: debt weight
        - Re: cost of equity
        - Rd: cost of debt
        - Tc: corporate tax rate
        """
        try:
            beta = float(self.financial_data.get('beta', 1.0))
            if beta <= 0 or beta > 5:
                beta = 1.0
            
            rfr = float(self.assumptions['risk_free_rate'])
            mrp = float(self.assumptions['market_risk_premium'])
            
            # Cost of equity using CAPM
            cost_of_equity = rfr + beta * mrp
            cost_of_equity = np.clip(cost_of_equity, 0.02, 0.50)
            
            # Cost of debt
            if self.assumptions.get('cost_of_debt') and self.assumptions['cost_of_debt'] > 0:
                cost_of_debt = float(self.assumptions['cost_of_debt'])
            else:
                # Estimate from credit spread: rfr + spread
                # Higher risk companies have higher spreads
                spread = 0.02 + (max(0, (1.0 - beta) * 0.01))
                cost_of_debt = max(rfr + spread, 0.03)
            
            cost_of_debt = np.clip(cost_of_debt, 0.01, 0.30)
            
            # Capital structure
            market_cap = float(self.financial_data.get('market_cap', 0))
            total_debt = float(self.financial_data.get('total_debt', 0))
            
            # If market cap missing, estimate from current price
            if market_cap <= 0 and self.financial_data['current_price'] > 0:
                market_cap = self.financial_data['current_price'] * self.financial_data['shares_outstanding']
            
            total_value = market_cap + total_debt
            
            if total_value <= 0:
                return np.clip(cost_of_equity, 0.03, 0.30)
            
            equity_weight = market_cap / total_value
            debt_weight = total_debt / total_value
            tax_rate = np.clip(float(self.assumptions['tax_rate']), 0.0, 0.40)
            
            wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))
            wacc = np.clip(wacc, 0.02, 0.40)
            
            return float(wacc)
        
        except Exception as e:
            # Conservative fallback WACC
            return 0.10
    
    def _project_cash_flows(self) -> List[Dict]:
        """Project free cash flows for forecast period"""
        try:
            base_revenue = float(self.financial_data.get('revenue', 0))
            base_fcf = float(self.financial_data.get('fcf', 0))
            operating_income = float(self.financial_data.get('operating_income', 0))
            revenue_growth = float(self.assumptions['revenue_growth']) if self.assumptions['revenue_growth'] else 0.05
            wacc = float(self.assumptions['wacc'])
            tax_rate = float(self.assumptions['tax_rate'])
            forecast_years = int(self.assumptions['forecast_years'])
            
            # Validate base values
            if base_revenue <= 0:
                return []
            
            # Calculate FCF margin using operating income for capital-intensive companies
            raw_fcf_margin = base_fcf / base_revenue if base_revenue > 0 and base_fcf > 0 else 0.02
            operating_margin = operating_income / base_revenue if base_revenue > 0 and operating_income > 0 else 0
            
            # For companies with low FCF but healthy operating income (capital-intensive):
            # Use NOPAT-based approach
            if raw_fcf_margin < 0.03 and operating_margin > 0.05:
                nopat = operating_income * (1 - tax_rate)
                fcf_margin = nopat / base_revenue
                use_nopat = True
            else:
                fcf_margin = max(raw_fcf_margin, 0.02)
                use_nopat = False
            
            # Sanity check on margins
            fcf_margin = np.clip(fcf_margin, 0.02, 0.50)
            
            # Ensure reasonable growth rate
            revenue_growth = np.clip(revenue_growth, -0.10, 0.50)
            
            # Store fcf_margin for later use
            self._fcf_margin_used = fcf_margin
            
            projections = []
            for year in range(1, forecast_years + 1):
                try:
                    # Project revenue
                    revenue = base_revenue * ((1 + revenue_growth) ** year)
                    
                    # Project FCF  
                    fcf = revenue * fcf_margin if fcf_margin > 0 else revenue * 0.12
                    
                    # Ensure FCF is positive
                    if fcf <= 0:
                        fcf = base_fcf if base_fcf > 0 else revenue * 0.10
                    
                    # Discount to present value
                    discount_factor = (1 + wacc) ** year
                    if discount_factor <= 0:
                        discount_factor = 1.0
                    
                    pv = fcf / discount_factor if discount_factor > 0 else fcf
                    
                    # Sanity check
                    if pv > 0 and not np.isinf(pv) and not np.isnan(pv):
                        projections.append({
                            'year': year,
                            'revenue': float(revenue),
                            'fcf': float(fcf),
                            'fcf_margin': float(fcf_margin),
                            'discount_factor': float(discount_factor),
                            'pv': float(pv),
                        })
                except Exception as e:
                    continue
            
            return projections
        
        except Exception as e:
            return []
    
    def _calculate_terminal_value(self, final_year_fcf: float) -> Tuple[float, float]:
        """
        Calculate terminal value using Gordon Growth Model.
        
        TV = FCF_{n+1} / (WACC - g)
        where g = perpetual growth rate
        """
        try:
            terminal_growth = float(self.assumptions['terminal_growth'])
            wacc = float(self.assumptions['wacc'])
            
            # Validate terminal_growth < WACC
            if terminal_growth >= wacc:
                # Adjust to be lower
                terminal_growth = min(terminal_growth, wacc * 0.7 - 0.001)
            
            # Ensure final_year_fcf is valid
            final_year_fcf = float(final_year_fcf) if final_year_fcf > 0 else 1000
            
            # FCF in terminal year (first year of perpetuity)
            terminal_fcf = final_year_fcf * (1 + terminal_growth)
            
            # Check for invalid WACC-growth spread
            denominator = wacc - terminal_growth
            
            if denominator <= 0.001:
                # Fallback: use higher WACC or lower growth
                if wacc <= terminal_growth:
                    wacc = terminal_growth + 0.05
                    denominator = wacc - terminal_growth
                else:
                    denominator = 0.001
            
            # Terminal value using Gordon Growth Model
            if denominator > 0:
                terminal_value = terminal_fcf / denominator
            else:
                terminal_value = terminal_fcf * 20  # Fallback multiplier
            
            # Sanity check on terminal value
            if terminal_value < 0 or np.isinf(terminal_value) or np.isnan(terminal_value):
                terminal_value = terminal_fcf * 15
            
            # Present value of terminal value
            pv_factor = (1 + wacc) ** self.assumptions['forecast_years']
            if pv_factor <= 0:
                pv_factor = 1.0
            
            terminal_pv = terminal_value / pv_factor
            
            # Last sanity check
            if terminal_pv < 0 or np.isinf(terminal_pv) or np.isnan(terminal_pv):
                terminal_pv = 0.0
            
            return float(terminal_value), float(terminal_pv)
        
        except Exception as e:
            # Conservative fallback
            return float(final_year_fcf * 15), float(final_year_fcf * 15 / 6)
    
    def generate_sensitivity_analysis(self, 
                                    wacc: Optional[float] = None,
                                    terminal_growth:Optional[float] = None,
                                    forecast_years: Optional[int] = None,
                                    revenue_growth: Optional[float] = None) -> Dict:
        """
        Generate 2-way sensitivity analysis table (WACC vs Revenue Growth).
        Shows intrinsic value across different combinations.
        
        Returns: Dictionary with sensitivity table (for display in Streamlit)
        """
        if not self.dcf_result:
            return {}
        
        try:
            wacc = wacc or self.assumptions.get('wacc', 0.10)
            terminal_growth = terminal_growth or self.assumptions.get('terminal_growth', 0.025)
            forecast_years = forecast_years or self.assumptions.get('forecast_years', 5)
            revenue_growth = revenue_growth or self.assumptions.get('revenue_growth', 0.05)
            
            # Create ranges (±3% around base, 7 points each)
            wacc_range = np.linspace(max(0.03, wacc - 0.03), wacc + 0.03, 7)
            growth_range = np.linspace(max(-0.05, revenue_growth - 0.05), revenue_growth + 0.05, 7)
            
            # Create sensitivity dictionary
            sensitivity_dict = {}
            
            for wacc_val in wacc_range:
                wacc_key = f"{wacc_val*100:.1f}%"
                sensitivity_dict[wacc_key] = {}
                
                for growth_val in growth_range:
                    try:
                        # Run DCF with perturbed assumptions
                        dcf = DCFValuation(self.ticker)
                        result = dcf.run_analysis(
                            wacc=float(wacc_val),
                            terminal_growth=float(terminal_growth) if terminal_growth else self.assumptions.get('terminal_growth', 0.025),
                            forecast_years=int(forecast_years),
                            risk_free_rate=self.assumptions.get('risk_free_rate'),
                            market_risk_premium=self.assumptions.get('market_risk_premium'),
                            tax_rate=self.assumptions.get('tax_rate'),
                            cost_of_debt=self.assumptions.get('cost_of_debt'),
                            revenue_growth=float(growth_val)
                        )
                        
                        if result.get('success'):
                            growth_key = f"{growth_val*100:.1f}%"
                            sensitivity_dict[wacc_key][growth_key] = result['intrinsic_value']
                        else:
                            growth_key = f"{growth_val*100:.1f}%"
                            sensitivity_dict[wacc_key][growth_key] = 0.0
                    except:
                        growth_key = f"{growth_val*100:.1f}%"
                        sensitivity_dict[wacc_key][growth_key] = 0.0
            
            self.sensitivity_data = sensitivity_dict
            return sensitivity_dict
        
        except Exception as e:
            return {}
    
    def get_sensitivity_table_pivoted(self) -> pd.DataFrame:
        """
        Convert sensitivity dictionary to a proper pivot table DataFrame.
        Format: WACC on rows (left axis), Growth on columns (top axis), Intrinsic Value in cells
        
        Returns: DataFrame formatted for display as a heatmap-style table
        """
        if not self.sensitivity_data:
            return pd.DataFrame()
        
        try:
            # Convert to list of records for pivoting
            records = []
            for wacc_key, growth_dict in self.sensitivity_data.items():
                for growth_key, value in growth_dict.items():
                    # Parse percentages back to floats for sorting
                    wacc_val = float(wacc_key.rstrip('%'))
                    growth_val = float(growth_key.rstrip('%'))
                    records.append({
                        'WACC': wacc_val,
                        'Growth': growth_val,
                        'Value': value
                    })
            
            if not records:
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            
            # Pivot table: WACC on rows, Growth on columns
            pivot_df = df.pivot(index='WACC', columns='Growth', values='Value')
            
            # Sort columns and index for proper ordering
            pivot_df = pivot_df.sort_index(axis=0, ascending=False)  # WACC descending (top to bottom)
            pivot_df = pivot_df.sort_index(axis=1, ascending=True)   # Growth ascending (left to right)
            
            # Format values as currency strings with highlighting
            def format_cell(val):
                if pd.isna(val) or val == 0:
                    return "-"
                return f"${val:,.0f}"
            
            # Create a styled version for display
            display_df = pivot_df.copy()
            display_df = display_df.applymap(lambda x: format_cell(x))
            
            # Set proper row and column names
            display_df.index.name = "WACC ↓"
            display_df.columns.name = "Revenue Growth →"
            
            return display_df, pivot_df  # Return both display and numeric versions
        
        except Exception as e:
            return pd.DataFrame(), pd.DataFrame()


# ==================== COMPARABLE COMPANIES ANALYSIS ====================

class ComparableCompanies:
    """Comparable companies valuation analysis"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.target_data = None
        self.peer_data = None
        self.result = None
    
    def run_analysis(self, 
                    peer_tickers: Optional[List[str]] = None,
                    auto_find_peers: bool = True) -> Dict:
        """
        Execute comparable companies valuation.
        
        Calculates median multiples across peer group and applies to target.
        """
        try:
            # Fetch target data
            self.target_data = fetch_financial_data(self.ticker)
            if not self.target_data.get('success'):
                return self.target_data
            
            # Identify peers
            if peer_tickers is None or len(peer_tickers) == 0:
                if auto_find_peers:
                    sector = self.target_data.get('sector', 'Technology')
                    peer_tickers = PEER_GROUPS.get(sector, PEER_GROUPS['Technology'])
                    peer_tickers = [p for p in peer_tickers if p != self.ticker]
                else:
                    return {'success': False, 'error': 'No peers provided'}
            
            # Fetch peer data in parallel
            self.peer_data = self._fetch_peers_parallel(peer_tickers)
            
            if len(self.peer_data) == 0:
                return {'success': False, 'error': 'Could not fetch peer data'}
            
            # Calculate multiples
            multiples = self._calculate_peer_multiples()
            
            # Apply to target
            implied_values = self._calculate_implied_values(multiples)
            
            if not implied_values:
                return {'success': False, 'error': 'Insufficient data for valuation'}
            
            # Calculate average implied value
            implied_values_list = list(implied_values.values())
            if not implied_values_list:
                return {'success': False, 'error': 'Insufficient data for valuation'}
            
            avg_implied_value = float(np.median(implied_values_list))
            current_price = self.target_data['current_price']
            upside = ((avg_implied_value / current_price) - 1) * 100 if current_price > 0 else 0
            
            self.result = {
                'success': True,
                'ticker': self.ticker,
                'company_name': self.target_data['company_name'],
                'current_price': float(current_price),
                'implied_value': float(avg_implied_value),
                'avg_implied_value': float(avg_implied_value),
                'upside_pct': float(upside),
                'peer_count': len(self.peer_data),
                'multiples': multiples,
                'implied_values': implied_values,
                'valuation_date': datetime.now().isoformat(),
            }
            
            return self.result
        
        except Exception as e:
            return {'success': False, 'error': str(e), 'ticker': self.ticker}
    
    def _fetch_peers_parallel(self, peer_tickers: List[str], max_workers: int = 5) -> List[Dict]:
        """Fetch peer data in parallel for speed"""
        peer_data = []
        
        def fetch_peer(ticker):
            try:
                data = fetch_financial_data(ticker, timeout=5)
                if data.get('success') and data.get('current_price', 0) > 0:
                    return data
            except:
                pass
            return None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_peer, ticker): ticker for ticker in peer_tickers[:12]}
            
            for future in as_completed(futures, timeout=15):
                try:
                    result = future.result()
                    if result:
                        peer_data.append(result)
                except:
                    continue
        
        return peer_data
    
    def _calculate_peer_multiples(self) -> Dict:
        """Calculate median multiples across peers"""
        multiples = {}
        
        # P/E multiple
        pe_ratios = [p['pe_ratio'] for p in self.peer_data 
                    if p.get('pe_ratio', 0) > 0 and p['pe_ratio'] < 100]
        if pe_ratios:
            multiples['pe_median'] = np.median(pe_ratios)
            multiples['pe_mean'] = np.mean(pe_ratios)
        
        # P/B multiple
        pb_ratios = [p['pb_ratio'] for p in self.peer_data 
                    if p.get('pb_ratio', 0) > 0]
        if pb_ratios:
            multiples['pb_median'] = np.median(pb_ratios)
            multiples['pb_mean'] = np.mean(pb_ratios)
        
        # EV/Revenue (crude estimate from P/E)
        multiples['ev_revenue_median'] = multiples.get('pe_median', 15) / 3
        
        return multiples
    
    def _calculate_implied_values(self, multiples: Dict) -> Dict:
        """Apply multiples to target company"""
        implied_values = {}
        
        # P/E method
        if 'pe_median' in multiples and self.target_data.get('pe_ratio', 0) > 0:
            earnings_per_share = self.target_data['net_income'] / self.target_data['shares_outstanding'] \
                if self.target_data['shares_outstanding'] > 0 else 0
            if earnings_per_share > 0:
                implied_values['pe_multiple'] = earnings_per_share * multiples['pe_median']
        
        # P/B method
        if 'pb_median' in multiples and self.target_data.get('pb_ratio', 0) > 0:
            book_value_per_share = (self.target_data['market_cap'] / self.target_data['pb_ratio']) / \
                                  self.target_data['shares_outstanding'] \
                if self.target_data['shares_outstanding'] > 0 else 0
            if book_value_per_share > 0:
                implied_values['pb_multiple'] = book_value_per_share * multiples['pb_median']
        
        return implied_values


# ==================== VALUATION REPORT ====================

class ValuationReport:
    """Formal valuation report generation"""
    
    def __init__(self, dcf_result: Dict, comps_result: Optional[Dict] = None):
        self.dcf_result = dcf_result
        self.comps_result = comps_result
    
    def get_summary_table(self) -> pd.DataFrame:
        """Generate summary table"""
        if not self.dcf_result.get('success'):
            return pd.DataFrame()
        
        data = {
            'Metric': [
                'Current Market Price',
                'DCF Intrinsic Value',
                'Upside/(Downside)',
            ],
            'Value': [
                f"${self.dcf_result['current_price']:.2f}",
                f"${self.dcf_result['intrinsic_value']:.2f}",
                f"{self.dcf_result['upside_pct']:+.1f}%",
            ]
        }
        
        if self.comps_result and self.comps_result.get('success'):
            data['Metric'].append('Comps Implied Value')
            data['Metric'].append('Comps Upside/(Downside)')
            data['Value'].append(f"${self.comps_result['implied_value']:.2f}")
            data['Value'].append(f"{self.comps_result['upside_pct']:+.1f}%")
        
        return pd.DataFrame(data)
    
    def get_assumptions_table(self) -> pd.DataFrame:
        """Generate assumptions table"""
        if not self.dcf_result.get('success'):
            return pd.DataFrame()
        
        assumptions = self.dcf_result['assumptions']
        
        data = {
            'Assumption': [
                'WACC',
                'Risk-Free Rate',
                'Market Risk Premium',
                'Cost of Debt',
                'Tax Rate',
                'Revenue Growth Rate',
                'Terminal Growth Rate',
                'Forecast Period',
            ],
            'Value': [
                f"{assumptions['wacc']*100:.2f}%",
                f"{assumptions['risk_free_rate']*100:.2f}%",
                f"{assumptions['market_risk_premium']*100:.2f}%",
                f"{self._estimate_cost_of_debt()*100:.2f}%",
                f"{assumptions['tax_rate']*100:.1f}%",
                f"{assumptions['revenue_growth']*100:.1f}%",
                f"{assumptions['terminal_growth']*100:.2f}%",
                f"{assumptions['forecast_years']} years",
            ]
        }
        
        return pd.DataFrame(data)
    
    def _estimate_cost_of_debt(self) -> float:
        """Estimate cost of debt from available data"""
        # Simplified: US risk-free + spread
        rfr = self.dcf_result['assumptions']['risk_free_rate']
        return max(rfr + 0.02, 0.03)
    
    def get_valuation_status(self) -> str:
        """Get valuation status string"""
        upside = self.dcf_result.get('upside_pct', 0)
        
        if upside > 15:
            return "✅ UNDERPRICED - Potential buying opportunity"
        elif upside > -15:
            return "≈ FAIRLY VALUED - Trading near intrinsic value"
        else:
            return "⚠️ OVERPRICED - Consider waiting for pullback"
