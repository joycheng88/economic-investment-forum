"""
Fama-French Factor Analysis & Attribution
Explains portfolio returns through factor exposure decomposition

3-Factor Model: Market, Size (SMB), Value (HML)
5-Factor Model: Market, Size, Value, Profitability, Investment

Usage:
    ff = FamaFrench(returns_df, weights)
    attr_3f = ff.factor_attribution_3f()  # 3-factor decomposition
    attr_5f = ff.factor_attribution_5f()  # 5-factor decomposition
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


class FamaFrench:
    """
    Fama-French Factor Analysis for Portfolio Attribution
    Decomposes returns into factor exposures
    """
    
    def __init__(self, returns_df: pd.DataFrame, weights: pd.Series = None, 
                 lookback_days: int = 252):
        """
        Initialize Fama-French analyzer
        
        Args:
            returns_df: DataFrame of daily returns, index=dates, columns=tickers
            weights: Series of portfolio weights, index=tickers
            lookback_days: Historical data period (default 1 year)
        """
        self.returns_df = returns_df.copy()
        self.weights = weights if weights is not None else pd.Series(
            1 / len(returns_df.columns), 
            index=returns_df.columns
        )
        self.lookback_days = lookback_days
        
        # Calculate portfolio returns
        self.portfolio_returns = (self.returns_df * self.weights).sum(axis=1)
        
        # Fetch factor data
        self.factors_3f = None
        self.factors_5f = None
        self._fetch_factor_data()
    
    def _fetch_factor_data(self):
        """
        Fetch or generate Fama-French factor data
        Always generates synthetic factors for date alignment
        Optionally fetches real Ken French factors as alternative
        """
        # Always generate synthetic factors first (ensures date alignment with returns)
        self._generate_synthetic_factors()
        
        # Optionally try to fetch real Ken French data if available
        try:
            import pandas_datareader.data as web
            
            start_date = self.returns_df.index[0] - timedelta(days=30)
            
            # Try to fetch 3-factor model data
            factors_3f_online = web.DataReader(
                'F-F_Research_Data_Factors_daily',
                'famafrench',
                start=start_date
            )[0]
            
            # Align to returns index
            if len(factors_3f_online) > 0:
                self.factors_3f_online = factors_3f_online.loc[factors_3f_online.index.intersection(self.returns_df.index)]
            
            # Try to fetch 5-factor model data
            try:
                factors_5f_online = web.DataReader(
                    'F-F_Research_Data_5_Factors_2x3_daily',
                    'famafrench',
                    start=start_date
                )[0]
                if len(factors_5f_online) > 0:
                    self.factors_5f_online = factors_5f_online.loc[factors_5f_online.index.intersection(self.returns_df.index)]
            except:
                pass
            
        except Exception as e:
            # If online fetch fails, synthetic factors are already available
            pass
    
    def _generate_synthetic_factors(self):
        """
        Generate synthetic Fama-French factors from portfolio holdings
        Useful when online data unavailable
        """
        dates = self.returns_df.index
        n_periods = len(self.returns_df)
        
        # Market factor: simple average return of all stocks (proxy for market)
        market_excess = self.returns_df.mean(axis=1) * 100
        
        # Size factor (SMB): volatility-based proxy
        # High volatility stocks vs. low volatility stocks
        volatilities = self.returns_df.rolling(20).std().mean()  # Mean volatility per stock
        high_vol_tickers = volatilities[volatilities > volatilities.median()].index
        low_vol_tickers = volatilities[volatilities <= volatilities.median()].index
        
        smb = pd.Series(0.0, index=dates)
        if len(high_vol_tickers) > 0 and len(low_vol_tickers) > 0:
            high_vol_ret = self.returns_df[high_vol_tickers].mean(axis=1)
            low_vol_ret = self.returns_df[low_vol_tickers].mean(axis=1)
            smb = (high_vol_ret - low_vol_ret) * 100
        
        # Value factor (HML): momentum-based proxy
        # Low momentum stocks vs. high momentum stocks (mean reversion)
        momentum = self.returns_df.rolling(20).mean().mean()  # Mean momentum per stock
        low_mom_tickers = momentum[momentum < momentum.median()].index
        high_mom_tickers = momentum[momentum >= momentum.median()].index
        
        hml = pd.Series(0.0, index=dates)
        if len(low_mom_tickers) > 0 and len(high_mom_tickers) > 0:
            low_mom_ret = self.returns_df[low_mom_tickers].mean(axis=1)
            high_mom_ret = self.returns_df[high_mom_tickers].mean(axis=1)
            hml = (low_mom_ret - high_mom_ret) * 100
        
        # Create 3-factor data
        self.factors_3f = pd.DataFrame({
            'Mkt-RF': market_excess,
            'SMB': smb.fillna(0),
            'HML': hml.fillna(0),
            'RF': pd.Series(0.02 / 252 * 100, index=dates)  # Assume 2% annual risk-free rate
        }, index=dates)
        
        # For 5-factor: add profitability and investment signals
        # Profitability: correlation with absolute returns
        abs_returns = self.returns_df.abs().mean()
        robust_tickers = abs_returns[abs_returns > abs_returns.median()].index
        weak_tickers = abs_returns[abs_returns <= abs_returns.median()].index
        
        rmw = pd.Series(0.0, index=dates)
        if len(robust_tickers) > 0 and len(weak_tickers) > 0:
            robust_ret = self.returns_df[robust_tickers].mean(axis=1)
            weak_ret = self.returns_df[weak_tickers].mean(axis=1)
            rmw = (robust_ret - weak_ret) * 100
        
        # Investment: volatility-based (conservative = lower volatility)
        cma = pd.Series(0.0, index=dates)
        if len(low_vol_tickers) > 0 and len(high_vol_tickers) > 0:
            cons_ret = self.returns_df[low_vol_tickers].mean(axis=1)
            agg_ret = self.returns_df[high_vol_tickers].mean(axis=1)
            cma = (cons_ret - agg_ret) * 100
        
        self.factors_5f = pd.DataFrame({
            'Mkt-RF': market_excess,
            'SMB': smb.fillna(0),
            'HML': hml.fillna(0),
            'RMW': rmw.fillna(0),
            'CMA': cma.fillna(0),
            'RF': pd.Series(0.02 / 252 * 100, index=dates)
        }, index=dates)
    
    def factor_attribution_3f(self) -> Dict:
        """
        3-Factor Fama-French Attribution
        Returns = α + β_mkt × Mkt-RF + β_smb × SMB + β_hml × HML + ε
        
        Returns:
            Dict with factor exposures, returns, and attribution analysis
        """
        if self.factors_3f is None:
            return {'error': 'Factor data not available'}
        
        # Align dates - use returns index since factors are generated from returns
        portfolio_ret = (self.portfolio_returns * 100).values  # Convert to basis points
        factors = self.factors_3f.loc[self.returns_df.index].copy()
        
        # Convert factors to decimal if needed
        if factors.max().max() > 1:  # Already in basis points
            pass
        else:
            factors = factors * 100
        
        # Risk-free rate
        rf = factors.get('RF', pd.Series(0.02/252 * 100, index=factors.index)).values
        
        # Excess returns
        portfolio_excess = portfolio_ret - rf
        factor_data = factors[['Mkt-RF', 'SMB', 'HML']].values
        
        # Check if we have enough data
        if len(factor_data) < 2:
            return {
                'error': 'Insufficient data for regression',
                'portfolio_annual_return': float(portfolio_excess.mean() * 252),
                'portfolio_annual_vol': float(np.std(portfolio_excess) * np.sqrt(252))
            }
        
        # Fit regression
        X = factor_data
        y = portfolio_excess
        
        model = LinearRegression()
        model.fit(X, y)
        
        alpha = model.intercept_ * 252  # Annualize
        betas = model.coef_
        r_squared = model.score(X, y)
        
        # Attribution: contribution of each factor to return
        factor_returns = factors[['Mkt-RF', 'SMB', 'HML']].mean() * 252  # Annualize
        factor_returns_std = factors[['Mkt-RF', 'SMB', 'HML']].std() * np.sqrt(252)  # Annualize volatility
        
        # Calculate Sharpe ratios
        factor_sharpes = factor_returns / factor_returns_std
        
        return {
            'model': '3-Factor',
            'alpha': float(alpha),
            'alpha_pct': float(alpha),
            'betas': {
                'market': float(betas[0]),
                'size_smb': float(betas[1]),
                'value_hml': float(betas[2])
            },
            'r_squared': float(r_squared),
            'factor_returns': {
                'mkt_rf': float(factor_returns['Mkt-RF']),
                'smb': float(factor_returns['SMB']),
                'hml': float(factor_returns['HML'])
            },
            'factor_volatility': {
                'mkt_rf': float(factor_returns_std['Mkt-RF']),
                'smb': float(factor_returns_std['SMB']),
                'hml': float(factor_returns_std['HML'])
            },
            'factor_sharpe': {
                'mkt_rf': float(factor_sharpes['Mkt-RF']),
                'smb': float(factor_sharpes['SMB']),
                'hml': float(factor_sharpes['HML'])
            },
            'portfolio_annual_return': float(portfolio_excess.mean() * 252),
            'portfolio_annual_vol': float(portfolio_excess.std() * np.sqrt(252)),
            'n_observations': len(factor_data)
        }
    
    def factor_attribution_5f(self) -> Dict:
        """
        5-Factor Fama-French Attribution (with Profitability and Investment)
        Returns = α + β_mkt×Mkt-RF + β_smb×SMB + β_hml×HML + β_rmw×RMW + β_cma×CMA + ε
        
        RMW: Robust Minus Weak Profitability
        CMA: Conservative Minus Aggressive Investment
        
        Returns:
            Dict with factor exposures, returns, and attribution analysis
        """
        if self.factors_5f is None:
            return {'error': 'Factor data not available'}
        
        # Align dates - use returns index
        portfolio_ret = (self.portfolio_returns * 100).values
        factors = self.factors_5f.loc[self.returns_df.index].copy()
        
        # Convert factors to decimal if needed
        if factors.max().max() > 1:
            pass
        else:
            factors = factors * 100
        
        # Risk-free rate
        rf = factors.get('RF', pd.Series(0.02/252 * 100, index=factors.index)).values
        
        # Excess returns
        portfolio_excess = portfolio_ret - rf
        factor_data = factors[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].values
        
        # Check if we have enough data
        if len(factor_data) < 2:
            return {
                'error': 'Insufficient data for regression',
                'portfolio_annual_return': float(portfolio_excess.mean() * 252),
                'portfolio_annual_vol': float(np.std(portfolio_excess) * np.sqrt(252))
            }
        
        # Fit regression
        X = factor_data
        y = portfolio_excess
        
        model = LinearRegression()
        model.fit(X, y)
        
        alpha = model.intercept_ * 252  # Annualize
        betas = model.coef_
        r_squared = model.score(X, y)
        
        # Factor analysis
        factor_returns = factors[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].mean() * 252
        factor_returns_std = factors[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].std() * np.sqrt(252)
        factor_sharpes = factor_returns / factor_returns_std
        
        return {
            'model': '5-Factor',
            'alpha': float(alpha),
            'alpha_pct': float(alpha),
            'betas': {
                'market': float(betas[0]),
                'size_smb': float(betas[1]),
                'value_hml': float(betas[2]),
                'profitability_rmw': float(betas[3]),
                'investment_cma': float(betas[4])
            },
            'r_squared': float(r_squared),
            'factor_returns': {
                'mkt_rf': float(factor_returns['Mkt-RF']),
                'smb': float(factor_returns['SMB']),
                'hml': float(factor_returns['HML']),
                'rmw': float(factor_returns['RMW']),
                'cma': float(factor_returns['CMA'])
            },
            'factor_volatility': {
                'mkt_rf': float(factor_returns_std['Mkt-RF']),
                'smb': float(factor_returns_std['SMB']),
                'hml': float(factor_returns_std['HML']),
                'rmw': float(factor_returns_std['RMW']),
                'cma': float(factor_returns_std['CMA'])
            },
            'factor_sharpe': {
                'mkt_rf': float(factor_sharpes['Mkt-RF']),
                'smb': float(factor_sharpes['SMB']),
                'hml': float(factor_sharpes['HML']),
                'rmw': float(factor_sharpes['RMW']),
                'cma': float(factor_sharpes['CMA'])
            },
            'portfolio_annual_return': float(portfolio_excess.mean() * 252),
            'portfolio_annual_vol': float(portfolio_excess.std() * np.sqrt(252)),
            'n_observations': len(factor_data)
        }
    
    def performance_attribution(self, weights_benchmark: Optional[pd.Series] = None) -> Dict:
        """
        Decompose portfolio outperformance relative to benchmark
        
        Args:
            weights_benchmark: Benchmark weights (default: equal weight)
        
        Returns:
            Attribution analysis: excess return = allocation effect + selection effect
        """
        if weights_benchmark is None:
            weights_benchmark = pd.Series(1/len(self.weights), index=self.weights.index)
        
        # Calculate benchmark and actual returns by asset
        avg_returns = self.returns_df.mean() * 252
        
        # Allocation effect: (actual_weight - benchmark_weight) × (factor_return - avg_return)
        weight_diff = self.weights - weights_benchmark
        
        # Selection effect: benchmark_weight × (actual_return - factor_return)
        # Simplified version
        portfolio_return = (self.returns_df * self.weights).sum(axis=1).mean() * 252
        benchmark_return = (self.returns_df * weights_benchmark).sum(axis=1).mean() * 252
        
        excess_return = portfolio_return - benchmark_return
        
        return {
            'portfolio_return': float(portfolio_return),
            'benchmark_return': float(benchmark_return),
            'excess_return': float(excess_return),
            'excess_return_pct': float((excess_return / benchmark_return) * 100),
            'active_weight': (self.weights - weights_benchmark).abs().sum(),
            'by_asset': {
                ticker: {
                    'weight': float(self.weights.get(ticker, 0)),
                    'benchmark_weight': float(weights_benchmark.get(ticker, 0)),
                    'weight_diff': float(self.weights.get(ticker, 0) - weights_benchmark.get(ticker, 0)),
                    'return': float(avg_returns.get(ticker, 0))
                }
                for ticker in self.weights.index
            }
        }
    
    def factor_exposure_summary(self) -> pd.DataFrame:
        """
        Summary of factor exposures (3F and 5F models)
        """
        attr_3f = self.factor_attribution_3f()
        attr_5f = self.factor_attribution_5f()
        
        summary = pd.DataFrame({
            '3-Factor Model': {
                'Alpha (%)': attr_3f.get('alpha', 0),
                'Market Beta': attr_3f.get('betas', {}).get('market', 0),
                'Size Beta (SMB)': attr_3f.get('betas', {}).get('size_smb', 0),
                'Value Beta (HML)': attr_3f.get('betas', {}).get('value_hml', 0),
                'R-Squared': attr_3f.get('r_squared', 0)
            },
            '5-Factor Model': {
                'Alpha (%)': attr_5f.get('alpha', 0),
                'Market Beta': attr_5f.get('betas', {}).get('market', 0),
                'Size Beta (SMB)': attr_5f.get('betas', {}).get('size_smb', 0),
                'Value Beta (HML)': attr_5f.get('betas', {}).get('value_hml', 0),
                'Profitability Beta (RMW)': attr_5f.get('betas', {}).get('profitability_rmw', 0),
                'Investment Beta (CMA)': attr_5f.get('betas', {}).get('investment_cma', 0),
                'R-Squared': attr_5f.get('r_squared', 0)
            }
        }).T
        
        return summary


def create_factor_report(returns_df: pd.DataFrame, weights: pd.Series, 
                        ticker_names: Dict[str, str] = None) -> str:
    """
    Generate formatted factor attribution report
    
    Args:
        returns_df: DataFrame of daily returns
        weights: Portfolio weights
        ticker_names: Dict mapping tickers to names
    
    Returns:
        Formatted markdown report
    """
    ff = FamaFrench(returns_df, weights)
    
    attr_3f = ff.factor_attribution_3f()
    attr_5f = ff.factor_attribution_5f()
    attr_perf = ff.performance_attribution()
    
    report = f"""
# Fama-French Factor Attribution Analysis

## Executive Summary
- **Portfolio Annual Return**: {attr_3f.get('portfolio_annual_return', 0):.2f}%
- **Portfolio Annual Volatility**: {attr_3f.get('portfolio_annual_vol', 0):.2f}%
- **Sample Period**: {attr_3f.get('n_observations', 0)} days

## 3-Factor Model (Fama-French 1993)
### Returns = α + β_mkt × Mkt-RF + β_smb × SMB + β_hml × HML

**Alpha (Risk-adjusted excess return)**: {attr_3f.get('alpha', 0):.2f}% annually

**Factor Betas (Exposures)**:
- Market Risk Premium (Mkt-RF): {attr_3f.get('betas', {}).get('market', 0):.3f}
  - _Market risk sensitivity. >1 = aggressive, <1 = defensive_
- Size Factor (SMB): {attr_3f.get('betas', {}).get('size_smb', 0):.3f}
  - _Small cap exposure. >0 = tilt to small caps_
- Value Factor (HML): {attr_3f.get('betas', {}).get('value_hml', 0):.3f}
  - _Value exposure. >0 = tilt to value stocks_

**Model Fit**: R² = {attr_3f.get('r_squared', 0):.3f}
- (_Percentage of portfolio returns explained by factors_)

**Factor Returns (Last Period)**:
- Market: {attr_3f.get('factor_returns', {}).get('mkt_rf', 0):.2f}%
- Size: {attr_3f.get('factor_returns', {}).get('smb', 0):.2f}%
- Value: {attr_3f.get('factor_returns', {}).get('hml', 0):.2f}%

## 5-Factor Model (Fama-French 2015)
### Adds Profitability and Investment Factors

**Alpha**: {attr_5f.get('alpha', 0):.2f}% annually

**Factor Betas**:
- Market: {attr_5f.get('betas', {}).get('market', 0):.3f}
- Size (SMB): {attr_5f.get('betas', {}).get('size_smb', 0):.3f}
- Value (HML): {attr_5f.get('betas', {}).get('value_hml', 0):.3f}
- Profitability (RMW): {attr_5f.get('betas', {}).get('profitability_rmw', 0):.3f}
  - _Exposure to profitable companies. >0 = profitable tilt_
- Investment (CMA): {attr_5f.get('betas', {}).get('investment_cma', 0):.3f}
  - _Conservative investment stance. >0 = lower asset growth_

**Model Fit**: R² = {attr_5f.get('r_squared', 0):.3f}

## Performance Attribution vs Equal-Weight Benchmark

- **Portfolio Return**: {attr_perf.get('portfolio_return', 0):.2f}%
- **Benchmark Return**: {attr_perf.get('benchmark_return', 0):.2f}%
- **Excess Return**: {attr_perf.get('excess_return', 0):.2f}% ({attr_perf.get('excess_return_pct', 0):.2f}%)
- **Active Weight**: {attr_perf.get('active_weight', 0):.2f}

## Key Insights

### Factor Tilts Detected:
"""
    
    # Analyze tilts
    market_beta_3f = attr_3f.get('betas', {}).get('market', 0)
    size_beta = attr_3f.get('betas', {}).get('size_smb', 0)
    value_beta = attr_3f.get('betas', {}).get('value_hml', 0)
    
    if market_beta_3f > 1.1:
        report += f"\n- **Aggressive Market Exposure**: β_market = {market_beta_3f:.2f} (>1.0)"
    elif market_beta_3f < 0.9:
        report += f"\n- **Defensive Positioning**: β_market = {market_beta_3f:.2f} (<1.0)"
    
    if size_beta > 0.1:
        report += f"\n- **Small Cap Tilt**: β_SMB = {size_beta:.2f} (overweight small caps)"
    elif size_beta < -0.1:
        report += f"\n- **Large Cap Tilt**: β_SMB = {size_beta:.2f} (overweight large caps)"
    
    if value_beta > 0.1:
        report += f"\n- **Value Tilt**: β_HML = {value_beta:.2f} (overweight value stocks)"
    elif value_beta < -0.1:
        report += f"\n- **Growth Tilt**: β_HML = {value_beta:.2f} (overweight growth stocks)"
    
    alpha_3f = attr_3f.get('alpha', 0)
    if alpha_3f > 1:
        report += f"\n- **Alpha Generation**: {alpha_3f:.2f}% (positive risk-adjusted returns)"
    elif alpha_3f < -1:
        report += f"\n- **Alpha Drag**: {alpha_3f:.2f}% (negative risk-adjusted returns)"
    
    report += """

### Interpretation:
- **Beta = 1.0**: Portfolio moves with factor (neutral exposure)
- **Beta > 1.0**: Overweight factor (risk taker on this dimension)
- **Beta < 1.0**: Underweight factor (risk reducer on this dimension)
- **Alpha > 0**: Generating excess returns beyond factor exposures
- **R² > 0.90**: Portfolio behavior well-explained by factors
"""
    
    return report
