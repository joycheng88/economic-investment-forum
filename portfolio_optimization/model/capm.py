"""
CAPM and Fama-French Portfolio Optimization

Models:
1. CAPM: Uses market risk premium and beta to estimate expected returns
2. Fama-French 3-Factor Model: Extends CAPM with SMB (Size) and HML (Value) factors

Risk-Return Optimization: max w^T μ - λ/2 * w^T Σ w
where μ = expected returns, Σ = covariance, λ = risk aversion parameter
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
from scipy.optimize import minimize


@dataclass
class CAPMConfig:
    """Configuration for CAPM-based portfolio optimization."""
    model_type: str = "capm"  # "capm" or "ff3"
    risk_aversion: float = 2.0  # λ: higher = more conservative
    risk_free_rate: float = 0.02  # assumed annual risk-free rate
    long_only: bool = True
    max_weight: Optional[float] = None
    solver: str = "OSQP"


def get_market_premium(returns: pd.DataFrame) -> float:
    """
    Estimate market risk premium from excess returns.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns (assumed excess returns if rf already subtracted)
        
    Returns:
    --------
    float
        Estimated market risk premium (annualized)
    """
    # Use average of all assets as proxy for market return
    market_return = returns.mean().mean()
    return market_return * 252  # annualize


def estimate_betas(returns: pd.DataFrame, market_returns: pd.Series) -> pd.Series:
    """
    Estimate asset betas relative to market.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        T x N asset returns
    market_returns : pd.Series
        T market returns
        
    Returns:
    --------
    pd.Series
        Beta for each asset
    """
    betas = {}
    market_var = np.var(market_returns, ddof=1)
    
    for ticker in returns.columns:
        cov = np.cov(returns[ticker], market_returns, ddof=1)[0, 1]
        beta = cov / market_var
        betas[ticker] = beta
    
    return pd.Series(betas)


def estimate_expected_returns_capm(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.02,
    market_premium: Optional[float] = None
) -> pd.Series:
    """
    Estimate expected returns using CAPM.
    
    E[r_i] = r_f + β_i * (E[r_m] - r_f)
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical asset returns
    risk_free_rate : float
        Risk-free rate (annualized)
    market_premium : float, optional
        Market risk premium. If None, estimated from data.
        
    Returns:
    --------
    pd.Series
        Expected return for each asset (annualized)
    """
    if market_premium is None:
        market_premium = get_market_premium(returns)
    
    # Use simple average of returns as market proxy
    market_returns = returns.mean(axis=1)
    
    # Estimate betas
    betas = estimate_betas(returns, market_returns)
    
    # CAPM: E[r_i] = r_f + β_i * MRP
    expected_returns = risk_free_rate + betas * market_premium
    
    return expected_returns


def estimate_expected_returns_ff3(
    returns: pd.DataFrame,
    ff_factors: Optional[pd.DataFrame] = None,
    risk_free_rate: float = 0.02
) -> pd.Series:
    """
    Estimate expected returns using Fama-French 3-factor model.
    
    E[r_i] = r_f + β_mkt,i * MKT + β_smb,i * SMB + β_hml,i * HML
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical asset returns
    ff_factors : pd.DataFrame, optional
        Fama-French factors (MKT, SMB, HML). If None, simple defaults used.
    risk_free_rate : float
        Risk-free rate (annualized)
        
    Returns:
    --------
    pd.Series
        Expected return for each asset (annualized)
    """
    # If FF factors not provided, use simple factor proxies
    if ff_factors is None:
        # Create synthetic factors from returns
        mkt_factor = returns.mean(axis=1)  # market proxy
        
        # SMB (small minus big): use smallest vs largest assets
        n = returns.shape[1]
        small_idx = list(range(n // 2))
        large_idx = list(range(n // 2, n))
        smb_factor = returns.iloc[:, small_idx].mean(axis=1) - returns.iloc[:, large_idx].mean(axis=1)
        
        # HML (high minus low): create from returns momentum
        hml_factor = returns.std(axis=1)  # simplified: volatility as proxy
        
        ff_factors = pd.DataFrame({
            'MKT': mkt_factor,
            'SMB': smb_factor,
            'HML': hml_factor
        })
    
    # Estimate factor loadings for each asset
    expected_returns = pd.Series(index=returns.columns, dtype=float)
    
    for ticker in returns.columns:
        # Regress asset returns on factors
        X = ff_factors.values
        X = np.column_stack([np.ones(len(X)), X])  # add intercept
        y = returns[ticker].values
        
        try:
            betas = np.linalg.lstsq(X, y, rcond=None)[0]
        except:
            betas = np.zeros(4)
        
        # Expected return = alpha + β_mkt*E[MKT] + β_smb*E[SMB] + β_hml*E[HML]
        alpha = betas[0]
        factor_expectations = ff_factors.mean()
        expected_ret = alpha + betas[1:] @ factor_expectations.values
        expected_returns[ticker] = expected_ret * 252  # annualize
    
    return expected_returns


def estimate_covariance(returns: pd.DataFrame, annualize: bool = True, periods_per_year: int = 252) -> pd.DataFrame:
    """
    Estimate covariance matrix using Ledoit-Wolf shrinkage.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        T x N asset returns
    annualize : bool
        If True, annualize the covariance
    periods_per_year : int
        Number of periods per year (default 252 for daily data)
        
    Returns:
    --------
    pd.DataFrame
        N x N covariance matrix
    """
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError:
        raise ImportError("Install scikit-learn: pip install scikit-learn")
    
    lw = LedoitWolf().fit(returns.values)
    cov = pd.DataFrame(
        lw.covariance_,
        index=returns.columns,
        columns=returns.columns
    )
    
    if annualize:
        cov = cov * periods_per_year
    
    return cov


def capm_optimal_weights(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    config: CAPMConfig = CAPMConfig()
) -> pd.Series:
    """
    Compute optimal portfolio weights by maximizing the Sharpe ratio.
    
    This finds the tangency portfolio (maximum Sharpe ratio):
    
    max (w^T μ - r_f) / sqrt(w^T Σ w)
    
    subject to: w^T 1 = 1, w ≥ 0 (if long_only), w ≤ max_weight
    
    The Sharpe ratio is invariant to the risk aversion parameter, 
    so we directly optimize it rather than using λ.
    
    Parameters:
    -----------
    expected_returns : pd.Series
        Expected return for each asset (from CAPM)
    cov : pd.DataFrame
        Covariance matrix
    config : CAPMConfig
        Configuration for optimization
        
    Returns:
    --------
    pd.Series
        Portfolio weights that maximize Sharpe ratio (tangency portfolio)
    """
    if expected_returns is None or expected_returns.empty:
        raise ValueError("expected_returns cannot be None or empty")
    
    if cov is None or cov.empty:
        raise ValueError("Covariance matrix cannot be None or empty")
    
    # Ensure symmetry
    cov = (cov + cov.T) / 2
    
    tickers = cov.index.tolist()
    n = len(tickers)
    
    # Convert to numpy for optimization
    mu = expected_returns.values
    sigma = cov.values
    rf = config.risk_free_rate
    
    def negative_sharpe(w):
        """Negative Sharpe ratio (for minimization)"""
        portfolio_return = w @ mu
        portfolio_variance = w @ sigma @ w
        portfolio_std = np.sqrt(portfolio_variance)
        
        if portfolio_std < 1e-10:
            return 1e10
        
        sharpe = (portfolio_return - rf) / portfolio_std
        return -sharpe  # Negative for minimization
    
    # Initial guess: equal weight
    x0 = np.array([1.0 / n] * n)
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    # Bounds: long-only constraints and max weight
    if config.long_only:
        bounds = tuple((0, config.max_weight if config.max_weight else 1.0) for _ in range(n))
    else:
        bounds = tuple((None, None) for _ in range(n))
    
    # Minimize negative Sharpe ratio
    result = minimize(
        negative_sharpe,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if not result.success:
        # Fallback to equal weight
        weights = np.array([1.0 / n] * n)
    else:
        weights = result.x
    
    # Ensure weights sum to 1 (numerical precision)
    weights = weights / weights.sum()
    
    return pd.Series(weights, index=tickers)


def get_capm_weights(
    returns: pd.DataFrame,
    config: CAPMConfig = CAPMConfig(),
    market_premium: Optional[float] = None,
    ff_factors: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Main function to compute CAPM/Fama-French optimal weights.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical asset returns
    config : CAPMConfig
        Configuration for the model
    market_premium : float, optional
        Market risk premium (if None, estimated from data)
    ff_factors : pd.DataFrame, optional
        Fama-French factors (required for FF3 model)
        
    Returns:
    --------
    pd.Series
        Optimal portfolio weights
    """
    # Estimate covariance
    cov = estimate_covariance(returns, annualize=True)
    
    # Estimate expected returns based on model type
    if config.model_type == "capm":
        expected_returns = estimate_expected_returns_capm(returns, config.risk_free_rate, market_premium)
    elif config.model_type == "ff3":
        expected_returns = estimate_expected_returns_ff3(returns, ff_factors, config.risk_free_rate)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
    
    # Optimize
    weights = capm_optimal_weights(expected_returns, cov, config)
    
    return weights
