"""
Black-Litterman Portfolio Optimization

The Black-Litterman model combines market equilibrium returns with investor views
to produce revised expected returns. It addresses several issues with traditional MVO:
- Extreme concentration in a few assets
- Instability to small changes in expected returns
- Difficulty in estimating expected returns

The model outputs posterior expected returns that are less extreme than the prior.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np


@dataclass
class View:
    """A single view on asset returns or relative performance."""
    assets: list  # tickers involved in the view
    expected_return: float  # expected return in the view (annualized)
    confidence: float  # confidence in view, between 0 and 1 (1 = very confident)


@dataclass
class BLConfig:
    """Configuration for Black-Litterman portfolio optimization."""
    risk_aversion: float = 2.0  # λ in market equilibrium
    tau: float = 0.05  # scaling factor for uncertainty in prior (typical: 0.01-0.1)
    risk_free_rate: float = 0.02
    long_only: bool = True
    max_weight: Optional[float] = None
    solver: str = "OSQP"
    views: list = field(default_factory=list)  # List of View objects


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


def market_implied_returns(
    market_weights: pd.Series,
    cov: pd.DataFrame,
    risk_aversion: float = 2.0,
    risk_free_rate: float = 0.02
) -> pd.Series:
    """
    Back out market-implied expected returns from market capitalization weights.
    
    Uses the Sharpe Ratio condition: implied returns = r_f + λ * Σ * w_market
    
    Parameters:
    -----------
    market_weights : pd.Series
        Market capitalization weights (or initial portfolio weights)
    cov : pd.DataFrame
        Covariance matrix (annualized)
    risk_aversion : float
        Market risk aversion parameter
    risk_free_rate : float
        Risk-free rate (annualized)
        
    Returns:
    --------
    pd.Series
        Market-implied expected returns for each asset
    """
    # Align weights with covariance index
    market_weights = market_weights[cov.index]
    
    # Implied excess returns = λ * Σ * w_market
    implied_excess_returns = risk_aversion * (cov @ market_weights)
    
    # Add back risk-free rate
    implied_returns = risk_free_rate + implied_excess_returns
    
    return implied_returns


def construct_view_matrix(views: list, tickers: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct view matrix P and view return vector Q from views.
    
    Parameters:
    -----------
    views : list
        List of View objects
    tickers : list
        All asset tickers
        
    Returns:
    --------
    P : np.ndarray
        K x N view matrix (K: number of views, N: number of assets)
    Q : np.ndarray
        K x 1 vector of view expected returns
    """
    n_views = len(views)
    n_assets = len(tickers)
    
    P = np.zeros((n_views, n_assets))
    Q = np.zeros(n_views)
    
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    
    for i, view in enumerate(views):
        Q[i] = view.expected_return
        for asset in view.assets:
            if asset in ticker_to_idx:
                P[i, ticker_to_idx[asset]] = 1.0 / len(view.assets)  # equal weight within view
    
    return P, Q


def posterior_expected_returns(
    prior_returns: pd.Series,
    cov: pd.DataFrame,
    views: list,
    tau: float = 0.05
) -> pd.Series:
    """
    Compute posterior expected returns by combining prior with investor views.
    
    Uses Bayesian inference:
    μ_posterior = μ_prior + cov * P^T * (P * cov * P^T + Σ_view)^{-1} * (Q - P * μ_prior)
    
    where Σ_view is the uncertainty in views (related to tau and view confidence).
    
    Parameters:
    -----------
    prior_returns : pd.Series
        Prior expected returns (typically from market equilibrium)
    cov : pd.DataFrame
        Covariance matrix (annualized)
    views : list
        List of View objects (constraints on expected returns)
    tau : float
        Uncertainty scaling factor (higher = less confident in views)
        
    Returns:
    --------
    pd.Series
        Posterior expected returns
    """
    if not views:
        return prior_returns
    
    tickers = cov.index.tolist()
    P, Q = construct_view_matrix(views, tickers)
    
    # Uncertainty in views: Σ_view = diag(1 / confidence)
    Sigma_view = np.zeros((len(views), len(views)))
    for i, view in enumerate(views):
        Sigma_view[i, i] = (1.0 - view.confidence) / view.confidence
    
    cov_array = cov.values
    
    # Posterior mean adjustment
    # P Σ P^T + Σ_view
    pcp_t = P @ cov_array @ P.T + Sigma_view
    
    try:
        pcp_t_inv = np.linalg.inv(pcp_t)
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        pcp_t_inv = np.linalg.pinv(pcp_t)
    
    # Adjustment term: Σ P^T (P Σ P^T + Σ_view)^{-1} (Q - P μ_prior)
    view_error = Q - P @ prior_returns.values
    adjustment = cov_array @ P.T @ pcp_t_inv @ view_error
    
    posterior_returns = prior_returns.values + tau * adjustment
    
    return pd.Series(posterior_returns, index=prior_returns.index)


def bl_optimal_weights(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    config: BLConfig = BLConfig()
) -> pd.Series:
    """
    Compute optimal portfolio weights using mean-variance optimization with BL returns.
    
    Objective: max w^T μ - (λ/2) * w^T Σ w
    
    Parameters:
    -----------
    expected_returns : pd.Series
        Expected returns (posterior from Black-Litterman)
    cov : pd.DataFrame
        Covariance matrix
    config : BLConfig
        Configuration
        
    Returns:
    --------
    pd.Series
        Optimal portfolio weights
    """
    if expected_returns is None or expected_returns.empty:
        raise ValueError("expected_returns cannot be None or empty")
    
    if cov is None or cov.empty:
        raise ValueError("Covariance matrix cannot be None or empty")
    
    # Ensure symmetry
    cov = (cov + cov.T) / 2
    
    tickers = cov.index.tolist()
    n = len(tickers)
    
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError("Install cvxpy: pip install cvxpy")
    
    w = cp.Variable(n)
    
    # Objective: maximize return - (λ/2) * variance
    lambda_param = config.risk_aversion
    objective = cp.Minimize(
        -w @ expected_returns.values + (lambda_param / 2) * cp.quad_form(w, cov.values)
    )
    
    # Constraints
    constraints = [cp.sum(w) == 1]
    
    if config.long_only:
        constraints.append(w >= 0)
    
    if config.max_weight is not None:
        constraints.append(w <= config.max_weight)
    
    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=getattr(cp, config.solver), verbose=False)
    
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"Optimization failed with status: {problem.status}")
    
    weights = pd.Series(w.value, index=tickers)
    
    return weights


def get_bl_weights(
    returns: pd.DataFrame,
    market_weights: Optional[pd.Series] = None,
    config: BLConfig = BLConfig()
) -> pd.Series:
    """
    Main function to compute Black-Litterman optimal weights.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical asset returns (used to estimate covariance)
    market_weights : pd.Series, optional
        Market cap weights. If None, uses inverse volatility weighting (realistic market proxy).
    config : BLConfig
        Configuration with views
        
    Returns:
    --------
    pd.Series
        Posterior optimal portfolio weights
    """
    # Estimate covariance
    cov = estimate_covariance(returns, annualize=True)
    tickers = returns.columns.tolist()
    
    # Default market weights: inverse volatility (more realistic than equal weights)
    if market_weights is None:
        # Use inverse volatility as a proxy for market cap weights
        # Assets with lower volatility get higher weight (more stable/liquid)
        volatilities = np.sqrt(np.diag(cov.values))
        inv_vol = 1.0 / volatilities
        market_weights = pd.Series(inv_vol / inv_vol.sum(), index=tickers)
    else:
        market_weights = market_weights[tickers]
    
    # Get market-implied returns as prior
    prior_returns = market_implied_returns(market_weights, cov, config.risk_aversion, config.risk_free_rate)
    
    # Update with investor views to get posterior
    posterior_returns = posterior_expected_returns(prior_returns, cov, config.views, config.tau)
    
    # Optimize with posterior returns
    weights = bl_optimal_weights(posterior_returns, cov, config)
    
    return weights
