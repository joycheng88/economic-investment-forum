"""
Global Minimum Variance (GMV) Portfolio Model

The GMV portfolio minimizes portfolio volatility without considering expected returns.
It solves: min w^T Σ w  subject to: Σ w_i = 1, and constraint constraints (long-only, max weight, etc)

This is the most conservative approach, purely based on historical covariance.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class GMVConfig:
    """Configuration for Global Minimum Variance portfolio optimization."""
    long_only: bool = True
    max_weight: Optional[float] = None
    solver: str = "OSQP"  # OSQP, ECOS, SCS
    
    
def estimate_covariance(returns: pd.DataFrame, annualize: bool = True, periods_per_year: int = 252) -> pd.DataFrame:
    """
    Estimate covariance matrix from historical returns using Ledoit-Wolf shrinkage.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        T x N DataFrame of returns (T: time periods, N: assets)
    annualize : bool
        If True, annualize the covariance matrix
    periods_per_year : int
        Number of periods per year (default 252 for daily data)
        
    Returns:
    --------
    pd.DataFrame
        N x N covariance matrix
    """
    if returns is None or returns.empty:
        raise ValueError("returns cannot be None or empty")
    
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError:
        raise ImportError("Install scikit-learn: pip install scikit-learn")
    
    # Use Ledoit-Wolf shrinkage estimator
    lw = LedoitWolf().fit(returns.values)
    cov = pd.DataFrame(
        lw.covariance_,
        index=returns.columns,
        columns=returns.columns
    )
    
    # Annualize if requested
    if annualize:
        cov = cov * periods_per_year
    
    return cov


def gmv_weights(cov: pd.DataFrame, config: GMVConfig = GMVConfig()) -> pd.Series:
    """
    Compute Global Minimum Variance portfolio weights.
    
    Parameters:
    -----------
    cov : pd.DataFrame
        N x N covariance matrix
    config : GMVConfig
        Configuration for optimization constraints and solver
        
    Returns:
    --------
    pd.Series
        Portfolio weights (one weight per asset)
        
    Raises:
    -------
    ValueError
        If covariance matrix is invalid or optimization fails
    """
    if cov is None or cov.empty:
        raise ValueError("Covariance matrix cannot be None or empty")
    
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square (N x N)")
    
    # Ensure symmetry
    cov = (cov + cov.T) / 2
    
    tickers = cov.index.tolist()
    n = len(tickers)
    
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError("Install cvxpy: pip install cvxpy")
    
    w = cp.Variable(n)
    
    # Objective: minimize portfolio variance w^T Σ w
    objective = cp.Minimize(cp.quad_form(w, cov.values))
    
    # Constraints
    constraints = [cp.sum(w) == 1]  # weights sum to 1
    
    if config.long_only:
        constraints.append(w >= 0)  # non-negative weights
    
    if config.max_weight is not None:
        constraints.append(w <= config.max_weight)  # max weight constraint
    
    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=getattr(cp, config.solver), verbose=False)
    
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"Optimization failed with status: {problem.status}")
    
    weights = pd.Series(w.value, index=tickers)
    
    return weights


def portfolio_volatility(weights: pd.Series, cov: pd.DataFrame) -> float:
    """
    Compute portfolio volatility.
    
    Parameters:
    -----------
    weights : pd.Series
        Portfolio weights
    cov : pd.DataFrame
        Covariance matrix
        
    Returns:
    --------
    float
        Portfolio volatility (standard deviation)
    """
    # Align weights with covariance matrix
    weights = weights[cov.index]
    portfolio_var = weights @ cov @ weights
    return np.sqrt(portfolio_var)
