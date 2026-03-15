"""
Conditional Value at Risk (CVaR) / Expected Shortfall Optimization

Minimizes the expected shortfall (average loss beyond VaR) rather than variance.
More robust to extreme losses and tail risk than traditional mean-variance.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def estimate_covariance(returns, method='ledoit_wolf'):
    """
    Estimate covariance matrix with Ledoit-Wolf shrinkage.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns (T x N)
    method : str
        Shrinkage method ('ledoit_wolf')
    
    Returns:
    --------
    cov : np.ndarray
        Covariance matrix (N x N)
    """
    lw = LedoitWolf()
    cov, _ = lw.fit(returns).covariance_, lw.shrinkage_
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def calculate_cvar(returns, weights, alpha=0.05):
    """
    Calculate Conditional Value at Risk (CVaR) at confidence level (1-alpha).
    
    Parameters:
    -----------
    returns : np.ndarray or pd.Series
        Portfolio returns (daily returns)
    weights : np.ndarray
        Portfolio weights
    alpha : float
        Confidence level (e.g., 0.05 for 95% CVaR)
    
    Returns:
    --------
    cvar : float
        Conditional Value at Risk
    """
    portfolio_returns = np.asarray(returns) @ weights
    var = np.percentile(portfolio_returns, alpha * 100)
    cvar = np.mean(portfolio_returns[portfolio_returns <= var])
    return cvar


def portfolio_cvar(weights, returns, alpha=0.05):
    """
    Calculate negative CVaR (for minimization).
    
    Parameters:
    -----------
    weights : np.ndarray
        Portfolio weights
    returns : np.ndarray
        Asset returns (T x N)
    alpha : float
        Confidence level
    
    Returns:
    --------
    cvar : float
        Negative CVaR (to minimize)
    """
    portfolio_returns = returns @ weights
    var = np.percentile(portfolio_returns, alpha * 100)
    cvar = np.mean(portfolio_returns[portfolio_returns <= var])
    return -cvar  # Negative because we minimize


def cvar_optimal_weights(returns, alpha=0.05, max_weight=0.15):
    """
    Find optimal portfolio weights minimizing CVaR.
    
    Also minimizes a small amount of variance to encourage diversification.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns (T x N)
    alpha : float
        Confidence level for CVaR (default 0.05 = 95% CVaR)
    max_weight : float
        Maximum weight per asset
    
    Returns:
    --------
    weights : pd.Series
        Optimal portfolio weights
    """
    n_assets = len(returns.columns)
    returns_array = returns.values
    
    # Estimate covariance for diversification penalty
    cov = estimate_covariance(returns).values
    
    # Objective: minimize -CVaR (i.e., maximize CVaR)
    # Add small variance penalty for diversification
    def objective(w):
        cvar_loss = portfolio_cvar(w, returns_array, alpha)
        var_penalty = 0.01 * w @ cov @ w  # Small diversification penalty
        return cvar_loss + var_penalty
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Sum to 1
    ]
    
    # Bounds
    bounds = tuple((0, max_weight) for _ in range(n_assets))
    
    # Initial guess: equal weight
    x0 = np.array([1.0 / n_assets] * n_assets)
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if not result.success:
        # Fallback to equal weight
        weights = np.array([1.0 / n_assets] * n_assets)
    else:
        weights = result.x
    
    # Ensure weights sum to 1
    weights = weights / weights.sum()
    
    return pd.Series(weights, index=returns.columns)


def portfolio_volatility(weights, cov):
    """
    Calculate portfolio volatility.
    
    Parameters:
    -----------
    weights : np.ndarray or pd.Series
        Portfolio weights
    cov : np.ndarray or pd.DataFrame
        Covariance matrix
    
    Returns:
    --------
    volatility : float
        Annualized portfolio volatility (assuming 252 trading days)
    """
    if isinstance(weights, pd.Series):
        weights = weights.values
    if isinstance(cov, pd.DataFrame):
        cov = cov.values
    
    variance = weights @ cov @ weights
    volatility = np.sqrt(variance * 252)
    return volatility


def get_cvar_weights(returns, max_weight=0.15, alpha=0.05):
    """
    Main interface for CVaR optimization.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns
    max_weight : float
        Maximum weight per asset
    alpha : float
        Confidence level for CVaR
    
    Returns:
    --------
    weights : pd.Series
        Optimal portfolio weights
    """
    weights = cvar_optimal_weights(returns, alpha=alpha, max_weight=max_weight)
    return weights
