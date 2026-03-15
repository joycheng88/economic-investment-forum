"""
Sparse Portfolio Selection (LASSO-type) Optimization

Uses L1 regularization (LASSO) to encourage sparse portfolios (fewer nonzero holdings).
Combines expected return maximization with L1 penalty on weights.
Useful for portfolios with transaction costs or when simplicity is desired.
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


def estimate_expected_returns(returns):
    """
    Estimate expected returns from historical data.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical asset returns (T x N)
    
    Returns:
    --------
    expected_returns : np.ndarray
        Estimated expected returns (annualized)
    """
    return returns.mean() * 252


def lasso_optimal_weights(returns, cov, expected_returns, lasso_penalty=0.01, max_weight=0.15, 
                          risk_aversion=1.0, num_assets_target=10, max_iterations=5):
    """
    Find optimal sparse portfolio weights using cardinality-controlled LASSO (L1 regularization).
    
    FIXED FORMULATION: Uses iterative hard thresholding to enforce true sparsity.
    Note: L1 penalty alone is ineffective under long-only + fully-invested constraints
    (since ||w||_1 = sum(w_i) = 1 is constant). This method uses iterative thresholding
    to select top K assets by weight magnitude, then re-optimizes on the reduced support.
    
    Objective (per iteration): maximize w^T μ - (λ/2) w^T Σ w - γ ||w||_1
    
    where:
    - μ = expected returns
    - Σ = covariance matrix
    - ||w||_1 = L1 norm of weights (soft regularization, hardened by iterative selection)
    - γ = LASSO penalty strength
    - λ = risk aversion coefficient
    - card(w) ≤ K = cardinality constraint (max K non-zero weights)
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns (T x N)
    cov : np.ndarray
        Covariance matrix (N x N)
    expected_returns : np.ndarray
        Expected asset returns
    lasso_penalty : float
        L1 regularization strength (default 0.01)
    max_weight : float
        Maximum weight per asset
    risk_aversion : float
        Risk aversion parameter (default 1.0)
    num_assets_target : int
        Target sparsity: max number of non-zero weights (default 10)
    max_iterations : int
        Max iterations for hard thresholding (default 5)
    
    Returns:
    --------
    weights : np.ndarray
        Optimal sparse portfolio weights (with <= num_assets_target non-zero entries)
    """
    n_assets = len(returns.columns)
    num_assets_target = min(num_assets_target, n_assets)  # Can't exceed available assets
    
    # Objective function: negative of (return - risk - sparsity)
    def objective(w):
        expected_return = np.dot(expected_returns, w)
        risk = 0.5 * risk_aversion * (w @ cov @ w)
        l1_penalty = lasso_penalty * np.sum(np.abs(w))
        return -(expected_return - risk - l1_penalty)
    
    # Constraint: weights sum to 1
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    ]
    
    # Bounds
    bounds = tuple((0, max_weight) for _ in range(n_assets))
    
    # Initial guess: equal weight
    x0 = np.array([1.0 / n_assets] * n_assets)
    
    # Optimize on full set
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if not result.success:
        weights = np.array([1.0 / n_assets] * n_assets)
    else:
        weights = result.x
    
    # ITERATIVE HARD THRESHOLDING: enforce sparsity
    for iteration in range(max_iterations):
        n_nonzero = np.sum(weights > 1e-6)
        
        if n_nonzero <= num_assets_target:
            break  # Already sparse enough
        
        # Hard threshold: keep only top num_assets_target assets by weight magnitude
        sorted_idx = np.argsort(np.abs(weights))
        weights[sorted_idx[:n_assets - num_assets_target]] = 0.0
        weights /= weights.sum()  # Renormalize to maintain sum=1
        
        # Re-optimize on the active set (assets with non-zero bounds)
        active_mask = weights > 1e-6
        active_idx = np.where(active_mask)[0]
        
        if len(active_idx) >= 1:
            # Build constraints and bounds for active assets only
            def obj_active(w_active):
                w_full = np.zeros(n_assets)
                w_full[active_idx] = w_active
                return objective(w_full)
            
            def const_sum_active(w_active):
                return np.sum(w_active) - 1.0
            
            bounds_active = tuple((0, max_weight) for _ in range(len(active_idx)))
            constraints_active = [{'type': 'eq', 'fun': const_sum_active}]
            x0_active = weights[active_idx]
            
            result_active = minimize(
                obj_active,
                x0_active,
                method='SLSQP',
                bounds=bounds_active,
                constraints=constraints_active,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            
            if result_active.success:
                weights = np.zeros(n_assets)
                weights[active_idx] = result_active.x
            # else: keep the thresholded weights from iteration
    
    # Ensure weights sum to 1
    weights = weights / weights.sum()
    
    # Zero out very small weights (near-zero sparsity)
    weights[weights < 1e-4] = 0
    weights = weights / weights.sum()
    
    return weights


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


def get_lasso_weights(returns, max_weight=0.15, lasso_penalty=0.01, risk_aversion=1.0, num_assets_target=10):
    """
    Main interface for LASSO sparse portfolio optimization.
    
    FIXED: Now uses iterative hard thresholding to enforce true cardinality-based sparsity.
    The original L1 penalty was ineffective under long-only + fully-invested constraints
    because ||w||_1 = 1 is constant. This method selects the top K assets by weight
    magnitude and re-optimizes, achieving actual sparsity.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns
    max_weight : float
        Maximum weight per asset
    lasso_penalty : float
        L1 regularization strength (controls smoothness, not primary sparsity mechanism)
    risk_aversion : float
        Risk aversion coefficient
    num_assets_target : int
        Target number of non-zero weights (sparsity control, default 10)
    
    Returns:
    --------
    weights : pd.Series
        Optimal sparse portfolio weights (with ≤ num_assets_target non-zero entries)
    """
    cov = estimate_covariance(returns).values * 252  # Annualize to match expected_returns
    exp_returns = estimate_expected_returns(returns)
    
    weights = lasso_optimal_weights(
        returns,
        cov,
        exp_returns,
        lasso_penalty=lasso_penalty,
        max_weight=max_weight,
        risk_aversion=risk_aversion,
        num_assets_target=num_assets_target,
        max_iterations=5
    )
    
    return pd.Series(weights, index=returns.columns)
