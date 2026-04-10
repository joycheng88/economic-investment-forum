"""
Distributionally Robust Optimization (DRO) for Portfolio Allocation

Implements worst-case portfolio optimization under distributional uncertainty
using Wasserstein ambiguity sets.

Classical optimization assumes: R ~ P̂ (empirical distribution)
DRO assumes: P ∈ U(P̂) where U is an ambiguity set.

Key idea: Optimize for worst-case distribution within Wasserstein ball around
empirical distribution.

Formulation:
    min_w  sup_{P ∈ W_ε(P̂)}  E_P[L(w, R)]

where W_ε(P̂) = {P : W(P, P̂) ≤ ε} is Wasserstein ambiguity set.

References:
- Blanchet, Kang, Murthy (2019): "Robust Wasserstein Profile Inference"
- Esfahani & Kuhn (2018): "Data-driven Distributionally Robust Optimization"
- Mohajerin Esfahani & Kuhn (2018): "Data-driven DRO using Wasserstein metric"
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def estimate_covariance(returns: pd.DataFrame, method: str = 'ledoit_wolf') -> pd.DataFrame:
    """
    Estimate covariance matrix with shrinkage.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns (T x N)
    method : str
        Covariance estimation method ('ledoit_wolf' or 'sample')
    
    Returns:
    --------
    cov : pd.DataFrame
        Covariance matrix (N x N)
    """
    if method == 'ledoit_wolf':
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns).covariance_
        return pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)
    else:
        return returns.cov()


def estimate_expected_returns(returns: pd.DataFrame, method: str = 'sample_mean') -> np.ndarray:
    """
    Estimate expected returns.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns (T x N)
    method : str
        Estimation method ('sample_mean')
    
    Returns:
    --------
    expected_returns : np.ndarray
        Expected returns (annualized)
    """
    if method == 'sample_mean':
        return returns.mean().values * 252
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_wasserstein_distance_approx(
    returns1: np.ndarray,
    returns2: np.ndarray,
    order: int = 2
) -> float:
    """
    Approximate Wasserstein distance between two return distributions.
    
    For univariate distributions with equal mass, W_p distance has closed form.
    For multivariate, we use moment-based approximation.
    
    Parameters:
    -----------
    returns1 : np.ndarray
        First sample (T1 x N)
    returns2 : np.ndarray
        Second sample (T2 x N)
    order : int
        Order of Wasserstein distance (default 2)
    
    Returns:
    --------
    distance : float
        Approximate Wasserstein-2 distance
    """
    # Moment-based approximation for multivariate case
    mu1 = returns1.mean(axis=0)
    mu2 = returns2.mean(axis=0)
    
    # W_2^2(P, Q) ≈ ||μ_P - μ_Q||^2 + trace(Σ_P + Σ_Q - 2(Σ_P^{1/2} Σ_Q Σ_P^{1/2})^{1/2})
    # Simplified: use Frobenius norm of difference
    diff_mean = np.linalg.norm(mu1 - mu2)
    
    if len(returns1) > 1 and len(returns2) > 1:
        cov1 = np.cov(returns1, rowvar=False)
        cov2 = np.cov(returns2, rowvar=False)
        diff_cov = np.linalg.norm(cov1 - cov2, ord='fro')
        distance = np.sqrt(diff_mean**2 + 0.1 * diff_cov)
    else:
        distance = diff_mean
    
    return distance


# ==================== MEAN-VARIANCE DRO ====================

def dro_mean_variance_weights(
    returns: pd.DataFrame,
    cov: np.ndarray,
    expected_returns: np.ndarray,
    epsilon: float = 0.5,
    risk_aversion: float = 1.0,
    max_weight: float = 0.15,
    long_only: bool = True
) -> np.ndarray:
    """
    Distributionally robust mean-variance portfolio optimization.
    
    Worst-case formulation:
        max_w  (μ̂ - ε·||Σ^{1/2} w||)^T w - (λ/2) w^T Σ w
    
    subject to: sum(w) = 1, w_i ≥ 0 (if long_only)
    
    The worst-case expected return is:
        μ_worst = μ̂ - ε · ||Σ^{1/2} w||
    
    This penalizes portfolios with high uncertainty exposure.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns (T x N)
    cov : np.ndarray
        Covariance matrix (N x N)
    expected_returns : np.ndarray
        Expected returns (N,)
    epsilon : float
        Wasserstein radius (ambiguity level, default 0.5)
        Higher ε = more conservative (larger ambiguity set)
    risk_aversion : float
        Risk aversion coefficient (default 1.0)
    max_weight : float
        Maximum weight per asset (default 0.15)
    long_only : bool
        Enforce non-negative weights (default True)
    
    Returns:
    --------
    weights : np.ndarray
        Optimal portfolio weights under DRO
    """
    n_assets = len(returns.columns)
    
    # Precompute Σ^{1/2} for worst-case adjustment
    try:
        sqrt_cov = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # If not positive definite, use eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-8)  # Ensure positive
        sqrt_cov = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    
    # Objective: maximize worst-case utility
    # U(w) = (μ̂ - ε·||Σ^{1/2} w||)^T w - (λ/2) w^T Σ w
    def objective(w):
        # Worst-case return adjustment
        uncertainty_penalty = epsilon * np.linalg.norm(sqrt_cov @ w)
        worst_case_return = expected_returns @ w - uncertainty_penalty
        
        # Risk term
        risk = 0.5 * risk_aversion * (w @ cov @ w)
        
        # Negative because we minimize
        return -(worst_case_return - risk)
    
    # Gradient (for faster optimization)
    def gradient(w):
        Sigma_w = cov @ w
        sqrt_Sigma_w = sqrt_cov @ w
        norm_sqrt = np.linalg.norm(sqrt_Sigma_w) + 1e-8
        
        # Gradient of worst-case return
        grad_worst_return = expected_returns - epsilon * (sqrt_cov.T @ sqrt_Sigma_w) / norm_sqrt
        
        # Gradient of risk term
        grad_risk = risk_aversion * Sigma_w
        
        return -(grad_worst_return - grad_risk)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Budget constraint
    ]
    
    # Bounds
    if long_only:
        bounds = tuple((0, max_weight) for _ in range(n_assets))
    else:
        bounds = tuple((-max_weight, max_weight) for _ in range(n_assets))
    
    # Initial guess: equal weight
    x0 = np.ones(n_assets) / n_assets
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if not result.success:
        warnings.warn(f"DRO optimization did not converge: {result.message}. Using equal weights.")
        weights = np.ones(n_assets) / n_assets
    else:
        weights = result.x
    
    # Ensure weights sum to 1
    weights = weights / weights.sum()
    
    return weights


# ==================== CVAR-DRO ====================

def dro_cvar_weights(
    returns: pd.DataFrame,
    epsilon: float = 0.3,
    alpha: float = 0.05,
    max_weight: float = 0.15,
    long_only: bool = True,
    n_scenarios: int = 1000
) -> np.ndarray:
    """
    Distributionally robust CVaR optimization.
    
    Formulation:
        min_{w, η}  η + (1/(1-α)) · sup_{P ∈ W_ε} E_P[(L(w,R) - η)_+]
    
    where L(w, R) = -w^T R is the portfolio loss.
    
    This is approximated using sample average approximation (SAA) with
    worst-case scenario generation.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns (T x N)
    epsilon : float
        Wasserstein radius (ambiguity level, default 0.3)
    alpha : float
        CVaR confidence level (default 0.05 for 95% CVaR)
    max_weight : float
        Maximum weight per asset (default 0.15)
    long_only : bool
        Enforce non-negative weights (default True)
    n_scenarios : int
        Number of scenarios for SAA (default 1000)
    
    Returns:
    --------
    weights : np.ndarray
        Optimal portfolio weights under CVaR-DRO
    """
    n_assets = returns.shape[1]
    T = len(returns)
    
    # Generate perturbed scenarios (worst-case within Wasserstein ball)
    # Simple approach: add Gaussian noise proportional to epsilon
    scenarios = returns.values.copy()
    
    # Add worst-case perturbations
    if epsilon > 0:
        cov_empirical = np.cov(returns.values, rowvar=False)
        noise_std = epsilon * np.sqrt(np.diag(cov_empirical))
        noise = np.random.randn(n_scenarios, n_assets) * noise_std
        
        # Worst-case: shift scenarios downward (conservative)
        worst_case_scenarios = scenarios[np.random.choice(T, n_scenarios)] - np.abs(noise)
    else:
        worst_case_scenarios = scenarios[np.random.choice(T, n_scenarios)]
    
    # Decision variables: [w (N), η (1), z (n_scenarios)]
    # Objective: η + (1/(1-α)) * sum(z_i) / n_scenarios
    n_vars = n_assets + 1 + n_scenarios
    
    def objective(x):
        w = x[:n_assets]
        eta = x[n_assets]
        z = x[n_assets + 1:]
        
        return eta + (1.0 / (1.0 - alpha)) * np.mean(z)
    
    # Constraints
    constraints = []
    
    # Budget constraint: sum(w) = 1
    constraints.append({
        'type': 'eq',
        'fun': lambda x: np.sum(x[:n_assets]) - 1.0
    })
    
    # CVaR auxiliary variables: z_i >= -(w^T R_i) - η
    # z_i >= 0
    for i in range(n_scenarios):
        def make_cvar_constraint(scenario_idx):
            def constraint(x):
                w = x[:n_assets]
                eta = x[n_assets]
                z_i = x[n_assets + 1 + scenario_idx]
                portfolio_loss = -np.dot(w, worst_case_scenarios[scenario_idx])
                return z_i - max(portfolio_loss - eta, 0)
            return constraint
        
        # Simplified: z_i >= loss - eta (implicitly z_i >= 0 from bounds)
    
    # Bounds
    bounds = []
    
    # Weights
    if long_only:
        bounds += [(0, max_weight) for _ in range(n_assets)]
    else:
        bounds += [(-max_weight, max_weight) for _ in range(n_assets)]
    
    # η (VaR): unbounded
    bounds.append((None, None))
    
    # z_i: non-negative
    bounds += [(0, None) for _ in range(n_scenarios)]
    
    # Initial guess
    x0 = np.zeros(n_vars)
    x0[:n_assets] = 1.0 / n_assets  # Equal weights
    x0[n_assets] = -0.01  # Initial VaR estimate
    
    # Optimize (using SLSQP or trust-constr)
    # For large n_scenarios, this becomes computationally expensive
    # We use a simpler approximation: standard CVaR on perturbed scenarios
    
    # Simplified approach: use standard CVaR formulation on worst-case scenarios
    from scipy.optimize import linprog
    
    # LP formulation: min η + (1/(1-α))·(1/K)·sum(z_i)
    # Variables: [w (N), η (1), z (K)]
    # Subject to: z_i >= -R_i^T w - η, z_i >= 0, sum(w)=1, bounds on w
    
    K = n_scenarios
    
    # Objective coefficients: [0...0 (N), 1 (η), (1/(1-α))·(1/K)...] 
    c = np.zeros(n_assets + 1 + K)
    c[n_assets] = 1.0  # η coefficient
    c[n_assets + 1:] = 1.0 / ((1.0 - alpha) * K)  # z coefficients
    
    # Inequality constraints: -z_i + R_i^T w + η <= 0
    # Rewrite as: -η - z_i + sum(w_j * (-R_ij)) <= 0
    A_ub = []
    b_ub = []
    
    for i in range(K):
        row = np.zeros(n_assets + 1 + K)
        row[:n_assets] = worst_case_scenarios[i]  # R_i^T w
        row[n_assets] = 1.0  # η
        row[n_assets + 1 + i] = -1.0  # -z_i
        A_ub.append(row)
        b_ub.append(0.0)
    
    # Equality constraint: sum(w) = 1
    A_eq = np.zeros((1, n_assets + 1 + K))
    A_eq[0, :n_assets] = 1.0
    b_eq = np.array([1.0])
    
    # Bounds (linprog format)
    if long_only:
        bounds_lp = [(0, max_weight) for _ in range(n_assets)]
    else:
        bounds_lp = [(-max_weight, max_weight) for _ in range(n_assets)]
    
    bounds_lp += [(None, None)]  # η unbounded
    bounds_lp += [(0, None) for _ in range(K)]  # z_i >= 0
    
    # Solve LP
    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds_lp,
        method='highs'
    )
    
    if not result.success:
        warnings.warn(f"CVaR-DRO LP did not converge. Using equal weights.")
        weights = np.ones(n_assets) / n_assets
    else:
        weights = result.x[:n_assets]
        weights = weights / weights.sum()  # Renormalize
    
    return weights


# ==================== MAIN INTERFACE ====================

def get_dro_weights(
    returns: pd.DataFrame,
    method: str = 'mean_variance',
    epsilon: float = 0.5,
    risk_aversion: float = 1.0,
    alpha: float = 0.05,
    max_weight: float = 0.15,
    long_only: bool = True
) -> pd.Series:
    """
    Get optimal portfolio weights using Distributionally Robust Optimization.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical asset returns (T x N)
    method : str
        DRO method: 'mean_variance' or 'cvar' (default 'mean_variance')
    epsilon : float
        Wasserstein radius (ambiguity level)
        - For mean_variance: 0.3-0.7 typical (default 0.5)
        - For cvar: 0.1-0.5 typical (default in function)
        Higher ε → more conservative
    risk_aversion : float
        Risk aversion (only for mean_variance, default 1.0)
    alpha : float
        CVaR confidence level (only for cvar, default 0.05)
    max_weight : float
        Maximum weight per asset (default 0.15)
    long_only : bool
        Enforce non-negative weights (default True)
    
    Returns:
    --------
    weights : pd.Series
        Optimal portfolio weights
    """
    if method == 'mean_variance':
        # Mean-variance DRO
        cov = estimate_covariance(returns).values * 252  # Annualize
        expected_returns = estimate_expected_returns(returns)
        
        weights = dro_mean_variance_weights(
            returns,
            cov,
            expected_returns,
            epsilon=epsilon,
            risk_aversion=risk_aversion,
            max_weight=max_weight,
            long_only=long_only
        )
    
    elif method == 'cvar':
        # CVaR-DRO
        weights = dro_cvar_weights(
            returns,
            epsilon=epsilon,
            alpha=alpha,
            max_weight=max_weight,
            long_only=long_only,
            n_scenarios=500  # Reduced for computational efficiency
        )
    
    else:
        raise ValueError(f"Unknown DRO method: {method}. Choose 'mean_variance' or 'cvar'.")
    
    return pd.Series(weights, index=returns.columns)


def compute_dro_cv_scores(returns, n_splits=5, epsilon=1.0, risk_aversion=1.0):
    """
    Compute K-fold cross-validation scores for DRO model.
    
    Diagnoses overfitting by comparing train vs test performance.
    High train_score - test_score indicates overfitting to training distribution.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns
    n_splits : int
        Number of CV folds
    epsilon : float
        Wasserstein radius (ambiguity level)
    risk_aversion : float
        Risk aversion coefficient
    
    Returns:
    --------
    cv_results : dict
        CV scores and diagnostics
    """
    from model.cross_validation import compute_kfold_cv_scores, diagnose_overfitting
    
    cv_results = compute_kfold_cv_scores(
        returns,
        get_dro_weights,
        model_name="DRO",
        n_splits=n_splits,
        risk_free_rate=0.02,
        method='mean_variance',
        epsilon=epsilon,
        risk_aversion=risk_aversion,
        alpha=0.05,
        max_weight=0.15,
        long_only=True
    )
    
    # Add diagnosis
    diagnosis = diagnose_overfitting(cv_results)
    cv_results['diagnosis'] = diagnosis
    
    return cv_results


def get_dro_weights_with_cv(
    returns,
    method: str = 'mean_variance',
    epsilon: float = None,
    risk_aversion: float = 1.0,
    max_weight: float = 0.15,
    cv_diagnose: bool = True,
    n_cv_splits: int = 5,
    auto_adjust_epsilon: bool = False
):
    """
    Get DRO weights with optional cross-validation diagnostics.
    
    Can auto-adjust epsilon based on CV overfitting diagnosis.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns
    method : str
        'mean_variance' or 'cvar'
    epsilon : float, optional
        Wasserstein radius. If None, uses default (1.0 for mean-variance, 0.3 for cvar)
    risk_aversion : float
        Risk aversion
    max_weight : float
        Max weight per asset
    cv_diagnose : bool
        If True, compute CV diagnostics
    n_cv_splits : int
        Number of CV folds
    auto_adjust_epsilon : bool
        If True, increase epsilon if overfitting detected
    
    Returns:
    --------
    weights : pd.Series
        Portfolio weights
    cv_results : dict, optional
        If cv_diagnose=True, includes CV scores and overfitting diagnosis
    """
    if epsilon is None:
        epsilon = 1.0 if method == 'mean_variance' else 0.3
    
    # Get weights
    weights = get_dro_weights(
        returns,
        method=method,
        epsilon=epsilon,
        risk_aversion=risk_aversion,
        max_weight=max_weight,
        long_only=True
    )
    
    if cv_diagnose:
        try:
            cv_results = compute_dro_cv_scores(
                returns,
                n_splits=n_cv_splits,
                epsilon=epsilon,
                risk_aversion=risk_aversion
            )
            
            # Auto-adjust if severe overfitting detected
            if auto_adjust_epsilon and cv_results['diagnosis']['severity'] in ['severe', 'moderate']:
                # Increase epsilon for more robustness
                new_epsilon = epsilon * 1.5
                weights = get_dro_weights(
                    returns,
                    method=method,
                    epsilon=new_epsilon,
                    risk_aversion=risk_aversion,
                    max_weight=max_weight,
                    long_only=True
                )
                cv_results['note'] = f"Epsilon increased from {epsilon:.2f} to {new_epsilon:.2f} due to overfitting"
            
            return weights, cv_results
        except Exception as e:
            import warnings
            warnings.warn(f"CV diagnostics failed: {e}")
            return weights, None
    
    return weights

def compute_worst_case_return(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov: np.ndarray,
    epsilon: float
) -> float:
    """
    Compute worst-case expected return under DRO.
    
    Formula:
        μ_worst = μ̂^T w - ε · ||Σ^{1/2} w||
    
    Parameters:
    -----------
    weights : np.ndarray
        Portfolio weights
    expected_returns : np.ndarray
        Expected returns
    cov : np.ndarray
        Covariance matrix
    epsilon : float
        Wasserstein radius
    
    Returns:
    --------
    worst_case_return : float
        Worst-case expected return
    """
    nominal_return = expected_returns @ weights
    
    # Compute ||Σ^{1/2} w||
    try:
        sqrt_cov = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        sqrt_cov = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    
    uncertainty_exposure = np.linalg.norm(sqrt_cov @ weights)
    
    worst_case_return = nominal_return - epsilon * uncertainty_exposure
    
    return worst_case_return


def dro_sensitivity_analysis(
    returns: pd.DataFrame,
    epsilon_range: np.ndarray = np.linspace(0.0, 1.0, 11),
    max_weight: float = 0.15
) -> pd.DataFrame:
    """
    Analyze sensitivity of DRO weights to epsilon (ambiguity level).
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns
    epsilon_range : np.ndarray
        Range of epsilon values to test (default 0.0 to 1.0)
    max_weight : float
        Maximum weight per asset
    
    Returns:
    --------
    sensitivity : pd.DataFrame
        DataFrame with columns: epsilon, worst_case_return, volatility, max_weight_value
    """
    results = []
    
    cov = estimate_covariance(returns).values * 252
    expected_returns = estimate_expected_returns(returns)
    
    for eps in epsilon_range:
        weights = dro_mean_variance_weights(
            returns,
            cov,
            expected_returns,
            epsilon=eps,
            risk_aversion=1.0,
            max_weight=max_weight,
            long_only=True
        )
        
        nominal_return = expected_returns @ weights
        worst_case = compute_worst_case_return(weights, expected_returns, cov, eps)
        volatility = np.sqrt(weights @ cov @ weights)
        max_w = weights.max()
        
        results.append({
            'epsilon': eps,
            'nominal_return': nominal_return,
            'worst_case_return': worst_case,
            'volatility': volatility,
            'max_weight': max_w,
            'num_nonzero': np.sum(weights > 1e-4)
        })
    
    return pd.DataFrame(results)
