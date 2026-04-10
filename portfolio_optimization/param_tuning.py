"""
Walk-Forward Parameter Tuning for Portfolio Optimization Models

Implements validation-based hyperparameter selection to avoid look-ahead bias
and improve out-of-sample performance.

Methodology:
- Split lookback window into train (60%) and validation (40%)
- Grid search over parameter space on training set
- Evaluate each candidate on validation set using Sharpe ratio
- Select best parameters and apply to out-of-sample test period

This prevents overfitting to the full lookback period and makes results more defensible.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
import warnings
warnings.filterwarnings('ignore')


# ==================== PARAMETER GRIDS ====================

DEFAULT_PARAM_GRIDS = {
    'GMV': {
        'solver': ['OSQP', 'ECOS'],
    },
    
    'CAPM': {
        'risk_aversion': [1.0, 2.0, 3.0, 5.0],
    },
    
    'BL': {
        'tau': [0.01, 0.025, 0.05, 0.1, 0.2],
        'risk_aversion': [1.0, 2.0, 3.0],
    },
    
    'HRP': {
        'linkage_method': ['single', 'average', 'complete', 'ward'],
    },
    
    'CVaR': {
        'alpha': [0.01, 0.025, 0.05, 0.10, 0.15],
    },
    
    'LASSO': {
        'num_assets_target': [5, 6, 7, 8, 10],  # Increased range to reduce concentration risk, more diversified
        'lasso_penalty': [0.01, 0.02, 0.05, 0.1],  # Increased penalty for stronger regularization
    },
    
    'RL': {
        'n_epochs': [3, 5, 8],
        'actor_lr': [0.0005, 0.001, 0.002],
        'gamma': [0.95, 0.99],
    },
    
    'DRO': {
        'epsilon': [0.7, 0.9, 1.1, 1.3, 1.5],  # INCREASED from [0.2,0.3,0.5,0.7,1.0] - more robust to uncertainty
        'risk_aversion': [0.5, 1.0, 2.0, 3.0],  # Extended for flexibility
    }
}


def split_train_validation(
    returns: pd.DataFrame,
    train_ratio: float = 0.6
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split returns into train and validation sets.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns (T x N)
    train_ratio : float
        Fraction of data for training (default 0.6)
    
    Returns:
    --------
    train_returns : pd.DataFrame
        Training returns
    val_returns : pd.DataFrame
        Validation returns
    """
    split_idx = int(len(returns) * train_ratio)
    train_returns = returns.iloc[:split_idx]
    val_returns = returns.iloc[split_idx:]
    
    return train_returns, val_returns


def evaluate_portfolio_on_validation(
    weights: pd.Series,
    val_returns: pd.DataFrame,
    risk_free_rate: float = 0.02
) -> float:
    """
    Evaluate portfolio performance on validation set.
    
    Parameters:
    -----------
    weights : pd.Series
        Portfolio weights
    val_returns : pd.DataFrame
        Validation returns
    risk_free_rate : float
        Annual risk-free rate
    
    Returns:
    --------
    sharpe_ratio : float
        Annualized Sharpe ratio on validation set
    """
    if weights is None or val_returns is None or len(val_returns) == 0:
        return -np.inf
    
    # Align weights to validation returns
    aligned_weights = weights.reindex(val_returns.columns, fill_value=0.0)
    aligned_weights = aligned_weights / aligned_weights.sum() if aligned_weights.sum() > 0 else aligned_weights
    
    # Compute portfolio returns
    port_returns = (val_returns @ aligned_weights).dropna()
    
    if len(port_returns) < 2:
        return -np.inf
    
    # Sharpe ratio
    mean_ret = port_returns.mean() * 252
    std_ret = port_returns.std() * np.sqrt(252)
    
    if std_ret == 0:
        return -np.inf
    
    sharpe = (mean_ret - risk_free_rate) / std_ret
    
    return sharpe


# ==================== MODEL-SPECIFIC TUNING ====================

def tune_gmv_params(
    train_returns: pd.DataFrame,
    val_returns: pd.DataFrame,
    max_weight: float,
    risk_free_rate: float,
    param_grid: Dict = None
) -> Dict:
    """Tune GMV parameters."""
    from model.gmv import estimate_covariance, gmv_weights
    
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRIDS['GMV']
    
    best_params = {}
    best_sharpe = -np.inf
    
    cov = estimate_covariance(train_returns, annualize=True)
    
    for solver in param_grid.get('solver', ['OSQP']):
        try:
            cfg = type('C', (), {'long_only': True, 'max_weight': max_weight, 'solver': solver})()
            weights = gmv_weights(cov, cfg)
            sharpe = evaluate_portfolio_on_validation(weights, val_returns, risk_free_rate)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {'solver': solver}
        except:
            continue
    
    return best_params if best_params else {'solver': 'OSQP'}


def tune_capm_params(
    train_returns: pd.DataFrame,
    val_returns: pd.DataFrame,
    max_weight: float,
    risk_free_rate: float,
    param_grid: Dict = None
) -> Dict:
    """Tune CAPM parameters."""
    from model.capm import get_capm_weights, CAPMConfig
    
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRIDS['CAPM']
    
    best_params = {}
    best_sharpe = -np.inf
    
    for risk_aversion in param_grid.get('risk_aversion', [2.0]):
        try:
            cfg = CAPMConfig(
                model_type="capm",
                risk_aversion=risk_aversion,
                risk_free_rate=risk_free_rate,
                long_only=True,
                max_weight=max_weight
            )
            weights = get_capm_weights(train_returns, cfg)
            sharpe = evaluate_portfolio_on_validation(weights, val_returns, risk_free_rate)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {'risk_aversion': risk_aversion}
        except:
            continue
    
    return best_params if best_params else {'risk_aversion': 2.0}


def tune_bl_params(
    train_returns: pd.DataFrame,
    val_returns: pd.DataFrame,
    max_weight: float,
    risk_free_rate: float,
    param_grid: Dict = None
) -> Dict:
    """Tune Black-Litterman parameters."""
    from model.bl import get_bl_weights, BLConfig
    
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRIDS['BL']
    
    best_params = {}
    best_sharpe = -np.inf
    
    for tau in param_grid.get('tau', [0.05]):
        for risk_aversion in param_grid.get('risk_aversion', [2.0]):
            try:
                cfg = BLConfig(
                    risk_aversion=risk_aversion,
                    tau=tau,
                    risk_free_rate=risk_free_rate,
                    long_only=True,
                    max_weight=max_weight,
                    views=[]
                )
                weights = get_bl_weights(train_returns, None, cfg)
                sharpe = evaluate_portfolio_on_validation(weights, val_returns, risk_free_rate)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {'tau': tau, 'risk_aversion': risk_aversion}
            except:
                continue
    
    return best_params if best_params else {'tau': 0.05, 'risk_aversion': 2.0}


def tune_hrp_params(
    train_returns: pd.DataFrame,
    val_returns: pd.DataFrame,
    max_weight: float,
    risk_free_rate: float,
    param_grid: Dict = None
) -> Dict:
    """Tune HRP parameters."""
    from model.hrp import get_hrp_weights, HRPConfig
    
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRIDS['HRP']
    
    best_params = {}
    best_sharpe = -np.inf
    
    for linkage_method in param_grid.get('linkage_method', ['ward']):
        try:
            cfg = HRPConfig(
                linkage_method=linkage_method,
                long_only=True,
                max_weight=max_weight
            )
            weights = get_hrp_weights(train_returns, cfg)
            sharpe = evaluate_portfolio_on_validation(weights, val_returns, risk_free_rate)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {'linkage_method': linkage_method}
        except:
            continue
    
    return best_params if best_params else {'linkage_method': 'ward'}


def tune_cvar_params(
    train_returns: pd.DataFrame,
    val_returns: pd.DataFrame,
    max_weight: float,
    risk_free_rate: float,
    param_grid: Dict = None
) -> Dict:
    """Tune CVaR parameters."""
    from model.cvar import get_cvar_weights
    
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRIDS['CVaR']
    
    best_params = {}
    best_sharpe = -np.inf
    
    for alpha in param_grid.get('alpha', [0.05]):
        try:
            weights = get_cvar_weights(train_returns, max_weight=max_weight, alpha=alpha)
            sharpe = evaluate_portfolio_on_validation(weights, val_returns, risk_free_rate)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {'alpha': alpha}
        except:
            continue
    
    return best_params if best_params else {'alpha': 0.05}


def tune_lasso_params(
    train_returns: pd.DataFrame,
    val_returns: pd.DataFrame,
    max_weight: float,
    risk_free_rate: float,
    param_grid: Dict = None
) -> Dict:
    """Tune LASSO parameters."""
    from model.lasso import get_lasso_weights
    
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRIDS['LASSO']
    
    best_params = {}
    best_sharpe = -np.inf
    
    for num_assets in param_grid.get('num_assets_target', [10]):
        for penalty in param_grid.get('lasso_penalty', [0.01]):
            try:
                weights = get_lasso_weights(
                    train_returns,
                    max_weight=max_weight,
                    lasso_penalty=penalty,
                    num_assets_target=num_assets
                )
                sharpe = evaluate_portfolio_on_validation(weights, val_returns, risk_free_rate)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {'num_assets_target': num_assets, 'lasso_penalty': penalty}
            except:
                continue
    
    return best_params if best_params else {'num_assets_target': 10, 'lasso_penalty': 0.01}


def tune_rl_params(
    train_returns: pd.DataFrame,
    val_returns: pd.DataFrame,
    max_weight: float,
    risk_free_rate: float,
    param_grid: Dict = None
) -> Dict:
    """Tune RL parameters."""
    try:
        from model.rl import get_rl_weights
    except ImportError:
        return {'n_epochs': 5}
    
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRIDS['RL']
    
    best_params = {}
    best_sharpe = -np.inf
    
    # Simplified tuning (RL is expensive to train)
    for n_epochs in param_grid.get('n_epochs', [5]):
        try:
            weights = get_rl_weights(
                train_returns,
                agent=None,
                state_extractor=None,
                n_epochs=n_epochs,
                max_weight=max_weight,
                long_only=True
            )
            sharpe = evaluate_portfolio_on_validation(weights, val_returns, risk_free_rate)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {'n_epochs': n_epochs}
        except:
            continue
    
    return best_params if best_params else {'n_epochs': 5}


def tune_dro_params(
    train_returns: pd.DataFrame,
    val_returns: pd.DataFrame,
    max_weight: float,
    risk_free_rate: float,
    param_grid: Dict = None
) -> Dict:
    """Tune DRO parameters."""
    from model.dro import get_dro_weights
    
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRIDS['DRO']
    
    best_params = {}
    best_sharpe = -np.inf
    
    for epsilon in param_grid.get('epsilon', [0.5]):
        for risk_aversion in param_grid.get('risk_aversion', [1.0]):
            try:
                weights = get_dro_weights(
                    train_returns,
                    method='mean_variance',
                    epsilon=epsilon,
                    risk_aversion=risk_aversion,
                    max_weight=max_weight,
                    long_only=True
                )
                sharpe = evaluate_portfolio_on_validation(weights, val_returns, risk_free_rate)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {'epsilon': epsilon, 'risk_aversion': risk_aversion}
            except:
                continue
    
    return best_params if best_params else {'epsilon': 0.5, 'risk_aversion': 1.0}


# ==================== MAIN INTERFACE ====================

TUNING_FUNCTIONS = {
    'GMV': tune_gmv_params,
    'CAPM': tune_capm_params,
    'BL': tune_bl_params,
    'HRP': tune_hrp_params,
    'CVaR': tune_cvar_params,
    'LASSO': tune_lasso_params,
    'RL': tune_rl_params,
    'DRO': tune_dro_params,
}


def tune_model_params(
    model_name: str,
    train_returns: pd.DataFrame,
    val_returns: pd.DataFrame,
    max_weight: float = 0.15,
    risk_free_rate: float = 0.02,
    param_grid: Dict = None,
    verbose: bool = False
) -> Dict:
    """
    Tune hyperparameters for a specific model using train/validation split.
    
    Parameters:
    -----------
    model_name : str
        Model name ('GMV', 'CAPM', 'BL', 'HRP', 'CVaR', 'LASSO', 'RL', 'DRO')
    train_returns : pd.DataFrame
        Training returns
    val_returns : pd.DataFrame
        Validation returns
    max_weight : float
        Maximum weight per asset
    risk_free_rate : float
        Annual risk-free rate
    param_grid : Dict
        Custom parameter grid (optional, uses defaults if None)
    verbose : bool
        Print tuning progress
    
    Returns:
    --------
    best_params : Dict
        Best parameters found via validation
    """
    if model_name not in TUNING_FUNCTIONS:
        if verbose:
            print(f"  Warning: No tuning function for {model_name}, using defaults")
        return {}
    
    if verbose:
        print(f"  Tuning {model_name}...")
    
    tune_func = TUNING_FUNCTIONS[model_name]
    best_params = tune_func(
        train_returns, val_returns, max_weight, risk_free_rate, param_grid
    )
    
    if verbose:
        print(f"    Best params: {best_params}")
    
    return best_params


def walkforward_tune_all_models(
    lookback_returns: pd.DataFrame,
    max_weight: float = 0.15,
    risk_free_rate: float = 0.02,
    train_ratio: float = 0.6,
    models_to_tune: List[str] = None,
    verbose: bool = False
) -> Dict[str, Dict]:
    """
    Perform walk-forward parameter tuning for all models.
    
    Parameters:
    -----------
    lookback_returns : pd.DataFrame
        Full lookback window returns
    max_weight : float
        Maximum weight per asset
    risk_free_rate : float
        Annual risk-free rate
    train_ratio : float
        Fraction of lookback for training (default 0.6)
    models_to_tune : List[str]
        List of models to tune (default: all)
    verbose : bool
        Print tuning progress
    
    Returns:
    --------
    tuned_params : Dict[str, Dict]
        Dictionary mapping model names to best parameters
    """
    if models_to_tune is None:
        models_to_tune = list(TUNING_FUNCTIONS.keys())
    
    # Split data
    train_returns, val_returns = split_train_validation(lookback_returns, train_ratio)
    
    if verbose:
        print(f"\nWalk-Forward Parameter Tuning:")
        print(f"  Train period: {len(train_returns)} days")
        print(f"  Validation period: {len(val_returns)} days")
    
    tuned_params = {}
    
    for model_name in models_to_tune:
        try:
            best_params = tune_model_params(
                model_name,
                train_returns,
                val_returns,
                max_weight,
                risk_free_rate,
                param_grid=None,
                verbose=verbose
            )
            tuned_params[model_name] = best_params
        except Exception as e:
            if verbose:
                print(f"  Error tuning {model_name}: {e}")
            tuned_params[model_name] = {}
    
    return tuned_params


def get_default_params(model_name: str) -> Dict:
    """
    Get default parameters for a model (used when tuning is disabled).
    
    Parameters:
    -----------
    model_name : str
        Model name
    
    Returns:
    --------
    default_params : Dict
        Default parameter values
    """
    defaults = {
        'GMV': {'solver': 'OSQP'},
        'CAPM': {'risk_aversion': 2.0},
        'BL': {'tau': 0.05, 'risk_aversion': 2.0},
        'HRP': {'linkage_method': 'ward'},
        'CVaR': {'alpha': 0.05},
        'LASSO': {'num_assets_target': 3, 'lasso_penalty': 0.01},  # Ultra-sparse default K=3 for clear differentiation from DRO
        'RL': {'n_epochs': 5},
        'DRO': {'epsilon': 1.0, 'risk_aversion': 1.0},  # INCREASED epsilon from 0.5 to 1.0
    }
    
    return defaults.get(model_name, {})
