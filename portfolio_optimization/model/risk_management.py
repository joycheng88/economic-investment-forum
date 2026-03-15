"""
Institutional-Grade Risk Management Module

Comprehensive tail risk and VaR analysis for portfolio compliance and reporting:

1. **Value-at-Risk (VaR)**: Market loss at specified confidence levels
   - Historical VaR: empirical quantile from past returns
   - Parametric VaR: normal distribution assumption
   - Cornish-Fisher VaR: adjusts for skewness and kurtosis
   - Monte Carlo VaR: simulation-based tail estimates

2. **Expected Shortfall (ES)**: average loss beyond VaR (tail risk measure)
   - Conditional on being in the tail
   - Coher risk measure (unlike VaR)

3. **Stress Testing**: portfolio performance under crisis scenarios
   - Historical crises (2008, 2020, etc.)
   - Correlation breakdowns
   - Volatility spikes

4. **Tail Risk Metrics**: skewness, kurtosis, extreme loss quantiles

5. **Concentration Risk**: Herfindahl index, position concentration

6. **Crisis Correlation Analysis**: correlation during market stress periods

7. **Risk Attribution**: contribution to portfolio risk by asset
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')


#========================= VALUE AT RISK (VaR) =========================#

def calculate_var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Value-at-Risk using historical empirical quantile.
    
    VaR = percentile of negative returns at confidence level
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns
    confidence : float
        Confidence level (0.95 = 95% VaR, means 5% tail)
    
    Returns:
    --------
    var : float
        VaR as negative return (e.g., -0.02 = -2%)
    """
    alpha = 1 - confidence
    var = np.percentile(returns, alpha * 100)
    return var


def calculate_var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Value-at-Risk assuming normal distribution.
    
    Assumes returns ~ N(μ, σ²)
    VaR = μ - z_α * σ
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns
    confidence : float
        Confidence level
    
    Returns:
    --------
    var : float
        VaR under normality assumption
    """
    mu = returns.mean()
    sigma = returns.std()
    alpha = 1 - confidence
    z_alpha = norm.ppf(alpha)
    var = mu - z_alpha * sigma
    return var


def calculate_var_cornish_fisher(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Cornish-Fisher VaR: adjusts parametric VaR for skewness and excess kurtosis.
    
    More accurate than normal VaR when returns are non-normal.
    Accounts for fat tails and asymmetry.
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns
    confidence : float
        Confidence level
    
    Returns:
    --------
    var : float
        Cornish-Fisher adjusted VaR
    """
    mu = returns.mean()
    sigma = returns.std()
    skew = returns.skew()
    kurt = returns.kurtosis()
    
    alpha = 1 - confidence
    z_alpha = norm.ppf(alpha)
    
    # Cornish-Fisher adjustment
    z_cf = z_alpha + (z_alpha**2 - 1) * skew / 6 + (z_alpha**3 - 3 * z_alpha) * kurt / 24
    
    var = mu - z_cf * sigma
    return var


def calculate_var_montecarlo(
    returns: pd.Series,
    confidence: float = 0.95,
    n_simulations: int = 10000,
    periods: int = 1
) -> float:
    """
    Monte Carlo VaR: simulate future returns based on historical distribution.
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns for fitting distribution
    confidence : float
        Confidence level
    n_simulations : int
        Number of simulation paths
    periods : int
        Number of periods to simulate (default 1 = one day)
    
    Returns:
    --------
    var : float
        VaR from Monte Carlo simulation
    """
    mu = returns.mean()
    sigma = returns.std()
    
    # Simulate random normal returns
    simulated = np.random.normal(mu, sigma, size=(n_simulations, periods))
    
    # Compound returns over periods
    cumulative_simulated = np.prod(1 + simulated, axis=1) - 1
    
    # VaR as quantile
    alpha = 1 - confidence
    var = np.percentile(cumulative_simulated, alpha * 100)
    
    return var


def calculate_var_all_methods(
    returns: pd.Series,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Calculate VaR using all methods and return comparison.
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio or asset returns
    confidence : float
        Confidence level
    
    Returns:
    --------
    var_dict : dict
        VaR estimates from different methods
    """
    return {
        'historical': calculate_var_historical(returns, confidence),
        'parametric': calculate_var_parametric(returns, confidence),
        'cornish_fisher': calculate_var_cornish_fisher(returns, confidence),
        'monte_carlo': calculate_var_montecarlo(returns, confidence),
    }


#========================= EXPECTED SHORTFALL =========================#

def calculate_es(returns: pd.Series, confidence: float = 0.95, method: str = 'historical') -> float:
    """
    Calculate Expected Shortfall (Conditional Value-at-Risk).
    
    ES = expected loss conditional on being beyond VaR
    More coherent risk measure than VaR (respects subadditivity)
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio or asset returns
    confidence : float
        Confidence level
    method : str
        'historical' | 'parametric' | 'cornish_fisher'
    
    Returns:
    --------
    es : float
        Expected shortfall / CVaR
    """
    alpha = 1 - confidence
    
    if method == 'historical':
        var = calculate_var_historical(returns, confidence)
        es = returns[returns <= var].mean()
    
    elif method == 'parametric':
        mu = returns.mean()
        sigma = returns.std()
        z_alpha = norm.ppf(alpha)
        es = mu - sigma * norm.pdf(z_alpha) / alpha
    
    elif method == 'cornish_fisher':
        var = calculate_var_cornish_fisher(returns, confidence)
        es = returns[returns <= var].mean()
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return es


#========================= TAIL RISK ANALYSIS =========================#

def calculate_tail_metrics(returns: pd.Series, confidence: float = 0.95) -> Dict:
    """
    Comprehensive tail risk metrics.
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns
    confidence : float
        Confidence level for VaR/ES
    
    Returns:
    --------
    metrics : dict
        Tail risk metrics including:
        - VaR (multiple methods)
        - ES / CVaR
        - Extreme loss percentiles
        - Tail skewness/kurtosis
        - Largest daily loss
    """
    alpha = 1 - confidence
    
    # Quantiles in tail
    tail_returns = returns[returns <= returns.quantile(alpha)]
    
    metrics = {
        'var_95': calculate_var_historical(returns, 0.95),
        'var_99': calculate_var_historical(returns, 0.99),
        'es_95': calculate_es(returns, 0.95),
        'es_99': calculate_es(returns, 0.99),
        'worst_return': returns.min(),
        'worst_10_pct': returns.quantile(0.10),
        'worst_5_pct': returns.quantile(0.05),
        'worst_1_pct': returns.quantile(0.01),
        'mean_worst_10pct': tail_returns.mean(),
        'largest_loss_days': len(returns[returns < -0.02]),
        'crash_days': len(returns[returns < -0.05]),
        'tail_skewness': tail_returns.skew() if len(tail_returns) > 0 else 0,
        'tail_kurtosis': tail_returns.kurtosis() if len(tail_returns) > 0 else 0,
    }
    
    return metrics


#========================= STRESS TESTING =========================#

def stress_test_scenario(
    returns: pd.DataFrame,
    weights: pd.Series,
    scenario_name: str,
    scenario_returns: Dict[str, float]
) -> Dict:
    """
    Compute portfolio impact under a stressed scenario.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns (T x N)
    weights : pd.Series
        Portfolio weights
    scenario_name : str
        Name of scenario
    scenario_returns : dict
        Ticker -> stressed return mapping
    
    Returns:
    --------
    impact : dict
        Portfolio impact under stress
    """
    # Align weights to returns columns
    weights_aligned = weights.reindex(returns.columns, fill_value=0.0)
    
    # Build scenario return vector
    stress_vec = pd.Series(
        [scenario_returns.get(t, 0.0) for t in returns.columns],
        index=returns.columns
    )
    
    # Portfolio return under stress
    portfolio_stress_return = (weights_aligned * stress_vec).sum()
    
    # Individual asset impacts
    asset_impacts = weights_aligned * stress_vec
    
    # Contribution to portfolio loss
    contributions = asset_impacts / portfolio_stress_return if portfolio_stress_return < 0 else asset_impacts
    
    return {
        'scenario': scenario_name,
        'portfolio_return': portfolio_stress_return,
        'asset_impacts': asset_impacts.sort_values(),
        'worst_asset': asset_impacts.idxmin(),
        'max_loss': asset_impacts.min(),
        'concentration': (weights_aligned * asset_impacts).sum() / portfolio_stress_return if portfolio_stress_return < 0 else 0
    }


def builtin_stress_scenarios() -> Dict[str, Dict[str, float]]:
    """
    Pre-defined crisis scenarios based on historical events.
    
    Returns:
    --------
    scenarios : dict
        Scenario name -> {asset returns}
    """
    return {
        # 2008 Financial Crisis
        '2008_Crisis': {
            'SPY': -0.37,   # S&P 500
            'AGG': -0.05,   # US Bonds (flight to safety)
            'EFA': -0.43,   # International stocks
            'GLD': 0.05,    # Gold (safe haven)
        },
        
        # COVID-19 Crash (March 2020)
        'COVID_2020': {
            'SPY': -0.34,
            'AGG': -0.02,
            'EFA': -0.35,
            'GLD': 0.07,
        },
        
        # Volmageddon (Feb 2018 - vol spike)
        'Vol_Spike': {
            'SPY': -0.04,
            'AGG': 0.01,
            'EFA': -0.05,
            'GLD': 0.02,
        },
        
        # Recession scenario
        'Recession': {
            'SPY': -0.25,
            'AGG': 0.03,    # Flight to quality
            'EFA': -0.28,
            'GLD': 0.08,
        },
        
        # Correlation breakdown (all risky assets down, bonds up)
        'Correlation_Breakdown': {
            'SPY': -0.20,
            'AGG': 0.02,
            'EFA': -0.22,
            'GLD': 0.10,
        },
        
        # Stagflation (growth and bonds both down)
        'Stagflation': {
            'SPY': -0.15,
            'AGG': -0.08,
            'EFA': -0.18,
            'GLD': 0.12,
        },
    }


def run_stress_tests(
    returns: pd.DataFrame,
    weights: pd.Series,
    scenarios: Dict[str, Dict[str, float]] = None
) -> Dict:
    """
    Run battery of stress tests.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns
    weights : pd.Series
        Portfolio weights
    scenarios : dict
        Custom scenarios (uses builtin if None)
    
    Returns:
    --------
    results : dict
        Scenario results, ranked by severity
    """
    if scenarios is None:
        scenarios = builtin_stress_scenarios()
    
    results = {}
    for scenario_name, scenario_returns in scenarios.items():
        # Only test with assets we have
        scenario_filtered = {
            k: v for k, v in scenario_returns.items() 
            if k in returns.columns
        }
        
        if scenario_filtered:
            impact = stress_test_scenario(returns, weights, scenario_name, scenario_filtered)
            results[scenario_name] = impact
    
    # Sort by severity (most negative first)
    results_sorted = dict(sorted(
        results.items(),
        key=lambda x: x[1]['portfolio_return']
    ))
    
    return results_sorted


#========================= CONCENTRATION RISK =========================#

def calculate_concentration_metrics(weights: pd.Series) -> Dict:
    """
    Measure portfolio concentration and diversification.
    
    Parameters:
    -----------
    weights : pd.Series
        Portfolio weights
    
    Returns:
    --------
    metrics : dict
        Concentration metrics
    """
    w = weights[weights > 1e-6].values  # Non-zero weights
    
    # Herfindahl Index: sum of squared weights
    # HHI = 1/N (perfect diversification) to 1 (single position)
    herfindahl = (w ** 2).sum()
    
    # Effective number of positions
    n_eff = 1 / herfindahl if herfindahl > 0 else len(weights)
    
    # Concentration metrics
    metrics = {
        'herfindahl_index': herfindahl,
        'effective_positions': n_eff,
        'max_weight': weights.max(),
        'top_3_weight': weights.nlargest(3).sum(),
        'top_5_weight': weights.nlargest(5).sum(),
        'n_nonzero_positions': len(w),
        'is_concentrated': herfindahl > 0.3,  # > 30% industry standard
        'diversification_ratio': w.mean() / w.std() if len(w) > 1 else 1,
    }
    
    return metrics


#========================= CRISIS CORRELATION =========================#

def calculate_crisis_correlation(
    returns: pd.DataFrame,
    window: int = 252,
    crisis_threshold: float = -0.02
) -> Dict:
    """
    Measure correlation breakdown during market stress periods.
    
    During crises, correlations tend to spike (contagion effect).
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns (T x N)
    window : int
        Rolling correlation window
    crisis_threshold : float
        Return threshold to define crisis days
    
    Returns:
    --------
    metrics : dict
        Normal vs crisis correlation comparison
    """
    # Identify crisis days (any critical asset drops sharply)
    min_returns = returns.min(axis=1)
    crisis_days = min_returns < crisis_threshold
    
    # Normal and crisis correlation matrices
    normal_corr = returns[~crisis_days].corr()
    crisis_corr = returns[crisis_days].corr() if crisis_days.sum() > 5 else returns.corr()
    
    # Correlation increase during crises
    off_diag_normal = normal_corr.values[np.triu_indices_from(normal_corr.values, k=1)]
    off_diag_crisis = crisis_corr.values[np.triu_indices_from(crisis_corr.values, k=1)]
    
    avg_corr_normal = np.mean(off_diag_normal)
    avg_corr_crisis = np.mean(off_diag_crisis)
    
    metrics = {
        'normal_avg_correlation': avg_corr_normal,
        'crisis_avg_correlation': avg_corr_crisis,
        'correlation_increase': avg_corr_crisis - avg_corr_normal,
        'correlation_increase_pct': (avg_corr_crisis - avg_corr_normal) / abs(avg_corr_normal) * 100 if avg_corr_normal != 0 else 0,
        'n_crisis_days': crisis_days.sum(),
        'crisis_pct': crisis_days.sum() / len(returns) * 100,
        'normal_corr_matrix': normal_corr,
        'crisis_corr_matrix': crisis_corr,
    }
    
    return metrics


#========================= RISK ATTRIBUTION =========================#

def marginal_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence: float = 0.95
) -> pd.Series:
    """
    Marginal Value-at-Risk: change in portfolio VaR from 1% increase in position.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns
    weights : pd.Series
        Current weights
    confidence : float
        VaR confidence level
    
    Returns:
    --------
    marginal_var : pd.Series
        MVaR for each asset
    """
    portfolio_returns = returns @ weights.reindex(returns.columns, fill_value=0)
    portfolio_var = calculate_var_historical(portfolio_returns, confidence)
    
    marginal_vars = {}
    for asset in returns.columns:
        w_bumped = weights.copy()
        w_bumped[asset] = min(w_bumped.get(asset, 0) + 0.01, 1.0)
        w_bumped = w_bumped / w_bumped.sum()  # Renormalize
        
        portfolio_rets_bumped = returns @ w_bumped.reindex(returns.columns, fill_value=0)
        portfolio_var_bumped = calculate_var_historical(portfolio_rets_bumped, confidence)
        
        marginal_vars[asset] = (portfolio_var_bumped - portfolio_var) / 0.01
    
    return pd.Series(marginal_vars)


def var_contribution(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence: float = 0.95
) -> pd.Series:
    """
    VaR Contribution: marginal VaR × portfolio weight.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns
    weights : pd.Series
        Current weights
    confidence : float
        VaR confidence level
    
    Returns:
    --------
    var_contrib : pd.Series
        VaR contribution by asset
    """
    mvar = marginal_var(returns, weights, confidence)
    weights_aligned = weights.reindex(returns.columns, fill_value=0)
    var_contrib = mvar * weights_aligned
    
    return var_contrib.sort_values()


#========================= COMPREHENSIVE RISK REPORT =========================#

def generate_risk_report(
    returns: pd.Series | pd.DataFrame,
    weights: pd.Series = None,
    portfolio_name: str = "Portfolio",
    confidence_levels: List[float] = [0.90, 0.95, 0.99]
) -> Dict:
    """
    Generate comprehensive institutional risk report.
    
    Parameters:
    -----------
    returns : pd.Series or pd.DataFrame
        Daily returns (Series for single asset/portfolio, DataFrame for multi-asset)
    weights : pd.Series
        Portfolio weights (optional, for multi-asset analysis)
    portfolio_name : str
        Name for reporting
    confidence_levels : list
        VaR/ES confidence levels to report
    
    Returns:
    --------
    report : dict
        Comprehensive risk metrics
    """
    # Handle both single series and multi-asset
    if isinstance(returns, pd.DataFrame):
        if weights is None:
            portfolio_returns = returns.mean(axis=1)  # Equal-weight if no weights given
        else:
            portfolio_returns = returns @ weights.reindex(returns.columns, fill_value=0)
    else:
        portfolio_returns = returns
    
    report = {
        'portfolio': portfolio_name,
        'analysis_period': {
            'start_date': returns.index[0] if hasattr(returns, 'index') else None,
            'end_date': returns.index[-1] if hasattr(returns, 'index') else None,
            'n_days': len(returns),
        },
        'tail_risk': calculate_tail_metrics(portfolio_returns),
        'var_methods': calculate_var_all_methods(portfolio_returns, 0.95),
    }
    
    # Add multi-confidence VaR/ES
    for conf in confidence_levels:
        report[f'var_{int(conf*100)}'] = calculate_var_historical(portfolio_returns, conf)
        report[f'es_{int(conf*100)}'] = calculate_es(portfolio_returns, conf)
    
    # Concentration and stress tests (only if multi-asset with weights)
    if isinstance(returns, pd.DataFrame) and weights is not None:
        report['concentration'] = calculate_concentration_metrics(weights)
        report['stress_tests'] = run_stress_tests(returns, weights)
        report['crisis_correlation'] = calculate_crisis_correlation(returns)
    
    return report


if __name__ == "__main__":
    # Test example
    np.random.seed(42)
    
    # Generate sample returns
    returns_sample = pd.Series(np.random.normal(0.0005, 0.01, 500))
    
    # VaR comparison
    print("VaR Estimates (95% confidence):")
    var_estimates = calculate_var_all_methods(returns_sample, 0.95)
    for method, var in var_estimates.items():
        print(f"  {method:20s}: {var:6.2%}")
    
    # Expected Shortfall
    es_95 = calculate_es(returns_sample, 0.95)
    print(f"\nExpected Shortfall (95%): {es_95:6.2%}")
    
    # Tail metrics
    print("\nTail Risk Metrics:")
    tail_metrics = calculate_tail_metrics(returns_sample)
    for metric, value in list(tail_metrics.items())[:5]:
        print(f"  {metric:20s}: {value:8.4f}")
