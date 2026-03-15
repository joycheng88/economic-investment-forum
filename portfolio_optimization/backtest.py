"""
Rolling Window Backtesting Framework for Portfolio Models

Compares out-of-sample performance of 8 portfolio optimization models:
1. GMV (Global Minimum Variance)
2. CAPM (Mean-Variance maximizing Sharpe ratio / tangency portfolio)
3. Black-Litterman (Market equilibrium + views)
4. HRP (Hierarchical Risk Parity)
5. CVaR (Conditional Value at Risk / Expected Shortfall)
6. LASSO (Sparse Portfolio Selection with cardinality control)
7. RL (Reinforcement Learning with Actor-Critic policy)
8. DRO (Distributionally Robust Optimization with Wasserstein ambiguity)

Methodology:
- Rolling window with lookback period for training
- Monthly rebalancing
- Measure out-of-sample returns, risk, and risk-adjusted metrics
- Transaction cost deduction (0.1% per one-way rebalancing)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import warnings

from data import DataConfig, load_data, get_risk_free_rate
from model.gmv import estimate_covariance as gmv_cov, gmv_weights, portfolio_volatility
from model.capm import get_capm_weights, CAPMConfig
from model.bl import get_bl_weights, BLConfig
from model.hrp import get_hrp_weights, HRPConfig
from model.cvar import get_cvar_weights
from model.lasso import get_lasso_weights
from model.rl import get_rl_weights
from model.dro import get_dro_weights
from model.risk_management import (
    calculate_var_all_methods, calculate_es, calculate_tail_metrics,
    calculate_concentration_metrics, calculate_crisis_correlation
)
from param_tuning import walkforward_tune_all_models, get_default_params


def run_rolling_backtest(
    tickers: list,
    start: str = "2018-01-01",
    end: str = "2026-02-20",
    lookback_days: int = 504,  # ~2 years
    rebalance_freq: str = "ME",  # Month-end
    max_weight: float = 0.15,
    risk_free_rate: float = None,
    transaction_cost_rate: float = 0.001,  # 10 bps per one-way transaction
    use_param_tuning: bool = False,  # Enable walk-forward parameter tuning
    train_val_ratio: float = 0.6  # Train/validation split ratio
) -> Dict:
    """
    Run rolling window backtest for all 8 models + 3 benchmarks.

    Parameters:
    -----------
    tickers : list
        Stock ticker symbols
    start : str
        Backtest start date (need to go back further for initial training)
    end : str
        Backtest end date
    lookback_days : int
        Number of days of history for training window (default 504 ≈ 2 years)
    rebalance_freq : str
        Rebalancing frequency ("ME"=month-end, "QE"=quarter-end, etc.)
    max_weight : float
        Maximum weight per asset
    risk_free_rate : float
        Annual risk-free rate for Sharpe calculation (default: real-time US Treasury rate)
    transaction_cost_rate : float
        One-way transaction cost as a fraction (default 0.001 = 10 basis points, ~0.1% typical)
        Turnover cost = transaction_cost_rate * sum(|w_new_i - w_old_i|)

    Returns:
    --------
    results : dict
        Dictionary with results for each model + benchmarks:
        {
            'model_name': {
                'daily_returns': pd.Series,          # Out-of-sample daily returns (net of transaction costs)
                'cumulative_returns': pd.Series,     # Cumulative return growth
                'weights_history': [(...), ...],     # List of (date, weights) tuples
                'metrics': {...},                    # Performance metrics
                'turnover_costs': pd.Series,         # Cumulative turnover costs deducted
                'relative_metrics': {...}            # Alpha, IR, TE vs benchmarks
            },
            'benchmarks': {
                'SPY': {...},
                '60-40': {...},
                'Equal-Weight': {...}
            }
        }
    """
    
    # Use real-time risk-free rate if not provided
    if risk_free_rate is None:
        risk_free_rate = get_risk_free_rate()
    
    print("=" * 80)
    print("ROLLING WINDOW BACKTEST")
    print("=" * 80)
    print(f"Tickers: {len(tickers)} stocks")
    print(f"Period: {start} to {end}")
    print(f"Lookback: {lookback_days} days (~{lookback_days/252:.1f} years)")
    print(f"Rebalance: {rebalance_freq}")
    print(f"Max weight: {max_weight*100:.1f}%")
    print(f"Risk-free rate: {risk_free_rate*100:.2f}%")
    print(f"Parameter tuning: {'ENABLED (walk-forward)' if use_param_tuning else 'DISABLED (default params)'}")
    
    # Load full dataset
    print(f"\nLoading data...")
    
    # Load benchmark data (SPY for S&P 500)
    print(f"Loading benchmarks...")
    cfg_benchmark = DataConfig(start=start, end=end, return_type="log", min_non_na_ratio=0.60)
    try:
        spy_prices, spy_returns = load_data(['SPY'], cfg_benchmark)
        spy_returns = spy_returns.sort_index()
        print(f"  ✓ SPY benchmark loaded")
    except:
        spy_returns = None
        print(f"  ⚠ SPY benchmark unavailable")
    
    # Load portfolio data
    print(f"Loading portfolio tickers...")
    cfg = DataConfig(start=start, end=end, return_type="log", min_non_na_ratio=0.60)
    prices, returns = load_data(tickers, cfg)
    returns = returns.sort_index()
    
    # Build rebalance dates
    rebalance_dates = returns.resample(rebalance_freq).last().index
    rebalance_dates = [d for d in rebalance_dates if d in returns.index]
    
    if len(rebalance_dates) < 2:
        raise ValueError(f"Not enough rebalance dates (found {len(rebalance_dates)}). "
                        "Try longer date range or different rebalance frequency.")
    
    print(f"Rebalance dates: {len(rebalance_dates)} (from {rebalance_dates[0].date()} to {rebalance_dates[-1].date()})")
    
    # Initialize storage
    results = {
        'GMV': {'returns': [], 'dates': [], 'weights_list': [], 'turnover_costs': [], 'prev_weights': None},
        'CAPM': {'returns': [], 'dates': [], 'weights_list': [], 'turnover_costs': [], 'prev_weights': None},
        'BL': {'returns': [], 'dates': [], 'weights_list': [], 'turnover_costs': [], 'prev_weights': None},
        'HRP': {'returns': [], 'dates': [], 'weights_list': [], 'turnover_costs': [], 'prev_weights': None},
        'CVaR': {'returns': [], 'dates': [], 'weights_list': [], 'turnover_costs': [], 'prev_weights': None},
        'LASSO': {'returns': [], 'dates': [], 'weights_list': [], 'turnover_costs': [], 'prev_weights': None},
        'RL': {'returns': [], 'dates': [], 'weights_list': [], 'turnover_costs': [], 'prev_weights': None},
        'DRO': {'returns': [], 'dates': [], 'weights_list': [], 'turnover_costs': [], 'prev_weights': None},
    }
    
    # Rolling window backtest
    print(f"\nRunning backtest...")
    num_rebalances_used = 0
    
    for i in range(len(rebalance_dates) - 1):
        reb_date = rebalance_dates[i]
        next_reb_date = rebalance_dates[i + 1]
        
        idx = returns.index.get_loc(reb_date)
        
        # Skip if not enough training data
        if idx < lookback_days:
            continue
        
        num_rebalances_used += 1
        
        # Training window: past lookback_days
        window_returns = returns.iloc[idx - lookback_days:idx]
        window_cov = gmv_cov(window_returns, annualize=True)
        
        # Walk-forward parameter tuning (if enabled)
        if use_param_tuning:
            tuned_params = walkforward_tune_all_models(
                window_returns,
                max_weight=max_weight,
                risk_free_rate=risk_free_rate,
                train_ratio=train_val_ratio,
                verbose=False
            )
        else:
            tuned_params = {model: get_default_params(model) for model in ['GMV', 'CAPM', 'BL', 'HRP', 'CVaR', 'LASSO', 'RL', 'DRO']}
        
        # Get weights from each model (using tuned or default parameters)
        try:
            solver = tuned_params.get('GMV', {}).get('solver', 'OSQP')
            w_gmv = gmv_weights(window_cov, type('C', (), {'long_only': True, 'max_weight': max_weight, 'solver': solver})())
        except:
            w_gmv = None
        
        try:
            risk_aversion = tuned_params.get('CAPM', {}).get('risk_aversion', 2.0)
            capm_cfg = CAPMConfig(model_type="capm", risk_aversion=risk_aversion, risk_free_rate=risk_free_rate,
                                 long_only=True, max_weight=max_weight)
            w_capm = get_capm_weights(window_returns, capm_cfg)
        except:
            w_capm = None
        
        try:
            bl_params = tuned_params.get('BL', {})
            bl_cfg = BLConfig(
                risk_aversion=bl_params.get('risk_aversion', 2.0),
                tau=bl_params.get('tau', 0.05),
                risk_free_rate=risk_free_rate,
                long_only=True,
                max_weight=max_weight,
                views=[]
            )
            w_bl = get_bl_weights(window_returns, None, bl_cfg)
        except:
            w_bl = None
        
        try:
            linkage = tuned_params.get('HRP', {}).get('linkage_method', 'ward')
            hrp_cfg = HRPConfig(linkage_method=linkage, long_only=True, max_weight=max_weight)
            w_hrp = get_hrp_weights(window_returns, hrp_cfg)
        except:
            w_hrp = None
        
        try:
            alpha = tuned_params.get('CVaR', {}).get('alpha', 0.05)
            w_cvar = get_cvar_weights(window_returns, max_weight=max_weight, alpha=alpha)
        except:
            w_cvar = None
        
        try:
            lasso_params = tuned_params.get('LASSO', {})
            w_lasso = get_lasso_weights(
                window_returns,
                max_weight=max_weight,
                lasso_penalty=lasso_params.get('lasso_penalty', 0.01),
                num_assets_target=lasso_params.get('num_assets_target', 5)  # REDUCED from 10
            )
        except:
            w_lasso = None
        
        # RL (Reinforcement Learning) - skip if PyTorch not available or data insufficient
        w_rl = None
        if len(window_returns) >= 100:  # RL needs sufficient data
            try:
                n_epochs = tuned_params.get('RL', {}).get('n_epochs', 3)
                w_rl = get_rl_weights(window_returns, agent=None, state_extractor=None, n_epochs=n_epochs, max_weight=max_weight, long_only=True)
            except ImportError:
                pass  # PyTorch not available
            except:
                pass  # Other RL errors
        
        # DRO (Distributionally Robust Optimization)
        try:
            dro_params = tuned_params.get('DRO', {})
            w_dro = get_dro_weights(
                window_returns,
                method='mean_variance',
                epsilon=dro_params.get('epsilon', 1.0),  # INCREASED from 0.5
                risk_aversion=dro_params.get('risk_aversion', 1.0),
                max_weight=max_weight,
                long_only=True
            )
        except:
            w_dro = None
        
        # Holding period returns
        start_loc = idx + 1
        end_loc = returns.index.get_loc(next_reb_date)
        
        if start_loc > end_loc:
            continue
        
        hold_rets = returns.iloc[start_loc:end_loc + 1]
        hold_dates = hold_rets.index
        
        # Store portfolio returns for each model
        for model_name, weights in [('GMV', w_gmv), ('CAPM', w_capm), ('BL', w_bl), ('HRP', w_hrp), ('CVaR', w_cvar), ('LASSO', w_lasso), ('RL', w_rl), ('DRO', w_dro)]:
            if weights is not None:
                # Align and compute daily returns
                hold_rets_aligned = hold_rets[weights.index]
                port_daily = hold_rets_aligned @ weights
                
                # CALCULATE AND DEDUCT TRANSACTION COSTS
                # Turnover = sum of absolute weight changes (one-way costs)
                prev_w = results[model_name]['prev_weights']
                if prev_w is not None:
                    # Align previous weights to current assets
                    prev_w_aligned = prev_w.reindex(weights.index, fill_value=0.0)
                    turnover = np.sum(np.abs(weights.values - prev_w_aligned.values))
                    turnover_cost = transaction_cost_rate * turnover
                    
                    # Deduct from first day of holding period
                    port_daily_adjusted = port_daily.copy()
                    if len(port_daily_adjusted) > 0:
                        port_daily_adjusted.iloc[0] -= turnover_cost  # Deduct on rebalance day
                    
                    results[model_name]['turnover_costs'].append(turnover_cost)
                    port_daily = port_daily_adjusted
                else:
                    results[model_name]['turnover_costs'].append(0.0)  # No cost on first rebalance
                
                # Update previous weights for next rebalance
                results[model_name]['prev_weights'] = weights.copy()
                
                results[model_name]['returns'].extend(port_daily.values)
                results[model_name]['dates'].extend(hold_dates)
                results[model_name]['weights_list'].append((reb_date, weights.copy()))
    
    print(f"Completed {num_rebalances_used} rebalances with out-of-sample testing")
    
    # Convert to Series and compute metrics
    backtest_results = {}
    benchmark_results = {}
    
    # Get common date range for all models
    all_dates = None
    for model_name in results.keys():
        if len(results[model_name]['dates']) > 0:
            dates = pd.to_datetime(results[model_name]['dates'])
            if all_dates is None:
                all_dates = pd.DatetimeIndex(dates).sort_values()
            else:
                all_dates = all_dates.union(pd.DatetimeIndex(dates))
    
    # Process model results
    for model_name in results.keys():
        if len(results[model_name]['returns']) > 0:
            daily_rets = pd.Series(
                results[model_name]['returns'],
                index=pd.to_datetime(results[model_name]['dates']),
                name=f"{model_name}_returns"
            )
            daily_rets = daily_rets.sort_index()
            
            # Compute metrics
            metrics = _compute_metrics(daily_rets, risk_free_rate)
            
            # Store turnover costs as Series for reporting
            turnover_costs_series = pd.Series(
                results[model_name]['turnover_costs'],
                name=f"{model_name}_turnover_cost"
            )
            
            backtest_results[model_name] = {
                'daily_returns': daily_rets,
                'cumulative_returns': (1 + daily_rets).cumprod(),
                'weights_history': results[model_name]['weights_list'],
                'metrics': metrics,
                'turnover_costs': turnover_costs_series,
                'total_turnover_cost': np.sum(results[model_name]['turnover_costs'])
            }
        else:
            print(f"Warning: {model_name} produced no valid returns")
    
    # Compute benchmark returns
    if all_dates is not None and len(all_dates) > 0:
        print(f"\nComputing benchmarks...")
        
        # SPY (S&P 500) benchmark
        if spy_returns is not None:
            spy_rets_aligned = spy_returns['SPY'].reindex(all_dates, method='ffill')
            spy_rets_aligned = spy_rets_aligned.fillna(0)
            benchmark_results['SPY'] = {
                'daily_returns': spy_rets_aligned,
                'cumulative_returns': (1 + spy_rets_aligned).cumprod(),
                'metrics': _compute_metrics(spy_rets_aligned, risk_free_rate),
                'weights_history': [],
                'turnover_costs': pd.Series([0], index=[all_dates[0]]),
                'total_turnover_cost': 0.0
            }
            print(f"  ✓ SPY benchmark computed")
        
        # 60-40 Portfolio (60% SPY / 40% AGG)
        try:
            agg_prices, agg_returns = load_data(['AGG'], cfg_benchmark)
            agg_returns = agg_returns.sort_index()
            
            spy_aligned = spy_returns['SPY'].reindex(all_dates, method='ffill').fillna(0)
            agg_aligned = agg_returns['AGG'].reindex(all_dates, method='ffill').fillna(0)
            
            port_60_40 = 0.6 * spy_aligned + 0.4 * agg_aligned
            benchmark_results['60-40'] = {
                'daily_returns': port_60_40,
                'cumulative_returns': (1 + port_60_40).cumprod(),
                'metrics': _compute_metrics(port_60_40, risk_free_rate),
                'weights_history': [],
                'turnover_costs': pd.Series([0], index=[all_dates[0]]),
                'total_turnover_cost': 0.0
            }
            print(f"  ✓ 60-40 portfolio computed")
        except:
            print(f"  ⚠ 60-40 portfolio unavailable (AGG data missing)")
        
        # Equal-Weight Portfolio
        equal_weight_rets = returns.reindex(all_dates, method='ffill').mean(axis=1)
        benchmark_results['Equal-Weight'] = {
            'daily_returns': equal_weight_rets,
            'cumulative_returns': (1 + equal_weight_rets).cumprod(),
            'metrics': _compute_metrics(equal_weight_rets, risk_free_rate),
            'weights_history': [],
            'turnover_costs': pd.Series([0], index=[all_dates[0]]),
            'total_turnover_cost': 0.0
        }
        print(f"  ✓ Equal-weight portfolio computed")
        
        # Compute relative metrics for each model vs benchmarks
        print(f"\nComputing relative metrics...")
        for model_name, model_data in backtest_results.items():
            model_rets = model_data['daily_returns']
            relative_metrics = {}
            
            for bench_name, bench_data in benchmark_results.items():
                bench_rets = bench_data['daily_returns']
                rel_metrics = _compute_relative_metrics(
                    model_rets, bench_rets, risk_free_rate
                )
                relative_metrics[bench_name] = rel_metrics
            
            backtest_results[model_name]['relative_metrics'] = relative_metrics
        
        print(f"  ✓ Relative metrics computed for {len(backtest_results)} models")
    
    # Add benchmarks to results
    backtest_results['benchmarks'] = benchmark_results
    
    return backtest_results


def _compute_metrics(daily_returns: pd.Series, risk_free_rate: float = 0.02) -> Dict:
    """
    Compute performance metrics for a return series.

    Parameters:
    -----------
    daily_returns : pd.Series
        Daily return series
    risk_free_rate : float
        Annual risk-free rate

    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    """
    if len(daily_returns) == 0:
        raise ValueError("Empty return series")
    
    # Annualization factor
    periods_per_year = 252
    
    # Basic metrics
    total_days = len(daily_returns)
    total_years = total_days / periods_per_year
    
    # Returns
    cumulative_return = (1 + daily_returns).prod() - 1
    annual_return = (1 + cumulative_return) ** (1 / max(total_years, 1)) - 1
    
    # Volatility
    daily_vol = daily_returns.std()
    annual_vol = daily_vol * np.sqrt(periods_per_year)
    
    # Sharpe Ratio
    excess_daily_return = daily_returns - risk_free_rate / periods_per_year
    sharpe = (excess_daily_return.mean() / daily_vol * np.sqrt(periods_per_year)) if daily_vol > 0 else 0
    
    # Sortino Ratio (only downside volatility)
    downside_rets = daily_returns[daily_returns < 0]
    downside_daily_vol = downside_rets.std() if len(downside_rets) > 0 else daily_vol
    downside_annual_vol = downside_daily_vol * np.sqrt(periods_per_year)
    sortino = (annual_return - risk_free_rate) / downside_annual_vol if downside_annual_vol > 0 else 0
    
    # Maximum Drawdown
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar Ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win Rate
    win_rate = (daily_returns > 0).sum() / len(daily_returns)
    
    # Return/Volatility Ratio
    ret_vol_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    
    # ======================== VALUE-AT-RISK & TAIL RISK METRICS ========================
    # VaR at multiple confidence levels
    var_estimates = calculate_var_all_methods(daily_returns, 0.95)
    
    # Expected Shortfall (CVaR)
    es_95 = calculate_es(daily_returns, 0.95, method='historical')
    es_99 = calculate_es(daily_returns, 0.99, method='historical')
    
    # Comprehensive tail risk metrics
    tail_metrics = calculate_tail_metrics(daily_returns, 0.95)
    
    # ======================== BUILD METRICS DICTIONARY ========================
    metrics = {
        # Return metrics
        'total_days': total_days,
        'total_years': total_years,
        'cumulative_return': cumulative_return,
        'annual_return': annual_return,
        'avg_daily_return': daily_returns.mean(),
        'median_daily_return': daily_returns.median(),
        
        # Risk metrics
        'daily_volatility': daily_vol,
        'annual_volatility': annual_vol,
        'max_drawdown': max_drawdown,
        'skewness': daily_returns.skew(),
        'kurtosis': daily_returns.kurtosis(),
        
        # Risk-adjusted metrics
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'return_vol_ratio': ret_vol_ratio,
        'win_rate': win_rate,
        
        # ===== VALUE-AT-RISK METRICS (Institutional Grade) =====
        # VaR methods comparison
        'var_historical_95': var_estimates['historical'],
        'var_parametric_95': var_estimates['parametric'],
        'var_cornish_fisher_95': var_estimates['cornish_fisher'],
        'var_montecarlo_95': var_estimates['monte_carlo'],
        
        # Expected Shortfall (better than VaR for tail risk)
        'expected_shortfall_95': es_95,
        'expected_shortfall_99': es_99,
        
        # Tail risk extremity
        'worst_daily_return': tail_metrics['worst_return'],
        'worst_5pct_return': tail_metrics['worst_5_pct'],
        'worst_1pct_return': tail_metrics['worst_1_pct'],
        'mean_worst_10pct': tail_metrics['mean_worst_10pct'],
        'largest_loss_days': tail_metrics['largest_loss_days'],
        'crash_days': tail_metrics['crash_days'],
    }
    
    return metrics


def _compute_relative_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02
) -> Dict:
    """
    Compute relative performance metrics vs a benchmark.
    
    Parameters:
    -----------
    portfolio_returns : pd.Series
        Daily portfolio returns
    benchmark_returns : pd.Series
        Daily benchmark returns (aligned to same dates)
    risk_free_rate : float
        Annual risk-free rate
    
    Returns:
    --------
    metrics : dict
        Dictionary with:
        - alpha: Annualized alpha (excess return vs benchmark)
        - beta: Portfolio beta vs benchmark
        - information_ratio: IR = alpha / tracking_error
        - tracking_error: Annualized std of active returns
        - correlation: Correlation with benchmark
    """
    # Align returns
    aligned = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if len(aligned) < 2:
        return {
            'alpha': 0.0,
            'beta': 1.0,
            'information_ratio': 0.0,
            'tracking_error': 0.0,
            'correlation': 0.0
        }
    
    port_rets = aligned['portfolio']
    bench_rets = aligned['benchmark']
    
    # Active returns
    active_rets = port_rets - bench_rets
    
    # Tracking error (annualized)
    tracking_error = active_rets.std() * np.sqrt(252)
    
    # Beta (regression: portfolio ~ benchmark)
    if bench_rets.std() > 0:
        beta = port_rets.cov(bench_rets) / bench_rets.var()
    else:
        beta = 1.0
    
    # Alpha (annualized)
    # Jensen's alpha: alpha = R_p - [R_f + beta * (R_b - R_f)]
    port_annual = port_rets.mean() * 252
    bench_annual = bench_rets.mean() * 252
    alpha = port_annual - (risk_free_rate + beta * (bench_annual - risk_free_rate))
    
    # Information Ratio
    if tracking_error > 0:
        information_ratio = alpha / tracking_error
    else:
        information_ratio = 0.0
    
    # Correlation
    correlation = port_rets.corr(bench_rets)
    
    return {
        'alpha': alpha,
        'beta': beta,
        'information_ratio': information_ratio,
        'tracking_error': tracking_error,
        'correlation': correlation
    }


def _compute_stress_tests(
    daily_returns: pd.Series,
    weights_history: list,
    all_returns: pd.DataFrame = None
) -> Dict:
    """
    Compute stress test metrics for a portfolio strategy.
    
    Parameters:
    -----------
    daily_returns : pd.Series
        Daily portfolio returns
    weights_history : list
        History of portfolio weights over time
    all_returns : pd.DataFrame
        Full asset returns matrix (for scenario analysis)
    
    Returns:
    --------
    stress_metrics : dict
        Stress test results including maximum loss in crises
    """
    # If we have full asset returns, compute scenario analysis
    if all_returns is not None and len(weights_history) > 0:
        try:
            from model.risk_management import builtin_stress_scenarios
            
            # Use most recent weights as representative
            recent_weights = weights_history[-1] if weights_history else None
            
            if recent_weights is not None:
                scenarios = builtin_stress_scenarios()
                stress_results = {}
                
                for scenario_name, scenario_returns in scenarios.items():
                    # Build scenario return vector
                    scenario_vec = pd.Series(
                        [scenario_returns.get(t, 0.0) for t in all_returns.columns],
                        index=all_returns.columns
                    )
                    
                    # Align weights
                    weights_aligned = recent_weights.reindex(all_returns.columns, fill_value=0.0)
                    
                    # Portfolio impact
                    portfolio_loss = (weights_aligned * scenario_vec).sum()
                    
                    stress_results[scenario_name] = {
                        'percentage_loss': portfolio_loss,
                        'dollar_loss_per_1m': portfolio_loss * 1_000_000
                    }
                
                # Sort by severity
                stress_results_sorted = dict(sorted(
                    stress_results.items(),
                    key=lambda x: x[1]['percentage_loss']
                ))
                
                return {
                    'stress_scenarios': stress_results_sorted,
                    'worst_case_scenario': list(stress_results_sorted.items())[0][0],
                    'worst_case_loss': list(stress_results_sorted.items())[0][1]['percentage_loss']
                }
        except Exception as e:
            pass
    
    # Fallback: historical worst periods
    return {
        'historical_worst_day': daily_returns.min(),
        'historical_worst_week': daily_returns.rolling(5).sum().min(),
        'historical_worst_month': daily_returns.rolling(21).sum().min(),
    }


def print_backtest_results(backtest_results: Dict) -> None:
    """
    Print formatted backtest results and comparison.

    Parameters:
    -----------
    backtest_results : dict
        Results from run_rolling_backtest()
    """
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 80)
    
    # Summary table
    summary_data = []
    for model_name in sorted(backtest_results.keys()):
        if model_name != 'benchmarks':
            metrics = backtest_results[model_name]['metrics']
            summary_data.append({
                'Model': model_name,
                'Annual Return': f"{metrics['annual_return']*100:.2f}%",
                'Annual Vol': f"{metrics['annual_volatility']*100:.2f}%",
                'Sharpe': f"{metrics['sharpe_ratio']:.3f}",
                'VaR 95%': f"{metrics['var_historical_95']*100:.2f}%",
                'ES 95%': f"{metrics['expected_shortfall_95']*100:.2f}%",
                'Max DD': f"{metrics['max_drawdown']*100:.2f}%",
            })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Detailed metrics for each model
    print("\n" + "-" * 80)
    print("DETAILED METRICS BY MODEL (with Institutional Risk Management)")
    print("-" * 80)
    
    for model_name in sorted(backtest_results.keys()):
        if model_name == 'benchmarks':
            continue
            
        metrics = backtest_results[model_name]['metrics']
        
        print(f"\n[{model_name}]")
        print(f"  Period: {metrics['total_years']:.2f} years ({metrics['total_days']} days)")
        print(f"\n  Returns:")
        print(f"    Cumulative Return:  {metrics['cumulative_return']*100:>8.2f}%")
        print(f"    Annual Return:      {metrics['annual_return']*100:>8.2f}%")
        print(f"    Avg Daily Return:   {metrics['avg_daily_return']*100:>8.4f}%")
        
        print(f"\n  Risk Metrics:")
        print(f"    Annual Volatility:  {metrics['annual_volatility']*100:>8.2f}%")
        print(f"    Max Drawdown:       {metrics['max_drawdown']*100:>8.2f}%")
        print(f"    Skewness:           {metrics['skewness']:>8.3f}")
        print(f"    Kurtosis:           {metrics['kurtosis']:>8.3f}")
        
        print(f"\n  Risk-Adjusted Returns:")
        print(f"    Sharpe Ratio:       {metrics['sharpe_ratio']:>8.3f}")
        print(f"    Sortino Ratio:      {metrics['sortino_ratio']:>8.3f}")
        print(f"    Calmar Ratio:       {metrics['calmar_ratio']:>8.3f}")
        
        # ===== INSTITUTIONAL RISK MANAGEMENT SECTION =====
        print(f"\n  ★ VALUE-AT-RISK (VaR) - Institutional Grade:")
        print(f"    VaR (95% conf):     {metrics['var_historical_95']*100:>8.2f}%")
        print(f"    VaR (99% conf):     {metrics['var_parametric_95']*100:>8.2f}%  (parametric)")
        print(f"    Cornish-Fisher VaR: {metrics['var_cornish_fisher_95']*100:>8.2f}%  (adjusts for fat tails)")
        
        print(f"\n  ★ EXPECTED SHORTFALL (CVaR) - Tail Risk Measure:")
        print(f"    ES (95% conf):      {metrics['expected_shortfall_95']*100:>8.2f}%")
        print(f"    ES (99% conf):      {metrics['expected_shortfall_99']*100:>8.2f}%")
        
        print(f"\n  ★ EXTREME LOSS METRICS:")
        print(f"    Worst Daily Loss:   {metrics['worst_daily_return']*100:>8.2f}%")
        print(f"    Worst 5% Days:      {metrics['worst_5pct_return']*100:>8.2f}%")
        print(f"    Worst 1% Days:      {metrics['worst_1pct_return']*100:>8.2f}%")
        print(f"    Days with >2% Loss: {metrics['largest_loss_days']:>8.0f}")
        print(f"    Crash Days (<-5%):  {metrics['crash_days']:>8.0f}")
    
    # Model comparison
    print("\n" + "-" * 80)
    print("MODEL RANKING")
    print("-" * 80)
    
    ranking_metrics = [
        ('Highest Annual Return', 'annual_return', lambda x: -x),
        ('Lowest Volatility', 'annual_volatility', lambda x: x),
        ('Highest Sharpe Ratio', 'sharpe_ratio', lambda x: -x),
        ('Highest Sortino Ratio', 'sortino_ratio', lambda x: -x),
        ('Smallest Max Drawdown', 'max_drawdown', lambda x: x),
        ('Best Calmar Ratio', 'calmar_ratio', lambda x: -x),
    ]
    
    for rank_name, metric_key, sort_key in ranking_metrics:
        sorted_models = sorted(
            backtest_results.items(),
            key=lambda x: sort_key(x[1]['metrics'][metric_key]) if isinstance(x[1], dict) and 'metrics' in x[1] else 0
        )
        sorted_models = [(m, d) for m, d in sorted_models if isinstance(d, dict) and 'metrics' in d]
        
        print(f"\n{rank_name}:")
        for i, (model_name, data) in enumerate(sorted_models[:3], 1):
            value = data['metrics'][metric_key]
            if isinstance(value, (int, float)):
                if 'Return' in rank_name or 'Drawdown' in rank_name or 'Volatility' in rank_name:
                    print(f"  {i}. {model_name}: {value*100:.2f}%")
                else:
                    print(f"  {i}. {model_name}: {value:.3f}")


def save_backtest_results(backtest_results: Dict, output_dir: str = ".") -> None:
    """
    Save backtest results to CSV files.

    Parameters:
    -----------
    backtest_results : dict
        Results from run_rolling_backtest()
    output_dir : str
        Output directory for CSV files
    """
    # Daily returns comparison
    returns_df = pd.DataFrame({
        model: data['daily_returns']
        for model, data in backtest_results.items()
    })
    returns_df.to_csv(f"{output_dir}/backtest_daily_returns.csv")
    
    # Cumulative returns comparison
    cumulative_df = pd.DataFrame({
        model: data['cumulative_returns']
        for model, data in backtest_results.items()
    })
    cumulative_df.to_csv(f"{output_dir}/backtest_cumulative_returns.csv")
    
    # Metrics summary
    metrics_data = []
    for model_name, data in backtest_results.items():
        metrics = data['metrics'].copy()
        metrics['Model'] = model_name
        metrics_data.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(f"{output_dir}/backtest_metrics.csv", index=False)
    
    print(f"\nBacktest results saved:")
    print(f"  ✓ backtest_daily_returns.csv")
    print(f"  ✓ backtest_cumulative_returns.csv")
    print(f"  ✓ backtest_metrics.csv")
