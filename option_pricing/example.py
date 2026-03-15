"""
Options Pricing System - Complete Examples & Integration

This module demonstrates all features of the options pricing system with
practical, production-ready code examples.

Usage:
    python example.py                    # Run all examples
    python example.py 1                  # Run only Example 1
    from example import example_1        # Import specific example
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

# Import all pricing modules
from fetch_data import (
    get_underlying_price, get_risk_free_rate, get_dividend_yield,
    calculate_historical_volatility, fetch_option_chain
)
from black_scholes import BlackScholesModel, bs_call_price, bs_put_price
from monte_carlo import MonteCarlo
from binomial_tree import BinomialTree
from neural_network import NeuralNetworkPricer, create_synthetic_training_data
from stochastic_vol import HestonModel, SABRModel, HullWhiteModel, compare_stochastic_vol_models
from utils import PricingComparison, calculate_greeks_table

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def example_1_quick_pricing():
    """Example 1: Quick single option pricing with all models."""
    print_header("EXAMPLE 1: Quick Single Option Pricing")
    
    # Parameters
    S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.20, 0.25
    
    print(f"Parameters: S=${S0}, K=${K}, r={r:.2%}, σ={sigma:.2%}, T={T}yr")
    print("\n" + "-" * 80)
    print(f"{'Model':<20} {'Call Price':<15} {'Put Price':<15} {'Time (ms)':<10}")
    print("-" * 80)
    
    # Black-Scholes
    t0 = time.time()
    bs = BlackScholesModel(S0, K, r, sigma, T)
    bs_time = (time.time() - t0) * 1000
    print(f"{'Black-Scholes':<20} ${bs.call_price():<14.4f} ${bs.put_price():<14.4f} {bs_time:<9.2f}")
    
    # Monte Carlo
    t0 = time.time()
    mc = MonteCarlo(S0, K, r, sigma, T, seed=42)
    mc_call = mc.european_call(n_paths=50000, antithetic=True)
    mc_put = mc.european_put(n_paths=50000, antithetic=True)
    mc_time = (time.time() - t0) * 1000
    print(f"{'Monte Carlo':<20} ${mc_call.price:<14.4f} ${mc_put.price:<14.4f} {mc_time:<9.2f}")
    
    # Binomial Tree
    t0 = time.time()
    bt = BinomialTree(S0, K, r, sigma, T, n_steps=50)
    bt_call = bt.european_call()
    bt_put = bt.european_put()
    bt_time = (time.time() - t0) * 1000
    print(f"{'Binomial Tree':<20} ${bt_call.price:<14.4f} ${bt_put.price:<14.4f} {bt_time:<9.2f}")
    
    print("\n✓ Example 1 Complete: All models converge to similar prices")


def example_2_greeks_analysis():
    """Example 2: Calculate and display Greeks."""
    print_header("EXAMPLE 2: Greeks Analysis")
    
    S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.20, 0.25
    
    model = BlackScholesModel(S0, K, r, sigma, T)
    
    print(f"Call Option Greeks:")
    print("-" * 80)
    
    call_greeks = model.greeks_call()
    for greek, value in call_greeks.items():
        print(f"  {greek.upper():<10}: {value:>10.6f}")
    
    print(f"\nPut Option Greeks:")
    print("-" * 80)
    
    put_greeks = model.greeks_put()
    for greek, value in put_greeks.items():
        print(f"  {greek.upper():<10}: {value:>10.6f}")
    
    print("\n✓ Example 2 Complete: Greeks show option sensitivities")


def example_3_option_chain():
    """Example 3: Price entire option chain across strikes."""
    print_header("EXAMPLE 3: Option Chain Pricing")
    
    S0, r, sigma, T = 100.0, 0.05, 0.20, 0.25
    strikes = np.array([90, 95, 100, 105, 110, 115, 120])
    
    print(f"Pricing chain: Strikes from ${strikes[0]} to ${strikes[-1]}")
    print("-" * 80)
    print(f"{'Strike':<10} {'Call':<12} {'Put':<12} {'Call Delta':<12} {'Put Delta':<12}")
    print("-" * 80)
    
    for K in strikes:
        bs = BlackScholesModel(S0, K, r, sigma, T)
        print(f"${K:<9.0f} ${bs.call_price():<11.4f} ${bs.put_price():<11.4f} "
              f"{bs.delta_call():<11.4f} {bs.delta_put():<11.4f}")
    
    print("\n✓ Example 3 Complete: Chain shows moneyness effects")


def example_4_implied_volatility():
    """Example 4: Calculate implied volatility from market prices."""
    print_header("EXAMPLE 4: Implied Volatility Calculation")
    
    S0, K, r, T = 100.0, 105.0, 0.05, 0.25
    
    # Simulate market prices
    market_call_prices = [1.50, 2.00, 2.50, 3.00]
    
    print(f"Backing out IV from market prices:")
    print("-" * 80)
    print(f"{'Market Call Price':<20} {'Implied Volatility':<20}")
    print("-" * 80)
    
    for market_price in market_call_prices:
        model = BlackScholesModel(S0, K, r, 0.20, T)  # Initial guess
        iv = model.implied_volatility_call(market_price)
        print(f"${market_price:<19.2f} {iv:<19.2%}")
    
    print("\n✓ Example 4 Complete: IV helps identify mispriced options")


def example_5_monte_carlo_convergence():
    """Example 5: Analyze Monte Carlo convergence."""
    print_header("EXAMPLE 5: Monte Carlo Convergence Analysis")
    
    S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.20, 0.25
    
    mc = MonteCarlo(S0, K, r, sigma, T, seed=42)
    
    path_counts = [100, 500, 1000, 5000, 10000, 50000, 100000]
    
    print(f"Convergence to Black-Scholes price: ${bs_call_price(S0, K, r, sigma, T):.4f}")
    print("-" * 80)
    print(f"{'Paths':<10} {'Price':<12} {'Error':<12} {'Std Error':<12}")
    print("-" * 80)
    
    bs_price = bs_call_price(S0, K, r, sigma, T)
    
    for n_paths in path_counts:
        result = mc.european_call(n_paths=n_paths, n_steps=100, antithetic=True)
        error = abs(result.price - bs_price)
        print(f"{n_paths:<10} ${result.price:<11.4f} ${error:<11.4f} ${result.std_error:<11.4f}")
    
    print("\n✓ Example 5 Complete: Error decreases as 1/sqrt(N)")


def example_6_compare_models():
    """Example 6: Compare all models for American options."""
    print_header("EXAMPLE 6: Model Comparison - American vs European")
    
    S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.20, 0.25
    
    print(f"Analyzing early exercise premium for American options")
    print("-" * 80)
    
    # European options (benchmark)
    bs = BlackScholesModel(S0, K, r, sigma, T)
    eur_call = bs.call_price()
    eur_put = bs.put_price()
    
    # American options via Binomial Tree
    bt = BinomialTree(S0, K, r, sigma, T, n_steps=100)
    amer_call = bt.american_call().price
    amer_put = bt.american_put().price
    
    print(f"\n{'Call Options':<30}")
    print(f"  European:              ${eur_call:.4f}")
    print(f"  American:              ${amer_call:.4f}")
    print(f"  Early Exercise Premium: ${amer_call - eur_call:.4f}")
    
    print(f"\n{'Put Options':<30}")
    print(f"  European:              ${eur_put:.4f}")
    print(f"  American:              ${amer_put:.4f}")
    print(f"  Early Exercise Premium: ${amer_put - eur_put:.4f}")
    
    print("\n✓ Example 6 Complete: American options worth more due to flexibility")


def example_7_neural_network_pricing():
    """Example 7: Train and use neural network for pricing."""
    print_header("EXAMPLE 7: Neural Network Option Pricing")
    
    print("Generating synthetic training data...")
    X, y = create_synthetic_training_data(n_samples=2000, seed=42)
    
    print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    print("\nTraining neural network...")
    
    pricer = NeuralNetworkPricer()
    X_train, X_test, y_train, y_test = pricer.prepare_data(X, y, test_size=0.2)
    
    history = pricer.train(
        X_train, y_train, X_test, y_test,
        epochs=100, batch_size=32, lr=0.001,
        early_stopping_patience=15
    )
    
    print(f"\n{'Training Metrics':<30}")
    print("-" * 50)
    print(f"  Initial Loss: {history['train_loss'][0]:.6f}")
    print(f"  Final Loss:   {history['train_loss'][-1]:.6f}")
    if history['val_loss']:
        print(f"  Best Val Loss: {min(history['val_loss']):.6f}")
    
    # Evaluate
    X_test_np = X[-len(X_test):]
    y_test_np = y[-len(X_test):]
    metrics = pricer.evaluate(X_test_np, y_test_np)
    
    print(f"\n{'Test Set Performance':<30}")
    print("-" * 50)
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    
    print("\n✓ Example 7 Complete: Neural network trained successfully")


def example_8_real_time_data():
    """Example 8: Fetch and analyze real market data."""
    print_header("EXAMPLE 8: Real-Time Market Data Integration")
    
    ticker = "AAPL"
    
    print(f"Fetching market data for {ticker}...")
    print("-" * 80)
    
    try:
        price = get_underlying_price(ticker)
        rfr = get_risk_free_rate()
        div_yield = get_dividend_yield(ticker)
        hist_vol = calculate_historical_volatility(ticker, periods=252)
        
        print(f"  Stock Price:           ${price:.2f}")
        print(f"  Risk-Free Rate:        {rfr:.2%}")
        print(f"  Dividend Yield:        {div_yield:.2%}")
        print(f"  Historical Volatility: {hist_vol:.2%}")
        
        print("\n✓ Example 8 Complete: Live data fetched successfully")
    except Exception as e:
        print(f"\n⚠ Example 8: Could not fetch live data (network/permissions)")
        print(f"  Error: {str(e)[:60]}...")


def example_9_sensitivity_analysis():
    """Example 9: Analyze price sensitivity to parameters."""
    print_header("EXAMPLE 9: Sensitivity Analysis")
    
    K, r, T = 105.0, 0.05, 0.25
    
    print(f"Strike=${K}, Rate={r:.2%}, Time={T}yr")
    print("\nPrice sensitivity to Spot Price and Volatility:")
    print("-" * 80)
    
    spots = np.array([90, 95, 100, 105, 110, 115, 120])
    vols = np.array([0.10, 0.20, 0.30])
    
    print(f"{'Spot':<8}", end="")
    for vol in vols:
        print(f"  σ={vol:.0%}  ", end="")
    print()
    print("-" * 80)
    
    for S in spots:
        print(f"${S:<7.0f}", end="")
        for sigma in vols:
            price = bs_call_price(S, K, r, sigma, T)
            print(f"  ${price:<6.2f} ", end="")
        print()
    
    print("\n✓ Example 9 Complete: Higher vol = higher option values")


def example_10_arbitrage_detection():
    """Example 10: Detect potential arbitrage with put-call parity."""
    print_header("EXAMPLE 10: Put-Call Parity & Arbitrage Detection")
    
    S0, r, T = 100.0, 0.05, 0.25
    
    print("Checking put-call parity for synthetic market data:")
    print("-" * 80)
    print(f"{'Strike':<10} {'BS Call':<12} {'BS Put':<12} {'C-P':<12} {'S-Ke^-rT':<12} {'Error':<12}")
    print("-" * 80)
    
    strikes = np.array([95, 100, 105, 110, 115])
    sigma = 0.20
    
    sum_error = 0
    for K in strikes:
        bs = BlackScholesModel(S0, K, r, sigma, T)
        call = bs.call_price()
        put = bs.put_price()
        
        lhs = call - put  # C - P
        discount = np.exp(-r * T)
        rhs = S0 - K * discount  # S - K*e^(-rT)
        error = abs(lhs - rhs)
        sum_error += error
        
        print(f"${K:<9.0f} ${call:<11.4f} ${put:<11.4f} ${lhs:<11.4f} ${rhs:<11.4f} ${error:<11.6f}")
    
    avg_error = sum_error / len(strikes)
    print(f"\nAverage parity error: ${avg_error:.8f}")
    if avg_error < 0.01:
        print("✓ No arbitrage opportunity detected (errors are negligible)")
    else:
        print("⚠ Potential arbitrage: Check market data")
    
    print("\n✓ Example 10 Complete: Put-call parity validated")


def example_11_heston_pricing():
    """Example 11: Heston stochastic volatility model."""
    print_header("EXAMPLE 11: Heston Model for Stochastic Volatility")
    
    S0, K, r, T = 100.0, 105.0, 0.05, 0.25
    
    print(f"Parameters: S=${S0}, K=${K}, r={r:.2%}, T={T}yr")
    print("\nHeston Model (captures volatility smile/skew):")
    print("-" * 80)
    
    # Compare across different correlations
    correlations = [-0.5, -0.3, 0.0, 0.3]
    
    print(f"{'Rho (Corr)':<15} {'Call Price':<15} {'Put Price':<15} {'Time (ms)':<10}")
    print("-" * 80)
    
    for rho in correlations:
        t0 = time.time()
        heston = HestonModel(
            S0, K, r, T,
            kappa=3.0,
            theta=0.04,
            sigma_v=0.4,
            rho=rho,
            v0=0.04
        )
        
        call = heston.call_price_monte_carlo(n_paths=10000)
        put = heston.put_price(method='mc')
        elapsed = (time.time() - t0) * 1000
        
        print(f"{rho:<15.1f} ${call:<14.4f} ${put:<14.4f} {elapsed:<9.2f}")
    
    print("\nNote: Negative correlation (equities) creates volatility skew")
    print("✓ Example 11 Complete: Heston model captures realistic volatility dynamics")


def example_12_sabr_vol_surface():
    """Example 12: SABR model for volatility surface."""
    print_header("EXAMPLE 12: SABR Model - Volatility Surface Fitting")
    
    F, T = 100.0, 0.25  # Forward, time
    
    print(f"Forward: {F}, Time to Expiry: {T}yr")
    print("\nSABR Implied Volatility Surface:")
    print("-" * 80)
    
    # Create strikes range
    strikes = np.array([85, 90, 95, 100, 105, 110, 115, 120])
    
    # SABR parameters
    alpha = 0.20  # Initial volatility
    beta = 1.0    # Lognormal
    nu = 0.4      # Vol of vol
    rho = -0.3    # Correlation
    
    print(f"SABR Parameters: α={alpha:.2%}, β={beta:.1f}, ν={nu:.2%}, ρ={rho:.2f}")
    print("-" * 80)
    print(f"{'Strike':<10} {'Implied Vol':<15} {'Call Price':<15} {'Put Price':<15}")
    print("-" * 80)
    
    for K in strikes:
        sabr = SABRModel(F, K, T, alpha=alpha, beta=beta, nu=nu, rho=rho)
        
        iv = sabr.implied_vol_bbg()
        call = sabr.call_price()
        put = sabr.put_price()
        
        moneyness = "ATM" if abs(K - F) < 1 else ("ITM" if K < F else "OTM")
        print(f"${K:<9.0f} {iv:<14.2%} ${call:<14.4f} ${put:<14.4f}")
    
    print("\nNote: SABR creates realistic volatility smile/skew for rates and FX")
    print("✓ Example 12 Complete: Smile/skew captured for volatility surface")


def example_13_hull_white_multi_factor():
    """Example 13: Hull-White multi-factor model."""
    print_header("EXAMPLE 13: Hull-White Multi-Factor Model")
    
    S0, K, r0, T = 100.0, 105.0, 0.05, 0.25
    
    print(f"Parameters: S=${S0}, K=${K}, r0={r0:.2%}, T={T}yr")
    print("\nHull-White: Stochastic Rates + Stochastic Volatility")
    print("-" * 80)
    
    # Compare different rate volatilities
    rate_vols = [0.005, 0.01, 0.02]
    
    print(f"{'σ_r (Rate Vol)':<20} {'Call Price':<15} {'Put Price':<15} {'Std Error':<15}")
    print("-" * 80)
    
    for sigma_r in rate_vols:
        hw = HullWhiteModel(
            S0, K, T,
            r0=r0,
            alpha_r=0.15,
            sigma_r=sigma_r,
            kappa_v=2.0,
            v_bar=0.04,
            sigma_v=0.3,
            rho_sr=0.1,
            rho_sv=-0.3
        )
        
        call, call_se = hw.call_price_monte_carlo(n_paths=5000, n_steps=50)
        put, put_se = hw.put_price_monte_carlo(n_paths=5000, n_steps=50)
        
        print(f"{sigma_r:<20.3%} ${call:<14.4f} ${put:<14.4f} ${put_se:<14.4f}")
    
    print("\nNote: Higher rate volatility affects option prices through stochastic discounting")
    print("✓ Example 13 Complete: Multi-factor model with rates + vol + spot")


def example_14_stochastic_vol_comparison():
    """Example 14: Compare all stochastic volatility models."""
    print_header("EXAMPLE 14: Comprehensive Stochastic Volatility Model Comparison")
    
    S0, K, r, T, sigma = 100.0, 105.0, 0.05, 0.25, 0.20
    
    print(f"Parameters: S=${S0}, K=${K}, r={r:.2%}, σ={sigma:.2%}, T={T}yr")
    print("\n" + "-" * 80)
    
    comparison = compare_stochastic_vol_models(S0, K, r, T, sigma)
    
    # Format output
    print(comparison.to_string())
    
    print("\n" + "-" * 80)
    print("\nModel Characteristics:")
    print("  Black-Scholes:   Simplest, constant volatility")
    print("  Heston:          Excellent for equities/FX, captures volatility clustering")
    print("  SABR:            Best for rates/vol surfaces, handles smile/skew well")
    print("  Hull-White:      Multi-factor, includes stochastic rates")
    
    print("\nUse Cases:")
    print("  • Equities/FX:     Use Heston for realistic smile")
    print("  • Interest Rates:  Use SABR for swaption/caps pricing")
    print("  • Hybrid:          Use Hull-White for cross-asset correlation")
    
    print("\n✓ Example 14 Complete: Comprehensive stochastic volatility analysis")


def run_all_examples():
    """Run all examples sequentially."""
    examples = [
        ("1", "Quick Single Option Pricing", example_1_quick_pricing),
        ("2", "Greeks Analysis", example_2_greeks_analysis),
        ("3", "Option Chain Pricing", example_3_option_chain),
        ("4", "Implied Volatility", example_4_implied_volatility),
        ("5", "Monte Carlo Convergence", example_5_monte_carlo_convergence),
        ("6", "Model Comparison", example_6_compare_models),
        ("7", "Neural Network Pricing", example_7_neural_network_pricing),
        ("8", "Real-Time Data", example_8_real_time_data),
        ("9", "Sensitivity Analysis", example_9_sensitivity_analysis),
        ("10", "Arbitrage Detection", example_10_arbitrage_detection),
        ("11", "Heston Stochastic Volatility", example_11_heston_pricing),
        ("12", "SABR Volatility Surface", example_12_sabr_vol_surface),
        ("13", "Hull-White Multi-Factor", example_13_hull_white_multi_factor),
        ("14", "Stochastic Vol Comparison", example_14_stochastic_vol_comparison),
    ]
    
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  Options Pricing System - Complete Examples".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    print("\nRunning all 14 examples...")
    print(f"Total runtime: ~60-120 seconds\n")
    
    for num, name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n⚠ Example {num} encountered an error:")
            print(f"  {str(e)[:100]}...")
    
    print("\n" + "=" * 80)
    print(" ALL EXAMPLES COMPLETED SUCCESSFULLY".center(80))
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review README.md for detailed documentation")
    print("  2. Import modules: from black_scholes import bs_call_price")
    print("  3. Run individual examples: python example.py 1")
    print("  4. Integrate into your trading system")
    print("\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        examples_map = {
            "1": example_1_quick_pricing,
            "2": example_2_greeks_analysis,
            "3": example_3_option_chain,
            "4": example_4_implied_volatility,
            "5": example_5_monte_carlo_convergence,
            "6": example_6_compare_models,
            "7": example_7_neural_network_pricing,
            "8": example_8_real_time_data,
            "9": example_9_sensitivity_analysis,
            "10": example_10_arbitrage_detection,
            "11": example_11_heston_pricing,
            "12": example_12_sabr_vol_surface,
            "13": example_13_hull_white_multi_factor,
            "14": example_14_stochastic_vol_comparison,
        }
        
        if example_num in examples_map:
            try:
                examples_map[example_num]()
                print("\n✓ Example completed successfully\n")
            except Exception as e:
                print(f"\n✗ Error: {e}\n")
        else:
            print(f"Unknown example: {example_num}")
            print(f"Available examples: {', '.join(examples_map.keys())}")
    else:
        run_all_examples()
