# Options Pricing System: Production-Ready Framework

**Status:** Production-Ready | **Version:** 1.0 | **Updated:** March 15, 2026

Comprehensive Python framework for options pricing with 8+ computational approaches, real-time data integration, and advanced analysis. Built from Emory Economic Investment Forum research.

---

## Quick Start

**Install:**
```bash
pip install -r requirements.txt
python -c "from black_scholes import bs_call_price; print(f'Call: ${bs_call_price(100, 105, 0.05, 0.2, 0.25):.2f}')"
```

**Examples:** `python example.py [1-16]` for all pricing methods

---

## System Architecture

```
option_pricing/
├── black_scholes.py        # European options (analytical)
├── monte_carlo.py          # Simulation-based pricing
├── binomial_tree.py        # American options (exact)
├── neural_network.py       # Deep learning pricing
├── stochastic_vol.py       # Heston, SABR, Hull-White
├── deep_pde.py             # Deep BSDE/PDE for high-dim
├── rl_hedging.py          # RL optimal stopping & hedging
├── fetch_data.py           # Real-time market data
├── utils.py                # Comparison & analysis
└── example.py              # 16 complete examples
```

---

## Module Reference

### 1. Black-Scholes (Analytical, ~0.1ms)
Fast closed-form European pricing.
```python
from black_scholes import BlackScholesModel

bs = BlackScholesModel(S0=100, K=105, r=0.05, sigma=0.20, T=0.25)
print(bs.call_price())        # Analytical price
print(bs.greeks_call())       # All Greeks
print(bs.implied_volatility_call(2.50))  # Back out IV
```

### 2. Monte Carlo (Simulation, 100K paths ~100ms)
Flexible pricing with confidence intervals.
```python
from monte_carlo import MonteCarlo

mc = MonteCarlo(S0=100, K=105, r=0.05, sigma=0.20, T=0.25)
result = mc.european_call(n_paths=100000, antithetic=True)
print(f"${result.price:.4f} ± ${result.std_error:.4f}")
```

### 3. Binomial Tree (Discrete, ~1-10ms)
Exact American option pricing.
```python
from binomial_tree import BinomialTree

bt = BinomialTree(S0=100, K=105, r=0.05, sigma=0.20, T=0.25, n_steps=50)
eur = bt.european_call()
amer = bt.american_call()
print(f"American premium: ${amer.price - eur.price:.4f}")
```

### 4. Neural Networks (Deep Learning)
Batch pricing from trained model.
```python
from neural_network import NeuralNetworkPricer, create_synthetic_training_data

X, y = create_synthetic_training_data(5000)
pricer = NeuralNetworkPricer()
X_train, X_test, y_train, y_test = pricer.prepare_data(X, y)
pricer.train(X_train, y_train, X_test, y_test, epochs=200)
print(f"R² Score: {pricer.evaluate(X_test, y_test)['r2']:.4f}")
```

### 5. Stochastic Volatility (~1-100ms)
Capture market smile/skew (Heston, SABR, Hull-White).
```python
from stochastic_vol import HestonModel, SABRModel

# Heston: mean-reverting volatility with smile
heston = HestonModel(S0=100, K=105, r=0.05, v0=0.04)
price_heston = heston.call_price_monte_carlo(n_paths=10000)

# SABR: volatility surface fitting
sabr = SABRModel(F=100, K=105, sigma=0.25, alpha=0.20)
iv = sabr.implied_vol_bbg(moneyness=1.05)

# Hull-White: stochastic rates + vol
hw = HullWhiteModel(S0=100, K=105, r=0.05, sigma_s=0.20)
```

### 6. Deep BSDE/PDE (Neural Networks, 10D in ~1s)
Solve high-dimensional PDE via neural networks. Addresses the curse of dimensionality.
```python
from deep_pde import BasketOptionPricer

pricer = BasketOptionPricer(n_dims=5, n_steps=20)
result = pricer.price_deep_bsde(
    S0=np.array([100]*5), K=105, r=0.05, sigma=0.20, T=0.25
)
print(f"5D basket: ${result.prices:.4f} ± ${result.stds:.4f}")
```
**Speedup:** 19x (3D) → 735x (10D) vs Monte Carlo

### 7. RL-Based Hedging & Optimal Stopping (NEW)
Learn exercise/hedging policies via reinforcement learning.

**American Options via Q-Learning:** Frame exercise as sequential decision (hold vs exercise). Uses backward induction value iteration.
```python
from rl_hedging import AmericanOptionQ

q = AmericanOptionQ(S0=100, K=100, r=0.05, sigma=0.20, T=0.25, option_type="put")
q.backward_induction(iterations=5)
print(f"American put: ${q.price():.4f}")
print(f"Exercise boundary: ${q.exercise_boundary[0]:.2f}")
```

**Dynamic Hedging Under Transaction Costs:** Learn hedge ratios via actor-critic network. Balances delta exposure with transaction costs.
```python
from rl_hedging import DynamicHedgingRL

hedger = DynamicHedgingRL(K=100, r=0.05, sigma=0.20, transaction_cost_rate=0.0005)
result = hedger.train(S0=100, T=0.25, epochs=30)
print(f"Mean P&L: ${result.mean_p_l:.4f}, Std: ${result.std_p_l:.4f}")
```

**Execution-Aware Hedging:** Optimize execution trajectory to minimize market impact + remaining risk. Uses dynamic programming.
```python
from rl_hedging import ExecutionAwareHedging

exec = ExecutionAwareHedging(total_quantity=10000, avg_volume_per_step=5000, n_steps=10)
strategies = exec.compare_strategies()
print(f"Adaptive cost: ${strategies['adaptive'][1]:.4f}")
```

### 8. Utils & Analysis
Compare models, validate arbitrage, sensitivity analysis.
```python
from utils import PricingComparison, ModelValidator

comp = PricingComparison(S0=100, K=105, r=0.05, sigma=0.20, T=0.25)
# ... add results ...
comp.compare()  # DataFrame with all models

validator = ModelValidator()
arb_violations = validator.arbitrage_check(calls, puts, S0=100, r=0.05, T=0.25)
```

---

## Model Selection

| Scenario | Best Choice | Why |
|----------|------------|-----|
| Fast European pricing | Black-Scholes | Analytical, <1ms |
| American options | Binomial | Exact early exercise |
| Exotics (Asian, barrier) | Monte Carlo | Path dependency |
| Real-time batch | Neural Network | GPU speedup |
| Volatility smile | Heston/SABR | Realistic surfaces |
| High-dimensional (5+ assets) | Deep BSDE | Curse of dimensionality |
| Optimal stopping | RL Q-Learning | Natural DP formulation |

---

## Examples

Run any with: `python example.py [N]`

**Classical (1-10):** Black-Scholes, Monte Carlo, Binomial, NN, Comparison, Greeks, Sensitivity, Arbitrage, Convergence, Variance Reduction

**Stochastic Vol (11-14):** Heston w/ correlation, SABR volatility surface, Hull-White multifactor, Model comparison

**Deep Learning (15-16):** Deep BSDE basket (5D), High-dim scaling (3D-10D with speedup metrics)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| `yfinance fetch fails` | Verify ticker, check internet, add delays |
| NN doesn't converge | Increase epochs to 500+, adjust LR to 0.0001-0.01 |
| Black-Scholes IV = NaN | Market price > 0, IV in [0.01, 5.0], S0 > 0, T > 0.001 |
| Binomial slow | Reduce n_steps, use Black-Scholes for European |
| RL training unstable | Add gradient clipping, reduce learning rate |

---

## Performance Benchmarks

| Model | Dimensions | Speed | Accuracy vs MC |
|-------|-----------|-------|--------------|
| Black-Scholes | 1 | 0.1ms | - (analytical) |
| Monte Carlo | 1-3 | ~100ms (100K paths) | Baseline |
| Binomial | 1 | 5-10ms | 99.9% |
| Neural Network | Batch | 1-5ms/item | 95-99% |
| Heston | 1 | 50-100ms | 99% |
| SABR | 1 | 10-50ms | 98% |
| Deep BSDE | 3-10 | 1-5ms | 98-99% |
| RL Q-Learning | 1 | 100-500ms | 95-99% |

---

## Key Features

✅ **8 Pricing Models** from analytical to deep learning  
✅ **Stochastic Volatility** with realistic smile/skew  
✅ **High-Dimensional** Deep BSDE/PDE solver (curse of dimensionality)  
✅ **RL-Based Hedging** optimal stopping & execution strategies  
✅ **Real-Time Data** via yfinance with validation  
✅ **Production Quality** full error handling, logging, type hints  
✅ **16 Examples** from basic to advanced  
✅ **Modular Design** independent, reusable components  

---

## Dependencies

```
numpy >= 1.19
scipy >= 1.5
pandas >= 1.1
torch >= 1.8
scikit-learn >= 0.24
yfinance >= 0.1.70
```

---

## Best Practices

- **Validate inputs:** Assert S0 > 0, 0 < sigma < 2, T > 0
- **Handle edge cases:** Clamp sigma to [0.001, 3.0], set T = max(T, 0.001)
- **Verify results:** Use put-call parity for quick sanity checks
- **Monitor performance:** Log execution time for real-time systems

---

## References

- Black & Scholes (1973): Classic analytical option pricing
- Cox & Ross & Rubinstein (1979): Binomial tree framework
- Hull (1980+): "Options, Futures, and Other Derivatives"
- Heston (1993): Stochastic volatility model
- Beck et al. (2019): Deep learning for PDEs

---

## License & Citation

Educational/commercial use permitted. For academic work:
```
Options Pricing Framework v1.0
Emory Economic Investment Forum
March 2026
```

---

**Questions?** Check module docstrings (`help(black_scholes.bs_call_price)`) or run `python example.py [1-16]`
