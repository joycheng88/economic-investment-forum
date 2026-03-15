# Portfolio Optimization & Financial Analysis Platform

A comprehensive financial analysis platform featuring **8 portfolio optimization models**, **professional-grade time-series forecasting**, **DCF & comparable company valuation**, **AI-powered investment chatbot**, and **institutional risk management**.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
# Open http://localhost:8501
```

**Backtest with walk-forward validation:**
```python
from backtest import run_rolling_backtest
results = run_rolling_backtest(
    tickers=['MSFT', 'NVDA', 'GOOGL', 'TSLA', 'META', 'JPM', 'JNJ', 'WMT'],
    use_param_tuning=True  # Walk-forward validation ✓
)
```

---

## 📊 Portfolio Optimization: 8-Model Ensemble

**Two-year backtest (2023-2024) with walk-forward parameter tuning:**

| Model | Sharpe | Return | Volatility | Type |
|-------|--------|--------|------------|------|
| **LASSO** | **2.17** | **46.8%** | 22.8% | Sparse optimization |
| **DRO** | 2.11 | 32.9% | **16.7%** | Robust optimization |
| HRP | 2.04 | 28.8% | 15.1% | Hierarchical clustering |
| BL | 2.01 | 27.8% | 14.8% | Bayesian allocation |
| CVaR | 1.90 | 26.7% | 15.0% | Tail risk minimization |
| GMV | 1.92 | 26.4% | 14.7% | Minimum variance |
| CAPM | 1.90 | 33.6% | 19.2% | Factor-based |
| RL | 1.67 | 28.1% | 18.3% | Reinforcement learning |
| **SPY** | 1.50 | 18.2% | 12.6% | Benchmark |

### Why Walk-Forward Validation?

Traditional backtesting trains and tests on the same data → severe overfitting.

**Walk-forward prevents this:**
- Train (60%) → Validate (40%) → Test on future (out-of-sample) ✓
- Monthly retuning adapts to market regimes
- No look-ahead bias in backtests
- More conservative, realistic performance estimates

---

## 🔮 Professional Time-Series Forecasting

**Status:** ✅ Production-ready institutional-grade system

### Core Components

| Component | Purpose |
|-----------|---------|
| **StationarityTester** | ADF + KPSS dual hypothesis testing |
| **ARIMAOptimizer** | Automatic parameter selection (AIC/BIC) |
| **GARCHVolatilityModel** | Volatility clustering detection |
| **AdvancedReturnForecaster** | 4-model ensemble with confidence bands |
| **ForecastDiagnostics** | Ljung-Box, ARCH, Jarque-Bera tests |
| **WalkForwardValidator** | Out-of-sample backtesting |

### Quick Usage

```python
from forecasting import AdvancedReturnForecaster
from data import load_data, DataConfig

config = DataConfig(start='2024-01-01', end='2026-02-27')
_, returns = load_data(['AAPL'], config)

forecaster = AdvancedReturnForecaster('AAPL', returns['AAPL'])
ensemble = forecaster.forecast_ensemble(steps=2)

# Output: Day 1: -0.154% ± 0.457%
print(f"Day 1: {ensemble['forecast'][0]*100:+.3f}% ± {ensemble['confidence_band'][0]*100:.3f}%")

# Diagnostics: Ljung-Box ✓, ARCH ✓, JB ✓, Theil U=0.78
diags = forecaster.run_diagnostics()
```

### Validation Results

✅ Stationarity: All tickers ADF p < 0.0001  
✅ Ensemble: 4 models with proper uncertainty quantification  
✅ Diagnostics: Ljung-Box (no autocorr), ARCH (clustering), JB (fat tails)  
✅ Accuracy: Theil U = 0.78 (better than naive model)  

---

## 💰 Valuation

### DCF Model
- Real-time financial data (income statement, balance sheet, cash flow)
- 5-10 year FCF projections with terminal value
- WACC via CAPM, enterprise value & intrinsic value
- **Performance:** <2-5 seconds (3-layer caching)

### Comparable Companies
- Multi-factor peer screening (industry, market cap, sector)
- P/E, P/B, EV/EBITDA, PEG multiples
- **Performance:** <3-5 seconds (parallel fetching)

---

## 💬 AI Chatbot & ML

**AI Chatbot:** Stock analysis, sector trends, market conditions, macro commentary, portfolio advice

**ML Stock Ranking:** XGBoost with 15 features
- Technical (7): RSI, MACD, Bollinger, momentum, ATR, volume, MA
- Fundamental (6): P/E, P/B, earnings growth, ROE, dividend yield, D/E
- Market (1): Sector momentum

---

## 🛡️ Institutional Risk Management

**Value-at-Risk (4 Methods):** Historical, Parametric, Cornish-Fisher, Monte Carlo

**Expected Shortfall (CVaR):** Tail risk measure (worst 5%/1% average loss)

**Stress Testing:** 6 crisis scenarios (2008, COVID, recession, vol spike, stagflation, correlation breakdown)

**Concentration:** Herfindahl Index, effective positions, top-N weights

**Performance:** All calculations <8ms (real-time ready)

---

## 📁 Project Structure

```
├── app.py                    # Streamlit dashboard (2200+ lines)
├── backtest.py               # Rolling window backtest engine
├── forecasting.py            # Time-series forecasting (1200+ lines)
├── valuation.py              # DCF & Comps (800+ lines)
├── chatbot.py                # AI assistant (1600+ lines)
├── ml_prediction.py          # ML stock ranking
├── data.py                   # Yahoo Finance data
├── param_tuning.py           # Walk-forward tuning
│
├── model/                    # Optimization models
│   ├── gmv.py, capm.py, bl.py, hrp.py
│   ├── cvar.py, lasso.py, rl.py, dro.py
│   └── risk_management.py
```

---

## ⚙️ Configuration

### Model Parameters
```python
params = {
    'GMV': {'solver': 'OSQP'},
    'CAPM': {'risk_aversion': 2.0},
    'BL': {'tau': 0.05, 'risk_aversion': 2.0},
    'HRP': {'linkage_method': 'ward'},
    'CVaR': {'alpha': 0.05},
    'LASSO': {'num_assets_target': 3, 'lasso_penalty': 0.01},
    'RL': {'n_epochs': 5},
    'DRO': {'epsilon': 1.0, 'risk_aversion': 1.0},
}
```

### Backtest Parameters
```python
run_rolling_backtest(
    tickers=['MSFT', 'NVDA', 'GOOGL', 'JPM', 'JNJ'],
    start='2023-01-01', end='2024-12-31',
    lookback_days=252,            # 1-year training
    rebalance_freq='ME',          # Monthly
    max_weight=0.15,              # 15% per asset
    transaction_cost_rate=0.001,  # 10 bps
    use_param_tuning=True         # Walk-forward ✓
)
```

---

## 📈 Dashboard Pages

1. **Home** — 8-model comparison, concentration, heatmap
2. **Model Pages** — Math, parameters, limitations
3. **Portfolio Builder** — Add/remove stocks, real-time reweighting
4. **Holdings Analysis** — Sector breakdown, correlation, factors
5. **AI Analysis** — Market context, recommendations
6. **Valuation** — DCF & Comps with sensitivity
7. **Chatbot** — Natural language intelligence
8. **Forecasting** — Time-series predictions with diagnostics
9. **Backtest** — Performance, costs, benchmarks

---

## 🎯 Use Cases

| Use Case | Features |
|----------|----------|
| **Portfolio Construction** | 8 models + walk-forward tuning |
| **Return Forecasting** | Professional ensemble + diagnostics |
| **Valuation Research** | DCF + Comps + sensitivity |
| **Investment Screening** | ML ranking + AI chatbot |
| **Risk Management** | CVaR, DRO, HRP, VaR, stress testing |
| **Low-Cost Trading** | LASSO sparse models |
| **Performance Analysis** | Holdings breakdown, factors |

---

## ✅ Key Features

✅ **8 Portfolio Models** with walk-forward parameter tuning  
✅ **Professional Forecasting** with stationarity testing, ARIMA, GARCH, ensemble  
✅ **DCF & Comps Valuation** with <5sec runtimes  
✅ **AI Chatbot** with multi-domain intelligence  
✅ **ML Stock Ranking** with 15-feature XGBoost  
✅ **Institutional Risk Management** (VaR, ES, stress testing, concentration)  
✅ **Interactive Streamlit Dashboard** with real-time visualizations  
✅ **Production Code** with error handling & type hints  

---

## 💪 Advantages

**vs Simple Backtests:**
- Walk-forward validation (no look-ahead bias) ✓
- Monthly parameter retuning (regime-adaptive) ✓
- Out-of-sample testing (realistic performance) ✓

**vs Basic Forecasting:**
- Stationarity testing (ADF + KPSS) ✓
- Automatic ARIMA optimization (AIC/BIC) ✓
- 4-model ensemble (robustness) ✓
- Statistical diagnostics (Ljung-Box, ARCH, JB) ✓

**vs Black-Box ML:**
- Transparent predictions (residual analysis) ✓
- Uncertainty quantification (confidence bands) ✓
- Statistical rigor (hypothesis testing) ✓

---

## 📝 Dependencies

**Core:** numpy, pandas, scipy, cvxpy, yfinance, scikit-learn  
**Statistics:** statsmodels, arch  
**ML:** torch, xgboost  
**Analytics:** transformers (FinBERT)  
**Dashboard:** streamlit, plotly  

**Install:** `pip install -r requirements.txt`

---

## 📚 References

**Portfolio Theory:**
- Markowitz (1952) — Mean-variance optimization
- Black-Litterman (1992) — Bayesian portfolio construction
- Rockafellar & Uryasev (2000) — CVaR optimization
- Tibshirani (1996) — LASSO sparse regression

**Time-Series Forecasting:**
- Box & Jenkins (1970) — ARIMA methodology
- Engle (1982, Nobel Prize) — GARCH volatility
- Dickey-Fuller (1979) — Stationarity testing
- Akaike (1974) — Information criterion

---

## ⚠️ Disclaimers

1. **Not Financial Advice** — Analytical tools for research only
2. **Educational Purpose** — Learning & experimentation
3. **Risk Warning** — All investments carry risk
4. **Data Limitations** — Yahoo Finance dependent
5. **No Warranty** — Provided "as is"

---

## ✅ Status

**Version:** 2.0 | **Last Updated:** Feb 28, 2026 | **Status:** Production-ready

For more information, review the code documentation in each module.
