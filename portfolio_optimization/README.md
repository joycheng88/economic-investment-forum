# Portfolio Optimization & Financial Analysis Platform

A comprehensive platform featuring **8 optimization models**, **forecasting**, **DCF/Comps valuation**, **AI chatbot**, and **risk management**.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run portfolio_optimization/app.py  # http://localhost:8501
```

## 📊 Models & Performance

**2-Year Backtest (Walk-Forward Validated):**

| Model | Sharpe | Return | Volatility | Type |
|-------|--------|--------|------------|------|
| **LASSO** | **2.17** | **46.8%** | 22.8% | Sparse |
| DRO | 2.11 | 32.9% | 16.7% | Robust |
| HRP | 2.04 | 28.8% | 15.1% | Hierarchical |
| SPY | 1.50 | 18.2% | 12.6% | Benchmark |

**All 8 Models:** LASSO, DRO, HRP, BL, CVaR, GMV, CAPM, RL

### Why Walk-Forward Validation?
- Prevents look-ahead bias with monthly parameter retuning
- Monthly rebalancing adapts to market regimes
- Realistic out-of-sample performance

## 🔮 Time-Series Forecasting

**ARIMA + GARCH + Ensemble:**
- Stationarity testing (ADF + KPSS)
- Automatic parameter optimization (AIC/BIC)
- 4-model ensemble with uncertainty bands
- Statistical diagnostics (Ljung-Box, ARCH, JB)
- Accuracy: Theil U = 0.78

## 💰 Valuation

**DCF Model:** Real-time financials, 5-10 year FCF, WACC validation (~3-5 sec)  
**Comparables:** Industry peers, P/E/P/B/EV/EBITDA multiples (~3-5 sec)  
**Any Stock:** Support for unlimited tickers (custom symbol input)

## 💬 AI Chatbot

**Enhanced Question Routing (April 2026):**

| Question Type | Example | Analysis |
|---|---|---|
| **Stock Addition** | "Add MSFT to portfolio?" | Correlation, quality score, sizing |
| **Fed/Macro** | "Fed rate cuts impact?" | 3 scenarios + winners/losers |
| **Sector** | "Semiconductor performance?" | Key players, trends, valuations |
| **Portfolio** | "Portfolio stress test?" | Risk factors + recommendations |

**Specialized Sub-Sector Analyzers:**
- 🖥️ **Semiconductors:** NVDA, AMD, QCOM, TSM, INTC, ASML (real-time pricing, AI boom analysis, geopolitics)
- 💊 **Pharma:** GLP-1 drugs, pipelines, FDA approvals (framework ready)
- 🏦 **Banking:** Interest rate sensitivity, credit cycles (framework ready)

## 🛡️ Risk Management

**Methods:** VaR (Historical/Parametric/Cornish-Fisher/Monte Carlo), CVaR, Stress Testing (6 scenarios), Concentration  
**Performance:** All <8ms (real-time ready)

## 📁 Project Structure

```
├── app.py              # Streamlit dashboard + 2 UI optimization modules
├── chatbot.py          # AI assistant with smart routing (3500+ lines)
├── forecasting.py      # Time-series (ARIMA/GARCH/ensemble)
├── valuation.py        # DCF & Comps
├── backtest.py         # Walk-forward engine
├── ml_prediction.py    # XGBoost stock ranking (15 features)
└── model/              # 8 optimization models + risk management
```

## 📈 Dashboard Pages

**Portfolio:** Builder, holdings analysis | **Valuation:** DCF, Comps | **Strategy:** AI analysis, forecasting | **Risk:** VaR, stress testing | **Chat:** Natural language Q&A | **Backtest:** Performance validation

## 🎯 Use Cases

- **Portfolio construction** with 8 competing models
- **Return forecasting** with statistical rigor
- **Valuation research** with sensitivity analysis
- **Investment screening** with ML + AI
- **Risk monitoring** with institutional methods
- **Performance analysis** with attribution

## ✅ Key Features

✅ 8 models + walk-forward tuning | ✅ ARIMA/GARCH ensemble | ✅ DCF/Comps any stock | ✅ AI chatbot with smart routing | ✅ ML ranking (15 features) | ✅ Institutional risk tools | ✅ Interactive dashboard

## 📝 Dependencies

```
numpy pandas scipy cvxpy yfinance scikit-learn statsmodels arch
torch xgboost transformers streamlit plotly
```

## ⚠️ Disclaimer

Educational tools for research only. Not financial advice. All investments carry risk.

**Status:** Production-ready (v2.0) | **Last Updated:** Apr 10, 2026
