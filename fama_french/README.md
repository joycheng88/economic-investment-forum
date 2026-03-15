# Fama-French Factor Models - Portfolio Analysis Framework

A comprehensive Python implementation of CAPM, Fama-French 3-factor, and Fama-French 5-factor models for quantitative portfolio analysis.

## Project Structure

### Core Modules

1. **fetch_data.py** (700+ lines)
   - Multi-source data integration pipeline
   - **Sources:**
     - Ken French's 5-factor library: Mkt-RF, SMB, HML, RMW, CMA
     - yfinance: Individual stock prices (BAC, BLK, JPM, MS, MET, CG, CME)
     - FRED: 3-month Treasury Bill Rate (DGS3MO)
   - **Outputs:**
     - `fama_french.csv`: Merged dataset with 1,205 daily observations
     - Daily and monthly frequency support
     - Excess returns computed for all assets

2. **capm.py** (350+ lines)
   - Single-factor Capital Asset Pricing Model
   - **Analysis:**
     - In-sample: Full dataset regression with OLS
     - Out-of-sample: 5-fold cross-validation
     - Performance metrics: Sharpe ratio, Jensen's alpha, Information ratio, Treynor ratio
   - **Outputs:**
     - `capm_in_sample_daily.csv`: In-sample daily results
     - `capm_in_sample_monthly.csv`: In-sample monthly results (if available)
     - `capm_out_of_sample.csv`: Cross-validation performance metrics

3. **ff3.py** (450+ lines)
   - Fama-French 3-factor model (Mkt-RF, SMB, HML)
   - **Analysis:**
     - Factor regression with all diagnostics
     - Factor significance testing vs CAPM
     - Cross-validation error metrics
   - **Outputs:**
     - `ff3_in_sample_daily.csv`: In-sample regression results
     - `ff3_factor_significance.csv`: SMB/HML joint significance test
     - `ff3_out_of_sample.csv`: Out-of-sample prediction errors

4. **ff5.py** (500+ lines)
   - Fama-French 5-factor model (Mkt-RF, SMB, HML, RMW, CMA)
   - **Analysis:**
     - Full 5-factor regression
     - Comprehensive model comparison vs FF3 and CAPM
     - Cross-validation framework
   - **Outputs:**
     - `ff5_in_sample_daily.csv`: In-sample regression results
     - `ff5_model_comparison.csv`: R², AIC, BIC comparison table
     - `ff5_out_of_sample.csv`: Cross-validation metrics

5. **visual.py** (500+ lines)
   - Comprehensive visualization and analysis framework
   - **Visualizations:**
     - Efficiency frontier (risk-return scatter)
     - Factor loadings comparison across models
     - Model fit comparison (R², AIC, BIC)
     - Statistical significance heatmap (p-values)
     - Performance metrics comparison
     - Out-of-sample error analysis
   - **Outputs:**
     - 6 publication-quality PNG charts
     - Summary comparison table

## Usage

### Step 1: Fetch Data
```bash
python fetch_data.py
```
Generates: `fama_french.csv` (merged market data + factors + excess returns)

### Step 2: Run Factor Models
```bash
python capm.py      # Single-factor model
python ff3.py       # 3-factor model with significance testing
python ff5.py       # 5-factor model with full comparison
```

Each model generates:
- In-sample regression results (CSV)
- Cross-validation performance metrics (CSV)
- Model diagnostics (R², adj-R², AIC, BIC)
- Risk-adjusted returns (Sharpe ratio, Jensen's alpha, Information ratio)

### Step 3: Generate Visualizations (Optional)
```bash
python visual.py
```
Creates: 6 analysis charts + summary table

## Data Structure

### Input: fama_french.csv
- **Frequency:** Daily observations (1,205 rows)
- **Price Data:** 7 financial stocks (BAC, BLK, JPM, MS, MET, CG, CME)
- **Returns:** Daily log returns for each asset + equal-weight portfolio
- **Factors:** Mkt-RF, SMB, HML, RMW, CMA (5-factor system)
- **Risk-Free Rate:** RF (from Ken French data)
- **Excess Returns:** All returns minus risk-free rate

### Output: Model Results
Each model generates CSV files with:
- **Alpha:** Intercept (abnormal return)
- **Betas:** Factor loadings (sensitivity)
- **Standard Errors:** Coefficient uncertainty
- **P-values:** Statistical significance (two-tailed)
- **R-squared:** Model fit quality
- **Performance Metrics:** Sharpe ratio, Jensen's alpha, Information ratio

## Key Assumptions

1. **Daily Data:** Primary analysis on daily frequency (monthly insufficient data)
2. **Equal-Weight Portfolio:** Averages returns across all 7 stocks
3. **Log Returns:** Computed as ln(P_t / P_t-1)
4. **Excess Returns:** Return - Risk-free rate (basis points)
5. **OLS Regression:** Normal errors, homoscedasticity assumed
6. **Cross-Validation:** 5-fold time-series KFold on daily data

## Results Summary

### Model Fit (R²)
- **CAPM:** 0.6-3.1% (typically weakest)
- **FF3:** 2.8-3.9% (modest improvement)
- **FF5:** 3.0-3.9% (marginal gain)

### Factor Significance
- **Market Beta:** Highly significant across all assets (p < 0.001)
- **SMB/HML:** Mixed significance (p depends on asset)
- **RMW/CMA:** Generally less significant

### Execution

All modules include:
✅ Error handling for missing data
✅ Logging for debugging
✅ CSV export for downstream analysis
✅ Descriptive statistics
✅ Regression diagnostics

## Performance Metrics Explained

- **Sharpe Ratio:** Excess return per unit risk
- **Jensen's Alpha:** Abnormal return after accounting for beta
- **Information Ratio:** Active return per unit active risk
- **Treynor Ratio:** Excess return per unit systematic risk

## Dependencies

```
pandas>=3.0
numpy>=2.4
yfinance>=0.2
statsmodels
scikit-learn
scipy
matplotlib
seaborn
requests
```

## File Cleanup

Removed intermediate files:
- Individual stock CSVs (BAC.csv, BLK.csv, etc.)
- Old merged data (merged_data.csv)
- Trading signal files (legacy format)

**Final Size:** 2.1 MB (core modules: 84 KB, data: 502 KB, results: ~2 KB each)

## Future Enhancements

1. Add monthly frequency support (currently daily-only)
2. Implement rolling-window factor analysis
3. Add risk factor analysis (VIX, credit spreads)
4. Sector-level analysis
5. Time-varying betas
6. Machine learning factor extraction

---

**Last Updated:** March 15, 2026
**Framework:** Fama-French 5-Factor Model
**Data Period:** 5 years (March 2021 - March 2026)
**Assets:** 7 financial stocks + equal-weight portfolio
