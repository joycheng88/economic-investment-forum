# EEIF Complete Implementation Plan & Work Division

## Executive Summary

This project integrates real-time sentiment analysis with portfolio optimization, valuation models, and risk management to create a production-ready system for data-driven investment decisions. The platform combines market sentiment from multiple sources (Reddit, Twitter, news aggregators) with quantitative portfolio construction using eight different optimization strategies. The end goal is a unified decision support system that leverages sentiment signals, valuation metrics, and technical indicators to guide investment allocation.

The system focuses initially on the GLP-1 sector but is designed to scale to other markets and sectors over time.

---

## Phase 1: Core Infrastructure (Complete)

The foundation consists of four major components that are already implemented. The sentiment analysis pipeline uses weak supervised labeling via VADER (a rule-based sentiment analyzer) combined with TF-IDF and Word2Vec feature engineering. Two models—Logistic Regression and Neural Network—are trained independently and then combined through ensemble prediction using weighted averaging based on their F1-scores. Real-time data collection integrates Reddit (via PRAW API) and NewsAPI, with an automated alert system that detects sentiment shifts, extreme values, and model disagreements.

Portfolio optimization includes eight distinct models that each solve the portfolio allocation problem differently. These models have completed walk-forward backtesting (rolling window validation that prevents look-ahead bias) with monthly parameter retuning to adapt to changing market regimes.

Time-series forecasting is implemented through stationarity testing (ADF and KPSS tests), ARIMA optimization with automatic parameter selection, GARCH volatility modeling to capture volatility clustering, and an ensemble forecasting approach combining multiple models with confidence bands. Diagnostic tests (Ljung-Box, ARCH, Jarque-Bera) validate the quality of forecasts.

Valuation models include discounted cash flow (DCF) analysis with 5-10 year projections, comparable company analysis using multi-factor peer screening, real-time financial data, and WACC calculation through CAPM.

---

## Phase 2: Integration & Real-Time Index (New Work Required)

### 1. Real-Time Sentiment Index (Priority: CRITICAL)

**Purpose and Concept:** The real-time sentiment index aggregates raw sentiment signals from multiple sources into a single, continuously updated metric that represents market sentiment toward GLP-1 medications and related topics. This index becomes a tradeable, benchmarkable metric that can be used to inform investment decisions.

The index uses multiple data sources including Reddit, Twitter/X, NewsAPI, and financial news aggregators to ensure comprehensive market coverage. A distributed collection architecture handles scaling as data volume increases. The key innovation is time-decay weighting, which mathematically emphasizes recent sentiment observations while gradually reducing the influence of older data. The formula combines positive and negative sentiment proportions:

$$SI(t) = 100 + 10 \times [0.6 \times P_{pos}(t) - 0.4 \times P_{neg}(t)]$$

where each observation is weighted by an exponential decay function based on time lag. Low-pass filtering reduces noise spikes that result from random fluctuations in sentiment.

The system produces multiple sentiment variants. The daily sentiment index provides a 0-100 scale representation of pure sentiment. Momentum sentiment uses 20- and 50-day moving averages to identify trends. Regime detection identifies whether the market is trending, ranging, or experiencing reversals. A volatility index measures sentiment uncertainty and model disagreement.

**Deliverables include** a multi-source data aggregator that pulls from Reddit, Twitter, news feeds, and RSS sources; sentiment index calculation with time-decay weighting; PostgreSQL/MongoDB schema for time-series storage; API endpoints for real-time queries; and Plotly/Streamlit dashboard visualization.

**Responsible team:** Backend/Data Engineer leads with NLP/ML Engineer support. Time estimate: 2-3 weeks.

### 2. Sentiment-Driven Portfolio Construction (Priority: HIGH)

**Purpose and Concept:** This component flows sentiment signals from the index into actual portfolio weight adjustments using the eight portfolio optimization models. Rather than ignoring sentiment, the system translates sentiment extremes into optimization constraints that dynamically adjust sector and security weights.

When sentiment is extremely positive, the system increases the maximum allowable weight on GLP-1-related companies and decreases the maximum weight on competing therapeutic approaches. Conversely, negative sentiment triggers more conservative positioning. Sector-level sentiment mapping groups companies into the GLP-1 ecosystem, allowing sentiment to influence allocation at the sector level rather than individual stock level.

The constraint generator transforms sentiment extremes into min/max weight bounds that feed into whichever optimization model is active (LASSO, DRO, HRP, Bayesian, CVaR, GMV, CAPM, or RL). An A/B testing framework allows comparison of sentiment-adjusted portfolios against benchmark allocations to measure the actual impact of sentiment on returns.

**Deliverables include** sector-level sentiment aggregator; dynamic constraint builder for optimization models; integration layer connecting sentiment to the existing portfolio optimization module; backtesting framework for sentiment strategies; and performance comparison reports showing sentiment-adjusted vs. benchmark returns.

**Responsible team:** Quant/Portfolio Analyst leads with Data Engineer and ML Engineer support. Time estimate: 2-3 weeks.

### 3. Multi-Factor Scoring System (Priority: HIGH)

**Purpose and Concept:** A single composite score synthesizes sentiment, fundamental, and technical analysis into one number that represents investment attractiveness. Rather than requiring analysts to manually weigh multiple signals, the scoring system automates this integration and provides transparency about which factors drive investment recommendations.

The sentiment component (40% weight) includes absolute sentiment level, sentiment momentum using 20/50-day moving average crossovers, and detectability of manipulation through extreme value analysis. The fundamental component (35% weight) incorporates DCF valuation upside/downside percentage, P/E, P/B, and EV/EBITDA ratios relative to peers, earnings and revenue growth metrics, and quality metrics like ROE and debt ratios. The technical component (25% weight) includes RSI, MACD, Bollinger Bands, support/resistance levels, momentum indicators, and volume trends. Each sub-score normalizes to a 0-100 scale before combining into the final score.

The composite score formula is:
$$Score = 0.40 \times Sentiment + 0.35 \times Fundamental + 0.25 \times Technical$$

This approach forces the team to think systematically about weights. Sensitivity analysis reveals which factors drive scores most and allows for weight optimization based on out-of-sample backtesting performance.

**Deliverables include** a scoring engine with weighted components; real-time score calculation and updates; sensitivity analysis showing factor contributions; API endpoints for score queries; and dashboard visualization of composite scores across the portfolio.

**Responsible team:** Quant Analyst leads with Data Scientist and Backend Engineer support. Time estimate: 2 weeks.

### 4. Advanced Forecasting Integration (Priority: MEDIUM)

**Purpose and Concept:** Ensemble return and volatility forecasts provide forward-looking inputs to portfolio optimization. Instead of assuming constant returns and volatilities, the optimizer works with predicted values for the next 5-10 days, allowing the portfolio to adapt to expected market conditions. Covariance matrix forecasting captures correlation changes, ensuring that portfolio diversification benefits remain valid.

Forecasts include ARIMA (autoregressive integrated moving average), GARCH (generalized autoregressive conditional heteroskedasticity), and machine learning ensemble approaches. Confidence bands quantify forecast uncertainty, allowing the risk management system to account for forecast error when computing portfolio risk metrics.

**Deliverables include** ensemble return forecaster for 5-10 day horizons; volatility forecast integration; covariance matrix forecasting; confidence interval computation; and daily or intraday update frequency pipelines.

**Responsible team:** Data Scientist/Quant leads with ML Engineer support. Time estimate: 1-2 weeks.

### 5. Risk Management & Monitoring System (Priority: HIGH)

**Purpose and Concept:** Institutional-grade risk oversight requires multiple complementary risk measures. Value-at-Risk (VaR) determines the maximum loss expected with a given confidence level (e.g., 1-day 95% VaR). Expected Shortfall (CVaR) measures tail risk by averaging losses beyond the VaR threshold, capturing the severity of worst-case scenarios.

The system implements four VaR calculation methods. Historical VaR uses the empirical distribution of past returns. Parametric VaR assumes returns follow a normal distribution. Cornish-Fisher VaR adjusts for non-normal distributions using higher moments (skewness and kurtosis). Monte Carlo VaR uses simulation to estimate tail risk.

Stress testing applies six shock scenarios: 2008 financial crisis conditions, COVID-style sudden drops, recession-like environments, volatility spikes, stagflation regimes, and correlation breakdowns (when historically uncorrelated assets suddenly move together). Concentration risk metrics identify whether the portfolio is too exposed to a few holdings. Greeks calculation (Delta, Gamma, Vega, Theta) manages option positions.

Real-time alerts trigger when portfolio risk exceeds thresholds, enabling immediate response to emerging risks.

**Deliverables include** risk metrics calculator executing in less than 8 milliseconds; stress testing engine with scenario library; real-time risk dashboard; alert system for risk breaches; and historical risk analytics archive.

**Responsible team:** Risk Analyst/Quant leads with Backend Engineer and Data Scientist support. Time estimate: 2-3 weeks.

### 6. API & Backend Infrastructure (Priority: CRITICAL)

**Purpose and Concept:** A production-ready REST API serves all models and indices to frontends, reports, and external consumers. The backend consists of FastAPI (modern Python framework), PostgreSQL for relational data, TimescaleDB for time-series compression, Redis for caching frequent queries, and MongoDB for flexible document storage.

The API exposes endpoints for sentiment (current index, historical data, forecasts, alerts), portfolio (allocations from 8 models, backtest results, live performance, current holdings), risk (VaR metrics, stress test results, Greeks, risk breach alerts), valuation (DCF intrinsic values, comparable multiples, growth scenarios), and forecasts (predicted returns, volatility forecasts, confidence bands).

Authentication uses JWT tokens. Rate limiting prevents API abuse. Comprehensive logging enables debugging and monitoring. Swagger documentation auto-generates from code annotations, enabling easy API exploration for developers.

**Deliverables include** FastAPI backend application (2,000+ lines); PostgreSQL and TimescaleDB setup with migration scripts; Redis caching layer; Docker containerization for ML models; comprehensive API documentation; load testing and performance optimization targeting sub-200ms response times; and monitoring/logging setup.

**Responsible team:** Backend Engineer leads with DevOps Engineer and Full-Stack Developer support. Time estimate: 3-4 weeks.

### 7. Web Dashboard & Monitoring (Priority: MEDIUM)

**Purpose and Concept:** A production-grade dashboard makes the entire system observable and actionable for decision-makers. The dashboard displays real-time sentiment index with animated time-series charts, portfolio allocation heatmaps showing all eight models' recommendations side-by-side, risk metrics (VaR, CVaR, concentration), performance tracking (Sharpe ratio, returns, drawdowns), and alert notifications. Additional views show return/volatility forecasts with confidence intervals, valuation comparisons (DCF vs. market price, peer multiples), and an interactive backtesting simulator where analysts can adjust parameters and immediately see performance impact.

Dashboard technology can use React.js for rich interactivity or Streamlit for rapid prototyping. Real-time updates use WebSocket connections to the API, enabling sub-second data feeds. Export functionality allows PDF reports, Excel sheets, and CSV downloads for further analysis or distribution.

**Deliverables include** React.js frontend (1,500+ lines); WebSocket connection for real-time data; export functionality (PDF, Excel, CSV); user authentication and role-based access control; mobile-responsive design; and performance optimization targeting less than 2-second load times.

**Responsible team:** Full-Stack Developer/Frontend Engineer leads with Backend Engineer and UI/UX Designer support. Time estimate: 2-3 weeks.

---

## Work Division & Team Structure

### Team Organization

The project requires four specialized roles reporting to a project director/quant lead who oversees overall architecture and integration strategy. The project director handles quant methodology oversight and performance validation.

**Team 1: ML/NLP Engineer (Person A)** owns sentiment analysis and forecasting. Current work includes sentiment index enhancement (adding Twitter and Bloomberg to data sources, implementing time-decay weighting, filtering out noise), forecasting integration (connecting ARIMA and GARCH models to the portfolio engine), and ongoing model monitoring (detecting sentiment model performance degradation, setting up automatic retraining triggers).

Deliverables include the SentimentIndexCalculator class (300+ lines), ForecastingAdapter class, model monitoring dashboard, and deployment documentation. Tools include Python, NLTK, spaCy, TensorFlow, scikit-learn, APScheduler, and MLflow.

**Team 2: Quant Analyst / Risk Officer (Person B)** handles portfolio construction and risk management. Work includes sentiment-driven portfolio construction (mapping sentiment to sector weights, generating dynamic constraints), multi-factor scoring system (designing weights, optimizing based on backtest results), risk management system (implementing VaR, CVaR, stress testing, Greeks calculation), and risk monitoring (configuring real-time alerts and threshold settings).

Deliverables include SentimentPortfolioBuilder (400+ lines), CompositeScoreCalculator (200+ lines), RiskMetricsCalculator (500+ lines), backtesting results, and risk methodology documentation. Tools include Python, pandas, scipy, statsmodels, and Bloomberg API.

**Team 3: Backend Engineer (Person C)** provides the infrastructure foundation. Work includes FastAPI backend development (2,000+ lines covering all endpoints), database architecture (PostgreSQL and TimescaleDB schema design), caching layer implementation (Redis for frequent queries), and deployment infrastructure (Docker containerization, CI/CD setup).

Deliverables include the FastAPI application, database schema with migration scripts, Docker setup files, API Swagger documentation, deployment guide, and logging/monitoring configuration. Tools include FastAPI, Pydantic, PostgreSQL, TimescaleDB, Redis, Docker, and GitHub Actions.

**Team 4: Frontend/Full-Stack Developer (Person D)** builds the user interface. Work includes dashboard frontend (React.js, 1,500+ lines with eight main views), real-time updates (WebSocket integration), reporting functionality (PDF generation, Excel export), and performance optimization (sub-2-second load times, mobile responsiveness).

Deliverables include the React frontend, WebSocket integration, report generator, user documentation, and design assets. Tools include React.js, TypeScript, Plotly.js, Material-UI or Ant Design, and PDF/Excel libraries.

---

## Implementation Timeline

Foundation and integration work occurs during weeks 1-2 with code reviews. Multi-team parallel development runs weeks 2-3 on core components. Integration and testing continue through weeks 4-5. Production work including Docker, CI/CD, and reporting features happen weeks 6-7. Optimization and end-to-end testing occur week 8. Final launch preparation with training and deployment happens weeks 9-10. Total estimate: 8-10 weeks.

---

## Core Technology: Portfolio Optimization Models

Understanding the eight portfolio optimization models is critical for this project. Each solves the portfolio problem with different tradeoff assumptions. The goal is to choose which constraints and inputs (including sentiment signals) to use based on market conditions and performance.

LASSO (Least Absolute Shrinkage and Selection Operator) creates sparse portfolios where many weights are exactly zero. This naturally eliminates low-conviction positions and creates concentrated portfolios that are easier to manage and lower in transaction costs. LASSO works best when conviction is high and portfolio simplicity is valued.

Distributionally Robust Optimization (DRO) assumes uncertainty about the true return distribution. Rather than assuming a single estimated distribution (which ignores estimation error), DRO constructs confidence sets around the estimated distribution and optimizes for the worst case within that set. This produces conservative portfolios that perform better when estimates are unreliable.

Hierarchical Risk Parity (HRP) first clusters assets based on correlation structure ("which assets move together?"), then assigns risk budgets hierarchically through the tree structure. This approach is more stable than mean-variance when correlations change and produces more intuitive sector allocations.

Bayesian optimization performs portfolio optimization while incorporating Bayesian priors about expected returns based on historical relationships. This smooths return estimates and reduces estimation noise compared to raw historical averages.

Conditional Value-at-Risk (CVaR) optimization minimizes tail risk (the worst expected losses) rather than total volatility. This produces portfolios that perform better in crisis scenarios even if their overall volatility appears similar.

Global Minimum Variance (GMV) simply minimizes portfolio volatility without considering expected returns. This is the most conservative approach and works well for protective allocations or when conviction on returns is very low.

CAPM (Capital Asset Pricing Model) uses the market factor and risk factors to construct systematic, factor-aligned portfolios. This approach is theoretically grounded and produces diversified allocations aligned with systematic risk exposure.

Reinforcement Learning (RL) learns optimal portfolio weights through simulation feedback, adapting dynamically to changing market environment. This data-driven approach captures complex patterns but requires careful validation to avoid overfitting.

---

## Sentiment Analysis: Research Direction for the Team

Before implementation begins, the team must understand what sentiment analysis is and study existing sentiment indices. This research should establish baselines and prevent duplicating existing work.

**For the ML Engineer:** Research the following questions. What is sentiment analysis? How does it differ from emotion detection or opinion mining? Study VADER (the rule-based lexicon approach we use), but also investigate transformer-based approaches like DistilBERT and RoBERTa fine-tuned on financial text. Why did we choose VADER initially? What are its limitations? What are examples of sentiment indices already deployed in finance (e.g., CNN Fear & Greed Index, Refinitiv Sentiment Indicators)? How do they aggregate sentiment and what update frequency do they use? Look at the University of Michigan Sentiment Index in macroeconomics—how does sentiment become tradeable?

The key finding should be that sentiment is most valuable when forward-looking (predicting future returns), cross-sectional (comparing one stock to sector average), and regime-dependent (sentiment strength varies by market environment). Study how professional indices handle false signals and data quality issues.

**For the Quant Analyst:** Research how sentiment signals integrate into portfolio construction. Study papers on sentiment-driven anomalies (do stocks with high sentiment underperform or outperform?). Look at case studies of sentiment-based funds (good and bad examples). Understand why sentiment alone is insufficient and why multi-factor validation matters. How do you distinguish between sentiment-driven price momentum and genuine fundamental changes?

**Practical Assignment:** Each team member should identify one publicly available sentiment index today, study how it's constructed, and present findings in the next meeting. Document effective approaches and common pitfalls.

---

## Key Quantitative Concepts

The system requires understanding several quantitative methods. Time-decay weighting emphasizes recent observations exponentially. The formula $weight_i = e^{-\lambda \cdot age_i}$ gives maximum weight to today's data and exponentially less weight as observations age. The decay parameter $\lambda$ controls responsiveness—higher values make the index more sensitive to recent changes.

Stationarity testing determines whether a time series has a constant mean and variance (required for ARIMA). The ADF test and KPSS test provide complementary perspectives. Time series must usually be differenced (taking returns instead of prices) to achieve stationarity.

Value-at-Risk answers the question "what's the maximum I could lose in a given day with 95% confidence?" Four methods calculate this: Historical VaR uses empirical return distribution, Parametric assumes normality (fast but inaccurate for tail risk), Cornish-Fisher adjusts for skewness/kurtosis, and Monte Carlo simulates paths. Real portfolios require multiple methods—no single approach is always correct.

Portfolio optimization maximizes return per unit of risk. The classical formulation is $\text{maximize } \mu^T w - \lambda \sigma^2(w)$ subject to $\sum w_i = 1$. Sentiment-adjusted versions add constraints like $w_i^{min}(sentiment) \leq w_i \leq w_i^{max}(sentiment)$ to encode conviction.

---

## ✅ CONSTRUCTION CHECKLIST: REAL-TIME INDEX

### Prerequisites
- [ ] Database infrastructure (PostgreSQL + TimescaleDB)
- [ ] Data sources configured (Reddit API key, NewsAPI key, etc.)
- [ ] Model artifacts saved (.joblib, .h5 files)
- [ ] API skeleton deployed (FastAPI framework)

### Core Components
- [ ] **Sentiment Index Calculator**
  - [ ] Multi-source data aggregator class
  - [ ] Time-decay weighting function
  - [ ] Outlier detection & handling
  - [ ] Index computation formula
  - [ ] Unit tests & validation

- [ ] **Real-Time Update Pipeline**
  - [ ] Scheduled collection (hourly/daily)
  - [ ] Deduplication logic
  - [ ] Model inference (ensemble prediction)
  - [ ] Index recalculation
  - [ ] Database append (time-series storage)

- [ ] **Monitoring & Alerts**
  - [ ] Alert detection (sentiment shifts, extremes, model disagreement)
  - [ ] Alert storage & history
  - [ ] Alert delivery (email, API, dashboard)
  - [ ] Alert configuration (thresholds, severity)

- [ ] **Visualization & Reporting**
  - [ ] Real-time index chart (updated hourly)
  - [ ] Historical trend view (1 month, 3 months, 1 year)
  - [ ] Component breakdown (positive%, negative%, neutral%)
  - [ ] Data source attribution (Reddit vs. News vs. Twitter)
  - [ ] Export functionality (CSV, JSON, PDF)

- [ ] **Performance Validation**
  - [ ] Correlation with market indices (SPY, sector ETFs)
  - [ ] Lead/lag analysis with actual returns
  - [ ] Index stability test (coefficient of variation)
  - [ ] Forecasting power (logistic regression on future returns)

### Maintenance & Operations
- [ ] Automated data quality checks
- [ ] Model performance monitoring (drift detection)
- [ ] Alert threshold optimization (quarterly review)
- [ ] Regular backups & disaster recovery
- [ ] Documentation & runbooks

---

## 📈 SUCCESS METRICS & KPIs

### Technical KPIs
- **API Response Time**: <200ms for 95th percentile
- **Data Freshness**: <30 minutes lag for sentiment index
- **Model Accuracy**: F1-score > 0.85 on test set
- **System Uptime**: 99.5% availability
- **Database Query Performance**: <100ms for real-time queries

### Business KPIs
- **Index Predictive Power**: Correlation > 0.60 with next-day returns
- **Strategy Performance**: Sharpe ratio > 1.5 on backtests
- **Alert Precision**: False positive rate < 10%
- **User Adoption**: Platform usage metrics & dashboard views
- **Cost Efficiency**: Infrastructure cost per transaction < $0.01

### Research/Science KPIs
- **Model Generalization**: Out-of-sample performance within 5% of in-sample
- **Factor Stability**: Weights stable across market regimes (Sharpe consistency)
- **Risk Model Accuracy**: VaR backtesting pass rate > 95%

---

## 🔗 DEPENDENCIES & INTEGRATION POINTS

```
GLP-1 Sentiment Analysis
      ↓
Real-Time Sentiment Index
      ↓
┌─────────────────────────────────────────────┐
│  Multi-Factor Scoring System                │
│  (Combines: Sentiment + Fundamental + Tech) │
└──────────────┬──────────────────────────────┘
               ↓
    Portfolio Optimization Engine
         (8 models with constraints)
               ↓
    Asset Allocation + Rebalancing
               ↓
    Risk Management System
   (VaR, CVaR, Stress Testing)
               ↓
┌─────────────────────────────────────────────┐
│  Production Dashboard & API                 │
│  (Real-time monitoring & reporting)         │
└─────────────────────────────────────────────┘
```

### Key Integration Points
1. **Sentiment → Portfolio**: Constraint generation from sentiment signals
2. **Forecast → Portfolio**: Return/volatility forecasts for optimization
3. **Risk → Portfolio**: VaR constraints for risk-aware allocation
4. **Valuation → Scoring**: DCF & comps feed into multi-factor score
5. **Portfolio → Dashboard**: Live holdings & performance tracking

---

## Real-Time Index Construction Checklist

Prerequisites include database infrastructure (PostgreSQL + TimescaleDB), configured data sources (Reddit API key, NewsAPI key), saved model artifacts (.joblib, .h5 files), and FastAPI framework deployment.

Core components must include the sentiment index calculator with multi-source data aggregation, time-decay weighting, outlier detection, index computation, and unit tests. The real-time update pipeline requires scheduled collection (hourly or daily), deduplication logic, ensemble model inference, index recalculation, and database append operations. Monitoring and alerts should detect sentiment shifts, extreme values, and model disagreement, with configurable thresholds and alert delivery methods.

Visualization should include real-time and historical charts, component breakdown (positive/negative/neutral percentages), data source attribution, and export capability. Performance validation must confirm correlation with market indices, lead/lag analysis with actual returns, index stability testing, and forecasting power measurement. Maintenance requires automated data quality checks, drift detection, threshold optimization (quarterly review), backups, and runbooks.

---

## Success Metrics

Technical KPIs include API response time under 200ms (95th percentile), sentiment index data freshness under 30 minutes, model F1-score above 0.85, system uptime of 99.5%, and database query performance under 100ms. Business KPIs include sentiment index correlation above 0.60 with next-day returns, strategy Sharpe ratio above 1.5 on backtests, alert false positive rate below 10%, and measurable user adoption. Research KPIs target out-of-sample performance within 5% of in-sample, stable weights across market regimes, and VaR backtesting pass rate above 95%.

---

## System Architecture & Integration Points

Confidence always flows forward in the system: GLP-1 sentiment analysis feeds into the real-time sentiment index, which feeds into the multi-factor scoring system (combining sentiment with fundamental and technical signals), which informs portfolio optimization using one of the eight models with dynamic sentiment-adjusted constraints, which generates asset allocations that feed into the risk management system (VaR, CVaR, stress testing), which displays results on the dashboard and serves results via API.

Five critical integration points require attention. Sentiment signals must translate into portfolio constraints. Forecasts (returns and volatility) feed forward-looking expectations into optimization. Risk metrics constrain portfolio weights (never exceed VaR limits). Valuation metrics (DCF, comps) feed the multi-factor scoring system. Portfolio holdings stream live performance to the dashboard.

---

## Final Implementation Notes

Success requires clear communication through weekly team syncs and documented decisions. Each component must remain independently testable despite integration. Code review, automatic testing, and continuous integration prevent regressions. Performance must be monitored from day one—don't optimize at the end. Always backtest strategies before production deployment. Expect iterations and improvements after initial launch.

The winning approach combines systematic rigor (validated backtesting, controlled experiments) with pragmatism (start simple, add complexity incrementally). Code quality matters early. Technical debt compounds; paying it down weekly is cheaper than paying it down later. Documentation payoff is asymmetric—invest upfront, then team velocity increases substantially.

Success is confirmed when the API is fast and reliable with clear documentation, when sentiment index shows predictive power for returns, when portfolio strategies outperform benchmarks with good Sharpe ratios, when the dashboard becomes used for actual decision-making, and when the team can modify and extend the system independently. Begin with Phase 2 tasks in week 1 and move forward systematically.
