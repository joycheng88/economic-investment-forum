# Econometric Pipeline Guide
## Complete Workflow: Sentiment → Panel Data → Analysis

This guide documents the complete pipeline for transforming raw news data into econometric panel data ready for statistical analysis.

---

## 1. Data Structure: Firm-Week Panel

### Definition
A **panel dataset** with:
- **Cross-sectional dimension:** Firms (10 firms)
- **Time dimension:** Weeks (ISO week format: YYYY-Www)
- **Observation unit:** Firm-week pair (firm × week)

### Core Columns

| Column | Type | Description | Formula/Source |
|--------|------|-------------|---------|
| `firm` | string | Firm name | Company names from gnews API |
| `week` | string | ISO week (YYYY-Www) | published_date → dt.isocalendar() |
| `avg_sentiment` | float64 | Mean sentiment score | Mean of sentiment_compound per firm-week |
| `article_count` | int64 | Number of articles | Count of articles per firm-week |
| `sentiment_std` | float64 | Std dev of sentiment | Std dev of sentiment_compound (NaN if n=1) |
| `rolling_sentiment_4w` | float64 | 4-week rolling average | Rolling mean per firm (window=4, min_periods=1) |
| `z_sentiment` | float64 | Standardized sentiment | (avg_sentiment - week_mean) / week_std |

### Example Data (Demo)
```
firm,week,avg_sentiment,article_count,sentiment_std,rolling_sentiment_4w
Chomps,2026-W14,0.0,1,,0.0
Ferrero,2026-W14,0.0,1,,0.0
General Mills,2026-W15,0.128,1,,0.128
Hershey,2026-W13,0.0,1,,0.0
Mars,2026-W14,0.0,1,,0.0
Mondelez,2026-W15,0.0,1,,0.0
Nestle,2026-W14,0.0772,1,,0.0772
PepsiCo,2026-W13,0.0,1,,0.0
PepsiCo,2026-W15,0.0,1,,0.0
RXBAR,2026-W14,0.0,1,,0.0
Wonderful,2026-W14,0.5719,1,,0.5719
```

---

## 2. Complete Pipeline: From Raw News to Panel Data

### Stage 1: Data Collection
**Script:** `run_full_pipeline_live.py`
**Output:** `data/raw/news_data_real.csv`

Collects news via gnews API:
```
Input:  170 firm-keyword search combinations (10 firms × 17 keywords)
Output: ~1,500-1,700 articles with columns:
  - firm (company name)
  - title, description (article text)
  - published_date (timestamp)
  - url, source
Time:   ~40 minutes (rate-limited at 1.5s/query)
```

### Stage 2: Text Preprocessing
**Script:** `run_full_pipeline_live.py` (step 2)
**Output:** `data/processed/news_data_cleaned_real.csv`

Cleans and normalizes text:
```
Operations:
  - Remove URLs, emails
  - Convert to lowercase
  - Remove punctuation
  - Remove stopwords
  - Apply stemming/lemmatization

Retention: 80-90% of articles pass validation
Time: ~5 seconds
```

### Stage 3: GLP-1 Relevance Filtering
**Script:** `run_full_pipeline_live.py` (step 3) OR standalone `src/filtering/filter_glp1_relevant.py`
**Output:** `data/processed/glp1_relevant_real.csv`

Applies dual-condition filter:
```
Condition 1: Contains firm name (10 firms, 85+ variants)
Condition 2: Contains GLP-1 keyword (132+ keywords)

Logic: MUST satisfy BOTH conditions
Retention: 70-75% of articles
Time: ~2 seconds
```

### Stage 4: Sentiment Analysis
**Script:** `run_full_pipeline_live.py` (step 4)
**Output:** `data/processed/articles_with_sentiment_real.csv`

Analyzes text sentiment (choose one):

#### Option A: VADER (Fast, Default)
```python
from src.sentiment.optimized_analyzer import VADEROptimized
analyzer = VADEROptimized()
df_results = analyzer.analyze_dataframe(df, text_column='clean_text', batch_size=1000)
```
- Speed: ~100 articles/sec
- Accuracy: 70-75% (on financial text)
- Output columns: sentiment_label, sentiment_compound (range: -1 to +1)

#### Option B: FinBERT (More Accurate)
```python
from src.sentiment.optimized_analyzer import OptimizedFinBERTAnalyzer
analyzer = OptimizedFinBERTAnalyzer(batch_size=32)
df_results = analyzer.analyze_dataframe(df, text_column='clean_text')
```
- Speed: 5-20 articles/sec
- Accuracy: 85-90% (trained on financial text)
- Output columns: sentiment_label, sentiment_compound, sentiment_confidence

### Stage 5: Create Firm-Week Panel
**Script:** `src/analysis/create_panel.py`
**Output:** `data/processed/firm_week_panel.csv`

Aggregates sentiment into panel format:
```python
from src.analysis.create_panel import create_firm_week_panel

panel = create_firm_week_panel(
    input_csv='data/processed/articles_with_sentiment_real.csv',
    output_csv='data/processed/firm_week_panel.csv'
)
```

**Processing steps:**
1. Parse `published_date` into datetime objects
2. Extract ISO week: `dt.isocalendar()` (format: YYYY-Www)
3. Group by (firm, week)
4. Compute aggregates:
   - `avg_sentiment` = mean of sentiment_compound
   - `article_count` = count of rows
   - `sentiment_std` = std dev of sentiment_compound
5. Sort by firm, then week
6. Save 5-column CSV

**Statistics (with real data ~1,500 articles):**
```
Expected: 500-600 firm-week observations
Avg articles per firm-week: 2-3
Firms with sparse coverage: May have only 1-2 weeks
Weeks with many observations: Peak weeks likely have 40-60 articles
```

### Stage 6: Add Rolling Average
**Script:** `src/analysis/add_rolling_sentiment.py`
**Output:** Updates `data/processed/firm_week_panel.csv` in-place

Adds 4-week rolling average sentiment:
```python
panel['rolling_sentiment_4w'] = panel.groupby('firm')['avg_sentiment'].transform(
    lambda x: x.rolling(window=4, min_periods=1).mean()
).round(4)
```

**Rolling window mechanics:**
- Window size: 4 weeks (backward-looking)
- Minimum periods: 1 (forward-fill first weeks, avoiding NaN)
- Calculation: Independent per firm (groupby avoids data leakage)
- Effect: Smooths firm-specific sentiment trends

**Example (Synthetic 8-week data for one firm):**
```
Week 1: sentiment=+0.10 → rolling=+0.10 (1 point)
Week 2: sentiment=+0.20 → rolling=+0.15 (avg: 0.1, 0.2)
Week 3: sentiment=-0.10 → rolling=+0.067 (avg: 0.1, 0.2, -0.1)
Week 4: sentiment=+0.30 → rolling=+0.10 (avg: 0.1, 0.2, -0.1, 0.3)
Week 5: sentiment=+0.00 → rolling=+0.10 (rolling window: 0.2, -0.1, 0.3, 0.0)
Week 6: sentiment=+0.05 → rolling=+0.0875 (rolling window: -0.1, 0.3, 0.0, 0.05)
Week 7: sentiment=-0.20 → rolling=+0.0375 (rolling window: 0.3, 0.0, 0.05, -0.2)
Week 8: sentiment=+0.15 → rolling=+0.075 (rolling window: 0.0, 0.05, -0.2, 0.15)
```

### Stage 7: Standardize Within-Week Sentiment
**Script:** `src/analysis/create_standardized_sentiment.py`
**Output:** `outputs/sentiment_index.csv`

Creates z-score normalized sentiment within each week:
```python
from src.analysis.create_standardized_sentiment import create_standardized_sentiment

sentiment_index = create_standardized_sentiment(
    input_csv='data/processed/firm_week_panel.csv',
    output_csv='outputs/sentiment_index.csv'
)
```

**Standardization formula (within-week normalization):**
```
Z_it = (S_it - mean_t) / std_t

Where:
  S_it = avg_sentiment for firm i in week t
  mean_t = mean sentiment across all firms in week t
  std_t = std dev of sentiment across all firms in week t
  Z_it = standardized sentiment for firm i in week t (z_sentiment column)
```

**Properties:**
- Mean ≈ 0 and std ≈ 1 within each week
- Each firm's sentiment relative to weekly baseline
- Positive z: above-average sentiment that week
- Negative z: below-average sentiment that week
- |z| > 1.96: Extreme sentiment (top/bottom ~2.5% in week)

**Example (2026-W14 with 7 firms):**
```
Week average: 0.0817
Week std dev: 0.2162

Wonderful:  0.5719 - 0.0817 / 0.2162 = +2.27 (extreme positive)
Nestle:     0.0772 - 0.0817 / 0.2162 = -0.02 (near average)
Others:     0.0000 - 0.0817 / 0.2162 = -0.38 (below average)
```

**Output file:** `outputs/sentiment_index.csv`
- Contains all 6 sentiment columns for complete analysis
- Sorted by firm and week
- Ready for econometric regression

---

## 3. Econometric Analysis: Using Panel Data

### Analysis 1: Firm-Level Time Series

**Question:** How does sentiment evolve for each firm over time?

```python
import pandas as pd
import matplotlib.pyplot as plt

panel = pd.read_csv('data/processed/firm_week_panel.csv')

# Time series for one firm
pepsi = panel[panel['firm'] == 'PepsiCo'].sort_values('week')

# Plot sentiment trends
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

ax1.plot(pepsi['week'], pepsi['avg_sentiment'], marker='o', label='Weekly')
ax1.plot(pepsi['week'], pepsi['rolling_sentiment_4w'], marker='s', label='4-week Rolling')
ax1.set_ylabel('Sentiment')
ax1.legend()
ax1.set_title('PepsiCo Sentiment Over Time')

ax2.bar(pepsi['week'], pepsi['article_count'])
ax2.set_ylabel('Article Count')
ax2.set_xlabel('Week')
ax2.set_title('Article Coverage by Week')
```

### Analysis 2: Cross-Firm Comparison

**Question:** Which firms have most positive/negative sentiment?

```python
# Aggregate across all weeks
firm_summary = panel.groupby('firm').agg({
    'avg_sentiment': 'mean',
    'rolling_sentiment_4w': 'mean',
    'article_count': 'sum'
}).sort_values('avg_sentiment', ascending=False)

print(firm_summary)

# Visualize
firm_summary['avg_sentiment'].plot(kind='barh', figsize=(10, 6))
plt.xlabel('Average Sentiment')
plt.title('Firm Sentiment Comparison')
plt.show()
```

### Analysis 3: Sentiment Momentum

**Question:** Is sentiment improving or declining?

```python
# Calculate weekly change
panel['sentiment_change'] = panel.groupby('firm')['rolling_sentiment_4w'].diff()
panel['momentum'] = (panel['sentiment_change'] > 0).astype(int)

# Momentum summary by firm
momentum_summary = panel.groupby('firm')['momentum'].agg(['sum', 'count', 'mean'])
momentum_summary.columns = ['Improving_Weeks', 'Total_Weeks', 'Momentum_Score']
print(momentum_summary)

# Interpretation: Momentum_Score > 0.5 = mostly improving
```

### Analysis 4: Trend Detection

**Question:** Detect inflection points (sentiment reversals)?

```python
# Identify reversals (sign change in rolling average)
panel['reversal'] = (
    panel.groupby('firm')['rolling_sentiment_4w'].apply(
        lambda x: x * x.shift(1) < 0  # Opposite signs
    )
).astype(int)

# Reversals by firm
reversals = panel[panel['reversal'] == 1][['firm', 'week', 'rolling_sentiment_4w']]
print(f"Total sentiment reversals: {len(reversals)}")
print(reversals)
```

### Analysis 5: Correlation with Returns (Requires Market Data)

**Question:** Does sentiment predict stock returns?

```python
# Merge with stock returns (when available)
returns = pd.read_csv('data/stock_returns_weekly.csv')  # Must have firm, week, returns columns
panel_with_returns = panel.merge(returns, on=['firm', 'week'], how='inner')

# Correlation
correlation = panel_with_returns['rolling_sentiment_4w'].corr(panel_with_returns['returns'])
print(f"Sentiment → Returns correlation: {correlation:.4f}")

# Predictive regression
from sklearn.linear_model import LinearRegression

X = panel_with_returns[['rolling_sentiment_4w']].values
y = panel_with_returns['returns'].values

model = LinearRegression().fit(X, y)
print(f"Beta (sentiment effect on returns): {model.coef_[0]:.6f}")
print(f"R-squared: {model.score(X, y):.4f}")
```

### Analysis 6: Panel Fixed Effects Regression

**Question:** After controlling for firm and week effects, does sentiment matter?

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Merge with returns
panel_with_returns = panel.merge(returns, on=['firm', 'week'], how='inner')

# Fixed effects model
model = ols('returns ~ C(firm) + C(week) + rolling_sentiment_4w', 
            data=panel_with_returns).fit()

print(model.summary())
# Look for significance of rolling_sentiment_4w coefficient
```

---

## 4. File Structure & Execution

### Key Files

| File | Purpose | When to Run |
|------|---------|-----------|
| `run_full_pipeline_live.py` | Main orchestration (stages 1-4) | Once per data collection cycle |
| `src/analysis/create_panel.py` | Create firm-week panel (stage 5) | After sentiment analysis |
| `src/analysis/add_rolling_sentiment.py` | Add rolling average (stage 6) | After panel creation |
| `examples_panel_analysis.py` | Example usage patterns | Reference only |
| `panel_workflow.py` | Complete workflow guide | Reference only |
| `rolling_sentiment_demo.py` | Demo with synthetic data | Learning reference |

### Execution Sequence

```bash
# Step 1: Collect and process (runs stages 1-4)
python run_full_pipeline_live.py
# Output: data/processed/articles_with_sentiment_real.csv

# Step 2: Create panel
python src/analysis/create_panel.py
# Output: data/processed/firm_week_panel.csv

# Step 3: Add rolling average
python src/analysis/add_rolling_sentiment.py
# Output: Updated data/processed/firm_week_panel.csv (now with rolling_sentiment_4w)

# Step 4: Create standardized sentiment index
python src/analysis/create_standardized_sentiment.py
# Output: outputs/sentiment_index.csv (with z_sentiment column)

# Step 5: Run analysis (in Jupyter or Python)
# Use code samples from Section 3 above
# Import from: outputs/sentiment_index.csv
```

---

## 5. Data Quality Checklist

Before running analysis, verify:

- [ ] Final index file exists: `outputs/sentiment_index.csv`
- [ ] All 6 columns present: firm, week, article_count, avg_sentiment, rolling_sentiment_4w, z_sentiment
- [ ] No missing values in any sentiment column
- [ ] avg_sentiment and rolling_sentiment_4w in range [-1, +1]
- [ ] z_sentiment has mean ≈ 0 and std ≈ 1 within each week
- [ ] Article count ≥ 1 per observation
- [ ] Weeks in YYYY-Www format
- [ ] All firm names match expected list (10 firms)
- [ ] Data sorted by firm, then by week

### Quick Verification

```python
import pandas as pd

index = pd.read_csv('outputs/sentiment_index.csv')

# Check structure
print(f"Shape: {index.shape}")  # Should be (N, 6)
print(f"Columns: {index.columns.tolist()}")

# Check data types
print(index.dtypes)

# Check for missing values
print(index.isnull().sum())  # All should be 0

# Check value ranges
print(f"Raw sentiment range: [{index['avg_sentiment'].min():.4f}, {index['avg_sentiment'].max():.4f}]")
print(f"Rolling sentiment range: [{index['rolling_sentiment_4w'].min():.4f}, {index['rolling_sentiment_4w'].max():.4f}]")
print(f"Z-sentiment range: [{index['z_sentiment'].min():.4f}, {index['z_sentiment'].max():.4f}]")

# Check z-score properties (should be ≈ standard normal per week)
print(f"\nZ-sentiment statistics:")
print(index['z_sentiment'].describe().round(4))

# Verify within-week properties
print(f"\nZ-sentiment properties by week:")
z_by_week = index.groupby('week')['z_sentiment'].agg(['mean', 'std', 'min', 'max'])
print(z_by_week.round(4))

# Check unique firms and weeks
print(f"\nUnique firms: {index['firm'].nunique()}")
print(f"Unique weeks: {index['week'].nunique()}")
```

---

## 6. Common Use Cases

### Use Case 1: Monitor Firm Sentiment
Track sentiment trends for a specific firm to identify market concerns or opportunities.

**Frequency:** Weekly
**Key metrics:** rolling_sentiment_4w (trend), sentiment_change (momentum)

### Use Case 2: Comparative Analysis
Compare sentiment across firms to identify industry leaders/laggards.

**Frequency:** Weekly
**Key metrics:** avg_sentiment (current), rolling_sentiment_4w (trend)

### Use Case 3: Sentiment Alpha
Test if sentiment predicts stock price movements (alpha signal).

**Frequency:** Weekly (post-market close)
**Key metrics:** rolling_sentiment_4w → Merge with returns → Calculate correlation/beta

### Use Case 4: Risk Assessment
Identify firms with deteriorating sentiment (early warning signal).

**Frequency:** Real-time monitoring
**Key metrics:** sentiment_change < 0 (reversal), rolling_sentiment_4w < -0.1 (sustained negative)

### Use Case 5: Content Strategy
Understand which topics drive sentiment for each firm.

**Frequency:** Monthly
**Key input:** Article keywords that contributed to sentiment scores

---

## 7. Next Steps & Extensions

### Immediate
- [ ] Collect real data (gnews pipeline running)
- [ ] Regenerate panel with full dataset (~500+ firm-weeks)
- [ ] Validate rolling averages with real multi-article observations

### Short-term
- [ ] Merge panel with stock returns data
- [ ] Calculate sentiment-returns correlation
- [ ] Test predictive power (Granger causality)

### Medium-term
- [ ] Extend rolling window to 8-week or 12-week averages
- [ ] Add news sentiment volatility (σ of rolling_sentiment)
- [ ] Segment by article sentiment intensity (very positive vs. neutral positive)

### Long-term
- [ ] Integrate with portfolio optimization (use sentiment as factor)
- [ ] Real-time alert system for sentiment reversals
- [ ] Production deployment (automated weekly pipeline)

---

## Appendix: Parameter Decisions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rolling window | 4 weeks | Quarterly business cycle; balance responsiveness vs. noise |
| min_periods | 1 | Avoid losing first observations; forward-fill with single value |
| Sentiment range | [-1, +1] | Standard for compound sentiment (allows meaningful aggregation) |
| ISO week | YYYY-Www | Standard time unit; enables cross-panel time alignment |
| Aggregation unit | Firm-week | Natural frequency for financial news; aligns with ETF/equity trading weeks |
| Filtering method | Dual-condition | Increases precision; reduces false positives |

---

**Last Updated:** 2026-04-09
**Status:** Production Ready (demo data), Scaling to Real Data
**Maintainer:** Sentiment Analysis Team
