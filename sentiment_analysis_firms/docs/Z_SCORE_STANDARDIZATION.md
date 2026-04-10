# Z-Score Sentiment Standardization

## Overview

The standardized sentiment index applies **within-week z-score normalization** to enable cross-firm and cross-week comparisons. This transforms raw sentiment scores into relative measures of how each firm's sentiment compares to its week's average.

## Formula

**Within-week standardization (z-score normalization):**

$$Z_{it} = \frac{S_{it} - \mu_t}{\sigma_t}$$

Where:
- $S_{it}$ = Average sentiment for firm $i$ in week $t$
- $\mu_t$ = Mean sentiment across all firms in week $t$
- $\sigma_t$ = Standard deviation of sentiment across all firms in week $t$
- $Z_{it}$ = Standardized sentiment for firm $i$ in week $t$ (the `z_sentiment` column)

## Output File

**File:** `outputs/sentiment_index.csv`

**Columns:**
1. `firm` - Firm name
2. `week` - ISO week (YYYY-Www format)
3. `article_count` - Number of articles in firm-week
4. `avg_sentiment` - Raw sentiment score (-1 to +1)
5. `rolling_sentiment_4w` - 4-week rolling average per firm
6. `z_sentiment` - Standardized within-week sentiment (the new column)

## Example: Week 2026-W14

Let's illustrate with real data from the demo dataset.

**Raw sentiment across 7 firms in week 2026-W14:**
```
Wonderful:  0.5719
Nestle:     0.0772
Others:     0.0000 (Chomps, Ferrero, Hershey, Mars, Mondelez, RXBAR)

Week statistics:
  Mean: 0.0817
  Std:  0.2162
```

**Z-score transformation:**
```
Wonderful: (0.5719 - 0.0817) / 0.2162 = +2.268  (extreme positive)
Nestle:    (0.0772 - 0.0817) / 0.2162 = -0.021  (near average)
Others:    (0.0000 - 0.0817) / 0.2162 = -0.378  (below average)
```

**Interpretation:**
- **Wonderful** (z = +2.268): Massive positive sentiment outlier that week
- **Nestle** (z = -0.021): Essentially at the weekly average
- **Others** (z = -0.378): Slightly below average, but uniform

## Properties of Z-Scores

### 1. **Mean Standardization**
Within each week, the z-score mean = 0
```
mean(z_sentiment | week) ≈ 0  (by construction)
```

### 2. **Std Dev Normalization**
Within each week, the z-score std = 1
```
std(z_sentiment | week) ≈ 1  (by construction)
```

### 3. **Distribution Interpretation**

For a normal distribution:
- **|z| < 1.0** (~68%): Typical sentiment
- **1.0 < |z| < 1.96** (~27%): Unusual sentiment
- **1.96 < |z| < 2.58** (~5%): Very unusual (top/bottom 2.5%)
- **|z| > 2.58** (~1%): Extreme (top/bottom 0.5%)

### 4. **Sign Interpretation**
- **z > 0** : Firm sentiment **above** weekly average (outperformer)
- **z < 0** : Firm sentiment **below** weekly average (underperformer)
- **z ≈ 0** : Firm sentiment **near** weekly average

## Use Cases

### Use Case 1: Weekly Rankings
Identify which firms have best/worst sentiment relative to their peers each week.
```python
# Rank firms by z_sentiment within each week
weekly_rankings = index.groupby('week').apply(
    lambda x: x.nlargest(3, 'z_sentiment')[['firm', 'z_sentiment']]
)
```

### Use Case 2: Extreme Sentiment Detection
Flag firm-weeks with unusual sentiment.
```python
# Find outliers
outliers = index[index['z_sentiment'].abs() > 1.96]
print(f"Extreme sentiment observations: {len(outliers)}")
```

### Use Case 3: Cross-Week Comparison
Compare the same firm across weeks on a standardized scale.
```python
# Track PepsiCo's relative sentiment over time
pepsi = index[index['firm'] == 'PepsiCo'].sort_values('week')
print(pepsi[['week', 'avg_sentiment', 'z_sentiment']])
# z_sentiment shows: good week vs. bad week vs. average week
```

### Use Case 4: Econometric Regression
Use z-scores as independent variables in panel regression.
```python
# Panel fixed effects model
# returns ~ z_sentiment + rolling_sentiment_4w + firm_FE + week_FE
# Benefits: 
#   - Standardized coefficients (easier interpretation)
#   - Controls for week-level sentiment shocks
#   - Identifies firm-specific effects
```

### Use Case 5: Anomaly Detection
Identify when a firm has unexpectedly positive/negative sentiment.
```python
# Flag when |z| > 2
index['anomaly'] = index['z_sentiment'].abs() > 2.0
anomalies = index[index['anomaly']]
```

## Comparison: Raw vs. Z-Score Sentiment

| Question | Raw Sentiment | Z-Score |
|----------|--------------|---------|
| **Is Wonderful positive?** | Yes (0.5719) | Extremely (z=2.27) |
| **Is PepsiCo negative?** | No (0.0000) | Yes, relative to week (z=-1.06) |
| **Who did better: RXBAR in W14 or PepsiCo in W15?** | RXBAR (tied at 0) | Comparable: both below week avg (z≈-0.4 vs z≈-1.1) |
| **Is sentiment trending?** | Can't tell (single values) | Can see vs. week baseline |
| **Statistical modeling** | Unstandardized | Standardized (preferred) |

## When to Use Each Sentiment Type

### Use `avg_sentiment` (Raw) when:
- You want interpretable [-1, +1] scores
- Comparing to established benchmarks
- Communicating to non-technical audiences
- Assessing overall market sentiment tone

### Use `rolling_sentiment_4w` (Rolling) when:
- Tracking firm-specific sentiment trends
- Smoothing short-term noise
- Identifying momentum reversal points
- Analyzing within-firm dynamics

### Use `z_sentiment` (Standardized) when:
- Comparing firms within a week (relative performance)
- Running statistical/econometric models
- Identifying outliers and anomalies
- Cross-week analysis
- Building predictive features

## Statistical Properties

### Z-Score Advantages
✅ Mean-centered (zero baseline)
✅ Variance-normalized (consistent scale across weeks)
✅ Directly interpretable (σ units from mean)
✅ Suitable for linear regression
✅ Identifies outliers naturally
✅ Handles week-level effects automatically

### Z-Score Limitations
⚠️ Requires multiple observations per week (to calculate std)
⚠️ Single-firm weeks have z=0 (no variance)
⚠️ Less interpretable to general audience (not [-1, +1])
⚠️ Weights all firms equally (heavy firms don't get more weight)

## Implementation Details

### Code
```python
from src.analysis.create_standardized_sentiment import create_standardized_sentiment

# Generate standardized sentiment
index = create_standardized_sentiment(
    input_csv='data/processed/firm_week_panel.csv',
    output_csv='outputs/sentiment_index.csv'
)
```

### Algorithm (within each week)
1. Load all firms' `avg_sentiment` scores for the week
2. Calculate mean: $\mu_t = \text{mean}(\text{avg_sentiment})$
3. Calculate std: $\sigma_t = \text{std}(\text{avg_sentiment})$
4. For each firm: $z_i = \frac{s_i - \mu_t}{\sigma_t}$
5. Handle edge case: if $\sigma_t = 0$ (all firms identical), set $z_i = 0$

### Edge Cases
- **Single firm-week**: z = 0 (can't standardize vs. single observation)
- **All firms same sentiment**: z = 0 for all (no variance)
- **Perfect anti-correlation**: z scores spread across full range

## Practical Examples from Demo Data

### Example 1: Wonderful in W14
```
Position: Highest sentiment that week (z = +2.27)
Interpretation: Extraordinary positive news
Action: Consider as potential buy signal
```

### Example 2: PepsiCo in W15
```
Position: Lowest sentiment that week (z = -1.06)
Interpretation: Underperforms peers despite neutral raw score
Action: Investigate relative to week's consensus
```

### Example 3: Six Firms in W14 (z ≈ -0.38)
```
Position: Slightly below week average
Interpretation: Part of neutral cluster
Action: Not actionable individually; watch for clustering changes
```

## Next Steps

1. **Merge with stock returns** → Test if z_sentiment predicts returns
2. **Create trading signals** → Buy z > +1.5, Sell z < -1.5
3. **Panel regression** → Estimate effect size of z_sentiment on returns
4. **Rolling window** → Extend to 8-week, 12-week z-scores
5. **Sector analysis** → Compare z-scores across food company sectors

## References

**Z-score normalization** is standard preprocessing in:
- Machine learning (feature standardization)
- Econometrics (standardized coefficients)
- Statistics (hypothesis testing)
- Finance (risk-adjusted returns like Sharpe ratio)

Also known as: **standardization, z-normalization, standard score**

---

**File:** `outputs/sentiment_index.csv`
**Created by:** `src/analysis/create_standardized_sentiment.py`
**Last updated:** 2026-04-09
