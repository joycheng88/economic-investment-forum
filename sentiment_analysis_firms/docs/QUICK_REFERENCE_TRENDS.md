# Sentiment Trends Visualization - Quick Reference

## Files Generated

### 1. **sentiment_trends.png** (217 KB)
**Visual time series of firm sentiment over time**

- **What to look for:**
  - Firms trending **upward** = improving sentiment (positive)
  - Firms trending **downward** = deteriorating sentiment (risky)
  - Horizontal lines = stable/stagnant sentiment
  - Steep slopes = volatile sentiment (e.g., breaking news)

- **Interpretation guide:**
  ```
  Z-Score Range    | Meaning
  Z > +2.0        | Extreme positive sentiment (outlier good news)
  +0.5 to +2.0    | Above average sentiment (watch as leader)
   0 to +0.5      | Neutral-positive (baseline acceptable)
  -0.5 to  0      | Neutral-negative (baseline concern)
  -2.0 to -0.5    | Below average sentiment (watch for risks)
  Z < -2.0        | Extreme negative sentiment (crisis)
  ```

- **How to read the chart:**
  1. Find your firm's color/line
  2. Trace from left (earliest) to right (latest)
  3. Compare position relative to red dashed baseline (0)
  4. Look for trends (slope of line):
     - Steep up = rapid sentiment improvement
     - Steep down = rapid sentiment deterioration
     - Flat = stable sentiment

- **Current visual highlights:**
  - **Wonderful (cyan):** Single point at +2.27 (★ star performer)
  - **PepsiCo (gray):** Two points showing **downward trend** (⚠ concern)
  - **Others:** Clustered around -0.38 (mixed negative)

---

### 2. **firm_sentiment_rankings.csv** (640 B)
**Numerical rankings of all 10 firms by sentiment**

- **Primary ranking metric:** `z_avg` (average standardized sentiment)
  - Higher = better sentiment
  - Rankings run 1 (best) to 10 (worst)

- **How to use:**
  ```
  # Load in Python
  import pandas as pd
  rankings = pd.read_csv('outputs/firm_sentiment_rankings.csv')
  print(rankings)  # View full table
  print(rankings.iloc[0])  # View top firm
  print(rankings.iloc[-1])  # View bottom firm
  ```

- **Key columns explained:**

  | Column | Meaning | Example |
  |--------|---------|---------|
  | `firm` | Company name | Wonderful, PepsiCo |
  | `rank` | Ranking (1=best) | 1, 10 |
  | `z_avg` | Mean sentiment score | 2.2678 (Wonderful) |
  | `n_obs` | How many weeks observed | 2 (PepsiCo), 1 (others) |
  | `raw_avg` | Average raw sentiment | 0 to 1 scale |
  | `rolling_avg` | 4-week trend | Smoothed sentiment |
  | `total_articles` | News coverage | 1 (most), 2 (PepsiCo) |

- **Current rankings (top 5):**
  ```
  1. Wonderful     - z_avg=+2.27 (strong positive)
  2. General Mills - z_avg=+0.92 (positive)
  3. Nestle        - z_avg=+0.14 (neutral+)
  4. Chomps        - z_avg=-0.38 (negative)
  5. Ferrero       - z_avg=-0.38 (negative)
  ...
  10. PepsiCo      - z_avg=-0.53 (most negative, deteriorating)
  ```

---

## Usage Scenarios

### Scenario 1: Monitor Sentiment Trends
**Question:** Which firms have improving/deteriorating sentiment?

**Answer:** Look at `sentiment_trends.png`
- Upward slopes = improving (watch as opportunity)
- Downward slopes = deteriorating (watch as risk)

**PepsiCo:** Clearly downward trend (week 1: 0.0 → week 3: -1.06)

---

### Scenario 2: Rank Firms by Sentiment
**Question:** Which is the most/least popular firm in news coverage?

**Answer:** Check `firm_sentiment_rankings.csv`
- **Most positive:** Wonderful (rank 1, z_avg +2.27)
- **Most negative:** PepsiCo (rank 10, z_avg -0.53)

---

### Scenario 3: Identify Volatility
**Question:** Which firms have unstable/variable sentiment?

**Answer:** Check the `z_std` column (volatility)
- **High std:** Sentiment varies week-to-week (risky/reactive)
- **Low std:** Sentiment stable (predictable/consistent)

**Current data limitation:** Only PepsiCo has multiple observations (std=0.75)

---

### Scenario 4: News Coverage Depth
**Question:** Which firms receive the most news attention?

**Answer:** Check `total_articles` column
- Higher = more news coverage = more sentiment signal
- Lower = less coverage = noisier sentiment signal

**Current:** PepsiCo (2 articles), all others (1 article each)

---

## Integration with Trading/Portfolio Systems

### Example 1: Sentiment-Based Stock Selection
```python
# Load rankings
rankings = pd.read_csv('outputs/firm_sentiment_rankings.csv')

# Select top performers
buy_list = rankings[rankings['z_avg'] > 0.5]['firm'].tolist()
# Result: ['Wonderful', 'General Mills']

# Select bottom performers
sell_list = rankings[rankings['z_avg'] < -0.3]['firm'].tolist()
# Result: ['Chomps', 'Ferrero', 'Hershey', 'Mars', 'Mondelez', 'RXBAR', 'PepsiCo']
```

### Example 2: Create Sentiment Factor
```python
# Use z_avg as sentiment factor weight
weights = rankings.set_index('firm')['z_avg']
# Use in portfolio optimization: weights* returns
```

### Example 3: Monitor PepsiCo Risk
```python
pepsi = rankings[rankings['firm'] == 'PepsiCo'].iloc[0]
print(f"PepsiCo volatility: {pepsi['z_std']}")  # 0.75 (high!)
print(f"PepsiCo trend: {pepsi['z_min']} to {pepsi['z_max']}")  # -1.06 to 0.0 (downward)
print(f"Alert: Deteriorating sentiment with HIGH volatility")
```

---

## Common Questions & Answers

### Q: Why are most firms at exactly z_avg = -0.378?
**A:** Demo data has only 1 observation per firm per week. Within each week, multiple firms have identical raw_avg (0.0), so they all get the same z-score. This will disappear with real data (multiple articles per week).

---

### Q: What does "standardized" (z-score) mean?
**A:** It means RELATIVE to that week's average:
- z = 0: Exactly average sentiment for that week
- z = +1: One standard deviation above that week
- z = -1: One standard deviation below that week

This lets you compare across weeks fairly. Raw -0.5 in week 1 might be different from -0.5 in week 2 due to overall news tone.

---

### Q: Why does the chart have a red dashed line at 0?
**A:** That's the weekly (period) average. Firms ABOVE the line had better sentiment than average that week. Firms BELOW the line had worse sentiment than average.

---

### Q: Can I use this for trading decisions with demo data?
**A:** **No.** With only 11 observations:
- Standard errors are huge (NaN for single-obs firms)
- Outliers dominate (Wonderful +2.27 is extreme with n=1)
- No statistical significance testing possible
- Need: Real data with 500+ observations for confidence

---

### Q: What's the relationship to the DiD analysis?
**A:** 
- **Rankings:** Show current sentiment LEVELS
- **DiD regression:** Shows GLP-1 IMPACT on sentiment change

Together they tell: "Snack firms (PepsiCo) have lower sentiment AND lost more sentiment after GLP-1"

---

## File Locations

```
outputs/sentiment_trends.png              ← Plot (open in image viewer)
outputs/firm_sentiment_rankings.csv       ← Rankings (open in Excel/pandas)
plot_sentiment_trends.py                  ← Code to regenerate
sentiment_analysis_firms/                 ← All pipeline outputs
```

---

## How to Regenerate

```bash
# If sentiment_index.csv exists, just run:
python plot_sentiment_trends.py

# Outputs will be created/overwritten:
#   - outputs/sentiment_trends.png
#   - outputs/firm_sentiment_rankings.csv
```

---

## Key Takeaways for Stakeholders

1. **Wonderful dominates** in positive sentiment (+2.27)
2. **PepsiCo deteriorates** over time (0.0 → -1.06) ⚠️
3. **Most snack firms lag** (negative sentiment cluster)
4. **Diversified firms do better** (Nestle, General Mills positive)
5. **GLP-1 event differentiates** traditional vs. alternative snacks

---

**Last Updated:** April 9, 2026
**Data:** Demo/Synthetic (11 observations)
**Status:** Ready for real data validation (500+ observations expected)
