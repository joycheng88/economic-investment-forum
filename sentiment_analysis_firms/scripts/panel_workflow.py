"""
Complete Workflow: From Sentiment Analysis to Econometric Panel
Shows how to use the firm-week panel data
"""

import pandas as pd

print("=" * 90)
print("FIRM-WEEK PANEL DATA: Complete Workflow")
print("=" * 90)

# Load the panel data
print("\n1. LOAD PANEL DATA")
print("-" * 90)

panel = pd.read_csv('data/processed/firm_week_panel.csv')

print(f"\nLoaded: {len(panel)} firm-week observations")
print(f"\nData structure:")
print(f"{'Column':<20} {'Type':<15} {'Description':<50}")
print("-" * 85)
print(f"{'firm':<20} {'string':<15} {'Firm name':<50}")
print(f"{'week':<20} {'string (YYYY-Www)':<15} {'ISO week':<50}")
print(f"{'avg_sentiment':<20} {'float':<15} {'Mean sentiment score (-1 to +1)':<50}")
print(f"{'article_count':<20} {'integer':<15} {'# articles in firm-week':<50}")
print(f"{'sentiment_std':<20} {'float (or NaN)':<15} {'Std dev of sentiment scores':<50}")

# Show sample
print(f"\nSample data (first 10 rows):")
print(panel.head(10).to_string(index=False))

# Descriptive statistics
print("\n2. DESCRIPTIVE STATISTICS")
print("-" * 90)

print(f"\nOverall statistics:")
stats = panel[['avg_sentiment', 'article_count', 'sentiment_std']].describe().round(4)
print(stats)

# By firm
print(f"\n\nBy Firm:")
firm_summary = panel.groupby('firm').agg({
    'avg_sentiment': ['mean', 'min', 'max'],
    'article_count': 'sum',
    'week': 'count'
}).round(3)
firm_summary.columns = ['avg_sent_mean', 'avg_sent_min', 'avg_sent_max', 'total_articles', 'num_weeks']
print(firm_summary.to_string())

# By week
print(f"\n\nBy Week (Market-Level Sentiment):")
week_summary = panel.groupby('week').agg({
    'avg_sentiment': ['mean', 'std', 'count'],
    'article_count': 'sum'
}).round(3)
week_summary.columns = ['market_sentiment', 'sentiment_volatility', 'num_firms', 'total_articles']
print(week_summary.to_string())

# Data preparation for econometrics
print("\n3. DATA PREPARATION FOR ECONOMETRICS")
print("-" * 90)

# Create firm identifiers
panel['firm_id'] = pd.Categorical(panel['firm']).codes
panel['week_id'] = pd.Categorical(panel['week']).codes

print(f"\nAdded panel indices:")
print(f"  firm_id: numeric firm identifier")
print(f"  week_id: numeric week identifier")

print(f"\nPrepared data (ready for panel regression):")
print(panel[['firm', 'firm_id', 'week', 'week_id', 'avg_sentiment', 'article_count']].head(8).to_string(index=False))

# Econometric usage
print("\n4. EXAMPLE ECONOMETRIC ANALYSES")
print("-" * 90)

print(f"""
a) Panel Fixed Effects Regression:
   -------
   from statsmodels.formula.api import ols
   
   model = ols('avg_sentiment ~ C(firm) + C(week)', data=panel).fit()
   print(model.summary())

b) Firm-Specific Time Series:
   -------
   nestle_ts = panel[panel['firm'] == 'Nestle'].sort_values('week')
   print(nestle_ts)

c) Compute firm-level averages:
   -------
   firm_means = panel.groupby('firm')['avg_sentiment'].mean().sort_values(ascending=False)
   print(firm_means)

d) Cross-sectional analysis:
   -------
   # Average sentiment by firm
   firm_avg = panel.groupby('firm')['avg_sentiment'].mean()
   # Average articles by firm
   firm_volume = panel.groupby('firm')['article_count'].sum()
   
   # Correlation: Does higher volume correlate with higher sentiment?
   corr = firm_avg.corr(firm_volume)
   print(f"Correlation: {{corr:.3f}}")

e) Merge with returns (when stock data available):
   -------
   # Stock returns by firm-week
   returns = pd.read_csv('stock_returns_weekly.csv')
   
   merged = panel.merge(returns, on=['firm', 'week'], how='left')
   
   # Sentiment-return correlation
   corr = merged['avg_sentiment'].corr(merged['returns'])
   
   # Predictive regression
   model = ols('returns ~ avg_sentiment', data=merged).fit()
""")

# Summary for real data
print("\n5. EXPECTED RESULTS WITH REAL DATA")
print("-" * 90)

print(f"""
Current demo data:
  • 11 observations (mostly 1 article per firm-week)
  • sentiment_std is NaN (need multiple articles per firm-week)
  • Weak statistical power

Expected with real data (from live collection):
  • ~1,000+ firm-week observations (50-100 firms × 10-20 weeks)
  • Multiple articles per firm-week → meaningful sentiment_std
  • Strong data for econometric analysis
  • Can test economic hypotheses:
    - Does sentiment predict returns?
    - Is sentiment mean-reverting?
    - How persistent are firm shocks?
""")

# Usage going forward
print("\n6. NEXT STEPS")
print("-" * 90)

print(f"""
1. Run full live collection:
   python run_full_pipeline_live.py
   
2. This will automatically:
   • Collect ~1,500-1,700 articles
   • Create articles_with_sentiment_real.csv
   • Generate firm_week_panel.csv

3. Then analyze:
   python -c "
import pandas as pd
panel = pd.read_csv('data/processed/firm_week_panel.csv')

# Your econometric analysis...
print(panel.describe())
   "
""")

print("\n" + "=" * 90)
print("✓ Panel data ready. See data/processed/firm_week_panel.csv")
print("=" * 90)
