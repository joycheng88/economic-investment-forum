"""
Rolling Sentiment Demonstration
Shows how rolling 4-week average works for time series analysis
"""

import pandas as pd
import numpy as np

print("=" * 90)
print("ROLLING 4-WEEK SENTIMENT AVERAGE: Demo & Usage")
print("=" * 90)

# Load panel data
panel = pd.read_csv('data/processed/firm_week_panel.csv')

print(f"\nPanel Data Summary:")
print(f"  Rows: {len(panel)}")
print(f"  Columns: {list(panel.columns)}")
print(f"  Firms: {panel['firm'].nunique()}")

# Show data with rolling sentiment
print(f"\n{'='*90}")
print("COMPLETE PANEL WITH ROLLING SENTIMENT")
print(f"{'='*90}")
print(f"\n{panel[['firm', 'week', 'avg_sentiment', 'rolling_sentiment_4w']].to_string(index=False)}")

# Demonstration with synthetic data
print(f"\n{'='*90}")
print("EXAMPLE: How Rolling 4-Week Average Works")
print(f"{'='*90}")

synthetic_data = pd.DataFrame({
    'firm': ['TestCo'] * 8,
    'week': ['2026-W01', '2026-W02', '2026-W03', '2026-W04', '2026-W05', '2026-W06', '2026-W07', '2026-W08'],
    'avg_sentiment': [0.1, 0.2, -0.1, 0.3, 0.0, 0.2, 0.4, -0.2]
})

synthetic_data = synthetic_data.sort_values(['firm', 'week']).reset_index(drop=True)
synthetic_data['rolling_sentiment_4w'] = synthetic_data.groupby('firm')['avg_sentiment'].transform(
    lambda x: x.rolling(window=4, min_periods=1).mean()
).round(4)

print(f"\nSynthetic time series (TestCo, 8 weeks):")
print(f"  Week 1: sentiment=+0.10 → rolling=+0.10 (only 1 data point)")
print(f"  Week 2: sentiment=+0.20 → rolling=+0.15 (avg of 2 points: 0.1, 0.2)")
print(f"  Week 3: sentiment=-0.10 → rolling=+0.07 (avg of 3 points: 0.1, 0.2, -0.1)")
print(f"  Week 4: sentiment=+0.30 → rolling=+0.10 (avg of 4 points: 0.1, 0.2, -0.1, 0.3)")
print(f"  Week 5: sentiment=+0.00 → rolling=+0.10 (rolling window, last 4: 0.2, -0.1, 0.3, 0.0)")
print(f"  ... (rolling window continues)")

print(f"\n{synthetic_data[['firm', 'week', 'avg_sentiment', 'rolling_sentiment_4w']].to_string(index=False)}")

# Use cases
print(f"\n{'='*90}")
print("USE CASES: Rolling Sentiment in Analysis")
print(f"{'='*90}")

use_cases = """
1. TREND DETECTION
   - Compare avg_sentiment vs rolling_sentiment_4w
   - If rolling > avg_sentiment: recent negative sentiment reversal
   - If rolling < avg_sentiment: recent positive sentiment reversal
   
2. SMOOTHING NOISE
   - Individual weeks can be noisy (few articles)
   - Rolling average smooths out weekly volatility
   - Reveals underlying trend

3. MOMENTUM ANALYSIS
   - Calculate change in rolling sentiment: diff(rolling_sentiment_4w)
   - Positive change = improving firm sentiment
   - Negative change = deteriorating firm sentiment

4. SENTIMENT PERSISTENCE
   - If rolling_sentiment_4w ≈ avg_sentiment: sentiment stable
   - If rolling_sentiment_4w >> avg_sentiment: recent sharp decline
   - If rolling_sentiment_4w << avg_sentiment: recent sharp rise

5. PREDICTIVE SIGNAL
   - Does rolling sentiment predict next week's returns?
   - Test: corr(rolling_sentiment_4w, future_returns)
   
6. COMPARATIVE ANALYSIS
   - Which firms have improving sentiment? (increasing rolling_sentiment_4w)
   - Which firms have deteriorating sentiment? (decreasing rolling_sentiment_4w)
   - Compare sentiment trends across firms
"""

print(use_cases)

# Analysis examples
print(f"\n{'='*90}")
print("EXAMPLE ANALYSES")
print(f"{'='*90}")

# Firms with multiple observations
multi_week_firms = panel.groupby('firm').size()
multi_week_firms = multi_week_firms[multi_week_firms > 1]

print(f"\nFirms with multiple time periods (for rolling average):")
for firm in multi_week_firms.index:
    firm_data = panel[panel['firm'] == firm].sort_values('week')[['week', 'avg_sentiment', 'rolling_sentiment_4w']]
    print(f"\n  {firm}:")
    print(f"    {firm_data.to_string(index=False)}")

# Sentiment dynamics
print(f"\n\nSentiment Dynamics:")
print(f"  Firms with positive sentiment:")
positive = panel[panel['avg_sentiment'] > 0][['firm', 'week', 'avg_sentiment', 'rolling_sentiment_4w']]
if len(positive) > 0:
    print(f"    {positive.to_string(index=False)}")
else:
    print(f"    None")

print(f"\n  Firms with zero sentiment:")
neutral = panel[panel['avg_sentiment'] == 0][['firm', 'week', 'avg_sentiment', 'rolling_sentiment_4w']]
if len(neutral) > 0:
    print(f"    {neutral.shape[0]} observations")
else:
    print(f"    None")

# Next steps with real data
print(f"\n{'='*90}")
print("EXPECTED WITH REAL DATA (1,000+ observations)")
print(f"{'='*90}")

print(f"""
Once live collection generates rich panel data:
  • Hundreds of firm-week observations
  • Multiple observations per firm
  • Rolling sentiment will smooth noise
  • Can identify sentiment trends

Example 1: Detect sentiment reversal
   firm_data = panel[panel['firm'] == 'PepsiCo'].sort_values('week')
   sentiment_improving = firm_data['rolling_sentiment_4w'].diff() > 0
   
Example 2: Momentum indicator
   panel['sentiment_momentum'] = panel.groupby('firm')['rolling_sentiment_4w'].diff()
   
Example 3: Mean reversion signal
   panel['sentiment_deviation'] = panel['avg_sentiment'] - panel['rolling_sentiment_4w']
   
Example 4: Predictive regression
   from statsmodels.formula.api import ols
   model = ols('next_week_return ~ rolling_sentiment_4w + C(firm)', data=merged).fit()
   
Example 5: Cross-firm comparison
   firm_trends = panel.groupby('firm')['rolling_sentiment_4w'].mean().sort_values(ascending=False)
""")

print(f"\n{'='*90}")
print("✓ Rolling sentiment added to panel data")
print(f"{'='*90}")
print(f"\nFile: data/processed/firm_week_panel.csv")
print(f"New column: rolling_sentiment_4w (rolling mean of avg_sentiment with 4-week window)")
