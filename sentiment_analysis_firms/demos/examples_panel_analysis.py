"""
Example: Using Firm-Week Panel Data for Analysis
Shows how to use the panel data for econometric analysis
"""

import pandas as pd
import numpy as np

# Load panel data
panel = pd.read_csv('data/processed/firm_week_panel.csv')

print("=" * 80)
print("FIRM-WEEK PANEL DATA EXAMPLE")
print("=" * 80)

print(f"\nDataset: {len(panel)} observations")
print(f"Dimensions: {panel['firm'].nunique()} firms × {panel['week'].nunique()} weeks")

# Show data structure
print(f"\nColumns:")
print(f"  firm            - Firm name (10 total)")
print(f"  week            - ISO week (format: YYYY-Www)")
print(f"  avg_sentiment   - Average sentiment score (-1 to +1)")
print(f"  article_count   - Number of articles in firm-week")
print(f"  sentiment_std   - Std dev of sentiment (NaN if only 1 article)")

# Example 1: Aggregate by firm
print("\n" + "=" * 80)
print("EXAMPLE 1: Aggregate by Firm")
print("=" * 80)

firm_stats = panel.groupby('firm').agg({
    'avg_sentiment': ['mean', 'std', 'min', 'max'],
    'article_count': 'sum',
    'week': 'count'
}).round(3)

firm_stats.columns = ['avg_sentiment_mean', 'avg_sentiment_std', 'min_sentiment', 'max_sentiment', 'total_articles', 'num_weeks']
print(f"\n{firm_stats.to_string()}")

print(f"\nInterpretation:")
print(f"  Wonderful: Highest average sentiment ({firm_stats.loc['Wonderful', 'avg_sentiment_mean']:.3f})")
print(f"  PepsiCo:   Appeared in most weeks ({firm_stats.loc['PepsiCo', 'num_weeks']:.0f} weeks)")

# Example 2: Aggregate by week
print("\n" + "=" * 80)
print("EXAMPLE 2: Aggregate by Week")
print("=" * 80)

week_stats = panel.groupby('week').agg({
    'avg_sentiment': 'mean',
    'article_count': ['sum', 'count'],
}).round(3)

week_stats.columns = ['market_sentiment', 'total_articles', 'num_firms']
print(f"\n{week_stats.to_string()}")

print(f"\nInterpretation:")
print(f"  Market-level sentiment by week")
print(f"  Can track sentiment trends over time)")

# Example 3: Filter and analyze
print("\n" + "=" * 80)
print("EXAMPLE 3: Filtering Examples")
print("=" * 80)

# High sentiment weeks
high_sentiment = panel[panel['avg_sentiment'] > 0.1]
print(f"\nHigh sentiment observations (score > 0.1):")
print(f"{high_sentiment[['firm', 'week', 'avg_sentiment', 'article_count']].to_string(index=False)}")

# Most recent week
latest_week = panel['week'].max()
print(f"\nLatest week ({latest_week}) data:")
latest_data = panel[panel['week'] == latest_week].sort_values('avg_sentiment', ascending=False)
print(f"{latest_data[['firm', 'avg_sentiment', 'article_count']].to_string(index=False)}")

# Example 4: Time series
print("\n" + "=" * 80)
print("EXAMPLE 4: Time Series Analysis (for real data)")
print("=" * 80)

print(f"""
Once you have real data (multiple articles per firm-week):

1. Create time series for each firm:
   firm_ts = panel[panel['firm'] == 'PepsiCo'].sort_values('week')
   
2. Calculate returns:
   firm_ts['sentiment_change'] = firm_ts['avg_sentiment'].diff()
   
3. Merge with stock returns:
   df_merged = firm_ts.merge(stock_returns, on=['firm', 'week'])
   
4. Run regression:
   from sklearn.linear_model import LinearRegression
   X = firm_ts[['avg_sentiment']]
   y = stock_returns['returns']
   model = LinearRegression().fit(X, y)

5. Test predictability:
   # Does sentiment predict next week's returns?
   df['next_week_return'] = df['return'].shift(-1)
   correlation = df['avg_sentiment'].corr(df['next_week_return'])
""")

# Example 5: Data summary
print("\n" + "=" * 80)
print("DATA SUMMARY")
print("=" * 80)

print(f"\nColumns and dtypes:")
print(panel.dtypes)

print(f"\nMissing values:")
print(panel.isna().sum())

print(f"\nValue ranges:")
print(f"  avg_sentiment: [{panel['avg_sentiment'].min():.3f}, {panel['avg_sentiment'].max():.3f}]")
print(f"  article_count: [{panel['article_count'].min()}, {panel['article_count'].max()}]")
print(f"  weeks: {panel['week'].min()} to {panel['week'].max()}")

print("\n" + "=" * 80)
print("✓ Panel data ready for econometric analysis!")
print("=" * 80)

# Save as Python object for easy import
print("\nTo use in your analysis script:")
print("""import pandas as pd

panel = pd.read_csv('data/processed/firm_week_panel.csv')

# Your analysis code here...
""")
