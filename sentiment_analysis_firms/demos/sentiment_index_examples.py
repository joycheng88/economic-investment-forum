"""
Sentiment Index Analysis Examples

Demonstrates how to use the standardized sentiment index for econometric analysis.

Three sentiment measures in outputs/sentiment_index.csv:
  1. avg_sentiment: Raw sentiment (-1 to +1)
  2. rolling_sentiment_4w: 4-week moving average per firm
  3. z_sentiment: Standardized within-week (mean≈0, std≈1)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_sentiment_index(filepath='outputs/sentiment_index.csv'):
    """Load final sentiment index."""
    index = pd.read_csv(filepath)
    index['week'] = pd.Categorical(index['week'], ordered=True)
    return index.sort_values(['firm', 'week'])


def example_1_weekly_comparison():
    """
    EXAMPLE 1: Compare firms within each week
    
    Question: Which firms are outperforming vs. underperforming each week?
    Solution: Use z_sentiment (standardized within-week)
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: WEEKLY FIRM COMPARISON")
    print("="*70)
    
    index = load_sentiment_index()
    
    # Show z-sentiment by week (firms ranked by sentiment)
    for week in index['week'].unique():
        week_data = index[index['week'] == week].sort_values('z_sentiment', ascending=False)
        print(f"\nWeek {week}:")
        print(f"  Mean sentiment: {week_data['avg_sentiment'].mean():.4f}")
        print(f"  Std sentiment: {week_data['avg_sentiment'].std():.4f}")
        print(f"  Rankings (by z-sentiment):")
        for _, row in week_data.iterrows():
            star = "⭐" if row['z_sentiment'] > 1 else "📍" if row['z_sentiment'] > 0 else "📉"
            print(f"    {star} {row['firm']:<20} z={row['z_sentiment']:>7.3f} raw={row['avg_sentiment']:>7.3f}")


def example_2_firm_rankings():
    """
    EXAMPLE 2: Firm-level sentiment persistence
    
    Question: Which firms consistently have positive/negative sentiment?
    Solution: Aggregate z_sentiment across weeks
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: FIRM SENTIMENT PERSISTENCE")
    print("="*70)
    
    index = load_sentiment_index()
    
    # Aggregate sentiment across all weeks
    firm_stats = index.groupby('firm').agg({
        'avg_sentiment': ['mean', 'std', 'min', 'max'],
        'z_sentiment': ['mean', 'count'],
        'article_count': 'sum'
    }).round(4)
    
    firm_stats.columns = ['raw_mean', 'raw_std', 'raw_min', 'raw_max', 'z_mean', 'observations', 'total_articles']
    firm_stats = firm_stats.sort_values('raw_mean', ascending=False)
    
    print("\nFirm Sentiment Ranking (across all weeks):")
    print(firm_stats.to_string())
    
    print("\nInterpretation:")
    print("  raw_mean > 0        : Mostly positive sentiment")
    print("  raw_mean ≈ 0        : Mixed/neutral sentiment")
    print("  raw_mean < 0        : Mostly negative sentiment")
    print("  z_mean > 0          : Above-average sentiment across weeks")
    print("  raw_std high        : Volatile sentiment (swings between +/- )")


def example_3_extreme_sentiment():
    """
    EXAMPLE 3: Identify extreme sentiment observations
    
    Question: Which firm-weeks had unusual sentiment?
    Solution: Filter on |z_sentiment| > threshold
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: EXTREME SENTIMENT DETECTION")
    print("="*70)
    
    index = load_sentiment_index()
    
    # Define extremes
    extreme_threshold = 1.0  # Standard deviation units (±1σ is ~68% of distribution)
    
    positive_extreme = index[index['z_sentiment'] > extreme_threshold].sort_values('z_sentiment', ascending=False)
    negative_extreme = index[index['z_sentiment'] < -extreme_threshold].sort_values('z_sentiment')
    
    print(f"\nPositive Extremes (z > {extreme_threshold}):")
    if len(positive_extreme) > 0:
        print(positive_extreme[['firm', 'week', 'avg_sentiment', 'z_sentiment']].to_string(index=False))
    else:
        print("  (No observations)")
    
    print(f"\nNegative Extremes (z < -{extreme_threshold}):")
    if len(negative_extreme) > 0:
        print(negative_extreme[['firm', 'week', 'avg_sentiment', 'z_sentiment']].to_string(index=False))
    else:
        print("  (No observations)")
    
    print(f"\nStatistical interpretation:")
    print(f"  |z| > {extreme_threshold:.1f}  : ~32% of observations (1 SD from mean)")
    print(f"  |z| > 1.96: ~5% of observations (top/bottom 2.5%)")
    print(f"  |z| > 2.58: ~1% of observations (top/bottom 0.5%)")


def example_4_rolling_trends():
    """
    EXAMPLE 4: Identify sentiment trends using rolling averages
    
    Question: Which firms have improving vs. deteriorating sentiment?
    Solution: Compare rolling_sentiment_4w across time
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: SENTIMENT TRENDS (ROLLING AVERAGES)")
    print("="*70)
    
    index = load_sentiment_index()
    
    # For firms with multiple observations, show trend
    print("\nSentiment Trends (rolling_sentiment_4w can show multi-week momentum):")
    for firm in sorted(index['firm'].unique()):
        firm_data = index[index['firm'] == firm].sort_values('week')
        if len(firm_data) > 1:
            print(f"\n  {firm}:")
            for _, row in firm_data.iterrows():
                trend = "↑" if row['rolling_sentiment_4w'] > row['avg_sentiment'] else "↓" if row['rolling_sentiment_4w'] < row['avg_sentiment'] else "→"
                print(f"    {row['week']}: raw={row['avg_sentiment']:>7.4f}  rolling={row['rolling_sentiment_4w']:>7.4f}  {trend}")


def example_5_cross_week_standardization():
    """
    EXAMPLE 5: Why standardization matters
    
    Question: How does standardization differ from raw sentiment?
    Solution: Compare raw vs z-sentiment
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: STANDARDIZATION BENEFITS")
    print("="*70)
    
    index = load_sentiment_index()
    
    print("\nComparison by week:")
    for week in index['week'].unique():
        week_data = index[index['week'] == week]
        avg_raw = week_data['avg_sentiment'].mean()
        
        print(f"\n  {week}:")
        print(f"    Week average sentiment: {avg_raw:.4f}")
        print(f"    Firms with avg_sentiment < week_avg (raw perspective):")
        below_avg = week_data[week_data['avg_sentiment'] < avg_raw]
        for _, row in below_avg.iterrows():
            print(f"      {row['firm']:<20} raw={row['avg_sentiment']:>7.4f} (below average: {row['avg_sentiment']-avg_raw:>7.4f})")
        
        print(f"    Same firms (in z-score perspective):")
        for _, row in below_avg.iterrows():
            print(f"      {row['firm']:<20} z={row['z_sentiment']:>7.4f} (relative to week std)")
    
    print("\n  Why standardization (z-score) is useful:")
    print("    - Compares firms relative to their week's baseline")
    print("    - Accounts for weeks with high/low average sentiment")
    print("    - Enables cross-week comparison (same scale)")
    print("    - Identifies outliers in a statistically meaningful way")


def example_6_regression_features():
    """
    EXAMPLE 6: Using sentiment for econometric regression
    
    Question: How to prepare sentiment features for predictive models?
    Solution: Create lagged features and interaction terms
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: FEATURES FOR ECONOMETRIC REGRESSION")
    print("="*70)
    
    index = load_sentiment_index()
    
    # Create lagged features (for predictive models)
    index = index.sort_values(['firm', 'week'])
    index['z_sentiment_lag1'] = index.groupby('firm')['z_sentiment'].shift(1)
    index['z_sentiment_change'] = index['z_sentiment'].diff()
    index['z_extreme'] = (index['z_sentiment'].abs() > 1.0).astype(int)  # Binary indicator
    
    print("\nFeatures created for regression:")
    print("  z_sentiment         : Current standardized sentiment")
    print("  z_sentiment_lag1    : Previous week's standardized sentiment")
    print("  z_sentiment_change  : Change in standardized sentiment (momentum)")
    print("  z_extreme           : Binary indicator (|z| > 1 = unusual sentiment)")
    
    print("\nExample features for regression (with lags):")
    print(index[['firm', 'week', 'z_sentiment', 'z_sentiment_lag1', 'z_sentiment_change', 'z_extreme']].head(10).to_string(index=False))
    
    print("\nRegression formula (pseudo-code):")
    print("  stock_returns ~ z_sentiment + z_sentiment_lag1 + z_sentiment_change + z_extreme + firm_FE + week_FE")
    print("\n  Interpretation:")
    print("    z_sentiment         : Contemporaneous sentiment effect")
    print("    z_sentiment_lag1    : Lagged effect (past sentiment predicts returns)")
    print("    z_sentiment_change  : Momentum (acceleration of sentiment)")
    print("    z_extreme           : Non-linear effect (unusual sentiment matters more)")


def print_summary():
    """Print summary statistics."""
    print("\n" + "="*70)
    print("SUMMARY: SENTIMENT INDEX INTERPRETATION")
    print("="*70)
    
    index = load_sentiment_index()
    
    print(f"\nDataset: {len(index)} observations")
    print(f"Firms: {index['firm'].nunique()}")
    print(f"Weeks: {index['week'].nunique()}")
    print(f"Total articles covered: {index['article_count'].sum()}")
    
    print(f"\nSentiment Measures:")
    print(f"  avg_sentiment (raw):")
    print(f"    Range: [{index['avg_sentiment'].min():.4f}, {index['avg_sentiment'].max():.4f}]")
    print(f"    Mean:  {index['avg_sentiment'].mean():.4f}")
    print(f"    Std:   {index['avg_sentiment'].std():.4f}")
    
    print(f"\n  rolling_sentiment_4w (4-week moving average):")
    print(f"    Range: [{index['rolling_sentiment_4w'].min():.4f}, {index['rolling_sentiment_4w'].max():.4f}]")
    print(f"    Mean:  {index['rolling_sentiment_4w'].mean():.4f}")
    print(f"    Std:   {index['rolling_sentiment_4w'].std():.4f}")
    
    print(f"\n  z_sentiment (standardized within-week):")
    print(f"    Range: [{index['z_sentiment'].min():.4f}, {index['z_sentiment'].max():.4f}]")
    print(f"    Mean:  {index['z_sentiment'].mean():.4f} (≈0 as expected)")
    print(f"    Std:   {index['z_sentiment'].std():.4f} (≈1 as expected)")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("SENTIMENT INDEX ANALYSIS EXAMPLES")
    print("File: outputs/sentiment_index.csv")
    print("="*70)
    
    # Run examples
    example_1_weekly_comparison()
    example_2_firm_rankings()
    example_3_extreme_sentiment()
    example_4_rolling_trends()
    example_5_cross_week_standardization()
    example_6_regression_features()
    print_summary()
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Merge with stock returns data (weekly returns)")
    print("2. Run panel regression: returns ~ z_sentiment + controls + FE")
    print("3. Test Granger causality: sentiment → returns")
    print("4. Examine heterogeneous effects: which firms respond most to sentiment?")
    print("5. Deploy as real-time trading signal (top/bottom z quartiles)")
