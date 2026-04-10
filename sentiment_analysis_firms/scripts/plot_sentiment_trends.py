"""
Sentiment Trends Visualization and Analysis

Creates:
1. Time series plot of z_sentiment for each firm
2. Rankings table of firms by average sentiment
3. Saves both to outputs/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def week_to_date(week_str):
    """
    Convert ISO week string (YYYY-Www) to actual date (Monday of that week).
    """
    year, week = week_str.split('-W')
    year = int(year)
    week = int(week)
    
    jan4 = datetime(year, 1, 4)
    week_one_monday = jan4 - pd.Timedelta(days=jan4.weekday())
    target_monday = week_one_monday + pd.Timedelta(weeks=week-1)
    
    return target_monday


def plot_sentiment_trends(input_csv='outputs/sentiment_index.csv',
                         output_png='outputs/sentiment_trends.png'):
    """
    Create time series plot of z_sentiment by firm.
    
    Args:
        input_csv: Path to sentiment index CSV
        output_png: Path to save plot
    """
    
    print("="*70)
    print("SENTIMENT TRENDS ANALYSIS")
    print("="*70)
    
    # Load data
    print(f"\n1. Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"   Loaded {len(df)} observations across {df['firm'].nunique()} firms")
    
    # Convert weeks to dates
    print(f"\n2. Converting weeks to dates...")
    df['week_date'] = df['week'].apply(week_to_date)
    df = df.sort_values(['firm', 'week_date'])
    
    print(f"   Date range: {df['week_date'].min().date()} to {df['week_date'].max().date()}")
    
    # Get firms and colors
    firms = sorted(df['firm'].unique())
    n_firms = len(firms)
    print(f"\n3. Plotting {n_firms} firms...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color palette
    colors = plt.cm.tab20(np.linspace(0, 1, n_firms))
    
    # Plot each firm
    for i, firm in enumerate(firms):
        firm_data = df[df['firm'] == firm].sort_values('week_date')
        
        ax.plot(firm_data['week_date'], 
               firm_data['z_sentiment'],
               marker='o',
               linewidth=2,
               markersize=6,
               label=firm,
               color=colors[i],
               alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('Standardized Sentiment (z-score)', fontsize=12, fontweight='bold')
    ax.set_title('Firm Sentiment Trends Over Time\n(Standardized within-week)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # X-axis formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, ha='right')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Zero (weekly avg)')
    
    # Legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, framealpha=0.9)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    print(f"\n4. Saving plot to: {output_png}")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"   ✓ Plot saved")
    
    plt.close()  # Close without displaying
    
    return df


def compute_firm_rankings(df, output_csv='outputs/firm_sentiment_rankings.csv'):
    """
    Compute average sentiment per firm and create rankings.
    
    Args:
        df: DataFrame with sentiment data
        output_csv: Path to save rankings
    """
    
    print("\n" + "="*70)
    print("FIRM SENTIMENT RANKINGS")
    print("="*70)
    
    # Aggregate by firm
    print(f"\n1. Computing statistics by firm...")
    
    rankings = df.groupby('firm').agg({
        'z_sentiment': ['mean', 'median', 'std', 'min', 'max', 'count'],
        'avg_sentiment': ['mean', 'std'],
        'rolling_sentiment_4w': ['mean'],
        'article_count': 'sum'
    }).round(4)
    
    # Flatten column names
    rankings.columns = ['_'.join(col).strip() for col in rankings.columns.values]
    rankings = rankings.rename(columns={
        'z_sentiment_mean': 'z_avg',
        'z_sentiment_median': 'z_median',
        'z_sentiment_std': 'z_std',
        'z_sentiment_min': 'z_min',
        'z_sentiment_max': 'z_max',
        'z_sentiment_count': 'n_obs',
        'avg_sentiment_mean': 'raw_avg',
        'avg_sentiment_std': 'raw_std',
        'rolling_sentiment_4w_mean': 'rolling_avg',
        'article_count_sum': 'total_articles'
    })
    
    # Sort by z_sentiment average (most positive to most negative)
    rankings = rankings.sort_values('z_avg', ascending=False)
    rankings['rank'] = range(1, len(rankings) + 1)
    
    # Reorder columns
    rankings = rankings[['rank', 'z_avg', 'z_median', 'z_std', 'z_min', 'z_max', 'n_obs',
                         'raw_avg', 'raw_std', 'rolling_avg', 'total_articles']]
    
    print("\n2. Rankings Table (sorted by z_sentiment):")
    print(rankings.to_string())
    
    # Save to CSV
    print(f"\n3. Saving to: {output_csv}")
    rankings.to_csv(output_csv)
    print(f"   ✓ Rankings saved")
    
    return rankings


def print_summary(df, rankings):
    """Print summary statistics and insights."""
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nSample Overview:")
    print(f"  Total observations: {len(df)}")
    print(f"  Unique firms: {df['firm'].nunique()}")
    print(f"  Weeks covered: {df['week'].nunique()}")
    print(f"  Total articles: {df['article_count'].sum()}")
    
    print(f"\nZ-Sentiment Statistics (all firms combined):")
    print(f"  Mean:     {df['z_sentiment'].mean():.4f}")
    print(f"  Median:   {df['z_sentiment'].median():.4f}")
    print(f"  Std Dev:  {df['z_sentiment'].std():.4f}")
    print(f"  Min:      {df['z_sentiment'].min():.4f}")
    print(f"  Max:      {df['z_sentiment'].max():.4f}")
    
    print(f"\nTop 3 Most Positive Firms (by avg z_sentiment):")
    top_3 = rankings.head(3)
    for i, (firm, row) in enumerate(top_3.iterrows(), 1):
        print(f"  {i}. {firm:20} z_avg={row['z_avg']:7.4f}  raw_avg={row['raw_avg']:7.4f}")
    
    print(f"\nTop 3 Most Negative Firms (by avg z_sentiment):")
    bottom_3 = rankings.tail(3)
    for i, (firm, row) in enumerate(bottom_3.iloc[::-1].iterrows(), 1):
        print(f"  {i}. {firm:20} z_avg={row['z_avg']:7.4f}  raw_avg={row['raw_avg']:7.4f}")
    
    print(f"\nMost Volatile Firms (highest z_sentiment std):")
    volatile = rankings.nlargest(3, 'z_std')
    for i, (firm, row) in enumerate(volatile.iterrows(), 1):
        print(f"  {i}. {firm:20} z_std={row['z_std']:7.4f}  (range: {row['z_min']:.4f} to {row['z_max']:.4f})")
    
    print(f"\nMost Covered Firms (by article count):")
    coverage = rankings.nlargest(3, 'total_articles')
    for i, (firm, row) in enumerate(coverage.iterrows(), 1):
        print(f"  {i}. {firm:20} articles={int(row['total_articles']):3d}  obs={int(row['n_obs']):2d}")


def print_interpretation(df, rankings):
    """Print interpretation of results."""
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    print(f"""
What the plot shows:
  - X-axis: Time progression (weeks in April 2026)
  - Y-axis: Standardized sentiment (z-score, mean≈0 within each week)
  - Each line: One firm's sentiment trajectory
  - Red dashed line: Zero baseline (weekly average)
  
Reading the chart:
  ✓ Upward trend:  Improving sentiment over time
  ✓ Positive values: Above-average sentiment vs. week baseline
  ✓ Negative values: Below-average sentiment vs. week baseline
  ✓ Steep changes:  Volatile sentiment across weeks
  
Key insights from rankings:
  • Rank 1: Most positive average sentiment (defensive in GLP-1 era)
  • Rank {len(rankings)}: Most negative average sentiment (vulnerable position)
  • High z_std: Volatile perception (news-dependent)
  • Low z_std: Stable perception (steady narrative)
    """)


def main():
    """Main execution."""
    
    # Create plots and rankings
    df = plot_sentiment_trends()
    rankings = compute_firm_rankings(df)
    
    # Summary
    print_summary(df, rankings)
    print_interpretation(df, rankings)
    
    print("\n" + "="*70)
    print("OUTPUT FILES")
    print("="*70)
    print(f"\n✓ outputs/sentiment_trends.png")
    print(f"  Time series plot of z_sentiment by firm")
    print(f"  Size: 14x8 inches, 300 DPI")
    
    print(f"\n✓ outputs/firm_sentiment_rankings.csv")
    print(f"  Firm rankings by average sentiment")
    print(f"  Rows: {len(rankings)} firms")
    print(f"  Columns: rank, z_avg, z_median, z_std, z_min, z_max, n_obs, raw_avg, raw_std, rolling_avg, total_articles")
    
    return df, rankings


if __name__ == '__main__':
    df, rankings = main()
