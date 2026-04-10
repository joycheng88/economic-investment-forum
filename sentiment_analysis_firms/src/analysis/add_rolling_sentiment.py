"""
Add Rolling 4-Week Average to Panel Data
Computes rolling sentiment for each firm
"""

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_rolling_sentiment(input_csv, output_csv=None):
    """
    Add rolling 4-week average sentiment to panel data.
    
    Args:
        input_csv: Path to firm_week_panel.csv
        output_csv: Path to save updated panel (defaults to overwrite input)
        
    Returns:
        DataFrame with new rolling_sentiment_4w column
    """
    
    if output_csv is None:
        output_csv = input_csv
    
    logger.info(f"Loading panel data from: {input_csv}")
    panel = pd.read_csv(input_csv)
    
    logger.info(f"Loaded {len(panel)} observations")
    logger.info(f"Firms: {panel['firm'].nunique()}")
    logger.info(f"Weeks: {panel['week'].nunique()}")
    
    # Ensure data is sorted by firm and week
    logger.info("Sorting by firm and week...")
    panel = panel.sort_values(['firm', 'week']).reset_index(drop=True)
    
    # Convert week to datetime for proper sorting (optional - using string works too)
    # panel['week_dt'] = pd.to_datetime(panel['week'] + '-1', format='%Y-W%W-%w')
    
    # Calculate rolling 4-week average for each firm
    logger.info("Computing rolling 4-week average sentiment per firm...")
    
    panel['rolling_sentiment_4w'] = panel.groupby('firm')['avg_sentiment'].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean()
    )
    
    # Round to 4 decimal places
    panel['rolling_sentiment_4w'] = panel['rolling_sentiment_4w'].round(4)
    
    logger.info("✓ Rolling sentiment calculated")
    
    # Show sample by firm
    logger.info("\nSample data for first firm:")
    first_firm = panel['firm'].iloc[0]
    firm_data = panel[panel['firm'] == first_firm][['firm', 'week', 'avg_sentiment', 'rolling_sentiment_4w', 'article_count']]
    logger.info(f"\n{firm_data.to_string(index=False)}")
    
    # Summary statistics
    logger.info("\nRolling sentiment statistics:")
    logger.info(f"  Min:  {panel['rolling_sentiment_4w'].min():.4f}")
    logger.info(f"  Mean: {panel['rolling_sentiment_4w'].mean():.4f}")
    logger.info(f"  Max:  {panel['rolling_sentiment_4w'].max():.4f}")
    logger.info(f"  Std:  {panel['rolling_sentiment_4w'].std():.4f}")
    
    # Count rolling values by firm
    logger.info("\nRolling sentiment availability by firm:")
    rolling_counts = panel.groupby('firm').apply(
        lambda x: {
            'total_weeks': len(x),
            'rolling_available': (x['rolling_sentiment_4w'].notna()).sum(),
            'pct_available': f"{100*(x['rolling_sentiment_4w'].notna()).sum()/len(x):.0f}%"
        }
    )
    
    for firm, counts in rolling_counts.items():
        logger.info(f"  {firm:<15} {counts['total_weeks']} weeks, {counts['rolling_available']} with rolling ({counts['pct_available']})")
    
    # Save updated panel
    logger.info(f"\nSaving updated panel to: {output_csv}")
    panel.to_csv(output_csv, index=False)
    logger.info(f"✓ Saved {len(panel)} observations with rolling_sentiment_4w")
    
    return panel


if __name__ == '__main__':
    input_file = 'data/processed/firm_week_panel.csv'
    
    logger.info("=" * 80)
    logger.info("ADD ROLLING 4-WEEK SENTIMENT AVERAGE")
    logger.info("=" * 80 + "\n")
    
    panel = add_rolling_sentiment(input_file)
    
    print("\n" + "=" * 80)
    print("✓ ROLLING SENTIMENT ADDED")
    print("=" * 80)
    
    print(f"\nFinal columns: {list(panel.columns)}")
    print(f"\nFinal data (all rows):")
    print(panel.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Usage:")
    print("  panel = pd.read_csv('data/processed/firm_week_panel.csv')")
    print("  panel[['firm', 'week', 'avg_sentiment', 'rolling_sentiment_4w']]")
    print("=" * 80)
