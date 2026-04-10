"""
Create Firm-Week Panel Data
Groups sentiment analysis results by firm and ISO week
Computes average sentiment, article count, and sentiment std dev
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_firm_week_panel(input_csv, output_csv):
    """
    Create firm-week panel data from sentiment analysis results.
    
    Args:
        input_csv: Path to sentiment analysis output CSV
        output_csv: Path to save panel data
        
    Returns:
        DataFrame with columns: firm, week, avg_sentiment, article_count, sentiment_std
    """
    
    logger.info(f"Loading sentiment data from: {input_csv}")
    
    # Read data
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} articles")
    
    # Ensure published_date is datetime
    logger.info("Converting published_date to datetime...")
    df['published_date'] = pd.to_datetime(df['published_date'])
    
    # Extract ISO calendar (year, week, day)
    logger.info("Extracting year-week from published_date...")
    df['iso_year'] = df['published_date'].dt.isocalendar().year
    df['iso_week'] = df['published_date'].dt.isocalendar().week
    df['week'] = df['iso_year'].astype(str) + '-W' + df['iso_week'].astype(str).str.zfill(2)
    
    logger.info(f"Date range: {df['published_date'].min()} to {df['published_date'].max()}")
    logger.info(f"Weeks covered: {df['week'].nunique()}")
    
    # Get sentiment column name (handle different possible names)
    sentiment_col = None
    for col in ['sentiment_compound', 'finbert_score', 'sentiment_score']:
        if col in df.columns:
            sentiment_col = col
            break
    
    if sentiment_col is None:
        raise ValueError(f"No sentiment column found. Available: {df.columns.tolist()}")
    
    logger.info(f"Using sentiment column: {sentiment_col}")
    
    # Get firm column name (handle different possible names)
    firm_col = None
    for col in ['detected_firm', 'firm_name', 'firm']:
        if col in df.columns:
            firm_col = col
            break
    
    if firm_col is None:
        raise ValueError(f"No firm column found. Available: {df.columns.tolist()}")
    
    logger.info(f"Using firm column: {firm_col}")
    
    # Check for missing values
    logger.info(f"\nData quality check:")
    logger.info(f"  Missing dates: {df['published_date'].isna().sum()}")
    logger.info(f"  Missing firms: {df[firm_col].isna().sum()}")
    logger.info(f"  Missing sentiment: {df[sentiment_col].isna().sum()}")
    
    # Remove rows with missing critical values
    df_clean = df.dropna(subset=['published_date', firm_col, sentiment_col])
    logger.info(f"  After removing nulls: {len(df_clean)} articles ({100*len(df_clean)/len(df):.1f}%)")
    
    # Group by firm and week
    logger.info("\nGrouping by firm and week...")
    grouped = df_clean.groupby([firm_col, 'week']).agg({
        sentiment_col: ['mean', 'std', 'count'],
        'published_date': 'min'  # For reference
    }).reset_index()
    
    # Flatten column names
    grouped.columns = [firm_col, 'week', 'avg_sentiment', 'sentiment_std', 'article_count', 'week_start']
    
    # Reorder and rename columns to match requirements
    panel = grouped[[firm_col, 'week', 'avg_sentiment', 'article_count', 'sentiment_std']].copy()
    panel.columns = ['firm', 'week', 'avg_sentiment', 'article_count', 'sentiment_std']
    
    # Round numerical columns
    panel['avg_sentiment'] = panel['avg_sentiment'].round(4)
    panel['sentiment_std'] = panel['sentiment_std'].round(4)
    
    # Sort by firm and week
    panel = panel.sort_values(['firm', 'week']).reset_index(drop=True)
    
    # Log summary
    logger.info(f"\nPanel Summary:")
    logger.info(f"  Total rows: {len(panel)}")
    logger.info(f"  Unique firms: {panel['firm'].nunique()}")
    logger.info(f"  Unique weeks: {panel['week'].nunique()}")
    logger.info(f"  Total articles: {panel['article_count'].sum()}")
    logger.info(f"  Avg articles per firm-week: {panel['article_count'].mean():.1f}")
    
    # Show sample
    logger.info(f"\nSample rows:")
    logger.info(f"\n{panel.head(10).to_string(index=False)}")
    
    # Show firms
    logger.info(f"\nFirms in panel:")
    firm_counts = panel.groupby('firm')['article_count'].agg(['count', 'sum'])
    firm_counts.columns = ['num_weeks', 'total_articles']
    logger.info(f"\n{firm_counts.to_string()}")
    
    # Save to CSV
    logger.info(f"\nSaving to: {output_csv}")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_csv, index=False)
    logger.info(f"✓ Saved {len(panel)} firm-week observations")
    
    return panel


if __name__ == '__main__':
    # Find latest sentiment analysis file
    import glob
    
    sentiment_files = glob.glob('data/processed/*sentiment*.csv')
    
    if not sentiment_files:
        print("No sentiment analysis files found in data/processed/")
        print("Available files:")
        print(glob.glob('data/processed/*.csv'))
        exit(1)
    
    # Use most recent file
    input_file = max(sentiment_files, key=lambda x: Path(x).stat().st_mtime)
    output_file = 'data/processed/firm_week_panel.csv'
    
    logger.info(f"Using sentiment file: {input_file}")
    logger.info(f"Output file: {output_file}\n")
    
    panel = create_firm_week_panel(input_file, output_file)
    
    print("\n" + "="*80)
    print("✓ FIRM-WEEK PANEL CREATED")
    print("="*80)
    print(f"\nFile: {output_file}")
    print(f"Rows: {len(panel)}")
    print(f"Columns: {', '.join(panel.columns.tolist())}")
    print("\nReady for econometric analysis!")
