"""
Create standardized sentiment index across firms within each week.

Within-week standardization (z-score):
Z_it = (S_it - mean_t) / std_t

Where:
  S_it = average sentiment for firm i in week t
  mean_t = mean sentiment across all firms in week t
  std_t = standard deviation of sentiment across all firms in week t
  Z_it = standardized sentiment for firm i in week t

Output includes:
  - avg_sentiment (raw)
  - rolling_sentiment_4w (rolling)
  - z_sentiment (standardized within-week)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def create_standardized_sentiment(input_csv='data/processed/firm_week_panel.csv',
                                  output_csv='outputs/sentiment_index.csv'):
    """
    Add standardized sentiment column to panel data.
    
    Args:
        input_csv: Path to firm_week_panel.csv
        output_csv: Path to save final sentiment index
        
    Returns:
        DataFrame with z_sentiment column added
    """
    
    # Load panel data
    print(f"Loading panel data from: {input_csv}")
    panel = pd.read_csv(input_csv)
    
    print(f"Loaded {len(panel)} observations")
    print(f"Columns: {list(panel.columns)}")
    
    # Standardize within each week
    print("\nStandardizing sentiment within each week...")
    
    def standardize_week(group):
        """Calculate z-score for firms within week."""
        mean_sentiment = group['avg_sentiment'].mean()
        std_sentiment = group['avg_sentiment'].std()
        
        # Avoid division by zero (if all firms have same sentiment in week)
        if std_sentiment == 0 or np.isnan(std_sentiment):
            group['z_sentiment'] = 0.0
        else:
            group['z_sentiment'] = (group['avg_sentiment'] - mean_sentiment) / std_sentiment
        
        return group
    
    # Apply standardization per week
    panel = panel.groupby('week', group_keys=False).apply(standardize_week)
    
    # Round to 4 decimals
    panel['z_sentiment'] = panel['z_sentiment'].round(4)
    
    print(f"✓ Z-sentiment calculated for all {len(panel)} observations")
    
    # Verify standardization
    print("\nVerification (by week):")
    verification = panel.groupby('week').agg({
        'avg_sentiment': ['mean', 'std', 'count'],
        'z_sentiment': ['mean', 'std', 'min', 'max']
    }).round(4)
    print(verification)
    
    # Statistics
    print(f"\nStandardized sentiment statistics (Z_it):")
    print(panel['z_sentiment'].describe().round(4))
    
    # Create output with all three sentiment measures
    output_df = panel[['firm', 'week', 'article_count', 
                       'avg_sentiment', 'rolling_sentiment_4w', 'z_sentiment']].copy()
    
    # Ensure columns are in correct order
    output_df = output_df[['firm', 'week', 'article_count', 
                           'avg_sentiment', 'rolling_sentiment_4w', 'z_sentiment']]
    
    # Sort by firm and week
    output_df = output_df.sort_values(['firm', 'week']).reset_index(drop=True)
    
    # Create output directory if it doesn't exist
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    output_df.to_csv(output_csv, index=False)
    print(f"\n✓ Final sentiment index saved to: {output_csv}")
    print(f"  Rows: {len(output_df)}")
    print(f"  Columns: {list(output_df.columns)}")
    
    return output_df


if __name__ == '__main__':
    # Create standardized sentiment
    sentiment_index = create_standardized_sentiment()
    
    # Display sample
    print("\n" + "="*60)
    print("SAMPLE DATA (first 15 rows):")
    print("="*60)
    print(sentiment_index.head(15).to_string())
    
    # Show sentiment types for one firm
    print("\n" + "="*60)
    print("EXAMPLE: PepsiCo Sentiment Across Weeks")
    print("="*60)
    pepsi = sentiment_index[sentiment_index['firm'] == 'PepsiCo'].copy()
    if len(pepsi) > 0:
        print(pepsi[['week', 'avg_sentiment', 'rolling_sentiment_4w', 'z_sentiment']].to_string())
    else:
        print("PepsiCo not found in data")
    
    print("\n" + "="*60)
    print("SENTIMENT INTERPRETATION:")
    print("="*60)
    print("avg_sentiment:       Raw sentiment score (-1 to +1)")
    print("rolling_sentiment_4w: 4-week moving average per firm")
    print("z_sentiment:          Standardized within-week (mean=0, sd~1)")
    print("                      Positive z: above week average")
    print("                      Negative z: below week average")
    print("                      |z| > 1.96: Top/bottom ~2.5% in week")
