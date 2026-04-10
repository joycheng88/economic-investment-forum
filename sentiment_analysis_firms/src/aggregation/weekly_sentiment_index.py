"""
Weekly Sentiment Index Creation
Aggregates sentiment scores by week for market and firm-level analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WeeklySentimentIndex:
    """
    Creates weekly sentiment indices for market and firm-level analysis.
    
    Performs aggregation across all articles in each week to create
    a weekly sentiment pulse for the market and each firm.
    """
    
    def __init__(self):
        logger.info("Initialized WeeklySentimentIndex")
    
    def create_weekly_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create overall market weekly sentiment indices.
        
        Args:
            df: DataFrame with sentiment analysis results
            
        Returns:
            DataFrame with weekly market sentiment indices
        """
        # Ensure datetime column
        df['published_date'] = pd.to_datetime(df['published_date'])
        
        # Add week identifier
        df['week_start'] = df['published_date'].dt.to_period('W').apply(lambda r: r.start_time)
        
        logger.info(f"Creating weekly indices for {len(df)} articles")
        
        # Create weekly indices
        weekly_data = []
        
        for week, group in df.groupby('week_start'):
            if len(group) == 0:
                continue
            
            week_summary = {
                'week_start': week.date(),
                'week_end': (week + pd.Timedelta(days=6)).date(),
                'num_articles': len(group),
                'num_firms': group['detected_firm'].nunique(),
                'num_keywords': group['detected_keywords'].nunique(),
                
                # Sentiment metrics (compound score is primary metric)
                'avg_compound_sentiment': group['sentiment_compound'].mean(),
                'median_compound_sentiment': group['sentiment_compound'].median(),
                'std_compound_sentiment': group['sentiment_compound'].std(),
                'min_compound_sentiment': group['sentiment_compound'].min(),
                'max_compound_sentiment': group['sentiment_compound'].max(),
                
                # Sentiment distribution
                'positive_count': (group['sentiment_label'] == 'positive').sum(),
                'negative_count': (group['sentiment_label'] == 'negative').sum(),
                'neutral_count': (group['sentiment_label'] == 'neutral').sum(),
                'positive_pct': 100 * (group['sentiment_label'] == 'positive').sum() / len(group),
                'negative_pct': 100 * (group['sentiment_label'] == 'negative').sum() / len(group),
                'neutral_pct': 100 * (group['sentiment_label'] == 'neutral').sum() / len(group),
                
                # Confidence
                'avg_confidence': group['sentiment_confidence'].mean(),
                
                # Component scores
                'avg_positive_score': group['sentiment_positive'].mean(),
                'avg_negative_score': group['sentiment_negative'].mean(),
                'avg_neutral_score': group['sentiment_neutral'].mean(),
            }
            
            weekly_data.append(week_summary)
        
        weekly_df = pd.DataFrame(weekly_data)
        logger.info(f"Created {len(weekly_df)} weeks of market sentiment data")
        
        return weekly_df.sort_values('week_start').reset_index(drop=True)
    
    def create_firm_weekly_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weekly sentiment indices by firm.
        
        Args:
            df: DataFrame with sentiment analysis results
            
        Returns:
            DataFrame with firm-week sentiment indices
        """
        df['published_date'] = pd.to_datetime(df['published_date'])
        df['week_start'] = df['published_date'].dt.to_period('W').apply(lambda r: r.start_time)
        
        logger.info(f"Creating firm-level weekly indices")
        
        firm_weekly_data = []
        
        for firm in sorted(df['detected_firm'].unique()):
            firm_df = df[df['detected_firm'] == firm]
            
            for week, group in firm_df.groupby('week_start'):
                if len(group) == 0:
                    continue
                
                week_ts = pd.Timestamp(week)
                firm_summary = {
                    'firm': firm,
                    'week_start': week_ts.date(),
                    'week_end': (week_ts + pd.Timedelta(days=6)).date(),
                    'num_articles': len(group),
                    'num_keywords': group['detected_keywords'].nunique(),
                    
                    'avg_compound_sentiment': group['sentiment_compound'].mean(),
                    'median_compound_sentiment': group['sentiment_compound'].median(),
                    'std_compound_sentiment': group['sentiment_compound'].std(),
                    'min_compound_sentiment': group['sentiment_compound'].min(),
                    'max_compound_sentiment': group['sentiment_compound'].max(),
                    
                    'positive_count': (group['sentiment_label'] == 'positive').sum(),
                    'negative_count': (group['sentiment_label'] == 'negative').sum(),
                    'neutral_count': (group['sentiment_label'] == 'neutral').sum(),
                    'positive_pct': 100 * (group['sentiment_label'] == 'positive').sum() / len(group),
                    'negative_pct': 100 * (group['sentiment_label'] == 'negative').sum() / len(group),
                    'neutral_pct': 100 * (group['sentiment_label'] == 'neutral').sum() / len(group),
                    
                    'avg_confidence': group['sentiment_confidence'].mean(),
                    
                    'avg_positive_score': group['sentiment_positive'].mean(),
                    'avg_negative_score': group['sentiment_negative'].mean(),
                    'avg_neutral_score': group['sentiment_neutral'].mean(),
                }
                
                firm_weekly_data.append(firm_summary)
        
        firm_weekly_df = pd.DataFrame(firm_weekly_data)
        logger.info(f"Created {len(firm_weekly_df)} firm-week combinations")
        
        return firm_weekly_df.sort_values(['firm', 'week_start']).reset_index(drop=True)
    
    def create_keyword_weekly_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weekly sentiment indices by keyword.
        
        Args:
            df: DataFrame with sentiment analysis results
            
        Returns:
            DataFrame with keyword-week sentiment indices
        """
        df['published_date'] = pd.to_datetime(df['published_date'])
        df['week_start'] = df['published_date'].dt.to_period('W').apply(lambda r: r.start_time)
        
        # Expand keywords (each article may have multiple keywords)
        expanded_rows = []
        for idx, row in df.iterrows():
            keywords = str(row['detected_keywords']).split(',')
            for keyword in keywords:
                keyword = keyword.strip()
                new_row = row.copy()
                new_row['keyword'] = keyword
                expanded_rows.append(new_row)
        
        df_expanded = pd.DataFrame(expanded_rows)
        
        logger.info(f"Creating keyword-level weekly indices")
        
        keyword_weekly_data = []
        
        for keyword in sorted(df_expanded['keyword'].unique()):
            keyword_df = df_expanded[df_expanded['keyword'] == keyword]
            
            for week, group in keyword_df.groupby('week_start'):
                if len(group) == 0:
                    continue
                
                week_ts = pd.Timestamp(week)
                keyword_summary = {
                    'keyword': keyword,
                    'week_start': week_ts.date(),
                    'week_end': (week_ts + pd.Timedelta(days=6)).date(),
                    'num_articles': len(group),
                    'num_firms': group['detected_firm'].nunique(),
                    
                    'avg_compound_sentiment': group['sentiment_compound'].mean(),
                    'median_compound_sentiment': group['sentiment_compound'].median(),
                    'std_compound_sentiment': group['sentiment_compound'].std(),
                    
                    'positive_pct': 100 * (group['sentiment_label'] == 'positive').sum() / len(group),
                    'negative_pct': 100 * (group['sentiment_label'] == 'negative').sum() / len(group),
                    'neutral_pct': 100 * (group['sentiment_label'] == 'neutral').sum() / len(group),
                    
                    'avg_confidence': group['sentiment_confidence'].mean(),
                }
                
                keyword_weekly_data.append(keyword_summary)
        
        keyword_weekly_df = pd.DataFrame(keyword_weekly_data)
        logger.info(f"Created {len(keyword_weekly_df)} keyword-week combinations")
        
        return keyword_weekly_df.sort_values(['keyword', 'week_start']).reset_index(drop=True)
