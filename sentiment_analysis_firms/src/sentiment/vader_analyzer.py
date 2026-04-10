"""
Sentiment Analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner)
Rule-based sentiment analysis optimized for news text and social media
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

logger = logging.getLogger(__name__)


class VADERSentimentAnalyzer:
    """
    VADER sentiment analyzer optimized for news articles and financial text.
    
    Output scores:
    - positive: 0.0-1.0
    - negative: 0.0-1.0
    - neutral: 0.0-1.0
    - compound: -1.0 to +1.0 (overall sentiment sentiment score)
    
    Classification:
    - positive: compound >= 0.05
    - negative: compound <= -0.05
    - neutral: -0.05 < compound < 0.05
    """
    
    def __init__(self):
        """Initialize VADER sentiment analyzer"""
        self.analyzer = SentimentIntensityAnalyzer()
        logger.info("Initialized VADER SentimentIntensityAnalyzer")
    
    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            dict with sentiment scores and label
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0,
                'label': 'neutral',
                'confidence': 0.0
            }
        
        # Get VADER scores
        scores = self.analyzer.polarity_scores(text)
        
        # Determine label based on compound score
        compound = scores['compound']
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        # Confidence based on absolute compound score
        confidence = abs(compound)
        
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': compound,
            'label': label,
            'confidence': confidence
        }
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'clean_text') -> pd.DataFrame:
        """
        Analyze sentiment for all articles in DataFrame.
        
        Args:
            df: DataFrame with articles
            text_column: Column containing text to analyze
            
        Returns:
            DataFrame with added sentiment columns
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        sentiment_data = []
        logger.info(f"Starting sentiment analysis on {len(df)} articles")
        
        for idx, text in enumerate(df[text_column]):
            result = self.analyze(text)
            sentiment_data.append(result)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1} articles")
        
        # Create DataFrame from sentiment results
        sentiment_df = pd.DataFrame(sentiment_data)
        
        # Combine with original data
        result_df = df.copy()
        for col in sentiment_df.columns:
            result_df[f'sentiment_{col}'] = sentiment_df[col]
        
        logger.info(f"Sentiment analysis complete: {len(result_df)} articles")
        
        return result_df
    
    def get_summary_stats(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics of sentiment.
        
        Args:
            df: DataFrame with sentiment columns
            
        Returns:
            dict: Summary statistics
        """
        if 'sentiment_compound' not in df.columns:
            raise ValueError("DataFrame must have 'sentiment_compound' column")
        
        label_counts = df['sentiment_label'].value_counts()
        
        return {
            'total_articles': len(df),
            'positive_count': label_counts.get('positive', 0),
            'negative_count': label_counts.get('negative', 0),
            'neutral_count': label_counts.get('neutral', 0),
            'positive_pct': 100 * label_counts.get('positive', 0) / len(df),
            'negative_pct': 100 * label_counts.get('negative', 0) / len(df),
            'neutral_pct': 100 * label_counts.get('neutral', 0) / len(df),
            'avg_compound_score': df['sentiment_compound'].mean(),
            'median_compound_score': df['sentiment_compound'].median(),
            'std_compound_score': df['sentiment_compound'].std(),
            'max_compound_score': df['sentiment_compound'].max(),
            'min_compound_score': df['sentiment_compound'].min(),
            'avg_confidence': df['sentiment_confidence'].mean()
        }
