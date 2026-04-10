"""
FinBERT Sentiment Analyzer for Financial Text
Uses HuggingFace's FinBERT model trained on financial data.
FinBERT outperforms general VADER on financial news sentiment.
"""

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from typing import Dict, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analysis for financial text.
    
    Designed specifically for financial news and earnings calls.
    Better performance than VADER on finance domain compared to general text.
    
    Model: ProsusAI/finbert
    Labels: positive (1), neutral (0), negative (-1)
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = None):
        """
        Initialize FinBERT model and tokenizer.
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cuda' or 'cpu', auto-detect if None
        """
        logger.info(f"Loading FinBERT model: {model_name}")
        
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # FinBERT label mapping
        # Original FinBERT: 0=negative, 1=neutral, 2=positive
        # We map to: -1=negative, 0=neutral, +1=positive
        self.label_map = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }
        self.score_map = {
            0: -1,  # negative
            1: 0,   # neutral
            2: 1    # positive
        }
        
        logger.info(f"Model loaded on device: {self.device}")
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze single text for sentiment.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with keys:
            - label: "positive", "neutral", or "negative"
            - score: +1, 0, or -1
            - confidence: probability of predicted label (0-1)
            - all_scores: dict with all label probabilities
        """
        if not text or len(str(text).strip()) == 0:
            return {
                'label': 'neutral',
                'score': 0,
                'confidence': 0.0,
                'all_scores': {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
            }
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get logits and probabilities
        logits = outputs.logits[0].cpu().numpy()
        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=0).numpy()
        
        # Get predicted label (argmax)
        predicted_label_idx = np.argmax(logits)
        predicted_label = self.label_map[predicted_label_idx]
        predicted_score = self.score_map[predicted_label_idx]
        confidence = float(probabilities[predicted_label_idx])
        
        return {
            'label': predicted_label,
            'score': predicted_score,
            'confidence': confidence,
            'all_scores': {
                'negative': float(probabilities[0]),
                'neutral': float(probabilities[1]),
                'positive': float(probabilities[2])
            }
        }
    
    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'clean_text',
        batch_size: int = 8
    ) -> pd.DataFrame:
        """
        Apply sentiment analysis to DataFrame column.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column to analyze
            batch_size: Batch processing size for efficiency
            
        Returns:
            DataFrame with added sentiment columns:
            - finbert_label: "positive", "neutral", "negative"
            - finbert_score: -1, 0, +1
            - finbert_confidence: 0-1 probability
            - finbert_positive_prob: Probability of positive (0-1)
            - finbert_neutral_prob: Probability of neutral (0-1)
            - finbert_negative_prob: Probability of negative (0-1)
        """
        logger.info(f"Analyzing {len(df)} texts with FinBERT (batch_size={batch_size})")
        
        results = {
            'label': [],
            'score': [],
            'confidence': [],
            'pos_prob': [],
            'neu_prob': [],
            'neg_prob': []
        }
        
        # Process in batches for efficiency
        for idx in tqdm(range(0, len(df), batch_size), desc="FinBERT Sentiment"):
            batch_texts = df[text_column].iloc[idx:idx+batch_size].fillna('').tolist()
            
            for text in batch_texts:
                result = self.analyze(text)
                results['label'].append(result['label'])
                results['score'].append(result['score'])
                results['confidence'].append(result['confidence'])
                results['pos_prob'].append(result['all_scores']['positive'])
                results['neu_prob'].append(result['all_scores']['neutral'])
                results['neg_prob'].append(result['all_scores']['negative'])
        
        # Add to dataframe
        df['finbert_label'] = results['label']
        df['finbert_score'] = results['score']
        df['finbert_confidence'] = results['confidence']
        df['finbert_positive_prob'] = results['pos_prob']
        df['finbert_neutral_prob'] = results['neu_prob']
        df['finbert_negative_prob'] = results['neg_prob']
        
        logger.info(f"Sentiment analysis complete")
        logger.info(f"Label distribution:")
        logger.info(f"  Positive: {(df['finbert_label'] == 'positive').sum()} "
                   f"({100*(df['finbert_label'] == 'positive').sum()/len(df):.1f}%)")
        logger.info(f"  Neutral:  {(df['finbert_label'] == 'neutral').sum()} "
                   f"({100*(df['finbert_label'] == 'neutral').sum()/len(df):.1f}%)")
        logger.info(f"  Negative: {(df['finbert_label'] == 'negative').sum()} "
                   f"({100*(df['finbert_label'] == 'negative').sum()/len(df):.1f}%)")
        
        return df
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for sentiment analysis.
        
        Args:
            df: DataFrame with finbert_label and finbert_score columns
            
        Returns:
            Dictionary with statistics
        """
        if 'finbert_score' not in df.columns:
            logger.warning("No finbert_score column found")
            return {}
        
        stats = {
            'total_articles': len(df),
            'positive_count': (df['finbert_label'] == 'positive').sum(),
            'neutral_count': (df['finbert_label'] == 'neutral').sum(),
            'negative_count': (df['finbert_label'] == 'negative').sum(),
            'positive_pct': 100 * (df['finbert_label'] == 'positive').sum() / len(df),
            'neutral_pct': 100 * (df['finbert_label'] == 'neutral').sum() / len(df),
            'negative_pct': 100 * (df['finbert_label'] == 'negative').sum() / len(df),
            'avg_score': df['finbert_score'].mean(),
            'median_score': df['finbert_score'].median(),
            'std_score': df['finbert_score'].std(),
            'min_score': df['finbert_score'].min(),
            'max_score': df['finbert_score'].max(),
            'avg_confidence': df['finbert_confidence'].mean(),
        }
        
        return stats
