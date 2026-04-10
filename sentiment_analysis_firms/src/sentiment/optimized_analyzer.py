"""
Optimized FinBERT Sentiment Analyzer with Batch Processing
Uses transformers.pipeline for efficient batch processing and GPU acceleration
"""

import torch
import pandas as pd
import numpy as np
from transformers import pipeline
import logging
from typing import Dict, List
from tqdm import tqdm

logger = logging.getLogger(__name__)


class OptimizedFinBERTAnalyzer:
    """
    Optimized FinBERT sentiment analysis with batch processing.
    
    Key optimizations:
    - Uses transformers.pipeline (handles device placement, batching)
    - Batch processing via pipeline built-in capabilities
    - GPU acceleration support (auto-detects CUDA)
    - Minimal data copying between CPU/GPU
    - Progress tracking for large datasets
    
    Output mapping:
    - positive → sentiment_label="positive", sentiment_score=+1
    - neutral → sentiment_label="neutral", sentiment_score=0
    - negative → sentiment_label="negative", sentiment_score=-1
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert", device: int = None, batch_size: int = 32):
        """
        Initialize optimized FinBERT pipeline.
        
        Args:
            model_name: HuggingFace model identifier
            device: GPU device ID (-1 for CPU, 0+ for GPU), auto-detect if None
            batch_size: Batch size for processing
        """
        logger.info(f"Initializing OptimizedFinBERT (batch_size={batch_size})")
        
        # Auto-detect device
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        
        self.device = device
        self.batch_size = batch_size
        device_str = f"GPU {device}" if device >= 0 else "CPU"
        
        # Initialize pipeline (handles tokenization, model loading, batching)
        logger.info(f"Loading model on {device_str}...")
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device,
            batch_size=batch_size,
            truncation=True,
            max_length=512
        )
        
        logger.info(f"✓ Model loaded on {device_str}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Max length: 512 tokens")
    
    def _score_to_label_score(self, pipeline_result: Dict) -> tuple:
        """
        Map FinBERT pipeline output to (label, score).
        
        FinBERT model output:
        - "POSITIVE" → sentiment_score = +1
        - "NEUTRAL" → sentiment_score = 0
        - "NEGATIVE" → sentiment_score = -1
        
        Args:
            pipeline_result: Dict from transformers pipeline
            
        Returns:
            Tuple of (label, score) where label is lowercase
        """
        label_lower = pipeline_result['label'].lower()
        
        score_map = {
            'positive': 1,
            'neutral': 0,
            'negative': -1
        }
        
        score = score_map.get(label_lower, 0)
        return label_lower, score
    
    def analyze_texts_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze batch of texts efficiently.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dicts with keys: label, score, score_confidence
        """
        if not texts:
            return []
        
        # Remove None/empty
        texts = [str(t).strip() if t else "" for t in texts]
        
        # Pipeline processes batch efficiently
        results_raw = self.pipeline(texts)
        
        results = []
        for result in results_raw:
            label, score = self._score_to_label_score(result)
            results.append({
                'label': label,
                'score': score,
                'score_confidence': result['score']
            })
        
        return results
    
    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'clean_text',
        batch_size: int = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Analyze all texts in DataFrame column with optimized batching.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column to analyze
            batch_size: Override batch_size (uses instance default if None)
            show_progress: Show progress bar
            
        Returns:
            DataFrame with added columns:
            - sentiment_label: "positive", "neutral", "negative"
            - sentiment_score: -1, 0, +1
            - sentiment_confidence: Model confidence (0-1)
        """
        batch_size = batch_size or self.batch_size
        
        logger.info(f"Analyzing {len(df)} texts")
        logger.info(f"  Column: {text_column}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Total batches: {(len(df) + batch_size - 1) // batch_size}")
        
        all_labels = []
        all_scores = []
        all_confidences = []
        
        # Create batches
        num_batches = (len(df) + batch_size - 1) // batch_size
        iterator = tqdm(range(num_batches), desc="Sentiment Analysis", disable=not show_progress)
        
        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            
            batch_texts = df[text_column].iloc[start_idx:end_idx].tolist()
            
            # Analyze batch
            batch_results = self.analyze_texts_batch(batch_texts)
            
            # Extract results
            for result in batch_results:
                all_labels.append(result['label'])
                all_scores.append(result['score'])
                all_confidences.append(result['score_confidence'])
        
        # Add to dataframe
        df = df.copy()
        df['sentiment_label'] = all_labels
        df['sentiment_score'] = all_scores
        df['sentiment_confidence'] = all_confidences
        
        # Log statistics
        logger.info("✓ Analysis complete")
        label_counts = pd.Series(all_labels).value_counts()
        for label in ['positive', 'neutral', 'negative']:
            count = label_counts.get(label, 0)
            pct = 100 * count / len(all_labels)
            logger.info(f"  {label.capitalize():<10}: {count:>4} ({pct:>5.1f}%)")
        
        return df
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics.
        
        Args:
            df: DataFrame with sentiment_label and sentiment_score columns
            
        Returns:
            Dictionary with statistics
        """
        if 'sentiment_score' not in df.columns:
            logger.warning("No sentiment_score column found")
            return {}
        
        labels = df['sentiment_label']
        scores = df['sentiment_score']
        
        stats = {
            'total_articles': len(df),
            'positive_count': (labels == 'positive').sum(),
            'neutral_count': (labels == 'neutral').sum(),
            'negative_count': (labels == 'negative').sum(),
            'positive_pct': 100 * (labels == 'positive').sum() / len(df),
            'neutral_pct': 100 * (labels == 'neutral').sum() / len(df),
            'negative_pct': 100 * (labels == 'negative').sum() / len(df),
            'mean_score': scores.mean(),
            'median_score': scores.median(),
            'std_score': scores.std(),
            'min_score': scores.min(),
            'max_score': scores.max(),
            'mean_confidence': df['sentiment_confidence'].mean() if 'sentiment_confidence' in df.columns else np.nan,
        }
        
        return stats


class VADEROptimized:
    """
    Vectorized VADER sentiment analysis for simple, fast baseline.
    Much faster than FinBERT but lower accuracy on financial text.
    """
    
    def __init__(self):
        """Initialize VADER analyzer (no model download needed)."""
        from nltk.sentiment import SentimentIntensityAnalyzer
        import nltk
        
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            logger.info("Downloading VADER lexicon...")
            nltk.download('vader_lexicon', quiet=True)
        
        self.vader = SentimentIntensityAnalyzer()
        logger.info("✓ VADER initialized (lexicon-based, no GPU needed)")
    
    def analyze_texts(self, texts: List[str]) -> List[Dict]:
        """
        Analyze list of texts with VADER.
        
        Args:
            texts: List of texts
            
        Returns:
            List of dicts with label, score
        """
        results = []
        for text in texts:
            if not text or len(str(text).strip()) == 0:
                results.append({
                    'label': 'neutral',
                    'score': 0,
                    'score_confidence': 0.0
                })
            else:
                scores = self.vader.polarity_scores(str(text))
                compound = scores['compound']
                
                # Map to -1, 0, +1
                if compound >= 0.05:
                    label = 'positive'
                    score = 1
                elif compound <= -0.05:
                    label = 'negative'
                    score = -1
                else:
                    label = 'neutral'
                    score = 0
                
                results.append({
                    'label': label,
                    'score': score,
                    'score_confidence': abs(compound)
                })
        
        return results
    
    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'clean_text',
        batch_size: int = 1000,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Analyze DataFrame with VADER (vectorized).
        
        Args:
            df: Input DataFrame
            text_column: Text column name
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            DataFrame with sentiment_label, sentiment_score columns
        """
        logger.info(f"Analyzing {len(df)} texts with VADER")
        logger.info(f"  Batch size: {batch_size}")
        
        all_labels = []
        all_scores = []
        all_confidences = []
        
        num_batches = (len(df) + batch_size - 1) // batch_size
        iterator = tqdm(range(num_batches), desc="VADER Analysis", disable=not show_progress)
        
        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            
            batch_texts = df[text_column].iloc[start_idx:end_idx].tolist()
            batch_results = self.analyze_texts(batch_texts)
            
            for result in batch_results:
                all_labels.append(result['label'])
                all_scores.append(result['score'])
                all_confidences.append(result['score_confidence'])
        
        df = df.copy()
        df['sentiment_label'] = all_labels
        df['sentiment_score'] = all_scores
        df['sentiment_confidence'] = all_confidences
        
        logger.info("✓ VADER analysis complete")
        label_counts = pd.Series(all_labels).value_counts()
        for label in ['positive', 'neutral', 'negative']:
            count = label_counts.get(label, 0)
            pct = 100 * count / len(all_labels)
            logger.info(f"  {label.capitalize():<10}: {count:>4} ({pct:>5.1f}%)")
        
        return df
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics."""
        if 'sentiment_score' not in df.columns:
            return {}
        
        labels = df['sentiment_label']
        scores = df['sentiment_score']
        
        return {
            'total_articles': len(df),
            'positive_count': (labels == 'positive').sum(),
            'neutral_count': (labels == 'neutral').sum(),
            'negative_count': (labels == 'negative').sum(),
            'positive_pct': 100 * (labels == 'positive').sum() / len(df),
            'neutral_pct': 100 * (labels == 'neutral').sum() / len(df),
            'negative_pct': 100 * (labels == 'negative').sum() / len(df),
            'mean_score': scores.mean(),
            'median_score': scores.median(),
            'std_score': scores.std(),
            'mean_confidence': df['sentiment_confidence'].mean() if 'sentiment_confidence' in df.columns else np.nan,
        }
