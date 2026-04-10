"""
Text preprocessing utilities for sentiment analysis
Handles text cleaning, normalization, and feature extraction
"""

import re
import logging
from typing import List, Dict
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK 'stopwords'...")
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading NLTK 'wordnet'...")
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    logger.info("Downloading NLTK 'omw-1.4'...")
    nltk.download('omw-1.4')


class TextPreprocessor:
    """Handles text preprocessing for sentiment analysis"""
    
    def __init__(
        self,
        min_length: int = 50,
        remove_urls: bool = True,
        remove_emails: bool = True,
        lemmatize: bool = False
    ):
        """
        Initialize the text preprocessor.
        
        Args:
            min_length (int): Minimum text length to keep (default: 50 chars)
            remove_urls (bool): Remove URLs from text (default: True)
            remove_emails (bool): Remove email addresses (default: True)
            lemmatize (bool): Apply lemmatization (default: False, for speed)
        """
        self.min_length = min_length
        self.should_remove_urls = remove_urls
        self.should_remove_emails = remove_emails
        self.lemmatize_enabled = lemmatize
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        
        logger.info(
            f"Initialized TextPreprocessor (min_length={min_length}, "
            f"lemmatize={lemmatize})"
        )
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        # Match http/https/ftp URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Match www URLs
        text = re.sub(r'www\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,}', '', text)
        return text
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses from text"""
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
        return text
    
    def lowercase(self, text: str) -> str:
        """Convert text to lowercase"""
        return text.lower()
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text"""
        # Keep only alphanumeric characters, spaces, and hyphens
        text = re.sub(r'[^\w\s\-]', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def remove_stopwords(self, text: str) -> str:
        """Remove English stopwords"""
        words = text.split()
        filtered_words = [w for w in words if w not in self.stop_words]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text: str) -> str:
        """Apply lemmatization to text"""
        if not self.lemmatize_enabled or not self.lemmatizer:
            return text
        
        try:
            words = text.split()
            lemmatized = [self.lemmatizer.lemmatize(w) for w in words]
            return ' '.join(lemmatized)
        except Exception as e:
            logger.warning(f"Lemmatization error: {str(e)}")
            return text
    
    def check_length(self, text: str) -> bool:
        """Check if text meets minimum length requirement"""
        return len(text) >= self.min_length
    
    def preprocess(self, text: str) -> tuple:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            tuple: (cleaned_text, valid) where valid indicates if text met length requirement
        """
        if not isinstance(text, str) or not text.strip():
            return "", False
        
        # Remove URLs and emails
        if self.should_remove_urls:
            text = self.remove_urls(text)
        
        if self.should_remove_emails:
            text = self.remove_emails(text)
        
        # Convert to lowercase
        text = self.lowercase(text)
        
        # Remove punctuation
        text = self.remove_punctuation(text)
        
        # Remove stopwords
        text = self.remove_stopwords(text)
        
        # Apply lemmatization if enabled
        if self.lemmatize_enabled:
            text = self.lemmatize_text(text)
        
        # Check minimum length
        valid = self.check_length(text)
        
        return text, valid
    
    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_columns: List[str] = None,
        output_column: str = 'clean_text',
        combine_sources: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess text columns in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_columns (list): Columns to preprocess (default: ['title', 'description'])
            output_column (str): Name of output column (default: 'clean_text')
            combine_sources (bool): Combine multiple text sources (default: True)
            
        Returns:
            pd.DataFrame: DataFrame with added clean_text column
        """
        if text_columns is None:
            text_columns = ['title', 'description']
        
        # Verify columns exist
        available_cols = [col for col in text_columns if col in df.columns]
        if not available_cols:
            logger.warning("No text columns found in DataFrame")
            return df
        
        logger.info(f"Preprocessing {len(df)} articles with columns: {available_cols}")
        
        df = df.copy()
        cleaned_texts = []
        valid_entries = []
        removed_count = 0
        
        for idx, row in df.iterrows():
            # Combine text from multiple sources if enabled
            if combine_sources:
                combined_text = " ".join(
                    [str(row[col]) for col in available_cols if pd.notna(row[col])]
                )
            else:
                # Use first available column
                combined_text = ""
                for col in available_cols:
                    if pd.notna(row[col]):
                        combined_text = str(row[col])
                        break
            
            # Preprocess
            clean_text, valid = self.preprocess(combined_text)
            
            cleaned_texts.append(clean_text)
            valid_entries.append(valid)
            
            if not valid:
                removed_count += 1
            
            # Log progress every 100 articles
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} articles")
        
        df[output_column] = cleaned_texts
        df['is_valid_length'] = valid_entries
        
        logger.info(
            f"Preprocessing complete: {len(df) - removed_count} valid, "
            f"{removed_count} too short (<{self.min_length} chars)"
        )
        
        return df
    
    def get_statistics(self, df: pd.DataFrame, text_column: str = 'clean_text') -> Dict:
        """
        Get statistics about preprocessed text.
        
        Args:
            df (pd.DataFrame): DataFrame with preprocessed text
            text_column (str): Column containing preprocessed text
            
        Returns:
            dict: Statistics dictionary
        """
        df_valid = df[df['is_valid_length'] == True] if 'is_valid_length' in df.columns else df
        
        if text_column not in df.columns:
            logger.error(f"Column '{text_column}' not found")
            return {}
        
        texts = df_valid[text_column].fillna("")
        
        # Calculate word counts
        word_counts = [len(t.split()) for t in texts]
        char_counts = [len(t) for t in texts]
        
        stats = {
            'total_articles': len(df),
            'valid_articles': len(df_valid),
            'removed_articles': len(df) - len(df_valid),
            'total_words': sum(word_counts),
            'avg_words_per_article': sum(word_counts) / len(word_counts) if word_counts else 0,
            'min_words': min(word_counts) if word_counts else 0,
            'max_words': max(word_counts) if word_counts else 0,
            'avg_chars': sum(char_counts) / len(char_counts) if char_counts else 0,
            'total_chars': sum(char_counts),
            'unique_words': len(set(' '.join(texts).split()))
        }
        
        return stats


def preprocess_news_data(
    input_file: str,
    output_file: str = None,
    min_length: int = 50,
    lemmatize: bool = False
) -> pd.DataFrame:
    """
    Preprocess news data from CSV file.
    
    Args:
        input_file (str): Path to input CSV
        output_file (str): Path to save output CSV (optional)
        min_length (int): Minimum text length (default: 50)
        lemmatize (bool): Apply lemmatization (default: False)
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} articles")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        min_length=min_length,
        lemmatize=lemmatize
    )
    
    # Preprocess
    df = preprocessor.preprocess_dataframe(
        df,
        text_columns=['title', 'description'],
        output_column='clean_text',
        combine_sources=True
    )
    
    # Get statistics
    stats = preprocessor.get_statistics(df, 'clean_text')
    
    logger.info("Preprocessing Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Save if output path provided
    if output_file:
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Saved preprocessed data to {output_file}")
    
    return df


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/news_data_cleaned.csv"
        
        df_cleaned = preprocess_news_data(
            input_file=input_path,
            output_file=output_path,
            min_length=50,
            lemmatize=False
        )
        
        # Display sample
        print("\nSample cleaned text:")
        print("=" * 60)
        for idx, row in df_cleaned[df_cleaned['is_valid_length'] == True].head(5).iterrows():
            print(f"\nOriginal: {row['title'][:70]}...")
            print(f"Cleaned:  {row['clean_text'][:70]}...")
    else:
        print("Usage: python text_preprocessor.py <input_csv> [output_csv]")
        print("\nExample:")
        print("  python text_preprocessor.py data/raw/news_data.csv data/processed/news_data_cleaned.csv")
