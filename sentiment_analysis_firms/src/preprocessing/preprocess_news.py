"""
Preprocessing pipeline for news articles
Run this script to clean news data and prepare for sentiment analysis
"""

import sys
import logging
from pathlib import Path
import pandas as pd

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.text_preprocessor import TextPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def preprocess_news_articles(
    input_path: str = "data/raw/news_data.csv",
    output_path: str = "data/processed/news_data_cleaned.csv",
    min_text_length: int = 50,
    lemmatize: bool = False,
    sample_output: bool = True
):
    """
    Preprocess news articles for sentiment analysis.
    
    Args:
        input_path (str): Path to raw news CSV
        output_path (str): Path to save cleaned CSV
        min_text_length (int): Minimum valid text length (default: 50 chars)
        lemmatize (bool): Apply lemmatization (default: False for speed)
        sample_output (bool): Display sample results (default: True)
    """
    
    logger.info("=" * 70)
    logger.info("NEWS ARTICLE PREPROCESSING")
    logger.info("=" * 70)
    
    # Check if input file exists
    if not Path(input_path).exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Tip: Run collect_news.py first to generate raw data")
        return None
    
    # Load data
    logger.info(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} articles")
    
    # Initialize preprocessor
    logger.info(f"\nInitializing preprocessor...")
    logger.info(f"  - Minimum text length: {min_text_length} characters")
    logger.info(f"  - Lemmatization: {'Enabled' if lemmatize else 'Disabled'}")
    logger.info(f"  - Remove URLs: True")
    logger.info(f"  - Remove emails: True")
    
    preprocessor = TextPreprocessor(
        min_length=min_text_length,
        remove_urls=True,
        remove_emails=True,
        lemmatize=lemmatize
    )
    
    # Preprocess DataFrame
    logger.info(f"\nPreprocessing {len(df)} articles...")
    df = preprocessor.preprocess_dataframe(
        df,
        text_columns=['title', 'description'],
        output_column='clean_text',
        combine_sources=True
    )
    
    # Get statistics
    logger.info("\n" + "=" * 70)
    logger.info("PREPROCESSING STATISTICS")
    logger.info("=" * 70)
    
    stats = preprocessor.get_statistics(df, 'clean_text')
    
    logger.info(f"Input articles: {stats['total_articles']}")
    logger.info(f"Valid articles: {stats['valid_articles']}")
    logger.info(f"Removed (too short): {stats['removed_articles']}")
    logger.info(f"Removal rate: {100 * stats['removed_articles'] / stats['total_articles']:.1f}%")
    
    logger.info(f"\nText Statistics:")
    logger.info(f"  Total words: {stats['total_words']:,}")
    logger.info(f"  Unique words: {stats['unique_words']:,}")
    logger.info(f"  Avg words/article: {stats['avg_words_per_article']:.1f}")
    logger.info(f"  Min words: {stats['min_words']}")
    logger.info(f"  Max words: {stats['max_words']}")
    logger.info(f"  Avg chars/article: {stats['avg_chars']:.1f}")
    
    # Save preprocessed data
    logger.info(f"\nSaving preprocessed data...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Saved to: {output_path}")
    
    # Display samples
    if sample_output:
        logger.info("\n" + "=" * 70)
        logger.info("SAMPLE PREPROCESSING RESULTS")
        logger.info("=" * 70)
        
        df_valid = df[df['is_valid_length'] == True]
        sample_size = min(5, len(df_valid))
        
        for i, (idx, row) in enumerate(df_valid.head(sample_size).iterrows(), 1):
            logger.info(f"\n{i}. ARTICLE")
            logger.info(f"   Original title: {row['title'][:70]}...")
            if pd.notna(row['description']):
                logger.info(f"   Original desc:  {row['description'][:70]}...")
            logger.info(f"   → Clean text:   {row['clean_text'][:70]}...")
            
            # Metadata
            if pd.notna(row.get('firm_name')):
                logger.info(f"   Firm: {row['firm_name']}, Keyword: {row['keyword_used']}")
            if pd.notna(row.get('source_name')):
                logger.info(f"   Source: {row['source_name']}")
    
    # Show breakdown by valid/invalid
    logger.info("\n" + "=" * 70)
    logger.info("VALIDITY BREAKDOWN")
    logger.info("=" * 70)
    
    valid_count = len(df[df['is_valid_length'] == True])
    invalid_count = len(df[df['is_valid_length'] == False])
    
    logger.info(f"Valid length (≥{min_text_length} chars): {valid_count}")
    logger.info(f"Too short (<{min_text_length} chars): {invalid_count}")
    
    # Optional: Show examples of removed texts
    if invalid_count > 0 and invalid_count <= 10:
        logger.info("\nRemoved texts (too short):")
        for idx, row in df[df['is_valid_length'] == False].head(5).iterrows():
            combined = str(row['title']) + " " + str(row['description']) if pd.notna(row['description']) else str(row['title'])
            logger.info(f"  - `{combined[:50]}...` ({len(combined)} chars)")
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ PREPROCESSING COMPLETE")
    logger.info("=" * 70)
    
    # Return summary
    summary = {
        'input_file': input_path,
        'output_file': output_path,
        'total_articles': stats['total_articles'],
        'valid_articles': stats['valid_articles'],
        'removed_articles': stats['removed_articles'],
        'total_words': stats['total_words'],
        'unique_words': stats['unique_words']
    }
    
    return df, summary


def main():
    """Main execution function"""
    
    try:
        # Preprocess news data
        df_clean, summary = preprocess_news_articles(
            input_path="data/raw/news_data.csv",
            output_path="data/processed/news_data_cleaned.csv",
            min_text_length=50,
            lemmatize=False,
            sample_output=True
        )
        
        # Additional analyses (optional)
        if df_clean is not None:
            logger.info("\nNext steps:")
            logger.info("  1. Run sentiment analysis on clean_text column")
            logger.info("  2. Aggregate sentiment by firm and keyword")
            logger.info("  3. Create time series sentiment indices")
            logger.info("  4. Export results for visualization")
    
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
