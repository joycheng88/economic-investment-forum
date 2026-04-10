"""
Full Data Pipeline: Collection → Preprocessing → Filtering → Sentiment → Weekly Indices
Runs complete pipeline with REAL data collection
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add parent directory to Python path to access src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure output directories exist
os.makedirs('outputs', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/full_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def step1_collect_news():
    """STEP 1: Collect real news data"""
    print("\n" + "=" * 80)
    print("STEP 1/5: COLLECTING REAL NEWS DATA (live gnews API)")
    print("=" * 80)
    
    try:
        from src.data_collection.news_collector import GLP1NewsCollector
        from src.firm_names import SNACK_COMPANIES
        from src.glp1_keywords import (
            GLP1_MEDICATIONS,
            GLP1_MEDICAL_TERMS,
            GLP1_MARKET_TERMS,
            GLP1_HEALTH_TOPICS
        )
        
        logger.info("Initializing GLP1NewsCollector for live data collection...")
        
        collector = GLP1NewsCollector(
            max_results=50,  # Modest size to respect API limits
            language="en",
            country=None,
            max_retries=3,
            retry_delay=2
        )
        
        firms = SNACK_COMPANIES
        keywords = (
            GLP1_MEDICATIONS[:6] +
            GLP1_MEDICAL_TERMS[:4] +
            GLP1_MARKET_TERMS[:4] +
            ["weight loss", "GLP-1", "obesity"]
        )
        
        custom_queries = [
            "GLP-1 market",
            "Ozempic Wegovy competition",
            "snack company obesity medicine",
            "weight loss medication market"
        ]
        
        logger.info(f"Collecting from {len(firms)} firms × {len(keywords)} keywords")
        logger.info(f"Plus {len(custom_queries)} custom queries")
        
        # Collect articles
        df_raw = collector.collect_articles(
            firms=firms,
            keywords=keywords,
            custom_queries=custom_queries,
            rate_limit_delay=1.5
        )
        
        if df_raw is None or len(df_raw) == 0:
            logger.error("No articles collected")
            return None
        
        logger.info(f"Raw collection: {len(df_raw)} articles")
        
        # Deduplicate
        df_raw = collector.deduplicatae_articles(df_raw)
        logger.info(f"After deduplication: {len(df_raw)} unique articles")
        
        # Filter to recent
        df_raw = collector.filter_by_date(df_raw, days_back=30)
        logger.info(f"After date filter (30 days): {len(df_raw)} articles")
        
        # Save raw data
        os.makedirs('data/raw', exist_ok=True)
        output_path = 'data/raw/news_data_real.csv'
        df_raw.to_csv(output_path, index=False)
        logger.info(f"✓ Saved to: {output_path}")
        
        print(f"\n✓ Collected {len(df_raw)} REAL articles from live news APIs")
        print(f"  Output: {output_path}")
        return df_raw
        
    except Exception as e:
        logger.error(f"Collection error: {e}", exc_info=True)
        print(f"✗ Collection failed: {e}")
        return None


def step2_preprocess():
    """STEP 2: Preprocess collected data"""
    print("\n" + "=" * 80)
    print("STEP 2/5: PREPROCESSING TEXT")
    print("=" * 80)
    
    try:
        from src.preprocessing.text_preprocessor import TextPreprocessor
        
        # Use real data if available, otherwise use demo
        input_path = 'data/raw/news_data_real.csv'
        if not os.path.exists(input_path):
            logger.warning("Real data not found, using existing cleaned data")
            if os.path.exists('data/processed/news_data_cleaned.csv'):
                df = pd.read_csv('data/processed/news_data_cleaned.csv')
                logger.info(f"Loaded existing cleaned data: {len(df)} articles")
                return df
            else:
                logger.error("No data available")
                return None
        
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} articles for preprocessing")
        
        preprocessor = TextPreprocessor(
            min_length=50,
            remove_urls=True,
            remove_emails=True,
            lemmatize=False
        )
        
        df_clean = preprocessor.preprocess_dataframe(
            df,
            text_columns=['title', 'description'],
            output_column='clean_text'
        )
        
        # Filter to valid articles
        df_valid = df_clean[df_clean['is_valid_length'] == True].copy()
        logger.info(f"Valid articles: {len(df_valid)}/{len(df_clean)}")
        
        # Save cleaned data
        os.makedirs('data/processed', exist_ok=True)
        output_path = 'data/processed/news_data_cleaned_real.csv'
        df_valid.to_csv(output_path, index=False)
        logger.info(f"✓ Saved to: {output_path}")
        
        print(f"\n✓ Preprocessed {len(df_valid)} valid articles")
        print(f"  Output: {output_path}")
        return df_valid
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}", exc_info=True)
        print(f"✗ Preprocessing failed: {e}")
        return None


def step3_filter():
    """STEP 3: Filter for GLP-1 relevance"""
    print("\n" + "=" * 80)
    print("STEP 3/5: FILTERING FOR GLP-1 RELEVANCE")
    print("=" * 80)
    
    try:
        from filter_glp1_relevant import filter_glp1_relevant
        
        input_path = 'data/processed/news_data_cleaned_real.csv'
        
        # Use real if available, otherwise use cleaned (demo)
        if not os.path.exists(input_path):
            logger.warning("Real cleaned data not found, using existing")
            if not os.path.exists('data/processed/news_data_cleaned.csv'):
                logger.error("No data to filter")
                return None
            input_path = 'data/processed/news_data_cleaned.csv'
        
        output_path = 'data/processed/glp1_relevant_real.csv'
        
        df_filtered, stats = filter_glp1_relevant(input_path, output_path)
        
        if df_filtered is None:
            logger.error("Filtering failed")
            return None
        
        logger.info(f"Filtered to {len(df_filtered)} relevant articles")
        
        print(f"\n✓ Filtered to {len(df_filtered)} articles with BOTH firm + GLP-1")
        print(f"  Retention: {stats['kept_articles']}/{stats['total_articles']} ({100*stats['kept_articles']/stats['total_articles']:.1f}%)")
        print(f"  Output: {output_path}")
        return df_filtered
        
    except Exception as e:
        logger.error(f"Filtering error: {e}", exc_info=True)
        print(f"✗ Filtering failed: {e}")
        return None


def step4_sentiment():
    """STEP 4: Analyze sentiment with VADER"""
    print("\n" + "=" * 80)
    print("STEP 4/5: SENTIMENT ANALYSIS (VADER)")
    print("=" * 80)
    
    try:
        from src.sentiment.vader_analyzer import VADERSentimentAnalyzer
        
        # Load filtered data (try real first, then demo)
        input_paths = [
            'data/processed/glp1_relevant_real.csv',
            'data/processed/glp1_relevant.csv'
        ]
        
        df_filtered = None
        for path in input_paths:
            if os.path.exists(path):
                df_filtered = pd.read_csv(path)
                logger.info(f"Loaded {len(df_filtered)} articles from {path}")
                break
        
        if df_filtered is None:
            logger.error("No filtered data found")
            return None
        
        analyzer = VADERSentimentAnalyzer()
        df_sentiment = analyzer.analyze_dataframe(df_filtered, text_column='clean_text')
        
        output_path = 'data/processed/articles_with_sentiment_real.csv'
        df_sentiment.to_csv(output_path, index=False)
        logger.info(f"✓ Saved to: {output_path}")
        
        # Get summary stats
        stats = analyzer.get_summary_stats(df_sentiment)
        
        print(f"\n✓ Sentiment analysis complete")
        print(f"  Positive: {stats['positive_count']} ({stats['positive_pct']:.1f}%)")
        print(f"  Negative: {stats['negative_count']} ({stats['negative_pct']:.1f}%)")
        print(f"  Neutral: {stats['neutral_count']} ({stats['neutral_pct']:.1f}%)")
        print(f"  Avg compound: {stats['avg_compound_score']:.3f}")
        print(f"  Output: {output_path}")
        
        return df_sentiment
        
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}", exc_info=True)
        print(f"✗ Sentiment analysis failed: {e}")
        return None


def step5_weekly_indices(df_sentiment):
    """STEP 5: Create weekly sentiment indices"""
    print("\n" + "=" * 80)
    print("STEP 5/5: CREATING WEEKLY SENTIMENT INDICES")
    print("=" * 80)
    
    try:
        from src.aggregation.weekly_sentiment_index import WeeklySentimentIndex
        
        if df_sentiment is None or len(df_sentiment) == 0:
            logger.error("No sentiment data provided")
            return None
        
        indexer = WeeklySentimentIndex()
        
        # Overall market index
        print("\n  Creating market-level weekly index...")
        weekly_index = indexer.create_weekly_index(df_sentiment)
        
        market_output = 'outputs/weekly_market_sentiment_index_real.csv'
        weekly_index.to_csv(market_output, index=False)
        logger.info(f"✓ Market index: {market_output}")
        
        print(f"    ✓ {len(weekly_index)} weeks of market sentiment")
        
        # Firm-level indices
        print("\n  Creating firm-level weekly indices...")
        firm_weekly_index = indexer.create_firm_weekly_index(df_sentiment)
        
        firm_output = 'outputs/weekly_firm_sentiment_indices_real.csv'
        firm_weekly_index.to_csv(firm_output, index=False)
        logger.info(f"✓ Firm index: {firm_output}")
        
        num_firms = firm_weekly_index['firm'].nunique()
        print(f"    ✓ {len(firm_weekly_index)} firm-week combinations ({num_firms} firms)")
        
        # Keyword-level indices
        print("\n  Creating keyword-level weekly indices...")
        keyword_weekly_index = indexer.create_keyword_weekly_index(df_sentiment)
        
        keyword_output = 'outputs/weekly_keyword_sentiment_indices_real.csv'
        keyword_weekly_index.to_csv(keyword_output, index=False)
        logger.info(f"✓ Keyword index: {keyword_output}")
        
        num_keywords = keyword_weekly_index['keyword'].nunique()
        print(f"    ✓ {len(keyword_weekly_index)} keyword-week combinations ({num_keywords} keywords)")
        
        # Display summaries
        print("\n" + "-" * 80)
        print("MARKET SENTIMENT (WEEKLY)")
        print("-" * 80)
        cols_display = ['week_start', 'num_articles', 'avg_compound_sentiment', 'positive_pct', 'negative_pct']
        print(weekly_index[cols_display].to_string(index=False))
        
        print("\n" + "-" * 80)
        print("FIRM SENTIMENT (WEEKLY - Sample)")
        print("-" * 80)
        firm_cols = ['firm', 'week_start', 'num_articles', 'avg_compound_sentiment', 'positive_pct']
        print(firm_weekly_index[firm_cols].head(15).to_string(index=False))
        
        print("\n" + "-" * 80)
        print("KEYWORD SENTIMENT (WEEKLY - Sample)")
        print("-" * 80)
        kw_cols = ['keyword', 'week_start', 'num_articles', 'avg_compound_sentiment', 'positive_pct']
        print(keyword_weekly_index[kw_cols].head(10).to_string(index=False))
        
        print("\n✓ Weekly indices created:")
        print(f"  Market: {market_output}")
        print(f"  Firms: {firm_output}")
        print(f"  Keywords: {keyword_output}")
        
        return {
            'weekly_market': weekly_index,
            'weekly_firms': firm_weekly_index,
            'weekly_keywords': keyword_weekly_index
        }
        
    except Exception as e:
        logger.error(f"Weekly index error: {e}", exc_info=True)
        print(f"✗ Weekly index creation failed: {e}")
        return None


def main():
    """Execute full pipeline"""
    
    print("\n" + "=" * 80)
    print("FULL DATA PIPELINE: REAL DATA COLLECTION → WEEKLY SENTIMENT INDICES")
    print("=" * 80)
    print("\nThis pipeline will:")
    print("  1. Collect REAL news data from live gnews API")
    print("  2. Preprocess and clean articles")
    print("  3. Filter for GLP-1 relevance")
    print("  4. Analyze sentiment with VADER")
    print("  5. Create WEEKLY sentiment indices")
    print("\n" + "=" * 80)
    
    # Step 1: Collect
    df_raw = step1_collect_news()
    if df_raw is None:
        print("\n✗ Pipeline interrupted at collection step")
        return
    
    # Step 2: Preprocess
    df_cleaned = step2_preprocess()
    if df_cleaned is None:
        print("\n✗ Pipeline interrupted at preprocessing step")
        return
    
    # Step 3: Filter
    df_filtered = step3_filter()
    if df_filtered is None:
        print("\n✗ Pipeline interrupted at filtering step")
        return
    
    # Step 4: Sentiment
    df_sentiment = step4_sentiment()
    if df_sentiment is None:
        print("\n✗ Pipeline interrupted at sentiment step")
        return
    
    # Step 5: Weekly indices
    indices = step5_weekly_indices(df_sentiment)
    if indices is None:
        print("\n✗ Pipeline interrupted at weekly indexing step")
        return
    
    # Final summary
    print("\n" + "=" * 80)
    print("✓✓✓ FULL PIPELINE COMPLETE ✓✓✓")
    print("=" * 80)
    
    print("\nData Files Generated:")
    print(f"  Raw collection:     data/raw/news_data_real.csv")
    print(f"  Cleaned:            data/processed/news_data_cleaned_real.csv")
    print(f"  Filtered:           data/processed/glp1_relevant_real.csv")
    print(f"  With sentiment:     data/processed/articles_with_sentiment_real.csv")
    
    print("\nWeekly Sentiment Indices:")
    print(f"  Market index:       outputs/weekly_market_sentiment_index_real.csv")
    print(f"  Firm indices:       outputs/weekly_firm_sentiment_indices_real.csv")
    print(f"  Keyword indices:    outputs/weekly_keyword_sentiment_indices_real.csv")
    
    print("\nNext Steps:")
    print("  • Use weekly indices for econometric analysis")
    print("  • Correlate with returns data (yfinance)")
    print("  • Perform Granger causality tests")
    print("  • Build predictive models")
    print("\nLog file: logs/full_pipeline.log")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
