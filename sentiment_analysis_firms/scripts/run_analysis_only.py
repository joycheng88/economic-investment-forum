"""
Analysis-Only Pipeline: Preprocessing → Filtering → Sentiment → Weekly Indices
Skips data collection and runs analysis on existing raw data
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
        logging.FileHandler('logs/analysis_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def step1_preprocess():
    """STEP 1: Preprocess raw data"""
    print("\n" + "=" * 80)
    print("STEP 1/4: PREPROCESSING TEXT")
    print("=" * 80)
    
    try:
        from src.preprocessing.text_preprocessor import TextPreprocessor
        
        input_path = 'data/raw/news_data_real.csv'
        if not os.path.exists(input_path):
            logger.error(f"Raw data file not found: {input_path}")
            return None
        
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} articles for preprocessing")
        print(f"\nLoaded {len(df)} articles")
        
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
        print(f"Valid articles: {len(df_valid)}/{len(df_clean)}")
        
        # Save cleaned data
        os.makedirs('data/processed', exist_ok=True)
        output_path = 'data/processed/news_data_cleaned_real.csv'
        df_valid.to_csv(output_path, index=False)
        logger.info(f"✓ Saved to: {output_path}")
        
        print(f"✓ Preprocessed and saved to: {output_path}")
        return df_valid
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}", exc_info=True)
        print(f"✗ Preprocessing failed: {e}")
        return None


def step2_filter():
    """STEP 2: Filter for GLP-1 relevance"""
    print("\n" + "=" * 80)
    print("STEP 2/4: FILTERING FOR GLP-1 RELEVANCE")
    print("=" * 80)
    
    try:
        from src.filtering.filter_glp1_relevant import filter_glp1_relevant
        
        input_path = 'data/processed/news_data_cleaned_real.csv'
        output_path = 'data/processed/glp1_relevant_real.csv'
        
        df_filtered, stats = filter_glp1_relevant(input_path, output_path)
        
        if df_filtered is None:
            logger.error("Filtering failed")
            return None
        
        logger.info(f"Filtered to {len(df_filtered)} relevant articles")
        print(f"✓ Filtered to {len(df_filtered)} articles with BOTH firm + GLP-1")
        print(f"  Retention: {stats['kept_articles']}/{stats['total_articles']} ({100*stats['kept_articles']/stats['total_articles']:.1f}%)")
        
        return df_filtered
        
    except Exception as e:
        logger.error(f"Filtering error: {e}", exc_info=True)
        print(f"✗ Filtering failed: {e}")
        return None


def step3_sentiment():
    """STEP 3: Analyze sentiment with VADER"""
    print("\n" + "=" * 80)
    print("STEP 3/4: SENTIMENT ANALYSIS (VADER)")
    print("=" * 80)
    
    try:
        from src.sentiment.vader_analyzer import VADERSentimentAnalyzer
        
        # Load filtered data
        input_path = 'data/processed/glp1_relevant_real.csv'
        
        if not os.path.exists(input_path):
            logger.error(f"Filtered data not found: {input_path}")
            return None
        
        df_filtered = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df_filtered)} filtered articles")
        print(f"\nLoaded {len(df_filtered)} filtered articles for sentiment analysis")
        
        analyzer = VADERSentimentAnalyzer()
        df_sentiment = analyzer.analyze_dataframe(df_filtered, text_column='clean_text')
        
        output_path = 'data/processed/articles_with_sentiment_real.csv'
        df_sentiment.to_csv(output_path, index=False)
        logger.info(f"✓ Saved to: {output_path}")
        
        # Get summary stats
        stats = analyzer.get_summary_stats(df_sentiment)
        
        print(f"\n✓ Sentiment analysis complete")
        print(f"  Total articles: {len(df_sentiment)}")
        print(f"  Positive: {stats['positive_count']} ({stats['positive_pct']:.1f}%)")
        print(f"  Negative: {stats['negative_count']} ({stats['negative_pct']:.1f}%)")
        print(f"  Neutral: {stats['neutral_count']} ({stats['neutral_pct']:.1f}%)")
        print(f"  Avg compound sentiment: {stats['avg_compound_score']:.3f}")
        
        return df_sentiment
        
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}", exc_info=True)
        print(f"✗ Sentiment analysis failed: {e}")
        return None


def step4_weekly_indices(df_sentiment):
    """STEP 4: Create weekly sentiment indices"""
    print("\n" + "=" * 80)
    print("STEP 4/4: CREATING WEEKLY SENTIMENT INDICES")
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
        
        print(f"    ✓ {len(weekly_index)} weeks of market sentiment data")
        
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
        if 'week_start' in weekly_index.columns:
            print(weekly_index[cols_display].to_string(index=False))
        
        print("\n" + "-" * 80)
        print("FIRM SENTIMENT (WEEKLY - Top 15)")
        print("-" * 80)
        firm_cols = ['firm', 'week_start', 'num_articles', 'avg_compound_sentiment', 'positive_pct']
        available_cols = [c for c in firm_cols if c in firm_weekly_index.columns]
        print(firm_weekly_index[available_cols].head(15).to_string(index=False))
        
        print("\n" + "-" * 80)
        print("KEYWORD SENTIMENT (WEEKLY - Top 10)")
        print("-" * 80)
        kw_cols = ['keyword', 'week_start', 'num_articles', 'avg_compound_sentiment', 'positive_pct']
        available_kw_cols = [c for c in kw_cols if c in keyword_weekly_index.columns]
        print(keyword_weekly_index[available_kw_cols].head(10).to_string(index=False))
        
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
    """Execute analysis-only pipeline"""
    
    print("\n" + "=" * 80)
    print("ANALYSIS PIPELINE: PREPROCESSING → SENTIMENT → WEEKLY INDICES")
    print("=" * 80)
    print("\nThis pipeline will:")
    print("  1. Preprocess and clean raw articles")
    print("  2. Filter for GLP-1 relevance")
    print("  3. Analyze sentiment with VADER")
    print("  4. Create WEEKLY sentiment indices by firm and keyword")
    print("\n" + "=" * 80)
    
    # Step 1: Preprocess
    df_cleaned = step1_preprocess()
    if df_cleaned is None:
        print("\n✗ Pipeline interrupted at preprocessing step")
        return
    
    # Step 2: Filter
    df_filtered = step2_filter()
    if df_filtered is None:
        print("\n✗ Pipeline interrupted at filtering step")
        return
    
    # Step 3: Sentiment
    df_sentiment = step3_sentiment()
    if df_sentiment is None:
        print("\n✗ Pipeline interrupted at sentiment step")
        return
    
    # Step 4: Weekly indices
    indices = step4_weekly_indices(df_sentiment)
    if indices is None:
        print("\n✗ Pipeline interrupted at weekly indexing step")
        return
    
    # Final summary
    print("\n" + "=" * 80)
    print("✓✓✓ ANALYSIS PIPELINE COMPLETE ✓✓✓")
    print("=" * 80)
    
    print("\nData Files Generated:")
    print(f"  Cleaned:            data/processed/news_data_cleaned_real.csv")
    print(f"  Filtered:           data/processed/glp1_relevant_real.csv")
    print(f"  With sentiment:     data/processed/articles_with_sentiment_real.csv")
    
    print("\nWeekly Sentiment Indices:")
    print(f"  Market index:       outputs/weekly_market_sentiment_index_real.csv")
    print(f"  Firm indices:       outputs/weekly_firm_sentiment_indices_real.csv")
    print(f"  Keyword indices:    outputs/weekly_keyword_sentiment_indices_real.csv")
    
    print("\n" + "=" * 80)
    print("Ready for:")
    print("  - Visualization with: python scripts/plot_sentiment_trends.py")
    print("  - Econometric analysis with: python scripts/did_regression.py")
    print("  - Panel data analysis with: python scripts/panel_workflow.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
