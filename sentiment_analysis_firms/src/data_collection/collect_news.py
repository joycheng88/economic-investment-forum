"""
Example usage script for collecting GLP-1 related news
Run this script to collect articles and save to CSV
"""

import sys
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_collection.news_collector import GLP1NewsCollector
from src.firm_names import SNACK_COMPANIES
from src.glp1_keywords import (
    GLP1_MEDICATIONS,
    GLP1_MEDICAL_TERMS,
    GLP1_MARKET_TERMS,
    GLP1_HEALTH_TOPICS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/news_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def collect_news_for_firms():
    """Collect news articles for all firms and GLP-1 keywords"""
    
    # Initialize collector
    logger.info("Initializing GLP1NewsCollector...")
    collector = GLP1NewsCollector(
        max_results=100,  # Get up to 100 results per query
        language="en",
        country=None,  # None = all countries
        max_retries=3,
        retry_delay=2
    )
    
    # Define search terms
    firms = SNACK_COMPANIES
    
    # Combine different keyword categories
    keywords = (
        GLP1_MEDICATIONS[:8] +           # Top medications
        GLP1_MEDICAL_TERMS[:6] +         # Medical terms
        ["weight loss", "obesity", "GLP-1"]  # Core terms
    )
    
    # Custom queries for broader topics
    custom_queries = [
        "GLP-1 market growth",
        "obesity medicine competition",
        "pharmaceutical innovation weight loss",
        "snack company diversification GLP-1",
        "diabetes drug market",
        "Ozempic Wegovy competition",
        "appetite suppressant market",
        "healthcare spending GLP-1"
    ]
    
    logger.info(f"Collecting articles for:")
    logger.info(f"  - {len(firms)} firms: {firms}")
    logger.info(f"  - {len(keywords)} keywords (sample): {keywords[:5]}...")
    logger.info(f"  - {len(custom_queries)} custom queries")
    logger.info(f"  - Total firm-keyword combinations: {len(firms) * len(keywords)}")
    
    # Collect articles
    try:
        df = collector.collect_articles(
            firms=firms,
            keywords=keywords,
            custom_queries=custom_queries,
            rate_limit_delay=1.5  # 1.5 second delay between queries
        )
        
        logger.info(f"Raw collection: {len(df)} articles")
        
        # Deduplicate by URL and title+date
        df = collector.deduplicatae_articles(df)
        logger.info(f"After deduplication: {len(df)} unique articles")
        
        # Filter to recent articles (last 30 days)
        df = collector.filter_by_date(df, days_back=30)
        logger.info(f"After date filter (30 days): {len(df)} articles")
        
        # Sort by date
        df = df.sort_values('published_date', ascending=False)
        
        # Save to CSV
        output_path = "data/raw/news_data.csv"
        collector.save_to_csv(df, output_path)
        
        # Print summary statistics
        logger.info("\n" + "="*60)
        logger.info("COLLECTION SUMMARY")
        logger.info("="*60)
        
        stats = collector.get_summary_stats(df)
        
        logger.info(f"Total articles: {stats['total_articles']}")
        logger.info(f"Unique sources: {stats['unique_sources']}")
        logger.info(f"Date range: {stats['date_range']}")
        logger.info(f"Avg description length: {stats['average_description_length']:.0f} chars")
        
        logger.info("\nTop 5 sources:")
        for source, count in stats['top_sources'].items():
            logger.info(f"  {source}: {count} articles")
        
        logger.info("\nTop 5 search queries by volume:")
        for query, count in stats['articles_per_query'].items():
            logger.info(f"  {query}: {count} articles")
        
        # Display firm and keyword coverage
        logger.info("\n" + "="*60)
        logger.info("FIRM AND KEYWORD COVERAGE")
        logger.info("="*60)
        
        if 'firm_name' in df.columns:
            firm_counts = df[df['firm_name'].notna()]['firm_name'].value_counts()
            logger.info(f"\nFirm coverage ({len(firm_counts)} firms):")
            for firm, count in firm_counts.head(10).items():
                logger.info(f"  {firm}: {count} articles")
        
        if 'keyword_used' in df.columns:
            keyword_counts = df[df['keyword_used'].notna()]['keyword_used'].value_counts()
            logger.info(f"\nKeyword coverage ({len(keyword_counts)} keywords):")
            for keyword, count in keyword_counts.head(10).items():
                logger.info(f"  {keyword}: {count} articles")
        
        # Display sample articles
        logger.info("\n" + "="*60)
        logger.info("SAMPLE ARTICLES")
        logger.info("="*60)
        
        sample_df = df[['title', 'firm_name', 'keyword_used', 'source_name', 
                        'published_date']].head(10)
        for idx, row in sample_df.iterrows():
            logger.info(f"\n{idx + 1}. {row['title'][:60]}...")
            logger.info(f"   Firm: {row['firm_name']}, Keyword: {row['keyword_used']}")
            logger.info(f"   Source: {row['source_name']}")
            logger.info(f"   Date: {row['published_date']}")
        
        logger.info("\n" + "="*60)
        logger.info(f"✓ Collection complete! Data saved to {output_path}")
        logger.info("="*60)
    
    except Exception as e:
        logger.error(f"Error during collection: {str(e)}", exc_info=True)
        raise


def quick_search():
    """Quick search for a few articles (for testing)"""
    
    logger.info("Running quick search test...")
    
    collector = GLP1NewsCollector(max_results=10)
    
    # Just search for a few high-level queries
    test_queries = [
        "GLP-1 Ozempic",
        "PepsiCo weight loss",
        "Nestle diabetes"
    ]
    
    all_articles = []
    
    for query in test_queries:
        logger.info(f"Searching for: {query}")
        articles = collector.search_articles(query)
        logger.info(f"Found {len(articles)} articles")
        
        for article in articles:
            parsed = collector.parse_article(article, query, firm=None, keyword=None)
            if parsed:
                all_articles.append(parsed)
    
    if all_articles:
        df = pd.DataFrame(all_articles)
        logger.info(f"\nQuick search results: {len(df)} articles")
        logger.info("\nSample:")
        print(df[['title', 'source_name']].head())
    else:
        logger.warning("No articles found in quick search")


if __name__ == "__main__":
    import pandas as pd
    
    logger.info("="*60)
    logger.info("GLP-1 NEWS COLLECTION SCRIPT")
    logger.info("="*60)
    
    # Uncomment one of the following:
    
    # Option 1: Full collection
    collect_news_for_firms()
    
    # Option 2: Quick test (comment out Option 1 first)
    # quick_search()
