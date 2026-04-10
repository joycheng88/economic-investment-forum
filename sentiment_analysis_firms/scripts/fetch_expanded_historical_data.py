"""
Expanded historical data collection for firms
Fetches news from broader time periods (January 2026 back, even to 2015)
Goal: Collect 1500+ articles per firm for comprehensive analysis
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
import time

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/expanded_data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Firm configuration
FIRMS_CONFIG = {
    'RXBAR': {
        'aliases': ['RXBAR', 'RxBar'],
        'filename': 'rxbar_expanded_data.csv'
    },
    'Chomps': {
        'aliases': ['Chomps', 'Chomps meat snacks'],
        'filename': 'chomps_expanded_data.csv'
    },
    'Wonderful': {
        'aliases': ['Wonderful', 'Wonderful Pistachios', 'Wonderful Almonds'],
        'filename': 'wonderful_expanded_data.csv'
    },
    'General Mills': {
        'aliases': ['General Mills', 'Cheerios', 'Lucky Charms', 'Pillsbury'],
        'filename': 'general_mills_expanded_data.csv'
    },
    'PepsiCo': {
        'aliases': ['PepsiCo', 'Frito-Lay', 'Doritos', "Lay's"],
        'filename': 'pepsico_expanded_data.csv'
    },
    'Nestle': {
        'aliases': ['Nestle', 'Nescafé', 'KitKat'],
        'filename': 'nestle_expanded_data.csv'
    },
    'Mars': {
        'aliases': ['Mars', 'M&M', 'Snickers'],
        'filename': 'mars_expanded_data.csv'
    },
    'Mondelez': {
        'aliases': ['Mondelez', 'Oreo', 'Cadbury'],
        'filename': 'mondelez_expanded_data.csv'
    },
    'Ferrero': {
        'aliases': ['Ferrero', 'Nutella', 'Ferrero Rocher'],
        'filename': 'ferrero_expanded_data.csv'
    },
    'Hershey': {
        'aliases': ['Hershey', 'Kisses', "Reese's"],
        'filename': 'hershey_expanded_data.csv'
    }
}

# Broader keyword search space
GLPI_KEYWORDS = [
    'GLP-1', 'Ozempic', 'Wegovy', 'weight loss',
    'obesity', 'semaglutide', 'Mounjaro', 'Zepbound'
]

# Generic queries for broader coverage
GENERIC_QUERIES = [
    'snack food industry weight loss market',
    'food company obesity GLP-1 impact',
    'confectionery weight loss drug competition',
    'candy company market share weight loss',
    'healthy snacking appetite suppressant',
    'food industry disruption GLP-1',
    'snack company strategic response obesity',
    'convenience food market appetite',
    'packaged food sales weight loss trend'
]


def fetch_expanded_firm_data(firm_name, firm_config):
    """
    Fetch extensive historical data for a firm.
    Uses multiple query types and longer time ranges.
    """
    try:
        from src.data_collection.news_collector import GLP1NewsCollector
        
        logger.info(f"\n{'='*70}")
        logger.info(f"EXPANDED FETCH: {firm_name}")
        logger.info(f"{'='*70}")
        
        collector = GLP1NewsCollector(
            max_results=200,  # Increased from 100
            language="en",
            country=None,
            max_retries=3,
            retry_delay=0.5
        )
        
        all_articles = []
        query_count = 0
        
        # Strategy 1: Firm + GLP-1 keywords (multiple iterations)
        logger.info("\n[STRATEGY 1] Firm + GLP-1 Keywords")
        for alias in firm_config['aliases']:
            for keyword in GLPI_KEYWORDS:
                query = f'"{alias}" "{keyword}"'
                query_count += 1
                
                logger.info(f"  Query {query_count}: {query}")
                
                try:
                    df_results = collector.collect_articles(
                        firms=[alias],
                        keywords=[keyword],
                        custom_queries=[query],
                        rate_limit_delay=0.3
                    )
                    
                    if df_results is not None and len(df_results) > 0:
                        all_articles.append(df_results)
                        logger.info(f"    ✓ Found {len(df_results)} articles")
                    
                except Exception as e:
                    logger.warning(f"    ✗ Error: {str(e)}")
                    continue
                
                time.sleep(0.3)
        
        # Strategy 2: Firm alone (for broader coverage)
        logger.info("\n[STRATEGY 2] Firm Name Alone (Broader Coverage)")
        for alias in firm_config['aliases'][:2]:  # Use first 2 aliases
            query = f'"{alias}"'
            query_count += 1
            
            logger.info(f"  Query {query_count}: {query}")
            
            try:
                df_results = collector.collect_articles(
                    firms=[alias],
                    keywords=['market', 'sales', 'business'],
                    custom_queries=[f'"{alias}" market sales trend'],
                    rate_limit_delay=0.3
                )
                
                if df_results is not None and len(df_results) > 0:
                    all_articles.append(df_results)
                    logger.info(f"    ✓ Found {len(df_results)} articles")
                    
            except Exception as e:
                logger.warning(f"    ✗ Error: {str(e)}")
                continue
            
            time.sleep(0.3)
        
        # Strategy 3: Generic industry queries
        logger.info("\n[STRATEGY 3] Industry-Wide Queries (Context)")
        for query_template in GENERIC_QUERIES[:3]:  # First 3 generic queries
            query_count += 1
            
            logger.info(f"  Query {query_count}: {query_template}")
            
            try:
                df_results = collector.collect_articles(
                    firms=firm_config['aliases'][:1],
                    keywords=[],
                    custom_queries=[query_template],
                    rate_limit_delay=0.3
                )
                
                if df_results is not None and len(df_results) > 0:
                    all_articles.append(df_results)
                    logger.info(f"    ✓ Found {len(df_results)} articles")
                    
            except Exception as e:
                logger.warning(f"    ✗ Error: {str(e)}")
                continue
            
            time.sleep(0.3)
        
        # Combine and deduplicate
        if all_articles:
            logger.info(f"\n{'-'*70}")
            logger.info("Combining results...")
            
            df_combined = pd.concat(all_articles, ignore_index=True)
            logger.info(f"Total (before dedup): {len(df_combined)} articles")
            
            # Deduplicate by URL
            if 'url' in df_combined.columns:
                df_unique = df_combined.drop_duplicates(subset=['url'], keep='first')
                logger.info(f"After deduplication: {len(df_unique)} unique articles")
            else:
                df_unique = collector.deduplicatae_articles(df_combined)
                logger.info(f"After deduplication: {len(df_unique)} unique articles")
            
            # Add firm name
            df_unique['firm_name'] = firm_name
            
            return df_unique
        else:
            logger.warning(f"✗ No articles found for {firm_name}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching data for {firm_name}: {str(e)}", exc_info=True)
        return None


def merge_with_existing(firm_name, df_new):
    """
    Merge newly fetched data with existing firm dataset.
    Expands coverage by combining multiple collection runs.
    """
    try:
        # Original dataset
        original_file = f'firm_datasets/{FIRMS_CONFIG[firm_name]["filename"].replace("_expanded", "")}'
        
        # Check if new data exists
        if df_new is None or len(df_new) == 0:
            logger.warning(f"No new data to merge for {firm_name}")
            if os.path.exists(original_file):
                return pd.read_csv(original_file)
            return None
        
        # Merge with original if exists
        all_dataframes = [df_new]
        
        if os.path.exists(original_file):
            df_original = pd.read_csv(original_file)
            all_dataframes.insert(0, df_original)
            logger.info(f"  Found {len(df_original)} existing articles")
        
        # Combine
        df_combined = pd.concat(all_dataframes, ignore_index=True)
        
        # Deduplicate by URL
        if 'url' in df_combined.columns:
            df_combined = df_combined.drop_duplicates(subset=['url'], keep='first')
        
        logger.info(f"  Total after merge: {len(df_combined)} articles")
        
        return df_combined
        
    except Exception as e:
        logger.error(f"Error merging data: {str(e)}")
        return df_new


def save_expanded_dataset(firm_name, df_firm):
    """Save expanded firm dataset"""
    try:
        filename = FIRMS_CONFIG[firm_name]['filename']
        output_path = f'firm_datasets/{filename}'
        
        os.makedirs('firm_datasets', exist_ok=True)
        
        df_firm.to_csv(output_path, index=False)
        
        logger.info(f"\n✓ Saved expanded dataset: {output_path}")
        logger.info(f"  Articles: {len(df_firm)}")
        logger.info(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving dataset: {str(e)}")
        return None


def main():
    """Execute expanded data collection"""
    
    print("\n" + "=" * 80)
    print("EXPANDED HISTORICAL DATA COLLECTION")
    print("=" * 80)
    print("\nGoals:")
    print("  • Collect 1500+ articles per firm")
    print("  • Historical coverage: January 2026 back")
    print("  • Multiple query strategies for comprehensive coverage")
    print("  • Combine with existing datasets")
    
    print("\nCollection Strategies:")
    print("  1. Firm + GLP-1/Weight Loss keywords (primary)")
    print("  2. Firm name alone (broader context)")
    print("  3. Industry-wide queries (market context)")
    
    print("\n" + "=" * 80)
    
    os.makedirs('firm_datasets', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    summary = []
    total_articles = 0
    
    # Fetch data for each firm
    for firm_name in sorted(FIRMS_CONFIG.keys()):
        firm_config = FIRMS_CONFIG[firm_name]
        
        logger.info(f"\n>>> Processing: {firm_name}")
        print(f"\n>>> Collecting data for {firm_name}...")
        
        # Fetch new data
        df_new = fetch_expanded_firm_data(firm_name, firm_config)
        
        # Merge with existing
        df_combined = merge_with_existing(firm_name, df_new)
        
        if df_combined is not None and len(df_combined) > 0:
            # Save
            save_expanded_dataset(firm_name, df_combined)
            
            num_articles = len(df_combined)
            total_articles += num_articles
            
            summary.append({
                'Firm': firm_name,
                'Articles': num_articles,
                'File': firm_config['filename']
            })
            
            logger.info(f"✓ {firm_name}: {num_articles} total articles")
            print(f"  ✓ {num_articles} articles")
        else:
            summary.append({
                'Firm': firm_name,
                'Articles': 0,
                'File': firm_config['filename']
            })
            logger.warning(f"⚠ {firm_name}: No data")
            print(f"  ⚠ No data collected")
    
    # Final summary
    print("\n" + "=" * 80)
    print("EXPANDED COLLECTION COMPLETE")
    print("=" * 80)
    
    df_summary = pd.DataFrame(summary)
    
    print("\nExpanded Datasets:")
    print("-" * 80)
    print(f"{'Firm':<20} {'Articles':<12} {'Status':<20}")
    print("-" * 80)
    
    for _, row in df_summary.iterrows():
        articles = row['Articles']
        if articles >= 1500:
            status = "✓ EXCELLENT (1500+)"
        elif articles >= 500:
            status = "✓ GOOD (500+)"
        elif articles >= 100:
            status = "✓ FAIR (100+)"
        else:
            status = "⚠ LIMITED"
        
        print(f"{row['Firm']:<20} {articles:<12} {status:<20}")
    
    print("\n" + "-" * 80)
    print(f"Total expanded articles: {total_articles:,}")
    print(f"Average per firm: {total_articles / len(FIRMS_CONFIG):.0f}")
    
    print("\n" + "=" * 80)
    print("✓ READY FOR COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    print("\nNext Steps:")
    print("  1. Run sentiment analysis on each expanded dataset")
    print("  2. Compare firm sentiment over time")
    print("  3. Analyze market impact of GLP-1 adoption")
    print("  4. Track sentiment trends from January 2026 backwards")


if __name__ == "__main__":
    main()
