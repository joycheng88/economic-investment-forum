"""
Fetch firm-specific news data online
Collects targeted data for each of the 10 firms
Combines firm name with GLP-1/weight loss keywords
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
import time

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/firm_data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Firm configuration with brand names and aliases
FIRMS_CONFIG = {
    'RXBAR': {
        'aliases': ['RXBAR', 'RxBar'],
        'filename': 'rxbar_data.csv'
    },
    'Chomps': {
        'aliases': ['Chomps', 'Chomps meat snacks'],
        'filename': 'chomps_data.csv'
    },
    'Wonderful': {
        'aliases': ['Wonderful', 'Wonderful Pistachios', 'Wonderful Almonds'],
        'filename': 'wonderful_data.csv'
    },
    'General Mills': {
        'aliases': ['General Mills', 'Cheerios', 'Lucky Charms', 'Pillsbury'],
        'filename': 'general_mills_data.csv'
    },
    'PepsiCo': {
        'aliases': ['PepsiCo', 'Frito-Lay', 'Doritos', "Lay's"],
        'filename': 'pepsico_data.csv'
    },
    'Nestle': {
        'aliases': ['Nestle', 'Nescafé', 'KitKat'],
        'filename': 'nestle_data.csv'
    },
    'Mars': {
        'aliases': ['Mars', 'M&M', 'Snickers'],
        'filename': 'mars_data.csv'
    },
    'Mondelez': {
        'aliases': ['Mondelez', 'Oreo', 'Cadbury'],
        'filename': 'mondelez_data.csv'
    },
    'Ferrero': {
        'aliases': ['Ferrero', 'Nutella', 'Ferrero Rocher'],
        'filename': 'ferrero_data.csv'
    },
    'Hershey': {
        'aliases': ['Hershey', 'Kisses', "Reese's"],
        'filename': 'hershey_data.csv'
    }
}

# GLP-1 and weight loss keywords
WEIGHT_LOSS_KEYWORDS = [
    'GLP-1', 'Ozempic', 'Wegovy', 'Mounjaro', 'Zepbound',
    'weight loss', 'obesity', 'semaglutide', 'appetite suppressant',
    'anti-obesity drug', 'weight management'
]


def fetch_firm_data(firm_name, firm_config):
    """
    Fetch news data for a specific firm.
    Runs multiple targeted queries combining firm name with GLP-1/weight loss keywords.
    """
    try:
        from src.data_collection.news_collector import GLP1NewsCollector
        
        logger.info(f"\n{'='*70}")
        logger.info(f"FETCHING DATA: {firm_name}")
        logger.info(f"{'='*70}")
        
        collector = GLP1NewsCollector(
            max_results=100,
            language="en",
            country=None,
            max_retries=2,
            retry_delay=1
        )
        
        all_articles = []
        query_count = 0
        
        # Generate search queries: each firm alias × each GLP-1/weight loss keyword
        for alias in firm_config['aliases']:
            for keyword in WEIGHT_LOSS_KEYWORDS:
                query = f'"{alias}" "{keyword}"'
                query_count += 1
                
                logger.info(f"\n[Query {query_count}] {query}")
                
                try:
                    df_results = collector.collect_articles(
                        firms=[alias],
                        keywords=[keyword],
                        custom_queries=[query],
                        rate_limit_delay=0.5
                    )
                    
                    if df_results is not None and len(df_results) > 0:
                        all_articles.append(df_results)
                        logger.info(f"  ✓ Found {len(df_results)} articles")
                    else:
                        logger.info(f"  ⚠ No results")
                        
                except Exception as e:
                    logger.warning(f"  ✗ Error: {str(e)}")
                    continue
                
                # Rate limiting
                time.sleep(0.5)
        
        # Combine all results
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


def save_firm_dataset(firm_name, df_firm):
    """Save firm dataset to CSV"""
    try:
        firm_config = FIRMS_CONFIG[firm_name]
        filename = firm_config['filename']
        output_path = f'firm_datasets/{filename}'
        
        os.makedirs('firm_datasets', exist_ok=True)
        
        df_firm.to_csv(output_path, index=False)
        
        logger.info(f"\n✓ Saved: {output_path}")
        logger.info(f"  Size: {len(df_firm)} articles, {os.path.getsize(output_path) / 1024:.1f} KB")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving dataset: {str(e)}")
        return None


def main():
    """Main execution"""
    
    print("\n" + "=" * 80)
    print("FIRM-SPECIFIC DATA COLLECTION")
    print("=" * 80)
    print("\nThis will:")
    print("  1. Fetch fresh news data for each of 10 firms")
    print("  2. Use targeted queries: Firm + GLP-1/Weight Loss keywords")
    print("  3. Deduplicate results")
    print("  4. Save each firm to separate CSV")
    print("\nFirms:")
    for i, firm in enumerate(FIRMS_CONFIG.keys(), 1):
        print(f"  {i:2d}. {firm}")
    
    print("\n" + "=" * 80)
    
    os.makedirs('firm_datasets', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    summary = []
    total_articles = 0
    
    # Fetch data for each firm
    for firm_name, firm_config in sorted(FIRMS_CONFIG.items()):
        logger.info(f"\n>>> Starting collection for: {firm_name}")
        
        # Fetch
        df_firm = fetch_firm_data(firm_name, firm_config)
        
        if df_firm is not None and len(df_firm) > 0:
            # Save
            save_firm_dataset(firm_name, df_firm)
            
            num_articles = len(df_firm)
            total_articles += num_articles
            
            summary.append({
                'Firm': firm_name,
                'Articles': num_articles,
                'File': firm_config['filename']
            })
            
            logger.info(f"✓ {firm_name}: {num_articles} articles")
        else:
            summary.append({
                'Firm': firm_name,
                'Articles': 0,
                'File': firm_config['filename']
            })
            logger.warning(f"⚠ {firm_name}: No data collected")
    
    # Final summary
    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    
    df_summary = pd.DataFrame(summary)
    
    print("\nDatasets Created:")
    print("-" * 80)
    print(f"{'Firm':<20} {'Articles':<12} {'File':<30}")
    print("-" * 80)
    
    for _, row in df_summary.iterrows():
        file_path = f"firm_datasets/{row['File']}"
        if os.path.exists(file_path):
            print(f"{row['Firm']:<20} {row['Articles']:<12} {row['File']:<30}")
    
    print("\n" + "-" * 80)
    print(f"Total articles collected: {total_articles}")
    print(f"Average per firm: {total_articles / len(FIRMS_CONFIG):.1f}")
    print(f"Firms with data: {len(df_summary[df_summary['Articles'] > 0])}")
    
    print("\n" + "=" * 80)
    print("✓ FIRM DATASETS READY")
    print("=" * 80)
    
    print("\nNext steps:")
    print("  1. Run sentiment analysis on each firm dataset")
    print("  2. Compare sentiment across firms")
    print("  3. Analyze firm-specific GLP-1 market impact")


if __name__ == "__main__":
    main()
