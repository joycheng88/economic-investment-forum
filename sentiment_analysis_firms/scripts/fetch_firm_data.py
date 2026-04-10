"""
Fetch and create firm-specific datasets
Collects news for each firm combined with weight loss/GLP-1 keywords
Maximizes data collection per firm
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Firm configuration
FIRMS = {
    'RXBAR': ['RXBAR', 'RxBar'],
    'Chomps': ['Chomps', 'Chomps meat'],
    'Wonderful': ['Wonderful', 'Wonderful Pistachios', 'Wonderful Almonds'],
    'General Mills': ['General Mills', 'Cheerios', 'Lucky Charms'],
    'PepsiCo': ['PepsiCo', 'Frito-Lay', 'Doritos', 'Lay\'s'],
    'Nestle': ['Nestle', 'Nescafé', 'KitKat', 'Gerber'],
    'Mars': ['Mars', 'M&M', 'Snickers', 'Milky Way'],
    'Mondelez': ['Mondelez', 'Oreo', 'Cadbury', 'Trident'],
    'Ferrero': ['Ferrero', 'Nutella', 'Ferrero Rocher', 'Tic Tac'],
    'Hershey': ['Hershey', 'Kisses', 'Reese\'s', 'Twizzlers']
}

# GLP-1 and weight loss keywords (firm-focused)
WEIGHT_LOSS_KEYWORDS = [
    'GLP-1', 'Ozempic', 'Wegovy', 'Mounjaro', 'weight loss',
    'obesity', 'appetite suppressant', 'metabolic', 'diabetes',
    'semaglutide', 'tirzepatide', 'anti-obesity', 'weight management'
]


def fetch_firm_data(firm_name, firm_aliases):
    """
    Fetch news data for a specific firm with weight loss/GLP-1 keywords.
    Uses multiple search queries to maximize coverage.
    """
    try:
        from src.data_collection.news_collector import GLP1NewsCollector
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Fetching data for: {firm_name}")
        logger.info(f"{'='*60}")
        
        collector = GLP1NewsCollector(
            max_results=100,  # Maximum results per query
            language="en",
            max_retries=2,
            retry_delay=1
        )
        
        all_articles = []
        
        # Query 1: Firm + GLP-1
        for alias in firm_aliases:
            queries = [
                f'"{alias}" GLP-1',
                f'"{alias}" Ozempic',
                f'"{alias}" weight loss',
                f'"{alias}" obesity',
                f'"{alias}" appetite suppressant',
                f'"{alias}" weight management',
                f'"{alias}" semaglutide',
                f'"{alias}" tirzepatide',
            ]
            
            for query in queries:
                logger.info(f"Query: {query}")
                try:
                    df_results = collector.search_articles(query, max_results=50)
                    if df_results is not None and len(df_results) > 0:
                        all_articles.append(df_results)
                        logger.info(f"  ✓ Found {len(df_results)} articles")
                    else:
                        logger.info(f"  ⚠ No results")
                except Exception as e:
                    logger.warning(f"  ✗ Error: {str(e)}")
        
        # Combine all results
        if all_articles:
            df_firm = pd.concat(all_articles, ignore_index=True)
            
            # Deduplicate
            initial_count = len(df_firm)
            df_firm = collector.deduplicatae_articles(df_firm)
            final_count = len(df_firm)
            
            logger.info(f"\nFinal count: {final_count}/{initial_count} unique articles")
            
            # Add firm name
            df_firm['firm_name'] = firm_name
            
            return df_firm
        else:
            logger.warning(f"No articles found for {firm_name}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching data for {firm_name}: {str(e)}", exc_info=True)
        return None


def combine_with_existing(firm_name, df_new):
    """
    Combine newly fetched data with existing data to maximize coverage.
    """
    try:
        # Load existing data for this firm
        filename = f"firm_datasets/{firm_name.lower().replace(' ', '_').replace('/', '_')}_data.csv"
        
        if os.path.exists(filename):
            df_existing = pd.read_csv(filename)
            logger.info(f"  Found {len(df_existing)} existing articles")
            
            # Combine
            if df_new is not None and len(df_new) > 0:
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                # Deduplicate by URL (if available)
                if 'url' in df_combined.columns:
                    df_combined = df_combined.drop_duplicates(subset=['url'], keep='first')
                
                logger.info(f"  Combined: {len(df_combined)} total articles")
                return df_combined
        else:
            return df_new
            
    except Exception as e:
        logger.error(f"Error combining data: {str(e)}")
        return df_new


def save_firm_dataset(firm_name, df_firm):
    """
    Save firm-specific dataset to CSV.
    """
    try:
        # Create filename
        filename = firm_name.lower().replace(' ', '_').replace('/', '_')
        output_path = f'firm_datasets/{filename}_data.csv'
        
        os.makedirs('firm_datasets', exist_ok=True)
        
        df_firm.to_csv(output_path, index=False)
        
        logger.info(f"✓ Saved: {output_path} ({len(df_firm)} articles)")
        
        return output_path
    except Exception as e:
        logger.error(f"Error saving dataset: {str(e)}")
        return None


def main():
    """Main execution"""
    
    print("\n" + "=" * 80)
    print("FETCHING AND CREATING FIRM-SPECIFIC DATASETS")
    print("=" * 80)
    print("\nThis will:")
    print("  1. Fetch fresh data for each firm")
    print("  2. Combine with existing data")
    print("  3. Create firm-specific CSVs with GLP-1/weight loss focus")
    print("\n" + "=" * 80)
    
    os.makedirs('firm_datasets', exist_ok=True)
    
    summary = []
    
    for firm_name, aliases in sorted(FIRMS.items()):
        logger.info(f"\n>>> Processing: {firm_name}")
        
        # Fetch new data
        df_new = fetch_firm_data(firm_name, aliases)
        
        # Combine with existing
        df_combined = combine_with_existing(firm_name, df_new)
        
        # Save
        if df_combined is not None and len(df_combined) > 0:
            output_path = save_firm_dataset(firm_name, df_combined)
            
            summary.append({
                'Firm': firm_name,
                'Articles': len(df_combined),
                'File': f'{firm_name.lower().replace(" ", "_").replace("/", "_")}_data.csv'
            })
        else:
            logger.warning(f"No data available for {firm_name}")
            summary.append({
                'Firm': firm_name,
                'Articles': 0,
                'File': f'{firm_name.lower().replace(" ", "_").replace("/", "_")}_data.csv'
            })
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    df_summary = pd.DataFrame(summary)
    print("\n" + df_summary.to_string(index=False))
    
    print("\n" + "=" * 80)
    print(f"Total articles collected: {df_summary['Articles'].sum()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
