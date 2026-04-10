"""
News data collection using gnews API
Collects articles related to snack firms and GLP-1 keywords
"""

import os
import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Set
import time

try:
    from gnews import GNews
except ImportError:
    raise ImportError(
        "gnews package required. Install with: pip install gnews"
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GLP1NewsCollector:
    """Collects news articles about firms and GLP-1 keywords using gnews API"""
    
    def __init__(
        self,
        max_results: int = 100,
        language: str = "en",
        country: str = None,
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize the news collector.
        
        Args:
            max_results (int): Maximum results per query (default: 100)
            language (str): Language for articles (default: "en")
            country (str): Country code filter (optional)
            max_retries (int): Max retry attempts per query (default: 3)
            retry_delay (int): Delay between retries in seconds (default: 2)
        """
        self.google_news = GNews(
            language=language,
            country=country,
            max_results=max_results
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.collected_articles = []
        self.seen_urls: Set[str] = set()  # Track URLs to avoid duplicates
        self.seen_title_dates: Set[tuple] = set()  # Track (title, date) pairs
        
        logger.info(
            f"Initialized GLP1NewsCollector (max_results={max_results}, "
            f"language={language})"
        )
    
    def search_articles(
        self,
        query: str,
        top_news: bool = False
    ) -> List[Dict]:
        """
        Search for articles using a query.
        
        Args:
            query (str): Search query
            top_news (bool): Use top news endpoint if True
            
        Returns:
            list: List of article dictionaries
        """
        articles = []
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Searching for: '{query}' (attempt {attempt + 1})")
                
                if top_news:
                    result = self.google_news.get_top_news()
                else:
                    result = self.google_news.get_news(query)
                
                articles = result if isinstance(result, list) else []
                logger.info(f"Found {len(articles)} articles for '{query}'")
                break
                
            except Exception as e:
                logger.warning(
                    f"Error searching '{query}': {str(e)} "
                    f"(attempt {attempt + 1})"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to search '{query}' after {self.max_retries} attempts")
        
        return articles
    
    def parse_article(self, article: Dict, query: str, firm: str = None, keyword: str = None) -> Dict:
        """
        Parse raw article data into standardized format.
        
        Args:
            article (dict): Raw article from gnews
            query (str): Search query that found this article
            firm (str): Firm name (optional)
            keyword (str): GLP-1 keyword (optional)
            
        Returns:
            dict: Parsed article data
        """
        try:
            # Extract fields
            url = article.get('url', '')
            title = article.get('title', '')
            description = article.get('description', '')
            published_at = article.get('published at', '')
            source = article.get('source', {})
            source_name = source.get('title', '') if isinstance(source, dict) else ''
            source_url = source.get('href', '') if isinstance(source, dict) else ''
            image = article.get('image', '')
            
            # Try to get full text (gnews may not always provide it)
            full_text = article.get('content', description) or description
            
            # Parse publication date
            try:
                if published_at:
                    pub_date = pd.to_datetime(published_at)
                else:
                    pub_date = datetime.now()
            except Exception:
                pub_date = datetime.now()
            
            parsed = {
                'title': title,
                'description': description,
                'full_text': full_text,
                'url': url,
                'source_name': source_name,
                'source_url': source_url,
                'image_url': image,
                'published_date': pub_date,
                'search_query': query,
                'firm_name': firm,
                'keyword_used': keyword,
                'collection_date': datetime.now()
            }
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing article: {str(e)}")
            return None
    
    def collect_articles(
        self,
        firms: List[str],
        keywords: List[str],
        custom_queries: List[str] = None,
        rate_limit_delay: float = 1.0
    ) -> pd.DataFrame:
        """
        Collect articles for all firm-keyword combinations.
        Tracks which firm and keyword combination found each article.
        
        Args:
            firms (list): List of firm names
            keywords (list): List of GLP-1 keywords
            custom_queries (list): Additional custom queries (optional)
            rate_limit_delay (float): Delay between queries in seconds
            
        Returns:
            pd.DataFrame: DataFrame with collected articles
        """
        total_queries = len(firms) * len(keywords)
        query_count = 0
        
        logger.info(f"Collecting articles for {len(firms)} firms × {len(keywords)} keywords")
        logger.info(f"Total firm-keyword combinations: {total_queries}")
        
        # Loop over all firm-keyword combinations
        for firm in firms:
            for keyword in keywords:
                query_count += 1
                query = f'"{firm}" "{keyword}"'
                
                logger.info(f"[{query_count}/{total_queries}] Searching: {firm} + {keyword}")
                
                # Search for articles
                articles = self.search_articles(query)
                
                # Parse and store articles
                for article in articles:
                    parsed = self.parse_article(
                        article, 
                        query,
                        firm=firm,
                        keyword=keyword
                    )
                    
                    if parsed:
                        # Check if we've already seen this article by URL
                        if parsed['url'] in self.seen_urls:
                            logger.debug(f"Skipping duplicate URL: {parsed['url']}")
                            continue
                        
                        # Check if we've already seen this article by title+date
                        title_date_key = (parsed['title'], str(parsed['published_date']))
                        if title_date_key in self.seen_title_dates:
                            logger.debug(f"Skipping duplicate article: {parsed['title'][:50]}...")
                            continue
                        
                        # Add to collected articles
                        self.collected_articles.append(parsed)
                        self.seen_urls.add(parsed['url'])
                        self.seen_title_dates.add(title_date_key)
                
                # Rate limiting
                if query_count < total_queries:
                    time.sleep(rate_limit_delay)
        
        # Add custom queries if provided
        if custom_queries:
            logger.info(f"Processing {len(custom_queries)} custom queries")
            for i, query in enumerate(custom_queries):
                logger.info(f"Custom query {i + 1}/{len(custom_queries)}: {query}")
                
                articles = self.search_articles(query)
                
                for article in articles:
                    parsed = self.parse_article(article, query, firm=None, keyword=None)
                    
                    if parsed:
                        title_date_key = (parsed['title'], str(parsed['published_date']))
                        if title_date_key not in self.seen_title_dates:
                            self.collected_articles.append(parsed)
                            self.seen_urls.add(parsed['url'])
                            self.seen_title_dates.add(title_date_key)
                
                if i < len(custom_queries) - 1:
                    time.sleep(rate_limit_delay)
        
        logger.info(f"Total unique articles collected: {len(self.collected_articles)}")
        
        # Convert to DataFrame
        if self.collected_articles:
            df = pd.DataFrame(self.collected_articles)
            return df
        else:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=[
                'title', 'description', 'full_text', 'url',
                'source_name', 'source_url', 'image_url',
                'published_date', 'search_query', 'firm_name', 'keyword_used',
                'collection_date'
            ])
    
    def save_to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Save DataFrame to CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filepath (str): Path to save CSV file
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"Saved {len(df)} articles to {filepath}")
        except Exception as e:
            logger.error(f"Error saving to CSV: {str(e)}")
            raise
    
    def deduplicatae_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate articles by URL and by title+date combination.
        
        Args:
            df (pd.DataFrame): DataFrame with articles
            
        Returns:
            pd.DataFrame: Deduplicated DataFrame
        """
        initial_count = len(df)
        
        # First deduplicate by URL
        df_dedup = df.drop_duplicates(subset=['url'], keep='first')
        url_removed = initial_count - len(df_dedup)
        
        # Then deduplicate by title + published_date
        df_dedup['title_date_key'] = (
            df_dedup['title'].astype(str) + '|' + 
            df_dedup['published_date'].astype(str)
        )
        df_dedup = df_dedup.drop_duplicates(subset=['title_date_key'], keep='first')
        df_dedup = df_dedup.drop(columns=['title_date_key'])
        
        title_date_removed = initial_count - url_removed - len(df_dedup)
        total_removed = initial_count - len(df_dedup)
        
        logger.info(
            f"Deduplication: {initial_count} → {len(df_dedup)} articles "
            f"(removed {url_removed} by URL, {title_date_removed} by title+date)"
        )
        return df_dedup
    
    def filter_by_date(
        self,
        df: pd.DataFrame,
        days_back: int = 30
    ) -> pd.DataFrame:
        """
        Filter articles to only include recent ones.
        
        Args:
            df (pd.DataFrame): DataFrame with articles
            days_back (int): Number of days to look back (default: 30)
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        df['published_date'] = pd.to_datetime(df['published_date'])
        df_filtered = df[df['published_date'] >= cutoff_date]
        
        logger.info(
            f"Filtered to {len(df_filtered)} articles from last {days_back} days"
        )
        return df_filtered
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics about collected articles.
        
        Args:
            df (pd.DataFrame): DataFrame with articles
            
        Returns:
            dict: Summary statistics
        """
        return {
            'total_articles': len(df),
            'unique_sources': df['source_name'].nunique(),
            'date_range': f"{df['published_date'].min()} to {df['published_date'].max()}",
            'average_description_length': df['description'].str.len().mean(),
            'top_sources': df['source_name'].value_counts().head(5).to_dict(),
            'articles_per_query': df['search_query'].value_counts().head(5).to_dict()
        }


def main():
    """Main execution function - example usage"""
    from src.firm_names import SNACK_COMPANIES
    from src.glp1_keywords import GLP1_MEDICATIONS, GLP1_MEDICAL_TERMS
    
    # Initialize collector
    collector = GLP1NewsCollector(
        max_results=50,
        language="en",
        max_retries=2
    )
    
    # Prepare search terms
    firms = SNACK_COMPANIES
    keywords = GLP1_MEDICATIONS[:5] + GLP1_MEDICAL_TERMS[:5]  # Sample keywords
    
    custom_queries = [
        "GLP-1 weight loss market",
        "obesity medication competition",
        "pharmaceutical innovation snacks"
    ]
    
    logger.info(f"Starting collection for {len(firms)} firms and {len(keywords)} keywords")
    
    # Collect articles
    df = collector.collect_articles(
        firms=firms,
        keywords=keywords,
        custom_queries=custom_queries,
        rate_limit_delay=1.5
    )
    
    # Process results
    if len(df) > 0:
        # Deduplicate
        df = collector.deduplicatae_articles(df)
        
        # Filter by date (optional)
        df = collector.filter_by_date(df, days_back=30)
        
        # Save to CSV
        output_path = "data/raw/news_data.csv"
        collector.save_to_csv(df, output_path)
        
        # Print summary statistics
        stats = collector.get_summary_stats(df)
        logger.info("Collection Summary:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Display sample
        logger.info("\nSample articles:")
        print(df[['title', 'source_name', 'published_date']].head(10))
    else:
        logger.warning("No articles collected")


if __name__ == "__main__":
    main()
