"""
Real-time Data Ingestion Pipeline for GLP-1 Content
Fetches data from NewsAPI and Reddit (PRAW)
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import requests
import praw
from praw.exceptions import PRAWException
from requests.exceptions import RequestException

from config import (
    NEWS_API_KEY,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    build_query,
    NEWS_API_URL,
    REDDIT_SUBREDDITS,
    NEWS_RATE_LIMIT,
    REDDIT_RATE_LIMIT,
    MIN_TEXT_LENGTH,
    MAX_TEXT_LENGTH,
    UNIFIED_COLUMNS,
)

# ===========================
# LOGGING SETUP
# ===========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_ingestion.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ===========================
# NEWS API FETCHER
# ===========================


class NewsAPIFetcher:
    """Fetch articles from NewsAPI"""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("NEWS_API_KEY not configured. Set it in .env file")
        self.api_key = api_key
        self.url = NEWS_API_URL
        self.query = build_query()
        logger.info(f"NewsAPI initialized with query: {self.query[:50]}...")

    def fetch(self, days_back: int = 30, page_size: int = 100) -> pd.DataFrame:
        """
        Fetch articles from NewsAPI

        Args:
            days_back: Number of days to look back
            page_size: Number of results per page (max 100)

        Returns:
            DataFrame with columns: text, title, source, timestamp, url, author
        """
        logger.info(f"Fetching news articles from last {days_back} days...")

        from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime(
            "%Y-%m-%d"
        )

        params = {
            "q": self.query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "from": from_date,
            "apiKey": self.api_key,
        }

        articles = []
        try:
            response = requests.get(self.url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return pd.DataFrame()

            logger.info(f"Found {len(data.get('articles', []))} articles")

            for article in data.get("articles", []):
                text = self._clean_text(article)
                if self._is_valid(text):
                    articles.append(
                        {
                            "text": text,
                            "title": article.get("title", ""),
                            "source": "NewsAPI",
                            "timestamp": pd.to_datetime(article.get("publishedAt")),
                            "url": article.get("url", ""),
                            "author": article.get("author", ""),
                        }
                    )

            logger.info(f"Validated {len(articles)} articles from NewsAPI")

        except RequestException as e:
            logger.error(f"NewsAPI request failed: {e}")

        return pd.DataFrame(articles)

    @staticmethod
    def _clean_text(article: Dict) -> str:
        """Combine title and description"""
        title = (article.get("title") or "").strip()
        description = (article.get("description") or "").strip()
        return (title + " " + description).strip()

    @staticmethod
    def _is_valid(text: str) -> bool:
        """Check if text meets quality criteria"""
        return MIN_TEXT_LENGTH <= len(text) <= MAX_TEXT_LENGTH


# ===========================
# REDDIT FETCHER
# ===========================


class RedditFetcher:
    """Fetch posts from Reddit using PRAW"""

    def __init__(
        self, client_id: str, client_secret: str, user_agent: str
    ):
        if not client_id or not client_secret:
            raise ValueError(
                "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET not configured. "
                "Set them in .env file"
            )

        try:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )
            # Test connection
            self.reddit.user.me()
            logger.info("Reddit API initialized successfully")
        except PRAWException as e:
            logger.error(f"Reddit authentication failed: {e}")
            raise

        self.query = build_query()
        self.subreddits = REDDIT_SUBREDDITS

    def fetch(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch posts from Reddit subreddits

        Args:
            limit: Number of submissions to fetch per subreddit

        Returns:
            DataFrame with columns: text, title, source, timestamp, url, author
        """
        logger.info(
            f"Fetching Reddit posts from {len(self.subreddits)} subreddits..."
        )

        posts = []

        for subreddit_name in self.subreddits:
            try:
                logger.info(f"Searching r/{subreddit_name}...")
                subreddit = self.reddit.subreddit(subreddit_name)

                for submission in subreddit.search(self.query, limit=limit):
                    text = self._clean_text(submission)
                    if self._is_valid(text):
                        posts.append(
                            {
                                "text": text,
                                "title": submission.title,
                                "source": f"Reddit (r/{subreddit_name})",
                                "timestamp": pd.to_datetime(
                                    submission.created_utc, unit="s"
                                ),
                                "url": f"https://reddit.com{submission.permalink}",
                                "author": submission.author.name
                                if submission.author
                                else "[deleted]",
                            }
                        )

                time.sleep(REDDIT_RATE_LIMIT)  # Rate limiting

            except PRAWException as e:
                logger.warning(f"Error fetching from r/{subreddit_name}: {e}")
                continue

        logger.info(f"Validated {len(posts)} posts from Reddit")
        return pd.DataFrame(posts)

    @staticmethod
    def _clean_text(submission) -> str:
        """Combine title and selftext"""
        title = submission.title.strip()
        selftext = submission.selftext.strip()
        return (title + " " + selftext).strip()

    @staticmethod
    def _is_valid(text: str) -> bool:
        """Check if text meets quality criteria"""
        return MIN_TEXT_LENGTH <= len(text) <= MAX_TEXT_LENGTH


# ===========================
# UNIFIED PIPELINE
# ===========================


class GLPDataPipeline:
    """Unified pipeline to fetch and merge data from all sources"""

    def __init__(
        self,
        news_api_key: Optional[str] = None,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        reddit_user_agent: Optional[str] = None,
    ):
        """Initialize fetchers"""
        self.news_fetcher = None
        self.reddit_fetcher = None

        if news_api_key or NEWS_API_KEY:
            try:
                self.news_fetcher = NewsAPIFetcher(news_api_key or NEWS_API_KEY)
                logger.info("NewsAPI fetcher initialized")
            except ValueError as e:
                logger.warning(f"NewsAPI not available: {e}")

        if (reddit_client_id or REDDIT_CLIENT_ID) and (
            reddit_client_secret or REDDIT_CLIENT_SECRET
        ):
            try:
                self.reddit_fetcher = RedditFetcher(
                    reddit_client_id or REDDIT_CLIENT_ID,
                    reddit_client_secret or REDDIT_CLIENT_SECRET,
                    reddit_user_agent or REDDIT_USER_AGENT,
                )
                logger.info("Reddit fetcher initialized")
            except (ValueError, PRAWException) as e:
                logger.warning(f"Reddit not available: {e}")

    def build_dataset(
        self,
        fetch_news: bool = True,
        fetch_reddit: bool = True,
        news_days_back: int = 30,
        reddit_limit: int = 100,
    ) -> pd.DataFrame:
        """
        Build unified dataset from all sources

        Args:
            fetch_news: Whether to fetch from NewsAPI
            fetch_reddit: Whether to fetch from Reddit
            news_days_back: Number of days to look back for news
            reddit_limit: Number of posts per subreddit

        Returns:
            Combined DataFrame with all sources
        """
        logger.info("Starting data collection pipeline...")
        dfs = []

        if fetch_news and self.news_fetcher:
            try:
                df_news = self.news_fetcher.fetch(
                    days_back=news_days_back, page_size=100
                )
                if not df_news.empty:
                    dfs.append(df_news)
                    logger.info(f"Added {len(df_news)} articles from NewsAPI")
            except Exception as e:
                logger.error(f"Error fetching news: {e}")

        if fetch_reddit and self.reddit_fetcher:
            try:
                df_reddit = self.reddit_fetcher.fetch(limit=reddit_limit)
                if not df_reddit.empty:
                    dfs.append(df_reddit)
                    logger.info(f"Added {len(df_reddit)} posts from Reddit")
            except Exception as e:
                logger.error(f"Error fetching Reddit: {e}")

        if not dfs:
            logger.warning("No data collected from any source")
            return pd.DataFrame(columns=UNIFIED_COLUMNS)

        df_combined = pd.concat(dfs, ignore_index=True)

        # Sort by timestamp
        df_combined["timestamp"] = pd.to_datetime(df_combined["timestamp"])
        df_combined = df_combined.sort_values("timestamp", ascending=False).reset_index(
            drop=True
        )

        logger.info(
            f"Pipeline complete: {len(df_combined)} total records "
            f"(News: {len([d for d in dfs if 'NewsAPI' in d['source'].values[0]])}, "
            f"Reddit: {len([d for d in dfs if 'Reddit' in d['source'].values[0]])})"
        )

        return df_combined

    @staticmethod
    def save_dataset(df: pd.DataFrame, filepath: str) -> bool:
        """Save dataset to CSV"""
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Dataset saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            return False


# ===========================
# CONVENIENCE FUNCTIONS
# ===========================


def fetch_glp1_data(
    fetch_news: bool = True,
    fetch_reddit: bool = True,
    news_days_back: int = 30,
    reddit_limit: int = 100,
) -> pd.DataFrame:
    """
    Fetch GLP-1 data from all configured sources

    Args:
        fetch_news: Fetch from NewsAPI
        fetch_reddit: Fetch from Reddit
        news_days_back: Days back for news
        reddit_limit: Limit per Reddit subreddit

    Returns:
        Combined DataFrame
    """
    pipeline = GLPDataPipeline()
    return pipeline.build_dataset(
        fetch_news=fetch_news,
        fetch_reddit=fetch_reddit,
        news_days_back=news_days_back,
        reddit_limit=reddit_limit,
    )


def fetch_and_assign_firm_news(
    days_back: int = 30, 
    output_dir: str = "firm_data",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch GLP-1 news and assign each article to firms based on drug/firm mapping
    This approach gets more articles by fetching broadly first, then assigning
    
    Args:
        days_back: Number of days to look back
        output_dir: Directory to save firm CSVs
    
    Returns:
        Dictionary mapping firm names to their DataFrames
    """
    import os
    from pathlib import Path
    from drug_firm_matcher import DrugFirmMatcher
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Fetching GLP-1 news data (last {days_back} days)...")
    
    # Fetch all GLP-1 data using the standard pipeline
    pipeline = GLPDataPipeline()
    df_all = pipeline.build_dataset(fetch_news=True, fetch_reddit=False, news_days_back=days_back)
    
    logger.info(f"Fetched {len(df_all)} total articles from GLP-1 query")
    
    if df_all.empty:
        logger.warning("No articles fetched")
        return {}
    
    # Initialize matcher for drug/firm assignment
    matcher = DrugFirmMatcher()
    
    # Assign firms to each article
    firm_assignments = []
    for idx, row in df_all.iterrows():
        text = row["text"].lower()
        # Check which drugs are mentioned
        mentioned_drugs = []
        for drug_name, aliases in matcher.DRUG_MAP.items():
            for alias in aliases:
                if alias.lower() in text:
                    mentioned_drugs.append(drug_name)
                    break
        
        # Assign firms based on drugs
        assigned_firms = set()
        for drug in mentioned_drugs:
            if drug in matcher.FIRM_MAP:
                assigned_firms.add(matcher.FIRM_MAP[drug])
        
        if assigned_firms:
            firm_assignments.append({
                "idx": idx,
                "firms": list(assigned_firms),
                "drugs": mentioned_drugs
            })
    
    logger.info(f"Assigned firms to {len(firm_assignments)} articles")
    
    # Group articles by firm
    firm_data = {}
    for firm_name in matcher.TOP_10_FIRMS.keys():
        firm_data[firm_name] = []
    
    for assignment in firm_assignments:
        row_idx = assignment["idx"]
        article_row = df_all.iloc[row_idx].copy()
        for firm in assignment["firms"]:
            article_row_copy = article_row.copy()
            article_row_copy["firm"] = firm
            firm_data[firm].append(article_row_copy)
    
    # Convert to DataFrames and save
    firm_dataframes = {}
    for firm_name, articles_list in firm_data.items():
        if articles_list:
            df_firm = pd.DataFrame(articles_list)
            df_firm = df_firm.sort_values("timestamp", ascending=False).reset_index(drop=True)
            df_firm = df_firm.drop_duplicates(subset=['title'], keep='first')
            firm_dataframes[firm_name] = df_firm
            
            # Save to CSV
            filename = firm_name.lower().replace(" ", "_").replace("/", "_") + ".csv"
            filepath = os.path.join(output_dir, filename)
            df_firm.to_csv(filepath, index=False)
            
            logger.info(f"✓ {firm_name}: {len(df_firm)} articles → {filepath}")
        else:
            logger.info(f"✗ {firm_name}: No articles found")
            firm_dataframes[firm_name] = pd.DataFrame()
    
    return firm_dataframes


if __name__ == "__main__":
    # Example usage - fetch GLP-1 general data then assign to firms
    print("="*60)
    print("GLP-1 Real-time Data Ingestion Pipeline")
    print("="*60)
    
    pipeline = GLPDataPipeline()
    
    # Option 1: General GLP-1 data
    print("\nFetching general GLP-1 data...")
    df = pipeline.build_dataset(fetch_news=True, fetch_reddit=False)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Option 2: Firm-specific news (fetches GLP-1 data and assigns firms)
    print("\n" + "="*60)
    print("Fetching GLP-1 data and assigning to firm portfolios...")
    print("="*60)
    firm_data = fetch_and_assign_firm_news(days_back=30, output_dir="firm_data")
    
    print("\nFirm-assigned data saved to firm_data/ directory")
