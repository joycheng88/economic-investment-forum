"""
Configuration for GLP-1 Data Ingestion Pipeline
Handles API credentials and query parameters
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ===========================
# API CONFIGURATION
# ===========================

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "glp1_sentiment_analysis v1.0")

# ===========================
# QUERY CONFIGURATION
# ===========================

# GLP-1 related keywords
GLP1_DRUGS = [
    "GLP-1",
    "semaglutide",
    "tirzepatide",
    "liraglutide",
    "Ozempic",
    "Wegovy",
    "Mounjaro",
    "Zepbound",
]

# Context keywords
CONTEXT_KEYWORDS = [
    "weight loss",
    "obesity",
    "appetite",
    "diabetes",
    "Type 2 diabetes",
    "cardiovascular",
]

# Build dynamic query
def build_query() -> str:
    """Build query string for news and Reddit searches"""
    drug_query = " OR ".join(GLP1_DRUGS)
    context_query = " OR ".join(CONTEXT_KEYWORDS)
    return f"({drug_query}) AND ({context_query})"


# ===========================
# DATA INGESTION PARAMETERS
# ===========================

NEWS_API_URL = "https://newsapi.org/v2/everything"
REDDIT_SUBREDDITS = [
    "health",
    "fitness",
    "diabetes",
    "loseit",
    "EatCheapAndHealthy",
    "nutrition",
]

# API Limits (respect free tier limits)
NEWS_PAGE_SIZE = 100  # Max per request
REDDIT_LIMIT = 100  # Max submissions per search

# Rate limiting delays (seconds)
NEWS_RATE_LIMIT = 1.0  # NewsAPI: 450/day free tier
REDDIT_RATE_LIMIT = 2.0  # Reddit: respectful delay

# ===========================
# DATA VALIDATION
# ===========================

MIN_TEXT_LENGTH = 20  # Minimum characters in text
MAX_TEXT_LENGTH = 10000  # Maximum characters in text
MIN_TITLE_LENGTH = 5

# Language filter
ALLOWED_LANGUAGES = ["en"]

# ===========================
# OUTPUT CONFIGURATION
# ===========================

# Sample column names for unified DataFrame
UNIFIED_COLUMNS = ["text", "title", "source", "timestamp", "url", "author"]

# Data storage
DEFAULT_OUTPUT_PATH = "data_cache/glp1_data.csv"
BACKUP_OUTPUT_PATH = "data_cache/glp1_data_backup.csv"
