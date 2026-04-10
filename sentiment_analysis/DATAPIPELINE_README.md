# GLP-1 Real-Time Data Ingestion Pipeline

A production-ready Python pipeline for collecting GLP-1 (Glucagon-Like Peptide-1) related content from multiple sources: **NewsAPI** and **Reddit (PRAW)**.

## Features

✅ **Multi-Source Data Collection**
- NewsAPI for news articles (450 requests/day free tier)
- Reddit via PRAW for forum discussions
- Configurable query terms focusing on GLP-1 drugs and weight loss context

✅ **Production-Ready**
- Comprehensive error handling and logging
- Rate limiting to respect API limits
- Data validation (text length, quality checks)
- Unified DataFrame output with consistent schema

✅ **Easy Configuration**
- Environment-based API credentials (.env file)
- Modular design with separate fetchers
- Configurable parameters (query, limits, filters)

✅ **Data Quality**
- Text cleaning and validation
- Duplicate handling
- Timestamp standardization
- URL and author tracking

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Credentials

**Copy the template and add your credentials:**

```bash
cp .env.template .env
```

**Edit `.env` with your API keys:**

#### NewsAPI
1. Get free API key: https://newsapi.org
2. Free tier: 450 requests/day
3. Add to `.env`:
```
NEWS_API_KEY=your_api_key_here
```

#### Reddit (PRAW)
1. Register app at: https://www.reddit.com/prefs/apps
2. Select app type: **script**
3. You'll get `client_id` and `client_secret`
4. Add to `.env`:
```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=glp1_sentiment_analysis_v1.0
```

### 3. Verify Setup

```bash
python data_ingestion.py
```

## Quick Start

### Basic Usage

```python
from data_ingestion import fetch_glp1_data

# Fetch from all sources
df = fetch_glp1_data()
print(f"Fetched {len(df)} records")
print(df.head())
```

### Advanced Usage

```python
from data_ingestion import GLPDataPipeline

# Initialize pipeline
pipeline = GLPDataPipeline()

# Fetch with custom parameters
df = pipeline.build_dataset(
    fetch_news=True,
    fetch_reddit=True,
    news_days_back=7,      # Last 7 days of news
    reddit_limit=50         # 50 posts per subreddit
)

# Save to CSV
pipeline.save_dataset(df, "glp1_data.csv")
```

## Data Output

The unified DataFrame contains:

| Column | Type | Description |
|--------|------|-------------|
| `text` | str | Combined title + description/content |
| `title` | str | Article title or post title |
| `source` | str | Source identifier (NewsAPI, Reddit) |
| `timestamp` | datetime | Publication/submission time |
| `url` | str | Link to original content |
| `author` | str | Author or poster name |

**Example Output:**
```
                                                  text          title  source                 timestamp  
GLP-1 drugs show promise in cardiovascular...  GLP-1 drugs...  NewsAPI  2026-04-09 14:30:00
Started Ozempic last month and feeling great  Started Ozempic...  Reddit...  2026-04-09 12:15:00
```

## Configuration

Edit `config.py` to customize:

- **Drug names**: `GLP1_DRUGS` - Add/remove drug names
- **Context keywords**: `CONTEXT_KEYWORDS` - Topic focus
- **Reddit subreddits**: `REDDIT_SUBREDDITS` - Communities to search
- **Data validation**: `MIN_TEXT_LENGTH`, `MAX_TEXT_LENGTH`
- **Rate limits**: `NEWS_RATE_LIMIT`, `REDDIT_RATE_LIMIT`

### Dynamic Query Example

```python
from config import build_query
query = build_query()
print(query)
# Output: (GLP-1 OR semaglutide OR tirzepatide OR ...) AND (weight loss OR obesity OR ...)
```

## Examples

Run interactive examples:

```bash
python example_usage.py
```

Includes:
1. Basic fetch from all sources
2. NewsAPI only
3. Reddit only
4. Data analysis (counts, distributions, text stats)
5. Save and load data
6. Incremental collection strategy
7. Filtering and subsetting

## API Rate Limits

| Service | Free Tier Limit | Rate Limiting |
|---------|-----------------|---------------|
| NewsAPI | 450 req/day | 1 sec between requests |
| Reddit | No hard limit | 2 sec between requests |

The pipeline respects these limits automatically.

## Logging

All operations are logged to:
- **Console**: Real-time status
- **File**: `data_ingestion.log` (persistent)

Example log output:
```
2026-04-09 14:30:15 - root - INFO - NewsAPI initialized with query: (GLP-1 OR semaglutide...) AND ...
2026-04-09 14:30:20 - root - INFO - Found 87 articles
2026-04-09 14:30:20 - root - INFO - Validated 85 articles from NewsAPI
2026-04-09 14:30:22 - root - INFO - Searching r/health...
```

## Error Handling

The pipeline handles common errors gracefully:

- **Missing API credentials**: Skips that source with warning
- **Network timeouts**: Logs error and continues
- **Invalid data**: Filters with quality checks
- **Rate limits**: Respects delays
- **Reddit auth failures**: Clear error message

## Data Caching

Default output location:
```
data_cache/glp1_data.csv
```

Backup created at:
```
data_cache/glp1_data_backup.csv
```

## Integration with Sentiment Analysis

The output DataFrame works directly with sentiment analysis:

```python
import pandas as pd
from vader import SentimentIntensityAnalyzer
from data_ingestion import fetch_glp1_data

# Fetch data
df = fetch_glp1_data()

# Add sentiment
analyzer = SentimentIntensityAnalyzer()
df['sentiment'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Analyze
print(df.groupby('source')['sentiment'].mean())
```

## Troubleshooting

### "NEWS_API_KEY not configured"
- Check your `.env` file exists
- Verify `NEWS_API_KEY=<your_key>` is set
- Restart Python/notebook after creating .env

### "Invalid Reddit credentials"
- Verify correct `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`
- Check app type is "script" (not "web app")
- Ensure credentials match your registered app

### "No data collected"
- Verify internet connection
- Check API keys are valid
- Ensure query keywords match available content
- Try with specific subreddit, e.g., `reddit.subreddit("loseit")`

### Rate Limit Errors
- Reduce `pageSize` in NewsAPI config
- Reduce `reddit_limit` parameter
- Increase rate limit delays in config

## File Structure

```
sentiment_analysis/
├── config.py                    # Configuration management
├── data_ingestion.py           # Main pipeline module
├── example_usage.py            # Usage examples
├── requirements.txt            # Python dependencies
├── .env.template               # Environment template
├── data_ingestion.log          # Logging output
├── data_cache/
│   ├── glp1_data.csv          # Latest data
│   └── glp1_data_backup.csv   # Backup
└── README.md                   # This file
```

## API Credits & Limits

### NewsAPI
- **Free tier**: 450 requests/day
- **Registration**: https://newsapi.org
- **Rate limiting**: 1 request/second (1000/month estimated)

### Reddit (PRAW)
- **Rate limiting**: Per-user, ~60 requests/minute
- **Registration**: https://www.reddit.com/prefs/apps
- **Documentation**: https://praw.readthedocs.io/

## Performance

Typical run times (from cold start):

| Operation | Time |
|-----------|------|
| Fetch 100 news articles | 2-3 seconds |
| Fetch 500 Reddit posts | 10-15 seconds (multiple subreddits) |
| Combined fetch (all sources) | 15-20 seconds |
| Save 1000 records to CSV | <1 second |

## Next Steps

1. **Sentiment Analysis**: Use output with VADER or FinBERT
2. **Aggregation**: Weekly/daily sentiment indices
3. **Monitoring**: Set up automated daily collection
4. **Dashboard**: Visualize trends over time
5. **Storage**: Migrate to database for persistence

## Contributing

To extend the pipeline:
- **Add new sources**: Extend `GLPDataPipeline` with new fetcher classes
- **Custom queries**: Modify `config.py` query terms
- **New filters**: Add validation in `_is_valid()` methods
- **Data enrichment**: Add fields in standardized format

## License

Part of the Emory Economic Investment Forum project.

## Support

For issues or questions:
- Check logs: `cat data_ingestion.log`
- Review examples: `python example_usage.py`
- Verify credentials: `echo $NEWS_API_KEY`

---

**Last Updated**: April 9, 2026
**Python Version**: 3.8+
**Maintainer**: EEIF Team
