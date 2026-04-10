# GLP-1 Complete Analysis Pipeline: Data Ingestion to Sentiment Index

Complete end-to-end workflow for collecting GLP-1 content, matching to drugs/firms, and building sentiment indices for econometric analysis.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GLP-1 SENTIMENT ANALYSIS PIPELINE                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  STEP 1: DATA INGESTION                    STEP 2: DRUG-FIRM MATCHING       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • NewsAPI (450 req/day)                   • Regex-based drug tagging       │
│  • Reddit PRAW (all subreddits)            • Expand multi-drug documents    │
│  • Unified DataFrame output                • Map to top 10 firms            │
│  │                                         │                                │
│  └─→ glp1_data.csv                        └─→ glp1_matched_documents.csv  │
│      (text, title, source, timestamp, url)   (+ drugs, firm columns)       │
│                                                                               │
│  STEP 3: SENTIMENT EXTRACTION              STEP 4: FIRM-LEVEL INDEX       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Positive keyword counts                 • Aggregate by firm             │
│  • Negative keyword counts                 • Calculate composite index     │
│  • Clinical data indicators                • Output time series ready      │
│  │                                         │                                │
│  └─→ Added columns (doc level)            └─→ glp1_sentiment_index.csv   │
│      (positive_count, negative_count,        (article_count, avg_sentiment,│
│       sentiment_net, has_clinical_data)       clinical_ratio, sentiment_idx)│
│                                                                               │
│  STEP 5: ECONOMETRIC ANALYSIS                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Time series analysis                                                     │
│  • Merge with stock returns                                                │
│  • Panel data models (fixed/random effects)                                │
│  • Event study: trial announcements → sentiment → returns                  │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## File Structure

```
sentiment_analysis/
├── config.py                        # API config + query parameters
├── data_ingestion.py               # NewsAPI + PRAW fetchers
├── data_utils.py                   # Data persistence + management
├── drug_firm_matcher.py            # Core matching logic (module)
├── GLP1_Drug_Firm_Matching.ipynb  # Interactive matching notebook
├── DATAPIPELINE_README.md          # Data ingestion documentation
├── requirements.txt                # Dependencies
├── .env.template                   # API credentials template
│
├── data_cache/
│   ├── glp1_data.csv              # Raw ingested data
│   ├── glp1_data_backup.csv       # Backup
│   └── glp1_matched_documents.csv # Matched with drug/firm/sentiment
│
└── outputs/
    ├── glp1_sentiment_index.csv   # Firm-level indices
    ├── sentiment_analysis_report.txt
    └── visualizations/            # Charts and plots
```

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API credentials
cp .env.template .env
# Edit .env with your NewsAPI key and Reddit credentials
```

### 2. Fetch Data

```python
from data_ingestion import fetch_glp1_data
from data_utils import DataManager

# Fetch fresh data
df = fetch_glp1_data(fetch_news=True, fetch_reddit=True)

# Merge with existing data and save
manager = DataManager()
df_merged = manager.update(df, dedup=True)
manager.save(df_merged, backup=True)
```

### 3. Match Drugs to Firms

```python
from drug_firm_matcher import simple_match_and_score

# Load data
df = pd.read_csv('data_cache/glp1_data.csv')

# Run matching pipeline
df_matched, sentiment_index = simple_match_and_score(df)

# Save results
df_matched.to_csv('data_cache/glp1_matched_documents.csv', index=False)
sentiment_index.to_csv('outputs/glp1_sentiment_index.csv')
```

### 4. Open Interactive Notebook

```bash
jupyter notebook GLP1_Drug_Firm_Matching.ipynb
```

## Module Reference

### data_ingestion.py

**Classes:**
- `NewsAPIFetcher`: Fetch articles from NewsAPI
- `RedditFetcher`: Fetch posts from Reddit using PRAW
- `GLPDataPipeline`: Unified pipeline combining all sources

**Key Functions:**
```python
fetch_glp1_data(fetch_news=True, fetch_reddit=True, news_days_back=30, reddit_limit=100)
# Returns: Combined DataFrame with all documents
```

### drug_firm_matcher.py

**Classes:**
- `DrugFirmMatcher`: Core matching and scoring logic

**Key Methods:**
```python
matcher = DrugFirmMatcher()

# Tag documents with drugs
matcher.tag_drug(text)  # Returns: List[str] - drug names

# Complete pipeline
df_matched, sentiment_index = simple_match_and_score(df)
```

### data_utils.py

**Classes:**
- `DataManager`: Handle data persistence, updates, deduplication
- `ScheduleHelper`: Examples for automated collection

**Key Methods:**
```python
manager = DataManager()
df_merged = manager.update(new_df, dedup=True)
manager.save(df_merged, backup=True)
stats = manager.get_stats()
```

## Data Dictionary

### Raw Ingested Data (glp1_data.csv)

| Column | Type | Description |
|--------|------|-------------|
| text | str | Combined title + description |
| title | str | Article/post title |
| source | str | "NewsAPI" or "Reddit (r/subreddit)" |
| timestamp | datetime | Publication time |
| url | str | Link to original |
| author | str | Author/poster name |

### Matched Documents (glp1_matched_documents.csv)

**Includes all above +:**

| Column | Type | Description |
|--------|------|-------------|
| drugs | str | Canonical drug name (expanded rows) |
| firm | str | Manufacturer from top 10 |
| positive_count | int | Positive keyword matches |
| negative_count | int | Negative keyword matches |
| sentiment_net | int | positive_count - negative_count |
| has_clinical_data | bool | Contains trial/efficacy mentions |
| text_length | int | Character count |

### Sentiment Index (glp1_sentiment_index.csv)

| Column | Type | Description |
|--------|------|-------------|
| firm | str | Pharmaceutical firm name |
| article_count | int | Total articles mentioning firm |
| sentiment_net_mean | float | Avg sentiment (-1 to +1 scale) |
| positive_ratio | float | % articles with positive tone |
| clinical_articles | int | # articles with clinical data |
| clinical_ratio | float | Clinical articles / total |
| sentiment_index | float | Composite 0-100 index |

## Top 10 Firms

1. **Novo Nordisk A/S** - Market leader (Ozempic, Wegovy)
2. **Eli Lilly and Company** - Fast-growth competitor (Mounjaro, Zepbound)
3. **Amgen Inc.** - Novel MariTide
4. **Pfizer Inc.** - Oral formulation (danuglipron)
5. **Roche Holding AG** - Carmot acquisition (CT-388, CT-996)
6. **AstraZeneca Plc** - Small-molecule oral candidates
7. **Zealand Pharma A/S** - M&A target
8. **Structure Therapeutics, Inc.** - Oral specialization
9. **Viking Therapeutics, Inc.** - VK2735
10. **Boehringer Ingelheim GmbH** - Cardiometabolic focus

## Sentiment Measurement

### Keyword-Based Approach

**Positive indicators:**
```
"promising", "success", "breakthrough", "approved", "superior", "outperform", 
"momentum", "expansion", "growth", "advance"
```

**Negative indicators:**
```
"setback", "adverse", "risk", "delay", "weak", "competition", "challenge", 
"decline", "loss"
```

**Clinical indicators:**
```
"trial", "clinical", "phase", "FDA", "efficacy", "safety", "results"
```

### Composite Index

```
Sentiment Index = (0.4 × avg_net_sentiment) + (0.3 × positive_ratio) + (0.3 × clinical_ratio)
```

## Workflow Examples

### Example 1: Weekly Sentiment Tracking

```python
import pandas as pd

# Load matched documents
df = pd.read_csv('data_cache/glp1_matched_documents.csv', parse_dates=['timestamp'])

# Group by firm and week
df['week'] = df['timestamp'].dt.isocalendar().week
weekly_sentiment = df.groupby(['firm', 'week']).agg({
    'sentiment_net': 'mean',
    'text': 'count',
    'has_clinical_data': 'mean'
}).rename(columns={'text': 'article_count'})

# Visualize
weekly_sentiment.unstack('firm')['article_count'].plot(figsize=(14, 6))
plt.show()
```

### Example 2: Event Study Analysis

```python
# Event: Clinical trial announcement for Novo Nordisk on date X
event_date = pd.Timestamp('2026-03-15')
window_days = 30

df['days_to_event'] = (df['timestamp'] - event_date).dt.days

# Filter to event window
event_window = df[df['days_to_event'].between(-window_days, window_days)]

# Compare pre vs post
pre = event_window[event_window['days_to_event'] < 0]['sentiment_net'].mean()
post = event_window[event_window['days_to_event'] >= 0]['sentiment_net'].mean()
diff = post - pre

print(f"Pre-event sentiment: {pre:.2f}")
print(f"Post-event sentiment: {post:.2f}")
print(f"Sentiment shock: {diff:.2f}")
```

### Example 3: Firm Comparison

```python
# Load sentiment index
idx = pd.read_csv('outputs/glp1_sentiment_index.csv', index_col='firm')

# Categorize by sentiment
high_sentiment = idx[idx['sentiment_index'] > 60]
low_sentiment = idx[idx['sentiment_index'] < 40]

print("High sentiment firms:")
print(high_sentiment[['article_count', 'sentiment_index']])

print("\nLow sentiment firms:")
print(low_sentiment[['article_count', 'sentiment_index']])
```

### Example 4: Panel Data Model

```python
# Time-firm panel with rolling averages
df['date'] = df['timestamp'].dt.date
daily_sentiment = df.groupby(['date', 'firm']).agg({
    'sentiment_net': 'mean',
    'text': 'count',
    'has_clinical_data': 'mean'
}).rename(columns={'text': 'volume'}).reset_index()

# Create rolling 7-day average
daily_sentiment['sentiment_7d'] = daily_sentiment.groupby('firm')['sentiment_net'].transform(
    lambda x: x.rolling(window=7, center=True).mean()
)

# Merge with stock returns (assuming returns_df available)
panel = daily_sentiment.merge(
    returns_df,
    left_on=['date', 'firm'], 
    right_on=['date', 'ticker'],
    how='inner'
)

# Fixed effects model
from linearmodels.panel import FirstDifferenceOLS

# Set multi-index
panel = panel.set_index(['firm', 'date'])

# Run FD model
model = FirstDifferenceOLS(panel['stock_return'], panel[['sentiment_net', 'volume']])
results = model.fit()
print(results.summary)
```

## Integration with Existing GLP-1 Projects

### Merge with sentiment_analysis_firms Panel Data

```python
# Load this pipeline's sentiment index
glp1_sentiment = pd.read_csv('outputs/glp1_sentiment_index.csv')

# Load existing GLP-1 firms sentiment panel
existing_panel = pd.read_csv(
    '../sentiment_analysis_firms/outputs/firm_week_panel.csv'
)

# The datasets can be merged on firm name and time period
merged = pd.merge(
    existing_panel,
    glp1_sentiment,
    on='firm',
    how='left'
)
```

### Use as Instrumental Variable

Sentiment index from media mentions can serve as exogenous instrument for:
- Stock return regressions
- IV models with endogenous fundamental news
- Granger causality: sentiment → price movements

## Performance Considerations

**Typical Runtime:**
- Data ingestion (news + reddit): 15-20 seconds
- Drug-firm matching: <1 second (vectorized)
- Sentiment extraction: <1 second
- Index calculation: <100ms
- Total end-to-end: ~20 seconds

**Data Volume:**
- NewsAPI: 30-50 articles/day
- Reddit: 50-100 posts/search
- After filtering: 70-80 relevant documents/day
- Monthly dataset: ~2,000-2,500 documents

## Troubleshooting

### "No data returned from NewsAPI"

Check:
1. `.env` file has valid `NEWS_API_KEY`
2. API key allows "everything" endpoint
3. Query terms match available content
4. Rate limit not exceeded (450 req/day)

### "Invalid Reddit credentials"

Check:
1. App registered at reddit.com/prefs/apps (type: "script")
2. Correct `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`
3. Credentials match your app, not user password

### "Sentiment index very low/high"

Likely causes:
- Keywords not well-calibrated for finance domain
- Consider switching to FinBERT for financial sentiment
- Add domain-specific keywords

### "No match after firm filtering"

Check:
- Article text mentions specific drug names (not generic "GLP-1")
- Drug names spelled correctly
- Check stemming if using advanced NLP

## Next Steps

1. **FinBERT Integration**: Replace keyword-based sentiment with fine-tuned FinBERT model
2. **Time Series Modeling**: ARIMA/GARCH for sentiment volatility
3. **Stock Return Integration**: Correlation/causality tests with firm stock prices
4. **Real-time Dashboard**: Streamlit app for monitoring sentiment index
5. **Database Storage**: Migrate from CSV to PostgreSQL for scalability

## References

**Papers & Resources:**
- Tetlock (2007): "Giving Content to Investor Sentiment"
- Gentzkow et al. (2019): "Text as Data" Handbook chapter
- FinBERT: Financial Sentiment Analysis with Pre-trained Language Models

**Tools & APIs:**
- NewsAPI: https://newsapi.org
- PRAW: https://praw.readthedocs.io
- VADER: https://github.com/cjhutto/vaderSentimentAnalysis
- FinBERT: https://github.com/ProsusAI/finBERT

---

**Last Updated:** April 9, 2026
**Maintainer:** EEIF Sentiment Analysis Team
