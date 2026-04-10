# Scripts Reference

Main analysis scripts for sentiment pipeline. All scripts import from `src/` modules.

## Quick Start

```bash
# Full pipeline (collect → preprocess → filter → sentiment → aggregate)
python scripts/run_full_pipeline_live.py

# VADER analysis & FinBERT comparison
python scripts/recompute_vader_compare.py

# Sentiment visualization & firm rankings
python scripts/plot_sentiment_trends.py

# Econometric analysis (DiD regression)
python scripts/did_regression.py

# Panel workflow example
python scripts/panel_workflow.py
```

## Script Details

### 1. run_full_pipeline_live.py
**Purpose:** Master orchestration script (collection → analysis)

**Steps:**
1. Collect news via gnews API (170 firm-keyword searches)
2. Preprocess text (clean, normalize, stem)
3. Filter for GLP-1 relevance (firm name + keyword)
4. Analyze sentiment with VADER
5. Create weekly indices (market, firm, keyword)

**Outputs:**
- `data/raw/news_data_real.csv`
- `data/processed/news_data_cleaned_real.csv`
- `data/processed/glp1_relevant_real.csv`
- `data/processed/articles_with_sentiment_real.csv`
- `outputs/weekly_*_sentiment_index_real.csv`

**Runtime:** ~45-60 minutes (rate-limited API)

### 2. recompute_vader_compare.py
**Purpose:** VADER recomputation + FinBERT correlation study

**Does:**
- Recomputes VADER scores on all articles
- Generates comparison framework with FinBERT
- Computes correlations (Pearson, Spearman, Kendall)
- Creates visualizations (4-panel chart)
- Outputs statistical summary

**Outputs:**
- `outputs/vader_finbert_comparison.csv` - Side-by-side scores
- `outputs/vader_finbert_comparison.png` - Visualization
- `outputs/correlation_summary.txt` - Statistics

**Runtime:** ~2 seconds

**Key Results:**
- Label agreement: 91%
- Correlation: r = +0.598 (p=0.052)
- RMSE: 0.252

### 3. plot_sentiment_trends.py
**Purpose:** Sentiment trends visualization + firm rankings

**Does:**
- Loads panel data from sentiment index
- Creates time series plot (10 firms)
- Computes firm-level rankings
- Generates rankings table
- Exports to CSV

**Outputs:**
- `outputs/sentiment_trends.png` - Time series (14x8", 300 DPI)
- `outputs/firm_sentiment_rankings.csv` - Rankings table

**Runtime:** ~1 second

**Key Metrics:**
- Top firm: Wonderful (z = +2.27)
- Bottom firm: PepsiCo (z = -0.53)
- Observation period: 3 weeks (April 2026)

### 4. did_regression.py
**Purpose:** Difference-in-Differences econometric analysis

**Does:**
- Loads panel data
- Creates treatment variables (exposed, post_event, interaction)
- Runs OLS with firm + week fixed effects
- Outputs regression results
- Interprets treatment effect

**Outputs:**
- `outputs/DID_ANALYSIS_RESULTS.txt` - Regression summary
- Console output: Treatment assignment, sample data, results

**Specification:**
```
y_it = α_i + γ_t + β(exposed × post) + ε_it
```

**Runtime:** ~1 second

**Key Result:**
- Treatment effect: β = -0.328 (treated firms more negative)

### 5. panel_workflow.py
**Purpose:** Complete panel data workflow example

**Does:**
- Creates firm-week panel from sentiment articles
- Computes rolling 4-week averages
- Standardizes within-week z-scores
- Outputs panel data for econometric use

**Outputs:**
- Panel data CSV (ready for regression)

**Runtime:** ~1 second

---

## Module Dependencies

All scripts depend on `src/` modules:

```
src/
├── data_collection/       # gnews API collection
├── preprocessing/         # Text cleaning
├── filtering/             # GLP-1 relevance
├── sentiment/             # VADER & FinBERT
├── aggregation/           # Weekly indices
└── analysis/              # Panel data, DiD
```

---

## Execution Examples

### Minimal (Demo Only)
```bash
python scripts/plot_sentiment_trends.py    # 1 sec
python scripts/recompute_vader_compare.py  # 2 sec
python scripts/did_regression.py           # 1 sec
# Total: <5 seconds
```

### Full Analysis
```bash
python scripts/run_full_pipeline_live.py   # 45-60 min (collection)
python scripts/plot_sentiment_trends.py    # 1 sec
python scripts/recompute_vader_compare.py  # 2 sec
python scripts/did_regression.py           # 1 sec
# Total: ~1 hour
```

---

## Error Handling

**Collection Timeout:** Retry with smaller date range
**Memory Issues:** Reduce batch size parameters in scripts
**GPU Not Found:** CPU fallback enabled automatically
**API Rate Limit:** Normal, add delay between requests

---

**Location:** `/scripts/`  
**Total Scripts:** 5 main, 5 demo  
**Execution:** Sequential (not parallelizable due to data dependencies)
