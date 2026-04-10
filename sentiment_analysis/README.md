# GLP-1 Real-Time Sentiment Analysis System

A production-ready sentiment analysis system that aggregates real-time market sentiment via Reddit and NewsAPI, trains ensemble ML models, and generates actionable investment signals.

## Core System

### Phase 1: Implementation (Complete)

**ML Pipeline:**
- Text preprocessing (lemmatization, stopword removal)
- TF-IDF feature engineering (5000 features, 1-2 grams) + Word2Vec embeddings
- Weak supervised labeling via VADER sentiment analyzer
- **Logistic Regression**: ~83% accuracy, interpretable
- **Neural Network**: ~81% accuracy, flexible
- **Ensemble**: ~85% accuracy (weighted by F1-score)

**Real-Time Components:**
- `RealTimeDataCollector`: Reddit (4 subreddits) + NewsAPI (4 queries) aggregation
- `RealTimeSentimentMonitor`: Alerts for sentiment shifts, extremes, and model disagreement
- Scheduled collection (APScheduler/Cron/Celery)
- CSV storage with time-series accumulation

### Phase 2: Integration (In Progress)

1. **Real-Time Sentiment Index** - Multi-source aggregation with time-decay weighting
2. **Sentiment-Driven Portfolio** - Dynamic constraints for 8 optimization models
3. **Multi-Factor Scoring** - Composite score (40% Sentiment + 35% Fundamental + 25% Technical)
4. **Forecasting Integration** - ARIMA/GARCH with confidence bands
5. **Risk Management** - VaR, CVaR, stress testing
6. **API & Backend** - FastAPI + PostgreSQL + Redis
7. **Dashboard** - React frontend with real-time updates

## Key Concepts

### VADER Sentiment Analysis
```
compound score: -1.0 (very negative) to +1.0 (very positive)
├─ compound > 0.05  → Positive (label=1)
├─ compound < -0.05 → Negative (label=0)
└─ other            → Neutral (label=2)
```

### TF-IDF Feature Encoding
$$\text{TF-IDF} = \frac{\text{term frequency}}{\text{doc length}} \times \log\left(\frac{\text{total docs}}{\text{docs with term}}\right)$$
Config: max_features=5000, min_df=2, max_df=0.8, ngram_range=(1,2)

### Sentiment Index Formula
$$SI(t) = 100 + 10 \times [0.6 \times P_{pos}(t) - 0.4 \times P_{neg}(t)]$$
- SI > 105: Strongly bullish
- SI 95-105: Neutral
- SI < 95: Strongly bearish

### Model Performance Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Accuracy** | (TP+TN)/Total | Overall correctness |
| **Precision** | TP/(TP+FP) | Positive prediction reliability |
| **Recall** | TP/(TP+FN) | Positive case detection rate |
| **F1-Score** | 2×(P×R)/(P+R) | Precision-recall balance ⭐ primary metric |
| **ROC-AUC** | Area under ROC | Model discrimination ability (0.5-1.0) |

## Quick Start

### 1. Ensemble Prediction
```python
models = load_models('./models')
text = "GLP-1 shows promising clinical results"
result = predict_sentiment_ensemble(text, models)
print(f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']:.2%}")
```

### 2. Calculate Daily Index
```python
daily_sentiment = df.groupby(df['date'].dt.date)['prob_positive'].mean()
SI = 100 + 10 * (0.6 * daily_sentiment - 0.4 * (1 - daily_sentiment))
```

### 3. Load & Save Models
```python
import joblib
from tensorflow.keras.models import load_model
lr_model = joblib.load('lr_model.joblib')
nn_model = load_model('nn_model.h5')
```

## Data Preprocessing

```
Raw Text → Lowercase/Clean URLs → Tokenize → Remove Stopwords 
         → Lemmatize → Encode (TF-IDF or Embedding)
```
**Validation**: 10-5000 characters per document

## Real-Time Data Sources

### Reddit (Free)
- Subreddits: r/GLP1, r/diabetes, r/WeightLoss, r/Ozempic
- Rate: 60 req/min
- Setup: reddit.com/prefs/apps → create app → save credentials

### NewsAPI (Free tier: 500 req/day)
- Queries: "GLP-1", "Ozempic", "diabetes treatment", "weight loss drug"
- Updates: 15-min intervals
- Setup: newsapi.org → register → save API key

## Model Architectures

**Logistic Regression:**
```python
LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, solver='lbfgs')
```

**Neural Network:**
```python
Sequential([Embedding(5000, 100), GlobalAveragePooling1D(), 
           Dense(128, activation='relu'), Dropout(0.2),
           Dense(64, activation='relu'), Dropout(0.2), Dense(1, sigmoid)])
```

**Ensemble:**
```
prob_ensemble = w_lr × prob_lr + w_nn × prob_nn  (w normalized from F1-scores)
```

## System Architecture

```
Reddit/NewsAPI → RealTimeDataCollector → Preprocessing 
               → Ensemble Model (LR + NN) → RealTimeSentimentMonitor
               → Storage (CSV) + Alerts
```

## Notebook Structure

| Section | Content |
|---------|---------|
| 0-2 | Setup & initialization |
| 3-5 | Data prep, preprocessing, features |
| 6-10 | Training, evaluation, ensemble |
| 11-14 | Index, signals, visualization |
| 15+ | Real-time pipeline (NEW) |

**Section 15 additions:**
- RealTimeDataCollector (950+ lines)
- Configuration templates
- End-to-end demo & visualization
- RealTimeSentimentMonitor with alerts
- Deployment checklist (4 phases)

## Performance Baseline

| Model | Accuracy | F1 | ROC-AUC | Speed |
|-------|----------|----|---------|----|
| Logistic Reg | 83% | 0.85 | 0.91 | Very fast |
| Neural Net | 81% | 0.87 | 0.93 | Fast |
| **Ensemble** | **85%** | **0.88** | **0.94** | Fast |

## Troubleshooting

**F1 < 0.70:** Check data quality (nulls, length), VADER thresholds, increase features  
**Overfitting:** Increase Dropout (0.3-0.4), reduce Dense units, increase L2 regularization  
**Slow inference:** Reduce TF-IDF dims, simplify model, use GPU/batching

## File Structure
```
sentiment_analysis/
├── EEIF_Quant_Sentiment_Analysis.ipynb  # Main notebook
├── GLP1_Sentiment_Analysis.ipynb         # Alternative
├── models/
│   ├── lr_model.joblib
│   ├── nn_model.h5
│   ├── tfidf_vectorizer.joblib
│   └── model_metadata.json
├── data_cache/                           # Real-time storage
└── README.md
```

## Integration: Sentiment → Portfolio

**Constraint Generation:** Sentiment extremes → min/max weight adjustments  
**Multi-Factor Score:** 40% Sentiment + 35% Fundamental + 25% Technical  
**8 Optimization Models:** LASSO, DRO, HRP, Bayesian, CVaR, GMV, CAPM, RL

## Next Steps (2-3 weeks each)

1. **Sentiment Index Enhancement** - Add Twitter, time-decay, PostgreSQL
2. **Portfolio Integration** - Constraint generator, backtesting
3. **Risk System** - VaR, CVaR, stress testing, alerts
4. **API & Dashboard** - FastAPI backend, React frontend (<200ms response)

## Success KPIs

- **Technical:** API <200ms response, <30min lag, Model F1>0.85, 99.5% uptime
- **Business:** Index correlation >0.60 with returns, Sharpe >1.5, Alert precision >90%

## Deployment Options

1. **APScheduler** (Dev): Simple background scheduler
2. **Cron** (Prod): OS-level scheduling
3. **Celery+Redis** (Enterprise): Distributed scalability

## Tech Stack

Python 3.8+ • scikit-learn • TensorFlow • NLTK • pandas • numpy • PRAW • newsapi  
Future: PostgreSQL • TimescaleDB • Redis • FastAPI • React • Docker

---

**Status:** Production Ready (Phase 1) + Phase 2 Integration Underway  
**Maintenance:** Weekly updates, monthly performance reviews  
**Last Updated:** April 2026
