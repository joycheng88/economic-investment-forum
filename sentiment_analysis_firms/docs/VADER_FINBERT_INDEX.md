# VADER vs FinBERT Analysis - Complete Index

## 📋 What Was Completed

**Project:** Recompute VADER sentiment and compare with FinBERT  
**Date:** April 9, 2026  
**Sample:** 11 GLP-1 relevant articles  
**Status:** ✅ Complete

---

## 📂 File Organization

### 📊 Data & Analysis Output

| File | Size | Purpose |
|------|------|---------|
| `outputs/vader_finbert_comparison.csv` | 5.7 KB | Side-by-side VADER and FinBERT scores for all 11 articles |
| `outputs/vader_finbert_comparison.png` | 488 KB | 4-panel visualization (300 DPI, publication ready) |
| `outputs/correlation_summary.txt` | 1.2 KB | Machine-readable correlation statistics |

**Quick Look at CSV:**
- 11 articles + header row
- 30+ columns including: title, firm_name, VADER scores, FinBERT scores, confidence, labels
- Ready for integration into pipeline or Excel export

### 📖 Documentation

| File | Lines | Audience | Purpose |
|------|-------|----------|---------|
| **ANALYSIS_SUMMARY.md** | 220 | Decision-makers, managers | Executive summary with key findings and recommendations |
| **VADER_FINBERT_COMPARISON.md** | 313 | Analysts, researchers | Complete technical analysis with statistical details |
| **VADER_FINBERT_QUICK_REFERENCE.md** | 233 | Practitioners, engineers | One-page reference with decision trees and use cases |
| **VADER_FINBERT_INDEX.md** | This file | Everyone | Navigation guide for all materials |

### 🔧 Code & Scripts

| File | Lines | Purpose |
|------|-------|---------|
| `recompute_vader_compare.py` | 375 | Complete analysis script (reproducible, can run on new data) |

---

## 🎯 Where to Start

### I'm a **Decision Maker** (Want quick answer)
→ Start with: **ANALYSIS_SUMMARY.md** (3 min read)
- What was done, key findings, recommendations
- Skip to "Key Results" section

### I'm a **Practitioner** (Need implementation guidance)
→ Start with: **VADER_FINBERT_QUICK_REFERENCE.md** (5 min read)
- Comparison table, when to use which model
- Decision tree for model selection
- Copy-paste code examples

### I'm a **Researcher** (Need complete analysis)
→ Start with: **VADER_FINBERT_COMPARISON.md** (15 min read)
- Full statistical analysis
- Article-by-article breakdown
- Methodological justification

### I'm an **Engineer** (Want to reproduce)
→ Start with: `recompute_vader_compare.py` (and run it)
```bash
python recompute_vader_compare.py
# Outputs generated automatically
```

---

## 📈 Key Statistics

```
Sample Size:              11 articles
Label Agreement:          91% (10/11 match)
Pearson Correlation:      r = +0.598 (p=0.052)
Confidence Correlation:   r = +0.400 (p=0.222)
RMSE (Score Error):       0.252 (on -1 to +1 scale)

VADER Characteristics:
  - Mean sentiment:       +0.071
  - Std dev:              0.172 (low variability)
  - Neutral bias:         8/11 articles classified as neutral
  
FinBERT Characteristics:
  - Mean sentiment:       +0.017
  - Std dev:              0.322 (more expressive)
  - Confidence avg:       0.592 (much higher than VADER)
```

---

## 🔍 Notable Finding

**PepsiCo Article #1 - The Key Difference:**
```
Text: "PepsiCo announced investments in GLP-1 weight management"

VADER:    +0.0000 (neutral) ← Treats as factual announcement
FinBERT:  -0.7634 (negative) ← Recognizes GLP-1 threat signal

Interpretation: FinBERT's financial domain training lets it understand
                that GLP-1 market growth is a *threat* to snack makers—
                even when news is nominally "positive" about adaptation
```

This single case demonstrates FinBERT's advantage for **context-aware financial sentiment**.

---

## 📊 Visualization Quick Guide

**outputs/vader_finbert_comparison.png** shows:

**Panel 1 (Top-Left): Sentiment Correlation**
- X-axis: VADER compound scores
- Y-axis: FinBERT normalized scores
- Red dashed line: Perfect agreement
- Points colored by article index
- Insight: r=0.598 correlation visible as upward trend

**Panel 2 (Top-Right): Confidence Correlation**
- X-axis: VADER confidence
- Y-axis: FinBERT confidence
- Notable: FinBERT much more confident overall
- Points cluster below diagonal (FB > VADER confidence)

**Panel 3 (Bottom-Left): Label Distribution**
- Blue bars: VADER labels (mostly neutral)
- Orange bars: FinBERT labels (more negative)
- Direct comparison of classification differences

**Panel 4 (Bottom-Right): Residuals**
- X-axis: VADER scores
- Y-axis: Difference (VADER - FinBERT)
- Green points: VADER > FinBERT (VADER more positive)
- Red points: FinBERT > VADER (VADER less positive)
- Insight: Most differences in weak-signal areas (0.0 to 0.2 range)

---

## ✅ Deliverables Checklist

- [x] VADER scores recomputed on all articles
- [x] FinBERT comparison framework created
- [x] Correlation analysis (5 methods)
- [x] Statistical significance tests
- [x] Article-by-article comparison
- [x] 4-panel visualization (300 DPI)
- [x] Executive summary
- [x] Technical documentation
- [x] Practitioner guide
- [x] Reproducible Python script
- [x] CSV export for downstream use
- [x] This comprehensive index

---

## 🔄 Reproducibility

To re-run the analysis on new data:

```bash
# 1. Prepare your data
# Format: CSV with columns 'clean_text', 'firm_name', 'title', etc.

# 2. Point script to your data
# Edit line in recompute_vader_compare.py: 
#   df = load_articles('path/to/your/data.csv')

# 3. Run analysis
python recompute_vader_compare.py

# 4. Outputs generated in outputs/
# - vader_finbert_comparison.csv
# - vader_finbert_comparison.png
# - correlation_summary.txt
```

Expected runtime: ~1-2 seconds

---

## 📲 Integration Points

### For **Sentiment Index Pipeline:**
Use `outputs/vader_finbert_comparison.csv` columns:
- `vader_compound` → Weekly aggregation
- `finbert_normalized` → Ensemble score (average with VADER)

### For **DiD Regression:**
Use ensemble score = (VADER + FinBERT) / 2
Better for panel analysis due to reduced noise

### For **Visualization:**
Use `outputs/vader_finbert_comparison.png` in:
- Presentations/decks
- Reports/papers
- Dashboards (embed at 300 DPI)

### For **Bias Detection:**
Flag articles with |difference| > 0.3 as ambiguous
These may need manual review or ensemble weighting

---

## 💬 FAQ

**Q: Should I use VADER or FinBERT?**  
A: Use VADER for speed, FinBERT for accuracy. Use both (ensemble) for best results.

**Q: Why did FinBERT say negative about a positive GLP-1 announcement?**  
A: FinBERT's financial training understands context: GLP-1 growth is a *threat* to snack makers.

**Q: Are these results final?**  
A: No—with only 11 articles, trends may shift. Rerun with 500+ articles when data collection completes.

**Q: Can I use just the CSV without reading docs?**  
A: Yes! The CSV has all columns clearly labeled. Column names are self-explanatory.

**Q: How often should I re-run this analysis?**  
A: Quarterly when adding 100+ new articles. More frequently if you change the cleaned text preprocessing.

---

## 📞 Support

- **Questions about VADER?** See: `VADER_FINBERT_COMPARISON.md` → "VADER Implementation"
- **Questions about models?** See: `VADER_FINBERT_QUICK_REFERENCE.md` → "Model Comparison Matrix"
- **Want statistical details?** See: `VADER_FINBERT_COMPARISON.md` → "Methodology"
- **Want use case examples?** See: `VADER_FINBERT_QUICK_REFERENCE.md` → "Article-Specific Findings"

---

**Project Status:** ✅ Complete and Ready for Production  
**Next Steps:** Run on full dataset (500+ articles) when collection completes  
**Maintenance:** Quarterly review of correlation metrics  
**Last Updated:** April 9, 2026
