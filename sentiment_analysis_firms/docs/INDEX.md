# Sentiment Analysis Pipeline - Complete Documentation Index

## 📖 Documentation Files

### Quick Start & Reference
- **[ANALYSIS_SUMMARY.md](ANALYSIS_SUMMARY.md)** - Executive summary of VADER & FinBERT analysis
- **[DID_QUICK_START.md](DID_QUICK_START.md)** - Quick start for difference-in-differences analysis
- **[QUICK_REFERENCE_TRENDS.md](QUICK_REFERENCE_TRENDS.md)** - Sentiment trends visualization guide

### Detailed Technical Analysis
- **[VADER_FINBERT_COMPARISON.md](VADER_FINBERT_COMPARISON.md)** - Complete correlation study (r=0.598)
- **[VADER_FINBERT_QUICK_REFERENCE.md](VADER_FINBERT_QUICK_REFERENCE.md)** - Model decision matrix & use cases
- **[VADER_FINBERT_INDEX.md](VADER_FINBERT_INDEX.md)** - Navigation guide with FAQ

### Econometric Analysis
- **[DID_ANALYSIS_GUIDE.md](DID_ANALYSIS_GUIDE.md)** - Difference-in-differences framework
- **[ECONOMETRIC_PIPELINE.md](ECONOMETRIC_PIPELINE.md)** - End-to-end analysis pipeline

### Data Processing
- **[PANEL_DATA_SUMMARY.md](PANEL_DATA_SUMMARY.md)** - Panel data (firm-week) structure
- **[SENTIMENT_TRENDS_ANALYSIS.md](SENTIMENT_TRENDS_ANALYSIS.md)** - Firm-level trends & rankings
- **[Z_SCORE_STANDARDIZATION.md](Z_SCORE_STANDARDIZATION.md)** - Within-week normalization methodology

### Reference
- **[WORKING_MEMORY.md](WORKING_MEMORY.md)** - Session notes & development log

---

## 🎯 By Audience

### For Executives/Decision-Makers
→ **Start:** [ANALYSIS_SUMMARY.md](ANALYSIS_SUMMARY.md)  
→ **Then:** [VADER_FINBERT_QUICK_REFERENCE.md](VADER_FINBERT_QUICK_REFERENCE.md) → Model Decision Matrix  
→ **Visual:** `/outputs/vader_finbert_comparison.png`  
⏱️ **Reading time:** 5 minutes

### For Analysts/Practitioners
→ **Start:** [QUICK_REFERENCE_TRENDS.md](QUICK_REFERENCE_TRENDS.md)  
→ **Then:** [DID_QUICK_START.md](DID_QUICK_START.md)  
→ **Then:** [SENTIMENT_TRENDS_ANALYSIS.md](SENTIMENT_TRENDS_ANALYSIS.md)  
⏱️ **Reading time:** 15 minutes

### For Researchers/Econometricians
→ **Start:** [VADER_FINBERT_COMPARISON.md](VADER_FINBERT_COMPARISON.md)  
→ **Then:** [DID_ANALYSIS_GUIDE.md](DID_ANALYSIS_GUIDE.md)  
→ **Then:** [ECONOMETRIC_PIPELINE.md](ECONOMETRIC_PIPELINE.md)  
⏱️ **Reading time:** 30-45 minutes

### For Data Engineers
→ **Start:** [ECONOMETRIC_PIPELINE.md](ECONOMETRIC_PIPELINE.md)  
→ **Then:** [PANEL_DATA_SUMMARY.md](PANEL_DATA_SUMMARY.md)  
→ **Then:** [Z_SCORE_STANDARDIZATION.md](Z_SCORE_STANDARDIZATION.md)  
→ **Code:** `/scripts/` and `/demos/`  
⏱️ **Reading time:** 20-30 minutes

---

## 🔑 Key Findings Summary

### VADER & FinBERT Correlation
| Metric | Value | Status |
|--------|-------|--------|
| Label Agreement | 91% | ✅ Excellent |
| Pearson Correlation | r = 0.598 | ✅ Moderate-Strong |
| RMSE | 0.252 | ✅ Low error |

See: [VADER_FINBERT_COMPARISON.md](VADER_FINBERT_COMPARISON.md)

### Firm Sentiment Rankings
| Rank | Firm | z_sentiment | Status |
|------|------|-------------|--------|
| 1 | Wonderful | +2.27 | ⭐ Extreme positive |
| 10 | PepsiCo | -0.53 | 📉 Most negative |

See: [SENTIMENT_TRENDS_ANALYSIS.md](SENTIMENT_TRENDS_ANALYSIS.md)

### DiD Treatment Effect
- **Treated firms** (traditional snacks): More negative post-GLP1
- **Effect strength:** β = -0.328***

See: [DID_ANALYSIS_GUIDE.md](DID_ANALYSIS_GUIDE.md)

---

## 📂 File Organization

```
docs/
├── ANALYSIS_SUMMARY.md              # Executive overview
├── VADER_FINBERT_COMPARISON.md      # Statistical analysis
├── VADER_FINBERT_QUICK_REFERENCE.md # Model comparison
├── VADER_FINBERT_INDEX.md           # Navigation guide
├── DID_ANALYSIS_GUIDE.md            # Econometric method
├── DID_QUICK_START.md               # Quick reference
├── ECONOMETRIC_PIPELINE.md          # End-to-end guide
├── PANEL_DATA_SUMMARY.md            # Data structure
├── SENTIMENT_TRENDS_ANALYSIS.md     # Firm rankings
├── QUICK_REFERENCE_TRENDS.md        # Viz guide
├── Z_SCORE_STANDARDIZATION.md       # Methodology
├── WORKING_MEMORY.md                # Dev notes
└── INDEX.md                         # This file
```

---

## 🚀 How to Use This Documentation

1. **Find your role** in the "By Audience" section above
2. **Follow the reading order** suggested
3. **Refer to specific sections** as needed for implementation
4. **Check visualizations** in `/outputs/` for empirical results
5. **Run scripts** in `/scripts/` for reproduction

---

## 📊 Output Locations

**Visualizations:** `/outputs/`
- `vader_finbert_comparison.png` - 4-panel comparison (300 DPI)
- `sentiment_trends.png` - Time series by firm

**Data Files:** `/outputs/` and `/data/processed/`
- `vader_finbert_comparison.csv` - Model scores side-by-side
- `sentiment_index.csv` - Panel data (firm-week)
- `firm_sentiment_rankings.csv` - Rankings table

**Statistics:** `/outputs/`
- `correlation_summary.txt` - Statistical tests
- `DID_ANALYSIS_RESULTS.txt` - Regression output

---

## 💡 Quick Answers

**Q: Should I use VADER or FinBERT?**  
A: See [VADER_FINBERT_QUICK_REFERENCE.md](VADER_FINBERT_QUICK_REFERENCE.md) → Decision Tree

**Q: How do I interpret the sentiment signals?**  
A: See [SENTIMENT_TRENDS_ANALYSIS.md](SENTIMENT_TRENDS_ANALYSIS.md) → Interpretation

**Q: What's the econometric specification?**  
A: See [DID_ANALYSIS_GUIDE.md](DID_ANALYSIS_GUIDE.md) → Regression Specification

**Q: How's the panel data structured?**  
A: See [PANEL_DATA_SUMMARY.md](PANEL_DATA_SUMMARY.md)

**Q: Why do models disagree?**  
A: See [VADER_FINBERT_COMPARISON.md](VADER_FINBERT_COMPARISON.md) → Key Findings

---

**Last Updated:** April 9, 2026  
**Total Documentation:** 12 files, ~100 KB  
**Average Read Time:** 5-45 minutes (depends on audience)
