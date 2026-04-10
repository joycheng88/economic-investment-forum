# Demo & Example Scripts

Reference implementations and demonstrations. These scripts showcase usage patterns and are not production-critical.

## Overview

```
demos/
├── did_synthetic_demo.py           # DiD with synthetic data
├── examples_panel_analysis.py      # Panel data patterns
├── rolling_sentiment_demo.py       # Rolling average calculation
├── sentiment_index_examples.py     # Index aggregation patterns
└── compare_vader_finbert.py        # Model comparison reference
```

## Script Details

### did_synthetic_demo.py
**Purpose:** Demonstrate DiD methodology with synthetic data

**Shows:**
- Treatment variable creation (exposed, post_event, interaction)
- Fixed effects regression setup
- How treatment effect manifests
- Interpretation of regression output

**Key Features:**
- Generates synthetic panel data (10 firms × 8 quarters)
- Assigns treatment to 5 firms
- Creates fake but realistic sentiment patterns
- Runs regression with known ground truth

**Use:** Learning DiD methodology without real data

### examples_panel_analysis.py
**Purpose:** Demonstrate panel data structures and operations

**Shows:**
- Creating firm-week panels
- Groupby operations
- Time series aggregation
- Multi-level indexing

**Key Features:**
- Sample data loading
- Panel filtering by firm/week
- Merging datasets
- Panel statistics

**Use:** Understanding panel data workflows

### rolling_sentiment_demo.py
**Purpose:** Demonstrate rolling window calculations

**Shows:**
- 4-week rolling average
- Per-firm rolling operations
- Forward/backward fill strategies
- Rolling statistics (mean, std, min, max)

**Key Features:**
- GroupBy with rolling windows
- Handling NaN at boundaries
- Trend detection from rolling averages
- Volatility calculations

**Use:** Understanding rolling average mechanics

### sentiment_index_examples.py
**Purpose:** Demonstrate weekly aggregation patterns

**Shows:**
- Weekly grouping (ISO weeks)
- Multi-level aggregation
- Computing summary statistics
- Index structure creation

**Key Features:**
- Market-level indices
- Firm-level indices
- Keyword-level indices
- Label distribution computation

**Use:** Understanding index creation

### compare_vader_finbert.py
**Purpose:** Reference implementation for model comparison

**Shows:**
- VADER analysis
- FinBERT framework
- Correlation computation
- Agreement metrics

**Key Features:**
- Article-by-article comparison
- Statistical testing
- Visualization templates
- Summary statistics

**Use:** Understanding model differences

---

## Running Demos

```bash
# Individual demo
python demos/did_synthetic_demo.py
python demos/examples_panel_analysis.py
python demos/rolling_sentiment_demo.py
python demos/sentiment_index_examples.py
python demos/compare_vader_finbert.py

# All demos
for script in demos/*.py; do python "$script"; done
```

---

## Use Cases

**Learning:** Run demos to understand methodology  
**Testing:** Use as templates for custom analysis  
**Debugging:** Check demo patterns when implementing features  
**Documentation:** Reference correct implementations

---

**Location:** `/demos/`  
**Status:** Educational / Reference only  
**Outputs:** Printed to console
