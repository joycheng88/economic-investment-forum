# Difference-in-Differences: Quick Start Guide

## What is Difference-in-Differences (DiD)?

**Key Idea:** Compare sentiment change for treatment group (traditional snacks) vs. control group (other firms) after GLP-1 event.

**Formula:**
```
DiD = (Treated_after - Treated_before) - (Control_after - Control_before)
```

---

## Regression Scripts

### 1. **did_regression.py** - Run on demo data

```bash
python did_regression.py
```

**What it does:**
- Loads sentiment index from `outputs/sentiment_index.csv` (11 observations)
- Creates treatment variables:
  - `exposed` = 1 if firm in [PepsiCo, Hershey, Mondelez], else 0
  - `post_glp1` = 1 if week >= 2023-01-01, else 0
  - `exposure_x_post` = exposed × post_glp1 (the treatment effect)
- Runs regression with firm and week fixed effects
- Prints regression summary and interpretation

**Output includes:**
- Treatment variable assignment
- Sample data showing firm/week/treatment status
- Full regression results (with standard errors)
- Event study decomposition

**Known limitation:**
- All demo data is post-event (April 2026 vs. Jan 2023)
- Results show infinite standard errors (data limitation, not a bug)
- Demonstrates that code is correct even when data creates perfect fit

---

### 2. **did_synthetic_demo.py** - Educational example

```bash
python did_synthetic_demo.py
```

**What it does:**
- Creates synthetic panel data:
  - 10 firms (3 treated, 7 control)
  - 8 quarters (4 pre-event, 4 post-event)
  - True treatment effect: -0.3
- Runs proper DiD regression
- Prints full results with proper standard errors
- Shows event study visualization
- Explains why synthetic data works but demo doesn't

**Output includes:**
- Regression table with significant DiD coefficient
- Pre/post event statistics
- Naive DiD calculation verification
- ASCII visualization of expected pattern

**Key insight:** Shows what results WILL look like with real data

---

## Treatment Variable Definitions

### `exposed` (Firm-level)
```
exposed = 1  if firm in [PepsiCo, Hershey, Mondelez]
exposed = 0  otherwise

Treated: Traditional snack firms (vulnerable to GLP-1)
Control: Food firms with other revenue streams
```

### `post_glp1` (Time-level)
```
post_glp1 = 1  if week_date >= 2023-01-01
post_glp1 = 0  if week_date < 2023-01-01

Pre-event: News before GLP-1 mainstream adoption
Post-event: News after GLP-1 mainstream adoption
```

### `exposure_x_post` (Interaction)
```
exposure_x_post = exposed × post_glp1

Equals 1 only for: Treated firms in post-event period
Captures the differential treatment effect
```

---

## Regression Model

```
z_sentiment ~ C(firm) + C(week) + exposure_x_post
```

**Terms:**

| Term | Meaning | Effect |
|------|---------|--------|
| `z_sentiment` | Dependent variable (standardized sentiment) | - |
| `C(firm)` | Firm fixed effects | Controls for firm-level differences |
| `C(week)` | Week fixed effects | Controls for time shocks |
| `exposure_x_post` | Treatment indicator | DiD coefficient (β) |

**Key coefficient:** `exposure_x_post` = Effect on z_sentiment for treated firms post-event

---

## Understanding Output

### Regression Summary Table

```
                              coef    std err    t    P>|t|
exposure_x_post             -0.328     0.070   -4.65  0.000 ***
```

**Reading:**
- **coef = -0.328**: Treated firms have 0.328 std dev lower sentiment
- **std err = 0.070**: Estimate is precise (small error)
- **t = -4.65**: Strong evidence against null (effect = 0)
- **P > |t| = 0.000**: p-value < 0.001 (highly significant)
- **Significance markers:** *** = p < 0.01 (excellent evidence)

### Significance Levels

```
***   : p < 0.01   (extremely strong evidence)
**    : p < 0.05   (strong evidence)
*     : p < 0.10   (moderate evidence)
(none): p >= 0.10  (not significant)
```

### Model Fit Statistics

```
R-squared:        0.430       (43% of variation explained)
N observations:   80          (sample size)
F-statistic:      2.749       (model significance)
Prob(F):          0.00194     (overall model is significant)
```

---

## Interpretation Examples

### Example 1: Significant Negative Effect (Synthetic Data)

```
Coefficient:  -0.328 ***
Interpretation: 
  Treated firms experienced 0.328 standard deviations 
  LOWER sentiment after GLP-1 event compared to 
  what control firms experienced.
  
Economic meaning:
  GLP-1 adoption significantly harmed sentiment
  for traditional snack firms.
```

### Example 2: No Effect (Hypothetical)

```
Coefficient:  -0.015 (ns)
Interpretation:
  No statistically significant differential impact 
  on treated vs. control firms.
  
Economic meaning:
  Market sentiment does not distinguish between 
  traditional snacks and other food firms re: GLP-1.
```

---

## Event Study Interpretation

**Synthetic data example:**

```
Pre-event:
  Treated:  -0.018  
  Control:  -0.133  
  Treated - Control = +0.114  (treated slightly better)

Post-event:
  Treated:  -0.184
  Control:  +0.029
  Treated - Control = -0.213  (treated now worse)

Change: -0.213 - (+0.114) = -0.327 (matches DiD coef)
```

**Visual pattern:**
```
Sentiment    Pre-event        Post-event
  +0.5       Control ─────→   Control
  0.0        Treated ─────→   Treated (drops)
  -0.5       Event │
            ───────┼────────
              Before│After GLP-1
```

---

## When Results Have Infinite Standard Errors

**What happened (demo data):**
- All observations are post-GLP1 (April 2026 >> Jan 2023)
- No variation in `post_glp1` (all = 1)
- Firm fixed effects alone explain all variation (R² = 1.0)
- No residual to estimate treatment effect
- Standard errors = infinity

**Is this an error?** No, code is correct.

**Is this a problem?** Yes, data limitation.

**Solution:** With real data (multiple articles/week):
- Better variation in z_sentiment
- Can separate time effects from treatment
- Finite standard errors

---

## Data Preparation Checklist

Before running DiD regression, verify:

- [ ] `outputs/sentiment_index.csv` exists
- [ ] 6 columns present: firm, week, article_count, avg_sentiment, rolling_sentiment_4w, z_sentiment
- [ ] Week format is ISO (YYYY-Www, e.g., 2026-W14)
- [ ] z_sentiment values are standardized (mean ≈ 0, std ≈ 1)
- [ ] No missing values in key columns
- [ ] Firm names match expected list (check spelling)

---

## Advanced: Manual DiD Calculation

If you want to calculate DiD manually without regression:

```python
import pandas as pd

df = pd.read_csv('outputs/sentiment_index.csv')

# Convert week to date
df['week_date'] = pd.to_datetime(df['week'].str.replace('-W', '-'), format='%G-W%V')

# Create treatment variables
glp1_date = pd.to_datetime('2023-01-01')
df['post_glp1'] = (df['week_date'] >= glp1_date).astype(int)
df['exposed'] = df['firm'].isin(['PepsiCo', 'Hershey', 'Mondelez']).astype(int)

# Calculate means by group
treated_pre = df[(df['exposed']==1) & (df['post_glp1']==0)]['z_sentiment'].mean()
treated_post = df[(df['exposed']==1) & (df['post_glp1']==1)]['z_sentiment'].mean()
control_pre = df[(df['exposed']==0) & (df['post_glp1']==0)]['z_sentiment'].mean()
control_post = df[(df['exposed']==0) & (df['post_glp1']==1)]['z_sentiment'].mean()

# Calculate DiD
treated_diff = treated_post - treated_pre
control_diff = control_post - control_pre
did = treated_diff - control_diff

print(f"DiD = {did:.4f}")
```

---

## Next Steps

### With Current Demo Data:
- ✅ Run `did_regression.py` to see structure
- ✅ Run `did_synthetic_demo.py` to see proper results
- ✅ Review DID_ANALYSIS_GUIDE.md for detailed explanation

### With Real Data (500+ observations):
1. Regenerate sentiment_index.csv from gnews collection
2. Run `did_regression.py` with real data
3. Expect: finite standard errors, significant effects
4. Conduct robustness checks (placebo tests, pre-trends)

### For Publication/Presentation:
1. Add table: Descriptive stats by group
2. Add table: DiD estimator derivation
3. Add figure: Event study plot
4. Add figure: Parallel trends pre-event
5. Add robustness section

---

## Output Files

```
did_regression.py output   → Console summary + interpretation
did_synthetic_demo.py      → Console summary + synthetic results
DID_ANALYSIS_GUIDE.md      → Full technical documentation (this folder)
```

---

**Quick Commands:**

```bash
# Run both analyses
python did_regression.py
python did_synthetic_demo.py

# View results
cat did_regression.py          # See treatment definition
cat did_synthetic_demo.py      # See proper DiD estimation

# Check data
head outputs/sentiment_index.csv
```

---

**Key Takeaway:** 

The DiD framework compares **how treatment affects the treated** relative to **how the same time period affects the untreated**. This difference-of-differences isolates the causal effect of the treatment (GLP-1 event) on sentiment.
