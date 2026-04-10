"""
Difference-in-Differences: Educational Example with Synthetic Data

This script demonstrates what the DiD regression WILL look like with real data
that includes both pre-event and post-event observations.

Current limitation with demo data:
  - All observations are post-GLP1 (April 2026 vs. Jan 2023 event)
  - No variation in time periods for identifying treatment effects
  - Perfect separation → R² = 1.0, infinite standard errors

This script shows:
  1. How DiD estimation works with balanced data
  2. Interpretation of significant treatment effects
  3. Event study visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')


def create_synthetic_did_data():
    """
    Create realistic synthetic panel data with:
    - 10 firms (3 treated, 7 control)
    - 8 time periods (4 pre-event, 4 post-event)
    - Treatment effect on z_sentiment
    """
    
    np.random.seed(42)
    
    # Parameters
    firms = ['PepsiCo', 'Hershey', 'Mondelez',  # Treated
             'Nestle', 'General Mills', 'Ferrero', 'Mars', 'Chocolates', 'RXBAR', 'Wonderful']
    
    treated_firms = ['PepsiCo', 'Hershey', 'Mondelez']
    
    # Time periods (2022-Q1 through 2023-Q4)
    periods = []
    for year in [2022, 2023]:
        for quarter in range(1, 5):
            periods.append(f"{year}-Q{quarter}")
    
    glp1_date = "2023-Q1"
    
    # Generate data
    data = []
    for firm in firms:
        exposed = 1 if firm in treated_firms else 0
        
        for t, period in enumerate(periods):
            post = 1 if period >= glp1_date else 0
            
            # Generate z_sentiment with treatment effect
            base_sentiment = 0.0
            time_trend = 0.05 * (t - 3.5)  # Gradual trend
            firm_effect = np.random.normal(0.1, 0.15) if exposed == 1 else np.random.normal(-0.05, 0.1)
            
            # TREATMENT EFFECT: -0.3 for treated firms post-event
            treatment_effect = -0.3 if (exposed == 1 and post == 1) else 0
            
            # Random noise
            noise = np.random.normal(0, 0.1)
            
            z_sentiment = base_sentiment + time_trend + firm_effect + treatment_effect + noise
            
            data.append({
                'firm': firm,
                'period': period,
                'year': int(period.split('-')[0]),
                'quarter': int(period.split('-')[1][1]),
                'exposed': exposed,
                'post_glp1': post,
                't': t,
                'z_sentiment': z_sentiment
            })
    
    df = pd.DataFrame(data)
    
    return df


def run_did_on_synthetic(df):
    """Run DiD regression on synthetic data."""
    
    print("\n" + "="*70)
    print("SYNTHETIC DATA: DIFFERENCE-IN-DIFFERENCES REGRESSION")
    print("="*70)
    
    print(f"\nData Summary:")
    print(f"  Firms: {df['firm'].nunique()} (3 treated, 7 control)")
    print(f"  Time periods: {df['period'].nunique()}")
    print(f"  Observations: {len(df)}")
    print(f"  Treated firm-quarters: {(df['exposed']==1).sum()}")
    print(f"  Control firm-quarters: {(df['exposed']==0).sum()}")
    
    # Create interaction
    df['exposure_x_post'] = df['exposed'] * df['post_glp1']
    
    print(f"\nTreatment Assignment:")
    print(f"  Treated firms: PepsiCo, Hershey, Mondelez")
    print(f"  Event date: 2023-Q1")
    print(f"  Pre-event periods: 2022-Q1, Q2, Q3, Q4")
    print(f"  Post-event periods: 2023-Q1, Q2, Q3, Q4")
    
    print(f"\nSample Data:")
    sample = df[df['firm'].isin(['PepsiCo', 'Nestle'])][['firm', 'period', 'z_sentiment', 'exposed', 'post_glp1', 'exposure_x_post']]
    print(sample.to_string(index=False))
    
    # Fit model
    print("\n" + "="*70)
    print("REGRESSION RESULTS")
    print("="*70)
    
    formula = 'z_sentiment ~ C(firm) + C(period) + exposure_x_post'
    model = ols(formula, data=df).fit()
    
    print(f"\nFormula: {formula}\n")
    print(model.summary())
    
    # Extract key results
    did_coef = model.params.get('exposure_x_post', 0)
    did_se = model.bse.get('exposure_x_post', np.inf)
    did_pval = model.pvalues.get('exposure_x_post', 1.0)
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    print(f"\n📊 MAIN RESULT (Difference-in-Differences):")
    print(f"   Coefficient: {did_coef:.6f}")
    print(f"   Std Error:   {did_se:.6f}")
    print(f"   P-value:     {did_pval:.6f}")
    
    if did_pval < 0.01:
        sig = "*** (p < 0.01) - Highly significant"
    elif did_pval < 0.05:
        sig = "** (p < 0.05) - Significant at 5%"
    elif did_pval < 0.10:
        sig = "* (p < 0.10) - Significant at 10%"
    else:
        sig = "Not significant"
    
    print(f"   Significance: {sig}")
    
    print(f"\n📈 Interpretation:")
    if did_coef < 0:
        print(f"   Treated firms experienced {abs(did_coef):.3f} std deviations")
        print(f"   LOWER sentiment impact from GLP-1 event compared to control firms.")
        print(f"\n   Practical meaning:")
        print(f"   - GLP-1 dietary pills are substitutes for snacks")
        print(f"   - Treated firms (traditional snack makers) have lower sentiment")
        print(f"   - Control firms (candy, packaged foods) less impacted")
    
    print(f"\n💡 Event Study Summary:")
    pre_treated = df[(df['exposed']==1) & (df['post_glp1']==0)]['z_sentiment'].mean()
    post_treated = df[(df['exposed']==1) & (df['post_glp1']==1)]['z_sentiment'].mean()
    pre_control = df[(df['exposed']==0) & (df['post_glp1']==0)]['z_sentiment'].mean()
    post_control = df[(df['exposed']==0) & (df['post_glp1']==1)]['z_sentiment'].mean()
    
    print(f"\n   Pre-event (avg):")
    print(f"      Treated:  {pre_treated:.4f}")
    print(f"      Control:  {pre_control:.4f}")
    print(f"      Diff:     {pre_treated - pre_control:.4f}")
    
    print(f"\n   Post-event (avg):")
    print(f"      Treated:  {post_treated:.4f}")
    print(f"      Control:  {post_control:.4f}")
    print(f"      Diff:     {post_treated - post_control:.4f}")
    
    print(f"\n   Change in difference (DiD):")
    print(f"      {(post_treated - post_control) - (pre_treated - pre_control):.4f}")
    print(f"      (Compared to regression coef: {did_coef:.4f})")
    
    return df, model


def visualize_event_study(df):
    """Create event study visualization."""
    
    print("\n" + "="*70)
    print("VISUALIZING THE EVENT STUDY")
    print("="*70)
    
    # Aggregate by exposure and period
    summary = df.groupby(['period', 'exposed'])['z_sentiment'].agg(['mean', 'std', 'count']).reset_index()
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    
    treated_data = summary[summary['exposed'] == 1].sort_values('period')
    control_data = summary[summary['exposed'] == 0].sort_values('period')
    
    print("\nTreated firms (by quarter):")
    print(treated_data[['period', 'mean', 'se']].to_string(index=False))
    
    print("\nControl firms (by quarter):")
    print(control_data[['period', 'mean', 'se']].to_string(index=False))
    
    # Event indicator
    event_periods = df['period'].unique()
    event_idx = np.where(event_periods >= '2023-Q1')[0][0]
    
    print(f"\n📊 Event Study Plot:")
    print(f"   X-axis: Time periods")
    print(f"   Y-axis: Average z_sentiment")
    print(f"   Vertical line: 2023-Q1 (GLP-1 event)")
    print(f"\n   Expected pattern:")
    print(f"   - Treated (red): Drops after event ↓")
    print(f"   - Control (blue): Stays relatively flat →")
    print(f"   - The gap widens = DiD treatment effect")
    
    print("\n   ASCII Visualization:")
    print("   ")
    print("    z_sentiment")
    print("     0.5 |            ")
    print("     0.0 | O---O    O---O  (Control, flat)")
    print("    -0.5 | X---X··X···X   (Treated, drops at | event)")
    print("        +-----------+---- time")
    print("              2023-Q1|   (Event)")
    print("        Pre       | Post")


def main():
    """Main execution."""
    
    print("\n" + "="*70)
    print("SYNTHETIC DEMONSTRATION: DiD WITH BALANCED PRE/POST DATA")
    print("="*70)
    
    # Create synthetic data
    df = create_synthetic_did_data()
    
    # Run regression
    df, model = run_did_on_synthetic(df)
    
    # Visualize
    visualize_event_study(df)
    
    print("\n" + "="*70)
    print("WHY SYNTHETIC DATA WORKS HERE (but not with our demo data)")
    print("="*70)
    
    print("""
Synthetic data characteristics:
  ✓ Pre-event observations (2022): Can estimate treated vs. control baseline
  ✓ Post-event observations (2023): Can estimate new treated vs. control levels
  ✓ Variation in dependent variable: Enough to identify treatment effect
  ✓ Parallel trends: Treated/control track similarly pre-event (assumption check)

Demo data limitations:
  ✗ All observations post-event (April 2026 vs. Jan 2023)
  ✗ No pre-event baseline (post_glp1 = 1 for all)
  ✗ Perfect fit from fixed effects alone (R² = 1.0)
  ✗ Cannot disentangle treatment effect from fixed effects

With REAL DATA (when gnews collection completes):
  ✓ Will have multiple articles per week (richer variation)
  ✓ Can extend time window to include pre-2023 periods if historical data added
  ✓ Will have proper treatment variation (both exposed=0 and exposed=1)
  ✓ Will enable credible DiD estimation

Next action: Collect pre-event sentiment data (e.g., 2022 news archives)
    """)
    
    return df, model


if __name__ == '__main__':
    df, model = main()
