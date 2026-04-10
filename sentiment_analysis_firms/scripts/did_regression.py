"""
Difference-in-Differences Regression: GLP-1 Event Analysis

Research Question: Did the GLP-1 event (2023-01-01) differentially impact 
sentiment for traditional snack firms vs. others?

Specification:
  z_sentiment_it = alpha_i + gamma_t + beta*(exposed_i * post_t) + error_it

Where:
  - alpha_i = firm fixed effects
  - gamma_t = time (week) fixed effects
  - exposed_i = 1 if traditional snack firm, 0 otherwise
  - post_t = 1 if week >= GLP1_EVENT_DATE, 0 before
  - beta = Treatment effect (DiD coefficient)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def week_to_date(week_str):
    """
    Convert ISO week string (YYYY-Www) to actual date (Monday of that week).
    
    Args:
        week_str: String in format "YYYY-Www" (e.g., "2026-W14")
        
    Returns:
        datetime object (first day of the ISO week)
    """
    year, week = week_str.split('-W')
    year = int(year)
    week = int(week)
    
    # January 4th is always in week 1 of the year
    jan4 = datetime(year, 1, 4)
    
    # Find Monday of week 1
    week_one_monday = jan4 - pd.Timedelta(days=jan4.weekday())
    
    # Calculate the Monday of the target week
    target_monday = week_one_monday + pd.Timedelta(weeks=week-1)
    
    return target_monday


def create_did_data(input_csv='outputs/sentiment_index.csv',
                    glp1_event_date='2023-01-01'):
    """
    Prepare data for difference-in-differences regression.
    
    Args:
        input_csv: Path to sentiment index CSV
        glp1_event_date: GLP-1 event cutoff date (YYYY-MM-DD)
        
    Returns:
        DataFrame with treatment variables and date columns
    """
    
    print("="*70)
    print("DIFFERENCE-IN-DIFFERENCES REGRESSION: GLP-1 IMPACT")
    print("="*70)
    
    # Load data
    print(f"\n1. Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"   Loaded {len(df)} observations")
    
    # Convert GLP-1 event date
    glp1_dt = pd.to_datetime(glp1_event_date)
    print(f"\n2. GLP-1 Event Date: {glp1_dt.date()}")
    
    # Convert ISO weeks to dates
    print(f"\n3. Converting ISO weeks to dates...")
    df['week_date'] = df['week'].apply(week_to_date)
    
    # Create post_glp1 indicator
    df['post_glp1'] = (df['week_date'] >= glp1_dt).astype(int)
    
    print(f"   Date range in data: {df['week_date'].min().date()} to {df['week_date'].max().date()}")
    print(f"   Pre-GLP1 observations: {(df['post_glp1'] == 0).sum()}")
    print(f"   Post-GLP1 observations: {(df['post_glp1'] == 1).sum()}")
    
    # Define treatment groups
    print(f"\n4. Defining treatment groups...")
    treated_firms = ['PepsiCo', 'Hershey', 'Mondelez']
    df['exposed'] = (df['firm'].isin(treated_firms)).astype(int)
    
    print(f"   Treated firms (exposed=1): {', '.join(treated_firms)}")
    print(f"   Control firms (exposed=0): {len(df[df['exposed']==0]) // (df['week'].nunique() + 1)} firms")
    
    treated_count = df[df['exposed'] == 1].groupby('firm').ngroups
    control_count = df[df['exposed'] == 0].groupby('firm').ngroups
    print(f"   Total treated firm-weeks: {(df['exposed'] == 1).sum()}")
    print(f"   Total control firm-weeks: {(df['exposed'] == 0).sum()}")
    
    # Create interaction term
    df['exposure_x_post'] = df['exposed'] * df['post_glp1']
    
    # Display sample data
    print(f"\n5. Sample data (first 10 rows):")
    sample_cols = ['firm', 'week', 'z_sentiment', 'exposed', 'post_glp1', 'exposure_x_post']
    print(df[sample_cols].head(10).to_string(index=False))
    
    return df, glp1_dt


def run_did_regression(df):
    """
    Run difference-in-differences regression with fixed effects.
    
    Formula: z_sentiment ~ C(firm) + C(week) + exposure_x_post
    
    Args:
        df: DataFrame with treatment variables
        
    Returns:
        Regression results object
    """
    
    print("\n" + "="*70)
    print("REGRESSION SPECIFICATION")
    print("="*70)
    
    print("""
Dependent Variable: z_sentiment
  
Model: z_sentiment_it = alpha_i + gamma_t + beta*(exposed_i * post_t) + error_it

Terms:
  C(firm)          = Firm fixed effects (alpha_i)
  C(week)          = Week fixed effects (gamma_t)
  exposure_x_post  = Treatment effect (beta) - DiD coefficient
  
The coefficient on exposure_x_post represents:
  - The differential impact on sentiment of treated vs. control firms
  - After vs. before the GLP-1 event
  - Controlling for firm and time fixed effects
    """)
    
    print("\n" + "="*70)
    print("RUNNING REGRESSION...")
    print("="*70)
    
    # Fit OLS with fixed effects
    # Using firm and week fixed effects via C() categorical variables
    formula = 'z_sentiment ~ C(firm) + C(week) + exposure_x_post'
    
    print(f"\nFormula: {formula}\n")
    
    # Fit the model
    model = ols(formula, data=df).fit()
    
    # Print summary
    print(model.summary())
    
    return model


def interpret_results(model, df):
    """
    Interpret the difference-in-differences regression results.
    
    Args:
        model: Fitted regression model
        df: Original data
    """
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # Extract DiD coefficient
    did_coef = model.params.get('exposure_x_post', None)
    did_se = model.bse.get('exposure_x_post', None)
    did_pval = model.pvalues.get('exposure_x_post', None)
    
    if did_coef is not None:
        print(f"\n📊 MAIN RESULT (Difference-in-Differences):")
        print(f"   Coefficient: {did_coef:.6f}")
        print(f"   Std Error:   {did_se:.6f}")
        print(f"   P-value:     {did_pval:.6f}")
        
        # Significance interpretation
        if did_pval < 0.01:
            sig = "*** (p < 0.01) - Highly significant"
        elif did_pval < 0.05:
            sig = "** (p < 0.05) - Significant at 5%"
        elif did_pval < 0.10:
            sig = "* (p < 0.10) - Significant at 10%"
        else:
            sig = "Not significant (p >= 0.10)"
        
        print(f"   Significance: {sig}")
        
        print(f"\n📈 Interpretation:")
        if did_coef > 0:
            print(f"   Treated firms' sentiment")
            print(f"   {abs(did_coef):.6f} units HIGHER after GLP-1 event (vs. control firms)")
            print(f"   compared to the pre-event difference.")
        elif did_coef < 0:
            print(f"   Treated firms' sentiment")
            print(f"   {abs(did_coef):.6f} units LOWER after GLP-1 event (vs. control firms)")
            print(f"   compared to the pre-event difference.")
        else:
            print(f"   NO differential effect of GLP-1 event on treated firms.")
        
        # Practical significance
        print(f"\n💡 Context:")
        print(f"   Effect size in σ units: {did_coef:.4f}")
        print(f"   Overall z_sentiment std: {df['z_sentiment'].std():.4f}")
        if df['z_sentiment'].std() > 0:
            pct_effect = (did_coef / df['z_sentiment'].std()) * 100
            print(f"   As % of overall std: {pct_effect:.1f}%")
    
    # Model fit
    print(f"\n📉 Model Fit:")
    print(f"   R-squared:     {model.rsquared:.4f}")
    print(f"   Adj R-squared: {model.rsquared_adj:.4f}")
    print(f"   N:             {model.nobs}")
    print(f"   F-statistic:   {model.fvalue:.4f}")
    print(f"   Prob(F):       {model.f_pvalue:.6f}")
    
    # Summary statistics by group
    print(f"\n📋 Summary Statistics by Group:")
    
    for exposed_val in [0, 1]:
        for post_val in [0, 1]:
            subset = df[(df['exposed'] == exposed_val) & (df['post_glp1'] == post_val)]
            if len(subset) > 0:
                group_name = "Treated" if exposed_val == 1 else "Control"
                period = "Post-GLP1" if post_val == 1 else "Pre-GLP1"
                
                print(f"\n   {group_name} ({period}):")
                print(f"      N:    {len(subset)}")
                print(f"      Mean: {subset['z_sentiment'].mean():.6f}")
                print(f"      Std:  {subset['z_sentiment'].std():.6f}")
                print(f"      Min:  {subset['z_sentiment'].min():.6f}")
                print(f"      Max:  {subset['z_sentiment'].max():.6f}")
    
    # Naive difference-in-differences calculation
    print(f"\n🔍 Naive Difference-in-Differences (no controls):")
    
    treated_pre = df[(df['exposed'] == 1) & (df['post_glp1'] == 0)]['z_sentiment'].mean()
    treated_post = df[(df['exposed'] == 1) & (df['post_glp1'] == 1)]['z_sentiment'].mean()
    control_pre = df[(df['exposed'] == 0) & (df['post_glp1'] == 0)]['z_sentiment'].mean()
    control_post = df[(df['exposed'] == 0) & (df['post_glp1'] == 1)]['z_sentiment'].mean()
    
    treated_diff = treated_post - treated_pre
    control_diff = control_post - control_pre
    naive_did = treated_diff - control_diff
    
    print(f"\n   Treated (Post - Pre):  {treated_post:.6f} - {treated_pre:.6f} = {treated_diff:.6f}")
    print(f"   Control (Post - Pre):  {control_post:.6f} - {control_pre:.6f} = {control_diff:.6f}")
    print(f"   DiD (Treated - Control): {treated_diff:.6f} - {control_diff:.6f} = {naive_did:.6f}")
    print(f"\n   (Compare to regression coefficient: {did_coef:.6f})")


def main():
    """Main execution."""
    
    # Create treatment variables
    df, glp1_dt = create_did_data()
    
    # Run regression
    model = run_did_regression(df)
    
    # Interpret results
    interpret_results(model, df)
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
1. With real data (500+ observations):
   - More robust estimates
   - Better balance across time periods and firms
   
2. Alternative event dates:
   - Try different GLP-1 announcement dates
   - Test for pre-trends (parallel assumption)
   
3. Robustness checks:
   - Event window analysis
   - Placebo tests (fake treatment dates)
   - Heterogeneous effects by firm
   
4. Integration with returns:
   - Test if sentiment differences translate to return differences
   - Merge with stock price data
""")
    
    return df, model


if __name__ == '__main__':
    df, model = main()
