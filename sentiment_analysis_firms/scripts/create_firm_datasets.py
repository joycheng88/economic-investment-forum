"""
Create firm-specific datasets from real news data
Filters for each firm + weight loss/GLP-1 relevance
"""

import os
import sys
import pandas as pd
import re
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
FIRMS = {
    'RXBAR': 'rxbar_data.csv',
    'Chomps': 'chomps_data.csv',
    'Wonderful': 'wonderful_data.csv',
    'General Mills': 'general_mills_data.csv',
    'PepsiCo': 'pepsico_data.csv',
    'Nestle': 'nestle_data.csv',
    'Mars': 'mars_data.csv',
    'Mondelez': 'mondelez_data.csv',
    'Ferrero': 'ferrero_data.csv',
    'Hershey': 'hershey_data.csv'
}

# GLP-1 and weight loss keywords
RELEVANCE_KEYWORDS = [
    'GLP-1', 'GLP1', 'Ozempic', 'Wegovy', 'Mounjaro', 'Zepbound',
    'semaglutide', 'tirzepatide', 'liraglutide',
    'weight loss', 'weight management', 'obesity', 'anti-obesity',
    'appetite suppressant', 'appetite suppression',
    'diabetes', 'metabolic', 'metabolism',
    'pharmaceutical', 'drug', 'medication', 'treatment',
    'clinical trial', 'FDA approval', 'regulatory',
    'market share', 'market impact', 'competition'
]

GENERIC_TERMS = [
    'protein bar', 'snack', 'nutrition', 'healthy',
    'recipe', 'diet', 'health benefits'
]

def check_relevance(text, firm_name):
    """
    Check if article is relevant to firm's GLP-1/weight loss business.
    Must mention firm AND have weight loss/GLP-1 context (not generic).
    """
    if not isinstance(text, str) or not text.strip():
        return False
    
    text_lower = text.lower()
    firm_lower = firm_name.lower()
    
    # Must mention firm
    if firm_lower not in text_lower:
        return False
    
    # Must mention GLP-1 or weight loss keywords
    has_relevance = any(kw.lower() in text_lower for kw in RELEVANCE_KEYWORDS)
    
    if not has_relevance:
        return False
    
    return True


def filter_firm_data(df, firm_name):
    """
    Filter data for specific firm with relevance checks.
    """
    # Filter by firm name
    df_firm = df[df['firm_name'] == firm_name].copy()
    
    if len(df_firm) == 0:
        return df_firm
    
    # Check relevance in combined text (title + description + full_text)
    df_firm['combined_text'] = (
        df_firm['title'].fillna('') + ' ' +
        df_firm['description'].fillna('') + ' ' +
        df_firm['full_text'].fillna('')
    )
    
    # Filter for relevance
    df_firm['is_relevant'] = df_firm['combined_text'].apply(
        lambda x: check_relevance(x, firm_name)
    )
    
    # Keep only relevant articles
    df_filtered = df_firm[df_firm['is_relevant'] == True].copy()
    
    # Remove temporary columns
    df_filtered = df_filtered.drop(columns=['combined_text', 'is_relevant'])
    
    return df_filtered


def main():
    """Create firm-specific datasets"""
    
    print("\n" + "=" * 80)
    print("CREATING FIRM-SPECIFIC DATASETS")
    print("=" * 80)
    
    # Load raw data
    input_path = 'data/raw/news_data_real.csv'
    if not os.path.exists(input_path):
        print(f"\n✗ Error: {input_path} not found")
        return
    
    print(f"\n📥 Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Total articles: {len(df)}")
    
    # Create output directory
    output_dir = 'firm_datasets'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}/")
    
    # Process each firm
    print("\n" + "-" * 80)
    print("FILTERING BY FIRM AND RELEVANCE")
    print("-" * 80)
    
    summary = []
    
    for firm_name, filename in sorted(FIRMS.items()):
        # Filter data
        df_firm = filter_firm_data(df, firm_name)
        
        output_path = os.path.join(output_dir, filename)
        
        # Save to CSV
        df_firm.to_csv(output_path, index=False)
        
        # Summary
        total_articles = len(df[df['firm_name'] == firm_name])
        relevant_articles = len(df_firm)
        retention_pct = (relevant_articles / total_articles * 100) if total_articles > 0 else 0
        
        summary.append({
            'firm': firm_name,
            'total': total_articles,
            'relevant': relevant_articles,
            'retention': retention_pct,
            'file': filename
        })
        
        status = "✓" if relevant_articles > 0 else "⚠"
        print(f"{status} {firm_name:<20} : {relevant_articles:>4}/{total_articles:<4} ({retention_pct:>5.1f}%) → {filename}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    df_summary = pd.DataFrame(summary)
    print("\nFirm Statistics:")
    print("-" * 80)
    print(f"{'Firm':<20} {'Total':<8} {'Relevant':<10} {'Retention':<12} {'File':<30}")
    print("-" * 80)
    for _, row in df_summary.iterrows():
        print(f"{row['firm']:<20} {row['total']:<8} {row['relevant']:<10} {row['retention']:>10.1f}% {row['file']:<30}")
    
    print("\n📊 Total Statistics:")
    print(f"  Total articles across all firms: {df_summary['total'].sum()}")
    print(f"  Total relevant articles: {df_summary['relevant'].sum()}")
    print(f"  Overall retention rate: {df_summary['relevant'].sum() / df_summary['total'].sum() * 100:.1f}%")
    
    # Files created
    print("\n📁 Files Created:")
    print("-" * 80)
    for _, row in df_summary.iterrows():
        file_path = os.path.join(output_dir, row['file'])
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"  ✓ {file_path:<50} ({file_size:>6.1f} KB)")
    
    print("\n" + "=" * 80)
    print("✓ FIRM-SPECIFIC DATASETS CREATED")
    print("=" * 80)
    
    print("\nNext steps:")
    print("  1. Use these firm-specific CSVs for further analysis")
    print("  2. Run sentiment analysis on each firm separately")
    print("  3. Analyze firm-specific GLP-1 impact")
    
    return df_summary


if __name__ == "__main__":
    main()
