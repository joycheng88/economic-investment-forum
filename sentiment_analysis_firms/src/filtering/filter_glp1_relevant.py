"""
Filter preprocessed articles to keep only those with BOTH:
1. A firm name (or its variants)
2. At least one GLP-1 keyword

Output: data/processed/glp1_relevant.csv
"""

import pandas as pd
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.firm_names import FIRM_NAME_VARIATIONS
from src.glp1_keywords import ALL_GLP1_KEYWORDS


def contains_firm(text, firm_variations):
    """
    Check if text contains any firm name variant.
    
    Args:
        text: String to search
        firm_variations: Dict of firm -> [variations]
    
    Returns:
        tuple: (found, firm_name) where found is bool and firm_name is str or None
    """
    if not isinstance(text, str):
        return False, None
    
    text_lower = text.lower()
    
    for firm, variations in firm_variations.items():
        for variant in variations:
            if variant.lower() in text_lower:
                return True, firm
    
    return False, None


def contains_glp1_keyword(text, keywords):
    """
    Check if text contains at least one GLP-1 keyword.
    
    Args:
        text: String to search
        keywords: List of keywords
    
    Returns:
        tuple: (found, keywords_found) where found is bool and keywords_found is list
    """
    if not isinstance(text, str):
        return False, []
    
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in text_lower:
            found_keywords.append(keyword)
    
    return len(found_keywords) > 0, found_keywords


def filter_glp1_relevant(input_csv, output_csv):
    """
    Filter dataset to keep articles with BOTH firm name AND GLP-1 keyword.
    
    Args:
        input_csv: Path to cleaned data
        output_csv: Path to save filtered data
    
    Returns:
        dict: Statistics dictionary
    """
    
    print("=" * 80)
    print("FILTERING FOR GLP-1 RELEVANT ARTICLES")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from: {input_csv}")
    if not os.path.exists(input_csv):
        print(f"✗ File not found: {input_csv}")
        print(f"  Run preprocess_news.py first to generate cleaned data")
        return None
    
    df = pd.read_csv(input_csv)
    print(f"✓ Loaded {len(df):,} articles")
    
    # Prepare variations
    total_variants = sum(len(vars) for vars in FIRM_NAME_VARIATIONS.values())
    print(f"✓ Loaded {len(FIRM_NAME_VARIATIONS)} firms with {total_variants} total variants")
    print(f"✓ Loaded {len(ALL_GLP1_KEYWORDS)} GLP-1 keywords")
    
    # Track results
    stats = {
        'total_articles': len(df),
        'with_firm': 0,
        'with_glp1': 0,
        'with_both': 0,
        'kept_articles': 0,
        'removed_articles': 0
    }
    
    # Check clean_text column (primary source for filtering)
    # But fall back to title/description if clean_text is missing
    text_column = 'clean_text' if 'clean_text' in df.columns else 'title'
    print(f"\nUsing '{text_column}' column for filtering")
    
    # Apply filters
    print("\nAnalyzing articles...")
    
    df['has_firm'] = False
    df['has_glp1'] = False
    df['detected_firm'] = None
    df['detected_keywords'] = None
    
    for idx, row in df.iterrows():
        text = row[text_column]
        
        # Check for firm
        has_firm, firm_name = contains_firm(text, FIRM_NAME_VARIATIONS)
        if has_firm:
            stats['with_firm'] += 1
            df.at[idx, 'has_firm'] = True
            df.at[idx, 'detected_firm'] = firm_name
        
        # Check for GLP-1 keyword
        has_glp1, keywords = contains_glp1_keyword(text, ALL_GLP1_KEYWORDS)
        if has_glp1:
            stats['with_glp1'] += 1
            df.at[idx, 'has_glp1'] = True
            df.at[idx, 'detected_keywords'] = ', '.join(keywords)
        
        # Progress
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1:,} articles...")
    
    print(f"✓ Analysis complete")
    
    # Filter to keep articles with BOTH conditions
    df_filtered = df[(df['has_firm']) & (df['has_glp1'])].copy()
    stats['with_both'] = len(df_filtered)
    stats['kept_articles'] = len(df_filtered)
    stats['removed_articles'] = len(df) - len(df_filtered)
    
    # Save filtered data
    print(f"\nSaving {len(df_filtered):,} filtered articles...")
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)
    df_filtered.to_csv(output_csv, index=False)
    print(f"✓ Saved to: {output_csv}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("FILTERING RESULTS")
    print("=" * 80)
    
    print(f"\nInput Articles: {stats['total_articles']:,}")
    print(f"  - With firm name: {stats['with_firm']:,} ({100*stats['with_firm']/stats['total_articles']:.1f}%)")
    print(f"  - With GLP-1 keyword: {stats['with_glp1']:,} ({100*stats['with_glp1']/stats['total_articles']:.1f}%)")
    print(f"  - With BOTH: {stats['with_both']:,} ({100*stats['with_both']/stats['total_articles']:.1f}%)")
    
    print(f"\nOutput Articles: {stats['kept_articles']:,}")
    print(f"Removed Articles: {stats['removed_articles']:,}")
    print(f"Retention Rate: {100*stats['kept_articles']/stats['total_articles']:.1f}%")
    
    # Firm distribution in filtered data
    print("\n" + "-" * 80)
    print("FIRMS IN FILTERED DATA (top 10)")
    print("-" * 80)
    
    firm_counts = df_filtered['detected_firm'].value_counts().head(10)
    for firm, count in firm_counts.items():
        pct = 100 * count / len(df_filtered)
        bar = '█' * int(pct / 2)
        print(f"  {firm:20} {count:5} ({pct:5.1f}%) {bar}")
    
    # Keyword distribution in filtered data
    print("\n" + "-" * 80)
    print("GLP-1 KEYWORDS IN FILTERED DATA (top 10 unique keywords)")
    print("-" * 80)
    
    # Parse detected keywords and count
    all_keywords_found = []
    for keywords_str in df_filtered['detected_keywords'].dropna():
        all_keywords_found.extend([k.strip() for k in keywords_str.split(',')])
    
    keyword_counts = pd.Series(all_keywords_found).value_counts().head(10)
    for keyword, count in keyword_counts.items():
        pct = 100 * count / len(df_filtered)
        bar = '█' * int(pct / 2)
        print(f"  {keyword:25} {count:5} ({pct:5.1f}%) {bar}")
    
    # Sample articles
    print("\n" + "-" * 80)
    print("SAMPLE ARTICLES")
    print("-" * 80)
    
    for idx, row in df_filtered.head(3).iterrows():
        print(f"\nFirm: {row['detected_firm']}")
        print(f"Keywords: {row['detected_keywords']}")
        print(f"Text: {row[text_column][:100]}...")
    
    # Quality metrics
    print("\n" + "-" * 80)
    print("QUALITY METRICS")
    print("-" * 80)
    
    if len(df_filtered) > 0:
        df_filtered['word_count'] = df_filtered[text_column].str.split().str.len()
        print(f"\nText Length (words):")
        print(f"  Min: {df_filtered['word_count'].min()}")
        print(f"  Max: {df_filtered['word_count'].max()}")
        print(f"  Avg: {df_filtered['word_count'].mean():.1f}")
        print(f"  Median: {df_filtered['word_count'].median():.0f}")
        
        # Valid length check
        if 'is_valid_length' in df_filtered.columns:
            valid = df_filtered['is_valid_length'].sum()
            invalid = (~df_filtered['is_valid_length']).sum()
            print(f"\nValidity:")
            print(f"  Valid: {valid:,} ({100*valid/len(df_filtered):.1f}%)")
            print(f"  Invalid: {invalid:,} ({100*invalid/len(df_filtered):.1f}%)")
        
        # Temporal coverage
        if 'published_date' in df_filtered.columns:
            df_filtered['published_date'] = pd.to_datetime(df_filtered['published_date'])
            print(f"\nTemporal Coverage:")
            print(f"  Oldest: {df_filtered['published_date'].min()}")
            print(f"  Newest: {df_filtered['published_date'].max()}")
            print(f"  Span: {(df_filtered['published_date'].max() - df_filtered['published_date'].min()).days} days")
    
    print("\n" + "=" * 80)
    print(f"✓ Filtering complete! Ready for sentiment analysis.")
    print("=" * 80)
    
    return df_filtered, stats


def main():
    """Main execution"""
    input_csv = 'data/processed/news_data_cleaned.csv'
    output_csv = 'data/processed/glp1_relevant.csv'
    
    df_filtered, stats = filter_glp1_relevant(input_csv, output_csv)
    
    if df_filtered is None:
        sys.exit(1)
    
    return df_filtered


if __name__ == '__main__':
    main()
