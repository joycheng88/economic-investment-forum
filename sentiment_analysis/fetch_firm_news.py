#!/usr/bin/env python3
"""
Script to fetch firm-specific news for all 10 GLP-1 companies
and save each as individual CSV files

Usage:
    python fetch_firm_news.py
    python fetch_firm_news.py --days 30 --output firm_data
"""

import argparse
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and save high-quality firm-specific GLP-1 news"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days to look back (default: 90)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="firm_data",
        help="Output directory for CSV files (default: firm_data)"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=500,
        help="Target minimum articles per firm (default: 500)"
    )
    parser.add_argument(
        "--check-key",
        action="store_true",
        help="Check if NEWS_API_KEY is configured"
    )
    
    args = parser.parse_args()
    
    # Check API key if requested
    if args.check_key:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("NEWS_API_KEY", "")
        if api_key:
            print("✓ NEWS_API_KEY is configured")
            return 0
        else:
            print("✗ NEWS_API_KEY is NOT configured")
            print("\nTo use this script, you need to:")
            print("1. Get a free API key from: https://newsapi.org")
            print("2. Add it to .env file as: NEWS_API_KEY=your_key")
            return 1
    
    # Check if API key exists before fetching
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key:
        print("\n" + "="*60)
        print("ERROR: NEWS_API_KEY not configured!")
        print("="*60)
        print("\nTo fetch firm-specific news, you need:")
        print("1. A free API key from: https://newsapi.org")
        print("2. Edit the .env file and add your key:")
        print("   NEWS_API_KEY=your_actual_key_here")
        print("\n" + "="*60)
        return 1
    
    print("\n" + "="*60)
    print("GLP-1 Firm-Specific News Fetcher (Drug/Firm Assignment)")
    print("="*60)
    print(f"\nStrategy: Fetch all GLP-1 news, then assign to firms by drug mapping")
    print(f"Data range: Last {args.days} days")
    print(f"Output directory: {args.output}\n")
    
    try:
        from data_ingestion import fetch_and_assign_firm_news
        firm_data = fetch_and_assign_firm_news(
            days_back=args.days,
            output_dir=args.output,
        )
        
        print("\n" + "="*60)
        print("SUCCESS: Firm-specific news fetched and saved!")
        print("="*60)
        
        # Print summary
        print(f"\nFiles saved in: {args.output}/")
        print("\nGenerated CSV files:")
        
        firm_names = [
            "Novo Nordisk", "Eli Lilly", "Amgen", "Pfizer", "Roche",
            "AstraZeneca", "Zealand Pharma", "Structure Therapeutics",
            "Viking Therapeutics", "Boehringer Ingelheim"
        ]
        
        for firm_name in firm_names:
            filename = firm_name.lower().replace(" ", "_").replace("/", "_") + ".csv"
            filepath = Path(args.output) / filename
            if filepath.exists():
                # Count rows in CSV
                import pandas as pd
                df = pd.read_csv(filepath)
                print(f"  ✓ {filename:<40} ({len(df):>4} articles)")
            else:
                print(f"  ✗ {filename:<40} (not created)")
        
        print("\nYou can now use these CSV files for:")
        print("  - Sentiment analysis")
        print("  - News trend analysis")
        print("  - Word frequency analysis")
        print("  - Time series visualization")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: Failed to fetch firm-specific news")
        print(f"Details: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
