"""
Drug-Firm Matching Module for GLP-1 Sentiment Analysis

Provides functions for:
- Tagging documents with drug mentions
- Mapping drugs to pharmaceutical firms
- Calculating firm-level sentiment indices
"""

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple


class DrugFirmMatcher:
    """Core class for matching drugs and firms in GLP-1 documents"""

    # Drug mapping: canonical name -> brand names and aliases
    DRUG_MAP = {
        "semaglutide": ["semaglutide", "ozempic", "wegovy", "rybelsus"],
        "tirzepatide": ["tirzepatide", "mounjaro", "zepbound"],
        "liraglutide": ["liraglutide", "saxenda", "victoza"],
        "dulaglutide": ["dulaglutide", "trulicity"],
        "danuglipron": ["danuglipron"],
        "maritide": ["maritide", "amg133"],
        "vk2735": ["vk2735"],
        "ct388": ["ct388", "carmot"],
        "ct996": ["ct996"],
        "glp1_gip": ["glp-1/gip", "glp1/gip", "dual-acting"]
    }

    # Firm mapping: drug -> manufacturer (top 10 firms)
    FIRM_MAP = {
        "semaglutide": "Novo Nordisk A/S",
        "liraglutide": "Novo Nordisk A/S",
        "dulaglutide": "Eli Lilly and Company",
        "tirzepatide": "Eli Lilly and Company",
        "maritide": "Amgen Inc.",
        "danuglipron": "Pfizer Inc.",
        "ct388": "Roche Holding AG",
        "ct996": "Roche Holding AG",
        "glp1_gip": "AstraZeneca Plc",
        "vk2735": "Viking Therapeutics, Inc.",
    }

    # Top 10 firms with metadata
    TOP_10_FIRMS = {
        "Novo Nordisk A/S": {
            "strength": "Market leader with approved obesity drug (Wegovy)",
            "drugs": ["semaglutide", "liraglutide"],
        },
        "Eli Lilly and Company": {
            "strength": "Fast-growing competitor with strong obesity portfolio",
            "drugs": ["tirzepatide", "dulaglutide"],
        },
        "Amgen Inc.": {
            "strength": "Novel next-generation candidate",
            "drugs": ["maritide"],
        },
        "Pfizer Inc.": {
            "strength": "Oral formulation innovation",
            "drugs": ["danuglipron"],
        },
        "Roche Holding AG": {
            "strength": "Aggressive M&A strategy, both injectable and oral",
            "drugs": ["ct388", "ct996"],
        },
        "AstraZeneca Plc": {
            "strength": "Cardiovascular and metabolic expertise",
            "drugs": ["glp1_gip"],
        },
        "Zealand Pharma A/S": {
            "strength": "Acquisition target potential",
            "drugs": [],
        },
        "Structure Therapeutics, Inc.": {
            "strength": "Oral formulation specialization",
            "drugs": [],
        },
        "Viking Therapeutics, Inc.": {
            "strength": "Dual-mechanism approach with positive trial data",
            "drugs": ["vk2735"],
        },
        "Boehringer Ingelheim GmbH": {
            "strength": "Cardiometabolic integration",
            "drugs": [],
        },
    }

    # Sentiment keywords
    POSITIVE_KEYWORDS = [
        "positive", "promising", "success", "breakthrough", "approved", "efficacy",
        "superior", "outperform", "excellence", "innovation", "leading",
        "strong", "powerful", "effective", "impressive", "remarkable",
        "gain", "momentum", "surge", "expansion", "growth", "advance", "progress"
    ]

    NEGATIVE_KEYWORDS = [
        "negative", "declined", "failed", "setback", "adverse", "concern",
        "risk", "delay", "loss", "weak", "inferior", "criticism", "competition",
        "challenge", "pressure", "loss share", "decline", "reduction"
    ]

    TRIAL_KEYWORDS = [
        "trial", "clinical", "data", "phase", "fda", "approval", "approved",
        "efficacy", "safety", "endpoints", "results", "significant"
    ]

    def __init__(self):
        """Initialize the matcher"""
        self.drug_map = self.DRUG_MAP
        self.firm_map = self.FIRM_MAP
        self.top_10_firms = list(self.TOP_10_FIRMS.keys())

    def tag_drug(self, text: str) -> List[str]:
        """
        Identify all drug mentions in text

        Args:
            text: Text to search for drug mentions

        Returns:
            List of canonical drug names found (deduplicated)
        """
        if pd.isna(text):
            return []

        text = str(text).lower()
        found = []

        for drug, keywords in self.drug_map.items():
            for keyword in keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", text):
                    found.append(drug)
                    break

        return list(set(found))

    def match_documents(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """
        Match documents to drugs and firms

        Args:
            df: Input DataFrame
            text_column: Name of text column to analyze

        Returns:
            DataFrame with added 'drugs' and 'firm' columns
        """
        # Tag drugs
        df = df.copy()
        df["drugs"] = df[text_column].apply(self.tag_drug)

        # Expand rows for multi-drug documents
        df = df.explode("drugs").reset_index(drop=True)

        # Map to firms
        df["firm"] = df["drugs"].map(self.firm_map)

        # Filter to top 10 firms
        df = df[df["firm"].isin(self.top_10_firms)].copy()

        # Remove NaN firms
        df = df.dropna(subset=["firm"])

        return df

    def extract_sentiment_features(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """
        Extract sentiment features from text

        Args:
            df: Input DataFrame
            text_column: Name of text column to analyze

        Returns:
            DataFrame with added sentiment feature columns
        """
        df = df.copy()

        def _extract_features(text):
            text_lower = str(text).lower()
            return {
                "positive_count": sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text_lower),
                "negative_count": sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in text_lower),
                "trial_count": sum(1 for kw in self.TRIAL_KEYWORDS if kw in text_lower),
                "has_clinical_data": any(kw in text_lower for kw in self.TRIAL_KEYWORDS),
            }

        features_df = pd.DataFrame(
            df[text_column].apply(_extract_features).tolist()
        )
        df = pd.concat([df, features_df], axis=1)

        # Create composite scores
        df["sentiment_net"] = df["positive_count"] - df["negative_count"]
        df["sentiment_ratio"] = (df["positive_count"] + 1) / (df["negative_count"] + 1)
        df["content_quality"] = df["trial_count"] * 2 + (df[text_column].str.len() / 100)

        return df

    def calculate_sentiment_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate firm-level sentiment indices

        Args:
            df: DataFrame with matched documents and sentiment features

        Returns:
            DataFrame with firm-level sentiment metrics
        """
        if len(df) == 0:
            return pd.DataFrame()

        # Group by firm
        agg_dict = {
            "text": "count",  # or use appropriate column
            "sentiment_net": ["mean", "std", "min", "max"],
            "positive_count": "mean",
            "negative_count": "mean",
            "has_clinical_data": ["sum", "mean"],
            "content_quality": "mean",
        }

        # Only use columns that exist
        agg_dict_filtered = {}
        for col, agg in agg_dict.items():
            if col in df.columns:
                agg_dict_filtered[col] = agg
            elif col == "text" and "title" in df.columns:
                agg_dict_filtered["title"] = agg
                break

        sentiment_index = df.groupby("firm").agg(agg_dict_filtered).round(3)

        # Flatten column names
        sentiment_index.columns = ["_".join(col).strip("_") for col in sentiment_index.columns.values]

        # Rename for clarity
        if "text_count" in sentiment_index.columns:
            sentiment_index = sentiment_index.rename(columns={"text_count": "article_count"})
        elif "title_count" in sentiment_index.columns:
            sentiment_index = sentiment_index.rename(columns={"title_count": "article_count"})

        # Calculate ratios if data available
        if "has_clinical_data_sum" in sentiment_index.columns:
            sentiment_index["clinical_articles"] = sentiment_index["has_clinical_data_sum"].astype(int)
            sentiment_index["clinical_ratio"] = sentiment_index["has_clinical_data_mean"]

        if "positive_count_mean" in sentiment_index.columns:
            sentiment_index["positive_ratio"] = (
                (df.groupby("firm")["positive_count"] > 0).sum() / sentiment_index["article_count"]
            ).values

        # Create composite sentiment index
        if "sentiment_net_mean" in sentiment_index.columns and "positive_ratio" in sentiment_index.columns:
            sentiment_index["sentiment_index"] = (
                sentiment_index["sentiment_net_mean"] * 0.4 +
                sentiment_index.get("positive_ratio", 0) * 100 * 0.3 +
                sentiment_index.get("clinical_ratio", 0) * 100 * 0.3
            ).round(2)
            sentiment_index = sentiment_index.sort_values("sentiment_index", ascending=False)

        return sentiment_index


def simple_match_and_score(df: pd.DataFrame, text_column: str = "text") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function for complete matching and scoring pipeline

    Args:
        df: Input DataFrame with text column
        text_column: Name of text column to analyze

    Returns:
        Tuple of (matched_documents_df, sentiment_index_df)
    """
    matcher = DrugFirmMatcher()

    # Match documents to drugs/firms
    df_matched = matcher.match_documents(df, text_column)

    # Extract sentiment features
    df_matched = matcher.extract_sentiment_features(df_matched, text_column)

    # Calculate sentiment index
    sentiment_index = matcher.calculate_sentiment_index(df_matched)

    return df_matched, sentiment_index


if __name__ == "__main__":
    # Example usage
    matcher = DrugFirmMatcher()

    # Test texts
    test_texts = [
        "Ozempic and Wegovy are semaglutide drugs from Novo Nordisk",
        "Mounjaro tirzepatide shows promise for weight loss",
        "No drug mentions here",
    ]

    # Test tagging
    print("Testing drug tagging:")
    for text in test_texts:
        drugs = matcher.tag_drug(text)
        print(f"  '{text[:50]}...' → {drugs}")
