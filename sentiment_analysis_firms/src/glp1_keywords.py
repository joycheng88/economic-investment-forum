"""
GLP-1 related keywords and variations for sentiment analysis
Comprehensive dictionary of GLP-1 terms, medications, and related keywords
for text matching and classification
"""

# Primary GLP-1 keywords
GLP1_PRIMARY_KEYWORDS = [
    "GLP-1",
    "GLP1",
    "glucagon-like peptide-1",
    "glucagon like peptide 1",
]

# GLP-1 medications and brand names
GLP1_MEDICATIONS = [
    "Ozempic",
    "Wegovy",
    "Mounjaro",
    "Zepbound",
    "Victoza",
    "Saxenda",
    "Byetta",
    "Bydureon",
    "Trulicity",
    "Dulaglutide",
    "Semaglutide",
    "Tirzepatide",
    "Liraglutide",
    "Exenatide",
    "GLP1 agonist",
    "GLP-1 agonist",
    "GLP-1 receptor agonist",
    "GLP1 receptor agonist"
]

# Medical and clinical terms
GLP1_MEDICAL_TERMS = [
    "weight loss drug",
    "weight loss drugs",
    "obesity drug",
    "obesity drugs",
    "anti-obesity drug",
    "anti-obesity medication",
    "type 2 diabetes drug",
    "type 2 diabetes medication",
    "diabetes medication",
    "weight management",
    "appetite suppression",
    "appetite suppressant",
    "reduced calorie intake",
    "metabolic syndrome",
    "cardiovascular disease reduction",
    "hemoglobin A1C",
    "blood glucose control",
    "diabetes control"
]

# Related health conditions and topics
GLP1_HEALTH_TOPICS = [
    "obesity epidemic",
    "obesity crisis",
    "rising obesity rates",
    "weight gain treatment",
    "diabetes epidemic",
    "type 2 diabetes",
    "prediabetes",
    "metabolic disease",
    "cardiovascular risk",
    "chronic disease management",
    "lifestyle modification",
    "bariatric surgery",
    "gastric bypass",
    "weight-restrictive surgery"
]

# Market and business terms
GLP1_MARKET_TERMS = [
    "GLP-1 market",
    "GLP1 market",
    "obesity market",
    "weight loss market",
    "diabetes market",
    "blockbuster drug",
    "blockbuster medication",
    "weight loss industry",
    "obesity treatment market",
    "pharmaceutical innovation",
    "drug shortage",
    "drug availability",
    "healthcare spending",
    "insurance coverage",
    "Medicare coverage",
    "prescription demand"
]

# Celebrity and social references
GLP1_CELEBRITY_REFERENCES = [
    "weight loss celebrity",
    "celebrity weight loss",
    "celebrity using GLP-1",
    "celebrity on Ozempic",
    "Ozempic look",
    "semaglutide celebrity",
    "Hollywood weight loss",
    "red carpet weight loss"
]

# Adverse effects and concerns
GLP1_ADVERSE_EFFECTS = [
    "side effects",
    "nausea",
    "vomiting",
    "gastrointestinal",
    "GI side effects",
    "pancreatitis risk",
    "thyroid cancer",
    "medullary thyroid carcinoma",
    "drug safety",
    "safety concerns",
    "adverse events",
    "tolerability",
    "discontinuation rate",
    "GLP-1 jaw",
    "semaglutide jaw",
    "facial wasting",
    "muscle loss"
]

# Regulatory and approval terms
GLP1_REGULATORY_TERMS = [
    "FDA approval",
    "FDA authorized",
    "clinical trial",
    "clinical trials",
    "phase 3 trial",
    "efficacy data",
    "safety data",
    "regulatory approval",
    "EMA approval",
    "European approval",
    "off-label use",
    "off-label prescription",
    "black market",
    "tirzepatide approval"
]

# Economic and insurance terms
GLP1_ECONOMIC_TERMS = [
    "drug cost",
    "medication cost",
    "out-of-pocket cost",
    "insurance reimbursement",
    "coverage denial",
    "prior authorization",
    "formulary",
    "pharmaceutical pricing",
    "drug price",
    "cost-benefit analysis",
    "healthcare expenditure",
    "budget impact"
]

# Competitive and market dynamics
GLP1_COMPETITIVE_TERMS = [
    "GLP-1 competition",
    "GLP1 competition",
    "market competition",
    "competitor analysis",
    "market share",
    "first-mover advantage",
    "clinical differentiation",
    "once-weekly injection",
    "daily injection",
    "oral formulation",
    "combination therapy"
]

# Combine all keyword categories
ALL_GLP1_KEYWORDS = (
    GLP1_PRIMARY_KEYWORDS +
    GLP1_MEDICATIONS +
    GLP1_MEDICAL_TERMS +
    GLP1_HEALTH_TOPICS +
    GLP1_MARKET_TERMS +
    GLP1_CELEBRITY_REFERENCES +
    GLP1_ADVERSE_EFFECTS +
    GLP1_REGULATORY_TERMS +
    GLP1_ECONOMIC_TERMS +
    GLP1_COMPETITIVE_TERMS
)

# Dictionary mapping keyword categories for organization
GLP1_KEYWORD_CATEGORIES = {
    "primary": GLP1_PRIMARY_KEYWORDS,
    "medications": GLP1_MEDICATIONS,
    "medical_terms": GLP1_MEDICAL_TERMS,
    "health_topics": GLP1_HEALTH_TOPICS,
    "market_terms": GLP1_MARKET_TERMS,
    "celebrity_references": GLP1_CELEBRITY_REFERENCES,
    "adverse_effects": GLP1_ADVERSE_EFFECTS,
    "regulatory": GLP1_REGULATORY_TERMS,
    "economic": GLP1_ECONOMIC_TERMS,
    "competitive": GLP1_COMPETITIVE_TERMS
}


def normalize_keyword(keyword: str) -> str:
    """
    Normalize keyword to lowercase and strip whitespace.
    
    Args:
        keyword (str): Keyword to normalize
        
    Returns:
        str: Normalized keyword
    """
    return keyword.lower().strip()


def get_normalized_keywords() -> list:
    """
    Get all keywords in normalized (lowercase) form.
    
    Returns:
        list: All keywords normalized to lowercase
    """
    return [normalize_keyword(kw) for kw in ALL_GLP1_KEYWORDS]


def get_keyword_category(keyword: str) -> str:
    """
    Determine which category a keyword belongs to.
    
    Args:
        keyword (str): Keyword to categorize
        
    Returns:
        str: Category name, or None if not found
    """
    keyword_norm = normalize_keyword(keyword)
    for category, keywords in GLP1_KEYWORD_CATEGORIES.items():
        if keyword_norm in [normalize_keyword(kw) for kw in keywords]:
            return category
    return None


def contains_glp1_keyword(text: str) -> bool:
    """
    Check if text contains any GLP-1 related keyword.
    
    Args:
        text (str): Text to search
        
    Returns:
        bool: True if any GLP-1 keyword is found
    """
    text_norm = normalize_keyword(text)
    return any(normalize_keyword(kw) in text_norm for kw in ALL_GLP1_KEYWORDS)


def extract_glp1_keywords(text: str) -> dict:
    """
    Extract all GLP-1 keywords from text with category information.
    
    Args:
        text (str): Text to search
        
    Returns:
        dict: Mapping of found keywords to their categories
    """
    text_norm = normalize_keyword(text)
    found_keywords = {}
    
    for keyword in ALL_GLP1_KEYWORDS:
        kw_norm = normalize_keyword(keyword)
        if kw_norm in text_norm:
            category = get_keyword_category(keyword)
            if keyword not in found_keywords:  # Avoid duplicates
                found_keywords[keyword] = category
    
    return found_keywords


def extract_glp1_keywords_by_category(text: str) -> dict:
    """
    Extract GLP-1 keywords organized by category.
    
    Args:
        text (str): Text to search
        
    Returns:
        dict: Mapping of categories to found keywords
    """
    text_norm = normalize_keyword(text)
    results_by_category = {}
    
    for category, keywords in GLP1_KEYWORD_CATEGORIES.items():
        found = []
        for keyword in keywords:
            kw_norm = normalize_keyword(keyword)
            if kw_norm in text_norm and keyword not in found:
                found.append(keyword)
        if found:
            results_by_category[category] = found
    
    return results_by_category


def count_glp1_keyword_mentions(text: str) -> int:
    """
    Count total number of GLP-1 keyword mentions in text.
    Note: This counts actual occurrences, not unique keywords.
    
    Args:
        text (str): Text to search
        
    Returns:
        int: Total number of keyword mentions
    """
    text_norm = normalize_keyword(text)
    count = 0
    
    for keyword in ALL_GLP1_KEYWORDS:
        kw_norm = normalize_keyword(keyword)
        count += text_norm.count(kw_norm)
    
    return count


def get_sentiment_context(keyword: str) -> str:
    """
    Determine likely sentiment context for a keyword.
    
    Args:
        keyword (str): Keyword to analyze
        
    Returns:
        str: Sentiment context ('positive', 'negative', 'neutral', 'mixed')
    """
    category = get_keyword_category(keyword)
    
    if category == "positive_effects":
        return "positive"
    elif category == "adverse_effects":
        return "negative"
    elif category in ["medications", "primary", "medical_terms", "health_topics"]:
        return "neutral"
    else:
        return "mixed"


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 60)
    print("GLP-1 KEYWORD ANALYSIS")
    print("=" * 60)
    
    # Print all categories
    print("\nKeyword Categories:")
    for category, keywords in GLP1_KEYWORD_CATEGORIES.items():
        print(f"\n{category.upper()} ({len(keywords)} keywords):")
        print(f"  {', '.join(keywords[:3])}...")
    
    print(f"\nTotal GLP-1 keywords: {len(ALL_GLP1_KEYWORDS)}")
    
    # Test functions
    print("\n" + "=" * 60)
    print("FUNCTION TESTS")
    print("=" * 60)
    
    test_texts = [
        "Ozempic is a popular GLP-1 medication for weight loss.",
        "Semaglutide and tirzepatide are competing for market dominance.",
        "FDA approved Zepbound for obesity treatment.",
        "Patients experienced nausea and vomiting as side effects.",
        "This text has no relevant keywords here."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text}")
        print(f"  Contains GLP-1 keyword: {contains_glp1_keyword(text)}")
        print(f"  Mention count: {count_glp1_keyword_mentions(text)}")
        keywords = extract_glp1_keywords(text)
        if keywords:
            print(f"  Keywords found: {list(keywords.keys())[:3]}")
        by_category = extract_glp1_keywords_by_category(text)
        if by_category:
            print(f"  Categories: {list(by_category.keys())}")
    
    # Print keyword statistics
    print("KEYWORD STATISTICS")
    print(f"Primary keywords: {len(GLP1_PRIMARY_KEYWORDS)}")
    print(f"Medications: {len(GLP1_MEDICATIONS)}")
    print(f"Medical terms: {len(GLP1_MEDICAL_TERMS)}")
    print(f"Health topics: {len(GLP1_HEALTH_TOPICS)}")
    print(f"Market terms: {len(GLP1_MARKET_TERMS)}")
    print(f"Celebrity references: {len(GLP1_CELEBRITY_REFERENCES)}")
    print(f"Adverse effects: {len(GLP1_ADVERSE_EFFECTS)}")
    print(f"Regulatory terms: {len(GLP1_REGULATORY_TERMS)}")
    print(f"Economic terms: {len(GLP1_ECONOMIC_TERMS)}")
    print(f"Competitive terms: {len(GLP1_COMPETITIVE_TERMS)}")
    print(f"\nTotal unique keywords: {len(set(ALL_GLP1_KEYWORDS))}")
