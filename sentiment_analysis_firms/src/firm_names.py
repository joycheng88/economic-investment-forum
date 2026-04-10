"""
Firm names and variations for sentiment analysis
Maps snack company names to possible text variations for NER and matching
"""

# List of snack companies
SNACK_COMPANIES = [
    "RXBAR",
    "Chomps",
    "Wonderful",
    "General Mills",
    "PepsiCo",
    "Nestle",
    "Mars",
    "Mondelez",
    "Ferrero",
    "Hershey"
]

# Dictionary mapping firms to name variations
FIRM_NAME_VARIATIONS = {
    "RXBAR": [
        "RXBAR",
        "RxBar",
        "Rx Bar",
        "RX Bar"
    ],
    "Chomps": [
        "Chomps",
        "chomps",
        "CHOMPS"
    ],
    "Wonderful": [
        "Wonderful",
        "Wonderful Company",
        "The Wonderful Company",
        "Wonderful Pistachios",
        "Wonderful Almonds",
        "WONDERFUL"
    ],
    "General Mills": [
        "General Mills",
        "GeneralMills",
        "General Mills, Inc.",
        "GM",
        "GM Inc.",
        "GENERAL MILLS"
    ],
    "PepsiCo": [
        "PepsiCo",
        "Pepsi Co",
        "Pepsi-Co",
        "PepsiCo, Inc.",
        "Pepsi",
        "Frito-Lay",
        "Frito Lay",
        "Doritos",
        "Lay's",
        "Cheetos",
        "Fritos",
        "Quaker",
        "PEPSICO"
    ],
    "Nestle": [
        "Nestle",
        "Nestlé",
        "Nestle SA",
        "Nestle S.A.",
        "Nestle Group",
        "Nescafé",
        "KitKat",
        "Aero",
        "Butterfinger",
        "Baby Ruth",
        "NESTLE"
    ],
    "Mars": [
        "Mars",
        "Mars Inc.",
        "Mars, Incorporated",
        "Mars Confectionery",
        "M&M",
        "M&M's",
        "Snickers",
        "Milky Way",
        "Twix",
        "3 Musketeers",
        "MARS"
    ],
    "Mondelez": [
        "Mondelez",
        "Mondelēz",
        "Mondelez International",
        "Mondelez, Inc.",
        "Oreo",
        "Trident",
        "Cadbury",
        "Toblerone",
        "Chiclets",
        "Halls",
        "MONDELEZ"
    ],
    "Ferrero": [
        "Ferrero",
        "Ferrero Rocher",
        "Ferrero Group",
        "Ferrero SpA",
        "Nutella",
        "Tic Tac",
        "Kinder",
        "Ferrero Kinder",
        "FERRERO"
    ],
    "Hershey": [
        "Hershey",
        "The Hershey Company",
        "Hershey Co.",
        "Hershey Foods",
        "Hershey's",
        "Reese's",
        "Reese",
        "Kisses",
        "Twizzlers",
        "Lancaster",
        "HERSHEY"
    ]
}


def get_all_variations(firm_name: str) -> list:
    """
    Get all name variations for a given firm.
    
    Args:
        firm_name (str): The firm name to look up
        
    Returns:
        list: List of all name variations, or empty list if firm not found
    """
    return FIRM_NAME_VARIATIONS.get(firm_name, [])


def get_firm_from_variation(variation: str) -> str:
    """
    Get the canonical firm name from a variation.
    
    Args:
        variation (str): A potential name variation
        
    Returns:
        str: The canonical firm name, or None if variation not found
    """
    for firm, variations in FIRM_NAME_VARIATIONS.items():
        if variation.lower() in [v.lower() for v in variations]:
            return firm
    return None


def is_snack_company(text: str) -> bool:
    """
    Check if text contains any snack company name or variation.
    
    Args:
        text (str): Text to check
        
    Returns:
        bool: True if any company name variation is found
    """
    text_lower = text.lower()
    for firm, variations in FIRM_NAME_VARIATIONS.items():
        for variation in variations:
            if variation.lower() in text_lower:
                return True
    return False


def extract_firms_from_text(text: str) -> list:
    """
    Extract all snack companies mentioned in text.
    
    Args:
        text (str): Text to search
        
    Returns:
        list: Canonical firm names found in text (deduplicated)
    """
    firms_found = set()
    text_lower = text.lower()
    
    for firm, variations in FIRM_NAME_VARIATIONS.items():
        for variation in variations:
            if variation.lower() in text_lower:
                firms_found.add(firm)
    
    return sorted(list(firms_found))


if __name__ == "__main__":
    # Example usage
    print("Snack Companies:")
    for i, company in enumerate(SNACK_COMPANIES, 1):
        print(f"{i}. {company}")
    
    print("\n" + "="*50)
    print("Name Variations per Company:")
    print("="*50)
    for firm, variations in FIRM_NAME_VARIATIONS.items():
        print(f"\n{firm}: {len(variations)} variations")
        print(f"  {', '.join(variations[:3])}...")
    
    # Test functions
    print("\n" + "="*50)
    print("Function Tests:")
    print("="*50)
    
    test_text = "PepsiCo and Frito-Lay are expanding their snack lines. Nestle and KitKat are competing."
    print(f"\nTest text: {test_text}")
    print(f"Contains snack company: {is_snack_company(test_text)}")
    print(f"Companies found: {extract_firms_from_text(test_text)}")
    
    print(f'\nVariation "M&M\'s" maps to: {get_firm_from_variation("M&M\'s")}')
    print(f"Variation 'Frito Lay' maps to: {get_firm_from_variation('Frito Lay')}")
