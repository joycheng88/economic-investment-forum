#!/usr/bin/env python3
"""Test that app.py page_modeling() calls use correct parameter names"""

import re
import sys

print("=" * 70)
print("APP.PY PARAMETER NAME VERIFICATION")
print("=" * 70)

with open('app.py', 'r') as f:
    app_content = f.read()

# Find the page_modeling function
page_modeling_match = re.search(r'def page_modeling\(\):.*?(?=\ndef |$)', app_content, re.DOTALL)
if not page_modeling_match:
    print("✗ page_modeling() function not found!")
    sys.exit(1)

page_modeling_code = page_modeling_match.group(0)
print("\n✓ page_modeling() function found")

# Check for DCFValuation calls
dcf_calls = re.findall(r'dcf_analysis\.run_analysis\([^)]+\)', page_modeling_code)
print(f"\nFound {len(dcf_calls)} DCFValuation.run_analysis() call(s):")

issues = []
for i, call in enumerate(dcf_calls, 1):
    print(f"\n  Call {i}:")
    # Check for old parameter names
    if 'terminal_growth_rate=' in call:
        print(f"    ✗ FOUND: terminal_growth_rate= (should be terminal_growth=)")
        issues.append("terminal_growth_rate parameter")
    else:
        print(f"    ✓ Not using terminal_growth_rate=")
    
    if 'revenue_growth_rate=' in call:
        print(f"    ✗ FOUND: revenue_growth_rate= (should be revenue_growth=)")
        issues.append("revenue_growth_rate parameter")
    else:
        print(f"    ✓ Not using revenue_growth_rate=")
    
    if 'terminal_growth=' in call:
        print(f"    ✓ Correctly using terminal_growth=")
    if 'revenue_growth=' in call:
        print(f"    ✓ Correctly using revenue_growth=")

# Check sensitivity analysis calls
sensitivity_calls = re.findall(r'dcf\.run_analysis\([^)]+\)', page_modeling_code)
print(f"\nFound {len(sensitivity_calls)} sensitivity analysis call(s)")

print("\n" + "=" * 70)
if issues:
    print(f"✗ FOUND {len(issues)} ISSUE(S):")
    for issue in issues:
        print(f"  - {issue}")
    sys.exit(1)
else:
    print("✓ ALL PARAMETER NAMES ARE CORRECT!")
    print("✓ page_modeling() is ready to use!")
print("=" * 70)
