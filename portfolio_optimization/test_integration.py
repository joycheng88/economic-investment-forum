#!/usr/bin/env python3
"""Quick integration test for DCFValuation parameter names"""

import inspect
from modeling import DCFValuation, ComparableCompanies, ValuationReport

print("=" * 60)
print("MODELING.PY INTEGRATION TEST")
print("=" * 60)

# Test 1: Check DCFValuation.run_analysis signatures
dcf = DCFValuation('AAPL')
sig = inspect.signature(dcf.run_analysis)
params = list(sig.parameters.keys())

print("\n1. DCFValuation.run_analysis() parameters:")
for i, param in enumerate(params, 1):
    print(f"   {i}. {param}")

# Test 2: Verify key parameter names exist
required_params = ['wacc', 'terminal_growth', 'forecast_years', 'risk_free_rate', 
                   'market_risk_premium', 'tax_rate', 'cost_of_debt', 'revenue_growth']

print("\n2. Required parameter check:")
all_present = True
for param in required_params:
    present = param in params
    status = "✓" if present else "✗"
    print(f"   {status} {param}")
    if not present:
        all_present = False

# Test 3: Verify parameters are NOT using old names
old_params_to_avoid = ['terminal_growth_rate', 'revenue_growth_rate', 'forecast_years_rate']

print("\n3. Avoid old parameter names:")
no_old_params = True
for old_param in old_params_to_avoid:
    if old_param in params:
        print(f"   ✗ FOUND OLD PARAM: {old_param}")
        no_old_params = False
    else:
        print(f"   ✓ Not using {old_param}")

print("\n" + "=" * 60)
if all_present and no_old_params:
    print("✓ ALL TESTS PASSED - Ready to use!")
else:
    print("✗ TESTS FAILED - Check parameter names")
print("=" * 60)
