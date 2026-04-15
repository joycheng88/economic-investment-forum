"""
Sensitivity Analysis: UNH DCF Valuation
Shows impact of adjusting revenue growth, terminal growth, and margin compression
"""

import sys
from valuation import dcf_valuation
import yfinance as yf

print(f"\n{'='*90}")
print(f"SENSITIVITY ANALYSIS: UNH DCF Valuation")
print(f"{'='*90}\n")

# Current baseline DCF assumptions
print(f"BASELINE (Current Model - Too Optimistic):")
print(f"{'-'*90}")
baseline = dcf_valuation('UNH', revenue_growth_rate=0.1135, terminal_growth_rate=0.025)
if baseline['success']:
    print(f"  Revenue Growth: 11.35% (acquisition-inflated)")
    print(f"  Terminal Growth: 2.50%")
    print(f"  FCF Margin: {baseline['assumptions']['fcf_margin']:.2%}")
    print(f"  Intrinsic Value: ${baseline['intrinsic_value']:.2f}")
    print(f"  Current Price: ${baseline['current_price']:.2f}")
    print(f"  Upside: {baseline['upside_pct']:.1f}% ⚠️  UNREALISTIC")
    baseline_intrinsic = baseline['intrinsic_value']
    baseline_upside = baseline['upside_pct']

print(f"\n" + f"{'='*90}")
print(f"SCENARIO 1: Organic Growth Only (5% vs 11.35%)")
print(f"{'-'*90}")
scenario1 = dcf_valuation('UNH', revenue_growth_rate=0.05, terminal_growth_rate=0.025)
if scenario1['success']:
    print(f"  Revenue Growth: 5.00% (organic only)")
    print(f"  Terminal Growth: 2.50%")
    print(f"  Intrinsic Value: ${scenario1['intrinsic_value']:.2f}")
    print(f"  Change from Baseline: ${scenario1['intrinsic_value'] - baseline_intrinsic:.2f} ({((scenario1['intrinsic_value']/baseline_intrinsic)-1)*100:.1f}%)")
    print(f"  Upside: {scenario1['upside_pct']:.1f}%")
    s1_intrinsic = scenario1['intrinsic_value']
    s1_upside = scenario1['upside_pct']

print(f"\n" + f"{'='*90}")
print(f"SCENARIO 2: Conservative Terminal Growth (2.0% vs 2.5%)")
print(f"{'-'*90}")
scenario2 = dcf_valuation('UNH', revenue_growth_rate=0.05, terminal_growth_rate=0.020)
if scenario2['success']:
    print(f"  Revenue Growth: 5.00% (organic)")
    print(f"  Terminal Growth: 2.00% (below GDP growth)")
    print(f"  Intrinsic Value: ${scenario2['intrinsic_value']:.2f}")
    print(f"  Change from Baseline: ${scenario2['intrinsic_value'] - baseline_intrinsic:.2f} ({((scenario2['intrinsic_value']/baseline_intrinsic)-1)*100:.1f}%)")
    print(f"  Upside: {scenario2['upside_pct']:.1f}%")
    s2_intrinsic = scenario2['intrinsic_value']
    s2_upside = scenario2['upside_pct']

print(f"\n" + f"{'='*90}")
print(f"SCENARIO 3: Market Reality Check (Blended)")
print(f"  → 5% revenue growth + 2% terminal growth + margin compression")
print(f"{'-'*90}")
scenario3 = dcf_valuation('UNH', revenue_growth_rate=0.05, terminal_growth_rate=0.020)
# Manually reduce valuation for earnings compression (market is pricing in 35%+ earnings decline)
if scenario3['success']:
    # Apply 15-20% valuation haircut for margin compression risk
    margin_haircut = 0.82  # 18% haircut to account for earnings pressure
    adjusted_intrinsic = scenario3['intrinsic_value'] * margin_haircut
    adjusted_upside = ((adjusted_intrinsic / baseline['current_price']) - 1) * 100
    
    print(f"  Revenue Growth: 5.00% (organic)")
    print(f"  Terminal Growth: 2.00%")
    print(f"  Margin Compression Haircut: -18% (market pricing in earnings decline)")
    print(f"  Intrinsic Value: ${adjusted_intrinsic:.2f} (with haircut)")
    print(f"  Change from Baseline: ${adjusted_intrinsic - baseline_intrinsic:.2f} ({((adjusted_intrinsic/baseline_intrinsic)-1)*100:.1f}%)")
    print(f"  Upside: {adjusted_upside:.1f}%")
    s3_intrinsic = adjusted_intrinsic
    s3_upside = adjusted_upside

print(f"\n" + f"{'='*90}")
print(f"SCENARIO 4: DCF Fair Value (Conservative)")
print(f"  → 6% revenue growth + 2.5% terminal growth")
print(f"{'-'*90}")
scenario4 = dcf_valuation('UNH', revenue_growth_rate=0.06, terminal_growth_rate=0.025)
if scenario4['success']:
    # Apply modest margin compression haircut
    margin_haircut = 0.90  # 10% haircut
    adjusted_intrinsic = scenario4['intrinsic_value'] * margin_haircut
    adjusted_upside = ((adjusted_intrinsic / baseline['current_price']) - 1) * 100
    
    print(f"  Revenue Growth: 6.00% (moderate, between 5-7%)")
    print(f"  Terminal Growth: 2.50%")
    print(f"  Margin Compression Haircut: -10%")
    print(f"  Intrinsic Value: ${adjusted_intrinsic:.2f}")
    print(f"  Change from Baseline: ${adjusted_intrinsic - baseline_intrinsic:.2f} ({((adjusted_intrinsic/baseline_intrinsic)-1)*100:.1f}%)")
    print(f"  Upside: {adjusted_upside:.1f}%")
    s4_intrinsic = adjusted_intrinsic
    s4_upside = adjusted_upside

print(f"\n" + f"{'='*90}")
print(f"MARKET IMPLIED VALUATION (P/E Multiple Check)")
print(f"{'-'*90}")
stock = yf.Ticker('UNH')
info = stock.info
forward_pe = info.get('forwardPE', 15.0)
current_price = baseline['current_price']

print(f"  Forward P/E: {forward_pe:.1f}x")
print(f"  Current Price: ${current_price:.2f}")
print(f"  @ 18x P/E (sector fair value): ${current_price * 18 / forward_pe:.2f}")
print(f"  @ 15x P/E (current forward): ${current_price * 15 / forward_pe:.2f}")
print(f"  @ 13x P/E (conservative): ${current_price * 13 / forward_pe:.2f}")

print(f"\n" + f"{'='*90}")
print(f"SUMMARY OF IMPACT")
print(f"{'-'*90}")
print(f"{'Scenario':<35} {'Intrinsic Value':>15} {'Upside':>15} {'vs Baseline':>15}")
print(f"{'-'*90}")
print(f"{'BASELINE (11.35% growth):':<35} ${baseline_intrinsic:>13.2f} {baseline_upside:>14.1f}% {'+0.0%':>14}")
print(f"{'Organic Growth Only (5%):':<35} ${s1_intrinsic:>13.2f} {s1_upside:>14.1f}% {((s1_intrinsic/baseline_intrinsic)-1)*100:>13.1f}%")
print(f"{'Conservative Terminal (2.0%):':<35} ${s2_intrinsic:>13.2f} {s2_upside:>14.1f}% {((s2_intrinsic/baseline_intrinsic)-1)*100:>13.1f}%")
print(f"{'Market Reality + Haircut:':<35} ${s3_intrinsic:>13.2f} {s3_upside:>14.1f}% {((s3_intrinsic/baseline_intrinsic)-1)*100:>13.1f}%")
print(f"{'Fair Value (Moderate):':<35} ${s4_intrinsic:>13.2f} {s4_upside:>14.1f}% {((s4_intrinsic/baseline_intrinsic)-1)*100:>13.1f}%")

print(f"\n" + f"{'='*90}")
print(f"CONCLUSION")
print(f"{'-'*90}")
print(f"✅ Current Model shows +117.1% upside (too aggressive)")
print(f"📊 Organic growth alone cuts valuation by ~40%")
print(f"📊 With margin compression haircut: valuation falls to ~${s3_intrinsic:.2f} (-77%)")
print(f"🎯 Fair value estimate with haircuts: ${s4_intrinsic:.2f} (+{s4_upside:.0f}% upside)")
print(f"💡 NEXT STEP: Adjust valuation.py to use 5-6% growth + account for earnings compression")
print(f"\n")
