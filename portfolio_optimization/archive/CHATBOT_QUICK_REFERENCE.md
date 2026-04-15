# Chatbot Quick Reference - Questions Now Supported

## 📈 Stock Analysis Questions

### With Portfolio Context
**Example**: "Should I add MSFT given my current holdings?"

Returns:
- Correlation to your portfolio
- Sector overlap assessment
- Quality score (ROE, debt, liquidity)
- **Suggested allocation**: specific %

---

**Example**: "Is TSLA a good investment based on my portfolio?"

Returns:
- Diversification value (correlation)
- Fundamentals analysis
- **Sizing recommendation** with rebalancing triggers

---

## 🏭 Sector & Industry Questions

**Example**: "How is the semiconductor industry performing?"

Returns:
- Real-time sector ETF price
- YTD return vs S&P 500
- 1-month, 3-month momentum
- Technical position (vs moving averages)
- Volatility analysis

---

**Example**: "What's happening in the technology sector?"

Returns:
- Live tech sector (XLK) performance
- Performance table (1m, 3m, YTD, 1y returns)
- Technical trend analysis
- **Up/Down positioning** vs major indices

---

## 📊 Market Outlook Questions

**Example**: "What's the overall market outlook?"

Returns:
- S&P 500, Nasdaq, Russell 2000 performance
- International (EFA) and emerging markets (EEM)
- Bonds (TLT) and commodities (DBC, gold) 
- **Asset class comparison** across time horizons
- Volatility regime assessment
- Breadth interpretation

---

**Example**: "How are market conditions right now?"

Returns:
- Current market status (open/closed/premarket)
- Performance across asset classes (1d, 1m, 3m, 1y)
- Volatility analysis
- Correlation dynamics
- Risk-off/risk-on signals

---

## 🏦 Fed & Interest Rate Questions

**Example**: "How will Fed rate cuts impact my portfolio?"

Returns **3 scenarios**:

**Scenario 1: Aggressive Cuts (3-4)**
- Winners: Growth stocks, utilities, REITs
- Losers: Banks, high-yield bonds
- **Portfolio action**: Specific rebalancing

**Scenario 2: Mild Cuts (1-2)**
- Winners: Quality growth, dividend stocks
- Losers: Speculative names

**Scenario 3: No Cuts**
- Winners: Financials, energy, TIPS
- Losers: Growth, long-duration bonds

Plus: Historical lessons (2001, 2019)

---

**Example**: "What should I do if Fed hikes rates?"

Returns:
- Current 10-year yield and trend
- Interest rate environment context
- Sector impacts of tightening
- Portfolio positioning for rate rises
- Historical precedents

---

**Example**: "How does Fed policy affect my holdings?"

Returns:
- Current Fed stance interpretation
- Impact by sector (banks, REITs, utilities, growth)
- Duration considerations for bonds
- **Recommendation by scenario**

---

## 🔥 Inflation Questions

**Example**: "What are the inflation risks?"

Returns **current data**:
- TIPS +/- YTD return
- Gold +/- YTD return
- Commodities +/- YTD return

Returns **2 scenarios**:

**Scenario A: Sustained Inflation (3-4%)**
- Winners: Energy, utilities, staples, REITs, TIPS
- Losers: Tech growth, long-duration bonds
- **Allocation**: Add X% TIPS, reduce Y% growth

**Scenario B: Deflation (<1%)**
- Winners: Long-duration bonds, growth stocks, tech
- Losers: Energy, commodities, REITs
- **Allocation**: Shift to bonds, reduce commodities

Plus:
- Inflation-protected allocation template (percentages)
- Historical episodes (2021-22, 1980s Volcker)

---

**Example**: "How does inflation affect stocks vs bonds?"

Returns:
- Current inflation regime interpretation
- Impact on equities (by style: growth vs value)
- Impact on bonds (duration, real yields)
- Asset class comparison
- **Hedging strategies** with real asset exposure

---

**Example**: "Should I hedge against inflation?"

Returns:
- Current hedge performance
- Hedging allocation template
- Inflation vs deflation scenarios
- Portfolio composition recommendations
- Historical validation

---

## 📚 What's NOT Supported (Yet)

❌ Specific earnings predictions ("When will Apple report earnings?")
❌ Stock price targets ("What's MSFT's price target?")
❌ Day-trading signals ("Is it a buy signal?")
❌ Crypto/Alternative assets (separate module)
❌ Tax optimization ("Should I harvest losses?")
❌ Margin/leverage ("Should I margin up?")

---

## 💡 Pro Tips for Best Results

### Give Context
✅ "Should I add MSFT given my current holdings?" (with portfolio loaded)
❌ "Is MSFT good?" (generic)

### Mention Your Concern
✅ "How will Fed rate cuts impact my portfolio?" 
❌ "Tell me about rates"

### Be Specific About Scenario
✅ "What are the inflation risks to my tech heavy portfolio?"
❌ "What about inflation?"

### Include Portfolio
✅ Load your portfolio in Portfolio Builder before asking
❌ Ask without portfolio (generic response)

---

## 📊 Data Freshness

All responses include **timestamp** showing analysis time.
- Data from Yfinance (updated intraday)
- Market status shows if market is open
- Times in UTC

Example: "**Analysis Timestamp**: 2026-04-10 14:32:15 UTC"

---

## 🎯 Example Session

**User**: "I have 8 stocks: AAPL, MSFT, GOOGL, NVDA, JPM, JNJ, XOM, WMT. 
Should I add META?"

**Bot**: 
```
Portfolio Fit Analysis: META

Correlation to portfolio: 0.72 (high)
├─ Already have 4 tech in 8 holdings (50% weight)
├─ Adding META adds tech concentration risk
├─ Quality score: 78/100 (good but not exceptional)
└─ Suggestion: YES, but size at 3% (keep weighted average ~12.5%)
    Consider selling one existing tech name first

Alternative: Consider sector diversification
├─ Consumer sector: Only WMT (weak)
├─ Healthcare: Only JNJ (adequate)
├─ If sector rotation to defensive, reduce META
```

---

**User**: "Given the Fed might cut rates, what should I do?"

**Bot**:
```
[Shows 3 scenarios with your portfolio impact]

Your portfolio tech-heavy → High benefit from rate cuts
Recommendation:
├─ Scenario 1 (Aggressive cuts): Position increases ~15%
├─ Scenario 2 (Mild cuts): Position increases ~5-8%
├─ Scenario 3 (No cuts): Position decreases ~5%

Hedge strategy:
├─ Add 10% bonds (TLT) if worried about Scenario 3
├─ Reduce META concentration—highest risk if no cuts
└─ History lesson: 2019 showed cuts can trigger 30% rallies
```

---

**User**: "What about inflation?"

**Bot**:
```
Current hedges: TIPS +2.3%, Gold +18.7%, Commodities -8.2%

Your portfolio risk:
├─ Tech-heavy: Vulnerable in sustained inflation (multiple compression)
├─ Energy XOM: Partial hedge (gains in inflation)
├─ Consumer WMT: Good hedge (pricing power, staples)

If sustained 3-4% inflation:
├─ Add 8% TIPS allocation
├─ Reduce META/NVDA combined from 30% to 20%
├─ Increase XOM and WMT weights

If deflation:
├─ META/NVDA win again (growth narrative)
├─ Keep existing tech allocation
└─ Add long-duration bonds (TLT)
```

---

## 🚀 Ready to Try?

Go to **💬 Chat & Questions** page and start asking!
The bot will now provide data-driven, specific recommendations based on real market conditions.

**Sample questions to start:**
1. "Should I add [TICKER] to my portfolio?"
2. "How is the [SECTOR] sector performing?"
3. "How will Fed rate cuts impact my portfolio?"
4. "What are the inflation risks?"
5. "What's the market outlook?"
