"""
Advanced Investment Chatbot with Market Intelligence and Anti-Hallucination Features

Key Features:
- Real-time data fetching with validation
- Confidence indicators for all claims
- Explicit fact vs. analysis separation
- Graceful handling of uncertainty
- Source attribution
- Timestamp tracking for data freshness
- Disclaimers for investment advice

Capabilities:
- Stock-specific analysis (fundamentals, technicals, portfolio fit)
- Industry/sector analysis and trends
- Market regime analysis and outlook
- Macro economic commentary
- Portfolio context and recommendations
- General investment education
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')


# Investment disclaimer
INVESTMENT_DISCLAIMER = """
⚠️ **Disclaimer**: This analysis is for educational purposes only and does not constitute investment advice. 
Past performance does not guarantee future results. Please consult a licensed financial advisor before making investment decisions.
"""


def get_data_timestamp() -> str:
    """Return current timestamp for data freshness tracking"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")


def get_market_status() -> Dict[str, Any]:
    """
    Determine if US stock market is currently open, and return status
    
    Returns:
        - status: 'open', 'closed', 'pre_market', 'after_hours'
        - emoji: Visual indicator
        - message: Human-readable status
    """
    from datetime import datetime
    import pytz
    
    # Get current time in US Eastern Time
    eastern = pytz.timezone('US/Eastern')
    now_et = datetime.now(eastern)
    
    # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    pre_market_start = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
    after_hours_end = now_et.replace(hour=20, minute=0, second=0, microsecond=0)
    
    is_weekday = now_et.weekday() < 5  # 0-4 is Monday-Friday
    
    if not is_weekday:
        return {
            'status': 'closed',
            'emoji': '🔴',
            'message': 'Market Closed (Weekend)',
            'time': now_et.strftime('%I:%M %p ET')
        }
    
    if market_open <= now_et < market_close:
        time_to_close = market_close - now_et
        hours = time_to_close.seconds // 3600
        minutes = (time_to_close.seconds % 3600) // 60
        return {
            'status': 'open',
            'emoji': '🟢',
            'message': f'Market Open (closes in {hours}h {minutes}m)',
            'time': now_et.strftime('%I:%M %p ET')
        }
    elif pre_market_start <= now_et < market_open:
        return {
            'status': 'pre_market',
            'emoji': '🟡',
            'message': 'Pre-Market Trading',
            'time': now_et.strftime('%I:%M %p ET')
        }
    elif market_close <= now_et < after_hours_end:
        return {
            'status': 'after_hours',
            'emoji': '🟡',
            'message': 'After-Hours Trading',
            'time': now_et.strftime('%I:%M %p ET')
        }
    else:
        return {
            'status': 'closed',
            'emoji': '🔴',
            'message': 'Market Closed',
            'time': now_et.strftime('%I:%M %p ET')
        }


def get_live_price_data(ticker: str) -> Dict[str, Any]:
    """
    Fetch real-time price data for a ticker
    
    Returns live/delayed price, change, volume, and other real-time metrics
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        
        # Get most recent data - use 1d with 1m interval for real-time
        hist = ticker_obj.history(period='1d', interval='1m')
        info = ticker_obj.info
        
        if hist.empty:
            # Fallback to daily data
            hist = ticker_obj.history(period='5d')
        
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            open_price = hist['Open'].iloc[0]
            high_price = hist['High'].max()
            low_price = hist['Low'].min()
            volume = hist['Volume'].sum()
            
            # Calculate changes
            price_change = current_price - open_price
            price_change_pct = (price_change / open_price) * 100 if open_price > 0 else 0
            
            # Get previous close from info or calculate
            prev_close = info.get('previousClose', open_price)
            if prev_close and prev_close > 0:
                change_from_prev = current_price - prev_close
                change_from_prev_pct = (change_from_prev / prev_close) * 100
            else:
                change_from_prev = price_change
                change_from_prev_pct = price_change_pct
            
            return {
                'success': True,
                'current_price': current_price,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'volume': volume,
                'change': change_from_prev,
                'change_pct': change_from_prev_pct,
                'prev_close': prev_close,
                'timestamp': hist.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                'data_points': len(hist)
            }
        else:
            return {'success': False, 'error': 'No price data available'}
    except Exception as e:
        return {'success': False, 'error': str(e)[:100]}


def get_intraday_summary(ticker: str) -> str:
    """
    Get intraday trading summary with real-time metrics
    """
    price_data = get_live_price_data(ticker)
    
    if not price_data['success']:
        return f"⚠️ Real-time data unavailable: {price_data.get('error', 'Unknown error')}"
    
    current = price_data['current_price']
    change = price_data['change']
    change_pct = price_data['change_pct']
    
    # Format change with color indicator
    if change_pct > 0:
        change_emoji = '📈'
        change_str = f"+${change:.2f} (+{change_pct:.2f}%)"
    elif change_pct < 0:
        change_emoji = '📉'
        change_str = f"-${abs(change):.2f} ({change_pct:.2f}%)"
    else:
        change_emoji = '➡️'
        change_str = "Unchanged"
    
    summary = f"**Current Price**: ${current:.2f} {change_emoji} {change_str}\n"
    summary += f"**Day Range**: ${price_data['low']:.2f} - ${price_data['high']:.2f}\n"
    summary += f"**Volume**: {price_data['volume']:,.0f}\n"
    summary += f"**Last Updated**: {price_data['timestamp']}\n"
    
    return summary


def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """
    Validate if ticker exists and has data
    
    Returns: (is_valid, message)
    """
    try:
        data = yf.Ticker(ticker)
        info = data.info
        if info and 'symbol' in info:
            return True, f"{ticker} validated"
        return False, f"{ticker} data incomplete"
    except Exception as e:
        return False, f"Cannot validate {ticker}: {str(e)[:50]}"


def classify_question_type(question: str) -> Dict[str, Any]:
    """
    Classify user question into categories for intelligent routing
    
    Returns dict with:
        - type: 'stock', 'sector', 'market', 'macro', 'portfolio', 'general', 'education'
        - entities: extracted tickers, sector names, etc.
        - intent: 'analysis', 'comparison', 'forecast', 'risk', etc.
        - confidence: routing confidence (0-1)
    """
    question_lower = question.lower()
    
    classification = {
        'type': 'general',
        'entities': [],
        'intent': 'analysis',
        'keywords': [],
        'confidence': 0.5
    }
    
    # Comparative/ranking questions (highest priority - very specific pattern)
    comparative_patterns = ['which stock', 'what stock', 'best stock', 'worst stock', 
                           'top performer', 'best performer', 'worst performer',
                           'most gain', 'biggest gain', 'highest return', 'lowest return',
                           'top gainer', 'bottom performer', 'which is better',
                           'compare stocks', 'stock comparison', 'rank stocks',
                           'performing the best', 'performing the worst']
    if any(pattern in question_lower for pattern in comparative_patterns):
        classification['type'] = 'comparison'
        classification['confidence'] = 0.95
        # Check if timeframe specified
        if any(word in question_lower for word in ['today', 'this week', 'this month', 'this year']):
            classification['keywords'].append('timeframe_specified')
    
    # Educational/definition questions
    education_keywords = ['what is', 'what are', 'explain', 'how does', 'how do', 'define',
                         'definition of', 'meaning of', 'tell me about', 'teach me']
    if any(k in question_lower for k in education_keywords) and classification['type'] == 'general':
        classification['type'] = 'education'
        classification['confidence'] = 0.9
    
    # Macro keywords (highest priority for investing - specific topics)
    macro_keywords = ['fed', 'federal reserve', 'interest rate', 'inflation', 'cpi',
                     'unemployment', 'gdp', 'recession', 'economy', 'economic',
                     'monetary policy', 'fiscal policy', 'treasury', 'bond yield', 'yield curve',
                     'geopolitic', 'trade war', 'tariff', 'dollar strength', 'currency']
    if any(k in question_lower for k in macro_keywords) and classification['type'] == 'general':
        classification['type'] = 'macro'
        classification['keywords'] = [k for k in macro_keywords if k in question_lower]
        classification['confidence'] = 0.85
    
    # Market-level keywords (high priority - before sector)
    market_keywords = ['overall market', 'market outlook', 'market condition', 'sp500', 's&p 500', 
                       'dow jones', 'nasdaq', 'market index', 'broad market', 'equities overall',
                       'stock market', 'market trend', 'market is', 'are stocks']
    if any(k in question_lower for k in market_keywords) and classification['type'] not in ['macro', 'education']:
        classification['type'] = 'market'
        classification['confidence'] = 0.85
    
    # Sector/Industry keywords (medium priority)
    sectors = ['technology sector', 'tech sector', 'finance', 'financial sector', 'healthcare sector',
               'health sector', 'energy sector', 'utilities sector', 'real estate sector', 
               'consumer sector', 'industrial sector', 'materials sector', 'communication sector',
               'semiconductor industry', 'software industry', 'banking sector', 'insurance sector',
               'pharma', 'biotech', 'retail sector', 'automotive sector']
    
    detected_sectors = [s for s in sectors if s in question_lower]
    if detected_sectors and classification['type'] == 'general':
        classification['type'] = 'sector'
        classification['entities'].extend(detected_sectors)
        classification['confidence'] = 0.8
    
    # Portfolio-specific (clear signals)
    portfolio_keywords = ['my portfolio', 'current holdings', 'rebalance', 'my positions',
                         'my investments', 'my stocks', 'portfolio performance']
    if any(k in question_lower for k in portfolio_keywords) and classification['type'] == 'general':
        classification['type'] = 'portfolio'
        classification['confidence'] = 0.95
    
    # Extract potential tickers (lowest priority for classification)
    tickers = extract_ticker_from_question(question)
    if tickers:
        classification['entities'].append(tickers)
        # Only classify as stock if no other strong signals
        if classification['type'] == 'general':
            classification['type'] = 'stock'
            classification['confidence'] = 0.75
    
    # Intent classification
    if any(word in question_lower for word in ['compare', 'versus', 'vs', 'better than', 'or which']):
        classification['intent'] = 'comparison'
    elif any(word in question_lower for word in ['forecast', 'predict', 'future', 'outlook', 'expect', 'will be']):
        classification['intent'] = 'forecast'
    elif any(word in question_lower for word in ['risk', 'danger', 'concern', 'worry', 'safe', 'dangerous', 'volatile']):
        classification['intent'] = 'risk'
    elif any(word in question_lower for word in ['opportunity', 'undervalue', 'cheap', 'bargain', 'buy signal']):
        classification['intent'] = 'opportunity'
    elif any(word in question_lower for word in ['why', 'how come', 'reason', 'cause', 'explain why']):
        classification['intent'] = 'explanation'
    
    return classification


def analyze_stock_comparison(question: str, portfolio_stocks: Optional[List[str]] = None) -> str:
    """
    Compare stocks or identify top/bottom performers with REAL-TIME data
    Handles questions like: "Which stock is performing best today?"
    """
    market_status = get_market_status()
    
    response = f"## 📊 Stock Performance Comparison\n\n"
    response += f"{market_status['emoji']} **Market Status**: {market_status['message']} ({market_status['time']})\n"
    response += f"**Data Timestamp**: {get_data_timestamp()}\n\n"
    
    # Determine timeframe from question with real-time bias
    question_lower = question.lower()
    if 'today' in question_lower or 'day' in question_lower or 'now' in question_lower or 'current' in question_lower:
        period = '1d'
        timeframe = "Today (Real-Time)"
        use_intraday = True
    elif 'week' in question_lower:
        period = '5d'
        timeframe = "This Week"
        use_intraday = False
    elif 'month' in question_lower:
        period = '1mo'
        timeframe = "This Month"
        use_intraday = False
    elif 'year' in question_lower or 'ytd' in question_lower:
        period = 'ytd'
        timeframe = "Year-to-Date"
        use_intraday = False
    else:
        # Default to real-time if market is open, otherwise this week
        if market_status['status'] == 'open':
            period = '1d'
            timeframe = "Today (Real-Time)"
            use_intraday = True
        else:
            period = '5d'
            timeframe = "This Week"
            use_intraday = False
    
    # Determine stock universe
    if portfolio_stocks and len(portfolio_stocks) > 0:
        tickers = portfolio_stocks
        response += f"**Analyzing**: Your portfolio ({len(tickers)} stocks)\n"
    else:
        # Use popular stocks if no portfolio
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B', 
                   'JPM', 'V', 'JNJ', 'WMT', 'PG', 'XOM', 'DIS']
        response += f"**Analyzing**: Popular large-cap stocks ({len(tickers)} tickers)\n"
        response += f"_Note: Build a portfolio to compare your specific holdings_\n"
    
    response += f"**Timeframe**: {timeframe}\n\n"
    
    try:
        # Fetch REAL-TIME data for all tickers
        performance = {}
        live_prices = {}
        failed_tickers = []
        
        for ticker in tickers:
            try:
                if use_intraday:
                    # Use live price data for intraday
                    price_data = get_live_price_data(ticker)
                    if price_data['success']:
                        performance[ticker] = price_data['change_pct'] / 100
                        live_prices[ticker] = {
                            'price': price_data['current_price'],
                            'change': price_data['change'],
                            'change_pct': price_data['change_pct']
                        }
                else:
                    # Use historical data for longer periods
                    data = yf.download(ticker, period=period, progress=False, auto_adjust=False)
                    if not data.empty:
                        if isinstance(data.columns, pd.MultiIndex):
                            close_prices = data['Close', ticker]
                        else:
                            close_prices = data['Close']
                        
                        if len(close_prices) >= 2:
                            ret = (close_prices.iloc[-1] / close_prices.iloc[0] - 1)
                            performance[ticker] = ret
                            live_prices[ticker] = {'price': close_prices.iloc[-1]}
            except:
                failed_tickers.append(ticker)
        
        if len(performance) < 2:
            response += "⚠️ **Insufficient Data**: Could not fetch performance data for comparison.\n"
            response += "This may be due to market being closed or data provider issues.\n"
            return response
        
        # Sort by performance
        sorted_perf = sorted(performance.items(), key=lambda x: x[1], reverse=True)
        
        # Top performers with real-time prices
        response += f"### 🏆 Top Performers ({timeframe})\n\n"
        top_n = min(5, len(sorted_perf))
        for i, (ticker, ret) in enumerate(sorted_perf[:top_n], 1):
            emoji = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else '  '
            price_info = ""
            if ticker in live_prices and 'price' in live_prices[ticker]:
                price_info = f" | ${live_prices[ticker]['price']:.2f}"
            response += f"{emoji} **{ticker}**: {ret*100:+.2f}%{price_info}\n"
        
        # Bottom performers with real-time prices
        response += f"\n### 📉 Bottom Performers ({timeframe})\n\n"
        bottom_n = min(5, len(sorted_perf))
        for i, (ticker, ret) in enumerate(sorted_perf[-bottom_n:][::-1], 1):
            price_info = ""
            if ticker in live_prices and 'price' in live_prices[ticker]:
                price_info = f" | ${live_prices[ticker]['price']:.2f}"
            response += f"  **{ticker}**: {ret*100:+.2f}%{price_info}\n"
        
        # Summary statistics
        returns = list(performance.values())
        avg_return = np.mean(returns)
        median_return = np.median(returns)
        
        response += f"\n### 📈 Summary Statistics\n\n"
        response += f"• **Average Return**: {avg_return*100:+.2f}%\n"
        response += f"• **Median Return**: {median_return*100:+.2f}%\n"
        response += f"• **Best**: {sorted_perf[0][0]} ({sorted_perf[0][1]*100:+.2f}%)\n"
        response += f"• **Worst**: {sorted_perf[-1][0]} ({sorted_perf[-1][1]*100:+.2f}%)\n"
        response += f"• **Spread**: {(sorted_perf[0][1] - sorted_perf[-1][1])*100:.2f}pp\n"
        
        if failed_tickers:
            response += f"\n_Note: Could not fetch data for: {', '.join(failed_tickers)}_\n"
        
    except Exception as e:
        response += f"⚠️ **Error**: Could not complete comparison: {str(e)[:100]}\n"
        response += "\nPlease try again or be more specific with your question.\n"
    
    return response


def analyze_market_conditions() -> str:
    """
    Comprehensive REAL-TIME market regime analysis with live data
    """
    market_status = get_market_status()
    
    response = f"## 📊 Real-Time Market Analysis\n\n"
    response += f"{market_status['emoji']} **Market Status**: {market_status['message']} ({market_status['time']})\n"
    response += f"**Data Timestamp**: {get_data_timestamp()}\n\n"
    
    try:
        # Get REAL-TIME SPY data
        spy_live = get_live_price_data('SPY')
        
        if spy_live['success']:
            response += "### 📈 S&P 500 Live Status (SPY)\n\n"
            response += get_intraday_summary('SPY')
            response += "\n"
        
        # Fetch SPY historical for context (1 year)
        spy = yf.download('SPY', period='1y', progress=False, auto_adjust=False)
        
        if spy.empty:
            return response + "⚠️ **Data Unavailable**: Cannot fetch market data at this time. Please try again later.\n"
        
        if isinstance(spy.columns, pd.MultiIndex):
            spy_close = spy['Adj Close', 'SPY']
        else:
            spy_close = spy['Adj Close']
        
        spy_returns = spy_close.pct_change().dropna()
        
        if len(spy_returns) < 50:
            return response + "⚠️ **Insufficient Data**: Not enough historical data for analysis.\n"
        
        # Key metrics (FACTS - from real data)
        ytd_return = (spy_close.iloc[-1] / spy_close.iloc[0] - 1)
        vol = spy_returns.std() * np.sqrt(252)
        sharpe = (spy_returns.mean() * 252 - 0.03) / vol if vol > 0 else 0
        
        # Moving averages
        ma_50 = spy_close.rolling(50).mean().iloc[-1]
        ma_200 = spy_close.rolling(200).mean().iloc[-1]
        current_price = spy_close.iloc[-1]
        
        # Drawdown calculation
        running_max = spy_close.expanding().max()
        drawdown = (spy_close - running_max) / running_max
        max_dd = drawdown.min()
        current_dd = drawdown.iloc[-1]
        
        # === FACTS SECTION ===
        response += "### Market Metrics (Factual Data)\n\n"
        response += f"**S&P 500 Performance**:\n"
        response += f"• Current Price (SPY): ${current_price:.2f}\n"
        response += f"• YTD Return: {ytd_return*100:.1f}%\n"
        response += f"• Annualized Volatility: {vol*100:.1f}%\n"
        response += f"• Sharpe Ratio (est.): {sharpe:.2f}\n"
        response += f"• Current Drawdown: {current_dd*100:.1f}%\n"
        response += f"• Max Drawdown (1Y): {max_dd*100:.1f}%\n"
        response += f"• 50-day MA: ${ma_50:.2f}\n"
        response += f"• 200-day MA: ${ma_200:.2f}\n\n"
        
        # === ANALYSIS SECTION (interpretation) ===
        response += "### Technical Analysis (Interpretation)\n\n"
        
        # Regime classification (based on objective rules)
        if current_price > ma_50 > ma_200:
            regime = "📈 **Uptrend Confirmed**"
            regime_desc = "Price above both 50-day and 200-day moving averages. Technically bullish."
        elif current_price > ma_50 and current_price > ma_200:
            regime = "📈 **Mixed Uptrend**"
            regime_desc = "Price above key MAs but 50/200 relationship weakening."
        elif current_price < ma_50 < ma_200:
            regime = "📉 **Downtrend**"
            regime_desc = "Price below both moving averages. Technically bearish."
        elif current_price < ma_50 or current_price < ma_200:
            regime = "⚠️ **Transitional**"
            regime_desc = "Mixed signals - price crossing key moving averages."
        else:
            regime = "🔄 **Range-Bound**"
            regime_desc = "Sideways movement within defined range."
        
        response += f"**Market Regime**: {regime}\n"
        response += f"{regime_desc}\n\n"
        
        # Volatility context (factual comparison)
        vol_20d = spy_returns.tail(20).std() * np.sqrt(252)
        vol_60d = spy_returns.tail(60).std() * np.sqrt(252)
        
        response += f"**Volatility Context**:\n"
        response += f"• 20-day volatility: {vol_20d*100:.1f}% (recent)\n"
        response += f"• 60-day volatility: {vol_60d*100:.1f}% (medium-term)\n"
        
        if vol_20d > vol_60d * 1.3:
            response += "• Recent volatility spike detected (20d > 60d by 30%+)\n"
        elif vol_20d < vol_60d * 0.7:
            response += "• Volatility compression observed (20d < 60d by 30%+)\n"
        else:
            response += "• Volatility relatively stable\n"
        
        response += "\n"
        
        # Sector rotation analysis with real data
        try:
            sector_etfs = {
                'XLK': 'Technology',
                'XLF': 'Financials',
                'XLV': 'Healthcare',
                'XLE': 'Energy',
                'XLI': 'Industrials',
                'XLP': 'Staples',
                'XLU': 'Utilities',
                'XLY': 'Consumer Discr.',
                'XLRE': 'Real Estate'
            }
            
            sector_perf = {}
            for etf, name in sector_etfs.items():
                try:
                    data = yf.download(etf, period='3mo', progress=False, auto_adjust=False)
                    if not data.empty:
                        if isinstance(data.columns, pd.MultiIndex):
                            ret = (data['Adj Close', etf].iloc[-1] / data['Adj Close', etf].iloc[0] - 1)
                        else:
                            ret = (data['Adj Close'].iloc[-1] / data['Adj Close'].iloc[0] - 1)
                        sector_perf[name] = ret
                except:
                    pass
            
            if len(sector_perf) >= 5:  # Only show if we have enough data
                sorted_sectors = sorted(sector_perf.items(), key=lambda x: x[1], reverse=True)
                response += f"**Sector Performance (3-Month) - {len(sector_perf)} sectors tracked**:\n"
                response += f"• Top performers: {sorted_sectors[0][0]} ({sorted_sectors[0][1]*100:+.1f}%), {sorted_sectors[1][0]} ({sorted_sectors[1][1]*100:+.1f}%)\n"
                response += f"• Bottom performers: {sorted_sectors[-1][0]} ({sorted_sectors[-1][1]*100:+.1f}%), {sorted_sectors[-2][0]} ({sorted_sectors[-2][1]*100:+.1f}%)\n\n"
                
                # Sector rotation insight (objective analysis)
                cyclical = ['Energy', 'Financials', 'Industrials', 'Consumer Discr.']
                defensive = ['Utilities', 'Staples', 'Healthcare']
                
                cyclical_returns = [v for k, v in sector_perf.items() if k in cyclical]
                defensive_returns = [v for k, v in sector_perf.items() if k in defensive]
                
                if cyclical_returns and defensive_returns:
                    cyclical_avg = np.mean(cyclical_returns)
                    defensive_avg = np.mean(defensive_returns)
                    
                    response += f"**Sector Rotation Analysis**:\n"
                    response += f"• Cyclical avg: {cyclical_avg*100:+.1f}% | Defensive avg: {defensive_avg*100:+.1f}%\n"
                    
                    if cyclical_avg > defensive_avg + 0.05:
                        response += f"• **Observation**: Cyclicals outperforming defensives by {(cyclical_avg-defensive_avg)*100:.1f}pp\n"
                        response += f"• _Interpretation_: May indicate risk-on sentiment or growth expectations\n"
                    elif defensive_avg > cyclical_avg + 0.05:
                        response += f"• **Observation**: Defensives outperforming cyclicals by {(defensive_avg-cyclical_avg)*100:.1f}pp\n"
                        response += f"• _Interpretation_: May indicate risk-off sentiment or caution\n"
                    else:
                        response += f"• **Observation**: Balanced performance between cyclicals and defensives\n"
            
        except Exception as e:
            response += f"⚠️ _Sector rotation data partially unavailable_\n"
        
    except Exception as e:
        response += f"⚠️ **Error**: Could not complete market analysis: {str(e)[:100]}\n\n"
        response += "This may be due to data provider issues or network connectivity. Please try again.\n"
    
    return response


def analyze_sector_industry(sector_name: str, portfolio_weights: Optional[pd.Series] = None) -> str:
    """
    REAL-TIME sector analysis with live ETF data and current market conditions
    """
    market_status = get_market_status()
    
    # Map sector names to ETFs
    sector_map = {
        'tech': ('XLK', 'Technology'),
        'technology': ('XLK', 'Technology'),
        'software': ('XLK', 'Technology'),
        'semiconductor': ('XLK', 'Technology/Semiconductors'),
        'finance': ('XLF', 'Financials'),
        'financial': ('XLF', 'Financials'),
        'banking': ('XLF', 'Financials/Banking'),
        'healthcare': ('XLV', 'Healthcare'),
        'health': ('XLV', 'Healthcare'),
        'pharma': ('XLV', 'Healthcare/Pharmaceuticals'),
        'biotech': ('XLV', 'Healthcare/Biotechnology'),
        'energy': ('XLE', 'Energy'),
        'oil': ('XLE', 'Energy'),
        'consumer': ('XLY', 'Consumer Discretionary'),
        'retail': ('XLY', 'Consumer Discretionary/Retail'),
        'industrial': ('XLI', 'Industrials'),
        'staples': ('XLP', 'Consumer Staples'),
        'utilities': ('XLU', 'Utilities'),
        'real estate': ('XLRE', 'Real Estate'),
        'materials': ('XLB', 'Materials'),
        'communication': ('XLC', 'Communication Services')
    }
    
    etf, full_name = sector_map.get(sector_name.lower(), (None, sector_name.title()))
    
    response = f"## 🏭 Real-Time Sector Analysis: {full_name}\n\n"
    response += f"{market_status['emoji']} **Market Status**: {market_status['message']} ({market_status['time']})\n"
    response += f"**Data Timestamp**: {get_data_timestamp()}\n\n"
    
    if not etf:
        response += f"⚠️ **Cannot Map Sector**: '{sector_name}' is not recognized.\n\n"
        response += "**Supported sectors**: technology, finance, healthcare, energy, consumer, industrial, materials, utilities, real estate, communication\n"
        return response
    
    try:
        # Get REAL-TIME sector ETF price
        sector_live = get_live_price_data(etf)
        
        if sector_live['success']:
            response += f"### 💹 Live {etf} Data\n\n"
            response += get_intraday_summary(etf)
            response += "\n"
        
        # Fetch sector ETF historical data
        sector_data = yf.download(etf, period='1y', progress=False, auto_adjust=False)
        spy_data = yf.download('SPY', period='1y', progress=False, auto_adjust=False)
        
        if sector_data.empty or spy_data.empty:
            return response + f"⚠️ **Data Unavailable**: Cannot fetch data for {etf} or market benchmark.\n"
        
        if isinstance(sector_data.columns, pd.MultiIndex):
            sector_close = sector_data['Adj Close', etf]
            spy_close = spy_data['Adj Close', 'SPY']
        else:
            sector_close = sector_data['Adj Close']
            spy_close = spy_data['Adj Close']
        
        # === FACTUAL METRICS ===
        sector_ytd = (sector_close.iloc[-1] / sector_close.iloc[0] - 1)
        spy_ytd = (spy_close.iloc[-1] / spy_close.iloc[0] - 1)
        relative_perf = sector_ytd - spy_ytd
        
        sector_vol = sector_close.pct_change().std() * np.sqrt(252)
        sector_current = sector_close.iloc[-1]
        
        response += "### Performance Metrics (Factual Data)\n\n"
        response += f"**{etf} - {full_name}**:\n"
        response += f"• Current Price: ${sector_current:.2f}\n"
        response += f"• YTD Return: {sector_ytd*100:.1f}%\n"
        response += f"• Market (SPY) YTD: {spy_ytd*100:.1f}%\n"
        response += f"• Relative Performance: {relative_perf*100:+.1f}pp\n"
        response += f"• Volatility (annualized): {sector_vol*100:.1f}%\n\n"
        
        # Momentum (factual)
        if len(sector_close) >= 63:
            returns_1m = (sector_close.iloc[-1] / sector_close.iloc[-21] - 1) if len(sector_close) >= 21 else None
            returns_3m = (sector_close.iloc[-1] / sector_close.iloc[-63] - 1) if len(sector_close) >= 63 else None
            
            if returns_1m is not None and returns_3m is not None:
                response += f"**Momentum**:\n"
                response += f"• 1-Month: {returns_1m*100:+.1f}%\n"
                response += f"• 3-Month: {returns_3m*100:+.1f}%\n\n"
        
        # Technical position (objective)
        if len(sector_close) >= 200:
            ma_50 = sector_close.rolling(50).mean().iloc[-1]
            ma_200 = sector_close.rolling(200).mean().iloc[-1]
            current = sector_close.iloc[-1]
            
            response += f"### Technical Analysis\n\n"
            response += f"**Moving Averages**:\n"
            response += f"• 50-day MA: ${ma_50:.2f}\n"
            response += f"• 200-day MA: ${ma_200:.2f}\n"
            response += f"• Price vs 50d MA: {((current/ma_50 - 1)*100):+.1f}%\n"
            response += f"• Price vs 200d MA: {((current/ma_200 - 1)*100):+.1f}%\n\n"
            
            if current > ma_50 > ma_200:
                response += "• **Technical Status**: Uptrend (price > 50d > 200d MA)\n"
            elif current < ma_50 < ma_200:
                response += "• **Technical Status**: Downtrend (price < 50d < 200d MA)\n"
            else:
                response += "• **Technical Status**: Mixed signals\n"
        
        response += "\n"
        
    except Exception as e:
        response += f"⚠️ **Error**: Could not complete sector analysis: {str(e)[:100]}\n"
    
    return response


def analyze_macro_environment(keywords: List[str]) -> str:
    """
    Macro analysis with explicit separation of facts vs. interpretation
    """
    response = f"## 🌍 Macro Environment Analysis\n\n"
    response += f"**Analysis date**: {get_data_timestamp()}\n\n"
    
    response += "_Note: Macro analysis combines factual data (e.g., yields, indicators) with economic interpretation. " 
    response += "Interpretations reflect general market consensus but should not be taken as predictions._\n\n"
    
    # Detect specific topics
    topics = {
        'rates': any(k in ['interest rate', 'fed', 'federal reserve', 'monetary'] for k in keywords),
        'inflation': any(k in ['inflation', 'cpi'] for k in keywords),
        'growth': any(k in ['gdp', 'economy', 'recession', 'growth'] for k in keywords),
        'geopolitics': any(k in ['geopolitic', 'trade war', 'tariff'] for k in keywords)
    }
    
    try:
        # Fetch Treasury yields as factual data point
        tnx = yf.download('^TNX', period='6mo', progress=False, auto_adjust=False)
        
        if not tnx.empty:
            if isinstance(tnx.columns, pd.MultiIndex):
                tnx_close = tnx['Adj Close', '^TNX']
            else:
                tnx_close = tnx['Adj Close']
            
            current_10y = tnx_close.iloc[-1]
            prev_10y = tnx_close.iloc[0]
            
            response += "### Interest Rate Environment (Factual)\n\n"
            response += f"**10-Year Treasury Yield**:\n"
            response += f"• Current: {current_10y:.2f}%\n"
            response += f"• 6-month change: {current_10y - prev_10y:+.2f}%\n"
            response += f"• Trend: {'Rising' if current_10y > prev_10y else 'Falling' if current_10y < prev_10y else 'Stable'}\n\n"
            
            # Contextual interpretation (clearly labeled)
            response += "### Interest Rate Context (Interpretation)\n\n"
            
            if current_10y > 4.5:
                response += f"• **Historical context**: At {current_10y:.1f}%, yields are elevated relative to 2010-2020 averages\n"
                response += "• _Common interpretation_: Tight financial conditions; higher discount rates for equity valuations\n"
            elif current_10y < 3.0:
                response += f"• **Historical context**: At {current_10y:.1f}%, yields are relatively low\n"
                response += "• _Common interpretation_: Accommodative conditions; lower discount rates may benefit growth equities\n"
            else:
                response += f"• **Historical context**: At {current_10y:.1f}%, yields are in moderate range\n"
                response += "• _Common interpretation_: Balanced rate environment\n"
            
            response += "\n"
        else:
            response += "⚠️ **Treasury yield data unavailable**\n\n"
    
    except Exception as e:
        response += f"⚠️ Unable to fetch real-time yield data: {str(e)[:80]}\n\n"
    
    # Educational content (not predictions)
    response += "### General Macro Considerations\n\n"
    
    if topics.get('inflation'):
        response += "**Inflation Dynamics**:\n"
        response += "• High inflation typically leads to Fed tightening (rate hikes)\n"
        response += "• Companies with pricing power tend to weather inflation better\n"
        response += "• Real assets (commodities, real estate) may provide inflation hedges\n"
        response += "• Fixed-income investors face purchasing power erosion\n\n"
    
    if topics.get('growth'):
        response += "**Economic Growth Indicators**:\n"
        response += "• GDP growth, PMI indices, employment data are key leading indicators\n"
        response += "• Yield curve inversions have historically preceded recessions\n"
        response += "• Consumer spending (70% of US GDP) drives economic cycles\n"
        response += "• Corporate earnings growth typically correlates with economic expansion\n\n"
    
    if topics.get('rates'):
        response += "**Fed Policy Impact**:\n"
        response += "• Rate cuts: Generally supportive for equities (lower borrowing costs)\n"
        response += "• Rate hikes: Can pressure valuations (higher discount rates)\n"
        response += "• Pause periods: Markets assess economic impact of prior moves\n"
        response += "• Forward guidance matters as much as current rates\n\n"
    
    response += "**Investment Style Considerations**:\n"
    response += "• High rates environment → Favor value, quality, current cash flows\n"
    response += "• Low rates environment → Growth stocks may outperform\n"
    response += "• Recession concerns → Defensive sectors (healthcare, utilities, staples)\n"
    response += "• Expansion phase → Cyclicals (industrials, discretionary, financials)\n\n"
    
    return response


def analyze_portfolio_clustering(portfolio_weights: pd.Series, returns_data: pd.DataFrame) -> str:
    """
    Advanced portfolio analysis with stock clustering by correlation patterns
    Groups stocks with similar behavior together
    """
    response = "## 💼 Portfolio Behavior Clustering\n\n"
    
    try:
        # Get returns for portfolio stocks
        portfolio_stocks = portfolio_weights.index.tolist()
        portfolio_returns = returns_data[portfolio_stocks].dropna()
        
        if len(portfolio_returns) < 20:
            return response + "⚠️ Insufficient data for clustering analysis.\n"
        
        # Calculate correlation matrix
        corr_matrix = portfolio_returns.corr()
        
        # Perform simple hierarchical clustering based on correlation
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        # Convert correlation to distance (1 - correlation)
        distances = 1 - corr_matrix.abs()
        condensed_dist = squareform(distances)
        
        # Hierarchical clustering
        linkage_matrix = linkage(condensed_dist, method='average')
        
        # Form clusters (adjust threshold for desired number of clusters)
        clusters = fcluster(linkage_matrix, t=0.7, criterion='distance')
        
        # Group stocks by cluster
        cluster_groups = {}
        for i, stock in enumerate(portfolio_stocks):
            cluster_id = clusters[i]
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(stock)
        
        # Analyze each cluster
        response += f"**Identified {len(cluster_groups)} behavioral groups** in your portfolio:\n\n"
        
        for cluster_id, stocks in sorted(cluster_groups.items(), key=lambda x: len(x[1]), reverse=True):
            response += f"### 📊 Group {cluster_id}: {', '.join(stocks)}\n\n"
            
            # Calculate cluster characteristics
            cluster_returns = portfolio_returns[stocks]
            avg_corr = corr_matrix.loc[stocks, stocks].values[np.triu_indices_from(corr_matrix.loc[stocks, stocks].values, k=1)].mean()
            
            cluster_vol = cluster_returns.std().mean() * np.sqrt(252)
            cluster_ret = cluster_returns.mean().mean() * 252
            
            response += f"**Characteristics**:\n"
            response += f"• **Stocks**: {len(stocks)} ({sum(portfolio_weights[stocks])/portfolio_weights.sum()*100:.1f}% of portfolio)\n"
            response += f"• **Internal Correlation**: {avg_corr:.2f} (moves together {avg_corr*100:.0f}% of the time)\n"
            response += f"• **Avg Volatility**: {cluster_vol*100:.1f}%\n"
            response += f"• **Avg Return**: {cluster_ret*100:+.1f}%\n\n"
            
            # Interpretation
            if avg_corr > 0.7:
                response += "_High correlation - these stocks react similarly to market events. Consider if this concentration aligns with your risk tolerance._\n\n"
            elif avg_corr > 0.4:
                response += "_Moderate correlation - some similar behavior but maintain individual characteristics._\n\n"
            else:
                response += "_Low correlation - these stocks provide diversification benefits within the group._\n\n"
        
        # Cross-cluster correlation
        response += "### 🔗 Cross-Group Relationships\n\n"
        
        if len(cluster_groups) > 1:
            cluster_ids = list(cluster_groups.keys())
            for i in range(len(cluster_ids)):
                for j in range(i+1, len(cluster_ids)):
                    group_i = cluster_groups[cluster_ids[i]]
                    group_j = cluster_groups[cluster_ids[j]]
                    
                    # Average correlation between groups
                    cross_corr = corr_matrix.loc[group_i, group_j].values.mean()
                    
                    response += f"• **Group {cluster_ids[i]} ↔ Group {cluster_ids[j]}**: {cross_corr:.2f}"
                    if cross_corr > 0.5:
                        response += " (Still somewhat linked)\n"
                    elif cross_corr < 0.2:
                        response += " ✓ (Strong diversification)\n"
                    else:
                        response += " (Moderate diversification)\n"
            
            response += "\n"
        
        # Portfolio insights
        response += "### 💡 Diversification Insights\n\n"
        
        # Check if portfolio is too concentrated in one cluster
        largest_cluster_weight = max(sum(portfolio_weights[stocks])/portfolio_weights.sum() for stocks in cluster_groups.values())
        
        if largest_cluster_weight > 0.6:
            response += "⚠️ **Concentration Alert**: Over 60% of your portfolio moves together. Consider:\n"
            response += "• Adding stocks from uncorrelated sectors\n"
            response += "• Reducing weights in the dominant cluster\n"
            response += "• This increases systemic risk during market shocks\n\n"
        elif largest_cluster_weight > 0.4:
            response += "🟡 **Moderate Concentration**: 40-60% in one behavioral group.\n"
            response += "• Acceptable for focused strategies\n"
            response += "• Monitor for sector-specific risks\n\n"
        else:
            response += "✓ **Well Diversified**: No single behavioral group dominates.\n"
            response += "• Portfolio shows good dispersion across patterns\n"
            response += "• Lower exposure to cluster-specific shocks\n\n"
        
        # Sector analysis if possible
        try:
            sector_map = {}
            for stock in portfolio_stocks:
                ticker_obj = yf.Ticker(stock)
                sector = ticker_obj.info.get('sector', 'Unknown')
                if sector not in sector_map:
                    sector_map[sector] = []
                sector_map[sector].append(stock)
            
            response += "### 🏭 Sector Distribution Across Clusters\n\n"
            for cluster_id, stocks in sorted(cluster_groups.items()):
                sectors = {}
                for stock in stocks:
                    for sector, sector_stocks in sector_map.items():
                        if stock in sector_stocks:
                            sectors[sector] = sectors.get(sector, 0) + 1
                
                if sectors:
                    dominant_sector = max(sectors, key=sectors.get)
                    response += f"• **Group {cluster_id}**: Primarily {dominant_sector} ({sectors[dominant_sector]}/{len(stocks)} stocks)\n"
            
            response += "\n"
        except:
            pass  # Sector data optional
        
    except ImportError:
        response += "⚠️ **Clustering requires scipy**. Install with: `pip install scipy`\n\n"
    except Exception as e:
        response += f"⚠️ **Analysis Error**: {str(e)[:100]}\n\n"
    
    return response


def analyze_investment_decision(user_question: str,
                               portfolio_weights: Optional[pd.Series] = None,
                               candidate_ticker: str = None,
                               returns_data: Optional[pd.DataFrame] = None) -> str:
    """
    REAL-TIME stock analysis with live price data and instant updates
    Works with ANY ticker - fetches fresh data if needed
    """
    market_status = get_market_status()
    
    response = f"## 📈 Real-Time Stock Analysis: {candidate_ticker}\n\n"
    response += f"{market_status['emoji']} **Market Status**: {market_status['message']} ({market_status['time']})\n"
    response += f"**Analysis Timestamp**: {get_data_timestamp()}\n\n"
    
    # Validate ticker first
    is_valid, validation_msg = validate_ticker(candidate_ticker)
    if not is_valid:
        response += f"⚠️ **Ticker Validation Failed**: {validation_msg}\n\n"
        response += "Please verify the ticker symbol and try again.\n"
        return response
    
    try:
        # === REAL-TIME PRICE DATA ===
        live_data = get_live_price_data(candidate_ticker)
        
        if live_data['success']:
            response += "### 💹 Live Market Data\n\n"
            response += get_intraday_summary(candidate_ticker)
            response += "\n"
        else:
            response += f"⚠️ _Live price data unavailable: {live_data.get('error', 'Unknown')}_\n\n"
        
        # Fetch comprehensive data
        ticker_obj = yf.Ticker(candidate_ticker)
        info = ticker_obj.info
        hist = ticker_obj.history(period='1y')
        
        if not info or len(info) < 5:
            return response + "⚠️ **Insufficient Data**: Cannot retrieve fundamental data for this ticker.\n"
        
        # === COMPANY OVERVIEW ===
        response += "### 🏢 Company Overview\n\n"
        company_name = info.get('longName') or info.get('shortName') or candidate_ticker
        response += f"**{company_name}** ({candidate_ticker})\n"
        response += f"• **Sector**: {info.get('sector', 'N/A')} | **Industry**: {info.get('industry', 'N/A')}\n"
        
        market_cap = info.get('marketCap')
        if market_cap:
            cap_size = 'Mega-cap' if market_cap > 200e9 else 'Large-cap' if market_cap > 10e9 else 'Mid-cap' if market_cap > 2e9 else 'Small-cap'
            response += f"• **Market Cap**: ${market_cap/1e9:.1f}B ({cap_size})\n"
        
        # Business description (abbreviated)
        if business_summary := info.get('longBusinessSummary'):
            response += f"• **Business**: {business_summary[:200]}...\n"
        
        response += "\n"
        
        # === VALUATION ANALYSIS ===
        response += "### 💰 Valuation Analysis\n\n"
        
        pe_ratio = info.get('forwardPE') or info.get('trailingPE')
        pb_ratio = info.get('priceToBook')
        ps_ratio = info.get('priceToSalesTrailing12Months')
        peg_ratio = info.get('pegRatio')
        ev_ebitda = info.get('enterpriseToEbitda')
        
        # Comprehensive valuation metrics
        response += "**Valuation Multiples**:\n"
        
        valuation_score = 0
        valuation_max = 0
        
        if pe_ratio and pe_ratio > 0:
            valuation_max += 1
            response += f"• **P/E Ratio**: {pe_ratio:.1f}"
            if pe_ratio < 15:
                response += " ✓ (Undervalued territory)\n"
                valuation_score += 1
            elif pe_ratio > 30:
                response += " ⚠️ (Premium valuation - needs strong growth)\n"
            else:
                response += " (Fair range)\n"
                valuation_score += 0.5
        
        if pb_ratio and pb_ratio > 0:
            valuation_max += 1
            response += f"• **P/B Ratio**: {pb_ratio:.2f}"
            if pb_ratio < 1.5:
                response += " ✓ (Trading near book value)\n"
                valuation_score += 1
            elif pb_ratio > 5:
                response += " (Asset-light or intangible-heavy)\n"
            else:
                response += "\n"
                valuation_score += 0.5
        
        if peg_ratio and peg_ratio > 0:
            response += f"• **PEG Ratio**: {peg_ratio:.2f}"
            if peg_ratio < 1:
                response += " ✓ (Growth at reasonable price)\n"
            elif peg_ratio > 2:
                response += " (Expensive relative to growth)\n"
            else:
                response += "\n"
        
        if ev_ebitda and ev_ebitda > 0:
            response += f"• **EV/EBITDA**: {ev_ebitda:.1f}"
            if ev_ebitda < 10:
                response += " ✓ (Attractive valuation)\n"
            elif ev_ebitda > 20:
                response += " (Rich valuation)\n"
            else:
                response += "\n"
        
        # Valuation verdict
        if valuation_max > 0:
            val_score_pct = (valuation_score / valuation_max) * 100
            response += f"\n**Valuation Score**: {val_score_pct:.0f}/100"
            if val_score_pct > 66:
                response += " - _Appears undervalued_\n"
            elif val_score_pct < 33:
                response += " - _Appears overvalued_\n"
            else:
                response += " - _Fair valuation_\n"
        
        response += "\n"
        
        # === FINANCIAL HEALTH & PROFITABILITY ===
        response += "### 💪 Financial Health & Profitability\n\n"
        
        roe = info.get('returnOnEquity')
        roa = info.get('returnOnAssets')
        profit_margin = info.get('profitMargins')
        operating_margin = info.get('operatingMargins')
        revenue_growth = info.get('revenueGrowth')
        earnings_growth = info.get('earningsGrowth')
        
        response += "**Profitability Metrics**:\n"
        
        if roe and roe > 0:
            response += f"• **ROE**: {roe*100:.1f}%"
            if roe > 0.20:
                response += " 🔥 (Excellent - top quartile)\n"
            elif roe > 0.15:
                response += " ✓ (Strong)\n"
            elif roe > 0.10:
                response += " (Average)\n"
            else:
                response += " ⚠️ (Weak - below 10%)\n"
        
        if profit_margin and profit_margin > 0:
            response += f"• **Net Margin**: {profit_margin*100:.1f}%"
            if profit_margin > 0.20:
                response += " (High margin business)\n"
            elif profit_margin > 0.10:
                response += " (Healthy margins)\n"
            elif profit_margin < 0.05:
                response += " (Thin margins - cost pressure)\n"
            else:
                response += "\n"
        
        if operating_margin:
            response += f"• **Operating Margin**: {operating_margin*100:.1f}%\n"
        
        # Growth analysis
        response += "\n**Growth Trajectory**:\n"
        
        if revenue_growth:
            response += f"• **Revenue Growth (YoY)**: {revenue_growth*100:+.1f}%"
            if revenue_growth > 0.20:
                response += " 🚀 (High growth)\n"
            elif revenue_growth > 0.10:
                response += " (Solid growth)\n"
            elif revenue_growth < 0:
                response += " ⚠️ (Contracting)\n"
            else:
                response += "\n"
        
        if earnings_growth:
            response += f"• **Earnings Growth (YoY)**: {earnings_growth*100:+.1f}%"
            if earnings_growth > 0.25:
                response += " (Accelerating profits)\n"
            elif earnings_growth < 0:
                response += " (Earnings declining)\n"
            else:
                response += "\n"
        
        # Balance sheet strength
        debt_to_equity = info.get('debtToEquity')
        current_ratio = info.get('currentRatio')
        
        response += "\n**Balance Sheet Strength**:\n"
        
        if debt_to_equity is not None:
            response += f"• **Debt/Equity**: {debt_to_equity:.1f}"
            if debt_to_equity < 50:
                response += " ✓ (Conservative leverage)\n"
            elif debt_to_equity > 150:
                response += " ⚠️ (High leverage - balance sheet risk)\n"
            else:
                response += " (Moderate leverage)\n"
        
        if current_ratio and current_ratio > 0:
            response += f"• **Current Ratio**: {current_ratio:.2f}"
            if current_ratio > 1.5:
                response += " ✓ (Strong liquidity)\n"
            elif current_ratio < 1.0:
                response += " ⚠️ (Liquidity concerns)\n"
            else:
                response += " (Adequate liquidity)\n"
        
        response += "\n"
        
        # === PRICE MOMENTUM & TECHNICAL ===
        if not hist.empty:
            response += "### 📊 Price Momentum & Technical Analysis\n\n"
            
            current_price = hist['Close'].iloc[-1]
            
            # Calculate returns
            ret_1m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-21] - 1) if len(hist) >= 21 else None
            ret_3m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-63] - 1) if len(hist) >= 63 else None
            ret_6m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-126] - 1) if len(hist) >= 126 else None
            ret_1y = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) if len(hist) >= 200 else None
            
            response += "**Recent Performance**:\n"
            if ret_1m: response += f"• **1-Month**: {ret_1m*100:+.1f}%\n"
            if ret_3m: response += f"• **3-Month**: {ret_3m*100:+.1f}%\n"
            if ret_6m: response += f"• **6-Month**: {ret_6m*100:+.1f}%\n"
            if ret_1y: response += f"• **1-Year**: {ret_1y*100:+.1f}%\n"
            
            # Technical indicators
            if len(hist) >= 50:
                ma_50 = hist['Close'].rolling(50).mean().iloc[-1]
                ma_200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else None
                
                response += "\n**Technical Signals**:\n"
                response += f"• **50-day MA**: ${ma_50:.2f} | Price is {((current_price/ma_50 - 1)*100):+.1f}% from MA\n"
                
                if ma_200:
                    response += f"• **200-day MA**: ${ma_200:.2f} | Price is {((current_price/ma_200 - 1)*100):+.1f}% from MA\n"
                    
                    if current_price > ma_50 > ma_200:
                        response += "• **Trend**: 📈 Confirmed uptrend (golden cross territory)\n"
                    elif current_price < ma_50 < ma_200:
                        response += "• **Trend**: 📉 Confirmed downtrend (death cross territory)\n"
                    else:
                        response += "• **Trend**: Mixed signals / Transitioning\n"
            
            # Volatility
            returns = hist['Close'].pct_change().dropna()
            if len(returns) > 20:
                volatility = returns.std() * np.sqrt(252)
                response += f"\n**Volatility**: {volatility*100:.1f}% annualized"
                if volatility > 0.40:
                    response += " (High volatility - higher risk)\n"
                elif volatility < 0.20:
                    response += " (Low volatility - more stable)\n"
                else:
                    response += " (Moderate)\n"
            
            response += "\n"
        else:
            # Variables won't exist, set defaults
            ret_1m = ret_3m = ret_6m = ret_1y = volatility = None
        
        # === ANALYST & MARKET SENTIMENT ===
        response += "### 🎯 Market Sentiment & Targets\n\n"
        
        target_mean = info.get('targetMeanPrice')
        target_high = info.get('targetHighPrice')
        target_low = info.get('targetLowPrice')
        recommendation = info.get('recommendationKey')
        
        upside = None
        if target_mean and live_data['success']:
            current = live_data['current_price']
            upside = ((target_mean / current) - 1) * 100
            
            response += "**Analyst Consensus**:\n"
            response += f"• **Mean Price Target**: ${target_mean:.2f} (Upside: {upside:+.1f}%)\n"
            if target_high and target_low:
                response += f"• **Target Range**: ${target_low:.2f} - ${target_high:.2f}\n"
            
            if recommendation:
                rec_map = {'buy': '🟢 BUY', 'strong_buy': '🟢🟢 STRONG BUY', 
                          'hold': '🟡 HOLD', 'sell': '🔴 SELL', 'strong_sell': '🔴🔴 STRONG SELL'}
                response += f"• **Recommendation**: {rec_map.get(recommendation, recommendation.upper())}\n"
        
        # Short interest
        short_pct = info.get('shortPercentOfFloat')
        if short_pct:
            response += f"\n**Short Interest**: {short_pct*100:.1f}% of float"
            if short_pct > 0.10:
                response += " ⚠️ (High short interest - squeeze potential or concerns)\n"
            else:
                response += "\n"
        
        response += "\n"
        
        # === INVESTMENT THESIS ===
        response += "### 💡 Investment Thesis\n\n"
        
        # Build bull/bear case
        response += "**Bull Case** 🐂:\n"
        bull_points = []
        
        if revenue_growth and revenue_growth > 0.15:
            bull_points.append(f"Strong revenue growth at {revenue_growth*100:.0f}% YoY - market share gains")
        if roe and roe > 0.15:
            bull_points.append(f"High ROE of {roe*100:.0f}% indicates competitive moat")
        if profit_margin and profit_margin > 0.15:
            bull_points.append(f"Fat margins at {profit_margin*100:.0f}% provide earnings leverage")
        if debt_to_equity and debt_to_equity < 50:
            bull_points.append("Strong balance sheet with low leverage")
        if ret_1y and ret_1y > 0.20:
            bull_points.append(f"Strong momentum with {ret_1y*100:.0f}% 1Y return")
        if target_mean and upside > 15:
            bull_points.append(f"Analysts see {upside:.0f}% upside potential")
        
        if bull_points:
            for point in bull_points[:4]:  # Top 4
                response += f"• {point}\n"
        else:
            response += "• Limited data for positive catalysts\n"
        
        response += "\n**Bear Case** 🐻:\n"
        bear_points = []
        
        if pe_ratio and pe_ratio > 30:
            bear_points.append(f"High P/E of {pe_ratio:.0f}x leaves little room for disappointment")
        if roe and roe < 0.10:
            bear_points.append(f"Weak ROE of {roe*100:.0f}% suggests competitive pressure")
        if revenue_growth and revenue_growth < 0:
            bear_points.append(f"Revenue declining at {revenue_growth*100:.0f}% - market share loss?")
        if debt_to_equity and debt_to_equity > 150:
            bear_points.append("High leverage creates financial risk")
        if ret_1y and ret_1y < -0.10:
            bear_points.append(f"Negative momentum with {ret_1y*100:.0f}% 1Y loss")
        if volatility and volatility > 0.40:
            bear_points.append(f"High volatility at {volatility*100:.0f}% means choppy ride ahead")
        
        if bear_points:
            for point in bear_points[:4]:  # Top 4
                response += f"• {point}\n"
        else:
            response += "• Limited identifiable risks in data\n"
        
        response += "\n"
        
        # === PORTFOLIO FIT ANALYSIS ===
        if portfolio_weights is not None and returns_data is not None:
            response += "### 📊 Portfolio Fit & Diversification\n\n"
            
            # Fetch fresh data for candidate if not in portfolio universe
            if candidate_ticker not in returns_data.columns:
                try:
                    candidate_data = yf.download(candidate_ticker, period='2y', progress=False, auto_adjust=False)
                    if not candidate_data.empty:
                        if isinstance(candidate_data.columns, pd.MultiIndex):
                            candidate_returns = candidate_data['Adj Close', candidate_ticker].pct_change().dropna()
                        else:
                            candidate_returns = candidate_data['Adj Close'].pct_change().dropna()
                    else:
                        candidate_returns = None
                except:
                    candidate_returns = None
            else:
                candidate_returns = returns_data[candidate_ticker].dropna()
            
            if candidate_returns is not None and len(candidate_returns) > 20:
                # Calculate portfolio returns
                portfolio_returns = (returns_data[portfolio_weights.index] * portfolio_weights).sum(axis=1)
                
                # Correlation analysis
                common_idx = candidate_returns.index.intersection(portfolio_returns.index)
                if len(common_idx) > 20:
                    correlation = candidate_returns[common_idx].corr(portfolio_returns[common_idx])
                    
                    response += f"**Diversification Score**: "
                    if abs(correlation) < 0.3:
                        response += f"🟢 **Excellent** (corr: {correlation:.2f})\n"
                        response += f"• Low correlation means strong diversification benefit\n"
                        response += f"• Adding {candidate_ticker} would reduce portfolio risk\n"
                    elif abs(correlation) < 0.6:
                        response += f"🟡 **Good** (corr: {correlation:.2f})\n"
                        response += f"• Moderate correlation provides some diversification\n"
                    else:
                        response += f"🔴 **Poor** (corr: {correlation:.2f})\n"
                        response += f"• High correlation - moves similarly to your portfolio\n"
                        response += f"• Limited diversification benefit from adding this position\n"
                
                # Risk-adjusted comparison
                candidate_vol = candidate_returns.std() * np.sqrt(252)
                portfolio_vol = portfolio_returns.std() * np.sqrt(252)
                
                candidate_sharpe = (candidate_returns.mean() * 252 - 0.03) / candidate_vol if candidate_vol > 0 else 0
                portfolio_sharpe = (portfolio_returns.mean() * 252 - 0.03) / portfolio_vol if portfolio_vol > 0 else 0
                
                response += f"\n**Risk-Return Profile**:\n"
                response += f"• {candidate_ticker} Sharpe: {candidate_sharpe:.2f} vs Portfolio: {portfolio_sharpe:.2f}"
                
                if candidate_sharpe > portfolio_sharpe * 1.2:
                    response += " ✓ (Better risk-adjusted returns)\n"
                elif candidate_sharpe < portfolio_sharpe * 0.8:
                    response += " ⚠️ (Worse risk-adjusted returns)\n"
                else:
                    response += " (Similar risk-adjusted returns)\n"
                
                response += f"• {candidate_ticker} volatility: {candidate_vol*100:.1f}% vs Portfolio: {portfolio_vol*100:.1f}%\n"
                
                if candidate_vol > portfolio_vol * 1.5:
                    response += f"• ⚠️ Adding this position would **increase** portfolio volatility significantly\n"
                elif candidate_vol < portfolio_vol * 0.7:
                    response += f"• ✓ Adding this position could **stabilize** portfolio volatility\n"
            else:
                response += f"_Portfolio fit analysis unavailable - insufficient overlapping data._\n"
            
            response += "\n"
        
        # === FINAL VERDICT ===
        response += "### 🎯 Investment Verdict\n\n"
        
        # Calculate aggregate score
        verdict_score = 0
        verdict_max = 0
        
        # Valuation
        if valuation_max > 0:
            verdict_score += (valuation_score / valuation_max) * 25
            verdict_max += 25
        
        # Growth
        if revenue_growth is not None:
            verdict_max += 25
            if revenue_growth > 0.15:
                verdict_score += 25
            elif revenue_growth > 0.05:
                verdict_score += 15
            elif revenue_growth > 0:
                verdict_score += 10
        
        # Profitability
        if roe is not None:
            verdict_max += 25
            if roe > 0.15:
                verdict_score += 25
            elif roe > 0.10:
                verdict_score += 15
            elif roe > 0.05:
                verdict_score += 10
        
        # Momentum
        if ret_3m is not None:
            verdict_max += 25
            if ret_3m > 0.10:
                verdict_score += 25
            elif ret_3m > 0:
                verdict_score += 15
            elif ret_3m > -0.10:
                verdict_score += 10
        
        if verdict_max > 0:
            final_score = (verdict_score / verdict_max) * 100
            
            response += f"**Overall Score**: {final_score:.0f}/100\n\n"
            
            if final_score >= 75:
                response += "**Rating**: 🟢 **STRONG BUY**\n"
                response += "_Compelling valuation, strong fundamentals, positive momentum. High conviction opportunity._\n"
            elif final_score >= 60:
                response += "**Rating**: 🟢 **BUY**\n"
                response += "_Solid fundamentals with attractive entry point. Good risk/reward._\n"
            elif final_score >= 45:
                response += "**Rating**: 🟡 **HOLD**\n"
                response += "_Mixed signals. Wait for better entry or catalyst. Monitor closely._\n"
            elif final_score >= 30:
                response += "**Rating**: 🔴 **UNDERPERFORM**\n"
                response += "_Multiple red flags. Consider reducing exposure or avoiding._\n"
            else:
                response += "**Rating**: 🔴 **SELL**\n"
                response += "_Significant concerns across valuation, fundamentals, and momentum. High risk._\n"
        else:
            response += "_Insufficient data for comprehensive rating._\n"
        
    except Exception as e:
        response += f"⚠️ **Analysis Error**: {str(e)[:150]}\n\n"
        response += "Could not complete analysis. Please verify ticker and data availability.\n"
    
    return response


def handle_education_question(question: str) -> str:
    """
    Handle educational/definitional questions about investing concepts
    """
    response = "## 📚 Investment Education\n\n"
    
    question_lower = question.lower()
    
    # Common investment terms
    if 'p/e ratio' in question_lower or 'price earnings' in question_lower:
        response += "**P/E Ratio (Price-to-Earnings Ratio)**\n\n"
        response += "**Definition**: Stock price divided by earnings per share (EPS)\n\n"
        response += "**Interpretation**:\n"
        response += "• High P/E (>20-25): Market expects strong future growth OR stock is expensive\n"
        response += "• Low P/E (<15): May be undervalued OR company has structural issues\n"
        response += "• Compare within same sector for meaningful comparison\n\n"
        response += "**Limitations**: Doesn't account for growth rates, debt levels, or industry differences\n"
    
    elif 'diversification' in question_lower:
        response += "**Diversification**\n\n"
        response += "**Definition**: Spreading investments across different assets to reduce risk\n\n"
        response += "**Core Principle**: Don't put all eggs in one basket\n\n"
        response += "**Key Dimensions**:\n"
        response += "• Across asset classes (stocks, bonds, real estate)\n"
        response += "• Across sectors (tech, healthcare, finance, etc.)\n"
        response += "• Across geographies (domestic vs. international)\n"
        response += "• Across company sizes (large-cap, mid-cap, small-cap)\n\n"
        response += "**Goal**: Reduce unsystematic (company-specific) risk while maintaining returns\n"
    
    elif 'sharpe ratio' in question_lower:
        response += "**Sharpe Ratio**\n\n"
        response += "**Definition**: Measure of risk-adjusted return\n\n"
        response += "**Formula**: (Portfolio Return - Risk-Free Rate) / Portfolio Volatility\n\n"
        response += "**Interpretation**:\n"
        response += "• >1.0: Good risk-adjusted performance\n"
        response += "• >2.0: Excellent risk-adjusted performance\n"
        response += "• <1.0: Poor risk-adjusted performance\n"
        response += "• Negative: Portfolio underperformed risk-free rate\n\n"
        response += "**Use Case**: Compare investments with different risk levels\n"
    
    elif 'beta' in question_lower:
        response += "**Beta**\n\n"
        response += "**Definition**: Measure of systematic risk relative to market\n\n"
        response += "**Interpretation**:\n"
        response += "• Beta = 1: Moves with market\n"
        response += "• Beta > 1: More volatile than market (amplifies moves)\n"
        response += "• Beta < 1: Less volatile than market (defensive)\n"
        response += "• Beta < 0: Moves opposite to market (rare)\n\n"
        response += "**Example**: Beta of 1.5 means stock typically moves 1.5% for every 1% market move\n"
    
    elif 'volatility' in question_lower:
        response += "**Volatility**\n\n"
        response += "**Definition**: Statistical measure of price fluctuation\n\n"
        response += "**Measurement**: Standard deviation of returns (typically annualized)\n\n"
        response += "**Interpretation**:\n"
        response += "• High volatility: Large price swings, higher uncertainty\n"
        response += "• Low volatility: More stable prices, lower risk\n"
        response += "• S&P 500 historical volatility: ~15-20% annualized\n\n"
        response += "**Note**: Volatility measures price fluctuation, not direction\n"
    
    elif 'correlation' in question_lower:
        response += "**Correlation**\n\n"
        response += "**Definition**: Statistical measure of how two assets move together\n\n"
        response += "**Range**: -1 to +1\n\n"
        response += "**Interpretation**:\n"
        response += "• +1: Perfect positive correlation (move together)\n"
        response += "• 0: No correlation (independent movements)\n"
        response += "• -1: Perfect negative correlation (move opposite)\n\n"
        response += "**For Diversification**: Seek assets with low or negative correlation\n"
    
    elif 'market cap' in question_lower or 'capitalization' in question_lower:
        response += "**Market Capitalization**\n\n"
        response += "**Definition**: Total market value of a company's outstanding shares\n\n"
        response += "**Formula**: Share Price × Number of Shares Outstanding\n\n"
        response += "**Categories**:\n"
        response += "• Large-cap: >$10B (established, lower risk)\n"
        response += "• Mid-cap: $2B-$10B (growth potential, moderate risk)\n"
        response += "• Small-cap: <$2B (high growth potential, higher risk)\n\n"
        response += "**Use**: Indicates company size and typically correlates with stability\n"
    
    else:
        response += f"I can explain many investment concepts including:\n\n"
        response += "• **Valuation**: P/E ratio, P/B ratio, EV/EBITDA\n"
        response += "• **Risk metrics**: Beta, volatility, correlation, Sharpe ratio\n"
        response += "• **Portfolio concepts**: Diversification, asset allocation, rebalancing\n"
        response += "• **Market terms**: Market cap, sectors, indices\n"
        response += "• **Returns**: Total return, dividend yield, capital gains\n\n"
        response += "Please ask about a specific term or concept!\n"
    
    return response


def extract_ticker_from_question(question: str) -> str:
    """
    Extract ticker symbol from user question with strict filtering
    """
    # Find capitalized ticker patterns (2-5 letter caps, more restrictive)
    matches = re.findall(r'\b([A-Z]{2,5})\b', question)
    
    # Aggressive filtering of common words
    common_words = {'IS', 'THE', 'AND', 'OR', 'FOR', 'WITH', 'THIS', 'THAT', 'SHOULD', 
                   'WOULD', 'COULD', 'GOOD', 'BAD', 'BEST', 'NOW', 'BASED', 'ON', 'MY',
                   'PORTFOLIO', 'INVESTMENT', 'DECISION', 'BUY', 'ADD', 'CONSIDER',
                   'RIGHT', 'BETTER', 'WORSE', 'MORE', 'LESS', 'THINK', 'ABOUT',
                   'GIVEN', 'DO', 'YOU', 'WE', 'THEY', 'WHAT', 'HOW', 'WHY',
                   'ARE', 'WILL', 'HAS', 'HAVE', 'HAD', 'CAN', 'MAY', 'MUST',
                   'TO', 'FROM', 'AT', 'IN', 'OF', 'BY', 'AS', 'UP', 'DOWN',
                   'REBALANCE', 'INVEST', 'SECTOR', 'MARKET', 'STOCK', 'STOCKS',
                   'PERFORM', 'PERFORMING', 'TODAY', 'RATE', 'RATES', 'FED', 'ETF',
                   'BULLISH', 'BEARISH', 'INFLATION', 'GDP', 'USA', 'USD'}
    
    tickers = [t for t in matches if t not in common_words and len(t) >= 2]
    
    return tickers[0] if tickers else ""


def handle_user_question(question: str, 
                         portfolio_weights: Optional[pd.Series] = None,
                         returns_data: Optional[pd.DataFrame] = None) -> str:
    """
    Main chatbot dispatcher with anti-hallucination measures
    
    Key Features:
    - Intelligent question classification with confidence scoring
    - Real data validation before analysis
    - Explicit fact vs. interpretation separation
    - Graceful handling of out-of-scope questions
    - Investment disclaimers on all advice
    """
    
    # Classify the question
    classification = classify_question_type(question)
    question_type = classification['type']
    entities = classification['entities']
    intent = classification['intent']
    confidence = classification.get('confidence', 0.5)
    
    # Low confidence warning
    if confidence < 0.6:
        prefix = f"_Note: Question classification confidence is {confidence:.0%}. Response may not fully address your intent._\n\n"
    else:
        prefix = ""
    
    try:
        # Route based on question type
        if question_type == 'education':
            return prefix + handle_education_question(question)
        
        elif question_type == 'comparison':
            # Stock comparison/ranking questions
            portfolio_stocks = portfolio_weights.index.tolist() if portfolio_weights is not None else None
            return prefix + analyze_stock_comparison(question, portfolio_stocks)
        
        elif question_type == 'stock' and entities:
            # Stock-specific analysis - ALWAYS provide comprehensive analysis
            ticker = entities[0]
            return prefix + analyze_investment_decision(question, portfolio_weights, ticker, returns_data)
        
        elif question_type == 'sector':
            # Sector/industry analysis
            sector_name = entities[0] if entities else classification.get('keywords', ['technology'])[0]
            return prefix + analyze_sector_industry(sector_name, portfolio_weights)
        
        elif question_type == 'market':
            # Overall market conditions
            return prefix + analyze_market_conditions()
        
        elif question_type == 'macro':
            # Macro economic commentary
            return prefix + analyze_macro_environment(classification.get('keywords', []))
        
        elif question_type == 'portfolio':
            # Portfolio-specific advice with clustering analysis
            if portfolio_weights is None or len(portfolio_weights) == 0:
                response = "## 💼 Portfolio Analysis\n\n"
                response += "⚠️ **No portfolio data available**.\n\n"
                response += "To get personalized portfolio analysis:\n"
                response += "1. Go to **Portfolio Builder** page\n"
                response += "2. Select stocks and build a portfolio\n"
                response += "3. Return here for contextualized advice\n"
                return response
            
            if returns_data is not None and len(portfolio_weights) >= 3:
                # Provide clustering analysis
                return prefix + analyze_portfolio_clustering(portfolio_weights, returns_data)
            else:
                # Basic portfolio info if insufficient data
                response = "## 💼 Your Portfolio\n\n"
                response += f"**Holdings**: {len(portfolio_weights)} positions\n"
                response += f"**Top positions**: {', '.join(portfolio_weights.nlargest(3).index.tolist())}\n\n"
                response += "_For comprehensive clustering analysis, ensure you have at least 3 stocks with sufficient historical data._\n\n"
                response += "_For other insights, visit:_\n"
                response += "• **Holdings Analysis** page: Risk factor exposure, beta, performance\n"
                response += "• **AI Analysis** page: LLM-powered insights and recommendations\n"
                return response
        
        else:
            # Unknown or general question
            response = "## 💬 How I Can Help\n\n"
            response += "I'm an **investment-focused AI assistant** with the following capabilities:\n\n"
            
            response += "### 📈 Stock Analysis\n"
            response += "Ask about specific companies using their ticker symbols.\n"
            response += "_Example: \"Is TSLA a good investment?\" or \"Analyze AAPL fundamentals\"_\n\n"
            
            response += "### 🏭 Sector & Industry Analysis\n"
            response += "Get performance data and outlook for market sectors.\n"
            response += "_Example: \"How is the technology sector performing?\"_\n\n"
            
            response += "### 📊 Market Conditions\n"
            response += "Current market regime, volatility, and sector rotation.\n"
            response += "_Example: \"What's the overall market outlook?\"_\n\n"
            
            response += "### 🌍 Macro Environment\n"
            response += "Interest rates, Fed policy, inflation, economic growth.\n"
            response += "_Example: \"How will rising rates affect stocks?\"_\n\n"
            
            response += "### 📚 Investment Education\n"
            response += "Learn about investing concepts and terminology.\n"
            response += "_Example: \"What is a Sharpe ratio?\" or \"Explain diversification\"_\n\n"
            
            response += "---\n\n"
            response += "**What I Cannot Do**:\n"
            response += "• Provide guaranteed predictions or future prices\n"
            response += "• Offer personalized financial advice (consult a licensed advisor)\n"
            response += "• Access real-time news or sentiment (data is delayed)\n"
            response += "• Execute trades or manage accounts\n\n"
            
            response += "_Please ask a specific question and I'll provide data-driven analysis!_\n"
            return response
    
    except Exception as e:
        error_response = "## ⚠️ Analysis Error\n\n"
        error_response += f"An error occurred while processing your question:\n"
        error_response += f"```\n{str(e)[:200]}\n```\n\n"
        error_response += "**Possible causes**:\n"
        error_response += "• Data provider (Yahoo Finance) temporary unavailability\n"
        error_response += "• Network connectivity issues\n"
        error_response += "• Invalid ticker symbol or sector name\n\n"
        error_response += "Please try again or rephrase your question.\n"
        return error_response
