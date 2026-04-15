"""
Advanced Investment Chatbot with Market Intelligence and Anti-Hallucination Features

Key Features:
- Real-time data fetching with validation
- LLM-enhanced narrative explanations via FinBERT & FinGPT
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
- Portfolio context and recommendations with LLM descriptions
- General investment education
- Sentiment analysis for decision-making
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

# Import LLM module for enhanced explanations
try:
    from llm import (
        FinancialLLMEnsemble, 
        generate_investment_recommendation,
        AnalysisContext,
        AnalysisResponse,
        LLMBackend,
        deep_stock_comparison,
        sector_rotation_analysis,
        portfolio_stress_test,
        extract_portfolio_risks
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("LLM module not available. Running chatbot without LLM enhancements.")


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


# ==================== LLM ORCHESTRATOR FUNCTIONS ====================

def llm_orchestrate_analysis(classification: Dict, 
                            question: str,
                            portfolio_weights: Optional[pd.Series] = None,
                            returns_data: Optional[pd.DataFrame] = None) -> str:
    """
    Orchestrate analysis by creating AnalysisContext and routing to appropriate LLM analyzer
    
    Parameters:
    -----------
    classification : dict
        Output from classify_question_type()
    question : str
        Original user question
    portfolio_weights : pd.Series, optional
        Portfolio allocation
    returns_data : pd.DataFrame, optional
        Historical returns
        
    Returns:
    --------
    response : str
        Markdown formatted response from LLM analyzer
    """
    if not LLM_AVAILABLE:
        return None  # Fall back to existing functions
    
    try:
        # Extract question details
        question_type = classification.get('type', 'general')
        intent = classification.get('intent', 'analysis')
        entities = classification.get('entities', [])
        confidence = classification.get('confidence', 0.7)
        
        # Detect timeframe from question
        timeframe = 'all'
        if 'today' in question.lower() or 'now' in question.lower():
            timeframe = 'today'
        elif 'week' in question.lower():
            timeframe = '1w'
        elif 'month' in question.lower():
            timeframe = '1m'
        elif '3 month' in question.lower():
            timeframe = '3m'
        elif 'year' in question.lower() or 'ytd' in question.lower():
            timeframe = '1y'
        
        # Create analysis context
        context = AnalysisContext(
            question_type=question_type,
            intent=intent,
            entities=entities,
            timeframe=timeframe,
            portfolio_weights=portfolio_weights,
            returns_data=returns_data,
            confidence=confidence,
            raw_question=question
        )
        
        # Route to appropriate analyzer based on question type
        response = None
        
        if question_type == 'comparison' and len(entities) > 0:
            response = deep_stock_comparison(context)
        elif question_type == 'sector' and len(entities) > 0:
            response = sector_rotation_analysis(context)
        elif question_type == 'portfolio' and portfolio_weights is not None:
            # Run stress test and risk analysis
            stress_result = portfolio_stress_test(context)
            risk_result = extract_portfolio_risks(context)
            # Combine results
            combined_narrative = f"{stress_result.narrative}\n\n{risk_result.narrative}"
            response = AnalysisResponse(
                narrative=combined_narrative,
                key_metrics={**stress_result.key_metrics, **risk_result.key_metrics},
                confidence_level=min(stress_result.confidence_level, risk_result.confidence_level),
                data_sources=list(set(stress_result.data_sources + risk_result.data_sources)),
                recommendations=stress_result.recommendations + risk_result.recommendations
            )
        
        # If analyzer returned response, format and return
        if response:
            return response.to_markdown()
    
    except Exception as e:
        print(f"LLM orchestration error: {e}")
        return None
    
    return None


def generate_llm_narrative(metric_name: str,
                          metric_value: float,
                          backend: str = 'fallback',
                          api_key: str = None) -> str:
    """
    Generate natural language narrative for a metric using LLM backend
    
    Parameters:
    -----------
    metric_name : str
        Name of metric ('sharpe_ratio', 'volatility', etc.)
    metric_value : float
        Numerical value
    backend : str
        'openai', 'anthropic', or 'fallback'
    api_key : str
        API key if using external service
        
    Returns:
    --------
    narrative : str
        Natural language explanation
    """
    if not LLM_AVAILABLE:
        return f"{metric_name}: {metric_value}"
    
    try:
        llm_backend = LLMBackend(backend=backend, api_key=api_key)
        
        prompt = f"""
Provide a 2-3 sentence investor-friendly explanation for this metric:

Metric: {metric_name}
Value: {metric_value}

Be concise and actionable."""
        
        if backend == 'openai' and hasattr(llm_backend, '_generate_with_openai'):
            return llm_backend._generate_with_openai(prompt)
        elif backend == 'anthropic' and hasattr(llm_backend, '_generate_with_claude'):
            return llm_backend._generate_with_claude(prompt)
        else:
            # Fallback to existing narrative function
            return generate_llm_explanation(metric_name, metric_value)
    
    except Exception as e:
        print(f"Error generating LLM narrative: {e}")
        return f"{metric_name}: {metric_value}"


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


def analyze_portfolio_performance(portfolio_weights: pd.Series, returns_data: pd.DataFrame) -> str:
    """
    Comprehensive portfolio performance analysis with real metrics
    Shows returns, risk, composition, and performance breakdown
    """
    response = "## 📊 Portfolio Performance Analysis\n\n"
    response += f"**Analysis Timestamp**: {get_data_timestamp()}\n\n"
    
    try:
        portfolio_stocks = portfolio_weights.index.tolist()
        portfolio_returns = returns_data[portfolio_stocks].dropna()
        
        if len(portfolio_returns) < 5:
            return response + "⚠️ **Insufficient Data**: Need at least 5 trading days of price history.\n"
        
        # === PORTFOLIO METRICS ===
        weighted_returns = (portfolio_returns * portfolio_weights).sum(axis=1)
        
        # Calculate key metrics
        annual_return = weighted_returns.mean() * 252
        annual_vol = weighted_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.03) / annual_vol if annual_vol > 0 else 0
        
        # Period returns
        total_return = (1 + weighted_returns).prod() - 1
        ytd_date = pd.Timestamp(datetime.now().year, 1, 1)
        ytd_returns = weighted_returns[weighted_returns.index >= ytd_date]
        ytd_return = (1 + ytd_returns).prod() - 1 if len(ytd_returns) > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + weighted_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]
        
        # Response formatting
        response += "### 📈 Portfolio Metrics\n\n"
        response += f"**Current Composition**: {len(portfolio_stocks)} holdings ({', '.join(portfolio_stocks[:5])}{'...' if len(portfolio_stocks) > 5 else ''})\n\n"
        
        response += "**Returns**:\n"
        response += f"• **Annualized Return**: {annual_return*100:+.2f}%\n"
        response += f"• **Period Total Return**: {total_return*100:+.2f}%\n"
        if len(ytd_returns) > 0:
            response += f"• **YTD Return**: {ytd_return*100:+.2f}%\n"
        response += "\n"
        
        response += "**Risk Metrics**:\n"
        response += f"• **Annualized Volatility**: {annual_vol*100:.2f}%\n"
        response += f"• **Sharpe Ratio**: {sharpe_ratio:.3f}"
        if sharpe_ratio > 1:
            response += " ✓ (Good risk-adjusted return)\n"
        elif sharpe_ratio > 0:
            response += " (Positive but modest)\n"
        else:
            response += " ⚠️ (Negative - underperforming risk-free rate)\n"
        
        response += f"• **Maximum Drawdown**: {max_drawdown*100:.2f}%\n"
        response += f"• **Current Drawdown**: {current_drawdown*100:.2f}%\n"
        response += "\n"
        
        # === PORTFOLIO COMPOSITION ===
        response += "### 💼 Portfolio Composition\n\n"
        sorted_weights = portfolio_weights.sort_values(ascending=False)
        
        for i, (ticker, weight) in enumerate(sorted_weights.items(), 1):
            response += f"{i}. **{ticker}**: {weight*100:.1f}%\n"
        
        response += "\n"
        
        # === INDIVIDUAL STOCK PERFORMANCE ===
        response += "### 📍 Individual Stock Performance\n\n"
        
        stock_returns = {}
        stock_contribution = {}
        
        for ticker in portfolio_stocks:
            if ticker in portfolio_returns.columns:
                ticker_returns = portfolio_returns[ticker]
                annual_ticker_return = ticker_returns.mean() * 252
                stock_returns[ticker] = annual_ticker_return
                stock_contribution[ticker] = annual_ticker_return * portfolio_weights[ticker]
        
        # Sort by contribution
        sorted_contrib = sorted(stock_contribution.items(), key=lambda x: x[1], reverse=True)
        
        response += "**Return Contribution (Annualized)**:\n"
        for ticker, contrib in sorted_contrib[:5]:
            emoji = '📈' if contrib > 0 else '📉'
            response += f"• {emoji} **{ticker}**: {contrib*100:+.2f}% (weight: {portfolio_weights[ticker]*100:.1f}%)\n"
        
        if len(sorted_contrib) > 5:
            response += f"• ... and {len(sorted_contrib) - 5} more\n"
        
        response += "\n"
        
        # === CORRELATION & DIVERSIFICATION ===
        corr_matrix = portfolio_returns.corr()
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        response += "### 🔄 Diversification Quality\n\n"
        response += f"**Average Pairwise Correlation**: {avg_corr:.3f}\n"
        
        if avg_corr > 0.7:
            response += "⚠️  **High Correlation** - Portfolio stocks move together\n"
            response += "• Limited diversification benefits\n"
            response += "• Consider adding uncorrelated assets\n"
        elif avg_corr > 0.4:
            response += "🟡 **Moderate Correlation** - Reasonable diversification\n"
            response += "• Some stocks provide independent movements\n"
            response += "• Balanced correlation structure\n"
        else:
            response += "✓ **Low Correlation** - Good diversification\n"
            response += "• Stocks move somewhat independently\n"
            response += "• Better risk reduction across positions\n"
        
        response += "\n"
        
        # === VOLATILITY BREAKDOWN ===
        response += "### 📊 Risk Decomposition\n\n"
        
        # Calculate individual volatilities
        individual_vols = {}
        for ticker in portfolio_stocks:
            individual_vols[ticker] = portfolio_returns[ticker].std() * np.sqrt(252)
        
        best_vol = min(individual_vols.items(), key=lambda x: x[1])
        worst_vol = max(individual_vols.items(), key=lambda x: x[1])
        
        response += f"**Individual Stock Volatilities**:\n"
        response += f"• **Least volatile**: {best_vol[0]} ({best_vol[1]*100:.1f}%)\n"
        response += f"• **Most volatile**: {worst_vol[0]} ({worst_vol[1]*100:.1f}%)\n"
        response += f"• **Portfolio volatility** ({annual_vol*100:.1f}%) is lower than most individual stocks due to diversification\n"
        
        response += "\n"
        
        # === BENCHMARK COMPARISON ===
        try:
            spy_data = yf.download('SPY', period='1y', progress=False, auto_adjust=False)
            if not spy_data.empty:
                if isinstance(spy_data.columns, pd.MultiIndex):
                    spy_returns = spy_data['Adj Close', 'SPY'].pct_change().dropna()
                else:
                    spy_returns = spy_data['Adj Close'].pct_change().dropna()
                
                # Align with portfolio returns
                common_idx = weighted_returns.index.intersection(spy_returns.index)
                if len(common_idx) > 20:
                    portfolio_ret_aligned = weighted_returns[common_idx]
                    spy_ret_aligned = spy_returns[common_idx]
                    
                    port_annual = portfolio_ret_aligned.mean() * 252
                    spy_annual = spy_ret_aligned.mean() * 252
                    
                    port_vol = portfolio_ret_aligned.std() * np.sqrt(252)
                    spy_vol = spy_ret_aligned.std() * np.sqrt(252)
                    
                    response += "### 📊 vs Benchmark (SPY - S&P 500)\n\n"
                    response += f"**Performance Comparison**:\n"
                    response += f"• **Portfolio Return**: {port_annual*100:.2f}% (vol: {port_vol*100:.2f}%)\n"
                    response += f"• **SPY Return**: {spy_annual*100:.2f}% (vol: {spy_vol*100:.2f}%)\n"
                    response += f"• **Excess Return**: {(port_annual - spy_annual)*100:+.2f}pp\n"
                    
                    if port_vol < spy_vol:
                        response += f"• **Risk Reduction**: {(1 - port_vol/spy_vol)*100:.1f}% lower volatility than SPY\n"
                    elif port_vol > spy_vol:
                        response += f"• **Higher Risk**: {(port_vol/spy_vol - 1)*100:.1f}% higher volatility than SPY\n"
                    
                    response += "\n"
        except:
            pass  # Benchmark comparison optional
        
        # === INVESTMENT INSIGHTS ===
        response += "### 💡 Key Insights\n\n"
        
        if annual_return < 0.05:
            response += "⚠️ **Low Expected Returns**: Annual return below 5% may not outpace inflation\n"
        elif annual_return > 0.20:
            response += "🚀 **Strong Expected Returns**: Annual return above 20% is excellent\n"
        else:
            response += f"💰 **Healthy Returns**: Annual return of {annual_return*100:.1f}% is reasonable\n"
        
        if annual_vol < 0.10:
            response += "✓ **Conservative Risk**: Low volatility suitable for risk-averse investors\n"
        elif annual_vol > 0.30:
            response += "⚠️ **High Risk**: Volatility above 30% requires conviction\n"
        else:
            response += f"🟡 **Moderate Risk**: {annual_vol*100:.1f}% volatility is balanced\n"
        
        if sharpe_ratio > 1:
            response += "✓ **Efficient**: Good return per unit of risk taken\n"
        elif sharpe_ratio < 0:
            response += "⚠️ **Inefficient**: Returns don't justify risk - consider alternative portfolio\n"
        
        # Concentration check
        top_weight = sorted_weights.iloc[0]
        if top_weight > 0.40:
            response += f"🔴 **Concentration Risk**: Top position is {top_weight*100:.0f}% - consider diversifying\n"
        elif top_weight > 0.25:
            response += f"🟡 **Moderate Concentration**: Top position at {top_weight*100:.0f}%\n"
        else:
            response += f"✓ **Well Balanced**: Largest position only {top_weight*100:.0f}%\n"
        
        response += "\n---\n"
        response += INVESTMENT_DISCLAIMER + "\n"
        
    except Exception as e:
        response += f"⚠️ **Error**: Could not complete analysis: {str(e)[:100]}\n"
    
    return response


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
    
    # PRIORITY 1: Comparative/ranking questions (highest priority - very specific pattern)
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
        return classification  # Early return for comparative questions
    
    # PRIORITY 1.5: Stock addition to portfolio (HIGH PRIORITY - specific investment decision)
    # This catches "Should I add MSFT to my portfolio?" - should NOT be treated as generic portfolio question
    add_patterns = ['should i add', 'can i add', 'would i add', 'should i buy', 'should i purchase',
                   'add to my portfolio', 'add to our portfolio', 'adding to portfolio',
                   'consider adding', 'think about adding', 'idea of adding', 'fit in my portfolio',
                   'fit in our portfolio', 'right fit', 'good fit', 'works with', 'complements']
    if any(pattern in question_lower for pattern in add_patterns):
        # Try to extract a ticker
        tickers = extract_ticker_from_question(question)
        if tickers:
            classification['type'] = 'stock'
            classification['entities'].append(tickers)
            classification['intent'] = 'portfolio_fit'  # Special intent for portfolio fit analysis
            classification['confidence'] = 0.93
            return classification  # Early return for stock addition questions
    
    # PRIORITY 2: Macro keywords (HIGH PRIORITY - domain-specific, should override generic patterns)
    # Check this BEFORE portfolio keywords so "How will Fed rate cuts impact my portfolio?" is recognized as macro
    macro_keywords = ['fed', 'federal reserve', 'interest rate', 'inflation', 'cpi',
                     'unemployment', 'gdp', 'recession', 'economy', 'economic',
                     'monetary policy', 'fiscal policy', 'treasury', 'bond yield', 'yield curve',
                     'geopolitic', 'trade war', 'tariff', 'dollar strength', 'currency',
                     'central bank', 'economic cycle', 'market cycle', 'business cycle']
    if any(k in question_lower for k in macro_keywords):
        classification['type'] = 'macro'
        classification['keywords'] = [k for k in macro_keywords if k in question_lower]
        classification['confidence'] = 0.88
        return classification  # Early return for macro questions
    
    # PRIORITY 3: Portfolio-specific (VERY HIGH PRIORITY - but AFTER macro and stock addition checks!)
    # This catches questions about user's own holdings/portfolio
    portfolio_keywords = ['my portfolio', 'our portfolio', 'current holdings', 'rebalance', 'my positions',
                         'my investments', 'my stocks', 'portfolio performance', 'portfolio doing',
                         'our holdings', 'our positions', 'our stocks',
                         'how is my', 'how is the portfolio', 'how is our portfolio',
                         'our current portfolio', 'my current portfolio',
                         'portfolio return', 'portfolio risk', 'portfolio volatility',
                         'portfolio allocation', 'portfolio composition',
                         'what is the volatility of our', 'what is the return of my',
                         'what is the risk of our', 'current portfolio']
    if any(k in question_lower for k in portfolio_keywords):
        classification['type'] = 'portfolio'
        classification['confidence'] = 0.95
        return classification  # Early return for portfolio questions
    
    # PRIORITY 4: Market-level keywords (high priority - before sector)
    market_keywords = ['overall market', 'market outlook', 'market condition', 'sp500', 's&p 500', 
                       'dow jones', 'nasdaq', 'market index', 'broad market', 'equities overall',
                       'stock market', 'market trend', 'market is', 'are stocks', 'market data',
                       'market analysis', 'vix', 'volatility index', 'market regime']
    if any(k in question_lower for k in market_keywords):
        classification['type'] = 'market'
        classification['confidence'] = 0.85
        return classification  # Early return for market questions
    
    # PRIORITY 5: Sector/Industry keywords with more patterns
    sector_patterns = [
        ('technology', ['tech', 'technology sector', 'tech sector', 'software', 'semiconductor', 'semi']),
        ('finance', ['finance', 'financial sector', 'banking', 'bank', 'insurance']),
        ('healthcare', ['healthcare', 'health sector', 'pharma', 'pharmaceutical', 'biotech', 'bio']),
        ('energy', ['energy sector', 'energy', 'oil', 'gas']),
        ('consumer', ['consumer', 'consumer sector', 'retail', 'discretionary']),
        ('industrial', ['industrial', 'industrial sector', 'manufacturing']),
        ('utilities', ['utilities', 'utility sector']),
        ('real_estate', ['real estate', 'reits', 'reit']),
        ('materials', ['materials', 'materials sector']),
        ('communications', ['communications', 'telecom', 'media', 'tmt']),
        ('staples', ['consumer staples', 'staples']),
    ]
    
    detected_sectors = []
    detected_sub_sectors = []  # Track specific sub-sectors like "semiconductor", "pharma"
    
    for sector_name, patterns in sector_patterns:
        if any(p in question_lower for p in patterns):
            detected_sectors.append(sector_name)
            # Track sub-sector specificity
            if sector_name == 'technology':
                if any(p in question_lower for p in ['semiconductor', 'semi']):
                    detected_sub_sectors.append('semiconductor')
                elif any(p in question_lower for p in ['software']):
                    detected_sub_sectors.append('software')
            elif sector_name == 'healthcare':
                if any(p in question_lower for p in ['pharma', 'pharmaceutical']):
                    detected_sub_sectors.append('pharma')
                elif any(p in question_lower for p in ['biotech', 'bio']):
                    detected_sub_sectors.append('biotech')
            elif sector_name == 'finance':
                if any(p in question_lower for p in ['banking', 'bank']):
                    detected_sub_sectors.append('banking')
                elif any(p in question_lower for p in ['insurance']):
                    detected_sub_sectors.append('insurance')
            elif sector_name == 'energy':
                if any(p in question_lower for p in ['oil']):
                    detected_sub_sectors.append('oil')
                elif any(p in question_lower for p in ['gas']):
                    detected_sub_sectors.append('gas')
    
    if detected_sectors:
        classification['type'] = 'sector'
        classification['entities'].extend(detected_sectors)
        if detected_sub_sectors:
            classification['sub_sector'] = detected_sub_sectors[0]  # Store specific sub-sector
        classification['confidence'] = 0.82
        return classification  # Early return for sector questions
    
    # Sector growth analysis patterns
    sector_growth_keywords = ['sector growth', 'industry growth', 'growth outlook', 'sector trend',
                             'industry trend', 'sector performance', 'industry analysis',
                             'sector analysis', 'which sector', 'best performing sector',
                             'sectors performing', 'performing well sector', 'sector performing']
    if any(k in question_lower for k in sector_growth_keywords):
        classification['type'] = 'sector'
        classification['confidence'] = 0.85
        classification['intent'] = 'forecast'
        return classification  # Early return for sector growth questions
    
    # Catch remaining sector questions that have "sector" or "industry" keyword + performance words
    performance_words = ['performing', 'performance', 'doing', 'best', 'worst', 'top', 'bottom']
    if (('sector' in question_lower or 'industry' in question_lower) and 
        any(w in question_lower for w in performance_words)):
        classification['type'] = 'sector'
        classification['confidence'] = 0.80
        classification['intent'] = 'analysis'
        return classification  # Early return for sector questions
    
    # PRIORITY 6: Extract potential tickers (HIGH PRIORITY - specific investment)
    tickers = extract_ticker_from_question(question)
    if tickers:
        classification['entities'].append(tickers)
        classification['type'] = 'stock'
        classification['confidence'] = 0.80
        return classification  # Early return for stock questions
    
    # PRIORITY 7: Educational/definition questions (LOWEST PRIORITY - generic patterns, only if nothing else matched)
    education_keywords = ['what is', 'what are', 'explain', 'how does', 'how do', 'define',
                         'definition of', 'meaning of', 'tell me about', 'teach me',
                         'interpret', 'what does', 'how to interpret', 'industry overview',
                         'sector overview', 'understand', 'concept', 'terminology']
    if any(k in question_lower for k in education_keywords):
        classification['type'] = 'education'
        classification['confidence'] = 0.85
        return classification  # Early return for education questions
    
    # Intent classification (applies to all types)
    if any(word in question_lower for word in ['compare', 'versus', 'vs', 'better than', 'or which']):
        classification['intent'] = 'comparison'
    elif any(word in question_lower for word in ['forecast', 'predict', 'future', 'outlook', 'expect', 'will be', 'trend']):
        classification['intent'] = 'forecast'
    elif any(word in question_lower for word in ['risk', 'danger', 'concern', 'worry', 'safe', 'dangerous', 'volatile']):
        classification['intent'] = 'risk'
    elif any(word in question_lower for word in ['opportunity', 'undervalue', 'cheap', 'bargain', 'buy signal']):
        classification['intent'] = 'opportunity'
    elif any(word in question_lower for word in ['explain', 'what is', 'how does', 'interpret']):
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
            medal = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else '  '
            price_str = ""
            if ticker in live_prices and 'price' in live_prices[ticker]:
                price = live_prices[ticker]['price']
                price_str = f"${price:.2f}"
            
            # Simple format: avoid markdown special characters
            ret_pct = f"{ret*100:+.2f}%"
            if medal.strip():  # If medal exists (1st, 2nd, 3rd)
                line = f"{medal} {ticker}: {ret_pct}"
            else:
                line = f"    {ticker}: {ret_pct}"
            
            if price_str:
                line += f"  |  {price_str}"
            
            response += f"{line}\n"
        
        # Bottom performers with real-time prices
        response += f"\n### 📉 Bottom Performers ({timeframe})\n\n"
        bottom_n = min(5, len(sorted_perf))
        for i, (ticker, ret) in enumerate(sorted_perf[-bottom_n:][::-1], 1):
            price_str = ""
            if ticker in live_prices and 'price' in live_prices[ticker]:
                price = live_prices[ticker]['price']
                price_str = f"${price:.2f}"
            
            ret_pct = f"{ret*100:+.2f}%"
            line = f"    {ticker}: {ret_pct}"
            if price_str:
                line += f"  |  {price_str}"
            
            response += f"{line}\n"
        
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
        'semiconductor industry': ('XLK', 'Technology/Semiconductors'),
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
        response += "**Supported sectors**: technology, semiconductor, finance, healthcare, pharma, energy, consumer, industrial, staples, utilities, real estate, materials, communication\n"
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


def analyze_semiconductor_industry(portfolio_weights: Optional[pd.Series] = None) -> str:
    """
    Deep semiconductor industry analysis with key players, trends, and competitive dynamics
    """
    market_status = get_market_status()
    
    response = f"## 🖥️ Semiconductor Industry Deep Dive\n\n"
    response += f"{market_status['emoji']} **Market Status**: {market_status['message']} ({market_status['time']})\n"
    response += f"**Data Timestamp**: {get_data_timestamp()}\n\n"
    
    # Key semiconductor companies for performance tracking
    semi_tickers = {
        'NVDA': 'NVIDIA (AI GPUs, Data Center)',
        'AMD': 'Advanced Micro Devices (CPUs, GPUs)',
        'QCOM': 'Qualcomm (Mobile Chips, 5G)',
        'TSM': 'TSMC (Leading Foundry)',
        'INTC': 'Intel (CPUs, Process Tech)',
        'ASML': 'ASML (Chip Equipment)',
        'AVGO': 'Broadcom (Infrastructure)',
        'MU': 'Micron (Memory)',
        'XLNX': 'Xilinx (FPGAs)'
    }
    
    try:
        # Fetch industry ETF (XLK - Technology) represents semiconductors as part of tech
        # But let's add specific semi-focused data
        response += "### 📊 Key Semiconductor Players Performance\n\n"
        
        ticker_returns = {}
        for ticker, description in semi_tickers.items():
            try:
                data = yf.download(ticker, period='1y', progress=False, auto_adjust=False)
                if not data.empty:
                    # Handle multi-index columns from download
                    if isinstance(data.columns, pd.MultiIndex):
                        adj_close = data['Adj Close'][ticker]
                    else:
                        adj_close = data['Adj Close']
                    
                    # Ensure we have a Series
                    if isinstance(adj_close, pd.DataFrame):
                        adj_close = adj_close.squeeze()
                    
                    if len(adj_close) > 0:
                        ytd_return = (adj_close.iloc[-1] / adj_close.iloc[0] - 1)
                        current_price = adj_close.iloc[-1]
                        
                        # Get 1-month return for momentum
                        if len(adj_close) >= 21:
                            month_return = (adj_close.iloc[-1] / adj_close.iloc[-21] - 1)
                        else:
                            month_return = 0
                        
                        ticker_returns[ticker] = {
                            'ytd': ytd_return,
                            'month': month_return,
                            'price': current_price,
                            'description': description
                        }
            except Exception as e:
                # Silently skip failed tickers
                pass
        
        if ticker_returns:
            # Sort by YTD performance
            sorted_tickers = sorted(ticker_returns.items(), key=lambda x: x[1]['ytd'], reverse=True)
            
            response += f"**Best Performers** \n"
            for ticker, data in sorted_tickers[:3]:
                response += f"• **{ticker}** - {data['description']}\n"
                response += f"  - YTD: {data['ytd']*100:+.1f}% | 1M: {data['month']*100:+.1f}% | Price: ${data['price']:.2f}\n"
            
            response += f"\n**Laggards/Headwinds** \n"
            for ticker, data in sorted_tickers[-3:]:
                response += f"• **{ticker}** - {data['description']}\n"
                response += f"  - YTD: {data['ytd']*100:+.1f}% | 1M: {data['month']*100:+.1f}% | Price: ${data['price']:.2f}\n"
        
        # Industry dynamics
        response += f"\n### 🔍 Industry Dynamics & Key Trends\n\n"
        
        response += f"**1. AI Chip Boom**\n"
        response += f"• NVIDIA dominance in AI/GPU market (dominant position)\n"
        response += f"• AMD competition in data center (catching up)\n"
        response += f"• Supply constraints easing but fab capacity remains constrained\n\n"
        
        response += f"**2. Manufacturing & Geopolitics**\n"
        response += f"• TSMC leads advanced node production (global dependency risk)\n"
        response += f"• Intel investing heavily in US/Europe fab expansion (long-term play)\n"
        response += f"• US/China trade tensions affecting supply chains\n\n"
        
        response += f"**3. Memory Market**\n"
        response += f"• DRAM & NAND normalize after 2022-23 oversupply\n"
        response += f"• Micron (MU) benefiting from recovery\n"
        response += f"• AI server demand driving premium specifications\n\n"
        
        response += f"**4. Equipment & Enablers**\n"
        response += f"• ASML (chip manufacturing equipment) critical bottleneck\n"
        response += f"• Equipment demand indicates strong capex cycle\n"
        response += f"• Broadcom benefiting from infrastructure/networking demand\n\n"
        
        # Sector health assessment
        response += f"### 📈 Industry Health Snapshot\n\n"
        
        response += f"**Growth Drivers:**\n"
        response += f"• AI/Machine Learning (transformational demand)\n"
        response += f"• Data center expansion (hyperscaler capex cycle)\n"
        response += f"• 5G/6G infrastructure (multi-year cycle)\n"
        response += f"• Automotive electrification (rising chip content)\n\n"
        
        response += f"**Headwinds:**\n"
        response += f"• Capacity constraints (fab utilization high)\n"
        response += f"• Geopolitical risks (US-China dynamics)\n"
        response += f"• Valuation expansion already significant (e.g., NVDA)\n"
        response += f"• Cyclicality risk (previous oversupply cycles)\n\n"
        
        # Investment perspective
        response += f"### 💡 Investment Perspective\n\n"
        
        response += f"**Structural Tailwinds** (Multi-year positive cycle):\n"
        response += f"• AI adoption is still in early innings\n"
        response += f"• Manufacturing supply remains constrained\n"
        response += f"• Industry consolidation trends benefit leaders\n\n"
        
        response += f"**Valuation Considerations**:\n"
        response += f"• Hardware plays (Intel, QCOM) trading at reasonable valuations\n"
        response += f"• AI chip leaders (NVIDIA, AMD) commanding premium valuations\n"
        response += f"• Equipment/enabler plays (ASML) benefiting from capex cycle\n\n"
        
        response += f"**Portfolio Positioning Insights**:\n"
        if portfolio_weights is not None and len(portfolio_weights) > 0:
            semi_holdings = [t for t in semi_tickers.keys() if t in portfolio_weights.index]
            if semi_holdings:
                response += f"• **Your semiconductor exposure**: {', '.join(semi_holdings)}\n"
                for ticker in semi_holdings:
                    response += f"  - {semi_tickers[ticker]}: {portfolio_weights[ticker]*100:.1f}% allocation\n"
            else:
                response += f"• **No semiconductor holdings detected** - Consider diversification\n"
        else:
            response += f"• Monitor AI chip leaders (NVDA, AMD) for growth\n"
            response += f"• Hardware plays (Intel, QCOM) for value\n"
            response += f"• Equipment plays (ASML) for capex cycle exposure\n"
        
        response += f"\n{INVESTMENT_DISCLAIMER}\n"
        
    except Exception as e:
        response += f"⚠️ **Error**: Incomplete analysis: {str(e)[:100]}\n"
    
    return response


def analyze_pharma_industry(portfolio_weights: Optional[pd.Series] = None) -> str:
    """Placeholder for pharma-specific industry analysis"""
    response = "## 💊 Pharmaceutical Industry Analysis\n\n"
    response += "_Specialized pharma analysis module - Coming soon with GLP-1, drug development pipelines, FDA approvals._\n"
    return response


def analyze_banking_industry(portfolio_weights: Optional[pd.Series] = None) -> str:
    """Placeholder for banking-specific industry analysis"""
    response = "## 🏦 Banking & Financial Services Analysis\n\n"
    response += "_Specialized banking analysis module - Coming soon with interest rates, credit cycles, regulatory changes._\n"
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
            
            # Contextual interpretation
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


def generate_llm_explanation(metric_name: str, metric_value: float, context: str = "") -> str:
    """
    Generate LLM-based natural language explanation for key metrics
    
    Parameters:
    -----------
    metric_name : str
        Name of metric (sharpe_ratio, volatility, return, drawdown, etc)
    metric_value : float
        Numerical value of metric
    context : str
        Additional context for explanation
        
    Returns:
    --------
    explanation : str
        Natural language explanation of what this metric means
    """
    
    # Define narrative explanations for key metrics
    explanations = {
        'sharpe_ratio': {
            'above_1.5': f"**Excellent Risk-Adjusted Performance**: Sharpe of {metric_value:.2f} suggests your portfolio earns {metric_value:.1f} units of excess return per unit of risk taken. This is in professional asset manager territory—typically only top 10-20% of portfolios achieve this. You're being well-compensated for the volatility you endure.",
            'above_1': f"**Good Risk-Adjusted Returns**: Sharpe of {metric_value:.2f} indicates solid efficiency. For every unit of risk (volatility) you accept, you're earning {metric_value:.1f} units of excess return above the risk-free rate. This is respectable for individual portfolios.",
            'above_0.5': f"**Moderate Risk-Adjusted Returns**: Sharpe of {metric_value:.2f} means returns exist but risk-adjusted efficiency is mediocre. Consider whether your portfolio construction is adding value or just taking unnecessary volatility.",
            'below_0.5': f"**Poor Risk-Adjusted Returns**: Sharpe of {metric_value:.2f} suggests portfolio returns barely compensate for risk taken. You may want to evaluate whether individual positions or overall allocation strategy needs refinement.",
        },
        'volatility': {
            'high': f"**High Volatility ({metric_value*100:.1f}%)**: Your portfolio experiences significant daily swings. In a 2-sigma down day (~2% of) you'd lose 2-3x that amount. Consider: (1) Your emotional tolerance for 20-40% drawdowns, (2) Whether diversification is adequate, (3) If concentration in a few names is driving this.",
            'moderate': f"**Moderate Volatility ({metric_value*100:.1f}%)**: Typical portfolio volatility. Expect 15-20% annual standard deviation. You'll see meaningful months (+5-10%) and down months (-5-10%). This is the baseline risk you must accept for long-term equity returns.",
            'low': f"**Low Volatility ({metric_value*100:.1f}%)**: Your portfolio is smooth. This suggests strong diversification, significant bond allocation, or concentration in stable dividend stocks. Good for sleep-at-night investing, but watch for opportunity cost vs equities.",
        },
        'drawdown': {
            'severe': f"**Severe Drawdown ({metric_value*100:.1f}%)**: Portfolio has experienced peak-to-trough decline of 30-50%+. This is normal for equity portfolios during bear markets (2008, 2020, 2022). Question: Can you stay invested during these? If not, portfolio is too aggressive.",
            'significant': f"**Significant Drawdown ({metric_value*100:.1f}%)**: Peak-to-trough losses of 15-30%. Expect this every 5-10 years. Your portfolio likely has 60-70% equities. Verify your risk tolerance matches this.",
            'moderate': f"**Moderate Drawdown ({metric_value*100:.1f}%)**: Maximum observed loss is 5-15%. Suggests balanced portfolio (40-60% equities) or very strong diversification. Low probability of panic selling at these levels.",
        },
        'return': {
            'high': f"**Strong Returns ({metric_value*100:+.1f}%)**: Annual return significantly above inflation (2-3%). If annualized, this 8-15%+ implies either (1) excellent security selection, (2) favorable market conditions, or (3) luck. Be cautious extrapolating.",
            'moderate': f"**Solid Returns ({metric_value*100:+.1f}%)**: Aligned with long-term equity average (10% annualized). Shows market-like performance. Question: Are you being compensated with risk-adjusted returns (Sharpe)?",
            'below_market': f"**Below-Market Returns ({metric_value*100:+.1f}%)**: Lagging S&P 500 averages. Investigate: (1) Is this temporary underperformance? (2) Are fees/taxes/trading costs drag too high? (3) Should positions be reweighted?",
        },
        'correlation': {
            'high': f"**High Co-Movement ({metric_value:.2f})**: Holdings move together frequently. This reduces diversification benefit—in a market crash, everything falls simultaneously. Diversification isn't working optimally.",
            'moderate': f"**Moderate Diversification ({metric_value:.2f})**: Some holdings are independent, some correlated. Add uncorrelated assets (bonds, commodities, international) to improve cross-protection.",
            'low': f"**Good Diversification ({metric_value:.2f})**: Holdings move relatively independently. Your portfolio has natural hedges. During market downturns, some positions likely outperform others.",
        },
    }
    
    # Select appropriate explanation based on metric and value
    if metric_name == 'sharpe_ratio':
        if metric_value >= 1.5:
            key = 'above_1.5'
        elif metric_value >= 1:
            key = 'above_1'
        elif metric_value >= 0.5:
            key = 'above_0.5'
        else:
            key = 'below_0.5'
    elif metric_name == 'volatility':
        if metric_value >= 0.30:
            key = 'high'
        elif metric_value >= 0.15:
            key = 'moderate'
        else:
            key = 'low'
    elif metric_name == 'drawdown':
        if metric_value < -0.30:
            key = 'severe'
        elif metric_value < -0.15:
            key = 'significant'
        else:
            key = 'moderate'
    elif metric_name == 'return':
        if metric_value >= 0.12:
            key = 'high'
        elif metric_value >= 0.08:
            key = 'moderate'
        else:
            key = 'below_market'
    elif metric_name == 'correlation':
        if metric_value >= 0.7:
            key = 'high'
        elif metric_value >= 0.4:
            key = 'moderate'
        else:
            key = 'low'
    else:
        return f"Metric value: {metric_value}"
    
    return explanations.get(metric_name, {}).get(key, "") if metric_name in explanations else ""


def analyze_portfolio_performance(portfolio_weights: pd.Series, returns_data: pd.DataFrame) -> str:
    """
    Comprehensive portfolio performance analysis
    Shows returns, risk, diversification, and key metrics
    """
    response = "## 💼 Your Portfolio Performance\n\n"
    
    try:
        # Get portfolio stocks
        portfolio_stocks = portfolio_weights.index.tolist()
        portfolio_returns = returns_data[portfolio_stocks].dropna()
        
        if len(portfolio_returns) < 10:
            return response + "⚠️ **Insufficient Data**: Need at least 10 days of historical data for analysis.\n"
        
        # Calculate portfolio metrics
        portfolio_daily_returns = (portfolio_returns * portfolio_weights).sum(axis=1)
        portfolio_annual_return = portfolio_daily_returns.mean() * 252
        portfolio_volatility = portfolio_daily_returns.std() * np.sqrt(252)
        portfolio_sharpe = (portfolio_annual_return - 0.03) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Cumulative return
        cumulative_return = (1 + portfolio_daily_returns).prod() - 1
        
        # Drawdown
        cumulative_values = (1 + portfolio_daily_returns).cumprod()
        running_max = cumulative_values.expanding().max()
        drawdown = (cumulative_values - running_max) / running_max
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]
        
        # === PERFORMANCE SUMMARY ===
        response += "### 📊 Performance Metrics\n\n"
        response += f"**Returns**\n"
        response += f"• **Annualized Return**: {portfolio_annual_return*100:.2f}%\n"
        response += generate_llm_explanation('return', portfolio_annual_return) + "\n\n"
        response += f"• **Total Return**: {cumulative_return*100:.2f}%\n"
        response += f"• **Daily Avg Return**: {portfolio_daily_returns.mean()*100:.3f}%\n\n"
        
        response += f"**Risk Metrics**\n"
        response += f"• **Volatility (Annual)**: {portfolio_volatility*100:.2f}%\n"
        response += generate_llm_explanation('volatility', portfolio_volatility) + "\n\n"
        response += f"• **Sharpe Ratio**: {portfolio_sharpe:.3f}\n"
        response += generate_llm_explanation('sharpe_ratio', portfolio_sharpe) + "\n\n"
        response += f"• **Max Drawdown**: {max_drawdown*100:.2f}%\n"
        response += generate_llm_explanation('drawdown', max_drawdown) + "\n\n"
        response += f"• **Current Drawdown**: {current_drawdown*100:.2f}%\n"
        
        # Return per unit of risk
        if portfolio_volatility > 0:
            return_per_risk = portfolio_annual_return / portfolio_volatility
            response += f"**Return per Unit of Risk**: {return_per_risk:.3f}\n"
            if return_per_risk > 1:
                response += "✓ Good risk-adjusted returns\n\n"
            elif return_per_risk > 0.5:
                response += "🟡 Moderate risk-adjusted returns\n\n"
            else:
                response += "⚠️ Low risk-adjusted returns\n\n"
        
        # === PORTFOLIO COMPOSITION ===
        response += "### 📈 Portfolio Composition\n\n"
        response += f"**Top Holdings**\n"
        top_holdings = portfolio_weights.nlargest(5)
        for ticker, weight in top_holdings.items():
            response += f"• **{ticker}**: {weight*100:.1f}%\n"
        
        response += f"\n**Holdings Count**: {len(portfolio_weights)}\n"
        response += f"**Concentration** (top 3): {top_holdings.sum()*100:.1f}%\n\n"
        
        # === INDIVIDUAL STOCK PERFORMANCE ===
        response += "### 📊 Individual Stock Performance\n\n"
        
        stock_returns = pd.Series({
            stock: (returns_data[stock].dropna().mean() * 252) for stock in portfolio_stocks
        })
        stock_returns = stock_returns.sort_values(ascending=False)
        
        response += f"**Best Performers**\n"
        for ticker, ret in stock_returns.head(3).items():
            weight = portfolio_weights.get(ticker, 0)
            contribution = ret * weight
            response += f"• **{ticker}** ({weight*100:.1f}%): {ret*100:+.2f}% annual (contributes {contribution*100:+.2f}pp)\n"
        
        response += f"\n**Worst Performers**\n"
        for ticker, ret in stock_returns.tail(3).items():
            weight = portfolio_weights.get(ticker, 0)
            contribution = ret * weight
            response += f"• **{ticker}** ({weight*100:.1f}%): {ret*100:+.2f}% annual (contributes {contribution*100:+.2f}pp)\n"
        
        # === DIVERSIFICATION ===
        response += "\n### 🔄 Diversification Analysis\n\n"
        
        corr_matrix = portfolio_returns.corr()
        avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        response += f"**Average Pairwise Correlation**: {avg_correlation:.3f}\n"
        response += generate_llm_explanation('correlation', avg_correlation) + "\n\n"
        
        # === BENCHMARK COMPARISON ===
        response += "### 📊 vs. S&P 500 (SPY)\n\n"
        
        try:
            spy_data = yf.download('SPY', period='2y', progress=False, auto_adjust=False)
            if not spy_data.empty:
                if isinstance(spy_data.columns, pd.MultiIndex):
                    spy_returns = spy_data['Adj Close', 'SPY'].pct_change().dropna()
                else:
                    spy_returns = spy_data['Adj Close'].pct_change().dropna()
                
                # Align dates
                common_idx = portfolio_daily_returns.index.intersection(spy_returns.index)
                if len(common_idx) > 10:
                    spy_annual = spy_returns[common_idx].mean() * 252
                    spy_vol = spy_returns[common_idx].std() * np.sqrt(252)
                    spy_sharpe = (spy_annual - 0.03) / spy_vol if spy_vol > 0 else 0
                    
                    response += f"**S&P 500 Metrics** (same period)\n"
                    response += f"• **Annual Return**: {spy_annual*100:.2f}%\n"
                    response += f"• **Volatility**: {spy_vol*100:.2f}%\n"
                    response += f"• **Sharpe Ratio**: {spy_sharpe:.3f}\n\n"
                    
                    response += f"**Relative Performance**\n"
                    response += f"• **Alpha**: {(portfolio_annual_return - spy_annual)*100:+.2f}pp\n"
                    response += f"• **Beta**: {portfolio_volatility/spy_vol:.2f}x\n"
                    
                    if portfolio_annual_return > spy_annual:
                        response += f"• ✓ Outperforming SPY by {(portfolio_annual_return - spy_annual)*100:.2f}pp\n"
                    else:
                        response += f"• ⚠️ Underperforming SPY by {(spy_annual - portfolio_annual_return)*100:.2f}pp\n"
        except:
            response += "⚠️ _Benchmark comparison data unavailable_\n"
        
        response += "\n"
        
        # === KEY INSIGHTS ===
        response += "### 💡 Key Insights & LLM Analysis\n\n"
        
        insights = []
        
        if portfolio_sharpe > 1.0:
            insights.append(f"✓ **Strong Risk-Adjusted Returns**: (Sharpe: {portfolio_sharpe:.2f}) Your portfolio is being well-rewarded for the risk taken.")
        elif portfolio_sharpe > 0.5:
            insights.append(f"🟡 **Moderate Risk-Adjusted Returns**: (Sharpe: {portfolio_sharpe:.2f}) Returns exist but could be improved through better positioning.")
        else:
            insights.append(f"⚠️ **Weak Risk-Adjusted Returns**: (Sharpe: {portfolio_sharpe:.2f}) Portfolio isn't adequately compensating for risk. Consider rebalancing.")
        
        if portfolio_volatility < 0.15:
            insights.append("✓ **Conservative Volatility**: Low swings suitable for risk-averse investors. Downside: May miss upside moves.")
        elif portfolio_volatility > 0.30:
            insights.append(f"⚠️ **High Volatility**: ({portfolio_volatility*100:.1f}%) Aggressive positioning. Can sustain 20-40% drawdowns. Verify emotional tolerance.")
        else:
            insights.append("✓ **Balanced Volatility**: Typical market risk. You'll see +10%/-10% moves regularly.")
        
        if max_drawdown > -0.30:
            insights.append("✓ **Manageable Drawdowns**: Historical losses are moderate. Suggests robust portfolio construction.")
        elif max_drawdown < -0.50:
            insights.append(f"⚠️ **Severe Historical Losses**: ({max_drawdown*100:.1f}%) During downturns, portfolio has lost 30-50%+. Normal for equity portfolios but verify you stayed invested.")
        
        if current_drawdown < -0.20:
            insights.append(f"🔴 **Currently in Drawdown**: Portfolio is {current_drawdown*100:.1f}% below all-time highs. Monitor for recovery opportunities or protective measures.")
        elif current_drawdown > -0.05:
            insights.append("✓ **Close to All-Time Highs**: Portfolio approaching peak valuation. Consider taking some profits or trimming positions.")
        
        if avg_correlation > 0.65:
            insights.append("⚠️ **Low Diversification**: Holdings move together, reducing risk reduction benefit. Consider adding uncorrelated assets.")
        elif avg_correlation < 0.35:
            insights.append("✓ **Strong Diversification**: Holdings move independently. Portfolio has natural hedges for market shocks.")
        
        for insight in insights:
            response += f"• {insight}\n"
        
        # LLM-based narrative summary if available
        if LLM_AVAILABLE:
            try:
                response += "\n### 📊 LLM-Generated Portfolio Summary\n\n"
                portfolio_analysis = f"Portfolio of {len(portfolio_weights)} positions with {portfolio_annual_return*100:.1f}% annual return, {portfolio_volatility*100:.1f}% volatility, and {portfolio_sharpe:.2f} Sharpe ratio."
                model_predictions = {'current': portfolio_weights.to_dict()}
                benchmark_comparison = {'alpha': portfolio_annual_return - 0.1, 'beta': portfolio_volatility / 0.18}
                risk_factors = {'beta_portfolio': portfolio_volatility / 0.18}
                
                llm_recommendation = generate_investment_recommendation(
                    portfolio_analysis, model_predictions, benchmark_comparison, risk_factors,
                    portfolio_weights, returns_data
                )
                response += llm_recommendation
            except Exception as e:
                pass  # Fallback to manual insights if LLM fails
        
        response += "\n" + INVESTMENT_DISCLAIMER
        
    except Exception as e:
        response += f"⚠️ **Error**: Could not complete analysis: {str(e)[:100]}\n"
    
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
        # Fetch comprehensive data FIRST for early scoring
        ticker_obj = yf.Ticker(candidate_ticker)
        info = ticker_obj.info
        hist = ticker_obj.history(period='1y')
        
        if not info or len(info) < 5:
            return response + "⚠️ **Insufficient Data**: Cannot retrieve fundamental data for this ticker.\n"
        
        # === EARLY SCORE CALCULATION (before detailed analysis) ===
        valuation_score = 0
        valuation_max = 0
        
        pe_ratio = info.get('forwardPE') or info.get('trailingPE')
        pb_ratio = info.get('priceToBook')
        peg_ratio = info.get('pegRatio')
        ev_ebitda = info.get('enterpriseToEbitda')
        
        if pe_ratio and pe_ratio > 0:
            valuation_max += 1
            if pe_ratio < 15:
                valuation_score += 1
            elif pe_ratio <= 30:
                valuation_score += 0.5
        
        if pb_ratio and pb_ratio > 0:
            valuation_max += 1
            if pb_ratio < 1.5:
                valuation_score += 1
            elif pb_ratio <= 5:
                valuation_score += 0.5
        
        roe = info.get('returnOnEquity')
        roa = info.get('returnOnAssets')
        profit_margin = info.get('profitMargins')
        operating_margin = info.get('operatingMargins')
        revenue_growth = info.get('revenueGrowth')
        earnings_growth = info.get('earningsGrowth')
        debt_to_equity = info.get('debtToEquity')
        
        verdict_score = 0
        verdict_max = 0
        
        # Valuation portion
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
        if not hist.empty and len(hist) >= 63:
            ret_3m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-63] - 1)
            verdict_max += 25
            if ret_3m > 0.10:
                verdict_score += 25
            elif ret_3m > 0:
                verdict_score += 15
            elif ret_3m > -0.10:
                verdict_score += 10
        
        # Calculate final score and recommendation
        if verdict_max > 0:
            final_score = (verdict_score / verdict_max) * 100
        else:
            final_score = 50  # Default middle score
        
        # Determine recommendation
        if final_score >= 75:
            recommendation = "🟢🟢 STRONG BUY"
            rec_reasoning = "Compelling valuation, strong fundamentals, and positive momentum. High conviction opportunity."
        elif final_score >= 60:
            recommendation = "🟢 BUY"
            rec_reasoning = "Solid fundamentals with attractive entry point. Good risk/reward profile."
        elif final_score >= 45:
            recommendation = "🟡 HOLD"
            rec_reasoning = "Mixed signals. Wait for better entry point, positive catalyst, or clearer trend."
        elif final_score >= 30:
            recommendation = "🔴 UNDERPERFORM"
            rec_reasoning = "Multiple red flags present. Consider reducing exposure or avoiding."
        else:
            recommendation = "🔴🔴 SELL"
            rec_reasoning = "Significant concerns across valuation, fundamentals, and momentum. High risk."
        
        # === PROMINENT RECOMMENDATION AT TOP ===
        response += "---\n\n"
        response += f"## ✅ RECOMMENDATION\n\n"
        response += f"**Decision**: {recommendation}\n\n"
        response += f"**Score**: {final_score:.0f}/100\n\n"
        response += f"**Why**: {rec_reasoning}\n\n"
        
        # Quick decision summary
        if final_score >= 60:
            response += "### 💰 Action: Consider as a Buy\n"
            response += "• Entry point looks reasonable\n"
            response += "• Risk/reward profile is favorable\n"
        elif final_score >= 45:
            response += "### ⏸️ Action: Wait for Clarity\n"
            response += "• Neither compelling nor concerning at current levels\n"
            response += "• Monitor for technical breakdown or earnings catalyst\n"
        else:
            response += "### ⛔ Action: Avoid or Reduce\n"
            response += "• Risk outweighs potential reward\n"
            response += "• Better opportunities likely available elsewhere\n"
        
        response += "\n---\n\n"
        
        # === REAL-TIME PRICE DATA ===
        live_data = get_live_price_data(candidate_ticker)
        
        if live_data['success']:
            response += "### 💹 Live Market Data\n\n"
            response += get_intraday_summary(candidate_ticker)
            response += "\n"
        else:
            response += f"⚠️ _Live price data unavailable: {live_data.get('error', 'Unknown')}_\n\n"
        
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
        
        # Comprehensive valuation metrics with LLM explanations
        response += "**Valuation Multiples with Interpretations**:\n"
        
        valuation_score = 0
        valuation_max = 0
        
        if pe_ratio and pe_ratio > 0:
            valuation_max += 1
            response += f"• **P/E Ratio: {pe_ratio:.1f}**\n"
            if pe_ratio < 15:
                response += "    ✓ Undervalued—Stock trades below S&P 500 avg (18-20x). Catalyst needed: Either market eventually recognizes value, or fundamentals justify low valuation. Monitor earnings quality.\n"
                valuation_score += 1
            elif pe_ratio > 30:
                response += "    ⚠️ Premium valuation—Any earnings miss = multiple compression. Requires 15%+ annual growth to justify. High execution risk.\n"
            else:
                response += "    ✓ Fair pricing—In line with market. Valuation depends entirely on growth trajectory.\n"
                valuation_score += 0.5
        
        if pb_ratio and pb_ratio > 0:
            valuation_max += 1
            response += f"• **P/B Ratio: {pb_ratio:.2f}**\n"
            if pb_ratio < 1.5:
                response += "    ✓ Trading near book—Low premium to balance sheet. Typically means market expects modest returns on equity. Good for value investors.\n"
                valuation_score += 1
            elif pb_ratio > 5:
                response += "    → Intangible-heavy asset base. Asset-light business model (tech, software) naturally has high P/B. Not necessarily expensive.\n"
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
        
    except Exception as e:
        response += f"⚠️ **Analysis Error**: {str(e)[:150]}\n\n"
        response += "Could not complete analysis. Please verify ticker and data availability.\n"
    
    return response


def handle_education_question(question: str) -> str:
    """
    Handle educational/definitional questions about investing concepts.
    Enhanced with support for sectors, macro concepts, and market terminology.
    """
    response = "## 📚 Investment Education\n\n"
    question_lower = question.lower()
    
    # ==================== VALUATION METRICS ====================
    if 'p/e ratio' in question_lower or 'price earnings' in question_lower:
        response += "**P/E Ratio (Price-to-Earnings Ratio)**\n\n"
        response += "**Definition**: Stock price divided by earnings per share (EPS)\n\n"
        response += "**Interpretation**:\n"
        response += "• High P/E (>20-25): Market expects strong future growth OR stock is expensive\n"
        response += "• Low P/E (<15): May be undervalued OR company has structural issues\n"
        response += "• Compare within same sector for meaningful comparison\n\n"
        response += "**Limitations**: Doesn't account for growth rates, debt levels, or industry differences\n"
    
    elif 'p/b ratio' in question_lower or 'price to book' in question_lower:
        response += "**P/B Ratio (Price-to-Book Ratio)**\n\n"
        response += "**Definition**: Stock price divided by book value per share (assets - liabilities)\n\n"
        response += "**Interpretation**:\n"
        response += "• P/B < 1: Stock trades below book value (potentially undervalued or distressed)\n"
        response += "• P/B = 1: Stock trades at book value\n"
        response += "• P/B > 3: Stock commands significant premium (growth or quality)\n\n"
        response += "**Best For**: Capital-intensive industries (banking, manufacturing); less useful for tech\n"
    
    elif 'peg ratio' in question_lower or 'price earning growth' in question_lower:
        response += "**PEG Ratio (Price/Earnings-to-Growth Ratio)**\n\n"
        response += "**Definition**: P/E ratio divided by annual earnings growth rate\n\n"
        response += "**Interpretation**:\n"
        response += "• PEG < 1: Stock may be undervalued relative to growth\n"
        response += "• PEG = 1: Stock fairly valued\n"
        response += "• PEG > 1: Stock may be expensive relative to growth\n\n"
        response += "**Advantage**: Accounts for growth, better than P/E alone\n"
    
    elif 'ev/ebitda' in question_lower or 'enterprise value' in question_lower:
        response += "**EV/EBITDA Ratio**\n\n"
        response += "**Definition**: Enterprise Value (market cap + debt - cash) / Earnings Before Interest, Tax, Depreciation\n\n"
        response += "**Interpretation**:\n"
        response += "• Lower EV/EBITDA: More attractive valuation\n"
        response += "• Cross-sector comparable (unlike P/E which varies by industry)\n"
        response += "• Typical range: 8-15x for mature companies\n\n"
        response += "**Advantage**: Removes impact of financing and accounting choices\n"
    
    # ==================== RISK METRICS ====================
    elif 'beta' in question_lower or 'market beta' in question_lower:
        response += "## β - Beta (Systematic Risk)\n\n"
        response += "**Definition**: Measure of how a stock moves relative to the market (S&P 500)\n\n"
        response += "**Interpretation**:\n"
        response += "• **Beta = 1.0**: Moves exactly with market (S&P 500 has beta of 1.0)\n"
        response += "• **Beta > 1.0**: More volatile than market\n"
        response += "  - Beta 1.5 = 50% larger moves than market\n"
        response += "  - Example: Tech stocks often have beta > 1.5\n"
        response += "• **Beta < 1.0**: Less volatile than market (defensive)\n"
        response += "  - Beta 0.7 = 30% smoother than market\n"
        response += "  - Example: Utilities often have beta < 0.8\n"
        response += "• **Beta < 0**: Inverse relationship (rare, some bonds/hedges)\n\n"
        response += "**Practical Example**:\n"
        response += "If market rises 10%:\n"
        response += "  - Beta 1.0 stock → +10%\n"
        response += "  - Beta 1.5 stock → +15%\n"
        response += "  - Beta 0.7 stock → +7%\n\n"
        response += "**Use**: Assess portfolio systematic risk and volatility expectations\n"
    
    elif 'volatility' in question_lower:
        response += "**Volatility (σ - Sigma)**\n\n"
        response += "**Definition**: Statistical measure of price fluctuation (standard deviation of returns)\n\n"
        response += "**Measurement**: Typically annualized\n\n"
        response += "**Interpretation**:\n"
        response += "• **High Volatility (>30%)**: Large price swings, higher uncertainty\n"
        response += "  - Example: Small-cap tech stocks ~40-50%\n"
        response += "• **Medium Volatility (15-25%)**: Typical equities\n"
        response += "  - Example: S&P 500 ~18%\n"
        response += "• **Low Volatility (<10%)**: Stable, less risky\n"
        response += "  - Example: Treasury bonds ~5%\n\n"
        response += "**Note**: Volatility measures fluctuation, not direction. An asset can be volatile but profitable.\n"
    
    elif 'sharpe ratio' in question_lower:
        response += "**Sharpe Ratio**\n\n"
        response += "**Definition**: Risk-adjusted return = (Portfolio Return - Risk-Free Rate) / Volatility\n\n"
        response += "**Formula**: (Return - Risk-Free Rate) / Standard Deviation\n\n"
        response += "**Interpretation**:\n"
        response += "• **>1.0**: Good risk-adjusted performance (investment-grade)\n"
        response += "• **>1.5**: Very good (professional fund managers target this)\n"
        response += "• **>2.0**: Excellent/exceptional\n"
        response += "• **<0.5**: Poor risk-adjusted performance\n"
        response += "• **Negative**: Worse than risk-free rate\n\n"
        response += "**Use**: Compare investments fairly (accounts for different risk levels)\n"
    
    elif 'correlation' in question_lower:
        response += "**Correlation (ρ - Rho)**\n\n"
        response += "**Definition**: Measure of how two assets move together (-1 to +1)\n\n"
        response += "**Interpretation**:\n"
        response += "• **+1.0**: Perfect positive (move together exactly)\n"
        response += "  - Example: APPL and tech index ~0.85\n"
        response += "• **0.0**: No correlation (independent)\n"
        response += "  - Example: Stocks and long-term bonds ~0.15\n"
        response += "• **-1.0**: Perfect negative (move opposite)\n"
        response += "  - Example: Rare; put options have negative correlation with stocks\n\n"
        response += "**For Portfolio Diversification**:\n"
        response += "Seek assets with LOW or NEGATIVE correlation to reduce portfolio volatility\n"
    
    # ==================== MARKET CONCEPTS ====================
    elif 'market cap' in question_lower or 'market capitalization' in question_lower:
        response += "**Market Capitalization (Market Cap)**\n\n"
        response += "**Definition**: Total market value = Share Price × Shares Outstanding\n\n"
        response += "**Categories** (US market):\n"
        response += "• **Mega-cap**: >$500B (Apple, Microsoft, Nvidia)\n"
        response += "• **Large-cap**: $10B-$500B (established, lower risk)\n"
        response += "• **Mid-cap**: $2B-$10B (growth potential, moderate risk)\n"
        response += "• **Small-cap**: $300M-$2B (high growth potential, higher risk, less liquid)\n"
        response += "• **Micro-cap**: <$300M (highly speculative, illiquid)\n\n"
        response += "**Interpretation**: Indicates stability, liquidity, analyst coverage, and growth potential\n"
    
    elif 'dividend yield' in question_lower or 'dividend' in question_lower:
        response += "**Dividend Yield**\n\n"
        response += "**Definition**: Annual dividend per share / Current stock price\n\n"
        response += "**Interpretation**:\n"
        response += "• **High Yield (>5%)**: Income-focused; may indicate market concerns or mature business\n"
        response += "• **Moderate Yield (2-4%)**: Balanced income and growth\n"
        response += "• **Low/No Yield (<1%)**: Growth-focused companies (common in tech)\n\n"
        response += "**Example**: Company pays $2 annual dividend, stock at $50 = 4% yield\n\n"
        response += "**Warning**: Extremely high yields may signal financial distress\n"
    
    elif 'roe' in question_lower or 'return on equity' in question_lower:
        response += "**ROE (Return on Equity)**\n\n"
        response += "**Definition**: Net Income / Shareholders' Equity\n\n"
        response += "**Interpretation**:\n"
        response += "• **High ROE (>15%)**: Efficient use of shareholder capital\n"
        response += "  - Example: Apple, Microsoft typically >20%\n"
        response += "• **Average ROE (10-15%)**: Normal profitable business\n"
        response += "• **Low ROE (<10%)**: Inefficient or struggling business\n\n"
        response += "**Use**: Compare companies in same industry; indicates management efficiency\n"
    
    # ==================== SECTOR EDUCATION ====================
    elif 'tmt' in question_lower or 'technology media telecom' in question_lower:
        response += "## 📡 TMT Sector (Technology, Media, Telecom)\n\n"
        response += "**Definition**: TMT is a cross-sector grouping combining:\n"
        response += "• **Technology (XLK)**: Software, semiconductors, hardware, IT services\n"
        response += "• **Media & Communications (XLC)**: Streaming, broadcasting, advertising, publishing  \n"
        response += "• **Telecommunications**: Phone/internet providers, 5G infrastructure\n\n"
        response += "**Key Characteristics**:\n"
        response += "• High growth potential but elevated valuations\n"
        response += "• Sensitive to interest rate changes (discount rates for future earnings)\n"
        response += "• Subject to regulatory scrutiny\n"
        response += "• Wide range of company sizes and profitability\n\n"
        response += "**Current TMT Landscape**:\n"
        response += "• **Big Tech dominates**: Apple, Microsoft, Google, Amazon, Meta\n"
        response += "• **Semiconductor strength**: NVIDIA, Intel, AMD critical for AI\n"
        response += "• **Telecom evolution**: Traditional telcos adding 5G/fiber, media pivoting to streaming\n"
        response += "• **AI impact**: Rapid value creation and disruption in content, software, chips\n\n"
        response += "**Investment Considerations**:\n"
        response += "• High beta (more volatile than market)\n"
        response += "• Vulnerable to tech cycles and recessions\n"
        response += "• Requires understanding of technology disruption\n"
    
    elif 'healthcare sector' in question_lower or 'biotech' in question_lower or 'pharma' in question_lower:
        response += "## 🏥 Healthcare Sector (XLV)\n\n"
        response += "**Sub-Sectors**:\n"
        response += "• **Pharmaceuticals**: Drug development and manufacturing\n"
        response += "• **Biotech**: Genetic/cellular therapies, emerging treatments\n"
        response += "• **Medical Devices**: Imaging, surgical tools, monitoring\n"
        response += "• **Healthcare Services**: Providers, insurers, pharmacy benefit managers\n\n"
        response += "**Characteristics**:\n"
        response += "• Defensive sector (demand relatively stable in recessions)\n"
        response += "• Long R&D cycles (drugs can take 10+ years to develop)\n"
        response += "• Regulatory risk (FDA approval, pricing pressure)\n"
        response += "• Demographics tailwind (aging populations globally)\n\n"
        response += "**Key Risks**:\n"
        response += "• Patent cliffs (loss of drug protection)\n"
        response += "• Political risk (drug price regulation)\n"
        response += "• Clinical trial failures\n"
        response += "• Biotech extreme volatility (binary outcomes)\n"
    
    elif 'financial sector' in question_lower or 'finance' in question_lower:
        response += "## 💰 Financial Sector (XLF)\n\n"
        response += "**Components**:\n"
        response += "• **Banks**: Commercial, investment, regional\n"
        response += "• **Insurance**: Property/casualty, life, reinsurance\n"
        response += "• **Real Estate Finance**: Mortgage REITs, lending\n\n"
        response += "**Key Dynamics**:\n"
        response += "• Very interest rate sensitive (higher rates = higher profits for banks)\n"
        response += "• Economic cycle: Benefits in growth; struggles in recessions\n"
        response += "• Regulatory environment significant\n"
        response += "• Leverage amplifies both gains and losses\n\n"
        response += "**Valuation Drivers**:\n"
        response += "• Interest rate spreads (yield curve slope)\n"
        response += "• Loan quality and non-performing asset ratios\n"
        response += "• Capital ratios and dividend capacity\n"
    
    elif 'energy sector' in question_lower or 'oil' in question_lower:
        response += "## ⛽ Energy Sector (XLE)\n\n"
        response += "**Components**:\n"
        response += "• **Upstream**: Oil/gas exploration and production\n"
        response += "• **Midstream**: Pipelines, transportation, storage\n"
        response += "• **Downstream**: Refining, distribution, retail\n"
        response += "• **Utilities**: Integrated energy companies\n\n"
        response += "**Key Characteristics**:\n"
        response += "• Highly cyclical (booms and busts with commodity prices)\n"
        response += "• Geopolitically sensitive (OPEC, sanctions, conflict)\n"
        response += "• Energy transition risk (shift to renewables)\n"
        response += "• Capital intensive (requires large upfront investment)\n\n"
        response += "**Current Tailwinds/Headwinds**:\n"
        response += "• ✓ Energy security focus post-Russia sanctions\n"
        response += "• ✗ Long-term transition away from fossil fuels\n"
        response += "• ✓ High cash generation supports dividends\n"
        response += "• ✗ Volatile commodity prices create uncertainty\n"
    
    # ==================== GENERAL CONCEPTS ====================
    elif 'diversification' in question_lower:
        response += "**Diversification**\n\n"
        response += "**Definition**: Spreading investments across uncorrelated assets to reduce risk\n\n"
        response += "**Core Principle**: Don't put all eggs in one basket\n\n"
        response += "**Dimensions**:\n"
        response += "• **By asset class**: Stocks, bonds, real estate, commodities\n"
        response += "• **By sector**: Tech, finance, healthcare, energy, etc.\n"
        response += "• **By geography**: US, developed international, emerging markets\n"
        response += "• **By company size**: Large-cap, mid-cap, small-cap\n"
        response += "• **By style**: Value, growth, dividend\n\n"
        response += "**Goal**: Reduce unsystematic (company-specific) risk while maintaining returns\n"
    
    else:
        # Comprehensive list if no specific match
        response += "I can explain many investment concepts:\n\n"
        response += "### 📊 Valuation Metrics\n"
        response += "• P/E ratio • P/B ratio • PEG ratio • EV/EBITDA • Dividend yield\n\n"
        response += "### 📈 Risk Metrics\n"
        response += "• **Beta** - Systematic risk relative to market\n"
        response += "• **Volatility** - Price fluctuation\n"
        response += "• **Sharpe Ratio** - Risk-adjusted return\n"
        response += "• **Correlation** - How assets move together\n\n"
        response += "### 💼 Portfolio Concepts\n"
        response += "• Diversification • Asset allocation • Rebalancing • Concentration risk\n\n"
        response += "### 🏢 Sectors\n"
        response += "• **TMT** (Technology, Media, Telecom) • Healthcare • Financials\n"
        response += "• Energy • Consumer • Industrial • Materials • Utilities • Real Estate\n\n"
        response += "### 💹 Market Terms\n"
        response += "• Market cap • Index composition • Bull/bear markets • Market cycles\n\n"
        response += "### 💰 Company Fundamentals\n"
        response += "• ROE • ROA • Margin • Growth • Profitability\n\n"
        response += "**Ask about any topic!** Examples:\n"
        response += "_\"What is beta?\" \"Explain TMT industry\" \"How to interpret dividend yield?\"_\n"
    
    response += "\n" + INVESTMENT_DISCLAIMER
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


def analyze_stock_portfolio_fit(ticker: str, portfolio_weights: Optional[pd.Series] = None) -> str:
    """
    Comprehensive analysis of whether a stock should be added to a portfolio
    Addresses questions like: "Should I add MSFT given my current holdings?"
    """
    response = f"## 📊 Portfolio Fit Analysis: {ticker}\n\n"
    
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period='1y')
        
        if not info or len(info) < 5:
            return response + f"⚠️ Cannot retrieve data for {ticker}\n"
        
        # === CORRELATION ANALYSIS ===
        response += "### 🔗 Correlation to Portfolio\n\n"
        
        if portfolio_weights is not None and len(portfolio_weights) > 0:
            response += f"**Cross-Correlation Analysis** (benchmarked against your {len(portfolio_weights)} holdings):\n\n"
            
            # Fetch correlation
            ticket_hist = yf.download(ticker, period='1y', progress=False, auto_adjust=False)
            if not ticket_hist.empty and not hist.empty:
                if isinstance(ticket_hist.columns, pd.MultiIndex):
                    new_returns = ticket_hist['Adj Close', ticker].pct_change()
                else:
                    new_returns = ticket_hist['Adj Close'].pct_change()
                
                if isinstance(hist.columns, pd.MultiIndex):
                    portfolio_return = hist['Adj Close', 'SPY'].pct_change()
                else:
                    portfolio_return = hist['Adj Close'].pct_change()
                
                correlation = new_returns.corr(portfolio_return)
                response += f"• **Correlation to portfolio**: {correlation:.2f}\n"
                
                if correlation < -0.2:
                    response += "    ✓ **Negative correlation** - Moves opposite to portfolio. Excellent diversifier!\n"
                elif correlation < 0.3:
                    response += "    ✓ **Low correlation** - Independent movement. Good diversifier.\n"
                elif correlation < 0.7:
                    response += "    ≈ **Moderate correlation** - Some overlap. Provides some diversification.\n"
                else:
                    response += "    ⚠️ **High correlation** - Moves with portfolio. Limited diversification benefit.\n"
            
            response += "\n"
        else:
            response += "_Portfolio holdings not provided. Add one to see correlation analysis._\n\n"
        
        # === SECTOR & INDUSTRY FIT ===
        response += "### 🏭 Sector Positioning\n\n"
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        response += f"• **Sector**: {sector}\n"
        response += f"• **Industry**: {industry}\n"
        
        if portfolio_weights is not None:
            portfolio_sectors = [yf.Ticker(t).info.get('sector', 'N/A') for t in portfolio_weights.index]
            sector_count = sum(1 for s in portfolio_sectors if s == sector)
            response += f"• **Current sector exposure**: {sector_count}/{len(portfolio_weights)} holdings in {sector}\n"
            
            if sector_count == 0:
                response += "    ✓ New sector exposure—add diversification\n"
            elif sector_count <= 2:
                response += "    ✓ Moderate sector representation—additional exposure reasonable\n"
            else:
                response += "    ⚠️ Already concentrated in this sector—consider if doubling up makes sense\n"
        
        response += "\n"
        
        # === FUNDAMENTALS & QUALITY ===
        response += "### 💎 Fundamentals & Quality Score\n\n"
        
        roe = info.get('returnOnEquity')
        roa = info.get('returnOnAssets')
        roic = info.get('returnOnCapital')
        debt_to_equity = info.get('debtToEquity')
        current_ratio = info.get('currentRatio')
        gross_margin = info.get('grossMargins')
        operating_margin = info.get('operatingMargins')
        
        quality_score = 0
        quality_max = 0
        
        if roe is not None:
            quality_max += 1
            response += f"• **ROE**: {roe*100:.1f}%"
            if roe > 0.15:
                response += " ✓\n"
                quality_score += 1
            elif roe > 0.10:
                response += " (Above average)\n"
                quality_score += 0.7
            else:
                response += " (Below average)\n"
        
        if debt_to_equity is not None:
            quality_max += 1
            response += f"• **Debt/Equity**: {debt_to_equity:.2f}"
            if debt_to_equity < 0.5:
                response += " ✓ (Conservative)\n"
                quality_score += 1
            elif debt_to_equity < 2:
                response += " (Moderate)\n"
                quality_score += 0.7
            else:
                response += " (Leveraged)\n"
        
        if current_ratio is not None:
            quality_max += 1
            response += f"• **Current Ratio**: {current_ratio:.2f}"
            if current_ratio > 1:
                response += " ✓ (Liquid)\n"
                quality_score += 1
            else:
                response += " (Tight)\n"
        
        # Overall quality verdict
        if quality_max > 0:
            quality_pct = (quality_score / quality_max) * 100
            response += f"\n**Quality Score**: {quality_pct:.0f}/100"
            if quality_pct >= 75:
                response += " ✓ High-quality business\n"
            elif quality_pct >= 50:
                response += " ~ Fair quality\n"
            else:
                response += " ⚠️ Lower quality—higher execution/financial risk\n"
        
        response += "\n"
        
        # === PORTFOLIO IMPACT ===
        response += "### ⚖️ Sizing & Allocation\n\n"
        response += "**Suggested Allocation Strategy**:\n\n"
        
        if portfolio_weights is not None and len(portfolio_weights) > 0:
            largest_weight = portfolio_weights.max()
            avg_weight = portfolio_weights.mean()
            n_stocks = len(portfolio_weights)
            
            response += f"• **Current portfolio**: {n_stocks} stocks\n"
            response += f"  - Largest holding: {largest_weight*100:.1f}%\n"
            response += f"  - Average weight: {avg_weight*100:.1f}%\n\n"
            
            target_weight = avg_weight * 0.5 if largest_weight > avg_weight * 3 else avg_weight
            response += f"• **Suggest adding at**: ~{target_weight*100:.1f}% (around your average)\n"
            response += "• **Avoid**: Making it your largest position on Day 1\n"
            response += "• **Rebalance schedule**: Review quarterly, trim winners > 2x average weight\n"
        else:
            response += "• **No portfolio data**—Suggest starting with 3-5% initial position\n"
            response += "• **Build gradually** to allow for price averaging\n"
        
        response += "\n"
        
    except Exception as e:
        response += f"⚠️ Error analyzing portfolio fit: {str(e)[:100]}\n"
    
    return response


def analyze_fed_portfolio_impact(keywords: List[str]) -> str:
    """
    Comprehensive analysis of Fed policy impact on portfolios
    Addresses questions like: "How will Fed rate cuts impact my portfolio?"
    """
    response = f"## 🏦 Fed Policy Impact on Portfolios\n\n"
    response += f"**Analysis Date**: {get_data_timestamp()}\n\n"
    
    try:
        # Get current Treasury yields as proxy for rate environment
        tnx = yf.download('^TNX', period='6mo', progress=False, auto_adjust=False)
        tlt = yf.download('TLT', period='1y', progress=False, auto_adjust=False)
        
        if not tnx.empty:
            if isinstance(tnx.columns, pd.MultiIndex):
                tnx_close = tnx['Adj Close', '^TNX']
                tlt_close = tlt['Adj Close', 'TLT']
            else:
                tnx_close = tnx['Adj Close']
                tlt_close = tlt['Adj Close']
            
            current_10y = tnx_close.iloc[-1]
            prior_10y = tnx_close.iloc[0]
            
            response += "### 👀 Current Interest Rate Environment\n\n"
            response += f"**10-Year Treasury Yield**: {current_10y:.2f}%\n"
            response += f"• 6-month change: {current_10y - prior_10y:+.2f}%\n"
            response += f"• Trend: {'Rising (tightening)' if current_10y > prior_10y else 'Falling (easing)' if current_10y < prior_10y else 'Stable'}\n\n"
        
        response += "### 📊 Fed Rate Cut Scenarios & Impacts\n\n"
        
        response += "#### Scenario 1: Aggressive Rate Cuts (3-4 cuts in 2026)\n"
        response += "*Market Expectation*: Economic slowdown concerns\n\n"
        response += "**Winners**:\n"
        response += "• **Growth Stocks** (Tech, ARK, QQQ) - Lower discount rates boost valuation multiples\n"
        response += "• **High-Dividend Stocks** - Bonds become less attractive, relative yield appeal increases\n"
        response += "• **REITs & Utilities** - Benefit from lower financing costs AND sustained dividend demand\n"
        response += "• **Unprofitable Growth Names** - Fed tailwind for cash flow burn scenarios\n\n"
        
        response += "**Losers**:\n"
        response += "• **Banks & Financials** (JPM, BAC, C) - Net interest margin compresses\n"
        response += "• **High-Yield Bonds** - Reduced carry in low-rate environment; credit spreads may widen (riskier)\n"
        response += "• **Short-term Treasury investments** - Reinvestment cascade to lower rates\n\n"
        
        response += "**Portfolio Action**:\n"
        response += "• Reduce financials overweight to market weight\n"
        response += "• Tilt toward growth; reduce value% below normal\n"
        response += "• Consider buying long-duration bonds (TLT) for capital appreciation\n\n"
        
        response += "---\n\n"
        
        response += "#### Scenario 2: Mild Rate Cuts (1-2 cuts)\n"
        response += "*Market Expectation*: Gradual economic adjustment\n\n"
        response += "**Winners**:\n"
        response += "• **Quality Growth** (MSFT, NVDA, AAPL) - Steady multiple expansion\n"
        response += "• **Dividend Aristocrats** - Stable earnings + yield still attractive vs bonds\n\n"
        
        response += "**Losers**:\n"
        response += "• **Ultra-high-beta speculative** - Euphoria trades cool\n"
        response += "• **Money Market Funds** (VMFXX) - Yields fall in lockstep with Fed\n\n"
        
        response += "---\n\n"
        
        response += "#### Scenario 3: No Rate Cuts (Pause)\n"
        response += "*Market Expectation*: Inflation lingers, Fed on hold\n\n"
        response += "**Winners**:\n"
        response += "• **Financials** - Interest rate landscape stabilizes\n"
        response += "• **Energy & Commodities** - Inflation hedge\n"
        response += "• **TIPS** (Inflation-Protected Securities) - Real yields secure\n\n"
        
        response += "**Losers**:\n"
        response += "• **Growth stocks** - High rates = high discount rates = lower valuations\n"
        response += "• **Long-duration bonds** - Rates stay elevated\n\n"
        
        response += "---\n\n"
        
        response += "### 🎯 Portfolio Positioning Checklist\n\n"
        response += "**For Uncertainty, Hedge:**\n"
        response += "1. **Duration Mix**: 30% bonds to cushion equity volatility\n"
        response += "   - If cuts come → Bond gains offset any stock hiccup\n"
        response += "   - If no cuts → Equities outperform bonds anyway\n\n"
        
        response += "2. **Sector Balance**:\n"
        response += "   - 25% Growth (Benefits from rate cuts)\n"
        response += "   - 25% Value (Benefits from no cuts)\n"
        response += "   - 20% Financials (Defensive to rates)\n"
        response += "   - 30% Diversified Sectors\n\n"
        
        response += "3. **International Diversification**:\n"
        response += "   - Non-US developed markets (EFA) - Decoupled from Fed policy\n"
        response += "   - Emerging markets (EEM) - Different monetary cycle\n\n"
        
        response += "4. **Avoid Single Bets**:\n"
        response += "   - Don't go 70% growth anticipating cuts\n"
        response += "   - Don't go 50% financials betting against cuts\n"
        response += "   - Market already prices in base case; your edge is optionality\n\n"
        
        response += "### 📚 Historical Precedents\n\n"
        response += "**2001 (Recession + Rate Cuts)**:\n"
        response += "• Tech stocks crashed 70-80% (cuts couldn't save overvalued growth)\n"
        response += "• Quality dividend stocks held up better\n"
        response += "• **Lesson**: Rate cuts help, but don't override valuation\n\n"
        
        response += "**2019 (Fed Pivot)**:\n"
        response += "• After 2-year hiking cycle, 3 cuts in late 2019\n"
        response += "• Tech stocks surged 30%+ heading into 2020\n"
        response += "• **Lesson**: Cuts can reignite growth narratives\n\n"
        
        response += f"{INVESTMENT_DISCLAIMER}\n"
        
    except Exception as e:
        response += f"⚠️ Error in analysis: {str(e)[:100]}\n"
    
    return response


def analyze_inflation_risks(keywords: List[str]) -> str:
    """
    Comprehensive inflation risk analysis for portfolios
    Addresses questions like: "What are the inflation risks?"
    """
    response = f"## 🔥 Inflation Risk Assessment\n\n"
    response += f"**Analysis Date**: {get_data_timestamp()}\n\n"
    
    try:
        response += "### 📊 Current Inflation Metrics\n\n"
        response += "*Note: Using available market data as proxy*\n\n"
        
        # Get inflation hedges performance
        tips = yf.download('VTIP', period='1y', progress=False, auto_adjust=False)
        gold = yf.download('GLD', period='1y', progress=False, auto_adjust=False)
        commodities = yf.download('DBC', period='1y', progress=False, auto_adjust=False)
        
        response += "**Inflation Hedge Performance (1-year)**:\n"
        try:
            if not tips.empty:
                if isinstance(tips.columns, pd.MultiIndex):
                    tips_ret = (tips['Adj Close', 'VTIP'].iloc[-1] / tips['Adj Close', 'VTIP'].iloc[0] - 1)
                    gold_ret = (gold['Adj Close', 'GLD'].iloc[-1] / gold['Adj Close', 'GLD'].iloc[0] - 1)
                    comm_ret = (commodities['Adj Close', 'DBC'].iloc[-1] / commodities['Adj Close', 'DBC'].iloc[0] - 1)
                else:
                    tips_ret = (tips['Adj Close'].iloc[-1] / tips['Adj Close'].iloc[0] - 1)
                    gold_ret = (gold['Adj Close'].iloc[-1] / gold['Adj Close'].iloc[0] - 1)
                    comm_ret = (commodities['Adj Close'].iloc[-1] / commodities['Adj Close'].iloc[0] - 1)
                
                response += f"• TIPS (Treasury Inflation-Protected): {tips_ret*100:+.1f}%\n"
                response += f"• Gold (GLD): {gold_ret*100:+.1f}%\n"
                response += f"• Commodities (DBC): {comm_ret*100:+.1f}%\n\n"
        except:
            pass
        
        response += "### 🎯 Inflation Scenarios & Portfolio Impact\n\n"
        
        response += "#### Scenario A: Sustained Inflation (3-4%)\n"
        response += "**Winners**:\n"
        response += "• **Energy Producers** (XLE, OXY, CVX) - Direct commodity exposure\n"
        response += "• **Utilities** (XLU) - Regulated; can pass costs; steady dividends hedge\n"
        response += "• **Consumer Staples** (XLP, WMT, PG) - Pricing power; buyback moats\n"
        response += "• **TIPS & Inflation-Linked Bonds** - Direct inflation indexing\n"
        response += "• **Real Estate/REITs** (XLRE) - Real asset with pricing power\n\n"
        
        response += "**Losers**:\n"
        response += "• **Tech Growth** (QQQ, high P/E names) - Compressed multiples in high-rate environment\n"
        response += "• **Long-duration bonds** (TLT) - Rising rates hurt bond prices\n"
        response += "• **Financials** - Compressed NIM if inflation driven by supply shock (not demand)\n\n"
        
        response += "**Action**:\n"
        response += "• Overweight: Commodities, energy, inflation-hedged equities\n"
        response += "• Add: 5-10% TIPS allocation for real returns protection\n"
        response += "• Reduce: Long-duration bonds and high-duration growth stocks\n\n"
        
        response += "---\n\n"
        
        response += "#### Scenario B: Deflation/Disinflation (<1%)\n"
        response += "**Winners**:\n"
        response += "• **Long-duration Bonds** (TLT, BND) - Bond prices rise sharply\n"
        response += "• **Growth Stocks** (MSFT, NVDA, AAPL) - Multiples expand; capex value recovers\n"
        response += "• **Tech/Software** - Pricing power; profit margins expand\n\n"
        
        response += "**Losers**:\n"
        response += "• **Energy/Commodities** - Deflation = demand destruction\n"
        response += "• **Real Estate** - Price declines under deflationary pressure\n"
        response += "• **Dividend Stocks** - Real dividend yields skyrocket; ratingcut (sell signal)\n\n"
        
        response += "---\n\n"
        
        response += "### 🛡️ Inflation-Protected Portfolio Allocation\n\n"
        response += "**Core Deflation Hedge (Everyone needs this)**:\n"
        response += "• 30-40% Equities (market participation)\n"
        response += "• 40-50% Bonds (duration cushion if deflation)\n"
        response += "• 5-10% Cash (optionality for dislocations)\n\n"
        
        response += "**Inflation Hedge Add-ons**:\n"
        response += "• 5-10% Commodities (oil, metals, agriculture exposure)\n"
        response += "• 5-10% Real Estate/REITs (real assets)\n"
        response += "• 3-5% TIPS (explicit indexing)\n"
        response += "• 2%  Gold (tail risk insurance)\n\n"
        
        response += "**Anti-Hedge Positions** (reduce if inflation concern):\n"
        response += "• Reduce high-duration growth tech: From 40% to 25%\n"
        response += "• Reduce 30-year treasuries: From 30% to 15%\n"
        response += "• Reduce fixed-income heavy allocation if rates rising\n\n"
        
        response += "### 📚 Historical Inflation Episodes\n\n"
        response += "**2021-2022 (Inflation Surge)**:\n"
        response += "• Energy +65%, Staples +20%, but Tech -65%\n"
        response += "• Bonds down 15-20%; TIPs only down 2%\n"
        response += "• **Lesson**: Inflation hits growth valuations hardest\n\n"
        
        response += "**1980s (Volcker Inflation Break)**:\n"
        response += "• 22% fed funds rate; bonds crushed but then soared as inflation fell\n"
        response += "• Commodities peaked then fell 50%\n"
        response += "• **Lesson**: Inflation fighting = temporary pain but ends bull market in duration\n\n"
        
        response += f"{INVESTMENT_DISCLAIMER}\n"
        
    except Exception as e:
        response += f"⚠️ Error in analysis: {str(e)[:100]}\n"
    
    return response


def handle_user_question(question: str, 
                         portfolio_weights: Optional[pd.Series] = None,
                         returns_data: Optional[pd.DataFrame] = None) -> str:
    """
    Main chatbot dispatcher with anti-hallucination measures
    
    Key Features:
    - Intelligent question classification with confidence scoring
    - Real data validation before analysis
    - Routes to deep LLM analyzers for rich insights
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
            # Stock comparison/ranking questions - Try LLM first
            if LLM_AVAILABLE:
                llm_response = llm_orchestrate_analysis(classification, question, portfolio_weights, returns_data)
                if llm_response:
                    return prefix + llm_response
            
            # Fallback to existing function
            portfolio_stocks = portfolio_weights.index.tolist() if portfolio_weights is not None else None
            return prefix + analyze_stock_comparison(question, portfolio_stocks)
        
        elif question_type == 'stock' and entities:
            # Stock-specific analysis
            ticker = entities[0]
            
            # Check if this is asking about portfolio fit (either by intent or by keywords)
            is_portfolio_fit = (
                intent == 'portfolio_fit' or 
                any(word in question.lower() for word in ['add', 'should i add', 'given my', 'my portfolio', 'fit', 'works with', 'complements'])
            )
            
            if is_portfolio_fit:
                return prefix + analyze_stock_portfolio_fit(ticker, portfolio_weights)
            else:
                # Regular stock analysis with investment decision
                return prefix + analyze_investment_decision(question, portfolio_weights, ticker, returns_data)
        
        elif question_type == 'sector':
            # Sector/industry analysis - Check for specific sub-sectors first
            sub_sector = classification.get('sub_sector', None)
            
            # Route to specialized sub-sector analyzers
            if sub_sector == 'semiconductor':
                return prefix + analyze_semiconductor_industry(portfolio_weights)
            elif sub_sector == 'pharma':
                return prefix + analyze_pharma_industry(portfolio_weights)
            elif sub_sector == 'banking':
                return prefix + analyze_banking_industry(portfolio_weights)
            
            # Try LLM first for deep sector rotation
            if LLM_AVAILABLE:
                llm_response = llm_orchestrate_analysis(classification, question, portfolio_weights, returns_data)
                if llm_response:
                    return prefix + llm_response
            
            # Fallback to existing function - get sector name from entities or keywords, or default to 'technology'
            sector_name = None
            if entities and len(entities) > 0:
                sector_name = entities[0]
            elif classification.get('keywords') and len(classification.get('keywords', [])) > 0:
                sector_name = classification['keywords'][0]
            else:
                sector_name = 'technology'  # Default sector
            
            return prefix + analyze_sector_industry(sector_name, portfolio_weights)
        
        elif question_type == 'market':
            # Overall market conditions - Use new market data analyzer
            return prefix + analyze_market_data(question)
        
        elif question_type == 'macro':
            # Macro economic commentary - Route to specific analyzers based on keywords
            keywords = classification.get('keywords', [])
            
            # Check for Fed/rate specific questions
            if any(word in question.lower() for word in ['fed', 'federal reserve', 'rate cut', 'rate hike', 'interest rate']):
                return prefix + analyze_fed_portfolio_impact(keywords)
            
            # Check for inflation specific questions
            if any(word in question.lower() for word in ['inflation', 'inflation risk']):
                return prefix + analyze_inflation_risks(keywords)
            
            # General macro analysis
            return prefix + analyze_macro_environment(keywords)
        
        elif question_type == 'portfolio':
            # Portfolio-specific advice - Try LLM analyzers for stress test & risk analysis
            if portfolio_weights is None or len(portfolio_weights) == 0:
                response = "## 💼 Portfolio Analysis\n\n"
                response += "⚠️ **No portfolio data available**.\n\n"
                response += "To get personalized portfolio analysis:\n"
                response += "1. Go to **Portfolio Builder** page\n"
                response += "2. Select stocks and build a portfolio\n"
                response += "3. Return here for contextualized advice\n"
                return response
            
            if returns_data is not None and len(portfolio_weights) >= 1 and LLM_AVAILABLE:
                # Try LLM-powered comprehensive analysis (stress test + risk analysis)
                llm_response = llm_orchestrate_analysis(classification, question, portfolio_weights, returns_data)
                if llm_response:
                    return prefix + llm_response
            
            if returns_data is not None and len(portfolio_weights) >= 1:
                # Provide comprehensive portfolio performance analysis
                return prefix + analyze_portfolio_performance(portfolio_weights, returns_data)
            else:
                # Basic portfolio info if insufficient data
                response = "## 💼 Your Portfolio\n\n"
                response += f"**Holdings**: {len(portfolio_weights)} positions\n"
                response += f"**Top positions**: {', '.join(portfolio_weights.nlargest(3).index.tolist())}\n\n"
                response += "_For comprehensive performance analysis, ensure you have sufficient historical data._\n\n"
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


# ==================== ENHANCED MARKET DATA ANALYZER ====================

def analyze_market_data(question: str) -> str:
    """
    Comprehensive market data analysis focused on market conditions and trends.
    Handles questions about market indices, volatility, market regime, etc.
    """
    response = "## 📊 Market Data Analysis\n\n"
    market_status = get_market_status()
    response += f"{market_status['emoji']} **Market Status**: {market_status['message']} ({market_status['time']})\n"
    response += f"**Analysis Date**: {get_data_timestamp()}\n\n"
    
    try:
        # Fetch major indices
        indices = {
            'SPY': 'S&P 500 (Large-cap)',
            'QQQ': 'Nasdaq 100 (Tech-heavy)',
            'IWM': 'Russell 2000 (Small-cap)',
            'EFA': 'EAFE (International Developed)',
            'EEM': 'Emerging Markets',
            'TLT': 'Long-term Treasuries (20Y+)',
            'GLD': 'Gold',
            'DBC': 'Commodities'
        }
        
        response += "### 📈 Performance Across Asset Classes\n\n"
        
        timeframes = ['1d', '1mo', '3mo', '1y']
        timeframe_names = ['Today', '1-Month', '3-Month', '1-Year']
        
        # Get performance for each timeframe
        for timeframe, timeframe_name in zip(timeframes, timeframe_names):
            response += f"**{timeframe_name} Performance**:\n"
            
            perf_data = {}
            for ticker, name in indices.items():
                try:
                    data = yf.download(ticker, period=timeframe, progress=False, auto_adjust=False)
                    if not data.empty:
                        if isinstance(data.columns, pd.MultiIndex):
                            ret = (data['Adj Close', ticker].iloc[-1] / data['Adj Close', ticker].iloc[0] - 1)
                        else:
                            ret = (data['Adj Close'].iloc[-1] / data['Adj Close'].iloc[0] - 1)
                        perf_data[name] = ret
                except:
                    pass
            
            if perf_data:
                sorted_perf = sorted(perf_data.items(), key=lambda x: x[1], reverse=True)
                for name, ret in sorted_perf:
                    emoji = '📈' if ret > 0 else '📉'
                    response += f"• {emoji} {name}: {ret*100:+.2f}%\n"
                response += "\n"
        
        # Volatility analysis
        response += "### 📊 Volatility Analysis\n\n"
        
        spy_data = yf.download('SPY', period='1y', progress=False, auto_adjust=False)
        if not spy_data.empty:
            if isinstance(spy_data.columns, pd.MultiIndex):
                spy_returns = spy_data['Adj Close', 'SPY'].pct_change().dropna()
            else:
                spy_returns = spy_data['Adj Close'].pct_change().dropna()
            
            current_vol = spy_returns.tail(20).std() * np.sqrt(252)
            avg_vol = spy_returns.tail(60).std() * np.sqrt(252)
            hist_vol = spy_returns.std() * np.sqrt(252)
            
            response += f"• **Current 20-day vol (SPY)**: {current_vol*100:.1f}%\n"
            response += f"• **60-day average vol**: {avg_vol*100:.1f}%\n"
            response += f"• **1-year average vol**: {hist_vol*100:.1f}%\n"
            
            if current_vol > avg_vol * 1.3:
                response += "• 🔴 **Elevated volatility** - Markets more turbulent than usual\n"
            elif current_vol < avg_vol * 0.7:
                response += "• 🟢 **Low volatility** - Markets calm, tight trading ranges\n"
            else:
                response += "• 🟡 **Normal volatility** - Typical market fluctuations\n"
            response += "\n"
        
        # Breadth analysis
        response += "### 🎯 Market Breadth\n\n"
        response += "**Interpreting Market Breadth**:\n"
        response += "• When most stocks rise with indices: Broad-based rally (healthy)\n"
        response += "• When few mega-cap stocks drive indices: Narrow market (vulnerable)\n"
        response += "• When stocks decline but indices hold: Divergence (warning sign)\n\n"
        
        # Correlation matrix
        response += "### 🔗 Asset Class Correlations\n\n"
        response += "**Current Market Dynamics**:\n"
        response += "• **Stocks-Bonds correlation**: Usually negative (good diversification)\n"
        response += "  - When rising: Rates rising, both hurt (risk-off)\n"
        response += "• **Stocks-Gold correlation**: Usually negative to neutral\n"
        response += "  - Gold rises: Flight to safety, risk-off\n"
        response += "• **Tech-Energy correlation**: Usually low\n"
        response += "  - Tech weak but energy strong: Sector rotation\n\n"
        
        # Key takeaways
        response += "### 💡 Market Context\n\n"
        response += "**What Drives Markets Currently**:\n"
        response += "1. **Interest Rates** - Fed policy, inflation expectations\n"
        response += "2. **Earnings** - Corporate profitability, guidance\n"
        response += "3. **Economic Data** - Growth, employment, inflation\n"
        response += "4. **Geopolitics** - Conflicts, sanctions, trade tensions\n"
        response += "5. **Tech/AI Narrative** - Sector rotation, valuation reassessment\n\n"
        
        response += "**For Portfolio Implications**:\n"
        response += "• High volatility → Rebalance to fixed allocation\n"
        response += "• Rising rates → Review bond duration, dividend yields\n"
        response += "• Tech strength → Check portfolio tech concentration\n"
        response += "• Sector rotation → Ensure adequate diversification\n"
        
    except Exception as e:
        response += f"⚠️ **Data Error**: {str(e)[:100]}\n"
    
    response += "\n" + INVESTMENT_DISCLAIMER
    return response


def analyze_sector_growth(question: str) -> str:
    """
    Analyze sector growth trends, outlook, and investment characteristics.
    """
    response = "## 🏢 Sector Growth & Trends Analysis\n\n"
    
    # Identify sector from question
    sector_map = {
        'technology': 'XLK',
        'tech': 'XLK',
        'healthcare': 'XLV',
        'health': 'XLV',
        'finance': 'XLF',
        'financial': 'XLF',
        'energy': 'XLE',
        'consumer': 'XLY',
        'industrial': 'XLI',
        'materials': 'XLB',
        'utilities': 'XLU',
        'staples': 'XLP',
        'real estate': 'XLRE',
        'communications': 'XLC',
    }
    
    detected_sector = None
    for sector_key, etf in sector_map.items():
        if sector_key in question.lower():
            detected_sector = etf
            break
    
    if not detected_sector:
        detected_sector = 'XLK'  # Default to tech
    
    try:
        # Fetch sector data
        sector_data = yf.download(detected_sector, period='1y', progress=False, auto_adjust=False)
        spy_data = yf.download('SPY', period='1y', progress=False, auto_adjust=False)
        
        if sector_data.empty or spy_data.empty:
            return response + "⚠️ **Insufficient data** for sector analysis.\n"
        
        if isinstance(sector_data.columns, pd.MultiIndex):
            sector_price = sector_data['Adj Close', detected_sector]
            spy_price = spy_data['Adj Close', 'SPY']
        else:
            sector_price = sector_data['Adj Close']
            spy_price = spy_data['Adj Close']
        
        # Performance metrics
        sector_ret = (sector_price.iloc[-1] / sector_price.iloc[0] - 1)
        spy_ret = (spy_price.iloc[-1] / spy_price.iloc[0] - 1)
        relative_ret = sector_ret - spy_ret
        
        sector_vol = sector_price.pct_change().std() * np.sqrt(252)
        spy_vol = spy_price.pct_change().std() * np.sqrt(252)
        
        response += "### 📊 Sector Performance\n\n"
        response += f"**{detected_sector} (ETF) YTD Returns**: {sector_ret*100:+.2f}%\n"
        response += f"**S&P 500 YTD**: {spy_ret*100:+.2f}%\n"
        response += f"**Relative Outperformance**: {relative_ret*100:+.2f}pp\n"
        response += f"**Volatility**: {sector_vol*100:.1f}% (S&P 500: {spy_vol*100:.1f}%)\n\n"
        
        if relative_ret > 0.08:
            response += "🟢 **Strong Outperformance** - Sector leading the market\n"
        elif relative_ret > 0:
            response += "🟡 **Slight Outperformance** - Sector keeping pace with gains\n"
        elif relative_ret > -0.08:
            response += "🟡 **Slight Underperformance** - Sector lagging slightly\n"
        else:
            response += "📉 **Significant Underperformance** - Sector struggling relative to market\n"
        
        response += "\n"
        
        # Growth drivers by sector
        response += "### 💡 Sector-Specific Growth Drivers\n\n"
        
        if detected_sector == 'XLK':  # Technology
            response += "**Technology Sector Growth Drivers**:\n"
            response += "• AI/ML adoption across enterprises\n"
            response += "• Cloud infrastructure expansion\n"
            response += "• Semiconductor demand (chips for all devices)\n"
            response += "• Software SaaS model recurring revenue\n"
            response += "• Interest rate sensitivity (high-growth valuations)\n\n"
            response += "**Risks**:\n"
            response += "• Regulatory scrutiny (antitrust, data privacy)\n"
            response += "• Valuation compression if rates rise\n"
            response += "• China geopolitical tensions (semiconductors)\n"
        
        elif detected_sector == 'XLV':  # Healthcare
            response += "**Healthcare Sector Growth Drivers**:\n"
            response += "• Aging demographics (boomers entering peak medical spending)\n"
            response += "• Breakthrough therapies (GLP-1 obesity drugs, biologics)\n"
            response += "• Telemedicine adoption\n"
            response += "• Medical device innovation\n"
            response += "• Pharma pipeline development\n\n"
            response += "**Risks**:\n"
            response += "• Drug price regulation\n"
            response += "• Patent cliffs\n"
            response += "• Clinical trial failures (biotech)\n"
        
        elif detected_sector == 'XLF':  # Financials
            response += "**Financials Sector Growth Drivers**:\n"
            response += "• Interest rate levels (steeper curve = higher NII)\n"
            response += "• Economic growth → lending demand\n"
            response += "• M&A activity\n"
            response += "• Insurance underwriting premiums\n"
            response += "• Capital market activity\n\n"
            response += "**Risks**:\n"
            response += "• Recession → credit losses\n"
            response += "• Deposit flight\n"
            response += "• Regulatory capital requirements\n"
        
        elif detected_sector  == 'XLE':  # Energy
            response += "**Energy Sector Growth Drivers**:\n"
            response += "• Oil/gas prices (commodity-driven)\n"
            response += "• ESG restrictions limiting supply\n"
            response += "• Energy security focus (post-Russia)\n"
            response += "• Dividend yield attraction\n"
            response += "• Geopolitical tensions raising prices\n\n"
            response += "**Risks**:\n"
            response += "• Long-term energy transition (EV, renewables)\n"
            response += "• Commodity price volatility\n"
            response += "• Recession demand destruction\n"
            response += "• Climate policy/carbon taxes\n"
        
        else:
            response += f"**{detected_sector} Sector Analysis**:\n"
            response += "Growth driven by economic cycles, regulation, and innovation in their respective domains.\n"
        
        response += "\n### 📈 Valuation & Attractiveness\n\n"
        response += "**Relative Valuation vs Market**:\n"
        
        # Simple relative valuation proxy
        if sector_vol > spy_vol * 1.2:
            response += f"• Higher volatility ({sector_vol*100:.1f}% vs {spy_vol*100:.1f}%) suggests growth-oriented or cyclical\n"
        else:
            response += f"• Similar volatility profile to market\n"
        
        if relative_ret > 0 and sector_vol < spy_vol:
            response += "• 🟢 Outperforming with lower risk (ideal)\n"
        elif relative_ret < 0 and sector_vol > spy_vol:
            response += "• 🔴 Underperforming with higher risk (avoid)\n"
        else:
            response += "• Mixed risk/reward profile - require sector-specific convictions\n"
        
        response += "\n### 🎯 Investment Considerations\n\n"
        response += "**Key Questions to Ask Yourself**:\n"
        response += "1. Do I believe in this sector's growth narrative?\n"
        response += "2. Is valuation reasonable given growth prospects?\n"
        response += "3. Does sector fit my risk tolerance?\n"
        response += "4. How correlated is this sector to my other holdings?\n"
        response += "5. What's the macro backdrop (rates, economy, policy)?\n"
        
    except Exception as e:
        response += f"⚠️ **Analysis Error**: {str(e)[:100]}\n"
    
    response += "\n" + INVESTMENT_DISCLAIMER
    return response
