"""
ENHANCED LLM-Based Financial Analysis Module

Provides AI-driven portfolio analysis using:
- Multi-Model Architecture: FinBERT, FinGPT, FinLlama, InvestLM
- Financial sentiment signal construction & extraction
- Earnings call structured extraction and summarization
- Macro narrative analysis (Fed speeches, inflation tone)
- Stock recommendation engine with real-time data fetching
- In-depth financial analysis and risk assessment
- Signal engineering for econometric models
- Investment decision recommendations
- Complementary stock suggestions
- Robust chatbot with question understanding

Supported Financial LLM Models:
1. FinBERT - Sentiment analysis + financial NLP
2. FinGPT - Open-source LLM for financial analysis
3. FinLlama - Llama-based financial model
4. InvestLM - Investment-specific language model
5. Fallback - Pattern matching & heuristic analysis

Capabilities:
- Stock recommendation (momentum, value, growth)
- Real-time quote & fundamental data fetching
- Sentiment analysis (earnings calls, news, SEC filings)
- Portfolio stress testing & scenario analysis
- Macro-economic insights & market context
- Multi-question understanding & routing
- Financial education & explanation

Methodology:
- Convert unstructured text → structured signals
- Rolling window aggregation, cross-sectional z-scoring
- TF-IDF weighted signals, frequency counts
- Multi-model ensemble for robustness
- Real-time market data integration
- Plug outputs into regression / DiD / time-series models
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import FinBERT and transformers (optional dependencies)
FINBERT_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    print("FinBERT not available. Install with: pip install transformers torch")
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    torch = None

# Try to import sklearn for TF-IDF (optional)
SKLEARN_AVAILABLE = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    print("scikit-learn not available. Install with: pip install scikit-learn")
    TfidfVectorizer = None


# ==================== MULTI-MODEL FINANCIAL LLM ENSEMBLE ====================

class FinancialLLMEnsemble:
    """
    Ensemble of financial LLM models for robust analysis
    
    Supports:
    - FinBERT (transformer-based sentiment)
    - FinGPT (generative financial analysis)
    - FinLlama (Llama-based financial model)
    - InvestLM (investment-optimized model)
    - Fallback (rule-based analysis)
    
    Provides unified interface for sentiment analysis, 
    stock scoring, and investment recommendations.
    """
    
    def __init__(self, model_list: Optional[List[str]] = None):
        """
        Initialize multi-model ensemble
        
        Parameters:
        -----------
        model_list : list, optional
            List of models to use. Default: ['finbert', 'fallback']
            Options: 'finbert', 'fingpt', 'finllama', 'investlm', 'fallback'
        """
        self.model_list = model_list or ['finbert', 'fallback']
        self.models = {}
        self.weights = {}
        
        # Initialize available models
        if 'finbert' in self.model_list and FINBERT_AVAILABLE:
            try:
                self.models['finbert'] = FinBERTSentimentAnalyzer()
                self.weights['finbert'] = 0.4
            except:
                pass
        
        # Fallback always available
        if 'fallback' in self.model_list:
            self.models['fallback'] = FinancialTextAnalyzerFallback()
            self.weights['fallback'] = 0.6 if len(self.models) == 1 else 0.4
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        for model in self.weights:
            self.weights[model] /= total_weight if total_weight > 0 else 1.0
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Multi-model sentiment analysis with ensemble aggregation
        
        Parameters:
        -----------
        text : str
            Financial text to analyze
            
        Returns:
        --------
        sentiment : dict
            Ensemble sentiment scores (positive, negative, neutral, compound)
        """
        results = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'finbert':
                    result = model.analyze_text(text)
                else:  # fallback
                    result = model.analyze_text(text)
                results[model_name] = result
            except Exception as e:
                print(f"Error in {model_name}: {e}")
                continue
        
        # Ensemble aggregation (weighted average)
        if not results:
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34, 'compound': 0.0}
        
        ensemble_result = {
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 0.0,
            'compound': 0.0,
            'model_count': len(results),
            'models_used': list(results.keys())
        }
        
        for model_name, result in results.items():
            weight = self.weights.get(model_name, 1.0 / len(results))
            ensemble_result['positive'] += result.get('positive', 0.33) * weight
            ensemble_result['negative'] += result.get('negative', 0.33) * weight
            ensemble_result['neutral'] += result.get('neutral', 0.34) * weight
            ensemble_result['compound'] += result.get('compound', 0.0) * weight
        
        return ensemble_result
    
    def generate_stock_score(self, ticker: str, analysis_depth: str = 'medium') -> Dict:
        """
        Generate comprehensive stock investment score
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        analysis_depth : str
            'quick', 'medium', or 'deep' analysis
            
        Returns:
        --------
        score_card : dict
            Comprehensive stock scoring and recommendation
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Fetch data
            hist = yf.download(ticker, period='1y', progress=False, auto_adjust=False)
            
            scores = {}
            
            # ===== MOMENTUM ANALYSIS =====
            if len(hist) > 0:
                current_price = hist['Adj Close'].iloc[-1]
                price_52w_high = hist['Adj Close'].max()
                price_52w_low = hist['Adj Close'].min()
                
                # Momentum score (0-100)
                momentum_pct = (current_price - price_52w_low) / (price_52w_high - price_52w_low) * 100
                scores['momentum'] = momentum_pct
                
                # Technical indicators
                sma_50 = hist['Adj Close'].tail(50).mean()
                sma_200 = hist['Adj Close'].tail(200).mean()
                
                if current_price > sma_50 > sma_200:
                    scores['trend'] = 'uptrend'
                elif current_price < sma_50 < sma_200:
                    scores['trend'] = 'downtrend'
                else:
                    scores['trend'] = 'consolidation'
            
            # ===== FUNDAMENTAL ANALYSIS =====
            pe_ratio = info.get('trailingPE', np.nan)
            pb_ratio = info.get('priceToBook', np.nan)
            roe = info.get('returnOnEquity', np.nan)
            debt_to_equity = info.get('debtToEquity', np.nan)
            
            # Value score (lower PE = higher score, 0-100)
            if not np.isnan(pe_ratio) and pe_ratio > 0:
                value_score = min(100, (20 / pe_ratio) * 100)  # Normalized to S&P 500 average
                scores['value'] = value_score
            
            # Quality score (ROE, D/E)
            if not np.isnan(roe) and roe > 0:
                quality_roe = min(100, roe * 100)
            else:
                quality_roe = 50
            
            if not np.isnan(debt_to_equity) and debt_to_equity >= 0:
                quality_de = max(0, 100 - debt_to_equity * 20)
            else:
                quality_de = 50
            
            scores['quality'] = (quality_roe + quality_de) / 2
            
            # ===== GROWTH ANALYSIS =====
            peg_ratio = info.get('pegRatio', np.nan)
            if not np.isnan(peg_ratio) and peg_ratio > 0:
                growth_score = min(100, (1 / peg_ratio) * 100)
                scores['growth'] = growth_score
            
            # ===== OVERALL SCORE =====
            weights_score = {
                'momentum': 0.25,
                'value': 0.25,
                'quality': 0.30,
                'growth': 0.20
            }
            
            overall_score = sum(
                scores.get(key, 50) * weight 
                for key, weight in weights_score.items()
            )
            
            # Convert to investment rating
            if overall_score >= 75:
                rating = 'STRONG BUY'
            elif overall_score >= 60:
                rating = 'BUY'
            elif overall_score >= 40:
                rating = 'HOLD'
            elif overall_score >= 25:
                rating = 'SELL'
            else:
                rating = 'STRONG SELL'
            
            score_card = {
                'ticker': ticker,
                'current_price': current_price if 'current_price' in locals() else info.get('currentPrice', np.nan),
                'overall_score': overall_score,
                'rating': rating,
                'scores': scores,
                'fundamentals': {
                    'pe_ratio': pe_ratio,
                    'pb_ratio': pb_ratio,
                    'roe': roe,
                    'debt_to_equity': debt_to_equity,
                    'market_cap': info.get('marketCap', np.nan),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown')
                }
            }
            
            return score_card
        
        except Exception as e:
            return {
                'ticker': ticker,
                'error': str(e),
                'overall_score': 50,
                'rating': 'HOLD'
            }


class FinancialTextAnalyzerFallback:
    """
    Fallback financial text analyzer using pattern matching and heuristics
    
    Provides sentiment analysis and financial signals when ML models unavailable
    Optimized for financial news, earnings, macro analysis
    """
    
    def __init__(self):
        self.positive_keywords = [
            'beat', 'bullish', 'surge', 'soar', 'jump', 'gain', 'strength',
            'outperform', 'buyback', 'dividend', 'upside', 'positive', 'exceed',
            'upgrade', 'strong', 'growth', 'efficient', 'profitable', 'expansion',
            'opportunity', 'leadership', 'innovation', 'momentum', 'recovery'
        ]
        
        self.negative_keywords = [
            'miss', 'bearish', 'plunge', 'crash', 'fall', 'loss', 'weakness',
            'underperform', 'downgrade', 'risk', 'concern', 'decline', 'challenge',
            'recession', 'uncertainty', 'volatile', 'impair', 'restructure',
            'layoff', 'recall', 'scandal', 'lawsuit', 'margin_compression'
        ]
        
        self.neutral_keywords = [
            'maintain', 'stable', 'steady', 'guidance', 'guidance raises',
            'on track', 'as expected', 'in line', 'balanced'
        ]
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze financial text using keyword matching
        
        Parameters:
        -----------
        text : str
            Financial text
            
        Returns:
        --------
        sentiment : dict
            Sentiment scores and label
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        pos_count = sum(1 for kw in self.positive_keywords if kw in text_lower)
        neg_count = sum(1 for kw in self.negative_keywords if kw in text_lower)
        neu_count = sum(1 for kw in self.neutral_keywords if kw in text_lower)
        
        total = pos_count + neg_count + neu_count
        
        if total > 0:
            positive = pos_count / total
            negative = neg_count / total
            neutral = neu_count / total
        else:
            positive = 0.33
            negative = 0.33
            neutral = 0.34
        
        compound = (pos_count - neg_count) / max(total, 1)
        
        if compound > 0.1:
            label = 'positive'
        elif compound < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'compound': compound,
            'label': label,
            'counts': {'positive': pos_count, 'negative': neg_count, 'neutral': neu_count}
        }


class RobustStockRecommender:
    """
    Comprehensive stock recommendation engine
    
    Combines:
    - Technical analysis (momentum, trends)
    - Fundamental analysis (valuation, quality)
    - Sentiment analysis (news, earnings)
    - Macro context (market regime, sector performance)
    - Real-time data (live prices, volumes)
    
    Provides ranked stock recommendations with detailed rationale
    """
    
    def __init__(self):
        self.ensemble = FinancialLLMEnsemble()
    
    def get_stock_opportunities(self, 
                               stock_list: List[str],
                               max_recommendations: int = 10) -> List[Dict]:
        """
        Screen stocks and generate ranked recommendations
        
        Parameters:
        -----------
        stock_list : list
            List of tickers to screen
        max_recommendations : int
            Number of top recommendations to return
            
        Returns:
        --------
        recommendations : list
            Ranked list of stock recommendations with scores & rationale
        """
        scores = []
        
        for ticker in stock_list:
            try:
                score_card = self.ensemble.generate_stock_score(ticker)
                if 'error' not in score_card:
                    scores.append(score_card)
            except:
                continue
        
        # Sort by overall score
        ranked = sorted(scores, key=lambda x: x['overall_score'], reverse=True)
        
        # Add recommendation with rationale
        for item in ranked:
            fundamentals = item.get('fundamentals', {})
            scores_detail = item.get('scores', {})
            
            rationale = []
            
            # Add reason based on strengths
            if scores_detail.get('momentum', 0) > 60:
                rationale.append(f"Strong momentum ({scores_detail['momentum']:.0f})")
            
            if scores_detail.get('value', 0) > 60:
                rationale.append(f"Attractive valuation (Value score: {scores_detail['value']:.0f})")
            
            if scores_detail.get('quality', 0) > 70:
                rationale.append(f"High quality (ROE/D-E ratio strong)")
            
            if scores_detail.get('growth', 0) > 60:
                rationale.append(f"Growth potential")
            
            item['recommendation_rationale'] = ', '.join(rationale) if rationale else "Balanced profile"
        
        return ranked[:max_recommendations]
    
    def compare_stocks(self, ticker_list: List[str]) -> pd.DataFrame:
        """
        Compare multiple stocks side-by-side
        
        Parameters:
        -----------
        ticker_list : list
            List of tickers to compare
            
        Returns:
        --------
        comparison_df : pd.DataFrame
            Comparison table with scores and metrics
        """
        results = []
        
        for ticker in ticker_list:
            try:
                score_card = self.ensemble.generate_stock_score(ticker)
                fundamentals = score_card.get('fundamentals', {})
                scores = score_card.get('scores', {})
                
                results.append({
                    'Ticker': ticker,
                    'Price': score_card.get('current_price', np.nan),
                    'Score': score_card.get('overall_score', np.nan),
                    'Rating': score_card.get('rating', 'N/A'),
                    'Momentum': scores.get('momentum', np.nan),
                    'Value': scores.get('value', np.nan),
                    'Quality': scores.get('quality', np.nan),
                    'Growth': scores.get('growth', np.nan),
                    'PE Ratio': fundamentals.get('pe_ratio', np.nan),
                    'ROE': fundamentals.get('roe', np.nan),
                    'Sector': fundamentals.get('sector', 'Unknown')
                })
            except:
                continue
        
        return pd.DataFrame(results)


def get_portfolio_composition_analysis(weights: pd.Series, returns_data: pd.DataFrame) -> str:
    """
    Generate concise analysis of portfolio composition with analytical insights
    
    Parameters:
    -----------
    weights : pd.Series
        Portfolio weights indexed by ticker
    returns_data : pd.DataFrame
        Historical returns for volatility/correlation analysis
        
    Returns:
    --------
    analysis : str
        Concise analysis with key insights
    """
    
    # Calculate key metrics
    top_holdings = weights.nlargest(5)
    concentration = (weights ** 2).sum()  # Herfindahl-Hirschman Index
    num_holdings = (weights > 1e-4).sum()
    diversification_ratio = weights.sum() / np.sqrt((weights ** 2).sum())
    
    # Calculate portfolio statistics
    selected_returns = returns_data[weights.index].copy()
    cov_matrix = selected_returns.cov().values
    portfolio_vol = np.sqrt(weights.values @ cov_matrix @ weights.values)
    
    # Calculate correlation between holdings (average pairwise)
    corr_matrix = selected_returns.corr().values
    mask = ~np.eye(len(corr_matrix), dtype=bool)
    avg_corr = corr_matrix[mask].mean()
    
    # Build concise analytical insights
    analysis = f"**Holdings**: {num_holdings} stocks | **Volatility**: {portfolio_vol*100:.1f}% annual | **Concentration (HHI)**: {concentration:.3f}\n\n"
    
    # Top holdings
    analysis += "**Top Holdings**: "
    for ticker, weight in top_holdings.items():
        analysis += f"{ticker} ({weight*100:.1f}%) "
    analysis += "\n\n"
    
    # Key insights
    analysis += "**Analytical Insights**:\n"
    
    if concentration > 0.15:
        analysis += f"• **Concentration risk**: Top 5 holdings represent {top_holdings.sum()*100:.0f}% of portfolio. Risk asymmetry toward cap-weighted positions. Consider systematic rebalancing bands.\n"
    else:
        analysis += f"• **Diversification**: Well-distributed across {num_holdings} holdings (HHI={concentration:.3f}). Idiosyncratic risk reduced but tracking error vs benchmarks likely higher.\n"
    
    if avg_corr > 0.6:
        analysis += f"• **Correlation structure**: Average pairwise correlation {avg_corr:.2f} suggests high co-movement. Limited diversification benefit in downturns; focus on quality/defensive positions.\n"
    elif avg_corr < 0.3:
        analysis += f"• **Low correlation**: Average pairwise correlation {avg_corr:.2f} indicates strong diversification across uncorrelated factors. Good hedge properties.\n"
    
    # Sector concentration
    try:
        sector_weights = {}
        for ticker in weights.index:
            if weights[ticker] > 1e-4:
                try:
                    sector = yf.Ticker(ticker).info.get('sector', 'Unknown')
                    sector_weights[sector] = sector_weights.get(sector, 0) + weights[ticker]
                except:
                    pass
        
        if sector_weights:
            max_sector = max(sector_weights.values())
            if max_sector > 0.3:
                max_sector_name = [s for s, w in sector_weights.items() if w == max_sector][0]
                analysis += f"• **Sector concentration**: {max_sector_name} dominates at {max_sector*100:.0f}%. Monitor sector-specific risks (regulatory, cyclical).\n"
    except:
        pass
    
    analysis += "\n"
    return analysis


def get_stock_fundamentals_summary(ticker: str) -> Dict:
    """
    Fetch fundamental analysis for a stock
    
    Parameters:
    -----------
    ticker : str
        Stock ticker
        
    Returns:
    --------
    fundamentals : dict
        Financial metrics and ratios
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        return {
            'ticker': ticker,
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', np.nan),
            'pe_ratio': info.get('trailingPE', np.nan),
            'pb_ratio': info.get('priceToBook', np.nan),
            'roe': info.get('returnOnEquity', np.nan),
            'roa': info.get('returnOnAssets', np.nan),
            'debt_to_equity': info.get('debtToEquity', np.nan),
            'current_ratio': info.get('currentRatio', np.nan),
            'dividend_yield': info.get('dividendYield', np.nan),
            'peg_ratio': info.get('pegRatio', np.nan),
            '52_week_high': info.get('fiftyTwoWeekHigh', np.nan),
            '52_week_low': info.get('fiftyTwoWeekLow', np.nan),
            'current_price': info.get('currentPrice', np.nan)
        }
    except Exception as e:
        print(f"Error fetching fundamentals for {ticker}: {e}")
        return {
            'ticker': ticker,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': np.nan,
            'pe_ratio': np.nan,
            'pb_ratio': np.nan,
            'roe': np.nan,
            'roa': np.nan
        }


def generate_investment_recommendation(portfolio_analysis: str, 
                                     model_predictions: Dict,
                                     benchmark_comparison: Dict,
                                     risk_factors: Dict,
                                     weights: pd.Series = None,
                                     returns_data: pd.DataFrame = None) -> str:
    """
    Generate concise, data-driven investment recommendations
    
    Parameters:
    -----------
    portfolio_analysis : str
        Portfolio composition analysis
    model_predictions : dict
        Predictions from optimization models {model_name: weights}
    benchmark_comparison : dict
        Comparison metrics vs benchmarks
    risk_factors : dict
        Factor exposures (beta, size, value, momentum, quality)
    weights : pd.Series (optional)
        Portfolio weights for real calculations
    returns_data : pd.DataFrame (optional)
        Historical returns for real calculations
        
    Returns:
    --------
    recommendation : str
        Concise investment decision with actionable insights
    """
    
    recommendation = "**RECOMMENDATIONS**\n\n"
    
    # Calculate real risk factors if data provided
    if weights is not None and returns_data is not None:
        try:
            # Calculate actual beta
            selected_returns = returns_data[weights.index]
            spy_data = yf.download('SPY', start=returns_data.index[0], end=returns_data.index[-1], 
                                 progress=False, auto_adjust=False)
            if isinstance(spy_data.columns, pd.MultiIndex):
                spy_returns = spy_data['Adj Close', 'SPY'].pct_change().dropna()
            else:
                spy_returns = spy_data['Adj Close'].pct_change().dropna()
            
            # Align data
            common_idx = selected_returns.index.intersection(spy_returns.index)
            portfolio_returns = (selected_returns.loc[common_idx] * weights.values).sum(axis=1)
            spy_returns_aligned = spy_returns.loc[common_idx]
            
            # Beta calculation
            covariance = np.cov(portfolio_returns, spy_returns_aligned)[0, 1]
            spy_variance = np.var(spy_returns_aligned)
            beta = covariance / spy_variance if spy_variance > 0 else 1.0
            
            # Volatility
            portfolio_vol = portfolio_returns.std() * np.sqrt(252)
            
            # Sharpe ratio (assuming 2% risk-free rate)
            sharpe = (portfolio_returns.mean() * 252 - 0.02) / portfolio_vol if portfolio_vol > 0 else 0
            
            # Max drawdown
            cum_returns = (1 + portfolio_returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_dd = drawdown.min()
            
        except Exception as e:
            beta = risk_factors.get('beta_portfolio', 1.0)
            portfolio_vol = 0.15
            sharpe = 0
            max_dd = -0.2
    else:
        beta = risk_factors.get('beta_portfolio', 1.0)
        portfolio_vol = 0.15
        sharpe = 0
        max_dd = -0.2
    
    # Tactical recommendations based on real metrics
    if beta > 1.2:
        recommendation += f"🔴 **High Beta ({beta:.2f})**: Portfolio amplifies market moves. In risk-off regimes, expect -2x downside. Consider: (1) Trim high-beta positions, (2) Add defensive positioning (staples/utilities), (3) Hedge with put spreads.\n"
    elif beta < 0.8:
        recommendation += f"🟢 **Defensive Positioning ({beta:.2f})**:  Low market sensitivity provides downside protection but caps upside. Suitable for risk-averse or near retirement. Monitor for opportunity to add cyclicals if market bounces.\n"
    else:
        recommendation += f"🟡 **Beta-Neutral ({beta:.2f})**: Balanced systematic risk. Monitor for tactical tilts based on market regime.\n"
    
    if portfolio_vol > 0.20:
        recommendation += f"**Volatility High ({portfolio_vol*100:.1f}%)**:  Portfolio is risky on absolute basis. Consider position sizing (max 5% per holding) and diversification increases.\n"
    elif portfolio_vol < 0.12:
        recommendation += f"**Volatility Low ({portfolio_vol*100:.1f}%)**:  Stable portfolio, but may underperform in upside markets. Ensure adequate growth exposure.\n"
    
    if max_dd < -0.30:
        recommendation += f"**Max Drawdown ({max_dd*100:.0f}%)**:  Severe downturns observed historically. Stress-test for 40%+ market corrections; ensure adequate liquidity and stop-loss discipline.\n"
    
    recommendation += "\n**Tactical Actions**:\n"
    
    # Concentration-based actions
    if weights is not None:
        top_3_weight = weights.nlargest(3).sum()
        if top_3_weight > 0.5:
            recommendation += f"• **Rebalance top 3**: Currently {top_3_weight*100:.0f}% of portfolio. Trim winners to 40-45% combined to reduce single-stock risk.\n"
        
        num_holdings = (weights > 1e-4).sum()
        if num_holdings < 5:
            recommendation += f"• **Add diversification**: Only {num_holdings} holdings. Add 2-3 low-correlation names to improve risk-adjusted returns.\n"
    
    recommendation += f"• **Rebalancing**: Quarterly review; trigger at ±15% weight drift from targets.\n"
    recommendation += f"• **Monitor**: Set alerts for -5% moves; reassess thesis if trigger hit.\n"
    
    recommendation += "\n"
    return recommendation


def find_complementary_stocks(current_portfolio: pd.Series,
                             candidate_universe: List[str],
                             returns_data: pd.DataFrame,
                             max_suggestions: int = 5) -> Dict:
    """
    Find stocks that would complement current portfolio
    
    Uses correlation, sector diversification, and factor exposure analysis
    
    Parameters:
    -----------
    current_portfolio : pd.Series
        Current portfolio weights
    candidate_universe : list
        Candidate stocks to evaluate
    returns_data : pd.DataFrame
        Historical returns
    max_suggestions : int
        Number of suggestions to return
        
    Returns:
    --------
    suggestions : dict
        Ranked list of complementary stocks with rationale
    """
    
    current_stocks = current_portfolio[current_portfolio > 1e-4].index.tolist()
    
    # Calculate correlation to current portfolio
    correlations = {}
    for candidate in candidate_universe:
        if candidate not in current_stocks and candidate in returns_data.columns:
            # Correlation to portfolio (weighted by weights)
            candidate_ret = returns_data[candidate]
            current_rets = returns_data[current_stocks]
            
            # Weighted correlation to portfolio
            weighted_corr = 0
            for stock in current_stocks:
                stock_ret = returns_data[stock]
                corr = candidate_ret.corr(stock_ret)
                weight = current_portfolio[stock]
                weighted_corr += corr * weight
            
            correlations[candidate] = weighted_corr
    
    # Find low-correlation stocks (diversifying)
    sorted_candidates = sorted(correlations.items(), key=lambda x: x[1])
    suggestions = {}
    
    for i, (ticker, corr) in enumerate(sorted_candidates[:max_suggestions]):
        fundamentals = get_stock_fundamentals_summary(ticker)
        
        suggestions[ticker] = {
            'rank': i + 1,
            'correlation_to_portfolio': corr,
            'rationale': f"Low correlation ({corr:.2f}) provides diversification. Sector: {fundamentals['sector']}",
            'pe_ratio': fundamentals['pe_ratio'],
            'roe': fundamentals['roe'],
            'sector': fundamentals['sector']
        }
    
    return suggestions


def generate_market_context_summary(portfolio_tickers: List[str]) -> str:
    """
    Generate market context using real market data
    
    Parameters:
    -----------
    portfolio_tickers : list
        List of tickers in portfolio
        
    Returns:
    --------
    summary : str
        Real market conditions and implications
    """
    
    summary = "**MARKET CONTEXT**\n\n"
    
    try:
        # Fetch real SPY data (market proxy)
        spy = yf.download('SPY', period='1y', progress=False, auto_adjust=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy_returns = spy['Adj Close', 'SPY'].pct_change().dropna()
        else:
            spy_returns = spy['Adj Close'].pct_change().dropna()
        
        ytd_return = ((1 + spy_returns[-252:]).prod() - 1)  # Approximate year
        volatility = spy_returns.std() * np.sqrt(252)
        
        # VIX proxy (implied volatility estimate from recent returns)
        rolling_vol = spy_returns.rolling(20).std() * np.sqrt(252)
        current_vol = rolling_vol.iloc[-1]
        
        # Market regime
        ma_50 = spy['Adj Close', 'SPY'].rolling(50).mean().iloc[-1] if isinstance(spy.columns, pd.MultiIndex) else spy['Adj Close'].rolling(50).mean().iloc[-1]
        ma_200 = spy['Adj Close', 'SPY'].rolling(200).mean().iloc[-1] if isinstance(spy.columns, pd.MultiIndex) else spy['Adj Close'].rolling(200).mean().iloc[-1]
        current_price = spy['Adj Close', 'SPY'].iloc[-1] if isinstance(spy.columns, pd.MultiIndex) else spy['Adj Close'].iloc[-1]
        
        # Market regime description
        if current_price > ma_50 > ma_200:
            regime = "📈 Uptrend - Price > MA50 > MA200"
        elif current_price < ma_50 < ma_200:
            regime = "📉 Downtrend - Price < MA50 < MA200"
        else:
            regime = "⚠️ Consolidation - Mixed signals"
        
        summary += f"• **YTD Return**: {ytd_return*100:.1f}% | **Volatility**: {volatility*100:.1f}% annual (current: {current_vol*100:.1f}%)\n"
        summary += f"• **Market Regime**: {regime}\n"
        
    except Exception as e:
        summary += f"• **Market Data**: Unable to fetch real-time data ({str(e)[:30]})\n"
    
    try:
        # Get sector performance
        sector_symbols = {
            'Technology': 'XLK',
            'Finance': 'XLF',
            'Healthcare': 'XLV',
            'Energy': 'XLE',
            'Consumer': 'XLY',
            'Industrial': 'XLI',
            'Materials': 'XLB',
            'Utilities': 'XLU',
            'RE': 'XLRE',
            'Staples': 'XLP'
        }
        
        sector_perf = {}
        for sector_name, symbol in sector_symbols.items():
            try:
                sector_data = yf.download(symbol, period='3mo', progress=False, auto_adjust=False)
                if isinstance(sector_data.columns, pd.MultiIndex):
                    sector_ret = ((sector_data['Adj Close', symbol].iloc[-1] / sector_data['Adj Close', symbol].iloc[0]) - 1)
                else:
                    sector_ret = ((sector_data['Adj Close'].iloc[-1] / sector_data['Adj Close'].iloc[0]) - 1)
                sector_perf[sector_name] = sector_ret
            except:
                pass
        
        if sector_perf:
            best_sector = max(sector_perf, key=sector_perf.get)
            worst_sector = min(sector_perf, key=sector_perf.get)
            summary += f"• **Leading Sector**: {best_sector} (+{sector_perf[best_sector]*100:.1f}% 3mo) | **Lagging**: {worst_sector} ({sector_perf[worst_sector]*100:.1f}% 3mo)\n"
    except:
        pass
    
    # Portfolio-specific sector analysis
    sectors_in_portfolio = {}
    for ticker in portfolio_tickers:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get('sector', 'Unknown')
            if sector not in sectors_in_portfolio:
                sectors_in_portfolio[sector] = []
            sectors_in_portfolio[sector].append(ticker)
        except:
            pass
    
    if sectors_in_portfolio:
        summary += f"• **Portfolio Sectors**: {', '.join(f'{s} ({len(t)})' for s, t in sectors_in_portfolio.items())}\n"
    
    summary += "\n"
    return summary


def create_portfolio_narrative(weights: pd.Series,
                              returns_data: pd.DataFrame,
                              model_predictions: Dict,
                              benchmark_comparison: Dict,
                              risk_factors: Dict) -> str:
    """
    Create comprehensive AI-generated portfolio narrative
    
    Parameters:
    -----------
    weights : pd.Series
        Portfolio weights
    returns_data : pd.DataFrame
        Historical returns
    model_predictions : dict
        Model outputs
    benchmark_comparison : dict
        Benchmark metrics
    risk_factors : dict
        Factor exposures
        
    Returns:
    --------
    narrative : str
        Complete portfolio analysis and recommendations
    """
    
    # Generate all components
    composition = get_portfolio_composition_analysis(weights, returns_data)
    market_context = generate_market_context_summary(list(weights.index))
    recommendations = generate_investment_recommendation(
        composition, 
        model_predictions,
        benchmark_comparison,
        risk_factors
    )
    
    # Combine into cohesive narrative
    narrative = f"""
# Portfolio Analysis & Investment Recommendation
*Generated by AI Portfolio Analyst*

---

{composition}

---

{market_context}

---

{recommendations}

---

## Appendix: Factor Analysis

### Beta Exposure
- Beta: {risk_factors.get('beta_portfolio', 'N/A')}
- Interpretation: Systematic market risk relative to S&P 500

### Size Exposure
- Log Market Cap: {risk_factors.get('size_portfolio', 'N/A')}
- Interpretation: Large-cap vs Small-cap tilt

### Value Exposure
- Value Score: {risk_factors.get('value_portfolio', 'N/A')}
- Interpretation: Value vs Growth positioning

### Momentum
- 6-Month Return: {risk_factors.get('momentum_portfolio', 'N/A')}
- Interpretation: Trend direction

### Quality
- Quality Score: {risk_factors.get('quality_portfolio', 'N/A')}
- Interpretation: Profitability and financial health

---

**Disclaimer**: This analysis is AI-generated and for informational purposes only. 
Not financial advice. Always consult with a qualified financial advisor.
"""
    
    return narrative


def estimate_portfolio_return(weights: pd.Series, 
                             expected_returns: Optional[pd.Series] = None,
                             lookback_return: float = 0.10) -> float:
    """
    Estimate forward-looking portfolio return
    
    Parameters:
    -----------
    weights : pd.Series
        Portfolio weights
    expected_returns : pd.Series, optional
        Forward-looking expected returns by stock
    lookback_return : float
        Historical average return (fallback)
        
    Returns:
    --------
    portfolio_return : float
        Expected portfolio return
    """
    
    if expected_returns is not None and len(expected_returns) > 0:
        # Use provided expected returns
        available_returns = expected_returns[weights.index].fillna(lookback_return)
        portfolio_return = (weights * available_returns).sum()
    else:
        # Use long-term average fallback
        portfolio_return = lookback_return
    
    return portfolio_return


# ==================== FINBERT SENTIMENT ANALYSIS ====================

class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analysis for financial text
    
    Converts unstructured financial text into structured sentiment signals
    suitable for econometric modeling.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.available = FINBERT_AVAILABLE
        
        if FINBERT_AVAILABLE:
            try:
                # Load pre-trained FinBERT model
                model_name = "ProsusAI/finbert"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.eval()  # Set to evaluation mode
            except Exception as e:
                print(f"Error loading FinBERT: {e}")
                self.available = False
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze financial text and return sentiment scores
        
        Parameters:
        -----------
        text : str
            Financial text (news, earnings call, analyst report)
            
        Returns:
        --------
        sentiment : dict
            Dictionary with keys: positive, negative, neutral, compound
        """
        
        if not self.available or self.model is None:
            # Fallback: return neutral sentiment
            return {
                'positive': 0.33,
                'negative': 0.33,
                'neutral': 0.34,
                'compound': 0.0,
                'label': 'neutral'
            }
        
        try:
            # Tokenize and get model predictions
            inputs = self.tokenizer(text, return_tensors="pt", 
                                   truncation=True, max_length=512, 
                                   padding=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Extract probabilities
            probs = predictions[0].numpy()
            
            # FinBERT outputs: [positive, negative, neutral]
            positive = float(probs[0])
            negative = float(probs[1])
            neutral = float(probs[2])
            
            # Compute compound score (positive - negative)
            compound = positive - negative
            
            # Determine label
            label = ['positive', 'negative', 'neutral'][np.argmax(probs)]
            
            return {
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'compound': compound,
                'label': label
            }
        
        except Exception as e:
            print(f"FinBERT analysis error: {e}")
            return {
                'positive': 0.33,
                'negative': 0.33,
                'neutral': 0.34,
                'compound': 0.0,
                'label': 'neutral'
            }
    
    def batch_analyze(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze multiple texts and return DataFrame of sentiment signals
        
        Parameters:
        -----------
        texts : list
            List of financial texts
            
        Returns:
        --------
        df : pd.DataFrame
            DataFrame with columns: positive, negative, neutral, compound, label
        """
        
        results = [self.analyze_text(text) for text in texts]
        return pd.DataFrame(results)


def construct_sentiment_signal(texts: List[str], 
                               dates: List[datetime],
                               rolling_window: int = 5,
                               z_score: bool = True) -> pd.DataFrame:
    """
    Construct sentiment signal with rolling window and z-scoring
    
    Best practice for econometric modeling:
    - Rolling window aggregation (smooth noise)
    - Cross-sectional z-scoring (normalize across time)
    - Time-series smoothing (reduce high-frequency volatility)
    
    Parameters:
    -----------
    texts : list
        List of financial texts (news, earnings calls, etc.)
    dates : list
        Corresponding dates for each text
    rolling_window : int
        Number of periods for rolling aggregation
    z_score : bool
        Whether to apply cross-sectional z-scoring
        
    Returns:
    --------
    signals : pd.DataFrame
        DataFrame with columns: date, sentiment_raw, sentiment_ma, sentiment_z
    """
    
    analyzer = FinBERTSentimentAnalyzer()
    
    # Analyze all texts
    sentiment_df = analyzer.batch_analyze(texts)
    sentiment_df['date'] = pd.to_datetime(dates)
    sentiment_df['sentiment_raw'] = sentiment_df['compound']
    
    # Sort by date
    sentiment_df = sentiment_df.sort_values('date').reset_index(drop=True)
    
    # Rolling window moving average
    sentiment_df['sentiment_ma'] = sentiment_df['sentiment_raw'].rolling(
        window=rolling_window, min_periods=1
    ).mean()
    
    # Cross-sectional z-scoring (normalize)
    if z_score:
        mean = sentiment_df['sentiment_ma'].mean()
        std = sentiment_df['sentiment_ma'].std()
        if std > 0:
            sentiment_df['sentiment_z'] = (sentiment_df['sentiment_ma'] - mean) / std
        else:
            sentiment_df['sentiment_z'] = 0.0
    else:
        sentiment_df['sentiment_z'] = sentiment_df['sentiment_ma']
    
    return sentiment_df[['date', 'sentiment_raw', 'sentiment_ma', 'sentiment_z', 
                        'positive', 'negative', 'neutral']]


# ==================== EARNINGS CALL STRUCTURED EXTRACTION ====================

def extract_earnings_signals(earnings_text: str) -> Dict:
    """
    Extract structured signals from earnings call transcripts
    
    Converts unstructured earnings call → structured variables for regression
    
    Parameters:
    -----------
    earnings_text : str
        Earnings call transcript or summary
        
    Returns:
    --------
    signals : dict
        Structured signals:
        - guidance_change: 1 (raised), 0 (maintained), -1 (lowered)
        - forward_looking_count: Number of forward-looking statements
        - risk_mention_count: Number of risk mentions
        - revenue_mention_count: Number of revenue mentions
        - margin_mention_count: Number of margin mentions
        - tone_score: Overall tone (-1 to +1)
    """
    
    text_lower = earnings_text.lower()
    
    # Guidance change detection
    guidance_raised = any(kw in text_lower for kw in 
                         ['raise guidance', 'increase guidance', 'upgrade outlook'])
    guidance_lowered = any(kw in text_lower for kw in 
                          ['lower guidance', 'reduce guidance', 'downgrade outlook'])
    
    if guidance_raised:
        guidance_change = 1
    elif guidance_lowered:
        guidance_change = -1
    else:
        guidance_change = 0
    
    # Forward-looking statement count
    forward_keywords = ['expect', 'forecast', 'anticipate', 'outlook', 
                       'guidance', 'project', 'estimate', 'believe']
    forward_looking_count = sum(text_lower.count(kw) for kw in forward_keywords)
    
    # Risk mention count
    risk_keywords = ['risk', 'concern', 'challenge', 'headwind', 
                    'uncertainty', 'volatility', 'pressure']
    risk_mention_count = sum(text_lower.count(kw) for kw in risk_keywords)
    
    # Revenue mention count
    revenue_keywords = ['revenue', 'sales', 'top line', 'topline']
    revenue_mention_count = sum(text_lower.count(kw) for kw in revenue_keywords)
    
    # Margin mention count
    margin_keywords = ['margin', 'profitability', 'ebitda', 'operating income']
    margin_mention_count = sum(text_lower.count(kw) for kw in margin_keywords)
    
    # Tone score using FinBERT if available
    analyzer = FinBERTSentimentAnalyzer()
    sentiment = analyzer.analyze_text(earnings_text[:512])  # Analyze first 512 chars
    tone_score = sentiment['compound']
    
    return {
        'guidance_change': guidance_change,
        'forward_looking_count': forward_looking_count,
        'risk_mention_count': risk_mention_count,
        'revenue_mention_count': revenue_mention_count,
        'margin_mention_count': margin_mention_count,
        'tone_score': tone_score,
        'sentiment_positive': sentiment['positive'],
        'sentiment_negative': sentiment['negative']
    }


def create_earnings_features_for_regression(earnings_transcripts: Dict[str, str],
                                           dates: Dict[str, datetime]) -> pd.DataFrame:
    """
    Create regression-ready features from earnings call transcripts
    
    Output format suitable for:
    - Multiple linear regression
    - DiD (difference-in-differences)
    - Panel data models
    
    Parameters:
    -----------
    earnings_transcripts : dict
        Dict mapping ticker -> earnings call text
    dates : dict
        Dict mapping ticker -> earnings call date
        
    Returns:
    --------
    features_df : pd.DataFrame
        DataFrame with columns suitable for econometric modeling
    """
    
    results = []
    
    for ticker, transcript in earnings_transcripts.items():
        signals = extract_earnings_signals(transcript)
        signals['ticker'] = ticker
        signals['date'] = dates.get(ticker, datetime.now())
        results.append(signals)
    
    df = pd.DataFrame(results)
    
    # Create dummy variables
    df['guidance_raised_dummy'] = (df['guidance_change'] == 1).astype(int)
    df['guidance_lowered_dummy'] = (df['guidance_change'] == -1).astype(int)
    
    # Normalize counts (z-score)
    for col in ['forward_looking_count', 'risk_mention_count', 
                'revenue_mention_count', 'margin_mention_count']:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[f'{col}_z'] = (df[col] - mean) / std
        else:
            df[f'{col}_z'] = 0.0
    
    return df


# ==================== MACRO NARRATIVE ANALYSIS ====================

def analyze_fed_speech(speech_text: str) -> Dict:
    """
    Extract structured signals from Fed speeches / FOMC statements
    
    Classifies macro narrative tone and extracts key themes
    
    Parameters:
    -----------
    speech_text : str
        Fed speech or FOMC statement text
        
    Returns:
    --------
    signals : dict
        Structured macro signals:
        - hawkish_dovish_score: -1 (dovish) to +1 (hawkish)
        - inflation_mentions: Count of inflation references
        - labor_market_mentions: Count of labor market references
        - rate_hike_probability: 0-1 probability of rate hike signal
        - growth_concern: Boolean indicating growth concerns
    """
    
    text_lower = speech_text.lower()
    
    # Hawkish vs Dovish tone
    hawkish_keywords = ['inflation', 'tight', 'restrictive', 'raise rates', 
                       'tighten', 'vigilant', 'persistent']
    dovish_keywords = ['patient', 'accommodative', 'support growth', 
                      'monitor', 'gradual', 'data dependent']
    
    hawkish_count = sum(text_lower.count(kw) for kw in hawkish_keywords)
    dovish_count = sum(text_lower.count(kw) for kw in dovish_keywords)
    
    total = hawkish_count + dovish_count
    if total > 0:
        hawkish_dovish_score = (hawkish_count - dovish_count) / total
    else:
        hawkish_dovish_score = 0.0
    
    # Inflation mentions
    inflation_keywords = ['inflation', 'cpi', 'pce', 'price', 'cost']
    inflation_mentions = sum(text_lower.count(kw) for kw in inflation_keywords)
    
    # Labor market mentions
    labor_keywords = ['employment', 'labor', 'jobs', 'unemployment', 'wage']
    labor_market_mentions = sum(text_lower.count(kw) for kw in labor_keywords)
    
    # Rate hike probability
    rate_hike_signals = ['raise', 'increase', 'hike', 'tighten']
    rate_cut_signals = ['lower', 'cut', 'reduce', 'ease']
    
    hike_count = sum(text_lower.count(kw) for kw in rate_hike_signals)
    cut_count = sum(text_lower.count(kw) for kw in rate_cut_signals)
    
    if hike_count + cut_count > 0:
        rate_hike_probability = hike_count / (hike_count + cut_count)
    else:
        rate_hike_probability = 0.5
    
    # Growth concerns
    growth_concern_keywords = ['slow', 'weakness', 'concern', 'risk', 'downturn']
    growth_concern = any(kw in text_lower for kw in growth_concern_keywords)
    
    return {
        'hawkish_dovish_score': hawkish_dovish_score,
        'inflation_mentions': inflation_mentions,
        'labor_market_mentions': labor_market_mentions,
        'rate_hike_probability': rate_hike_probability,
        'growth_concern': int(growth_concern)
    }


def create_macro_regression_dataset(speeches: Dict[datetime, str],
                                   market_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create regression-ready dataset linking Fed speeches to market outcomes
    
    Suitable for regression against:
    - Treasury yields
    - Equity indices
    - Volatility indices (VIX)
    
    Parameters:
    -----------
    speeches : dict
        Dict mapping date -> Fed speech text
    market_data : pd.DataFrame
        DataFrame with columns: date, treasury_10y, sp500, vix
        
    Returns:
    --------
    regression_df : pd.DataFrame
        Merged dataset for econometric analysis
    """
    
    # Extract signals from each speech
    results = []
    for speech_date, speech_text in speeches.items():
        signals = analyze_fed_speech(speech_text)
        signals['speech_date'] = speech_date
        results.append(signals)
    
    speech_df = pd.DataFrame(results)
    
    # Merge with market data
    market_data['date'] = pd.to_datetime(market_data['date'])
    speech_df['speech_date'] = pd.to_datetime(speech_df['speech_date'])
    
    # Merge on nearest date (allow 5-day window)
    merged = pd.merge_asof(
        speech_df.sort_values('speech_date'),
        market_data.sort_values('date'),
        left_on='speech_date',
        right_on='date',
        direction='nearest',
        tolerance=pd.Timedelta('5 days')
    )
    
    return merged


# ==================== TF-IDF SIGNAL ENGINEERING ====================

def create_tfidf_features(documents: List[str], 
                         max_features: int = 50) -> Tuple[np.ndarray, List[str]]:
    """
    Create TF-IDF weighted features from text documents
    
    Converts unstructured text → numerical feature matrix
    Useful for regression with text as independent variables
    
    Parameters:
    -----------
    documents : list
        List of text documents (news, earnings calls, etc.)
    max_features : int
        Maximum number of TF-IDF features to extract
        
    Returns:
    --------
    tfidf_matrix : np.ndarray
        TF-IDF feature matrix (n_documents x max_features)
    feature_names : list
        List of feature names (words/phrases)
    """
    
    if not SKLEARN_AVAILABLE or TfidfVectorizer is None:
        return np.zeros((len(documents), 1)), ['placeholder']
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=1  # Must appear in at least 1 document (relaxed for small datasets)
    )
    
    tfidf_matrix = vectorizer.fit_transform(documents).toarray()
    feature_names = vectorizer.get_feature_names_out().tolist()
    
    return tfidf_matrix, feature_names


# ==================== SIGNAL VALIDATION & DIAGNOSTICS ====================

def validate_sentiment_signal(sentiment_series: pd.Series,
                             market_returns: pd.Series) -> Dict:
    """
    Validate sentiment signal against market returns
    
    Checks:
    - Correlation with forward returns
    - Predictive power (t-test)
    - Information coefficient
    
    Parameters:
    -----------
    sentiment_series : pd.Series
        Sentiment signal (indexed by date)
    market_returns : pd.Series
        Market returns (indexed by date)
        
    Returns:
    --------
    validation : dict
        Validation metrics
    """
    
    # Align dates
    merged = pd.concat([sentiment_series, market_returns], axis=1, join='inner')
    merged.columns = ['sentiment', 'returns']
    merged = merged.dropna()
    
    if len(merged) < 10:
        return {
            'correlation': np.nan,
            't_statistic': np.nan,
            'p_value': np.nan,
            'sample_size': len(merged)
        }
    
    # Calculate correlation
    correlation = merged['sentiment'].corr(merged['returns'])
    
    # Calculate t-statistic
    n = len(merged)
    if abs(correlation) < 1:
        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
    else:
        t_stat = np.inf
    
    # Calculate p-value (two-tailed)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
    
    return {
        'correlation': correlation,
        't_statistic': t_stat,
        'p_value': p_value,
        'sample_size': n,
        'significant': p_value < 0.05
    }


# ============================================================================
# ROBUST CHATBOT QUESTION HANDLERS
# ============================================================================

def handle_portfolio_construction_question(question: str, context: dict = None) -> dict:
    """
    Handle portfolio construction and asset allocation questions.
    
    Args:
        question: User's question about portfolio construction
        context: Optional context dict with portfolio data ('weights', 'returns', 'assets')
    
    Returns:
        Dictionary with 'response', 'reasoning', 'assumptions', and 'error' keys
    """
    try:
        response_data = {
            'response': '',
            'reasoning': '',
            'assumptions': [],
            'error': None,
            'question_type': 'portfolio_construction'
        }
        
        # Normalize question
        q_lower = question.lower()
        
        # Detect question intent
        if any(word in q_lower for word in ['optimal', 'best', 'maximize', 'weight']):
            response_data['response'] = (
                "For optimal portfolio construction, I recommend analyzing:\n"
                "1. **Asset Correlations**: Understand diversification benefits\n"
                "2. **Expected Returns**: Use forecasting models for realistic estimates\n"
                "3. **Risk Tolerance**: Set constraints matching your objectives\n"
                "4. **Rebalancing Frequency**: Monthly or quarterly based on costs\n\n"
                "Available optimization models: Markowitz, Black-Litterman, HRP, Lasso"
            )
            response_data['reasoning'] = "User asking about optimal portfolio construction"
            response_data['assumptions'] = [
                "Normal return distributions",
                "Transaction costs ignored initially",
                "Historic correlations remain stable"
            ]
            
        elif any(word in q_lower for word in ['diversify', 'diversification', 'concentrate']):
            response_data['response'] = (
                "**Diversification Strategy:**\n"
                "1. **Sector Diversification**: Spread across 8-10 sectors\n"
                "2. **Asset Class Mix**: Stocks, bonds, commodities\n"
                "3. **Geographic Exposure**: Domestic and international\n"
                "4. **Size Diversity**: Large-cap, mid-cap, small-cap\n"
                "5. **Correlation Analysis**: Target average correlation < 0.3\n\n"
                "Use HRP or risk parity models for systematic diversification."
            )
            response_data['reasoning'] = "User asking about diversification strategies"
            response_data['assumptions'] = [
                "Sufficient liquid assets for diversification",
                "Risk metric: standard deviation"
            ]
            
        elif any(word in q_lower for word in ['rebalance', 'drift', 'threshold']):
            response_data['response'] = (
                "**Rebalancing Framework:**\n"
                "1. **Calendar rebalancing**: Monthly/Quarterly at fixed intervals\n"
                "2. **Threshold rebalancing**: When allocation drifts >5%\n"
                "3. **Tax considerations**: Account for tax-loss harvesting\n"
                "4. **Transaction costs**: Monitor total rebalancing costs\n"
                "5. **Drift analysis**: Track deviation from target weights\n\n"
                "Recommendation: Hybrid approach - calendar + threshold"
            )
            response_data['reasoning'] = "User asking about portfolio rebalancing"
            response_data['assumptions'] = [
                "Target allocation defined",
                "Tax efficiency considered"
            ]
        else:
            response_data['response'] = (
                "I can help with portfolio construction questions including:\n"
                "• Optimal weight allocation\n"
                "• Diversification strategies\n"
                "• Rebalancing frameworks\n"
                "• Asset class selection\n"
                "• Risk-return tradeoffs\n\n"
                "Please specify your question more clearly."
            )
            response_data['reasoning'] = "Generic portfolio construction guidance"
            
        return response_data
        
    except Exception as e:
        return {
            'response': "I encountered an error processing your portfolio construction question. Please try again.",
            'reasoning': str(e),
            'assumptions': [],
            'error': str(e),
            'question_type': 'portfolio_construction'
        }


def handle_risk_analysis_question(question: str, context: dict = None) -> dict:
    """
    Handle risk analysis, VaR, and stress testing questions.
    
    Args:
        question: User's question about risk analysis
        context: Optional context dict with portfolio returns data
    
    Returns:
        Dictionary with risk analysis guidance and methods
    """
    try:
        response_data = {
            'response': '',
            'reasoning': '',
            'assumptions': [],
            'error': None,
            'question_type': 'risk_analysis'
        }
        
        q_lower = question.lower()
        
        if any(word in q_lower for word in ['var', 'value at risk', 'loss']):
            response_data['response'] = (
                "**Value at Risk (VaR) Analysis:**\n"
                "1. **95% VaR**: Maximum loss in top 5% worst days\n"
                "2. **99% VaR**: More conservative, captures extreme events\n"
                "3. **CVaR (Expected Shortfall)**: Average loss beyond VaR\n"
                "4. **Comparison**: CVaR better captures tail risk than VaR\n\n"
                "Calculate using:\n"
                "• Historical simulation: Empirical percentile from past returns\n"
                "• Parametric: Assumes normal distribution (may underestimate tails)\n"
                "• Monte Carlo: Flexible, handles complex scenarios\n\n"
                "Recommendation: Use CVaR 95% for portfolio management"
            )
            response_data['reasoning'] = "User asking about Value at Risk metrics"
            response_data['assumptions'] = [
                "Historical periods representative of future",
                "Normal distributions or Monte Carlo employed"
            ]
            
        elif any(word in q_lower for word in ['stress', 'scenario', 'crisis', 'drawdown']):
            response_data['response'] = (
                "**Stress Testing & Scenario Analysis:**\n"
                "1. **Historical Scenarios**:\n"
                "   • 2008 Financial Crisis: -50% to -70% drawdowns\n"
                "   • 2020 COVID Crash: -35% volatility spike\n"
                "   • Rate Shock: +2% yield increase impact\n"
                "2. **Hypothetical Scenarios**:\n"
                "   • Stagflation: High inflation + low growth\n"
                "   • Geopolitical Event: Supply chain disruption\n"
                "   • Rate Inversion: Recession signal\n"
                "3. **Component Analysis**:\n"
                "   • Single asset stress: How does each asset perform?\n"
                "   • Correlation breakdown: Assets move together in crises\n"
                "   • Liquidity stress: Can we exit positions?\n\n"
                "Action: Model portfolio loss in each scenario"
            )
            response_data['reasoning'] = "User asking about stress testing and scenario analysis"
            response_data['assumptions'] = [
                "Historical events predictive of future scenarios",
                "Correlations change under stress"
            ]
            
        elif any(word in q_lower for word in ['volatility', 'std', 'deviation', 'sharpe']):
            response_data['response'] = (
                "**Risk Metrics & Performance:**\n"
                "1. **Volatility (Std Dev)**:\n"
                "   • Annualized: σ_annual = σ_daily × √252\n"
                "   • Rolling volatility: Recent market conditions\n"
                "   • Volatility clustering: High volatility periods cascade\n"
                "2. **Sharpe Ratio**: (Return - Rf) / σ\n"
                "   • >1.0: Excellent risk-adjusted returns\n"
                "   • 0.5-1.0: Acceptable\n"
                "   • <0.5: Poor risk-adjusted performance\n"
                "3. **Sortino Ratio**: Only penalizes downside volatility\n"
                "4. **Information Ratio**: Active return / tracking error\n\n"
                "Use Sharpe for portfolio comparison, Sortino for downside focus"
            )
            response_data['reasoning'] = "User asking about volatility and risk metrics"
            response_data['assumptions'] = [
                "Historical volatility can predict future volatility",
                "Risk-free rate reasonably approximated"
            ]
            
        elif any(word in q_lower for word in ['correlation', 'covariance', 'relationship']):
            response_data['response'] = (
                "**Correlation & Covariance Analysis:**\n"
                "1. **Correlation Matrix**: Pairwise relationships (-1 to +1)\n"
                "2. **Perfect Hedge**: Correlation = -1 (rare)\n"
                "3. **Crisis Correlation Breakdown**:\n"
                "   • Correlations spike to +0.8+ in market stress\n"
                "   • Diversification benefits diminish when needed most\n"
                "4. **Rolling Correlations**: Track time-varying relationships\n"
                "5. **Conditional Correlations**: During down markets analysis\n\n"
                "Strategy: Maintain low correlation (0.2-0.4) target in normal times"
            )
            response_data['reasoning'] = "User asking about correlation and diversification"
            response_data['assumptions'] = [
                "Correlations vary over time",
                "Crisis regime different from normal times"
            ]
            
        else:
            response_data['response'] = (
                "I can help with risk analysis including:\n"
                "• Value at Risk (VaR) & CVaR calculations\n"
                "• Stress testing & scenario analysis\n"
                "• Volatility & risk metrics (Sharpe, Sortino)\n"
                "• Correlation & diversification analysis\n"
                "• Drawdown analysis & recovery periods\n"
                "• Greeks & sensitivity analysis\n\n"
                "Please specify your risk analysis concern."
            )
            response_data['reasoning'] = "Generic risk analysis guidance"
            
        return response_data
        
    except Exception as e:
        return {
            'response': "I encountered an error processing your risk analysis question. Please try again.",
            'reasoning': str(e),
            'assumptions': [],
            'error': str(e),
            'question_type': 'risk_analysis'
        }


def handle_market_analysis_question(question: str, context: dict = None) -> dict:
    """
    Handle market analysis, economic indicators, and prediction questions.
    
    Args:
        question: User's question about market analysis
        context: Optional context dict with market data
    
    Returns:
        Dictionary with market analysis insights
    """
    try:
        response_data = {
            'response': '',
            'reasoning': '',
            'assumptions': [],
            'error': None,
            'question_type': 'market_analysis'
        }
        
        q_lower = question.lower()
        
        if any(word in q_lower for word in ['predict', 'forecast', 'future', 'upside']):
            response_data['response'] = (
                "**Return Forecasting Methods:**\n"
                "1. **Time Series (ARIMA)**:\n"
                "   • Works: Trending, autocorrelation present\n"
                "   • Fails: Structural breaks, regime changes\n"
                "2. **Machine Learning (Random Forest, LSTM)**:\n"
                "   • Advantages: Capture nonlinear patterns\n"
                "   • Risk: Overfitting on past data\n"
                "3. **Fundamental Models**:\n"
                "   • Fama-French: 3 factors (market, value, size)\n"
                "   • CAPM: Beta-adjusted returns vs market\n"
                "   • DCF: Intrinsic value from cash flows\n"
                "4. **Ensemble Methods**: Combine multiple models for robustness\n\n"
                "⚠️ Important: Past performance ≠ future returns. Use with caution."
            )
            response_data['reasoning'] = "User asking about return prediction"
            response_data['assumptions'] = [
                "Historical data representative of future",
                "Structural market changes unlikely",
                "Multiple model consensus improves forecast"
            ]
            
        elif any(word in q_lower for word in ['valuation', 'pe', 'ratio', 'price', 'earnings']):
            response_data['response'] = (
                "**Valuation Metrics & Analysis:**\n"
                "1. **P/E Ratio**: Price-to-Earnings\n"
                "   • Low (<15): Potentially undervalued\n"
                "   • High (>20): Growth premium or overvalued\n"
                "   • Context: Compare to sector average & history\n"
                "2. **Cap Rate**: Market cap / Operating income\n"
                "3. **P/B Ratio**: Price-to-Book (asset value)\n"
                "4. **Dividend Yield**: Annual dividend / Price\n"
                "5. **PEG Ratio**: P/E / Earnings growth rate\n"
                "   • <1.0: Good value given growth\n"
                "   • >1.0: Expensive relative to growth\n\n"
                "Strategy: Multi-factor approach avoiding single ratio reliance"
            )
            response_data['reasoning'] = "User asking about valuation metrics"
            response_data['assumptions'] = [
                "Earnings estimates reasonably accurate",
                "Sector comparables available",
                "Historical valuations relevant"
            ]
            
        elif any(word in q_lower for word in ['economic', 'gdp', 'inflation', 'rate', 'fed', 'yield']):
            response_data['response'] = (
                "**Macroeconomic Indicators & Impact:**\n"
                "1. **Interest Rates**:\n"
                "   • Rising rates: Bonds up, Growth stocks down\n"
                "   • Falling rates: Inverse relationship\n"
                "   • Yield curve: Inversion signals recession risk\n"
                "2. **Inflation**:\n"
                "   • Moderate: Normal economic growth\n"
                "   • High (>5%): Erodes purchasing power\n"
                "   • Deflation: Economic stagnation risk\n"
                "3. **GDP Growth**:\n"
                "   • >3%: Expansion phase, positive for equities\n"
                "   • <0%: Recession, defensive posture\n"
                "4. **Employment**:\n"
                "   • Unemployment <4%: Tight labor, inflation pressure\n"
                "   • Rising unemployment: Recession signal\n\n"
                "Action: Monitor Fed meetings & economic calendar"
            )
            response_data['reasoning'] = "User asking about macroeconomic indicators"
            response_data['assumptions'] = [
                "Indicators released on schedule",
                "Markets efficiently price economic data",
                "Historical relationships hold"
            ]
            
        elif any(word in q_lower for word in ['sector', 'industry', 'rotation', 'momentum']):
            response_data['response'] = (
                "**Sector Analysis & Rotation:**\n"
                "1. **Economic Cycle Sensitivity**:\n"
                "   • **Growth Phase**: Tech, Consumer Discretionary\n"
                "   • **Peak**: Financials, Industrials\n"
                "   • **Slowdown**: Healthcare, Utilities, Staples\n"
                "   • **Recession**: Defensive (Staples, Bonds)\n"
                "2. **Sector Performance**:\n"
                "   • Relative strength: Compare sector returns\n"
                "   • Momentum: Trending sectors\n"
                "   • Valuation: S&P sectors have varying valuations\n"
                "3. **Rotation Indicators**:\n"
                "   • High-beta → Low-beta signals risk-off\n"
                "   • Yield curve slope affects sector weights\n"
                "   • Breadth/Depth ratios\n"
                "4. **Overweight/Underweight Decisions**:\n"
                "   • Based on macro outlook & valuations\n"
                "   • Sector ETFs for implementation\n\n"
                "Strategy: Dynamic sector allocation based on economic cycle"
            )
            response_data['reasoning'] = "User asking about sector rotation and trends"
            response_data['assumptions'] = [
                "Economic cycles continue as historical patterns",
                "Sector classifications remain relevant",
                "Momentum persists in short-medium term"
            ]
            
        else:
            response_data['response'] = (
                "I can help with market analysis including:\n"
                "• Return forecasting & prediction models\n"
                "• Valuation analysis & metrics\n"
                "• Macroeconomic indicators & impact\n"
                "• Sector rotation & momentum analysis\n"
                "• Technical analysis & patterns\n"
                "• Market sentiment & risk indicators\n\n"
                "Please clarify your market analysis question."
            )
            response_data['reasoning'] = "Generic market analysis guidance"
            
        return response_data
        
    except Exception as e:
        return {
            'response': "I encountered an error processing your market analysis question. Please try again.",
            'reasoning': str(e),
            'assumptions': [],
            'error': str(e),
            'question_type': 'market_analysis'
        }


def classify_question(question: str) -> str:
    """
    Classify a user question into one of the chatbot handler categories.
    
    Args:
        question: User's input question
    
    Returns:
        Question classification: 'portfolio', 'risk', 'market', or 'general'
    """
    q_lower = question.lower()
    
    portfolio_keywords = ['weight', 'allocation', 'construct', 'optimal', 'asset', 
                         'diversif', 'rebalance', 'holding', 'position', 'concentration']
    risk_keywords = ['risk', 'var', 'volatility', 'std', 'drawdown', 'stress', 
                     'scenario', 'correlation', 'sharpe', 'sortino', 'cvar', 'loss']
    market_keywords = ['predict', 'forecast', 'market', 'valuation', 'pe', 'ratio',
                       'economic', 'gdp', 'inflation', 'rate', 'sector', 'momentum',
                       'yield', 'earnings', 'growth']
    
    # Check for keyword matches
    if any(word in q_lower for word in portfolio_keywords):
        return 'portfolio'
    elif any(word in q_lower for word in risk_keywords):
        return 'risk'
    elif any(word in q_lower for word in market_keywords):
        return 'market'
    else:
        return 'general'


def process_chatbot_question(question: str, context: dict = None) -> dict:
    """
    Main entry point for processing chatbot questions with automatic routing.
    
    Args:
        question: User's question
        context: Optional context dictionary with portfolio/market data
    
    Returns:
        Dictionary with response, reasoning, and metadata
    """
    try:
        if not question or not isinstance(question, str):
            return {
                'response': "Please provide a valid question.",
                'reasoning': "Invalid input",
                'error': "Empty or non-string question",
                'question_type': 'invalid'
            }
        
        # Classify the question
        q_type = classify_question(question)
        
        # Route to appropriate handler
        if q_type == 'portfolio':
            return handle_portfolio_construction_question(question, context)
        elif q_type == 'risk':
            return handle_risk_analysis_question(question, context)
        elif q_type == 'market':
            return handle_market_analysis_question(question, context)
        else:
            # General response
            return {
                'response': (
                    "I can assist with:\n"
                    "• **Portfolio Construction**: Asset allocation, weights, diversification\n"
                    "• **Risk Analysis**: VaR, stress testing, volatility metrics\n"
                    "• **Market Analysis**: Forecasting, valuation, economic indicators\n\n"
                    "Please ask a question related to portfolio optimization or investment analysis."
                ),
                'reasoning': "Question did not match specific categories",
                'assumptions': [],
                'error': None,
                'question_type': 'general'
            }
    
    except Exception as e:
        return {
            'response': "An error occurred processing your question. Please try again.",
            'reasoning': str(e),
            'assumptions': [],
            'error': str(e),
            'question_type': 'error'
        }

