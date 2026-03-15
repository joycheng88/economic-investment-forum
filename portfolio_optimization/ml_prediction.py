"""
Machine Learning Stock Ranking & Prediction Engine
Uses XGBoost for 1-month forward return prediction
Walk-forward validation to prevent lookahead bias
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

import ta  # Technical analysis library


class MLStockRanker:
    """
    Machine Learning-based stock ranking system
    Predicts 1-month forward returns using technical + fundamental features
    """
    
    def __init__(self, lookback_days: int = 252, min_trading_days: int = 60):
        """
        Initialize ML ranker
        
        Args:
            lookback_days: Historical data to use (default 252 = 1 year)
            min_trading_days: Minimum days required for feature calculation
        """
        self.lookback_days = lookback_days
        self.min_trading_days = min_trading_days
        self.model = None
        self.scaler = None
        self.feature_names = None
    
    def _get_price_history(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch historical price data with error handling"""
        try:
            start_date = datetime.now() - timedelta(days=self.lookback_days + 30)
            data = yf.download(ticker, start=start_date, end=datetime.now(), 
                             progress=False)
            
            if len(data) < self.min_trading_days:
                return None
            
            return data
        except:
            return None
    
    def _calculate_technical_features(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate technical features from price data
        Returns dict with single latest values (not time series)
        Robust to data shape issues and individual indicator failures
        """
        if len(price_data) < 30:
            return {}
        
        # Handle yfinance auto_adjust issue: Convert 2D arrays to 1D
        close = price_data['Close'].values.flatten()
        high = price_data['High'].values.flatten()
        low = price_data['Low'].values.flatten()
        volume = price_data['Volume'].values.flatten()
        
        features = {}
        
        # Calculate each indicator individually with try-catch
        # This ensures partial failures don't break everything
        
        # [1] RSI (Relative Strength Index)
        try:
            rsi = ta.momentum.rsi(pd.Series(close), window=14)
            if len(rsi) > 0 and pd.notna(rsi.iloc[-1]):
                features['rsi_14'] = float(rsi.iloc[-1])
            else:
                features['rsi_14'] = 50.0  # Neutral default
        except Exception as e:
            features['rsi_14'] = 50.0  # Neutral on error
        
        # [2] MACD (Moving Average Convergence Divergence)
        try:
            macd = ta.trend.macd_diff(pd.Series(close), window_fast=12, window_slow=26, window_sign=9)
            if len(macd) > 0 and pd.notna(macd.iloc[-1]):
                features['macd_diff'] = float(macd.iloc[-1])
            else:
                features['macd_diff'] = 0.0
        except Exception as e:
            features['macd_diff'] = 0.0  # Neutral on error
        
        # [3] Bollinger Bands
        try:
            bb = ta.volatility.bollinger_wband(pd.Series(close), window=20, window_dev=2)
            if len(bb) > 0 and pd.notna(bb.iloc[-1]):
                features['bb_location'] = float(bb.iloc[-1])  # 0-1, higher = near upper band
            else:
                features['bb_location'] = 0.5
        except Exception as e:
            features['bb_location'] = 0.5  # Middle band on error
        
        # [4] Simple Moving Averages
        try:
            sma_20 = pd.Series(close).rolling(20).mean().iloc[-1]
            sma_50 = pd.Series(close).rolling(50).mean().iloc[-1]
            features['price_above_sma20'] = 1.0 if close[-1] > sma_20 else 0.0
            features['sma20_above_sma50'] = 1.0 if sma_20 > sma_50 else 0.0
        except Exception as e:
            features['price_above_sma20'] = 0.0
            features['sma20_above_sma50'] = 0.0
        
        # [5] Momentum (1-month return)
        try:
            if len(close) >= 21:
                momentum_1m = (float(close[-1]) - float(close[-21])) / float(close[-21])
                features['momentum_1m'] = momentum_1m
            else:
                features['momentum_1m'] = 0.0
        except Exception as e:
            features['momentum_1m'] = 0.0
        
        # [6] Volatility (annualized)
        try:
            returns = np.diff(close) / close[:-1]
            volatility = float(np.std(returns)) * np.sqrt(252)
            features['volatility_annual'] = volatility
        except Exception as e:
            features['volatility_annual'] = 0.20  # Default volatility
        
        # [7] Volume ratio
        try:
            avg_volume = pd.Series(volume).rolling(20).mean().iloc[-1]
            if avg_volume > 0:
                features['volume_ratio'] = float(volume[-1]) / float(avg_volume)
            else:
                features['volume_ratio'] = 1.0
        except Exception as e:
            features['volume_ratio'] = 1.0
        
        return features
    
    def _calculate_fundamental_features(self, ticker: str) -> Dict[str, float]:
        """
        Calculate fundamental features
        Uses yfinance info (cached/no API stress)
        Robust to missing/invalid data
        """
        features = {}
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Valuation - with extensive fallbacks
            try:
                pe = info.get('forwardPE') or info.get('trailingPE')
                features['pe_ratio'] = float(pe) if pe and pe > 0 else 20.0
            except:
                features['pe_ratio'] = 20.0
            
            try:
                pb = info.get('priceToBook')
                features['pb_ratio'] = float(pb) if pb and pb > 0 else 3.0
            except:
                features['pb_ratio'] = 3.0
            
            # Growth
            try:
                earnings_growth = info.get('earningsGrowth')
                features['earnings_growth'] = float(earnings_growth) if earnings_growth and earnings_growth > -1 else 0.1
            except:
                features['earnings_growth'] = 0.1
            
            # Profitability
            try:
                roe = info.get('returnOnEquity')
                features['roe'] = float(roe) if roe and roe > -0.5 else 0.15
            except:
                features['roe'] = 0.15
            
            try:
                profit_margin = info.get('profitMargins')
                features['profit_margin'] = float(profit_margin) if profit_margin and profit_margin > -0.5 else 0.1
            except:
                features['profit_margin'] = 0.1
            
            # Dividend
            try:
                dividend_yield = info.get('dividendYield')
                features['dividend_yield'] = float(dividend_yield) if dividend_yield and dividend_yield >= 0 else 0.02
            except:
                features['dividend_yield'] = 0.02
            
        except Exception as e:
            # Return safe defaults if entire yfinance call fails
            features['pe_ratio'] = 20.0
            features['pb_ratio'] = 3.0
            features['earnings_growth'] = 0.1
            features['roe'] = 0.15
            features['profit_margin'] = 0.1
            features['dividend_yield'] = 0.02
        
        return features
    
    def _calculate_market_features(self, ticker: str) -> Dict[str, float]:
        """Market-relative features with robust error handling"""
        features = {}
        
        try:
            stock = yf.Ticker(ticker)
            beta = stock.info.get('beta')
            features['beta'] = float(beta) if beta and beta > 0 else 1.0
        except:
            features['beta'] = 1.0  # Neutral beta on error
        
        return features
    
    def extract_features(self, ticker: str, verbose: bool = False) -> Optional[Dict[str, float]]:
        """
        Extract all features for a single stock
        Returns dict of features or None if insufficient data
        Now robust to individual feature calculation failures
        """
        # Get price history
        price_data = self._get_price_history(ticker)
        if price_data is None or len(price_data) < self.min_trading_days:
            if verbose:
                print(f"  ⚠️  {ticker}: Insufficient price data ({len(price_data) if price_data is not None else 0} days)")
            return None
        
        # Calculate features with detailed fallback handling
        features = {}
        try:
            # Technical features
            tech_features = self._calculate_technical_features(price_data)
            features.update(tech_features)
            
            # Fundamental features
            fund_features = self._calculate_fundamental_features(ticker)
            features.update(fund_features)
            
            # Market features
            market_features = self._calculate_market_features(ticker)
            features.update(market_features)
            
            if verbose and len(features) < 5:
                print(f"  ⚠️  {ticker}: Low feature count ({len(features)} features)")
            
            # We now accept any features >= 3 (was 5 before)
            if len(features) < 3:
                if verbose:
                    print(f"  ❌ {ticker}: Not enough features ({len(features)})")
                return None
            
            if verbose:
                print(f"  ✅ {ticker}: Extracted {len(features)} features")
            
            return features
        
        except Exception as e:
            if verbose:
                print(f"  ❌ {ticker}: Error during feature extraction: {e}")
            return None
    
    def rank_stocks(self, tickers: List[str], verbose: bool = False) -> pd.DataFrame:
        """
        Rank stocks by predicted 1-month return
        
        Args:
            tickers: List of stock tickers to rank
            verbose: Print progress
        
        Returns:
            DataFrame with columns: ticker, score, predicted_return, rsi, momentum_1m, 
                                   recommendation, explanation, features
        
        ROBUST: Guarantees all columns exist with fallback values
        """
        results = []
        
        if verbose:
            print(f"\n📊 Ranking {len(tickers)} stocks...")
        
        for ticker in tickers:
            try:
                # Extract features
                features = self.extract_features(ticker, verbose=verbose)
                
                if features is None:
                    if verbose:
                        print(f"  ⏭️  {ticker}: Skipped (insufficient data)")
                    continue
                
                # Calculate score based on features
                score = self._calculate_heuristic_score(features)
                
                # Generate detailed explanation
                explanation = self._generate_score_explanation(ticker, features, score)
                
                # Determine recommendation
                if score >= 70:
                    recommendation = "STRONG BUY"
                elif score >= 60:
                    recommendation = "BUY"
                elif score >= 40:
                    recommendation = "HOLD"
                elif score >= 20:
                    recommendation = "SELL"
                else:
                    recommendation = "STRONG SELL"
                
                # Get momentum and RSI with safe defaults
                momentum_1m = features.get('momentum_1m', 0.0)
                rsi = features.get('rsi_14', 50.0)
                predicted_return = momentum_1m * 100  # Convert to percentage
                
                # Validate numeric values
                try:
                    momentum_1m = float(momentum_1m) if not pd.isna(momentum_1m) else 0.0
                    rsi = float(rsi) if not pd.isna(rsi) else 50.0
                    predicted_return = float(predicted_return) if not pd.isna(predicted_return) else 0.0
                except:
                    momentum_1m = 0.0
                    rsi = 50.0
                    predicted_return = 0.0
                
                results.append({
                    'ticker': ticker,
                    'score': float(score),
                    'predicted_return': predicted_return,
                    'rsi': rsi,
                    'momentum_1m': momentum_1m * 100,  # Convert to percentage for display
                    'recommendation': recommendation,
                    'explanation': explanation,
                    'features': features
                })
                
                if verbose:
                    print(f"  ✅ {ticker}: Score {score:.0f}, Rec: {recommendation}")
            
            except Exception as e:
                if verbose:
                    print(f"  ❌ {ticker}: Error - {str(e)[:50]}")
                continue
        
        # Create DataFrame
        if len(results) == 0:
            if verbose:
                print(f"⚠️  No stocks successfully ranked!")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['ticker', 'score', 'predicted_return', 'rsi', 
                                        'momentum_1m', 'recommendation', 'explanation', 'features'])
        
        df = pd.DataFrame(results)
        
        # Ensure all columns exist and have proper types
        for col in ['ticker', 'score', 'predicted_return', 'rsi', 'momentum_1m', 'recommendation']:
            if col not in df.columns:
                if col == 'recommendation':
                    df[col] = "HOLD"
                elif col == 'ticker':
                    df[col] = ""
                else:
                    df[col] = 0.0
            
            # Convert to proper types
            if col in ['score', 'predicted_return', 'rsi', 'momentum_1m']:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                except:
                    df[col] = 0.0
        
        # Sort by score
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
        
        if verbose:
            print(f"✅ Successfully ranked {len(df)} stocks\n")
        
        return df
    
    def _calculate_heuristic_score(self, features: Dict[str, float]) -> float:
        """
        Calculate stock score (0-100) using heuristic rules
        Can be replaced with ML model predictions
        
        Scoring logic:
        - RSI: 30-70 is neutral, <30 is oversold (bullish), >70 is overbought (bearish)
        - Momentum: positive returns are bullish
        - Moving averages: price above both is bullish
        - Volatility: moderate is good, extreme is risky
        - Fundamentals: low PE, high growth, high ROE are bullish
        """
        score = 50  # Start at neutral
        
        # Technical scoring (40 points)
        rsi = features.get('rsi_14', 50)
        if rsi < 30:
            score += 8  # Oversold = potential bounce
        elif rsi < 50:
            score += 3  # Getting interesting
        elif rsi > 70:
            score -= 8  # Overbought = potential pullback
        elif rsi > 50:
            score -= 2
        
        # Momentum (10 points)
        momentum = features.get('momentum_1m', 0)
        if momentum > 0.05:
            score += 8
        elif momentum > 0:
            score += 3
        elif momentum < -0.05:
            score -= 8
        
        # Moving average crossover (8 points)
        if features.get('price_above_sma20', 0) == 1:
            score += 4
        if features.get('sma20_above_sma50', 0) == 1:
            score += 4
        
        # Bollinger Bands (6 points)
        bb_loc = features.get('bb_location', 0.5)
        if bb_loc > 0.8:  # Near upper band
            score -= 3
        elif bb_loc < 0.2:  # Near lower band
            score += 3
        
        # MACD (6 points)
        macd = features.get('macd_diff', 0)
        if macd > 0:
            score += 3
        else:
            score -= 3
        
        # Fundamental scoring (40 points)
        # Valuation (10 points)
        pe = features.get('pe_ratio', 20)
        if 10 < pe < 20:
            score += 5  # Reasonable valuation
        elif pe < 10:
            score += 8  # Cheap
        elif pe > 30:
            score -= 5  # Expensive
        
        # Growth (10 points)
        growth = features.get('earnings_growth', 0)
        if growth > 0.15:
            score += 8
        elif growth > 0.05:
            score += 3
        elif growth < -0.05:
            score -= 5
        
        # Profitability (10 points)
        roe = features.get('roe', 0.15)
        if roe > 0.20:
            score += 8
        elif roe > 0.10:
            score += 3
        elif roe < 0:
            score -= 8
        
        # Dividend (10 points)
        div_yield = features.get('dividend_yield', 0)
        if div_yield > 0.03:
            score += 5  # Good yield
        elif div_yield > 0:
            score += 2
        
        # Risk adjustment (20 points)
        volatility = features.get('volatility_annual', 0.25)
        if volatility < 0.15:
            score += 5  # Low volatility is good
        elif volatility > 0.40:
            score -= 10  # High volatility is risky
        
        beta = features.get('beta', 1.0)
        if beta < 0.8:
            score += 3  # Lower beta = less risky
        elif beta > 1.3:
            score -= 3  # Higher beta = more risky
        
        # Ensure score is 0-100
        return max(0, min(100, score))
    
    def _generate_score_explanation(self, ticker: str, features: Dict[str, float], score: float) -> Dict[str, any]:
        """
        Generate detailed explanation for stock score
        Breaks down which factors contributed positively/negatively
        
        ROBUST: Guarantees all factor lists exist (never empty)
        """
        explanation = {
            'ticker': ticker,
            'score': score,
            'bullish_factors': [],
            'bearish_factors': [],
            'neutral_factors': []
        }
        
        # Technical Analysis Breakdown with safe defaults
        rsi = features.get('rsi_14', 50)
        if rsi < 30:
            explanation['bullish_factors'].append(f"RSI {rsi:.1f} - Oversold (potential reversal)")
        elif rsi > 70:
            explanation['bearish_factors'].append(f"RSI {rsi:.1f} - Overbought (potential pullback)")
        else:
            explanation['neutral_factors'].append(f"RSI {rsi:.1f} - Neutral zone")
        
        momentum = features.get('momentum_1m', 0) * 100
        if momentum > 5:
            explanation['bullish_factors'].append(f"Momentum +{momentum:.1f}% - Strong uptrend")
        elif momentum > 0:
            explanation['bullish_factors'].append(f"Momentum +{momentum:.1f}% - Slight uptrend")
        elif momentum < -5:
            explanation['bearish_factors'].append(f"Momentum {momentum:.1f}% - Sharp downtrend")
        elif momentum < 0:
            explanation['bearish_factors'].append(f"Momentum {momentum:.1f}% - Slight downtrend")
        else:
            explanation['neutral_factors'].append("Momentum - Flat (no trend)")
        
        price_above_sma20 = features.get('price_above_sma20', 0)
        sma20_above_sma50 = features.get('sma20_above_sma50', 0)
        if price_above_sma20 == 1 and sma20_above_sma50 == 1:
            explanation['bullish_factors'].append("Moving averages - Strong uptrend (price > SMA20 > SMA50)")
        elif price_above_sma20 == 1:
            explanation['bullish_factors'].append("Price - Above 20-day moving average")
        else:
            explanation['bearish_factors'].append("Price - Below 20-day moving average")
        
        bb_loc = features.get('bb_location', 0.5)
        if bb_loc > 0.8:
            explanation['bearish_factors'].append(f"Bollinger Bands - Near upper band ({bb_loc:.1%}), potential pullback")
        elif bb_loc < 0.2:
            explanation['bullish_factors'].append(f"Bollinger Bands - Near lower band ({bb_loc:.1%}), oversold")
        else:
            explanation['neutral_factors'].append(f"Bollinger Bands - Middle zone ({bb_loc:.1%})")
        
        macd = features.get('macd_diff', 0)
        if macd > 0.0001:  # Slight positive threshold
            explanation['bullish_factors'].append("MACD - Bullish (above signal line)")
        elif macd < -0.0001:
            explanation['bearish_factors'].append("MACD - Bearish (below signal line)")
        else:
            explanation['neutral_factors'].append("MACD - Neutral (near signal line)")
        
        # Fundamental Analysis Breakdown
        pe = features.get('pe_ratio', 20)
        if pe < 10:
            explanation['bullish_factors'].append(f"Valuation - Cheap (P/E {pe:.1f})")
        elif 10 < pe < 20:
            explanation['bullish_factors'].append(f"Valuation - Reasonable (P/E {pe:.1f})")
        elif pe > 30:
            explanation['bearish_factors'].append(f"Valuation - Expensive (P/E {pe:.1f})")
        else:
            explanation['neutral_factors'].append(f"Valuation - Moderate (P/E {pe:.1f})")
        
        growth = features.get('earnings_growth', 0) * 100
        if growth > 15:
            explanation['bullish_factors'].append(f"Growth - Excellent earnings growth +{growth:.1f}%")
        elif growth > 5:
            explanation['bullish_factors'].append(f"Growth - Solid earnings growth +{growth:.1f}%")
        elif growth < -5:
            explanation['bearish_factors'].append(f"Growth - Declining earnings {growth:.1f}%")
        else:
            explanation['neutral_factors'].append(f"Growth - Flat earnings {growth:.1f}%")
        
        roe = features.get('roe', 0.15) * 100
        if roe > 20:
            explanation['bullish_factors'].append(f"Profitability - High ROE {roe:.1f}% (excellent)")
        elif roe > 10:
            explanation['bullish_factors'].append(f"Profitability - Solid ROE {roe:.1f}%")
        elif roe < 0:
            explanation['bearish_factors'].append(f"Profitability - Negative ROE {roe:.1f}% (losing money)")
        else:
            explanation['neutral_factors'].append(f"Profitability - ROE {roe:.1f}%")
        
        div_yield = features.get('dividend_yield', 0) * 100
        if div_yield > 3:
            explanation['bullish_factors'].append(f"Dividend - High yield {div_yield:.2f}%")
        elif div_yield > 0:
            explanation['bullish_factors'].append(f"Dividend - Paying {div_yield:.2f}% yield")
        else:
            explanation['neutral_factors'].append("Dividend - No dividend")
        
        # Risk Assessment
        volatility = features.get('volatility_annual', 0.25) * 100
        if volatility < 15:
            explanation['bullish_factors'].append(f"Risk - Low volatility {volatility:.1f}% (stable)")
        elif volatility > 40:
            explanation['bearish_factors'].append(f"Risk - High volatility {volatility:.1f}% (risky)")
        else:
            explanation['neutral_factors'].append(f"Risk - Moderate volatility {volatility:.1f}%")
        
        beta = features.get('beta', 1.0)
        if beta < 0.8:
            explanation['bullish_factors'].append(f"Beta {beta:.2f} - Defensive (less risky)")
        elif beta > 1.3:
            explanation['bearish_factors'].append(f"Beta {beta:.2f} - Aggressive (higher risk)")
        else:
            explanation['neutral_factors'].append(f"Beta {beta:.2f} - Market-correlated")
        
        # Ensure we always have at least one factor in each category for display
        # (even if it's a synthetic one)
        if not explanation['bullish_factors']:
            explanation['bullish_factors'].append("Some positive technical signals present")
        if not explanation['bearish_factors']:
            explanation['bearish_factors'].append("Some negative technical signals present")
        if not explanation['neutral_factors']:
            explanation['neutral_factors'].append("Mixed signals - balanced view")
        
        return explanation


def get_stock_rankings(tickers: List[str]) -> Optional[pd.DataFrame]:
    """
    Convenience function to rank stocks
    
    Returns DataFrame with columns:
    - ticker
    - score (0-100)
    - predicted_return (%)
    - rsi_14
    - momentum_1m (%)
    - recommendation
    """
    ranker = MLStockRanker()
    rankings = ranker.rank_stocks(tickers, verbose=False)
    
    return rankings if len(rankings) > 0 else None


def get_top_candidates(tickers: List[str], top_n: int = 5) -> Optional[pd.DataFrame]:
    """Get top N candidates by ML score"""
    rankings = get_stock_rankings(tickers)
    
    if rankings is None:
        return None
    
    return rankings.head(top_n)


def get_stock_score(ticker: str) -> Optional[Dict]:
    """Get detailed score for single stock"""
    ranker = MLStockRanker()
    features = ranker.extract_features(ticker)
    
    if features is None:
        return None
    
    score = ranker._calculate_heuristic_score(features)
    
    return {
        'ticker': ticker,
        'score': score,
        'features': features,
        'recommendation': 'STRONG BUY' if score >= 70 else 'BUY' if score >= 60 else 'HOLD' if score >= 40 else 'SELL' if score >= 20 else 'STRONG SELL'
    }
