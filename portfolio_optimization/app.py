"""
Streamlit Interface for Portfolio Optimization Model Comparison

Multi-page app with:
1. Home: Display all 28 stocks with model comparison
2. Description: Model descriptions and assumptions
3. Portfolio Builder: Interactive add/remove stocks
4. Backtest: Historical performance comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import logging
import urllib3
import os

# Suppress all warnings and debug messages
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('requests').setLevel(logging.CRITICAL)
logging.getLogger('requests.packages.urllib3').setLevel(logging.CRITICAL)
urllib3.disable_warnings()
# Disable requests HTTPConnection verbose logging
logging.getLogger('urllib3.connectionpool').setLevel(logging.CRITICAL)

from data import load_data, DataConfig, get_risk_free_rate, validate_ticker
from model.gmv import estimate_covariance as gmv_cov, gmv_weights, portfolio_volatility
from model.capm import get_capm_weights, CAPMConfig
from model.bl import get_bl_weights, BLConfig
from model.hrp import get_hrp_weights, HRPConfig
from model.cvar import get_cvar_weights
from model.lasso import get_lasso_weights
from model.rl import get_rl_weights
from model.dro import get_dro_weights
from backtest import run_rolling_backtest
from sectors import get_sectors, get_stocks_by_sector, get_sector_allocation, fetch_sector_from_yfinance, update_custom_sector
from holdings import portfolio_factor_exposure, get_benchmark_exposures
from llm import (
    get_portfolio_composition_analysis,
    generate_investment_recommendation,
    find_complementary_stocks,
    generate_market_context_summary,
    create_portfolio_narrative,
    estimate_portfolio_return
)
from chatbot import analyze_investment_decision, extract_ticker_from_question, handle_user_question
from modeling import DCFValuation, ComparableCompanies, ValuationReport

# Page config
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = [
        "SHV", "BRK-B", "JNJ", "MKL", "IBN", "UNH", "MSFT", "HDB",
        "TSM", "HCC", "TDW", "KWEB", "GOOG", "IBKR", "NVDA", "COIN",
        "BN", "SHOP", "META", "UBER", "ZETA", "AMR", "HOOD", "AMZN",
        "VAL", "NE", "JPM", "EFX"
    ]
if 'custom_sector_mapping' not in st.session_state:
    st.session_state.custom_sector_mapping = {}  # Store sector assignments for custom tickers

# All available stocks (28)
ALL_STOCKS = [
    "SHV", "BRK-B", "JNJ", "MKL", "IBN", "UNH", "MSFT", "HDB",
    "TSM", "HCC", "TDW", "KWEB", "GOOG", "IBKR", "NVDA", "COIN",
    "BN", "SHOP", "META", "UBER", "ZETA", "AMR", "HOOD", "AMZN",
    "VAL", "NE", "JPM", "EFX"
]

# Cache risk-free rate to avoid repeated fetches
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_risk_free_rate():
    """Fetch real-time US Treasury 10-year yield as risk-free rate"""
    rate = get_risk_free_rate()
    return rate

# Get risk-free rate for use throughout app
RISK_FREE_RATE = fetch_risk_free_rate()

# Add Streamlit-level caching for valuation results (OPTIMIZATION)
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def run_dcf_cached(ticker, wacc, terminal_growth, forecast_years, risk_free_rate, 
                   market_risk_premium, tax_rate, cost_of_debt, revenue_growth=None):
    """Cached DCF analysis to avoid redundant calculations"""
    from valuation import dcf_valuation
    return dcf_valuation(
        ticker=ticker,
        wacc=wacc,
        terminal_growth_rate=terminal_growth,
        forecast_years=forecast_years,
        risk_free_rate=risk_free_rate,
        market_risk_premium=market_risk_premium,
        tax_rate=tax_rate,
        cost_of_debt=cost_of_debt,
        revenue_growth_rate=revenue_growth
    )

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def run_comps_cached(ticker, auto_find_peers=True, peer_tickers=None):
    """Cached comps analysis to avoid redundant peer screening"""
    from valuation import comparable_companies_analysis
    return comparable_companies_analysis(ticker=ticker, peer_tickers=peer_tickers, auto_find_peers=auto_find_peers)

def load_portfolio_data(tickers, start_date, end_date):
    """Load data from yfinance"""
    if not tickers:
        raise ValueError("No tickers provided. Please add at least one ticker.")
    
    # Ensure dates are comparable (convert datetime to date if needed)
    from datetime import datetime
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()
    
    # Validation: no future dates
    today = datetime.now().date()
    if end_date > today:
        raise ValueError(f"❌ End date cannot be in the future. Today is {today}. Please select {today} or earlier.")
    if start_date > today:
        raise ValueError(f"❌ Start date cannot be in the future. Today is {today}. Please select {today} or earlier.")
    
    # Validation: start must be before end
    if start_date >= end_date:
        raise ValueError(f"❌ Start date ({start_date}) must be earlier than end date ({end_date}).")

    with st.spinner('Loading data from Yahoo Finance...'):
        cfg = DataConfig(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            return_type="log",
            min_non_na_ratio=0.60  # Allows mixed asset types (stocks ~68%, crypto 100%)
        )
        prices, returns = load_data(tickers, cfg)
        if returns is None or returns.empty:
            raise ValueError("No valid return data loaded for selected tickers/date range.")
        return prices, returns

def run_portfolio_optimization(returns, max_weight=0.15, constraints=None):
    """
    Run all 8 models and return results
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns
    max_weight : float
        Maximum weight per asset
    constraints : dict, optional
        Portfolio constraints:
        - 'sector_caps': dict mapping sector -> max weight
        - 'min_weight': minimum weight for non-zero holdings
        - 'max_holdings': maximum number of non-zero holdings
    """
    if returns is None or returns.empty:
        raise ValueError("Input returns are empty.")
    
    cov = gmv_cov(returns)
    results = {}
    
    def apply_constraints(weights, ticker_list):
        """Apply optional constraints to portfolio weights"""
        if constraints is None or not constraints:
            return weights
        
        w = weights.copy()
        
        # Apply max holdings constraint (zero out smallest positions)
        if 'max_holdings' in constraints:
            max_h = constraints['max_holdings']
            nonzero = (w > 1e-4).sum()
            if nonzero > max_h:
                # Keep top max_h holdings, zero out rest
                sorted_idx = w.abs().argsort(ascending=False)
                keep_idx = sorted_idx[:max_h]
                zero_idx = sorted_idx[max_h:]
                w.iloc[zero_idx] = 0.0
                w = w / w.sum()  # Renormalize
        
        # Apply min weight constraint (if weight > 0, ensure >= min_weight)
        if 'min_weight' in constraints:
            min_w = constraints['min_weight']
            small = (0 < w) & (w < min_w)
            if small.any():
                w[small] = min_w
                w = w / w.sum()  # Renormalize
        
        # Apply sector caps (reduce over-cap holdings proportionally)
        if 'sector_caps' in constraints:
            from sectors import get_sector_for_stock
            sector_caps = constraints['sector_caps']
            sector_weights = {}
            
            for ticker in w.index:
                sector = get_sector_for_stock(ticker)
                if sector not in sector_weights:
                    sector_weights[sector] = []
                sector_weights[sector].append(ticker)
            
            for sector, tickers in sector_weights.items():
                cap = sector_caps.get(sector, 1.0)
                current_weight = w.loc[tickers].sum()
                if current_weight > cap:
                    # Reduce proportionally
                    reduction = (current_weight - cap) / current_weight
                    w.loc[tickers] = w.loc[tickers] * (1 - reduction)
                    w = w / w.sum()  # Renormalize
        
        return w
    
    # GMV
    try:
        cfg_gmv = type('C', (), {'long_only': True, 'max_weight': max_weight, 'solver': 'OSQP'})()
        w_gmv = gmv_weights(cov, cfg_gmv)
        w_gmv = apply_constraints(w_gmv, returns.columns)
        vol_gmv = portfolio_volatility(w_gmv, cov)
        results['GMV'] = {'weights': w_gmv, 'volatility': vol_gmv}
    except Exception as e:
        st.warning(f"GMV error: {e}")
    
    # CAPM
    try:
        cfg_capm = CAPMConfig(model_type="capm", risk_aversion=2.0, risk_free_rate=RISK_FREE_RATE, long_only=True, max_weight=max_weight)
        w_capm = get_capm_weights(returns, cfg_capm)
        w_capm = apply_constraints(w_capm, returns.columns)
        vol_capm = portfolio_volatility(w_capm, cov)
        results['CAPM'] = {'weights': w_capm, 'volatility': vol_capm}
    except Exception as e:
        st.warning(f"CAPM error: {e}")
    
    # BL
    try:
        cfg_bl = BLConfig(risk_aversion=2.0, tau=0.05, risk_free_rate=RISK_FREE_RATE, long_only=True, max_weight=max_weight, views=[])
        w_bl = get_bl_weights(returns, None, cfg_bl)
        w_bl = apply_constraints(w_bl, returns.columns)
        vol_bl = portfolio_volatility(w_bl, cov)
        results['BL'] = {'weights': w_bl, 'volatility': vol_bl}
    except Exception as e:
        st.warning(f"BL error: {e}")
    
    # HRP
    try:
        cfg_hrp = HRPConfig(linkage_method="ward", long_only=True, max_weight=max_weight)
        w_hrp = get_hrp_weights(returns, cfg_hrp)
        w_hrp = apply_constraints(w_hrp, returns.columns)
        vol_hrp = portfolio_volatility(w_hrp, cov)
        results['HRP'] = {'weights': w_hrp, 'volatility': vol_hrp}
    except Exception as e:
        st.warning(f"HRP error: {e}")
    
    # CVaR
    try:
        w_cvar = get_cvar_weights(returns, max_weight=max_weight, alpha=0.05)
        w_cvar = apply_constraints(w_cvar, returns.columns)
        vol_cvar = portfolio_volatility(w_cvar, cov)
        results['CVaR'] = {'weights': w_cvar, 'volatility': vol_cvar}
    except Exception as e:
        st.warning(f"CVaR error: {e}")
    
    # LASSO
    try:
        w_lasso = get_lasso_weights(returns, max_weight=max_weight, lasso_penalty=0.01, num_assets_target=10)
        w_lasso = apply_constraints(w_lasso, returns.columns)
        vol_lasso = portfolio_volatility(w_lasso, cov)
        results['LASSO'] = {'weights': w_lasso, 'volatility': vol_lasso}
    except Exception as e:
        st.warning(f"LASSO error: {e}")

    # Reinforcement Learning (RL)
    # Note: RL is computationally intensive, but now with improved training
    try:
        # Check if we have a cached RL agent for these tickers
        if 'rl_agent' not in st.session_state or 'rl_tickers' not in st.session_state or st.session_state.rl_tickers != list(returns.columns):
            with st.spinner("Training improved RL agent (this may take 60-90 seconds with better training)..."):
                w_rl = get_rl_weights(returns, agent=None, state_extractor=None, n_epochs=30, max_weight=max_weight, long_only=True)
                # Cache for future use
                st.session_state.rl_tickers = list(returns.columns)
        else:
            # Use cached agent
            w_rl = get_rl_weights(returns, agent=None, state_extractor=None, n_epochs=30, max_weight=max_weight, long_only=True)
        
        w_rl = apply_constraints(w_rl, returns.columns)
        vol_rl = portfolio_volatility(w_rl, cov)
        results['RL'] = {'weights': w_rl, 'volatility': vol_rl}
    except ImportError:
        st.info("RL model requires PyTorch. Install with: pip install torch")
    except Exception as e:
        st.warning(f"RL error: {e}")

    # Distributionally Robust Optimization (DRO)
    try:
        w_dro = get_dro_weights(
            returns,
            method='mean_variance',
            epsilon=1.0,  # Robust to uncertainty
            risk_aversion=1.0,
            max_weight=max_weight,
            long_only=True
        )
        w_dro = apply_constraints(w_dro, returns.columns)
        vol_dro = portfolio_volatility(w_dro, cov)
        results['DRO'] = {'weights': w_dro, 'volatility': vol_dro}
    except Exception as e:
        st.warning(f"DRO error: {e}")

    if not results:
        raise RuntimeError("All models failed. Try a wider date range or fewer/more liquid assets.")
    
    return results, cov

# ==================== HOME PAGE ====================

def page_home():
    st.title("📊 Portfolio Optimization")
    
    st.markdown("""
    Compare 8 portfolio optimization models with advanced analytics and risk metrics.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start date",
                value=datetime.now() - timedelta(days=365*2),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End date",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        # Max weight
        max_weight = st.slider(
            "Max weight per asset (%)",
            min_value=5,
            max_value=50,
            value=15,
            step=1
        ) / 100
        
        st.info(
            "**📌 Home Page Methodology**: One-time portfolio construction with **60% train / 40% test split**. "
            "Models are optimized on historical training data, then evaluated on held-out test data (last 40%). "
            "No rebalancing—shows realistic buy-and-hold performance. Compare with 📊 **Rolling Backtest** for historical rolling analysis."
        )
        
        # Load button
        if st.button("🔄 Run Analysis", use_container_width=True):
            try:
                prices, returns = load_portfolio_data(ALL_STOCKS, start_date, end_date)
                
                # Split into 60% train / 40% test for realistic out-of-sample performance
                split_idx = int(len(returns) * 0.6)
                train_returns = returns.iloc[:split_idx]
                test_returns = returns.iloc[split_idx:]
                
                # Optimize on training data only (not test data)
                results, cov = run_portfolio_optimization(train_returns, max_weight)
                
                st.session_state.data_loaded = True
                st.session_state.prices = prices
                st.session_state.returns = test_returns  # Store test returns for honest evaluation
                st.session_state.train_returns = train_returns  # Store train returns for reference
                st.session_state.results = results
                st.session_state.cov = cov
                st.session_state.max_weight = max_weight
                
                st.success("✅ Analysis complete! (Out-of-sample evaluation on last 40% of data)")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    if st.session_state.data_loaded:
        required_keys = ['results', 'returns', 'cov', 'prices']
        missing_keys = [k for k in required_keys if k not in st.session_state]
        if missing_keys:
            st.warning("Session data incomplete. Please rerun analysis.")
            st.session_state.data_loaded = False
            return

        st.markdown("---")
        
        results = st.session_state.results
        returns = st.session_state.returns  # This is now test/out-of-sample returns (last 40%)
        cov = st.session_state.cov
        prices = st.session_state.prices
        
        # Show data split info
        if 'train_returns' in st.session_state:
            train_len = len(st.session_state.train_returns)
            test_len = len(returns)
            st.info(f"📊 **Evaluation Methodology**: Models trained on first {train_len} days, evaluated on last {test_len} days (out-of-sample)")
        
        # ===== SECTION 1: SUMMARY METRICS =====
        st.subheader("📈 Performance & Risk Summary (Out-of-Sample)")

        
        summary_data = []
        # Recalculate covariance from TEST data for honest evaluation
        test_cov = gmv_cov(returns, annualize=True)
        
        for model_name, data in results.items():
            weights = data['weights'].reindex(returns.columns, fill_value=0.0)
            
            # Recalculate volatility using TEST period covariance (not training period)
            vol_test = portfolio_volatility(weights, test_cov)
            
            # Calculate additional metrics using test period
            ann_return = returns.mean() @ weights * 252
            sharpe = (ann_return - RISK_FREE_RATE) / vol_test if vol_test > 0 else 0
            herfindahl = (weights ** 2).sum()
            num_holdings = int((weights > 1e-4).sum())
            top_weight = weights.max() * 100
            
            summary_data.append({
                'Model': model_name,
                'Ann. Return': f"{ann_return*100:.2f}%",
                'Volatility': f"{vol_test*100:.2f}%",
                'Sharpe Ratio': f"{sharpe:.3f}",
                'Holdings': num_holdings,
                'HHI': f"{herfindahl:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # ===== SECTION 2: VOLATILITY & SHARPE COMPARISON =====
        st.subheader("📊 Risk-Return Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            vol_data = {}
            return_data = {}
            for m, data in results.items():
                weights = data['weights'].reindex(returns.columns, fill_value=0.0)
                # Use test period volatility for consistent evaluation
                vol_test = portfolio_volatility(weights, test_cov)
                ann_ret = returns.mean() @ weights * 252
                vol_data[m] = vol_test * 100
                return_data[m] = ann_ret * 100
            
            fig_scatter = px.scatter(
                x=list(vol_data.values()),
                y=list(return_data.values()),
                text=list(vol_data.keys()),
                labels={'x': 'Annual Volatility (%)', 'y': 'Annual Return (%)'},
                title='Risk-Return Profile',
                size=[8]*len(vol_data),
                color=list(vol_data.keys())
            )
            fig_scatter.update_traces(
                textposition='top center',
                mode='markers+text',
                hovertemplate='<b>%{text}</b><br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            sharpe_data = {}
            for m, data in results.items():
                weights = data['weights'].reindex(returns.columns, fill_value=0.0)
                # Use test period volatility for consistent Sharpe calculation
                vol_test = portfolio_volatility(weights, test_cov)
                ann_ret = returns.mean() @ weights * 252
                sharpe_data[m] = (ann_ret - RISK_FREE_RATE) / vol_test if vol_test > 0 else 0
            
            colors = ['#636EFA' if s >= 0 else '#EF553B' for s in sharpe_data.values()]
            fig_sharpe = px.bar(
                x=list(sharpe_data.keys()),
                y=list(sharpe_data.values()),
                color=list(sharpe_data.keys()),
                labels={'x': 'Model', 'y': 'Sharpe Ratio'},
                title='Risk-Adjusted Returns (Sharpe Ratio)',
                text=[f'{v:.3f}' for v in sharpe_data.values()]
            )
            fig_sharpe.update_traces(
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Sharpe Ratio: %{y:.3f}<extra></extra>'
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)
        
        st.markdown("---")
        
        # ===== SECTION 3: PORTFOLIO COMPOSITION =====
        st.subheader("🎯 Portfolio Composition Analysis")
        
        # Select model for detailed analysis
        selected_model = st.selectbox(
            "Select model for detailed breakdown:",
            list(results.keys()),
            key="model_selector"
        )
        
        weights = results[selected_model]['weights'].reindex(returns.columns, fill_value=0.0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Waffle chart - proportional grid squares
            top_n = 10
            top_weights = weights.nlargest(top_n)
            other_weight = weights[~weights.index.isin(top_weights.index)].sum()
            
            if other_weight > 1e-4:
                top_weights['Others'] = other_weight
            
            try:
                import matplotlib.pyplot as plt
                from matplotlib.patches import Rectangle
                import matplotlib.patches as mpatches

                normalized_weights = (top_weights.values / top_weights.sum()) * 100
                fig_waf, ax = plt.subplots(figsize=(8, 6))
                colors = plt.cm.viridis(np.linspace(0, 1, len(top_weights)))

                count = 0
                square_size = int(np.ceil(np.sqrt(100)))

                for idx, (ticker, weight) in enumerate(top_weights.items()):
                    num_squares = int(np.round(normalized_weights[idx]))
                    color = colors[idx]

                    for _ in range(num_squares):
                        row = count // square_size
                        col = count % square_size
                        rect = Rectangle((col, square_size - row - 1), 1, 1,
                                         facecolor=color, edgecolor='white', linewidth=1)
                        ax.add_patch(rect)
                        count += 1

                ax.set_xlim(0, square_size)
                ax.set_ylim(0, square_size)
                ax.set_aspect('equal')
                ax.axis('off')
                ax.set_title(f'{selected_model} - Top Holdings (Proportional by Area)', fontsize=14, fontweight='bold', pad=20)

                legend_patches = [
                    mpatches.Patch(facecolor=colors[i], edgecolor='white', label=f'{ticker}: {weight*100:.1f}%')
                    for i, (ticker, weight) in enumerate(top_weights.items())
                ]
                ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
                plt.tight_layout()
                st.pyplot(fig_waf)
            except Exception:
                st.info("Waffle chart unavailable (matplotlib dependency). Showing bar chart instead.")
                fig_fallback = px.bar(
                    x=top_weights.index,
                    y=top_weights.values * 100,
                    labels={'x': 'Asset', 'y': 'Weight (%)'},
                    title=f'{selected_model} - Top Holdings'
                )
                st.plotly_chart(fig_fallback, use_container_width=True)
        
        with col2:
            # Cumulative contribution
            sorted_weights = weights.sort_values(ascending=False)
            cumsum = np.cumsum(sorted_weights.values) * 100
            
            fig_cumul = px.bar(
                x=list(range(1, len(sorted_weights) + 1)),
                y=cumsum,
                labels={'x': 'Number of Assets', 'y': 'Cumulative Weight (%)'},
                title=f'{selected_model} - Cumulative Contribution',
                color=cumsum,
                color_continuous_scale='Viridis'
            )
            fig_cumul.update_traces(
                hovertemplate='<b>Top %{x} Assets</b><br>Cumulative Weight: %{y:.2f}%<extra></extra>'
            )
            fig_cumul.add_hline(y=80, line_dash="dash", line_color="red", 
                               annotation_text="80% threshold")
            st.plotly_chart(fig_cumul, use_container_width=True)
        
        st.markdown("---")
        
        # ===== SECTION 4: CONCENTRATION METRICS =====
        st.subheader("📌 Diversification & Concentration")
        
        col1, col2, col3 = st.columns(3)
        
        concentration_metrics = {}
        for model_name, data in results.items():
            w = data['weights']
            hhi = (w ** 2).sum()  # Herfindahl-Hirschman Index
            std_w = np.std(w)
            entropy = -np.sum(w[w > 1e-4] * np.log(w[w > 1e-4]))
            concentration_metrics[model_name] = {'HHI': hhi, 'Std': std_w, 'Entropy': entropy}
        
        with col1:
            fig_hhi = px.bar(
                x=list(concentration_metrics.keys()),
                y=[v['HHI'] for v in concentration_metrics.values()],
                labels={'x': 'Model', 'y': 'HHI Index'},
                title='Concentration (HHI)\nLower = More Diversified',
                color=[v['HHI'] for v in concentration_metrics.values()],
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig_hhi, use_container_width=True)
        
        with col2:
            fig_std = px.bar(
                x=list(concentration_metrics.keys()),
                y=[v['Std'] for v in concentration_metrics.values()],
                labels={'x': 'Model', 'y': 'Std Dev of Weights'},
                title='Weight Distribution\nStandard Deviation',
                color=[v['Std'] for v in concentration_metrics.values()],
                color_continuous_scale='Plasma'
            )
            st.plotly_chart(fig_std, use_container_width=True)
        
        with col3:
            fig_entropy = px.bar(
                x=list(concentration_metrics.keys()),
                y=[v['Entropy'] for v in concentration_metrics.values()],
                labels={'x': 'Model', 'y': 'Shannon Entropy'},
                title='Diversification (Entropy)\nHigher = More Diversified',
                color=[v['Entropy'] for v in concentration_metrics.values()],
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_entropy, use_container_width=True)
        
        st.markdown("---")
        
        # ===== SECTION 5: BETA & SYSTEMATIC RISK =====
        st.subheader("🎲 Beta Analysis - Systematic Risk")
        
        # Calculate market returns (equally-weighted)
        market_returns = returns.mean(axis=1)
        
        beta_data = {}
        for model_name, data in results.items():
            weights = data['weights']
            aligned_weights = weights[returns.columns]
            
            # Portfolio returns
            port_ret = returns @ aligned_weights
            
            # Beta = Cov(Rp, Rm) / Var(Rm)
            covariance = np.cov(port_ret, market_returns)[0, 1]
            market_var = np.var(market_returns)
            beta = covariance / market_var if market_var > 0 else 0
            
            # Correlation with market
            correlation = np.corrcoef(port_ret, market_returns)[0, 1]
            
            beta_data[model_name] = {'Beta': beta, 'Correlation': correlation}
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_beta = px.bar(
                x=list(beta_data.keys()),
                y=[v['Beta'] for v in beta_data.values()],
                labels={'x': 'Model', 'y': 'Beta'},
                title='Portfolio Beta vs Market\nBeta > 1: More volatile than market',
                color=[v['Beta'] for v in beta_data.values()],
                color_continuous_scale='RdBu_r'
            )
            fig_beta.add_hline(y=1.0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_beta, use_container_width=True)
        
        with col2:
            fig_corr = px.bar(
                x=list(beta_data.keys()),
                y=[v['Correlation'] for v in beta_data.values()],
                labels={'x': 'Model', 'y': 'Correlation with Market'},
                title='Market Correlation',
                color=[v['Correlation'] for v in beta_data.values()],
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("---")
        
        # ===== SECTION 6: RETURN DISTRIBUTION ANALYSIS =====
        st.subheader("📊 Return Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot of returns for each model
            box_data = []
            for model_name, data in results.items():
                weights = data['weights']
                aligned_weights = weights[returns.columns]
                port_ret = (returns @ aligned_weights) * 100  # Convert to %
                for ret in port_ret:
                    box_data.append({'Model': model_name, 'Daily Return (%)': ret})
            
            box_df = pd.DataFrame(box_data)
            fig_box = px.box(
                box_df,
                x='Model',
                y='Daily Return (%)',
                title='Return Distribution by Model',
                points='outliers'
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Volatility of returns
            vol_distribution = {}
            for model_name, data in results.items():
                weights = data['weights']
                aligned_weights = weights[returns.columns]
                port_ret = returns @ aligned_weights
                vol_distribution[model_name] = port_ret.std() * np.sqrt(252) * 100
            
            fig_vol_dist = px.bar(
                x=list(vol_distribution.keys()),
                y=list(vol_distribution.values()),
                labels={'x': 'Model', 'y': 'Daily Return Volatility (%)'},
                title='Annualized Volatility from Daily Returns',
                color=list(vol_distribution.values()),
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_vol_dist, use_container_width=True)
        
        st.markdown("---")
        
        # ===== SECTION 7: WEIGHT HEATMAP (Multiple Models) =====
        st.subheader("🗺️ Weight Distribution Across Models")
        
        weights_comparison = pd.DataFrame({
            model: data['weights'] for model, data in results.items()
        })
        
        # Sort by variance across models
        weights_comparison = weights_comparison.loc[weights_comparison.var(axis=1).nlargest(20).index]
        
        fig_heatmap = px.imshow(
            weights_comparison.T * 100,
            labels=dict(color='Weight (%)'),
            title='Top 20 Assets - Weight Distribution Across Models',
            color_continuous_scale='Blues',
            aspect='auto'
        )
        fig_heatmap.update_traces(
            hovertemplate='<b>Model:</b> %{y}<br><b>Asset:</b> %{x}<br><b>Weight:</b> %{z:.2f}%<extra></extra>'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("---")
        
        # ===== SECTION 8: CUMULATIVE RETURNS TIME SERIES =====
        st.subheader("📈 Cumulative Returns Over Time")
        
        # Calculate cumulative returns for each model
        cumulative_returns = {}
        for model_name, data in results.items():
            weights = data['weights']
            portfolio_returns = (returns * weights).sum(axis=1)
            cumulative_returns[model_name] = (1 + portfolio_returns).cumprod()
        
        # Create time series plot
        df_cumreturns = pd.DataFrame(cumulative_returns, index=returns.index)
        
        fig_timeseries = px.line(
            df_cumreturns,
            labels={'value': 'Cumulative Value', 'index': 'Date'},
            title='Portfolio Growth: $1 Invested Since Start Date',
            line_shape='linear'
        )
        
        fig_timeseries.update_layout(
            hovermode='x unified',
            height=450,
            yaxis_title='Portfolio Value ($)',
            xaxis_title='Date'
        )
        
        # Improve hover template
        fig_timeseries.update_traces(
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x|%b %d, %Y}<br>Value: $%{y:.2f}<extra></extra>'
        )
        
        st.plotly_chart(fig_timeseries, use_container_width=True)
        
    else:
        st.info("👈 Configure settings and click 'Run Analysis' to start")


# ==================== DESCRIPTION PAGE ====================

def page_description():
    st.title("📚 Portfolio Optimization Models")
    
    st.markdown("""
    This page explains each model with a consistent structure:
    **Objective**, **Developed by**, **Investment philosophy**, **mathematical objective**,
    **full mathematical derivation**, and **pros/cons**.
    """)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["GMV", "CAPM", "Black-Litterman", "HRP", "CVaR", "LASSO", "RL", "DRO"])

    with tab1:
        st.subheader("🔵 Global Minimum Variance (GMV)")
        st.markdown("**Objective:** Minimize total portfolio variance under long-only and budget constraints.")
        st.markdown("**Developed by:** Harry Markowitz (1952)")
        st.markdown("**Investment philosophy:** If return forecasts are noisy, risk control should come first. Diversification and covariance structure are the primary levers of robust allocation.")
        st.caption("Reference: Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.")

        st.markdown("#### Mathematical Formula (Objective Function)")
        st.latex(r"\min_w \quad w^\top \Sigma w")
        st.latex(r"\text{subject to: } \mathbf{1}^\top w = 1, \quad 0 \le w_i \le w_{\max}")

        st.markdown("#### Mathematical Derivation (Full)")
        st.write("Start from portfolio return and variance:")
        st.latex(r"R_p = w^\top R, \qquad \sigma_p^2 = \mathrm{Var}(R_p)=w^\top\Sigma w")
        st.write("Solve the unconstrained GMV core with budget constraint using a Lagrangian:")
        st.latex(r"\mathcal{L}(w,\eta)=w^\top\Sigma w-\eta(\mathbf{1}^\top w-1)")
        st.write("First-order condition:")
        st.latex(r"\nabla_w\mathcal{L}=2\Sigma w-\eta\mathbf{1}=0 \Rightarrow w=\frac{\eta}{2}\Sigma^{-1}\mathbf{1}")
        st.write(r"Apply the budget constraint $\mathbf{1}^\top w=1$:")
        st.latex(r"\frac{\eta}{2}\,\mathbf{1}^\top\Sigma^{-1}\mathbf{1}=1 \Rightarrow \frac{\eta}{2}=\frac{1}{\mathbf{1}^\top\Sigma^{-1}\mathbf{1}}")
        st.write("Closed-form GMV solution:")
        st.latex(r"w_{GMV}=\frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top\Sigma^{-1}\mathbf{1}}")
        st.write("And minimum achievable variance:")
        st.latex(r"\sigma_{GMV}^2 = w_{GMV}^\top\Sigma w_{GMV}=\frac{1}{\mathbf{1}^\top\Sigma^{-1}\mathbf{1}}")
        st.write(r"With box constraints $0\le w_i\le w_{\max}$, the problem remains convex but is solved numerically (QP).")

        st.markdown("#### Pros and Cons")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Pros:**")
            st.write("✓ No return forecasting needed\n✓ Convex, numerically stable\n✓ Strong risk-control baseline")
        with c2:
            st.write("**Cons:**")
            st.write("✗ Ignores expected return\n✗ Can over-allocate to low-vol assets\n✗ Sensitive to covariance estimation")

    with tab2:
        st.subheader("🟢 CAPM (Tangency / Max Sharpe)")
        st.markdown("**Objective:** Maximize Sharpe ratio of the portfolio using CAPM-implied expected returns.")
        st.markdown("**Developed by:** William Sharpe, John Lintner, Jan Mossin (1960s)")
        st.markdown("**Investment philosophy:** Investors are compensated for systematic risk, not idiosyncratic risk. The best risky portfolio is the tangency portfolio on the efficient frontier.")
        st.caption("Reference: Sharpe, W. (1964). Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk. Journal of Finance, 19(3), 425-442.")

        st.markdown("#### Mathematical Formula (Objective Function)")
        st.latex(r"\max_w \quad \mathrm{SR}(w)=\frac{w^\top\mu - R_f}{\sqrt{w^\top\Sigma w}}")
        st.latex(r"\text{subject to: } \mathbf{1}^\top w = 1, \quad 0 \le w_i \le w_{\max}")

        st.markdown("#### Mathematical Derivation (Full)")
        st.write("Step 1 — CAPM expected return for each asset:")
        st.latex(r"\mathbb{E}[R_i]=R_f+\beta_i\big(\mathbb{E}[R_m]-R_f\big),\qquad \beta_i=\frac{\mathrm{Cov}(R_i,R_m)}{\mathrm{Var}(R_m)}")
        st.write(r"Stack into vector form $\mu\in\mathbb{R}^n$ and define excess-return vector:")
        st.latex(r"\tilde{\mu}=\mu-R_f\mathbf{1}")
        st.write("Step 2 — Sharpe-ratio objective (scale-invariant):")
        st.latex(r"\max_w\;\frac{w^\top\tilde{\mu}}{\sqrt{w^\top\Sigma w}}")
        st.write("Because the ratio is homogeneous in $w$, solve equivalent problem:")
        st.latex(r"\max_w\; w^\top\tilde{\mu}\quad\text{s.t.}\quad w^\top\Sigma w=1")
        st.write("Lagrangian:")
        st.latex(r"\mathcal{L}(w,\kappa)=w^\top\tilde{\mu}-\frac{\kappa}{2}(w^\top\Sigma w-1)")
        st.write("First-order condition:")
        st.latex(r"\nabla_w\mathcal{L}=\tilde{\mu}-\kappa\Sigma w=0 \Rightarrow w\propto \Sigma^{-1}\tilde{\mu}")
        st.write(r"Step 3 — normalize to budget constraint $\mathbf{1}^\top w=1$:")
        st.latex(r"w_{tan}=\frac{\Sigma^{-1}(\mu-R_f\mathbf{1})}{\mathbf{1}^\top\Sigma^{-1}(\mu-R_f\mathbf{1})}")
        st.write("This is the tangency (maximum Sharpe) portfolio. With box constraints, solve numerically via constrained optimization.")
        st.write("Connection to minimum variance: if excess returns are uninformative, allocation emphasis shifts toward variance minimization, yielding the GMV portfolio.")

        st.markdown("#### Pros and Cons")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Pros:**")
            st.write("✓ Directly optimizes risk-adjusted return\n✓ Strong theoretical foundation\n✓ Interpretable via beta")
        with c2:
            st.write("**Cons:**")
            st.write("✗ Sensitive to expected-return errors\n✗ Assumes stable beta structure\n✗ Can produce concentrated weights")

    with tab3:
        st.subheader("🟡 Black-Litterman")
        st.markdown("**Objective:** Blend market equilibrium returns with investor views in a Bayesian framework, then optimize mean-variance utility.")
        st.markdown("**Developed by:** Fischer Black & Robert Litterman (1990, Goldman Sachs)")
        st.markdown("**Investment philosophy:** Start from market consensus (prior), then tilt gradually with explicit confidence in subjective views to avoid unstable allocations.")
        st.caption("Reference: Black, F., & Litterman, R. (1992). Global Portfolio Optimization. Financial Analysts Journal, 48(5), 28-43.")

        st.markdown("#### Mathematical Formula (Objective Function)")
        st.latex(r"\max_w \quad w^\top\mu_{post} - \frac{\lambda}{2}w^\top\Sigma w")
        st.latex(r"\text{subject to: } \mathbf{1}^\top w = 1, \quad 0 \le w_i \le w_{\max}")

        st.markdown("#### Mathematical Derivation (Full)")
        st.write("Step 1 — Reverse optimization (equilibrium prior):")
        st.latex(r"\mu_{eq}=\delta\Sigma w_{mkt}")
        st.write(r"where $\delta$ is market risk aversion and $w_{mkt}$ are market-cap weights.")

        st.write("Step 2 — Specify prior and view likelihood as Gaussian models:")
        st.latex(r"\mu \sim \mathcal{N}(\mu_{eq},\tau\Sigma)")
        st.latex(r"Q\mid\mu \sim \mathcal{N}(P\mu,\Omega)")
        st.write(r"Here, $P$ maps assets to views, $Q$ stores view returns, $\Omega$ encodes view uncertainty.")

        st.write("Step 3 — Apply Bayesian normal-normal conjugacy to obtain posterior moments:")
        st.latex(r"\Sigma_{post}=\left[(\tau\Sigma)^{-1}+P^\top\Omega^{-1}P\right]^{-1}")
        st.latex(r"\mu_{post}=\Sigma_{post}\left[(\tau\Sigma)^{-1}\mu_{eq}+P^\top\Omega^{-1}Q\right]")
        st.write("Equivalent commonly used form:")
        st.latex(r"\mu_{post}=\mu_{eq}+\tau\Sigma P^\top\left(P\tau\Sigma P^\top+\Omega\right)^{-1}(Q-P\mu_{eq})")

        st.write("Step 4 — Plug posterior mean into constrained mean-variance optimization:")
        st.latex(r"\max_w\; w^\top\mu_{post}-\frac{\lambda}{2}w^\top\Sigma w \quad \text{s.t.}\; \mathbf{1}^\top w=1,\;0\le w_i\le w_{\max}")
        st.write(r"Without box constraints, the first-order condition gives $w\propto\Sigma^{-1}\mu_{post}$; with constraints, solve numerically.")

        st.markdown("#### Pros and Cons")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Pros:**")
            st.write("✓ More stable than naive mean-variance\n✓ Explicitly incorporates views/confidence\n✓ Anchored to market equilibrium")
        with c2:
            st.write("**Cons:**")
            st.write(r"✗ More parameters ($\tau$, $\Omega$, views)" + "\n" + r"✗ Requires careful view specification" + "\n" + r"✗ Harder to communicate to non-technical users")

    with tab4:
        st.subheader("🔴 Hierarchical Risk Parity (HRP)")
        st.markdown("**Objective:** Allocate risk across hierarchically clustered assets, avoiding unstable covariance inversion.")
        st.markdown("**Developed by:** Marcos López de Prado (2016)")
        st.markdown("**Investment philosophy:** Treat estimation error as central. Use clustering plus recursive allocation to build diversified portfolios that are robust out of sample.")
        st.caption("Reference: López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of Sample. Journal of Portfolio Management, 42(4), 59-69.")

        st.markdown("#### Mathematical Formula (Objective Function)")
        st.write("HRP is algorithmic (not a single global objective), but core equations are:")
        st.latex(r"d_{ij}=\sqrt{\frac{1-\rho_{ij}}{2}}")
        st.latex(r"w_L=\frac{R_R}{R_L+R_R},\quad w_R=\frac{R_L}{R_L+R_R}")

        st.markdown("#### Mathematical Derivation (Full)")
        st.write("Step 1 — Convert correlation to a proper distance for hierarchical clustering:")
        st.latex(r"d_{ij}=\sqrt{\frac{1-\rho_{ij}}{2}}")
        st.write(r"This maps $\rho_{ij}\in[-1,1]$ into $d_{ij}\in[0,1]$ and preserves metric properties needed by linkage.")

        st.write(r"Step 2 — Build dendrogram and quasi-diagonalize covariance matrix $\Sigma$ according to leaf order.")

        st.write("Step 3 — Define cluster risk via inverse-variance portfolio (IVP) inside each cluster $C$:")
        st.latex(r"w^{IVP}_C = \frac{\mathrm{diag}(\Sigma_C)^{-1}\mathbf{1}}{\mathbf{1}^\top\mathrm{diag}(\Sigma_C)^{-1}\mathbf{1}}")
        st.latex(r"R_C = \left(w^{IVP}_C\right)^\top\Sigma_C\,w^{IVP}_C")

        st.write(r"Step 4 — Recursive bisection allocation between left/right child clusters $(L,R)$:")
        st.latex(r"\frac{w_L}{w_R}=\frac{R_R}{R_L}")
        st.latex(r"w_L=\frac{R_R}{R_L+R_R},\qquad w_R=\frac{R_L}{R_L+R_R}")
        st.write("Multiply current parent weight by $(w_L,w_R)$ and recurse until leaf assets are reached.")

        st.write("Final portfolio weight for asset $i$ equals product of all split factors along its path in the tree:")
        st.latex(r"w_i=\prod_{s\in\mathcal{P}(i)} a_s,\quad a_s\in\left\{\frac{R_R}{R_L+R_R},\frac{R_L}{R_L+R_R}\right\}")
        st.write(r"HRP avoids global matrix inversion of full $\Sigma$, reducing instability from estimation error.")

        st.markdown("#### Pros and Cons")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Pros:**")
            st.write("✓ Robust to estimation noise\n✓ No expected-return model required\n✓ Typically stable allocations")
        with c2:
            st.write("**Cons:**")
            st.write("✗ Not explicitly return-maximizing\n✗ Depends on clustering/linkage choice\n✗ Less transparent optimization target")

    with tab5:
        st.subheader("🔶 Conditional Value at Risk (CVaR)")
        st.markdown("**Objective:** Minimize expected tail loss beyond a quantile threshold.")
        st.markdown("**Developed by:** Rockafellar & Uryasev (2000)")
        st.markdown("**Investment philosophy:** Protect the portfolio from severe downside events, not just average volatility. Tail risk matters most for capital preservation.")
        st.caption("Reference: Rockafellar, R. T., & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. Journal of Risk, 2(3), 21-41.")

        st.markdown("#### Mathematical Formula (Objective Function)")
        st.latex(r"\min_w\;\mathrm{CVaR}_{\alpha}(L(w))")
        st.latex(r"\text{subject to: } \mathbf{1}^\top w=1,\quad 0\le w_i\le w_{\max}")

        st.markdown("#### Mathematical Derivation (Full)")
        st.write("Step 1 — Define portfolio loss from returns $r$:")
        st.latex(r"L(w,r)=-w^\top r")
        st.write("Step 2 — Quantile and tail expectation definitions:")
        st.latex(r"\mathrm{VaR}_{\alpha}(w)=\inf\{\zeta:\mathbb{P}(L(w,r)\le \zeta)\ge \alpha\}")
        st.latex(r"\mathrm{CVaR}_{\alpha}(w)=\mathbb{E}[L(w,r)\mid L(w,r)\ge \mathrm{VaR}_{\alpha}(w)]")

        st.write("Step 3 — Rockafellar-Uryasev convex reformulation:")
        st.latex(r"\mathrm{CVaR}_{\alpha}(w)=\min_{\zeta\in\mathbb{R}}\left\{\zeta+\frac{1}{1-\alpha}\,\mathbb{E}\big[(L(w,r)-\zeta)_+\big]\right\}")
        st.write(r"where $(x)_+=\max(x,0)$. This converts tail-risk minimization into a convex program.")

        st.write(r"Step 4 — Sample average approximation with scenarios $r_t,\; t=1,\dots,T$:")
        st.latex(r"\min_{w,\zeta,u}\; \zeta+\frac{1}{(1-\alpha)T}\sum_{t=1}^T u_t")
        st.latex(r"\text{s.t. } u_t\ge -w^\top r_t-\zeta,\; u_t\ge 0,\; \mathbf{1}^\top w=1,\;0\le w_i\le w_{\max}")
        st.write(r"This is a linear program in $(w,\zeta,u)$ after scenario discretization.")

        st.write(r"Step 5 — KKT intuition: at optimum, active scenarios satisfy $u_t=-w^\top r_t-\zeta>0$ and lie in the worst $1-\alpha$ tail; inactive scenarios have $u_t=0$.")

        st.markdown("#### Pros and Cons")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Pros:**")
            st.write("✓ Direct tail-risk control\n✓ Better for skewed/fat-tail returns\n✓ Useful for stress-aware mandates")
        with c2:
            st.write("**Cons:**")
            st.write("✗ Needs enough data for stable tails\n✗ Can ignore upside potential\n✗ Optimization can be less stable with sparse samples")

    with tab6:
        st.subheader("🟣 LASSO Portfolio (Cardinality-Sparse)")
        st.markdown("**Objective:** Maximize return and penalize risk via iterative hard thresholding, selecting exactly K assets.")
        st.markdown("**Developed by:** Robert Tibshirani (LASSO, 1996); sparse portfolio optimization via cardinality control")
        st.markdown("**Investment philosophy:** Simpler portfolios reduce transaction costs, improve execution, and are more resistant to estimation error. This implementation uses **iterative hard thresholding** to enforce true sparsity (K non-zero weights).")
        st.caption("Reference: Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. Journal of the Royal Statistical Society: Series B, 58(1), 267-288.")

        st.markdown("#### 🔧 Implementation Note (FIXED)")
        st.info("""
**Previous Issue:** The standard L1 penalty is mathematically ineffective under long-only + fully-invested constraints, 
since $\\|w\\|_1 = \\sum|w_i| = \\sum w_i = 1$ is constant and does not vary with the solution.

**Current Fix:** We now use **cardinality-based hard thresholding**:
1. Optimize mean-variance + L1 on the full asset set
2. Iteratively keep only the top K assets by weight magnitude
3. Re-optimize on the reduced support
4. Repeat until convergence (typically 2-3 iterations)

Result: Enforces cardinality constraint $\\text{card}(w) \\le K$ (default $K=10$) while maintaining mathematical rigor.
        """)

        st.markdown("#### Mathematical Formula (Cardinality-Constrained Objective)")
        st.latex(r"\max_w \quad w^\top\mu - \frac{\lambda}{2}w^\top\Sigma w \quad \text{subject to: } \text{card}(w) \le K,\; \mathbf{1}^\top w = 1,\quad 0\le w_i\le w_{\max}")

        st.markdown("#### Mathematical Derivation (Full)")
        st.write("**Step 1** — Start with unconstrained mean-variance + L1 regularization:")
        st.latex(r"\max_w\; w^\top\mu-\frac{\lambda}{2}w^\top\Sigma w-\gamma\|w\|_1 \quad \text{s.t. } \mathbf{1}^\top w=1,\;0\le w_i\le w_{\max}")
        
        st.write("**Step 2** — Standard L1 penalty ineffectiveness under long-only + fully-invested constraints:")
        st.latex(r"\|w\|_1 = \sum_{i=1}^n |w_i| = \sum_{i=1}^n w_i = 1 \quad \text{(constant, invariant in optimization)}")
        st.write("Since $\\|w\\|_1$ is constant, the L1 term does not influence the optimal solution. We need cardinality control instead.")
        
        st.write("**Step 3** — Iterative Hard Thresholding Algorithm:")
        st.latex(r"\text{For } t = 1, 2, \ldots, T_{\max}:")
        st.latex(r"w^{(t+1)} = \arg\max_{w} \; w^\top\mu-\frac{\lambda}{2}w^\top\Sigma w, \quad \text{s.t. } \text{supp}(w) \subseteq \mathcal{S}_t")
        st.latex(r"\mathcal{S}_{t+1} = \text{argmax}_{\\|S\\|=K} \sum_{i \in S} |w_i^{(t+1)}|")
        st.write(r"where $\mathcal{S}_t$ is the active support (top K assets by magnitude), and we re-optimize on the restricted set.")
        
        st.write("**Step 4** — Final solution satisfies approximate Karush-Kuhn-Tucker (KKT) conditions on active set:")
        st.latex(r"\lambda\Sigma w^* - \mu + \nu \mathbf{1}^{\text{active}} = 0")
        st.write(r"where $\nu$ enforces the budget constraint on the active subset, and non-active weights are set to zero.")

        st.markdown("#### Pros and Cons")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Pros:**")
            st.write("✓ True sparse portfolios (exactly K holdings)\n✓ Lower turnover and trading costs\n✓ Interpretable, simplified complexity\n✓ Empirically robust to estimation error")
        with c2:
            st.write("**Cons:**")
            st.write("✗ Cardinality selection ($K$) is a hyperparameter\n✗ May under-diversify if $K$ is too small\n✗ Discrete optimization can be slower\n✗ Sensitive to expected-return estimation")

    with tab7:
        st.subheader("🟠 Reinforcement Learning (RL)")
        st.markdown("**Objective:** Learn optimal portfolio policy through agent training on historical returns with reward feedback.")
        st.markdown("**Developed by:** Modern machine learning; applied to portfolio optimization via policy gradient methods")
        st.markdown("**Investment philosophy:** Rather than assuming static optimization, allow a learning agent to discover adaptive allocation strategies from data, capturing non-linear relationships and regime changes that traditional models miss.")
        st.caption("Reference: Charpentier, A., Élie, R., & Remlinger, C. (2021). Reinforcement Learning in Different Phases of the Money Management Process. arXiv:2110.03079.")

        st.markdown("#### Mathematical Formula (Policy Gradient Objective)")
        st.latex(r"\max_{\theta}\; \mathbb{E}[R_{\text{portfolio}} | a_t \sim \pi_\theta(s_t)]")
        st.latex(r"\text{subject to: } \mathbf{1}^\top w_t = 1, \quad 0 \le w_i(t) \le w_{\max}, \quad \text{(long-only and fully invested)}")

        st.markdown("#### Mathematical Derivation (Full)")
        st.write("**Step 1** — Define the trading environment as a Markov Decision Process (MDP):")
        st.latex(r"(S, A, P, R, \gamma) \text{ where:}")
        st.write(r"- $S$: State space = market covariance matrix + recent returns (lookback window)")
        st.write(r"- $A$: Action space = portfolio weight vector $w \in \Delta^n$ (simplex)")
        st.write(r"- $P$: Transition dynamics = next-day returns from $r_t \sim ?$")
        st.write(r"- $R$: Reward = portfolio return (Sharpe ratio, cumulative, or risk-adjusted)")
        st.write(r"- $\gamma$: Discount factor = 0.99 (value of future rewards)")

        st.write("**Step 2** — Policy parameterization using neural network:")
        st.latex(r"\pi_\theta(w_t | s_t) = \text{softmax}(\text{NN}_\theta(s_t))")
        st.write(r"where $\text{NN}_\theta$ is a learnable neural network with weights $\theta$, and softmax ensures $\mathbf{1}^\top w = 1$.")

        st.write("**Step 3** — Define the value function (expected cumulative return from state $s$):")
        st.latex(r"V^\pi(s) = \mathbb{E}[G_t | S_t = s] = \mathbb{E}\left[\sum_{k=0}^\infty \gamma^k R_{t+k} \mid S_t = s\right]")
        st.write("The agent learns to maximize this: better states have higher value functions.")

        st.write("**Step 4** — Policy gradient update (REINFORCE or Actor-Critic):")
        st.latex(r"\theta_{t+1} = \theta_t + \alpha \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot (R_t + \gamma V(s_{t+1}) - V(s_t))")
        st.write(r"(advantage $A_t = R_t + \gamma V(s_{t+1}) - V(s_t)$), scaled by learning rate $\alpha$.")

        st.write("**Step 5** — Iterative improvement:")
        st.latex(r"\text{For } \text{episode} = 1, 2, \ldots, N_{\text{epochs}}:")
        st.latex(r"1. \text{ Reset state } s_0 \text{ (e.g., current market regime)}")
        st.latex(r"2. \text{ For } t = 1, \ldots, T: \quad a_t = \pi_\theta(s_t), \quad s_{t+1} = \text{next market state}, \quad R_t = \text{return}")
        st.latex(r"3. \text{ Accumulate } A_t \text{ (advantage)}, \text{ update } \theta \text{ via gradient step}")
        st.write(r"Repeat until convergence (policy stabilizes across random episodes).")

        st.markdown("#### Pros and Cons")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Pros:**")
            st.write("✓ Adaptive to market regimes and changing conditions\n✓ Captures non-linear relationships\n✓ Can learn from large historical datasets\n✓ No closed-form solution needed")
        with c2:
            st.write("**Cons:**")
            st.write("✗ Computationally expensive (slow training)\n✗ Requires many training episodes (data hungry)\n✗ Overfitting risk on historical data\n✗ 'Black box' - difficult to interpret decisions\n✗ Unstable in scarce-data regimes (small portfolios)")

    with tab8:
        st.subheader("🔵 Distributionally Robust Optimization (DRO)")
        st.markdown("**Objective:** Optimize portfolio under ambiguity, protecting against model misspecification by considering all distributions in an uncertainty set.")
        st.markdown("**Developed by:** Ben-Tal, Ghaoui, & Nemirovski (2000+)")
        st.markdown("**Investment philosophy:** Traditional mean-variance assumes we know the true return distribution. DRO relaxes this: optimize for the worst-case return distribution within a **ball** of likely distributions, ensuring robustness.")
        st.caption("Reference: Ben-Tal, A., Ghaoui, L. E., & Nemirovski, A. (2009). Robust Optimization. Princeton University Press.")

        st.markdown("#### Mathematical Formula (Min-Max Objective)")
        st.latex(r"\max_w \; \min_{F \in \mathcal{F}_\epsilon} \mathbb{E}_F[r^\top w] - \lambda \cdot \text{Var}_F(r^\top w)")
        st.latex(r"\text{subject to: } \mathbf{1}^\top w = 1, \quad 0 \le w_i \le w_{\max}")

        st.markdown("#### Mathematical Derivation (Full)")
        st.write("**Step 1** — Standard mean-variance assumes exact distribution $F^*$ of returns is known:")
        st.latex(r"\mu = \mathbb{E}_{F^*}[r], \quad \Sigma = \text{Cov}_{F^*}[r]")
        st.write("But $F^*$ is unknown; we only observe samples. DRO hedges against this uncertainty.")

        st.write("**Step 2** — Define an uncertainty set $\\mathcal{F}_\\epsilon$ around the empirical distribution:")
        st.latex(r"\mathcal{F}_\epsilon = \{F : d(F, \hat{F}_N) \le \epsilon\}")
        st.write(r"where $\hat{F}_N$ is the empirical distribution (sample CDF) and $d(\cdot, \cdot)$ is a divergence (Wasserstein, KL, $\chi^2$, etc.).")
        st.write(r"Common choice: **Wasserstein distance** $W_p(F, \hat{F}) = \inf_{\pi} \left(\int |x-y|^p d\pi(x,y)\right)^{1/p}$")

        st.write("**Step 3** — Min-max reformulation (worst-case optimization):")
        st.latex(r"\max_w \min_{F \in \mathcal{F}_\epsilon} \left[ \mathbb{E}_F[w^\top r] - \frac{\lambda}{2} (w^\top \text{Cov}_F[r] w) \right]")
        st.write(r"Intuitively: choose $w$ that performs best even if Nature picks the worst distribution in the uncertainty set.")

        st.write("**Step 4** — Tractable reformulation via Lagrangian duality:")
        st.write(r"Under Wasserstein ambiguity, the inner min over distributions can be solved explicitly:")
        st.latex(r"\min_F \mathbb{E}_F[r^\top w] \approx \mu_N^\top w - \epsilon \|w\|_2")
        st.write(r"where $\mu_N$ is the sample mean and $\|w\|_2$ is the regularization (penalizes extreme allocations).")
        st.write(r"Similarly for variance, yielding a tractable least-squares problem with $\ell_2$ regularization.")

        st.write("**Step 5** — Final convex optimization:")
        st.latex(r"\max_w \quad \mu_N^\top w - \epsilon \|w\|_2 - \frac{\lambda}{2} w^\top (\Sigma_N + \epsilon I) w")
        st.latex(r"\text{subject to: } \mathbf{1}^\top w = 1, \quad 0 \le w_i \le w_{\max}")
        st.write(r"The $\epsilon I$ term adds regularization, making covariance more stable when $n$ is large and $N$ (samples) is limited.")

        st.markdown("#### Pros and Cons")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Pros:**")
            st.write("✓ Robust to model misspecification\n✓ Handles small-sample regimes well\n✓ Convex optimization (tractable)\n✓ Theoretically justified (worst-case guarantee)")
        with c2:
            st.write("**Cons:**")
            st.write("✗ Hyperparameter $\\epsilon$ (uncertainty radius) must be chosen\n✗ Conservative: may under-allocate to risky assets\n✗ Computationally heavier than GMV\n✗ Sensitive to choice of divergence metric ($W_p$, KL, etc.)")


# ==================== HOLDINGS PAGE (merged: Builder + Factor Analysis) ====================

def page_holdings():
    st.title("📈 Portfolio Builder & Factor Analysis")
    st.markdown("""
    Build your custom portfolio and analyze factor exposures:
    - **Section 1**: Add/remove stocks and set portfolio constraints
    - **Section 2**: Analyze factor exposure of your holdings
    """)
    
    st.divider()
    
    # ===== SECTION 1: PORTFOLIO CONSTRUCTION =====
    st.subheader("🎯 SECTION 1: Portfolio Construction & Configuration")
    
    with st.expander("📝 Stock Selection & Customization", expanded=True):
        add_col, rem_col = st.columns(2)
        with add_col:
            new_ticker = st.text_input("Add ticker", placeholder="e.g., AAPL").upper().strip()
            if st.button("Add", use_container_width=True):
                if not new_ticker:
                    st.warning("Please enter a ticker.")
                elif new_ticker in st.session_state.selected_stocks:
                    st.info(f"{new_ticker} is already in the portfolio.")
                else:
                    check = validate_ticker(new_ticker)
                    if check.get("valid", False):
                        # Check if ticker has known sector
                        from sectors import STOCK_SECTORS
                        if new_ticker not in STOCK_SECTORS:
                            # Try to fetch from yfinance
                            fetched_sector = fetch_sector_from_yfinance(new_ticker)
                            if fetched_sector:
                                st.session_state.custom_sector_mapping = update_custom_sector(
                                    new_ticker, fetched_sector, f"{fetched_sector} (via Yahoo Finance)",
                                    st.session_state.custom_sector_mapping
                                )
                                st.info(f"✓ Sector auto-assigned: {fetched_sector} (fetched from Yahoo Finance)")
                            else:
                                # Prompt user to assign sector
                                st.info("📌 New ticker detected. Please assign a sector:")
                                available_sectors = get_sectors() + ["Other"]
                                sector = st.selectbox(
                                    f"Select sector for {new_ticker}:",
                                    available_sectors,
                                    key=f"sector_select_{new_ticker}"
                                )
                                if st.button(f"Confirm sector '{sector}' for {new_ticker}"):
                                    st.session_state.custom_sector_mapping = update_custom_sector(
                                        new_ticker, sector, f"{sector} (Custom)",
                                        st.session_state.custom_sector_mapping
                                    )
                                    st.session_state.selected_stocks.append(new_ticker)
                                    st.success(f"Added {new_ticker} in {sector} sector")
                                    st.rerun()
                        else:
                            st.session_state.selected_stocks.append(new_ticker)
                            st.success(f"Added {new_ticker}")
                            st.rerun()
                        
                        # If we reach here for new custom stocks, add them
                        if new_ticker not in st.session_state.selected_stocks and new_ticker in st.session_state.custom_sector_mapping:
                            st.session_state.selected_stocks.append(new_ticker)
                            st.rerun()
                    else:
                        st.error(f"Invalid ticker: {new_ticker}")
        with rem_col:
            removable = [t for t in st.session_state.selected_stocks if t not in ALL_STOCKS]
            remove_ticker = st.selectbox("Remove custom ticker", ["-"] + removable)
            if st.button("Remove", use_container_width=True):
                if remove_ticker != "-":
                    st.session_state.selected_stocks.remove(remove_ticker)
                    # Also remove from custom mapping if present
                    if remove_ticker in st.session_state.custom_sector_mapping:
                        del st.session_state.custom_sector_mapping[remove_ticker]
                    st.success(f"Removed {remove_ticker}")
                    st.rerun()

    st.write(f"**Current universe:** {len(st.session_state.selected_stocks)} tickers")
    
    # Divider before configuration
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Start date", value=datetime.now() - timedelta(days=365 * 2), max_value=datetime.now(), key="builder_start")
    with col2:
        end_date = st.date_input("End date", value=datetime.now(), max_value=datetime.now(), key="builder_end")
    with col3:
        max_weight = st.slider("Max weight per stock (%)", 5, 50, 15, 1, key="builder_max_weight") / 100

    # ===== OPTIONAL CONSTRAINTS =====
    st.markdown("### 🔒 Optional Portfolio Constraints")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        enable_sector_caps = st.checkbox("Enable sector caps", key="enable_sector_caps")
    with col2:
        enable_min_weight = st.checkbox("Enable min weight per holding", key="enable_min_weight")
    with col3:
        enable_max_holdings = st.checkbox("Limit max holdings", key="enable_max_holdings")
    
    constraints = {}
    
    if enable_sector_caps:
        st.markdown("**Sector Weight Caps (% of portfolio)**")
        sectors = get_sectors()
        sector_cols = st.columns(3)
        sector_caps = {}
        for i, sector in enumerate(sectors):
            with sector_cols[i % 3]:
                cap = st.slider(f"{sector}", 5, 100, 30, 5, key=f"sector_cap_{sector}") / 100
                sector_caps[sector] = cap
        constraints['sector_caps'] = sector_caps
    
    if enable_min_weight:
        constraints['min_weight'] = st.slider(
            "Minimum weight per non-zero holding (%)",
            0.5, 10.0, 1.0, 0.5,
            key="min_weight_slider"
        ) / 100
        st.caption("⚠️ Any holding with weight > 0 must be >= this minimum")
    
    if enable_max_holdings:
        max_num_holdings = len(st.session_state.selected_stocks)
        constraints['max_holdings'] = st.slider(
            "Maximum number of holdings",
            1, max_num_holdings, min(10, max_num_holdings),
            key="max_holdings_slider"
        )
        st.caption(f"📊 Portfolio will hold at most {constraints['max_holdings']} assets")

    st.divider()
    
    # Display constraint validation
    if constraints:
        st.markdown("---")
        st.subheader("✓ Constraints Applied")
        constraint_text = []
        if 'sector_caps' in constraints:
            constraint_text.append(f"**Sector caps:** {', '.join([f'{s}: {v*100:.0f}%' for s, v in constraints['sector_caps'].items()])}")
        if 'min_weight' in constraints:
            constraint_text.append(f"**Min weight per holding:** {constraints['min_weight']*100:.1f}%")
        if 'max_holdings' in constraints:
            constraint_text.append(f"**Max holdings:** {constraints['max_holdings']}")
        for text in constraint_text:
            st.write(text)
    
    # ===== SECTION 2: FACTOR EXPOSURE ANALYSIS =====
    st.markdown("---")
    st.divider()
    st.subheader("🔍 SECTION 2: Factor Exposure Analysis")
    st.markdown("""
    Understand what risk factors your portfolio is exposed to:
    - **Market (Beta)**: Systematic risk relative to S&P 500
    - **Size**: Large-cap vs Small-cap bias
    - **Value**: Value stocks vs Growth stocks (P/B ratio)
    - **Momentum**: Trending vs Mean-reverting exposure
    - **Quality**: High-quality vs Low-quality companies (ROE, profitability)
    """)
    
    # Date range for factor exposure analysis
    col1, col2 = st.columns(2)
    with col1:
        start_date_factor = st.date_input("Start date", value=datetime.now() - timedelta(days=365), max_value=datetime.now(), key="holdings_start")
    with col2:
        end_date_factor = st.date_input("End date", value=datetime.now(), max_value=datetime.now(), key="holdings_end")
    
    st.markdown("### 📊 Enter Custom Portfolio Weights")
    st.caption("Build your custom portfolio and analyze its factor exposures. Total weights should sum to 100%")
    
    portfolio_tickers = []
    portfolio_weights = []
    
    # Input table for custom weights
    num_rows = 5
    cols = st.columns(3)
    with cols[0]:
        st.write("**Stock**")
    with cols[1]:
        st.write("**Weight (%)**")
    with cols[2]:
        st.write("")
    
    for i in range(num_rows):
        c1, c2, c3 = st.columns(3)
        with c1:
            ticker = st.text_input(f"Ticker {i+1}", placeholder="MSFT", key=f"holdings_ticker_{i}")
        with c2:
            weight = st.number_input(f"Weight {i+1}", 0.0, 100.0, 0.0, key=f"holdings_weight_{i}")
        
        if ticker and weight > 0:
            portfolio_tickers.append(ticker.upper())
            portfolio_weights.append(weight / 100)
    
    if not portfolio_tickers:
        st.warning("Please enter at least one ticker and weight")
        weights_factor = None
    else:
        # Normalize weights
        total_weight = sum(portfolio_weights)
        if total_weight <= 0:
            st.error("Total weight must be > 0")
            weights_factor = None
        else:
            portfolio_weights = [w / total_weight for w in portfolio_weights]
            weights_factor = pd.Series(dict(zip(portfolio_tickers, portfolio_weights)))
    
    if weights_factor is not None:
        if st.button("🚀 Analyze Factor Exposure", use_container_width=True):
            try:
                with st.spinner("Loading data and calculating exposures..."):
                    # Load data
                    prices, returns = load_portfolio_data(list(weights_factor.index), start_date_factor, end_date_factor)
                    
                    # Calculate portfolio exposures
                    portfolio_exp = portfolio_factor_exposure(weights_factor, returns)
                    
                    # Calculate benchmark exposures
                    benchmarks = get_benchmark_exposures(['SPY', 'QQQ'], returns)
                
                # Display results
                st.markdown("---")
                st.subheader("✓ Factor Exposure Summary")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    beta = portfolio_exp['beta_portfolio']
                    beta_text = f"{beta:.2f}" if not pd.isna(beta) else "N/A"
                    st.metric("Market (Beta)", beta_text, help="Sensitivity to S&P 500")
                
                with col2:
                    size = portfolio_exp['size_portfolio']
                    size_text = f"{size:.2f}" if not pd.isna(size) else "N/A"
                    st.metric("Size", size_text, help="Log market cap exposure")
                
                with col3:
                    value = portfolio_exp['value_portfolio']
                    value_text = f"{value:.2f}" if not pd.isna(value) else "N/A"
                    st.metric("Value", value_text, help="Lower P/B = value, Higher P/B = growth")
                
                with col4:
                    momentum = portfolio_exp['momentum_portfolio']
                    momentum_text = f"{momentum*100:.1f}%" if not pd.isna(momentum) else "N/A"
                    st.metric("Momentum (6M)", momentum_text, help="Positive = uptrend")
                
                with col5:
                    quality = portfolio_exp['quality_portfolio']
                    quality_text = f"{quality:.2f}" if not pd.isna(quality) else "N/A"
                    st.metric("Quality", quality_text, help="Higher ROE & Lower P/B = quality")
                
                # Factor comparison chart
                st.markdown("### Factor Comparison vs Benchmarks")
                
                factor_comparison = {
                    'Portfolio': {
                        'Beta': portfolio_exp.get('beta_portfolio', 0),
                        'Size': portfolio_exp.get('size_portfolio', 0),
                        'Value': portfolio_exp.get('value_portfolio', 0),
                        'Momentum': portfolio_exp.get('momentum_portfolio', 0),
                        'Quality': portfolio_exp.get('quality_portfolio', 0)
                    }
                }
                
                for bench_ticker, bench_exp in benchmarks.items():
                    factor_comparison[bench_ticker] = {
                        'Beta': bench_exp.get('beta', 0),
                        'Size': bench_exp.get('size', 0),
                        'Value': bench_exp.get('value', 0),
                        'Momentum': bench_exp.get('momentum', 0),
                        'Quality': bench_exp.get('quality', 0)
                    }
                
                comparison_df = pd.DataFrame(factor_comparison).T.fillna(0)
                
                fig_factors = go.Figure()
                for factor in comparison_df.columns:
                    fig_factors.add_trace(go.Bar(
                        name=factor,
                        x=comparison_df.index,
                        y=comparison_df[factor]
                    ))
                
                fig_factors.update_layout(
                    title="Factor Exposure Comparison",
                    barmode='group',
                    xaxis_title="Portfolio",
                    yaxis_title="Exposure",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_factors, use_container_width=True)
                
                # Individual stock breakdown
                st.markdown("### Individual Stock Factor Breakdown")
                
                df_breakdown = portfolio_exp['factor_breakdown']
                
                if not df_breakdown.empty:
                    # Display table
                    display_cols = ['Ticker', 'Weight', 'Beta', 'Size', 'Value', 'Momentum', 'Quality']
                    display_df = df_breakdown[display_cols].copy()
                    
                    # Format columns
                    display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x*100:.1f}%")
                    for col in ['Beta', 'Size', 'Value', 'Quality']:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                    display_df['Momentum'] = display_df['Momentum'].apply(lambda x: f"{x*100:.1f}%" if not pd.isna(x) else "N/A")
                    
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Size scatter (Market Cap vs Weight)
                    st.markdown("### Size Distribution")
                    scatter_df = df_breakdown[['Ticker', 'Weight', 'Market Cap']].copy()
                    scatter_df = scatter_df[scatter_df['Market Cap'].notna()]
                    
                    if not scatter_df.empty:
                        fig_scatter = px.scatter(
                            scatter_df,
                            x='Market Cap',
                            y='Weight',
                            size='Weight',
                            hover_name='Ticker',
                            log_x=True,
                            title="Market Cap vs Portfolio Weight",
                            labels={'Market Cap': 'Market Cap (log scale)', 'Weight': 'Weight in Portfolio'}
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Interpretation guide
                st.markdown("---")
                st.markdown("### 📖 Factor Interpretation Guide")
                
                with st.expander("What do these factors mean?"):
                    st.markdown("""
                    **Beta (Market Risk)**
                    - Beta > 1: Portfolio is more volatile than S&P 500 (higher systematic risk)
                    - Beta = 1: Moves in line with S&P 500
                    - Beta < 1: Portfolio is less volatile than S&P 500 (lower systematic risk)
                    
                    **Size**
                    - Higher values: Biased toward large-cap stocks
                    - Lower values: Biased toward small-cap stocks
                    
                    **Value**
                    - Positive: Value stock bias (lower P/B ratios)
                    - Negative: Growth stock bias (higher P/B ratios)
                    
                    **Momentum**
                    - Positive %: Stocks trending up (bullish exposure)
                    - Negative %: Stocks trending down (bearish exposure)
                    - Near 0%: Neutral momentum
                    
                    **Quality**
                    - Positive: High-quality companies (high ROE, low P/B, profitable)
                    - Negative: Lower-quality companies (low ROE, high P/B)
                    - Range typically: -1 to +1
                    """)
            
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                import traceback
                st.write(traceback.format_exc())


def page_backtest():
    st.title("📊 Rolling Backtest")
    st.markdown("Evaluate out-of-sample performance for all models using rolling window rebalancing with realistic transaction costs.")
    
    st.info("""
    **📌 Backtest Methodology:**
    - **Rolling Window:** Train on past N years of returns, test on future out-of-sample period
    - **Rebalancing:** Periodic rebalancing (monthly/quarterly/yearly) to maintain target weights
    - **Transaction Costs:** Each rebalance incurs trading costs (default: 10 basis points = 0.1% per one-way turn)
    - **Turnover Cost Formula:** $\\text{Cost} = 0.1\\% \\times \\sum_i |w_i^{\\text{new}} - w_i^{\\text{old}}|$ (one-way)
    
    This ensures realistic comparison across models, since sparse portfolios (ex: LASSO) typically have lower turnover.
    """)
    
    st.success("""
    **🔄 Backtest vs Home Page:**
    - **Backtest:** Rolling historical simulation showing how strategies would have performed since 2020 with monthly rebalancing
    - **Home Page:** One-time portfolio construction showing realistic performance on hold-to-test period
    
    Both use **out-of-sample evaluation** (honest assessment), but Backtest shows rolling returns over 6+ years while 
    Home Page shows point-in-time performance. Results should be in similar range (~15-25% annual return).
    """)

    tickers = st.session_state.selected_stocks if st.session_state.selected_stocks else ALL_STOCKS

    col1, col2 = st.columns(2)
    with col1:
        lookback_years = st.slider("Lookback window (years)", 1.0, 5.0, 2.0, 0.5)
    with col2:
        rebalance_freq = st.selectbox(
            "Rebalance frequency",
            ["ME", "QE", "YE"],
            format_func=lambda x: {"ME": "Monthly", "QE": "Quarterly", "YE": "Yearly"}[x],
        )
    
    col3, col4 = st.columns(2)
    with col3:
        use_param_tuning = st.checkbox(
            "Enable walk-forward parameter tuning",
            value=True,
            help="Auto-tune model hyperparameters (e.g., LASSO K, CVaR alpha, BL tau) using train/validation splits. More realistic but slower."
        )
    with col4:
        if use_param_tuning:
            train_val_ratio = st.slider("Train/validation split", 0.5, 0.8, 0.6, 0.05)
        else:
            train_val_ratio = 0.6

    if st.button("🚀 Run Backtest", use_container_width=True):
        try:
            timing_msg = "1-2 minutes" if not use_param_tuning else "3-5 minutes (parameter tuning enabled)"
            with st.spinner(f"Running backtest ({timing_msg})..."):
                backtest_results = run_rolling_backtest(
                    tickers=tickers,
                    start="2020-01-01",
                    end="2026-02-20",
                    lookback_days=int(lookback_years * 252),
                    rebalance_freq=rebalance_freq,
                    max_weight=0.15,
                    risk_free_rate=RISK_FREE_RATE,
                    transaction_cost_rate=0.001,  # 10 bps per one-way transaction
                    use_param_tuning=use_param_tuning,
                    train_val_ratio=train_val_ratio
                )
                st.session_state.backtest_results = backtest_results
            st.success("✅ Backtest complete!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    if "backtest_results" in st.session_state:
        results = st.session_state.backtest_results
        
        # Separate benchmarks from models
        benchmarks = results.get('benchmarks', {})
        model_results = {k: v for k, v in results.items() if k != 'benchmarks'}

        st.markdown("---")
        
        # Show evaluation methodology info
        if use_param_tuning:
            st.info(
                "📊 **Evaluation Methodology**: Walk-forward out-of-sample analysis with "
                f"{int(train_val_ratio*100)}/{int((1-train_val_ratio)*100)} train/validation split. "
                "Each portfolio is optimized only on training data, then evaluated on unseen validation data. "
                "Portfolio weights are rebalanced at each period. Results are honest, out-of-sample performance."
            )
        else:
            st.warning(
                "⚠️ **Note**: Using default parameters (walk-forward tuning disabled). "
                "Enable 'walk-forward parameter tuning' for more realistic out-of-sample results."
            )

        st.subheader("📊 Performance Metrics (Out-of-Sample)")
        
        # Model performance table
        st.markdown("### Models")
        metrics_data = []
        for model, data in model_results.items():
            m = data["metrics"]
            metrics_data.append({
                "Model": model,
                "Return": f"{m['annual_return']*100:.2f}%",
                "Volatility": f"{m['annual_volatility']*100:.2f}%",
                "Sharpe": f"{m['sharpe_ratio']:.3f}",
                "Sortino": f"{m['sortino_ratio']:.3f}",
                "Max DD": f"{m['max_drawdown']*100:.2f}%",
            })
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
        
        # Benchmark performance table
        if benchmarks:
            st.markdown("### Benchmarks")
            bench_data = []
            for bench, data in benchmarks.items():
                m = data["metrics"]
                bench_data.append({
                    "Benchmark": bench,
                    "Return": f"{m['annual_return']*100:.2f}%",
                    "Volatility": f"{m['annual_volatility']*100:.2f}%",
                    "Sharpe": f"{m['sharpe_ratio']:.3f}",
                    "Sortino": f"{m['sortino_ratio']:.3f}",
                    "Max DD": f"{m['max_drawdown']*100:.2f}%",
                })
            st.dataframe(pd.DataFrame(bench_data), use_container_width=True)

        st.subheader("📈 Cumulative Returns")
        
        # Build cumulative returns dataframe (models + benchmarks)
        cumulative_df = pd.DataFrame({
            model: (data["cumulative_returns"] - 1) * 100 for model, data in model_results.items()
        })
        
        # Add benchmarks to the chart
        if benchmarks:
            for bench_name, bench_data in benchmarks.items():
                cumulative_df[f"{bench_name} (Benchmark)"] = (bench_data["cumulative_returns"] - 1) * 100
        
        fig_returns = px.line(
            cumulative_df, 
            title="Cumulative Returns: Models vs Benchmarks", 
            labels={"value": "Return (%)", "index": "Date"}
        )
        
        # Style benchmark lines differently (dashed)
        if benchmarks:
            for i, col in enumerate(cumulative_df.columns):
                if "(Benchmark)" in col:
                    fig_returns.data[i].line.dash = 'dash'
                    fig_returns.data[i].line.width = 2
        
        st.plotly_chart(fig_returns, use_container_width=True)
        
        # Relative Performance Metrics
        st.subheader("📊 Relative Performance vs Benchmarks")
        st.markdown("""
        **Metrics:**
        - **Alpha**: Annualized excess return vs benchmark (Jensen's alpha)
        - **Beta**: Portfolio sensitivity to benchmark movements
        - **Information Ratio**: Alpha / Tracking Error (risk-adjusted excess return)
        - **Tracking Error**: Annualized volatility of active returns (portfolio - benchmark)
        - **Correlation**: Correlation coefficient with benchmark
        """)
        
        # Select benchmark for comparison
        benchmark_options = list(benchmarks.keys())
        if benchmark_options:
            selected_benchmark = st.selectbox(
                "Select benchmark for comparison",
                benchmark_options,
                index=0 if "SPY" in benchmark_options else 0
            )
            
            # Build relative metrics table
            rel_metrics_data = []
            for model, data in model_results.items():
                if 'relative_metrics' in data and selected_benchmark in data['relative_metrics']:
                    rel = data['relative_metrics'][selected_benchmark]
                    rel_metrics_data.append({
                        "Model": model,
                        "Alpha": f"{rel['alpha']*100:.2f}%",
                        "Beta": f"{rel['beta']:.3f}",
                        "Info Ratio": f"{rel['information_ratio']:.3f}",
                        "Tracking Error": f"{rel['tracking_error']*100:.2f}%",
                        "Correlation": f"{rel['correlation']:.3f}",
                    })
            
            if rel_metrics_data:
                st.dataframe(pd.DataFrame(rel_metrics_data), use_container_width=True)
            else:
                st.info("No relative metrics available for selected benchmark")
        
    else:
        st.info("Configure parameters and run backtest to view out-of-sample performance.")


# ==================== CHATBOT PAGE ====================

def page_chatbot():
    st.title("💬 Investment Decision Chatbot")
    
    st.info("""
    💡 **Ask Questions About Markets, Sectors, or Specific Stocks**
    
    **Stock Analysis**:
    - "Is TSLA a good investment based on my portfolio?"
    - "Should I add MSFT given my current holdings?"
    
    **Sector/Industry Trends**:
    - "How is the semiconductor industry performing?"
    - "What's happening in the technology sector?"
    
    **Market Outlook**:
    - "What's the overall market outlook?"
    - "How are market conditions right now?"
    
    **Macro Environment**:
    - "How will Fed rate cuts impact my portfolio?"
    - "What are the inflation risks?"
    """)
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display existing chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    user_input = st.chat_input("Ask about stocks, sectors, market conditions, or macro trends...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            try:
                # Get portfolio data if available
                portfolio_weights = None
                returns_data = None
                
                if st.session_state.selected_stocks:
                    try:
                        prices, returns = load_portfolio_data(
                            st.session_state.selected_stocks, 
                            datetime.now() - timedelta(days=365*2),
                            datetime.now()
                        )
                        returns_data = returns
                        
                        # Get portfolio weights from last run
                        if "last_weights" in st.session_state:
                            portfolio_weights = st.session_state.last_weights
                        else:
                            # Default to equal weight
                            portfolio_weights = pd.Series(
                                [1/len(st.session_state.selected_stocks)] * len(st.session_state.selected_stocks),
                                index=st.session_state.selected_stocks
                            )
                    except:
                        pass  # Continue without portfolio data
                
                # Use new intelligent handler
                response = handle_user_question(user_input, portfolio_weights, returns_data)
                
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            except Exception as e:
                response = f"❌ **Error**: {str(e)[:150]}"
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

# ==================== ML RANKING PAGE ====================

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def run_ml_ranking_cached(tickers):
    """Cached ML ranking to avoid redundant calculations"""
    from ml_prediction import MLStockRanker
    ranker = MLStockRanker()
    return ranker.rank_stocks(tickers)

def page_ml_ranking():
    st.title("🤖 AI Analysis")
    
    st.info("""
    **Machine Learning-Based Stock Ranking System**
    
    This page analyzes stocks using a hybrid scoring model that combines:
    - **Technical Features**: RSI, MACD, Bollinger Bands, momentum, moving averages
    - **Fundamental Features**: P/E ratio, earnings growth, ROE, dividend yield
    - **Market Features**: Beta, volatility
    
    **Output**: 0-100 ranking score + 1-month return prediction + buy/sell recommendation
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Stock selection
        st.markdown("### Select Stocks to Rank")
        selected_tickers = st.multiselect(
            "Choose stocks (or use your portfolio):",
            options=ALL_STOCKS,
            default=[],
            key="ml_stock_selection"
        )
        
        # Use portfolio button
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📊 Use My Portfolio", use_container_width=True):
                selected_tickers = st.session_state.selected_stocks
                st.rerun()
        
        with col_b:
            if st.button("⭐ Top 10 Holdings", use_container_width=True):
                selected_tickers = ALL_STOCKS[:10]
                st.rerun()
    
    with col2:
        st.markdown("### Quick Stats")
        if selected_tickers:
            st.metric("Selected Stocks", len(selected_tickers))
        else:
            st.warning("Select stocks to analyze")
    
    if selected_tickers:
        with st.spinner("🔄 Analyzing stocks with ML model..."):
            try:
                # Run ML ranking
                rankings_df = run_ml_ranking_cached(tuple(selected_tickers))
                
                # Display rankings table
                st.markdown("### 📊 Stock Rankings")
                
                # Sort by score descending
                rankings_df = rankings_df.sort_values("score", ascending=False).reset_index(drop=True)
                rankings_df["rank"] = range(1, len(rankings_df) + 1)
                
                # Display with formatting
                display_cols = ["rank", "ticker", "score", "predicted_return", "rsi", "momentum_1m", "recommendation"]
                if all(col in rankings_df.columns for col in display_cols):
                    display_df = rankings_df[display_cols].copy()
                    
                    # Format numeric columns
                    display_df["score"] = display_df["score"].apply(lambda x: f"{x:.1f}/100")
                    display_df["predicted_return"] = display_df["predicted_return"].apply(lambda x: f"{x:.2f}%")
                    display_df["momentum_1m"] = display_df["momentum_1m"].apply(lambda x: f"{x:.2f}%")
                    display_df["rsi"] = display_df["rsi"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "rank": st.column_config.NumberColumn("Rank", width="80px"),
                            "ticker": st.column_config.TextColumn("Ticker", width="100px"),
                            "score": st.column_config.TextColumn("ML Score", width="120px"),
                            "recommendation": st.column_config.TextColumn("Recommendation", width="150px"),
                            "predicted_return": st.column_config.TextColumn("Predicted 1M", width="130px"),
                            "rsi": st.column_config.TextColumn("RSI", width="80px"),
                            "momentum_1m": st.column_config.TextColumn("Momentum", width="100px"),
                        }
                    )
                    
                    # Visualization tabs
                    tab1, tab2, tab3 = st.tabs(["📈 Scores", "🎯 Recommendations", "📊 Features"])
                    
                    with tab1:
                        # Score distribution chart
                        fig = px.bar(
                            rankings_df.sort_values("score", ascending=True),
                            x="score",
                            y="ticker",
                            orientation="h",
                            color="score",
                            color_continuous_scale=["red", "yellow", "green"],
                            title="ML Ranking Scores",
                            labels={"score": "Score (0-100)", "ticker": "Stock"}
                        )
                        fig.update_layout(showlegend=False, height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Score statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Highest Score", f"{rankings_df['score'].max():.1f}")
                        with col2:
                            st.metric("Lowest Score", f"{rankings_df['score'].min():.1f}")
                        with col3:
                            st.metric("Average Score", f"{rankings_df['score'].mean():.1f}")
                        with col4:
                            buy_count = len(rankings_df[rankings_df['recommendation'].isin(['BUY', 'STRONG BUY'])])
                            st.metric("Buy Signals", buy_count)
                    
                    with tab2:
                        # Recommendation breakdown
                        rec_counts = rankings_df['recommendation'].value_counts()
                        fig = px.pie(
                            values=rec_counts.values,
                            names=rec_counts.index,
                            title="Recommendation Distribution",
                            color_discrete_map={
                                'STRONG BUY': '#00CC00',
                                'BUY': '#80DD80',
                                'HOLD': '#FFD700',
                                'SELL': '#FFA500',
                                'STRONG SELL': '#FF0000'
                            }
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        # Feature breakdown for each stock
                        st.markdown("#### Feature Breakdown by Stock")
                        
                        feature_cols = [col for col in rankings_df.columns if col not in 
                                       ['rank', 'ticker', 'score', 'recommendation', 'predicted_return']]
                        
                        if feature_cols:
                            selected_stock = st.selectbox(
                                "Select stock to view features:",
                                rankings_df['ticker'].values,
                                key="ml_feature_stock"
                            )
                            
                            stock_data = rankings_df[rankings_df['ticker'] == selected_stock].iloc[0]
                            
                            # Display features in columns
                            feature_data = []
                            for col in feature_cols:
                                if pd.notna(stock_data[col]):
                                    feature_data.append({
                                        "Feature": col.replace("_", " ").title(),
                                        "Value": f"{stock_data[col]:.2f}" if isinstance(stock_data[col], (int, float)) else str(stock_data[col])
                                    })
                            
                            if feature_data:
                                features_df = pd.DataFrame(feature_data)
                                st.dataframe(features_df, use_container_width=True, hide_index=True)
                    
                    # Detailed analysis section
                    st.markdown("### 📝 Detailed Analysis")
                    
                    analysis_col = st.selectbox(
                        "Select stock for detailed analysis:",
                        rankings_df['ticker'].values,
                        key="ml_analysis_stock"
                    )
                    
                    stock = rankings_df[rankings_df['ticker'] == analysis_col].iloc[0]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"#### {analysis_col}")
                        st.metric("ML Score", f"{stock['score']:.1f}/100")
                        st.metric("Recommendation", stock['recommendation'])
                    
                    with col2:
                        st.metric("Predicted 1M Return", f"{stock['predicted_return']:.2f}%")
                        if 'rsi' in stock and pd.notna(stock['rsi']):
                            st.metric("RSI (14)", f"{stock['rsi']:.1f}")
                    
                    with col3:
                        if 'momentum_1m' in stock and pd.notna(stock['momentum_1m']):
                            st.metric("1M Momentum", f"{stock['momentum_1m']:.2f}%")
                        if 'features' in stock and 'volatility_annual' in stock['features']:
                            vol = stock['features']['volatility_annual'] * 100
                            st.metric("Volatility (Annual)", f"{vol:.2f}%")
                    
                    # Recommendation explanation
                    rec = stock['recommendation']
                    score = stock['score']
                    
                    if rec == 'STRONG BUY':
                        st.success(f"🚀 **STRONG BUY** (Score: {score:.1f})\n\nThis stock shows exceptional fundamentals and technical setup. Strong buy signal.")
                    elif rec == 'BUY':
                        st.info(f"✅ **BUY** (Score: {score:.1f})\n\nThis stock presents good value with positive momentum. Consider adding to portfolio.")
                    elif rec == 'HOLD':
                        st.warning(f"⏸️ **HOLD** (Score: {score:.1f})\n\nThis stock is fairly valued. Hold if you own it, but not a strong buy signal.")
                    elif rec == 'SELL':
                        st.warning(f"⚠️ **SELL** (Score: {score:.1f})\n\nThis stock shows weakness. Consider reducing exposure.")
                    else:
                        st.error(f"🔴 **STRONG SELL** (Score: {score:.1f})\n\nThis stock shows significant downside risk. Avoid or exit position.")
                    
                    # Score Explanation Breakdown
                    st.markdown("### 🔍 What's Driving This Score?")
                    
                    if 'explanation' in stock and stock['explanation'] is not None:
                        explanation = stock['explanation']
                        
                        # Bullish factors
                        if explanation.get('bullish_factors'):
                            with st.expander("✅ Bullish Factors", expanded=True):
                                for factor in explanation['bullish_factors']:
                                    st.success(f"• {factor}")
                        
                        # Bearish factors
                        if explanation.get('bearish_factors'):
                            with st.expander("⚠️ Bearish Factors", expanded=True):
                                for factor in explanation['bearish_factors']:
                                    st.error(f"• {factor}")
                        
                        # Neutral factors
                        if explanation.get('neutral_factors'):
                            with st.expander("➖ Neutral Factors", expanded=False):
                                for factor in explanation['neutral_factors']:
                                    st.info(f"• {factor}")
                
            except Exception as e:
                st.error(f"❌ Error in ML ranking: {str(e)}")
                st.info("Please select stocks and try again. Make sure you have internet connection for data fetching.")


def page_modeling():
    """Financial Modeling & Valuation Analysis Page"""
    st.title("💰 Financial Modeling & Valuation")
    st.markdown("**DCF (Discounted Cash Flow) & Comparable Companies Analysis**")
    st.markdown("---")
    
    # Initialize session state for modeling
    if 'modeling_results' not in st.session_state:
        st.session_state.modeling_results = {}
    
    # ===== UI CONFIGURATION =====
    st.subheader("⚙️ Configure Analysis Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        # Stock selection
        valuation_source = st.radio("Stock Source", ["From Portfolio", "Custom Ticker"], key="model_source")
        
        if valuation_source == "From Portfolio":
            if len(st.session_state.selected_stocks) > 0:
                valuation_ticker = st.selectbox("Select Stock", st.session_state.selected_stocks, key="model_ticker_select")
            else:
                st.warning("📍 No stocks in portfolio. Please use 'Custom Ticker' or go to Portfolio Builder to add stocks.")
                valuation_ticker = None
        else:
            valuation_ticker = st.text_input("Enter Ticker Symbol", value="AAPL", key="model_ticker_input").upper()
    
    with col2:
        st.markdown("**Analysis Options**")
        run_dcf = st.checkbox("✓ Run DCF Analysis", value=True, key="model_dcf_check")
        run_comps = st.checkbox("✓ Run Comparable Companies", value=True, key="model_comps_check")
        auto_find_peers = st.checkbox("Auto-find peers (same sector)", value=True, key="model_peers_check",
                                      help="Automatically identify comparable companies")
    
    if not auto_find_peers and run_comps:
        peer_tickers_input = st.text_input("Enter Peer Tickers (comma-separated)", 
                                          placeholder="e.g., MSFT,GOOGL,META,AMZN", key="model_peers_input")
        peer_tickers = [t.strip().upper() for t in peer_tickers_input.split(',') if t.strip()] if peer_tickers_input else None
    else:
        peer_tickers = None
    
    # ===== DCF ASSUMPTIONS =====
    st.markdown("### 💼 DCF Model Assumptions")
    st.markdown("*Adjust parameters to test different valuation scenarios*")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        wacc_input = st.number_input("WACC (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.5,
                                    help="Weighted Average Cost of Capital - discount rate for future cash flows", 
                                    key="model_wacc")
        wacc = wacc_input / 100
        
        terminal_growth = st.number_input("Terminal Growth (%)", min_value=0.0, max_value=5.0, 
                                        value=2.5, step=0.1,
                                        help="Long-term perpetual growth rate (typically 2-3%)", 
                                        key="model_terminal")
        terminal_growth = terminal_growth / 100
    
    with col2:
        revenue_growth = st.number_input("Revenue Growth (%)", min_value=-20.0, max_value=50.0, 
                                       value=None, step=1.0,
                                       help="Leave empty to auto-calculate from historical data", 
                                       key="model_revenue")
        if revenue_growth is not None:
            revenue_growth = revenue_growth / 100
        
        forecast_years = st.number_input("Forecast Period (years)", min_value=3, max_value=10, 
                                       value=5, step=1, key="model_forecast")
    
    with col3:
        risk_free_rate = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, 
                                       value=4.0, step=0.1,
                                       help="US Treasury 10-year yield", 
                                       key="model_rf")
        risk_free_rate = risk_free_rate / 100
        
        market_risk_premium = st.number_input("Market Risk Premium (%)", min_value=0.0, max_value=15.0, 
                                            value=7.0, step=0.5,
                                            help="Expected market return - risk-free rate", 
                                            key="model_mrp")
        market_risk_premium = market_risk_premium / 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        tax_rate = st.number_input("Corporate Tax Rate (%)", min_value=0.0, max_value=50.0, 
                                  value=21.0, step=1.0, key="model_tax")
        tax_rate = tax_rate / 100
    
    with col2:
        cost_of_debt = st.number_input("Cost of Debt (%)", min_value=0.0, max_value=20.0, 
                                      value=5.0, step=0.5, key="model_debt_cost")
        cost_of_debt = cost_of_debt / 100
    
    with col3:
        st.markdown("*Cost of Debt helps calculate WACC*")
    
    st.markdown("---")
    
    # ===== RUN ANALYSIS =====
    if st.button("🔍 Run Valuation Analysis", use_container_width=True, key="model_run_button"):
        if not valuation_ticker:
            st.error("❌ Please enter or select a valid ticker symbol")
        else:
            try:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.info("⏳ Initializing valuation engines...")
                progress_bar.progress(5)
                
                dcf_result = None
                comps_result = None
                
                # ===== RUN DCF ANALYSIS =====
                if run_dcf:
                    status_text.info("⚡ Running DCF Analysis...")
                    progress_bar.progress(25)
                    
                    try:
                        dcf_analysis = DCFValuation(ticker=valuation_ticker)
                        dcf_result = dcf_analysis.run_analysis(
                            wacc=wacc,
                            terminal_growth=terminal_growth,
                            forecast_years=int(forecast_years),
                            risk_free_rate=risk_free_rate,
                            market_risk_premium=market_risk_premium,
                            tax_rate=tax_rate,
                            cost_of_debt=cost_of_debt,
                            revenue_growth=revenue_growth
                        )
                        progress_bar.progress(60)
                    except Exception as e:
                        st.error(f"❌ DCF Analysis Error: {str(e)}")
                        dcf_result = None
                
                # ===== RUN COMPS ANALYSIS =====
                if run_comps:
                    status_text.info("⚡ Running Comparable Companies Analysis...")
                    progress_bar.progress(75)
                    
                    try:
                        comps_analysis = ComparableCompanies(ticker=valuation_ticker)
                        comps_result = comps_analysis.run_analysis(
                            peer_tickers=peer_tickers,
                            auto_find_peers=auto_find_peers
                        )
                        progress_bar.progress(90)
                    except Exception as e:
                        st.warning(f"⚠️ Comps Analysis: {str(e)}")
                        comps_result = None
                
                status_text.success("✅ Analysis Complete!")
                progress_bar.progress(100)
                import time
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # ===== DISPLAY RESULTS =====
                st.markdown("---")
                
                # Check if DCF analysis succeeded
                if dcf_result and dcf_result.get('success'):
                    st.subheader("📊 Valuation Summary")
                    
                    # Create ValuationReport for formatting
                    report = ValuationReport(dcf_result, comps_result)
                    status_text_report = report.get_valuation_status()
                    
                    # Gauge & Metrics
                    col_gauge, col_metrics = st.columns([1.2, 1])
                    
                    with col_gauge:
                        upside_dcf = dcf_result.get('upside_pct', 0)
                        intrinsic_value = dcf_result.get('intrinsic_value', 0)
                        
                        # Create gauge chart showing upside percentage
                        gauge_fig = go.Figure(data=[go.Indicator(
                            mode="gauge+number+delta",
                            value=upside_dcf,
                            number={'suffix': "%", 'font': {'size': 28}},
                            title={'text': "DCF Upside/Downside"},
                            delta={'reference': 0, 'suffix': "%", 'font': {'size': 16}},
                            gauge={
                                'axis': {'range': [-50, 50]},
                                'bar': {'color': "darkblue" if upside_dcf > 15 else "orange" if upside_dcf > -15 else "darkred"},
                                'steps': [
                                    {'range': [-50, -15], 'color': "#ffcccc"},
                                    {'range': [-15, 15], 'color': "#ffffcc"},
                                    {'range': [15, 50], 'color': "#ccffcc"},
                                ],
                                'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': 0}
                            },
                            domain={'x': [0, 1], 'y': [0, 1]}
                        )])
                        
                        # Add annotation for intrinsic value at the center
                        gauge_fig.add_annotation(
                            text=f"<b>IV: ${intrinsic_value:.2f}</b>",
                            x=0.5, y=0.35,
                            showarrow=False,
                            font=dict(size=16, color="black"),
                            xref="paper", yref="paper"
                        )
                        
                        gauge_fig.update_layout(height=350, margin=dict(l=10, r=10, t=60, b=10))
                        st.plotly_chart(gauge_fig, use_container_width=True)
                        
                        # Status text
                        if upside_dcf > 15:
                            st.success("✅ **UNDERPRICED** - Trading below intrinsic value")
                        elif upside_dcf > -15:
                            st.info("≈ **FAIRLY VALUED** - Near intrinsic value")
                        else:
                            st.error("⚠️ **OVERPRICED** - Trading above intrinsic value")
                    
                    with col_metrics:
                        st.markdown("### Key Metrics")
                        st.metric("Current Price", f"${dcf_result.get('current_price', 0):.2f}")
                        st.metric("DCF Intrinsic Value", f"${dcf_result.get('intrinsic_value', 0):.2f}", 
                                 delta=f"{upside_dcf:+.1f}%")
                        if comps_result and comps_result.get('avg_implied_value'):
                            st.metric("Comps Implied Value", f"${comps_result.get('avg_implied_value', 0):.2f}",
                                     delta=f"{comps_result.get('upside_pct', 0):+.1f}%")
                    
                    st.markdown("---")
                    
                    # Detailed Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Enterprise Value", f"${dcf_result.get('enterprise_value', 0)/1e9:.2f}B")
                        st.caption("Total firm value")
                    with col2:
                        st.metric("Equity Value", f"${dcf_result.get('equity_value', 0)/1e9:.2f}B")
                        st.caption("Equity value to shareholders")
                    with col3:
                        st.metric("Terminal Value PV", f"${dcf_result.get('terminal_pv', 0)/1e9:.2f}B")
                        st.caption("Present value of terminal value")
                    
                    # ===== CHARTS & VISUALIZATIONS =====
                    st.markdown("### 📈 Valuation Details")
                    
                    # Projections chart
                    if dcf_result.get('projections'):
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            proj_df = pd.DataFrame(dcf_result['projections'])
                            fig_fcf = px.bar(proj_df, x='year', y='fcf', 
                                           title='Free Cash Flow Projections',
                                           labels={'fcf': 'FCF ($B)', 'year': 'Year'},
                                           color_discrete_sequence=['#1f77b4'])
                            fig_fcf.update_layout(height=350, hovermode='x unified')
                            st.plotly_chart(fig_fcf, use_container_width=True)
                        
                        with col_chart2:
                            # PV breakdown
                            fcf_pv = sum([p.get('pv', 0) for p in dcf_result['projections']])
                            terminal_pv = dcf_result.get('terminal_pv', 0)
                            
                            pv_data = {
                                'Component': ['Projected FCF PV', 'Terminal Value PV'],
                                'Value': [fcf_pv, terminal_pv]
                            }
                            fig_pv = px.pie(values=pv_data['Value'], names=pv_data['Component'],
                                          title='Enterprise Value Components')
                            fig_pv.update_layout(height=350)
                            st.plotly_chart(fig_pv, use_container_width=True)
                    
                    # Sensitivity analysis
                    st.markdown("### 🎯 Sensitivity Analysis")
                    
                    try:
                        sensitivity = dcf_analysis.generate_sensitivity_analysis(
                            wacc, terminal_growth, int(forecast_years), revenue_growth
                        )
                        if sensitivity is not None and len(sensitivity) > 0:
                            # Get pivoted table for better visualization
                            display_df, numeric_df = dcf_analysis.get_sensitivity_table_pivoted()
                            
                            if not display_df.empty:
                                st.markdown("**WACC (left axis) vs Revenue Growth (top axis) — Intrinsic Value**")
                                
                                # Display the pivot table
                                st.dataframe(
                                    display_df, 
                                    use_container_width=True,
                                    height=400
                                )
                                
                                # Additional info
                                col_sens_info1, col_sens_info2 = st.columns(2)
                                with col_sens_info1:
                                    assumptions = dcf_result.get('assumptions', {})
                                    st.markdown("**Current Base Assumptions**")
                                    st.write(f"• WACC: {assumptions.get('wacc', 0)*100:.2f}%")
                                    st.write(f"• Revenue Growth: {assumptions.get('revenue_growth', 0)*100:.1f}%")
                                    st.write(f"• Terminal Growth: {assumptions.get('terminal_growth', 0)*100:.1f}%")
                                
                                with col_sens_info2:
                                    st.markdown("**Table Legend**")
                                    st.write(f"🔵 Base WACC: {wacc*100:.2f}%")
                                    st.write(f"🟢 Base Growth: {revenue_growth*100:.1f}%")
                                    st.write(f"Range: ±3% around base values")
                            else:
                                st.info("🔄 Sensitivity table generation in progress...")
                        else:
                            st.info("🔄 Generating sensitivity analysis...")
                    except Exception as e:
                        st.info(f"🔄 Sensitivity analysis in progress... ({str(e)[:50]})")
                        st.write(f"• Tax Rate: {assumptions.get('tax_rate', 0)*100:.1f}%")
                        st.write(f"• Cost of Debt: {assumptions.get('cost_of_debt', 0)*100:.1f}%")
                    
                    st.markdown("---")
                    
                    # Assumptions table
                    st.markdown("### 📋 Detailed Assumptions")
                    assumptions_table = report.get_assumptions_table()
                    st.write(assumptions_table)
                    
                    st.markdown("---")
                    
                    # Download results
                    summary_data = {
                        'Metric': [
                            'Current Price', 'DCF Intrinsic Value', 'Upside/Downside %',
                            'Enterprise Value', 'WACC', 'Revenue Growth', 'Terminal Growth',
                            'Forecast Years', 'Cost of Debt', 'Tax Rate'
                        ],
                        'Value': [
                            f"${dcf_result.get('current_price', 0):.2f}",
                            f"${dcf_result.get('intrinsic_value', 0):.2f}",
                            f"{dcf_result.get('upside_pct', 0):.2f}%",
                            f"${dcf_result.get('enterprise_value', 0)/1e9:.3f}B",
                            f"{assumptions.get('wacc', 0)*100:.2f}%",
                            f"{assumptions.get('revenue_growth', 0)*100:.2f}%",
                            f"{assumptions.get('terminal_growth', 0)*100:.2f}%",
                            f"{assumptions.get('forecast_years', 0)} years",
                            f"{assumptions.get('cost_of_debt', 0)*100:.2f}%",
                            f"{assumptions.get('tax_rate', 0)*100:.2f}%"
                        ]
                    }
                    
                    summary_df = pd.DataFrame(summary_data)
                    csv_data = summary_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Valuation Summary (CSV)",
                        data=csv_data,
                        file_name=f"{valuation_ticker}_valuation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                else:
                    if dcf_result and dcf_result.get('error'):
                        st.error(f"❌ {dcf_result.get('error')}")
                    else:
                        st.error("❌ Valuation analysis failed. Please check ticker and try again.")
                    st.info("💡 Tip: Make sure the ticker is valid and has available financial data.")
            
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("Please try again or check your internet connection.")
    
    # ===== INFORMATION SECTION =====
    st.markdown("---")
    with st.expander("📖 About DCF & Comparable Analysis"):
        st.markdown("""
        ## Valuation Methods Explained
        
        ### **DCF (Discounted Cash Flow) Model**
        - **Methodology**: Project future free cash flows and discount them to present value using WACC
        - **Key Inputs**: 
          - Revenue Growth Rate: Expected growth in company revenue
          - WACC: Discount rate reflecting cost of capital
          - Terminal Growth: Long-term growth after forecast period
        - **Best For**: Mature, profitable companies with stable cash flows
        - **Strengths**: Fundamental approach; reflects intrinsic value
        - **Limitations**: Highly sensitive to growth assumptions; difficult for unprofitable firms
        
        ### **Comparable Companies (Comps) Analysis**
        - **Methodology**: Compare target company's multiples (P/E, EV/EBITDA) to similar peers
        - **Key Metrics**: Price-to-Earnings, Enterprise Value/EBITDA, Price-to-Book
        - **Best For**: Relative valuation; benchmarking vs industry peers  
        - **Strengths**: Market-based; easy to understand
        - **Limitations**: Depends on similar companies; assumes market is rational
        
        ### **How to Use This Tool**
        1. Select a stock from your portfolio or enter a custom ticker
        2. Adjust DCF assumptions based on your investment thesis
        3. Compare DCF intrinsic value with current price
        4. Use Comparable Companies analysis for validation
        5. Download results for your investment analysis
        """)


# ==================== MAIN APP ====================

def main():
    # Sidebar navigation
    with st.sidebar:
        st.markdown("# 📊 Portfolio Optimizer")
        st.markdown("*Advanced Portfolio Construction & Analysis*")
        st.divider()
        
        # Clean navigation menu
        st.markdown("### Navigation")
        page = st.radio(
            "Select Page:",
            ["🏠 Home", "📈 Holdings", "📊 Backtest", "💰 Modeling", "🤖 AI Analysis", "💬 Chatbot", "📚 Description"],
            label_visibility="collapsed",
            key="page_selection"
        )
        
        # Map radio selection to page names
        page_map = {
            "🏠 Home": "Home",
            "📈 Holdings": "Holdings",
            "📊 Backtest": "Backtest",
            "💰 Modeling": "Modeling",
            "🤖 AI Analysis": "AI Analysis",
            "💬 Chatbot": "Chatbot",
            "📚 Description": "Description"
        }
        selected_page = page_map.get(page, "Home")
        
        st.divider()
        
        # Model information
        st.markdown("### 8 Models")
        models_info = [
            ("GMV", "Minimize volatility"),
            ("CAPM", "Risk-adjusted returns"),
            ("BL", "Market equilibrium"),
            ("HRP", "Hierarchical clustering"),
            ("CVaR", "Tail risk control"),
            ("LASSO", "Sparse portfolio"),
            ("RL", "Reinforcement learning"),
            ("DRO", "Robust optimization")
        ]
        
        with st.expander("📋 Model Details", expanded=False):
            for model, desc in models_info:
                st.caption(f"**{model}**: {desc}")
        
        st.divider()
        
        # Information section
        st.markdown("### 📡 Data Source")
        st.caption("**40,000+ global securities** via Yahoo Finance")
        st.caption(f"**Risk-Free Rate**: {RISK_FREE_RATE*100:.2f}% (US 10Y Treasury)")
        
        # Initialize page state
        if 'page' not in st.session_state:
            st.session_state.page = "Home"
        else:
            st.session_state.page = selected_page
        
        page = st.session_state.page
    
    if page == "Home":
        page_home()
    elif page == "Description":
        page_description()
    elif page == "Holdings":
        page_holdings()
    elif page == "Backtest":
        page_backtest()
    elif page == "Modeling":
        page_modeling()
    elif page == "AI Analysis":
        page_ml_ranking()
    elif page == "Chatbot":
        page_chatbot()

if __name__ == "__main__":
    main()
