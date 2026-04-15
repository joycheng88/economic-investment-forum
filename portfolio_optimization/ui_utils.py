"""
Streamlit UI Utilities Module
Consolidates common UI patterns and helper functions for app.py
"""

import streamlit as st
from contextlib import contextmanager
from typing import Dict, Any, Tuple, Optional, Callable


# ============================================================================
# CONSTANTS & REGISTRIES
# ============================================================================

PAGE_REGISTRY = {
    "🏠 Home": "Home",
    "📈 Holdings": "Holdings",
    "🔄 Pairs Trading": "Pairs Trading",
    "💡 Modeling": "Modeling",
    "📊 Historical Analysis": "Historical Analysis",
    "🎯 Backtesting": "Backtesting",
    "🤖 Chatbot": "Chatbot",
}

MODELS_INFO = [
    ("GMV", "Minimize volatility"),
    ("CLA", "Maximize Sharpe ratio"),
    ("ERC", "Equal risk contribution"),
    ("RPB", "Risk parity"),
    ("MHP", "Max Sharpe hierarchical"),
    ("HRP", "Hierarchical risk parity"),
    ("BL", "Black-Litterman"),
    ("Lasso", "Sparse allocation"),
]


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Centralized session state initialization with all defaults."""
    defaults = {
        'data_loaded': False,
        'selected_stocks': [],
        'selected_date_range': None,
        'custom_sector_mapping': {},
        'page': 'Home',
        'ml_model_trained': False,
        'backtesting_results': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# METRIC DISPLAY HELPERS
# ============================================================================

def display_metrics_row(metrics: Dict[str, Tuple[str, Optional[str]]], 
                        cols: int = 4):
    """
    Display metrics in a clean grid layout.
    
    Args:
        metrics: Dict[metric_name, (value, delta)] where delta is optional
        cols: Number of columns for the grid
    
    Example:
        metrics = {
            "Current Price": ("$370.44", None),
            "Intrinsic Value": ("$182.35", "-50.8%"),
            "Upside": ("81%", None),
        }
        display_metrics_row(metrics, cols=3)
    """
    metric_items = list(metrics.items())
    
    for i in range(0, len(metric_items), cols):
        columns = st.columns(cols)
        for j, col in enumerate(columns):
            if i + j < len(metric_items):
                key, (value, delta) = metric_items[i + j]
                with col:
                    st.metric(key, value, delta=delta)


def display_single_metric(label: str, value: Any, delta: Optional[str] = None, 
                          col_width: float = 1.0):
    """Display a single metric with optional column width control."""
    with st.container():
        st.metric(label, value, delta=delta)


# ============================================================================
# COLUMN LAYOUT HELPERS
# ============================================================================

def create_two_column_layout(ratio: Tuple[float, float] = (1, 1)):
    """Create two columns with specified width ratio."""
    return st.columns(ratio)


def create_three_column_layout(ratio: Tuple[float, float, float] = (1, 1, 1)):
    """Create three columns with specified width ratio."""
    return st.columns(ratio)


def create_gauge_metrics_layout():
    """Create standard layout for gauge chart + metrics (1.2:1 ratio)."""
    return st.columns([1.2, 1])


# ============================================================================
# ERROR HANDLING & PROGRESS TRACKING
# ============================================================================

@contextmanager
def safe_run_analysis(analysis_name: str):
    """
    Context manager for safe error handling in analyses.
    
    Usage:
        with safe_run_analysis("DCF Analysis") as (progress, status):
            # Your analysis code here
            progress.update(25, "Running valuation...")
            # ...
            progress.update(100, "Complete!")
    """
    try:
        progress = st.progress(0, text=f"Starting {analysis_name}...")
        status = st.empty()
        yield progress, status
    except Exception as e:
        st.error(f"❌ {analysis_name} failed: {str(e)}")
        raise
    finally:
        pass


def run_with_error_handling(func: Callable, *args, **kwargs) -> Tuple[bool, Any]:
    """
    Execute a function with error handling.
    
    Returns:
        (success: bool, result: Any) - (True, result) on success or (False, None) on error
    
    Example:
        success, result = run_with_error_handling(dcf_analysis.run_analysis, **params)
        if success:
            st.write(result)
        else:
            st.error("Analysis failed")
    """
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return False, None


def create_progress_tracker(label: str = "Processing...") -> Tuple:
    """
    Create a progress bar and status text container.
    
    Returns:
        (progress_bar, status_container)
    """
    progress_bar = st.progress(0, text=label)
    status_container = st.empty()
    return progress_bar, status_container


def update_progress(progress_bar, status_container, 
                   percentage: int, message: str):
    """Update progress bar and status message."""
    progress_bar.progress(percentage, text=message)
    status_container.info(f"📝 {message}")


def finalize_progress(progress_bar, status_container, message: str = "Complete! ✅"):
    """Finalize progress bar."""
    progress_bar.progress(100, text=message)
    status_container.success(f"✅ {message}")


# ============================================================================
# DATA & MODEL UTILITIES
# ============================================================================

def display_model_comparison_header():
    """Display model comparison header with info."""
    st.markdown("### 📊 Model Comparison & Optimization")
    st.info("Select a model and customize parameters to find optimal portfolio allocation.")


def display_optimization_results(results: Dict[str, Any]):
    """Display optimization results in a standard format."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Expected Return", f"{results.get('return', 0):.2%}")
    with col2:
        st.metric("Volatility", f"{results.get('volatility', 0):.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{results.get('sharpe', 0):.2f}")


# ============================================================================
# SIDEBAR HELPERS
# ============================================================================

def render_sidebar_navigation() -> str:
    """
    Render sidebar navigation using PAGE_REGISTRY.
    
    Returns:
        selected_page: The selected page key
    """
    with st.sidebar:
        st.title("Navigation")
        selected = st.selectbox(
            "Choose Page",
            options=list(PAGE_REGISTRY.keys()),
            label_visibility="collapsed"
        )
        
        st.divider()
        st.markdown("---")
        
        return PAGE_REGISTRY[selected]


def render_sidebar_filters():
    """Render common filters in sidebar."""
    with st.sidebar:
        st.subheader("⚙️ Filters & Settings")
        
        export_format = st.selectbox(
            "Export Format",
            ["CSV", "Excel", "JSON"]
        )
        
        return export_format


# ============================================================================
# DATA DISPLAY HELPERS
# ============================================================================

def display_dataframe_with_settings(df, key: str = "", **kwargs):
    """
    Display dataframe with standard settings.
    
    Args:
        df: DataFrame to display
        key: Unique key for the dataframe
        **kwargs: Additional arguments for st.dataframe()
    """
    default_kwargs = {
        'use_container_width': True,
        'hide_index': False,
    }
    default_kwargs.update(kwargs)
    
    st.dataframe(df, key=key, **default_kwargs)


def display_performance_summary(returns: Dict[str, float]):
    """Display a summary of performance metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_summary = [
        ("Total Return", f"{returns.get('total', 0):.2%}"),
        ("Annual Return", f"{returns.get('annual', 0):.2%}"),
        ("Max Drawdown", f"{returns.get('max_dd', 0):.2%}"),
        ("Sharpe Ratio", f"{returns.get('sharpe', 0):.2f}"),
    ]
    
    columns = [col1, col2, col3, col4]
    for col, (label, value) in zip(columns, metrics_summary):
        with col:
            st.metric(label, value)


# ============================================================================
# EXPANDER HELPERS
# ============================================================================

def expander_with_info(title: str, info_text: str):
    """Create an expander with info icon."""
    return st.expander(f"📌 {title} - {info_text}")


# ============================================================================
# LOADING & CACHING DECORATORS
# ============================================================================

def with_loading_spinner(func: Callable, label: str = "Loading...") -> Callable:
    """Decorator to add loading spinner to a function."""
    def wrapper(*args, **kwargs):
        with st.spinner(label):
            return func(*args, **kwargs)
    return wrapper


# ============================================================================
# STANDARD LAYOUTS
# ============================================================================

class StandardLayouts:
    """Collection of standard UI layouts."""
    
    @staticmethod
    def metric_with_chart():
        """Layout: Left gauge chart, right metrics."""
        return st.columns([1.2, 1])
    
    @staticmethod
    def three_metrics():
        """Layout: Three equal metrics."""
        return st.columns(3)
    
    @staticmethod
    def sidebar_main():
        """Layout: Sidebar + main content (automatic)."""
        pass  # Streamlit handles this automatically
    
    @staticmethod
    def main_with_tabs():
        """Layout: Main content with multiple tabs."""
        return st.tabs


# ============================================================================
# DATE & TIME HELPERS
# ============================================================================

def get_date_range_selector():
    """Get date range from user."""
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date")
    with col2:
        end_date = st.date_input("End Date")
    
    return start_date, end_date


def date_range_to_period(start_date, end_date: str) -> str:
    """Convert date range to period string."""
    return f"{start_date} to {end_date}"


# ============================================================================
# ALERT & NOTIFICATION HELPERS
# ============================================================================

def show_success_message(message: str):
    """Display success message."""
    st.success(f"✅ {message}")


def show_error_message(message: str):
    """Display error message."""
    st.error(f"❌ {message}")


def show_warning_message(message: str):
    """Display warning message."""
    st.warning(f"⚠️ {message}")


def show_info_message(message: str):
    """Display info message."""
    st.info(f"ℹ️ {message}")
