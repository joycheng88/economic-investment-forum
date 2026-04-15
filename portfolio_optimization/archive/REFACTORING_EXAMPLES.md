# APP.PY REFACTORING EXAMPLES

## Key Refactoring Examples

This document shows concrete before/after examples of how to use the new `ui_utils.py` module to streamline app.py.

---

## EXAMPLE 1: Session State Initialization

### ❌ BEFORE (6 separate checks)
```python
# Before: Scattered throughout the code
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False

if 'selected_stocks' not in st.session_state:
    st.session_state['selected_stocks'] = []

if 'selected_date_range' not in st.session_state:
    st.session_state['selected_date_range'] = None

if 'custom_sector_mapping' not in st.session_state:
    st.session_state['custom_sector_mapping'] = {}

if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

if 'ml_model_trained' not in st.session_state:
    st.session_state['ml_model_trained'] = False
```

### ✅ AFTER (1 centralized call)
```python
from ui_utils import initialize_session_state

# In main():
initialize_session_state()

# That's it! All defaults are set in one place.
```

**Lines saved:** 8 lines → 1 function call  
**Benefit:** Single source of truth for defaults, easier to add new state items

---

## EXAMPLE 2: Metric Display

### ❌ BEFORE (22 st.metric() calls, ~50 lines)
```python
# Example from DCF analysis page
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Current Price", f"${current_price:.2f}", delta=None)
with col2:
    st.metric("Intrinsic Value", f"${intrinsic_value:.2f}", delta=delta_value)
with col3:
    st.metric("Upside", f"{upside_pct:.1f}%", delta=None)
with col4:
    st.metric("FCF Yield", f"{fcf_yield:.2%}", delta=None)

# ... repeated 5+ more times with different metrics
```

### ✅ AFTER (1 function call)
```python
from ui_utils import display_metrics_row

metrics = {
    "Current Price": (f"${current_price:.2f}", None),
    "Intrinsic Value": (f"${intrinsic_value:.2f}", delta_value),
    "Upside": (f"{upside_pct:.1f}%", None),
    "FCF Yield": (f"{fcf_yield:.2%}", None),
}
display_metrics_row(metrics, cols=4)
```

**Lines saved:** ~50 lines → ~5 lines  
**Benefit:** Cleaner code, consistent metric styling, easy to adjust number of columns

---

## EXAMPLE 3: Error Handling

### ❌ BEFORE (19 try-except blocks, ~60 lines)
```python
# Portfolio 1 analysis
try:
    st.write("Running analysis...")
    results = run_gvw_optimization()
    st.success("✅ Analysis complete")
    st.dataframe(results)
except Exception as e:
    st.error(f"❌ Analysis failed: {str(e)}")

# Portfolio 2 analysis  
try:
    st.write("Running analysis...")
    results = run_cla_optimization()
    st.success("✅ Analysis complete")
    st.dataframe(results)
except Exception as e:
    st.error(f"❌ Analysis failed: {str(e)}")

# ... repeated 17 more times
```

### ✅ AFTER (3 lines with error handling wrapper)
```python
from ui_utils import run_with_error_handling, display_dataframe_with_settings

# Portfolio 1
success, results = run_with_error_handling(run_gvw_optimization)
if success:
    display_dataframe_with_settings(results)

# Portfolio 2
success, results = run_with_error_handling(run_cla_optimization)
if success:
    display_dataframe_with_settings(results)
```

**Lines saved:** ~60 lines → ~12 lines  
**Benefit:** Consistent error handling, cleaner code, easier to maintain

---

## EXAMPLE 4: Page Navigation

### ❌ BEFORE (8 if-elif chains, ~20 lines)
```python
# Sidebar page selection
page = st.sidebar.selectbox(
    "Choose Page",
    ["🏠 Home", "📈 Holdings", "🔄 Pairs Trading", "💡 Modeling", 
     "📊 Historical Analysis", "🎯 Backtesting", "🤖 Chatbot"]
)

# Manual page routing
if page == "🏠 Home":
    page = "Home"
    page_home()
elif page == "📈 Holdings":
    page = "Holdings"
    page_holdings()
elif page == "🔄 Pairs Trading":
    page = "Pairs Trading"
    page_pairs_trading()
# ... etc
```

### ✅ AFTER (Uses PAGE_REGISTRY, ~5 lines)
```python
from ui_utils import PAGE_REGISTRY, render_sidebar_navigation

# Get page from sidebar (uses PAGE_REGISTRY internally)
selected_page = render_sidebar_navigation()

# Direct dispatch - no if-elif chains
if selected_page == "Home":
    page_home()
elif selected_page == "Holdings":
    page_holdings()
# ... much cleaner
```

**Lines saved:** ~20 lines → ~8 lines (with better structure)  
**Benefit:** Central registry makes it easy to add/remove pages, consistent naming

---

## EXAMPLE 5: Progress Tracking

### ❌ BEFORE (Complex progress management, ~40 lines)
```python
progress_bar = st.progress(0)
status_text = st.empty()

progress_bar.progress(0, text="Starting analysis...")
status_text.info("Initializing...")

try:
    # Step 1
    results1 = load_data()
    progress_bar.progress(25, text="Data loaded...")
    st.write("✅ Data loaded")
    
    # Step 2
    results2 = run_analysis(results1)
    progress_bar.progress(50, text="Running analysis...")
    st.write("✅ Analysis running")
    
    # Step 3
    results3 = process_results(results2)
    progress_bar.progress(75, text="Processing results...")
    st.write("✅ Results processed")
    
    # Step 4
    visualize_results(results3)
    progress_bar.progress(100, text="Complete!")
    st.success("✅ Analysis complete!")
    
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
```

### ✅ AFTER (Streamlined with helpers, ~15 lines)
```python
from ui_utils import create_progress_tracker, update_progress, finalize_progress

progress, status = create_progress_tracker("Starting analysis...")

try:
    results1 = load_data()
    update_progress(progress, status, 25, "Data loaded")
    
    results2 = run_analysis(results1)
    update_progress(progress, status, 50, "Running analysis")
    
    results3 = process_results(results2)
    update_progress(progress, status, 75, "Processing results")
    
    visualize_results(results3)
    finalize_progress(progress, status)
    
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
```

**Lines saved:** ~40 lines → ~15 lines  
**Benefit:** Much cleaner code flow, consistent progress tracking pattern

---

## EXAMPLE 6: Data Display (Holdings Page)

### ❌ BEFORE (Repeated display code, ~60 lines)
```python
st.subheader("Portfolio Holdings")

# Display with manual settings
st.dataframe(
    holdings_df,
    use_container_width=True,
    hide_index=False,
)

st.subheader("Performance Summary")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Return", f"{total_return:.2%}")
with col2:
    st.metric("Annual Return", f"{annual_return:.2%}")
with col3:
    st.metric("Max Drawdown", f"{max_dd:.2%}")
with col4:
    st.metric("Sharpe Ratio", f"{sharpe:.2f}")

st.subheader("Risk Metrics")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Volatility", f"{volatility:.2%}")
with col2:
    st.metric("Beta", f"{beta:.2f}")
with col3:
    st.metric("VaR (95%)", f"{var_95:.2%}")
```

### ✅ AFTER (Using unified helpers, ~10 lines)
```python
from ui_utils import display_dataframe_with_settings, display_performance_summary

st.subheader("Portfolio Holdings")
display_dataframe_with_settings(holdings_df)

st.subheader("Performance Summary")
display_performance_summary({
    'total': total_return,
    'annual': annual_return,
    'max_dd': max_dd,
    'sharpe': sharpe,
})

st.subheader("Risk Metrics")
risk_metrics = {
    "Volatility": (f"{volatility:.2%}", None),
    "Beta": (f"{beta:.2f}", None),
    "VaR (95%)": (f"{var_95:.2%}", None),
}
display_metrics_row(risk_metrics, cols=3)
```

**Lines saved:** ~60 lines → ~15 lines  
**Benefit:** Consistent styling, easier to update appearance globally

---

## MIGRATION CHECKLIST

When integrating these changes into app.py:

- [ ] Add `from ui_utils import *` at top of app.py
- [ ] Call `initialize_session_state()` in `main()` function
- [ ] Replace page navigation with `render_sidebar_navigation()`
- [ ] Replace all `st.metric()` groups with `display_metrics_row()`
- [ ] Replace try-except blocks with `run_with_error_handling()`
- [ ] Replace progress bar patterns with helper functions
- [ ] Replace dataframe display calls with `display_dataframe_with_settings()`
- [ ] Test each page: Home, Holdings, Pairs Trading, Modeling, Historical, Backtesting, Chatbot
- [ ] Verify all error messages display correctly
- [ ] Check that all data loads properly

---

## TESTING STRATEGY

1. **Unit Level:** Test each ui_utils function independently
   ```python
   # Test display_metrics_row
   metrics = {"Price": ("$100", "5%")}
   display_metrics_row(metrics)
   ```

2. **Page Level:** Run each page and verify output
   ```bash
   streamlit run app.py
   # Click through each page, verify it looks identical
   ```

3. **Integration:** Full end-to-end test
   - Load data
   - Run analyses
   - Check error handling
   - Verify progress tracking

4. **Regression:** Compare before/after screenshots
   - Should be visually identical
   - Functionality should be identical

---

## PERFORMANCE NOTES

No measurable performance change expected:
- Utility functions add negligible overhead
- Code is cleaner but not faster
- Caching already implemented elsewhere
- Should feel slightly faster due to reduced screen flicker

---

## ROLLBACK PLAN

If issues occur, simply:
1. Don't import from ui_utils
2. Keep original app.py code
3. No dependencies on ui_utils are required

The utility module is optional - existing code works fine without it.

---

## FUTURE ENHANCEMENTS

After initial refactoring, consider:

1. **Component-based UI**
   ```python
   class PortfolioCard:
       def __init__(self, title, returns, risk):
           self.title = title
       
       def render(self):
           # Render card with consistent styling
   ```

2. **Theme customization**
   ```python
   THEME = {
       'primary_color': '#1f77b4',
       'metric_font_size': 14,
   }
   ```

3. **Localization**
   ```python
   TRANSLATIONS = {
       'en': {'Home': 'Home', 'Holdings': 'Holdings'},
       'es': {'Home': 'Inicio', 'Holdings': 'Cartera'},
   }
   ```

4. **Accessibility improvements**
   - Better ARIA labels
   - Keyboard navigation
   - High contrast mode

---

**Ready to refactor? Start with Example 1 (Session State Initialization) - it's the quickest win!**
