# APP.PY REFACTORING QUICK REFERENCE CARD

**Print this page and keep it handy while refactoring app.py**

---

## 🚀 QUICK START

```python
# At top of app.py, add this import:
from ui_utils import (
    initialize_session_state,
    display_metrics_row,
    run_with_error_handling,
    create_progress_tracker,
    update_progress,
    finalize_progress,
)

# In main(), replace session state checks with:
initialize_session_state()
```

---

## 🔀 FIND & REPLACE PATTERNS

### PATTERN 1: Metrics (Search: "st.columns(4)")
```python
# ❌ FIND THIS:
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Price", price)
with col2:
    st.metric("Value", value)
with col3:
    st.metric("Return", ret)
with col4:
    st.metric("Risk", risk)

# ✅ REPLACE WITH:
metrics = {
    "Price": (f"${price:.2f}", None),
    "Value": (f"${value:.2f}", None),
    "Return": (f"{ret:.1%}", None),
    "Risk": (f"{risk:.0f}/100", None),
}
display_metrics_row(metrics, cols=4)
```

### PATTERN 2: Error Handling (Search: "try:")
```python
# ❌ FIND THIS:
try:
    st.write("Running...")
    result = some_function()
    st.success("✅ Done")
    st.dataframe(result)
except Exception as e:
    st.error(f"❌ Error: {str(e)}")

# ✅ REPLACE WITH:
success, result = run_with_error_handling(some_function)
if success:
    st.dataframe(result)
```

### PATTERN 3: Progress Tracking (Search: "st.progress(0)")
```python
# ❌ FIND THIS:
progress = st.progress(0)
status = st.empty()
progress.progress(0, "Starting...")
status.info("Init...")
# ... work ...
progress.progress(50, "Half...")
status.info("Mid...")
# ... work ...
progress.progress(100, "Done!")
status.success("✅ Complete")

# ✅ REPLACE WITH:
progress, status = create_progress_tracker("Starting...")
# ... work ...
update_progress(progress, status, 50, "Half...")
# ... work ...
finalize_progress(progress, status)
```

### PATTERN 4: Column Layouts (Search: "st.columns")
```python
# ❌ 2-column layout
left, right = st.columns(2)

# ✅ Use helper instead:
from ui_utils import create_two_column_layout
left, right = create_two_column_layout((1, 1))

# ❌ 3-column layout
col1, col2, col3 = st.columns(3)

# ✅ Use helper instead:
from ui_utils import create_three_column_layout
col1, col2, col3 = create_three_column_layout()

# ❌ Gauge + metrics layout
gauge_col, metrics_col = st.columns([1.2, 1])

# ✅ Use helper instead:
from ui_utils import create_gauge_metrics_layout
gauge_col, metrics_col = create_gauge_metrics_layout()
```

---

## 📝 UI UTILS FUNCTION REFERENCE

### Session State
```python
initialize_session_state()  # Call once in main()
```

### Display Functions
```python
display_metrics_row(metrics, cols=4)  # Dict[str, Tuple[str, Optional[str]]]
display_performance_summary(returns)  # Dict with 'total', 'annual', 'max_dd', 'sharpe'
display_dataframe_with_settings(df)   # Clean dataframe display
display_model_comparison_header()      # Standard header for model pages
```

### Error Handling
```python
success, result = run_with_error_handling(func, *args, **kwargs)
if success:
    # Use result
else:
    # Already showed error message
```

### Progress Tracking
```python
progress, status = create_progress_tracker("Starting...")
update_progress(progress, status, 50, "Half done...")
finalize_progress(progress, status, "Complete!")
```

### Layouts
```python
left, right = create_two_column_layout((1, 1))
col1, col2, col3 = create_three_column_layout()
gauge, metrics = create_gauge_metrics_layout()  # [1.2, 1] ratio
```

### Navigation
```python
selected_page = render_sidebar_navigation()  # Returns page name from PAGE_REGISTRY
```

---

## 🎯 REFACTORING CHECKLIST

### Before Starting
- [ ] Pull latest app.py
- [ ] Create backup of app.py
- [ ] Create branch: `git checkout -b optimize/ui-refactor`

### Phase 1: Setup (5 min)
- [ ] Copy ui_utils.py to portfolio_optimization/
- [ ] Add import statement to app.py
- [ ] Replace session state checks with `initialize_session_state()`
- [ ] Test: `streamlit run app.py`

### Phase 2: Metrics (15 min)
- [ ] Find all `st.columns()` followed by `st.metric()`
- [ ] Replace with `display_metrics_row()` pattern
- [ ] Test each page after change

### Phase 3: Error Handling (20 min)
- [ ] Find all `try:` blocks with `except Exception`
- [ ] Replace with `run_with_error_handling()` pattern
- [ ] Verify error messages still display

### Phase 4: Progress (10 min)
- [ ] Find all `st.progress()` calls
- [ ] Replace with helper pattern
- [ ] Test progress bars update correctly

### Phase 5: Navigation (5 min)
- [ ] Update sidebar navigation
- [ ] Use PAGE_REGISTRY

### Phase 6: Validation (15 min)
- [ ] [ ] Test Home page
- [ ] [ ] Test Holdings page
- [ ] [ ] Test Pairs Trading page
- [ ] [ ] Test Modeling page
- [ ] [ ] Test Historical Analysis page
- [ ] [ ] Test Backtesting page
- [ ] [ ] Test Chatbot page
- [ ] [ ] Check performance (not slower?)
- [ ] [ ] Verify caching still works

---

## 🐛 COMMON ISSUES & FIXES

| Issue | Cause | Fix |
|-------|-------|-----|
| "No module named 'ui_utils'" | Missing file | Copy ui_utils.py to same dir as app.py |
| Metrics don't align right | Wrong cols param | Use `display_metrics_row(metrics, cols=4)` |
| Progress bar doesn't show | Didn't call `finalize_progress` | Add `finalize_progress(progress, status)` |
| Error message missing | Using wrong function | Use `run_with_error_handling()` |
| Columns too wide/narrow | Wrong ratio | Use `create_gauge_metrics_layout()` for [1.2, 1] |

---

## ✅ BEFORE/AFTER COMPARISON

### Before Refactoring
```
app.py - 2566 lines
  ├─ 22 metric display patterns (50 lines)
  ├─ 19 error handling patterns (60 lines)
  ├─ 35+ column layout patterns (70 lines)
  ├─ 6 session state checks (8 lines)
  └─ Complex progress tracking (40 lines)
```

### After Refactoring
```
app.py - ~2300 lines (-10%)
ui_utils.py - 250 lines (+)

Cleaner patterns:
  ├─ 1 metric function (calls only)
  ├─ 1 error wrapper (calls only)
  ├─ Standard column helpers (calls only)
  ├─ 1 session init (1 call)
  └─ Clean progress helpers (calls only)
```

---

## 📊 TIME ESTIMATES

| Phase | Time | Lines Saved |
|-------|------|------------|
| 1: Setup | 5 min | - |
| 2: Metrics | 15 min | 50 |
| 3: Errors | 20 min | 40 |
| 4: Progress | 10 min | 30 |
| 5: Navigation | 5 min | 10 |
| 6: Validation | 15 min | - |
| **Total** | **70 min** | **130+ lines** |

---

## 🚨 IMPORTANT REMINDERS

1. **Test after each phase** - Don't wait until the end
2. **Keep git history** - Use `git commit` after each phase
3. **Save backups** - Before and after each major change
4. **Read error messages** - They usually tell you what's wrong
5. **Don't skip validation** - Test all 7 pages fully

---

## 🔗 RELATED DOCUMENTATION

- **OPTIMIZATION_SUMMARY.md** - Executive overview
- **OPTIMIZATION_GUIDE.md** - Detailed strategy
- **REFACTORING_EXAMPLES.md** - Code examples
- **IMPLEMENTATION_COOKBOOK.md** - Step-by-step guide
- **ui_utils.py** - Function source code

---

## 💾 GIT WORKFLOW SUGGESTED

```bash
# Start
git checkout -b optimize/ui-refactor

# After Phase 1
git add app.py
git commit -m "refactor: initialize session state centrally"

# After Phase 2
git add app.py
git commit -m "refactor: consolidate metric displays"

# After Phase 3
git add app.py
git commit -m "refactor: standardize error handling"

# After Phase 4
git add app.py
git commit -m "refactor: simplify progress tracking"

# After Phase 5
git add app.py
git commit -m "refactor: clarify navigation routing"

# After Phase 6
git push origin optimize/ui-refactor
# Create PR for review
```

---

## 📞 QUICK HELP

**Q: Can't find a pattern?**
A: Use Find & Replace (Ctrl+H / Cmd+H) to search for "st.columns"

**Q: Metrics look wrong?**
A: Check `cols` parameter: `display_metrics_row(metrics, cols=3)` or `cols=4`

**Q: Error message disappeared?**
A: Ensure you're using `run_with_error_handling()` not `run_with_error_handling`

**Q: Progress bar stuck?**
A: Add `finalize_progress(progress, status)` at end

**Q: Unsure about a change?**
A: Check REFACTORING_EXAMPLES.md for similar pattern

---

**Keep this page open while refactoring!**
**Bookmark: IMPLEMENTATION_COOKBOOK.md for step-by-step guidance**

Last updated: 2025-04-10
