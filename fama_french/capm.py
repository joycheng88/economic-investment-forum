"""
CAPM (Capital Asset Pricing Model) Implementation

Standard CAPM model:
    Ri - Rf = alpha + beta * (Rm - Rf) + epsilon

Where:
    Ri = Return on asset i
    Rf = Risk-free rate
    Rm = Market return (Mkt-RF factor)
    beta = Market sensitivity
    alpha = Jensen's alpha (abnormal return)

This module provides:
- In-sample and out-of-sample regression analysis
- Performance metrics (Sharpe ratio, Jensen's alpha, information ratio)
- Statistical significance tests
- Results export

Author: Emory Economic Investment Forum
Date: March 2026
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CAPMAnalyzer:
    """
    CAPM (Capital Asset Pricing Model) analyzer for individual stocks and portfolios.
    """

    def __init__(self, data_path="fama_french.csv"):
        """
        Initialize CAPM analyzer with data

        Args:
            data_path: Path to fama_french.csv
        """
        logger.info("Loading CAPM data...")
        self.df = pd.read_csv(data_path)
        self.df["Date"] = pd.to_datetime(self.df["Date"])

        # Separate daily and monthly data
        self.daily_data = self.df[self.df["Frequency"] == "Daily"].copy()
        self.monthly_data = self.df[self.df["Frequency"] == "Monthly"].copy()

        # Identify assets (stocks + portfolio)
        self.stocks = [col for col in self.df.columns if col in
                       ["BAC", "BLK", "JPM", "MS", "MET", "CG", "CME"]]
        self.portfolio = [col for col in self.df.columns if "Portfolio_EQ" in col]
        self.all_assets = self.stocks + self.portfolio

        logger.info(f"  Loaded {self.daily_data.shape[0]} daily observations")
        logger.info(f"  Loaded {self.monthly_data.shape[0]} monthly observations")
        logger.info(f"  Assets: {', '.join(self.all_assets)}")

        self.results = {}

    # ========================================================================
    # CAPM REGRESSION
    # ========================================================================

    def fit_capm(self, excess_returns, market_premium):
        """
        Fit CAPM model: y = alpha + beta * market_premium

        Args:
            excess_returns: Asset excess returns
            market_premium: Market risk premium (Mkt-RF)

        Returns:
            dict with regression results
        """
        # Remove NaN values
        valid_idx = ~(excess_returns.isna() | market_premium.isna())
        y = excess_returns[valid_idx].values
        X = market_premium[valid_idx].values

        # Add constant for alpha
        X_with_const = sm.add_constant(X)

        # Fit regression
        model = sm.OLS(y, X_with_const).fit()

        return {
            "alpha": model.params[0],
            "beta": model.params[1],
            "alpha_se": model.bse[0],
            "beta_se": model.bse[1],
            "alpha_pval": model.pvalues[0],
            "beta_pval": model.pvalues[1],
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "residuals": model.resid,
            "fitted": model.fittedvalues,
            "aic": model.aic,
            "bic": model.bic,
            "n_obs": model.nobs,
        }

    def analyze_in_sample(self, frequency="monthly"):
        """
        Analyze CAPM model on entire dataset (in-sample).

        Args:
            frequency: "daily" or "monthly"

        Returns:
            DataFrame with results for all assets
        """
        logger.info(f"Running in-sample CAPM analysis ({frequency})...")

        data = self.monthly_data if frequency == "monthly" else self.daily_data
        
        # Return empty if no data
        if data is None or len(data) == 0:
            logger.warning(f"  No {frequency} data available, skipping...")
            return pd.DataFrame()
        
        results_list = []

        for asset in self.all_assets:
            # Skip if asset has no data
            if f"{asset}_Excess" not in data.columns:
                continue

            excess_ret = data[f"{asset}_Excess"]
            market_prem = data["Mkt-RF"]
            
            # Skip if insufficient data
            if len(excess_ret) < 3 or excess_ret.isna().all():
                continue

            # Fit CAPM
            capm_res = self.fit_capm(excess_ret, market_prem)

            # Compute additional metrics
            metrics = self._compute_metrics(excess_ret, market_prem, capm_res)

            # Combine results
            result_row = {
                "Asset": asset,
                "Alpha": capm_res["alpha"],
                "Alpha_SE": capm_res["alpha_se"],
                "Alpha_t": capm_res["alpha"] / capm_res["alpha_se"],
                "Alpha_pval": capm_res["alpha_pval"],
                "Beta": capm_res["beta"],
                "Beta_SE": capm_res["beta_se"],
                "Beta_t": capm_res["beta"] / capm_res["beta_se"],
                "Beta_pval": capm_res["beta_pval"],
                "R_Squared": capm_res["r_squared"],
                "Adj_R_Squared": capm_res["adj_r_squared"],
                "Sharpe_Ratio": metrics["sharpe"],
                "Jensens_Alpha": metrics["jensens_alpha"],
                "Information_Ratio": metrics["info_ratio"],
                "Treynor_Ratio": metrics["treynor"],
                "Tracking_Error": metrics["tracking_error"],
                "Observations": capm_res["n_obs"],
            }

            results_list.append(result_row)
            self.results[f"in_sample_{frequency}_{asset}"] = capm_res

        results_df = pd.DataFrame(results_list)
        return results_df.sort_values("Sharpe_Ratio", ascending=False)

    # ========================================================================
    # OUT-OF-SAMPLE ANALYSIS
    # ========================================================================

    def analyze_out_of_sample(self, frequency="monthly", n_splits=5):
        """
        Cross-validation analysis using K-fold approach.

        Args:
            frequency: "daily" or "monthly"
            n_splits: Number of k-folds (default: 5)

        Returns:
            DataFrame with cross-validation results
        """
        logger.info(f"Running {n_splits}-fold cross-validation CAPM analysis ({frequency})...")

        data = self.monthly_data if frequency == "monthly" else self.daily_data
        
        # Return empty if no data
        if data is None or len(data) == 0:
            logger.warning(f"  No {frequency} data available, skipping...")
            return pd.DataFrame()
        
        kfold = KFold(n_splits=n_splits, shuffle=False)

        cv_results = {}

        for asset in self.all_assets:
            if f"{asset}_Excess" not in data.columns:
                continue

            excess_ret = data[f"{asset}_Excess"].values
            market_prem = data["Mkt-RF"].values

            # Remove NaN
            valid_idx = ~(np.isnan(excess_ret) | np.isnan(market_prem))
            excess_ret = excess_ret[valid_idx]
            market_prem = market_prem[valid_idx]
            
            # Skip if insufficient data
            if len(excess_ret) < n_splits:
                logger.warning(f"  Skipping {asset} - insufficient data for {n_splits}-fold CV")
                continue

            fold_results = []

            for train_idx, test_idx in kfold.split(excess_ret):
                # Training data
                X_train = sm.add_constant(market_prem[train_idx])
                y_train = excess_ret[train_idx]

                # Test data
                X_test = sm.add_constant(market_prem[test_idx])
                y_test = excess_ret[test_idx]

                # Fit on training
                model = sm.OLS(y_train, X_train).fit()

                # Predict on test
                y_pred = model.predict(X_test)

                # Error metrics
                mse = np.mean((y_test - y_pred) ** 2)
                mae = np.mean(np.abs(y_test - y_pred))
                rmse = np.sqrt(mse)

                fold_results.append({
                    "mse": mse,
                    "mae": mae,
                    "rmse": rmse,
                    "alpha": model.params[0],
                    "beta": model.params[1],
                    "r_squared": model.rsquared,
                })

            # Average across folds
            fold_df = pd.DataFrame(fold_results)
            cv_results[asset] = {
                "mean_mse": fold_df["mse"].mean(),
                "std_mse": fold_df["mse"].std(),
                "mean_mae": fold_df["mae"].mean(),
                "std_mae": fold_df["mae"].std(),
                "mean_rmse": fold_df["rmse"].mean(),
                "std_rmse": fold_df["rmse"].std(),
                "mean_alpha": fold_df["alpha"].mean(),
                "mean_beta": fold_df["beta"].mean(),
                "mean_r2": fold_df["r_squared"].mean(),
            }

        cv_df = pd.DataFrame(cv_results).T
        cv_df.index.name = "Asset"
        return cv_df.reset_index()

    # ========================================================================
    # PERFORMANCE METRICS
    # ========================================================================

    def _compute_metrics(self, excess_returns, market_premium, capm_results):
        """
        Compute performance metrics for CAPM model.

        Args:
            excess_returns: Excess returns (y)
            market_premium: Market risk premium (X)
            capm_results: CAPM regression results

        Returns:
            dict with performance metrics
        """
        alpha = capm_results["alpha"]
        beta = capm_results["beta"]
        residuals = capm_results["residuals"]

        # Sharpe Ratio = excess return / std dev
        excess_ret_mean = excess_returns.mean()
        excess_ret_std = excess_returns.std()
        sharpe = excess_ret_mean / excess_ret_std if excess_ret_std > 0 else 0

        # Jensen's Alpha (annualized if monthly data)
        jensens_alpha = alpha * 12  # Annualize monthly alpha

        # Information Ratio
        tracking_error = residuals.std()
        info_ratio = alpha / tracking_error if tracking_error > 0 else 0

        # Treynor Ratio
        market_premium_mean = market_premium.mean()
        treynor = excess_ret_mean / beta if beta != 0 else 0

        return {
            "sharpe": sharpe,
            "jensens_alpha": jensens_alpha,
            "info_ratio": info_ratio,
            "treynor": treynor,
            "tracking_error": tracking_error,
        }

    # ========================================================================
    # EXPORT RESULTS
    # ========================================================================

    def export_results(self, output_prefix="capm"):
        """Export CAPM results to CSV files."""
        logger.info(f"Exporting CAPM results...")

        # In-sample daily
        in_sample_daily = self.analyze_in_sample(frequency="daily")
        if not in_sample_daily.empty:
            in_sample_daily.to_csv(f"{output_prefix}_in_sample_daily.csv", index=False)
            logger.info(f"  ✓ {output_prefix}_in_sample_daily.csv")
        else:
            logger.warning(f"  ⚠ Skipping daily results - no data")

        # In-sample monthly
        in_sample_monthly = self.analyze_in_sample(frequency="monthly")
        if not in_sample_monthly.empty:
            in_sample_monthly.to_csv(f"{output_prefix}_in_sample_monthly.csv", index=False)
            logger.info(f"  ✓ {output_prefix}_in_sample_monthly.csv")
        else:
            logger.warning(f"  ⚠ Skipping monthly results - no data")

        # Out-of-sample (use daily data for cross-validation)
        oos = self.analyze_out_of_sample(frequency="daily")
        if not oos.empty:
            oos.to_csv(f"{output_prefix}_out_of_sample.csv", index=False)
            logger.info(f"  ✓ {output_prefix}_out_of_sample.csv")
        else:
            logger.warning(f"  ⚠ Skipping out-of-sample results - no data")

        return in_sample_daily, in_sample_monthly, oos

    def print_summary(self):
        """Print summary of CAPM analysis."""
        in_sample_daily = self.analyze_in_sample(frequency="daily")

        print("\n" + "=" * 100)
        print(" CAPM MODEL RESULTS (In-Sample, Daily Data)")
        print("=" * 100 + "\n")

        if not in_sample_daily.empty:
            print(in_sample_daily[[
                "Asset", "Alpha", "Beta", "Alpha_pval", "Beta_pval",
                "R_Squared", "Sharpe_Ratio", "Jensens_Alpha"
            ]].to_string(index=False))
        else:
            print("  No daily data available")

        print("\n" + "=" * 100 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution: CAPM analysis"""
    capm = CAPMAnalyzer("fama_french.csv")
    capm.export_results("capm")
    capm.print_summary()


if __name__ == "__main__":
    main()
