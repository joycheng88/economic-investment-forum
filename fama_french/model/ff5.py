"""
Fama-French 5-Factor Model Implementation

FF5 Model:
    Ri - Rf = alpha + b1*(Mkt-RF) + b2*SMB + b3*HML + b4*RMW + b5*CMA + epsilon

Where:
    Mkt-RF = Market risk premium
    SMB = Small Minus Big (size factor)
    HML = High Minus Low (value factor)
    RMW = Robust Minus Weak (profitability factor)
    CMA = Conservative Minus Aggressive (investment factor)

This is the most comprehensive model, adding profitability and investment factors.

This module provides:
- Full 5-factor regression analysis
- In-sample and out-of-sample validation
- Factor decomposition and interpretation
- Comparison with FF3 and CAPM

Author: Emory Economic Investment Forum
Date: March 2026
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
DEFAULT_DATA_PATH = PROJECT_DIR / "output" / "fama_french.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output"


class FamaFrench5Analyzer:
    """
    Fama-French 5-Factor model analyzer.
    """

    def __init__(self, data_path=None):
        """
        Initialize FF5 analyzer with data

        Args:
            data_path: Path to fama_french.csv
        """
        data_path = Path(data_path) if data_path is not None else DEFAULT_DATA_PATH

        logger.info("Loading Fama-French 5-Factor data...")
        self.df = pd.read_csv(data_path)
        self.df["Date"] = pd.to_datetime(self.df["Date"])

        # Separate daily and monthly data
        self.daily_data = self.df[self.df["Frequency"] == "Daily"].copy()
        self.monthly_data = self.df[self.df["Frequency"] == "Monthly"].copy()

        # Identify assets
        self.stocks = [col for col in self.df.columns if col in
                       ["BAC", "BLK", "JPM", "MS", "MET", "CG", "CME"]]
        self.portfolio = [col for col in self.df.columns if "Portfolio_EQ" in col]
        self.all_assets = self.stocks + self.portfolio

        logger.info(f"  Loaded {self.daily_data.shape[0]} daily observations")
        logger.info(f"  Loaded {self.monthly_data.shape[0]} monthly observations")
        logger.info(f"  Assets: {', '.join(self.all_assets)}")

        self.results = {}

    # ========================================================================
    # FF5 REGRESSION
    # ========================================================================

    def fit_ff5(self, excess_returns, factors_df):
        """
        Fit Fama-French 5-factor model.

        Args:
            excess_returns: Asset excess returns
            factors_df: DataFrame with Mkt-RF, SMB, HML, RMW, CMA

        Returns:
            dict with regression results
        """
        # Prepare data
        valid_idx = ~(excess_returns.isna() | factors_df.isna().any(axis=1))
        y = excess_returns[valid_idx].values
        X = factors_df[valid_idx].values

        # Add constant for alpha
        X_with_const = sm.add_constant(X)

        # Fit regression
        model = sm.OLS(y, X_with_const).fit()

        return {
            "alpha": model.params[0],
            "beta_mkt": model.params[1],
            "beta_smb": model.params[2],
            "beta_hml": model.params[3],
            "beta_rmw": model.params[4],
            "beta_cma": model.params[5],
            "alpha_se": model.bse[0],
            "beta_mkt_se": model.bse[1],
            "beta_smb_se": model.bse[2],
            "beta_hml_se": model.bse[3],
            "beta_rmw_se": model.bse[4],
            "beta_cma_se": model.bse[5],
            "alpha_pval": model.pvalues[0],
            "beta_mkt_pval": model.pvalues[1],
            "beta_smb_pval": model.pvalues[2],
            "beta_hml_pval": model.pvalues[3],
            "beta_rmw_pval": model.pvalues[4],
            "beta_cma_pval": model.pvalues[5],
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "residuals": model.resid,
            "fitted": model.fittedvalues,
            "f_statistic": model.fvalue,
            "f_pval": model.f_pvalue,
            "aic": model.aic,
            "bic": model.bic,
            "n_obs": model.nobs,
            "model": model,
        }

    def analyze_in_sample(self, frequency="monthly"):
        """
        In-sample Fama-French 5-factor analysis.

        Args:
            frequency: "daily" or "monthly"

        Returns:
            DataFrame with results for all assets
        """
        logger.info(f"Running in-sample FF5 analysis ({frequency})...")

        data = self.monthly_data if frequency == "monthly" else self.daily_data
        
        # Return empty if no data
        if data is None or len(data) == 0:
            logger.warning(f"  No {frequency} data available, skipping...")
            return pd.DataFrame()
        
        results_list = []

        for asset in self.all_assets:
            if f"{asset}_Excess" not in data.columns:
                continue

            excess_ret = data[f"{asset}_Excess"]
            factors = data[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].copy()
            
            # Skip if insufficient data
            if len(excess_ret) < 6 or excess_ret.isna().all():
                continue

            # Fit FF5
            ff5_res = self.fit_ff5(excess_ret, factors)

            # Compute metrics
            metrics = self._compute_metrics(excess_ret, factors, ff5_res)

            # Result row
            result_row = {
                "Asset": asset,
                "Alpha": ff5_res["alpha"],
                "Alpha_SE": ff5_res["alpha_se"],
                "Alpha_t": ff5_res["alpha"] / ff5_res["alpha_se"],
                "Alpha_pval": ff5_res["alpha_pval"],
                "Beta_Mkt": ff5_res["beta_mkt"],
                "Beta_Mkt_SE": ff5_res["beta_mkt_se"],
                "Beta_Mkt_pval": ff5_res["beta_mkt_pval"],
                "Beta_SMB": ff5_res["beta_smb"],
                "Beta_SMB_SE": ff5_res["beta_smb_se"],
                "Beta_SMB_pval": ff5_res["beta_smb_pval"],
                "Beta_HML": ff5_res["beta_hml"],
                "Beta_HML_SE": ff5_res["beta_hml_se"],
                "Beta_HML_pval": ff5_res["beta_hml_pval"],
                "Beta_RMW": ff5_res["beta_rmw"],
                "Beta_RMW_SE": ff5_res["beta_rmw_se"],
                "Beta_RMW_pval": ff5_res["beta_rmw_pval"],
                "Beta_CMA": ff5_res["beta_cma"],
                "Beta_CMA_SE": ff5_res["beta_cma_se"],
                "Beta_CMA_pval": ff5_res["beta_cma_pval"],
                "R_Squared": ff5_res["r_squared"],
                "Adj_R_Squared": ff5_res["adj_r_squared"],
                "F_Statistic": ff5_res["f_statistic"],
                "F_pval": ff5_res["f_pval"],
                "Sharpe_Ratio": metrics["sharpe"],
                "Jensens_Alpha": metrics["jensens_alpha"],
                "Information_Ratio": metrics["info_ratio"],
                "Observations": ff5_res["n_obs"],
            }

            results_list.append(result_row)
            self.results[f"in_sample_{frequency}_{asset}"] = ff5_res

        results_df = pd.DataFrame(results_list)
        return results_df.sort_values("Sharpe_Ratio", ascending=False)

    # ========================================================================
    # OUT-OF-SAMPLE ANALYSIS
    # ========================================================================

    def analyze_out_of_sample(self, frequency="monthly", n_splits=5):
        """
        K-fold cross-validation for FF5 model.

        Args:
            frequency: "daily" or "monthly"
            n_splits: Number of folds

        Returns:
            DataFrame with cross-validation results
        """
        logger.info(f"Running {n_splits}-fold cross-validation FF5 analysis ({frequency})...")

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
            factors = data[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].values

            # Remove NaN
            valid_idx = ~(np.isnan(excess_ret) | np.isnan(factors).any(axis=1))
            excess_ret = excess_ret[valid_idx]
            factors = factors[valid_idx]
            
            # Skip if insufficient data
            if len(excess_ret) < n_splits:
                logger.warning(f"  Skipping {asset} - insufficient data for {n_splits}-fold CV")
                continue

            fold_results = []

            for train_idx, test_idx in kfold.split(excess_ret):
                # Training data
                X_train = sm.add_constant(factors[train_idx])
                y_train = excess_ret[train_idx]

                # Test data
                X_test = sm.add_constant(factors[test_idx])
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
                    "beta_mkt": model.params[1],
                    "beta_smb": model.params[2],
                    "beta_hml": model.params[3],
                    "beta_rmw": model.params[4],
                    "beta_cma": model.params[5],
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
                "mean_beta_mkt": fold_df["beta_mkt"].mean(),
                "mean_beta_smb": fold_df["beta_smb"].mean(),
                "mean_beta_hml": fold_df["beta_hml"].mean(),
                "mean_beta_rmw": fold_df["beta_rmw"].mean(),
                "mean_beta_cma": fold_df["beta_cma"].mean(),
                "mean_r2": fold_df["r_squared"].mean(),
            }

        cv_df = pd.DataFrame(cv_results).T
        cv_df.index.name = "Asset"
        return cv_df.reset_index()

    # ========================================================================
    # PERFORMANCE METRICS
    # ========================================================================

    def _compute_metrics(self, excess_returns, factors_df, ff5_results):
        """Compute performance metrics."""
        alpha = ff5_results["alpha"]
        residuals = ff5_results["residuals"]

        # Sharpe Ratio
        sharpe = excess_returns.mean() / excess_returns.std()

        # Jensen's Alpha (annualized)
        jensens_alpha = alpha * 12

        # Information Ratio
        tracking_error = residuals.std()
        info_ratio = alpha / tracking_error if tracking_error > 0 else 0

        return {
            "sharpe": sharpe,
            "jensens_alpha": jensens_alpha,
            "info_ratio": info_ratio,
        }

    # ========================================================================
    # MODEL COMPARISON
    # ========================================================================

    def compare_with_ff3_capm(self, frequency="monthly"):
        """
        Compare FF5 fit with FF3 and CAPM models.

        Args:
            frequency: "daily" or "monthly"

        Returns:
            DataFrame comparing models
        """
        logger.info("Comparing FF5 with FF3 and CAPM...")

        data = self.monthly_data if frequency == "monthly" else self.daily_data
        results_list = []

        for asset in self.all_assets:
            if f"{asset}_Excess" not in data.columns:
                continue

            excess_ret = data[f"{asset}_Excess"]
            factors_ff5 = data[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
            factors_ff3 = data[["Mkt-RF", "SMB", "HML"]]
            factors_capm = data[["Mkt-RF"]]

            # Fit all three models
            valid_idx = ~(excess_ret.isna() | factors_ff5.isna().any(axis=1))
            data_clean = excess_ret[valid_idx]

            # CAPM
            X_capm = sm.add_constant(factors_capm[valid_idx])
            model_capm = sm.OLS(data_clean, X_capm).fit()

            # FF3
            X_ff3 = sm.add_constant(factors_ff3[valid_idx])
            model_ff3 = sm.OLS(data_clean, X_ff3).fit()

            # FF5
            X_ff5 = sm.add_constant(factors_ff5[valid_idx])
            model_ff5 = sm.OLS(data_clean, X_ff5).fit()

            # Compare
            result_row = {
                "Asset": asset,
                "CAPM_R2": model_capm.rsquared,
                "FF3_R2": model_ff3.rsquared,
                "FF5_R2": model_ff5.rsquared,
                "FF5_over_CAPM": model_ff5.rsquared - model_capm.rsquared,
                "FF5_over_FF3": model_ff5.rsquared - model_ff3.rsquared,
                "CAPM_AIC": model_capm.aic,
                "FF3_AIC": model_ff3.aic,
                "FF5_AIC": model_ff5.aic,
                "FF5_Better_than_FF3": model_ff5.rsquared > model_ff3.rsquared,
            }

            results_list.append(result_row)

        return pd.DataFrame(results_list)

    # ========================================================================
    # EXPORT RESULTS
    # ========================================================================

    def export_results(self, output_prefix="ff5", output_dir=None):
        """Export FF5 results to CSV."""
        logger.info("Exporting FF5 results...")

        output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # In-sample
        in_sample_daily = self.analyze_in_sample(frequency="daily")
        if not in_sample_daily.empty:
            daily_path = output_dir / f"{output_prefix}_in_sample_daily.csv"
            in_sample_daily.to_csv(daily_path, index=False)
            logger.info(f"  ✓ {daily_path}")

        in_sample_monthly = self.analyze_in_sample(frequency="monthly")
        if not in_sample_monthly.empty:
            monthly_path = output_dir / f"{output_prefix}_in_sample_monthly.csv"
            in_sample_monthly.to_csv(monthly_path, index=False)
            logger.info(f"  ✓ {monthly_path}")

        # Out-of-sample (use daily data)
        oos = self.analyze_out_of_sample(frequency="daily")
        if not oos.empty:
            oos_path = output_dir / f"{output_prefix}_out_of_sample.csv"
            oos.to_csv(oos_path, index=False)
            logger.info(f"  ✓ {oos_path}")

        # Model comparison (use daily data)
        comparison = self.compare_with_ff3_capm(frequency="daily")
        if not comparison.empty:
            comparison_path = output_dir / f"{output_prefix}_model_comparison.csv"
            comparison.to_csv(comparison_path, index=False)
            logger.info(f"  ✓ {comparison_path}")

        return in_sample_daily, in_sample_monthly, oos, comparison

    def print_summary(self):
        """Print summary of FF5 analysis."""
        in_sample = self.analyze_in_sample(frequency="daily")

        print("\n" + "=" * 140)
        print(" FAMA-FRENCH 5-FACTOR MODEL RESULTS (In-Sample, Daily Data)")
        print("=" * 140 + "\n")

        if not in_sample.empty:
            print(in_sample[[
                "Asset", "Alpha", "Alpha_pval", "Beta_Mkt", "Beta_Mkt_pval",
                "Beta_SMB", "Beta_SMB_pval", "Beta_HML", "Beta_HML_pval",
                "Beta_RMW", "Beta_RMW_pval", "Beta_CMA", "Beta_CMA_pval",
                "R_Squared", "Sharpe_Ratio"
            ]].to_string(index=False))
        else:
            print("  No daily data available")

        print("\n" + "=" * 140 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution: FF5 analysis"""
    ff5 = FamaFrench5Analyzer()
    ff5.export_results("ff5")
    ff5.print_summary()


if __name__ == "__main__":
    main()
