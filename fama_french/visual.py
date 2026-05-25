"""
Fama-French Models Visualization & Comparison

Creates comprehensive visualizations comparing CAPM, FF3, and FF5 models:
- Efficiency frontier
- Factor loadings comparison
- Cumulative returns over time
- Alphas and risk metrics
- Statistical significance heatmaps
- Model fit comparison (R², AIC, BIC)
- Risk-return scatter plots

Author: Emory Economic Investment Forum
Date: March 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"


class FamaFrenchVisualizer:
    """
    Comprehensive visualization for CAPM, FF3, and FF5 model comparison.
    """

    def __init__(self, data_path=None):
        """
        Initialize visualizer

        Args:
            data_path: Path to fama_french.csv
        """
        data_path = Path(data_path) if data_path is not None else OUTPUT_DIR / "fama_french.csv"

        self.df = pd.read_csv(data_path)
        self.df["Date"] = pd.to_datetime(self.df["Date"])

        # Load model results
        self.capm_results = self._load_model_results("capm")
        self.ff3_results = self._load_model_results("ff3")
        self.ff5_results = self._load_model_results("ff5")

    def _load_model_results(self, model_name):
        """Load model results from CSV files."""
        try:
            in_sample = pd.read_csv(OUTPUT_DIR / f"{model_name}_in_sample_monthly.csv")
            oos = pd.read_csv(OUTPUT_DIR / f"{model_name}_out_of_sample.csv")
            return {"in_sample": in_sample, "out_of_sample": oos}
        except Exception as e:
            print(f"Warning: Could not load {model_name} results: {e}")
            return {"in_sample": None, "out_of_sample": None}

    # ========================================================================
    # 1. EFFICIENCY FRONTIER
    # ========================================================================

    def plot_efficiency_frontier(self, output_file="efficiency_frontier.png"):
        """
        Plot risk-return scatter showing efficiency frontier.

        Points colored by model fit (R²)
        """
        if self.ff5_results["in_sample"] is None:
            print("Skipping efficiency frontier - missing data")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Extract monthly returns for risk calculation
        monthly_data = self.df[self.df["Frequency"] == "Monthly"].copy()

        # CAPM results
        capm_df = self.capm_results["in_sample"]
        for _, row in capm_df.iterrows():
            asset = row["Asset"]
            if f"{asset}_Excess" in monthly_data.columns:
                ret = monthly_data[f"{asset}_Excess"].mean() * 12
                risk = monthly_data[f"{asset}_Excess"].std() * np.sqrt(12)
                ax1.scatter(risk, ret, s=100, alpha=0.7, label=asset)
                ax1.annotate(asset, (risk, ret), fontsize=9, ha="center")

        ax1.set_xlabel("Risk (Std Dev, Annualized)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Expected Return (Annualized %)", fontsize=11, fontweight="bold")
        ax1.set_title("CAPM: Risk-Return Frontier", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # FF5 results
        monthly_data_ff5 = self.df[self.df["Frequency"] == "Monthly"].copy()
        ff5_df = self.ff5_results["in_sample"]
        
        scatter = ax2.scatter(
            ff5_df.index, ff5_df["R_Squared"],
            c=ff5_df["Sharpe_Ratio"], s=150, cmap="viridis", alpha=0.7
        )

        for idx, (_, row) in enumerate(ff5_df.iterrows()):
            ax2.annotate(row["Asset"], (idx, row["R_Squared"]), fontsize=9, ha="center")

        ax2.set_xlabel("Asset", fontsize=11, fontweight="bold")
        ax2.set_ylabel("R² (Model Fit)", fontsize=11, fontweight="bold")
        ax2.set_title("FF5: Model Fit vs Sharpe Ratio", fontsize=12, fontweight="bold")
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label("Sharpe Ratio", fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_file}")
        plt.close()

    # ========================================================================
    # 2. FACTOR LOADINGS COMPARISON
    # ========================================================================

    def plot_factor_loadings(self, output_file="factor_loadings.png"):
        """
        Compare factor loadings across all three models.
        """
        if self.ff5_results["in_sample"] is None:
            print("Skipping factor loadings - missing data")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Load data
        capm_df = self.capm_results["in_sample"]
        ff3_df = self.ff3_results["in_sample"]
        ff5_df = self.ff5_results["in_sample"]

        assets = capm_df["Asset"].tolist()

        # 1. Market Beta (all three models should have similar)
        ax = axes[0, 0]
        x = np.arange(len(assets))
        width = 0.25

        ax.bar(x - width, capm_df["Beta"], width, label="CAPM", alpha=0.8)
        ax.bar(x, ff3_df["Beta_Mkt"], width, label="FF3", alpha=0.8)
        ax.bar(x + width, ff5_df["Beta_Mkt"], width, label="FF5", alpha=0.8)

        ax.set_ylabel("Beta (Market Sensitivity)", fontweight="bold")
        ax.set_title("Market Beta (Mkt-RF) Comparison", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(assets, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # 2. SMB and HML loadings (FF3 vs FF5)
        ax = axes[0, 1]
        x = np.arange(len(assets))
        width = 0.25

        ax.bar(x - width, ff3_df["Beta_SMB"], width, label="SMB (FF3)", alpha=0.8)
        ax.bar(x, ff5_df["Beta_SMB"], width, label="SMB (FF5)", alpha=0.8)
        ax.bar(x + width, ff5_df["Beta_HML"], width, label="HML (FF5)", alpha=0.8)

        ax.set_ylabel("Beta", fontweight="bold")
        ax.set_title("Size (SMB) & Value (HML) Factors", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(assets, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # 3. Profitability (RMW) & Investment (CMA) - FF5 only
        ax = axes[1, 0]
        x = np.arange(len(assets))
        width = 0.35

        ax.bar(x - width/2, ff5_df["Beta_RMW"], width, label="RMW (Profitability)", alpha=0.8)
        ax.bar(x + width/2, ff5_df["Beta_CMA"], width, label="CMA (Investment)", alpha=0.8)

        ax.set_ylabel("Beta", fontweight="bold")
        ax.set_title("Profitability (RMW) & Investment (CMA) Factors", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(assets, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

        # 4. Alphas across models
        ax = axes[1, 1]
        x = np.arange(len(assets))
        width = 0.25

        ax.bar(x - width, capm_df["Alpha"] * 12, width, label="CAPM", alpha=0.8)
        ax.bar(x, ff3_df["Alpha"] * 12, width, label="FF3", alpha=0.8)
        ax.bar(x + width, ff5_df["Alpha"] * 12, width, label="FF5", alpha=0.8)

        ax.set_ylabel("Annual Alpha (%)", fontweight="bold")
        ax.set_title("Jensen's Alpha (Annualized) Comparison", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(assets, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0, color="r", linestyle="--", linewidth=1)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_file}")
        plt.close()

    # ========================================================================
    # 3. MODEL FIT COMPARISON
    # ========================================================================

    def plot_model_fit_comparison(self, output_file="model_fit.png"):
        """
        Compare model fit metrics (R², AIC, BIC).
        """
        if self.ff5_results["in_sample"] is None:
            print("Skipping model fit - missing data")
            return

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        capm_df = self.capm_results["in_sample"]
        ff3_df = self.ff3_results["in_sample"]
        ff5_df = self.ff5_results["in_sample"]

        assets = capm_df["Asset"].tolist()
        x = np.arange(len(assets))
        width = 0.25

        # R² Comparison
        ax = axes[0]
        ax.bar(x - width, capm_df["R_Squared"], width, label="CAPM", alpha=0.8)
        ax.bar(x, ff3_df["R_Squared"], width, label="FF3", alpha=0.8)
        ax.bar(x + width, ff5_df["R_Squared"], width, label="FF5", alpha=0.8)

        ax.set_ylabel("R² (Fit Quality)", fontweight="bold", fontsize=11)
        ax.set_title("Model Fit: R²", fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(assets, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis="y")

        # AIC Comparison (lower is better)
        ax = axes[1]
        ax.bar(x - width, capm_df["AIC"], width, label="CAPM", alpha=0.8)
        ax.bar(x, ff3_df["AIC"], width, label="FF3", alpha=0.8)
        ax.bar(x + width, ff5_df["AIC"], width, label="FF5", alpha=0.8)

        ax.set_ylabel("AIC (Lower = Better)", fontweight="bold", fontsize=11)
        ax.set_title("Model Comparison: AIC", fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(assets, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # BIC Comparison
        ax = axes[2]
        ax.bar(x - width, capm_df["BIC"], width, label="CAPM", alpha=0.8)
        ax.bar(x, ff3_df["BIC"], width, label="FF3", alpha=0.8)
        ax.bar(x + width, ff5_df["BIC"], width, label="FF5", alpha=0.8)

        ax.set_ylabel("BIC (Lower = Better)", fontweight="bold", fontsize=11)
        ax.set_title("Model Comparison: BIC", fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(assets, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_file}")
        plt.close()

    # ========================================================================
    # 4. STATISTICAL SIGNIFICANCE HEATMAP
    # ========================================================================

    def plot_significance_heatmap(self, output_file="significance_heatmap.png"):
        """
        Heatmap showing statistical significance of factor loadings (p-values).
        """
        if self.ff5_results["in_sample"] is None:
            print("Skipping significance heatmap - missing data")
            return

        ff5_df = self.ff5_results["in_sample"]

        # Create matrix of p-values
        pval_data = ff5_df[[
            "Beta_Mkt_pval", "Beta_SMB_pval", "Beta_HML_pval",
            "Beta_RMW_pval", "Beta_CMA_pval"
        ]].set_index(ff5_df["Asset"])

        pval_data.columns = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create heatmap
        sns.heatmap(
            pval_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            center=0.05,
            cbar_kws={"label": "P-value"},
            ax=ax,
            vmin=0,
            vmax=0.15,
            linewidths=0.5,
        )

        ax.set_title("FF5: Factor Loading Significance (P-values)\nGreen = Significant (p<0.05)", 
                     fontweight="bold", fontsize=12, pad=20)
        ax.set_xlabel("Factor", fontweight="bold", fontsize=11)
        ax.set_ylabel("Asset", fontweight="bold", fontsize=11)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_file}")
        plt.close()

    # ========================================================================
    # 5. PERFORMANCE METRICS COMPARISON
    # ========================================================================

    def plot_performance_metrics(self, output_file="performance_metrics.png"):
        """
        Compare Sharpe ratio, Jensen's alpha, and information ratio.
        """
        if self.ff5_results["in_sample"] is None:
            print("Skipping performance metrics - missing data")
            return

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        capm_df = self.capm_results["in_sample"]
        ff3_df = self.ff3_results["in_sample"]
        ff5_df = self.ff5_results["in_sample"]

        assets = capm_df["Asset"].tolist()
        x = np.arange(len(assets))
        width = 0.25

        # Sharpe Ratio
        ax = axes[0]
        ax.bar(x - width, capm_df["Sharpe_Ratio"], width, label="CAPM", alpha=0.8)
        ax.bar(x, ff3_df["Sharpe_Ratio"], width, label="FF3", alpha=0.8)
        ax.bar(x + width, ff5_df["Sharpe_Ratio"], width, label="FF5", alpha=0.8)

        ax.set_ylabel("Sharpe Ratio", fontweight="bold", fontsize=11)
        ax.set_title("Risk-Adjusted Return: Sharpe Ratio", fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(assets, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Jensen's Alpha (Annualized)
        ax = axes[1]
        ax.bar(x - width, capm_df["Jensens_Alpha"], width, label="CAPM", alpha=0.8)
        ax.bar(x, ff3_df["Jensens_Alpha"], width, label="FF3", alpha=0.8)
        ax.bar(x + width, ff5_df["Jensens_Alpha"], width, label="FF5", alpha=0.8)

        ax.set_ylabel("Annual Alpha (%)", fontweight="bold", fontsize=11)
        ax.set_title("Abnormal Return: Jensen's Alpha", fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(assets, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0, color="r", linestyle="--", linewidth=1)

        # Information Ratio
        ax = axes[2]
        ax.bar(x - width, capm_df["Information_Ratio"], width, label="CAPM", alpha=0.8)
        ax.bar(x, ff3_df["Information_Ratio"], width, label="FF3", alpha=0.8)
        ax.bar(x + width, ff5_df["Information_Ratio"], width, label="FF5", alpha=0.8)

        ax.set_ylabel("Information Ratio", fontweight="bold", fontsize=11)
        ax.set_title("Active Management: Information Ratio", fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(assets, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_file}")
        plt.close()

    # ========================================================================
    # 6. OUT-OF-SAMPLE COMPARISON
    # ========================================================================

    def plot_out_of_sample_comparison(self, output_file="out_of_sample.png"):
        """
        Compare out-of-sample prediction errors across models.
        """
        if self.capm_results["out_of_sample"] is None:
            print("Skipping OOS comparison - missing data")
            return

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        capm_oos = self.capm_results["out_of_sample"]
        ff3_oos = self.ff3_results["out_of_sample"]
        ff5_oos = self.ff5_results["out_of_sample"]

        assets = capm_oos["Asset"].tolist()
        x = np.arange(len(assets))
        width = 0.25

        # RMSE (Lower is better)
        ax = axes[0]
        ax.bar(x - width, capm_oos["mean_rmse"], width, label="CAPM", alpha=0.8)
        ax.bar(x, ff3_oos["mean_rmse"], width, label="FF3", alpha=0.8)
        ax.bar(x + width, ff5_oos["mean_rmse"], width, label="FF5", alpha=0.8)

        ax.set_ylabel("RMSE", fontweight="bold", fontsize=11)
        ax.set_title("Out-of-Sample Prediction Error: RMSE", fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(assets, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # MAE (Lower is better)
        ax = axes[1]
        ax.bar(x - width, capm_oos["mean_mae"], width, label="CAPM", alpha=0.8)
        ax.bar(x, ff3_oos["mean_mae"], width, label="FF3", alpha=0.8)
        ax.bar(x + width, ff5_oos["mean_mae"], width, label="FF5", alpha=0.8)

        ax.set_ylabel("MAE", fontweight="bold", fontsize=11)
        ax.set_title("Out-of-Sample Error: MAE", fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(assets, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # R² (Higher is better)
        ax = axes[2]
        ax.bar(x - width, capm_oos["mean_r2"], width, label="CAPM", alpha=0.8)
        ax.bar(x, ff3_oos["mean_r2"], width, label="FF3", alpha=0.8)
        ax.bar(x + width, ff5_oos["mean_r2"], width, label="FF5", alpha=0.8)

        ax.set_ylabel("R²", fontweight="bold", fontsize=11)
        ax.set_title("Out-of-Sample Model Fit: R²", fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(assets, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_file}")
        plt.close()

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================

    def create_summary_table(self, output_file="model_summary_table.csv"):
        """
        Create comprehensive summary table comparing all three models.
        """
        if self.ff5_results["in_sample"] is None:
            print("Skipping summary table - missing data")
            return

        capm_df = self.capm_results["in_sample"]
        ff3_df = self.ff3_results["in_sample"]
        ff5_df = self.ff5_results["in_sample"]

        summary = pd.DataFrame({
            "Asset": capm_df["Asset"],
            "CAPM_R2": capm_df["R_Squared"].round(4),
            "FF3_R2": ff3_df["R_Squared"].round(4),
            "FF5_R2": ff5_df["R_Squared"].round(4),
            "CAPM_Alpha": (capm_df["Alpha"] * 12).round(6),
            "FF3_Alpha": (ff3_df["Alpha"] * 12).round(6),
            "FF5_Alpha": (ff5_df["Alpha"] * 12).round(6),
            "CAPM_Sharpe": capm_df["Sharpe_Ratio"].round(4),
            "FF3_Sharpe": ff3_df["Sharpe_Ratio"].round(4),
            "FF5_Sharpe": ff5_df["Sharpe_Ratio"].round(4),
            "FF5_Beta_Mkt": ff5_df["Beta_Mkt"].round(4),
            "FF5_Beta_SMB": ff5_df["Beta_SMB"].round(4),
            "FF5_Beta_HML": ff5_df["Beta_HML"].round(4),
            "FF5_Beta_RMW": ff5_df["Beta_RMW"].round(4),
            "FF5_Beta_CMA": ff5_df["Beta_CMA"].round(4),
        })

        summary.to_csv(output_file, index=False)
        print(f"✓ Saved: {output_file}")

        return summary

    # ========================================================================
    # GENERATE ALL VISUALIZATIONS
    # ========================================================================

    def generate_all(self):
        """Generate all visualizations and summary table."""
        print("\n" + "=" * 80)
        print(" GENERATING FAMA-FRENCH MODEL VISUALIZATIONS")
        print("=" * 80 + "\n")

        self.plot_efficiency_frontier()
        self.plot_factor_loadings()
        self.plot_model_fit_comparison()
        self.plot_significance_heatmap()
        self.plot_performance_metrics()
        self.plot_out_of_sample_comparison()
        summary = self.create_summary_table()

        print("\n" + "=" * 80)
        print(" SUMMARY TABLE")
        print("=" * 80 + "\n")
        print(summary.to_string(index=False))
        print("\n" + "=" * 80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution: Generate all visualizations"""
    visualizer = FamaFrenchVisualizer()
    visualizer.generate_all()


if __name__ == "__main__":
    main()
