"""
Fama-French 5-Factor Model Data Fetching

Fetches and prepares comprehensive data for CAPM, Fama-French 3-factor,
and Fama-French 5-factor models including:

1. Fama-French 5 factors (Mkt-RF, SMB, HML, RMW, CMA) from Ken French's library
2. Individual stock prices from yfinance
3. Risk-free rate from FRED (Treasury bills)
4. Both daily and monthly frequency data
5. Computed returns and performance metrics

Data is merged and exported to fama_french.csv with the following structure:
- Raw prices and returns (daily/monthly)
- Fama-French factors aligned by date
- Risk-free rate
- Equal-weight portfolio returns

Output: fama_french.csv with multi-index columns for easy analysis

Author: Emory Economic Investment Forum
Date: March 2026
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import io
import logging
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FamaFrenchDataFetcher:
    """
    Fetches and prepares Fama-French 5-factor model data from multiple sources.
    """

    def __init__(self, stocks, end_date=None, years_back=5):
        """
        Initialize FamaFrenchDataFetcher

        Args:
            stocks: List of stock tickers to fetch (e.g., ['BAC', 'JPM', 'MS', 'BLK'])
            end_date: End date for data (default: today)
            years_back: How many years of data to fetch (default: 5 years)
        """
        self.stocks = stocks
        self.end_date = end_date or datetime.now()
        self.start_date = self.end_date - timedelta(days=365 * years_back)

        logger.info(f"Data range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Stocks: {', '.join(stocks)}")

        self.daily_returns = None
        self.monthly_returns = None
        self.ff_factors_daily = None
        self.ff_factors_monthly = None
        self.rf_daily = None
        self.rf_monthly = None
        self.data_merged = None

    # ========================================================================
    # 1. FAMA-FRENCH FACTORS
    # ========================================================================

    def fetch_fama_french_5_factors(self):
        """
        Fetch 5-factor data from Ken French's data library via direct download.
        Factors: Mkt-RF, SMB, HML, RMW, CMA + RF (risk-free rate)

        Returns:
            Tuple of (daily_factors_df, monthly_factors_df)
        """
        logger.info("Fetching Fama-French 5 factors from Ken French's library...")

        try:
            # Download monthly 5-factor data directly from Ken French's site (ZIP file)
            url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
            
            import zipfile
            import io as io_module
            from io import StringIO
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Extract CSV from ZIP
            with zipfile.ZipFile(io_module.BytesIO(response.content)) as zf:
                csv_files = [f for f in zf.namelist() if f.endswith('.csv') or f.endswith('.CSV')]
                
                if not csv_files:
                    raise ValueError(f"No CSV file found in zip")
                
                with zf.open(csv_files[0]) as csv_file:
                    # Read the file and parse manually
                    content = csv_file.read().decode('utf-8', errors='ignore')
                    lines = content.split('\n')
                    
                    # Find rows with numeric indices (YYYYMM format - 6 digits)
                    data_lines = []
                    header_line = None
                    
                    for i, line in enumerate(lines):
                        if not line.strip():
                            continue
                        parts = line.split(',')
                        if parts and len(parts[0].strip()) == 6 and parts[0].strip().isdigit():
                            if header_line is None and i > 0:
                                # This might be the header
                                header_line = i
                            data_lines.append(line)
                    
                    # If we found data lines, use them
                    if data_lines:
                        # Find the actual header (line before first data line)
                        for i in range(len(lines)):
                            parts = lines[i].split(',')
                            if parts and len(parts[0].strip()) == 6 and parts[0].strip().isdigit():
                                # Header is the line before this
                                if i > 0:
                                    header_idx = i - 1
                                else:
                                    header_idx = 0
                                break
                        
                        # Read from that header onwards
                        csv_content = '\n'.join(lines[header_idx:])
                        ff_factors_monthly = pd.read_csv(
                            StringIO(csv_content),
                            index_col=0,
                            na_values=-99.99
                        )
                    else:
                        # Fallback: read normally with skiprows
                        ff_factors_monthly = pd.read_csv(
                            StringIO(content),
                            skiprows=4,
                            index_col=0,
                            na_values=-99.99
                        )
            
            # Clean up column names
            ff_factors_monthly.columns = [col.strip() for col in ff_factors_monthly.columns]
            
            # Convert columns to numeric (some might be string)
            for col in ff_factors_monthly.columns:
                ff_factors_monthly[col] = pd.to_numeric(ff_factors_monthly[col], errors='coerce')
            
            # Convert index to datetime (index should be YYYYMM format)
            ff_factors_monthly.index = pd.to_datetime(
                ff_factors_monthly.index.astype(str).str.extract(r'(\d{6})')[0],
                format='%Y%m',
                errors='coerce'
            )
            
            # Remove rows with NaT dates
            ff_factors_monthly = ff_factors_monthly[ff_factors_monthly.index.notna()]
            ff_factors_monthly.index.name = 'Date'
            
            # Divide by 100 to convert from percentages to decimals
            ff_factors_monthly = ff_factors_monthly / 100
            
            logger.info(f"  Downloaded factors shape: {ff_factors_monthly.shape}")
            logger.info(f"  Factors: {ff_factors_monthly.columns.tolist()}")
            logger.info(f"  Fama-French Monthly: {ff_factors_monthly.shape[0]} observations")

            # For daily: resample monthly to daily with forward-fill
            ff_factors_daily = (
                ff_factors_monthly.resample("D").ffill().bfill()
            )

            logger.info(f"  Fama-French Daily (interpolated): {ff_factors_daily.shape[0]} observations")

            # Filter to requested date range
            ff_factors_daily = ff_factors_daily.loc[self.start_date:self.end_date]
            ff_factors_monthly = ff_factors_monthly.loc[self.start_date:self.end_date]

            self.ff_factors_monthly = ff_factors_monthly.copy()
            self.ff_factors_daily = ff_factors_daily.copy()

            return ff_factors_daily, ff_factors_monthly

        except Exception as e:
            logger.error(f"Error fetching Fama-French factors: {e}")
            raise

    # ========================================================================
    # 2. RISK-FREE RATE
    # ========================================================================

    def fetch_risk_free_rate_fred(self):
        """
        Fetch risk-free rate from Ken French's 5-factor data (RF column).
        
        Returns:
            Tuple of (daily_rf_df, monthly_rf_df)
        """
        logger.info("Fetching risk-free rate from Fama-French data...")

        try:
            # Use RF from Ken French's factors (already available)
            if self.ff_factors_daily is None:
                raise ValueError("Must fetch FF factors first")
            
            rf_daily = self.ff_factors_daily[['RF']].copy()
            rf_monthly = self.ff_factors_monthly[['RF']].copy()
            
            logger.info(f"  Daily RF: {rf_daily.shape[0]} observations")
            logger.info(f"  Monthly RF: {rf_monthly.shape[0]} observations")

            self.rf_daily = rf_daily.copy()
            self.rf_monthly = rf_monthly.copy()

            return rf_daily, rf_monthly

        except Exception as e:
            logger.error(f"Error fetching risk-free rate: {e}")
            raise

    # ========================================================================
    # 3. STOCK PRICES & RETURNS
    # ========================================================================

    def fetch_stock_data(self):
        """
        Fetch individual stock prices from yfinance and compute returns.

        Returns:
            Tuple of (daily_returns_df, monthly_returns_df)
        """
        logger.info(f"Fetching stock data from yfinance for {len(self.stocks)} stocks...")

        try:
            # Fetch adjusted close prices
            data = yf.download(
                " ".join(self.stocks),
                start=self.start_date,
                end=self.end_date,
                interval="1d",
                progress=False,
            )
            
            # Handle different yfinance return structures
            # yfinance now returns MultiIndex columns: (Price, Ticker)
            if isinstance(data.columns, pd.MultiIndex):
                # Get Close prices from first level
                if 'Close' in data.columns.get_level_values(0):
                    prices = data['Close'].copy()
                else:
                    # Use first level available
                    prices = data.iloc[:, data.columns.get_level_values(0) == data.columns.get_level_values(0)[0]].copy()
                    prices.columns = prices.columns.get_level_values(1)
            elif 'Adj Close' in data.columns:
                prices = data[['Adj Close']].copy()
                prices.columns = self.stocks
            else:
                # Assume first column is prices
                prices = data.iloc[:, :len(self.stocks)].copy()
                prices.columns = self.stocks

            logger.info(f"  Downloaded prices shape: {prices.shape}")
            logger.info(f"  Coverage: {prices.notna().sum()}")

            # Ensure prices is a DataFrame
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=self.stocks[0])

            # Drop rows with missing data
            prices = prices.dropna(axis=0, how="any")
            logger.info(f"  After removing NaN: {prices.shape[0]} daily observations")

            # Compute daily returns (log returns)
            daily_returns = np.log(prices / prices.shift(1)).dropna()
            daily_returns.index.name = "Date"

            logger.info(f"  Daily returns shape: {daily_returns.shape}")

            # Compute monthly returns (resample to month-end)
            monthly_returns = (1 + daily_returns).resample("ME").prod() - 1
            monthly_returns.index.name = "Date"

            logger.info(f"  Monthly returns shape: {monthly_returns.shape}")

            self.daily_returns = daily_returns.copy()
            self.monthly_returns = monthly_returns.copy()

            return daily_returns, monthly_returns

        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            raise

    # ========================================================================
    # 4. PORTFOLIO RETURNS (EQUAL-WEIGHT)
    # ========================================================================

    def compute_equal_weight_portfolio(self):
        """
        Compute equal-weight portfolio returns from individual stocks.

        Returns:
            Tuple of (daily_portfolio_returns, monthly_portfolio_returns)
        """
        logger.info("Computing equal-weight portfolio returns...")

        daily_portfolio = self.daily_returns.mean(axis=1)
        daily_portfolio.name = "Portfolio_EQ"

        monthly_portfolio = self.monthly_returns.mean(axis=1)
        monthly_portfolio.name = "Portfolio_EQ"

        logger.info(f"  Daily portfolio: {daily_portfolio.notna().sum()} observations")
        logger.info(f"  Monthly portfolio: {monthly_portfolio.notna().sum()} observations")

        return daily_portfolio, monthly_portfolio

    # ========================================================================
    # 5. MERGE & EXPORT
    # ========================================================================

    def merge_all_data(self):
        """
        Merge stock returns, Fama-French factors, and risk-free rate.
        Create both daily and monthly datasets.

        Returns:
            Tuple of (daily_merged_df, monthly_merged_df)
        """
        logger.info("Merging all data sources...")

        # Get equal-weight portfolio
        daily_portfolio, monthly_portfolio = self.compute_equal_weight_portfolio()

        # DAILY DATA
        logger.info("  Preparing daily data...")

        # Combine: stock returns + portfolio + FF factors + RF
        daily_merged = pd.concat(
            [
                self.daily_returns,
                daily_portfolio,
                self.ff_factors_daily[
                    ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
                ],  # Exclude RF from FF (using FRED RF)
                self.rf_daily,
            ],
            axis=1,
            join="inner",
        )

        daily_merged = daily_merged.dropna()
        daily_merged["Frequency"] = "Daily"

        logger.info(f"  Daily merged: {daily_merged.shape}")
        logger.info(f"  Columns: {daily_merged.columns.tolist()}")

        # MONTHLY DATA
        logger.info("  Preparing monthly data...")

        monthly_merged = pd.concat(
            [
                self.monthly_returns,
                monthly_portfolio,
                self.ff_factors_monthly[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]],
                self.rf_monthly,
            ],
            axis=1,
            join="inner",
        )

        monthly_merged = monthly_merged.dropna()
        monthly_merged["Frequency"] = "Monthly"

        logger.info(f"  Monthly merged: {monthly_merged.shape}")
        logger.info(f"  Columns: {monthly_merged.columns.tolist()}")

        # Combine both frequencies
        self.data_merged = pd.concat([daily_merged, monthly_merged], axis=0)
        self.data_merged = self.data_merged.sort_index()

        logger.info(f"Combined daily+monthly: {self.data_merged.shape}")

        return daily_merged, monthly_merged

    # ========================================================================
    # 6. COMPUTE EXCESS RETURNS
    # ========================================================================

    def compute_excess_returns(self, merged_df):
        """
        Compute excess returns (return - risk-free rate) for each asset.
        Fama-French factors are already in excess form.

        Args:
            merged_df: Merged dataframe with returns and RF

        Returns:
            DataFrame with excess returns added as new columns
        """
        logger.info("Computing excess returns...")

        merged_copy = merged_df.copy()

        # Add excess returns for each stock
        for stock in self.stocks:
            merged_copy[f"{stock}_Excess"] = merged_copy[stock] - merged_copy["RF"]

        # Portfolio excess return
        merged_copy["Portfolio_EQ_Excess"] = (
            merged_copy["Portfolio_EQ"] - merged_copy["RF"]
        )

        logger.info(f"  Added excess return columns: {len(self.stocks) + 1}")

        return merged_copy

    # ========================================================================
    # 7. EXPORT TO CSV
    # ========================================================================

    def export_to_csv(self, filepath="fama_french.csv"):
        """
        Export merged data to CSV with both raw returns and excess returns.

        Args:
            filepath: Output CSV file path
        """
        logger.info(f"Exporting data to {filepath}...")

        # Ensure data is merged
        if self.data_merged is None:
            self.merge_all_data()

        # Add excess returns
        export_df = self.compute_excess_returns(self.data_merged)

        # Reset index to include Date as column
        export_df = export_df.reset_index()

        # Export
        export_df.to_csv(filepath, index=False)

        logger.info(f"✓ Exported {export_df.shape[0]} rows × {export_df.shape[1]} columns")
        logger.info(f"✓ File size: {Path(filepath).stat().st_size / 1e6:.2f} MB")
        logger.info(f"✓ Columns: {export_df.columns.tolist()}")

        return export_df

    # ========================================================================
    # 8. SUMMARY STATISTICS
    # ========================================================================

    def print_summary(self):
        """Print summary statistics of fetched data."""
        if self.data_merged is None:
            self.merge_all_data()

        print("\n" + "=" * 80)
        print(" FAMA-FRENCH DATA SUMMARY")
        print("=" * 80 + "\n")

        daily_data = self.data_merged[self.data_merged["Frequency"] == "Daily"]
        monthly_data = self.data_merged[self.data_merged["Frequency"] == "Monthly"]

        print(f"Date Range: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Stocks: {', '.join(self.stocks)}\n")

        print("Data Coverage:")
        print(f"  Daily observations:   {daily_data.shape[0]:>6}")
        print(f"  Monthly observations: {monthly_data.shape[0]:>6}\n")

        print("Daily Returns Statistics:")
        for stock in self.stocks:
            if stock in daily_data.columns:
                mean = daily_data[stock].mean()
                std = daily_data[stock].std()
                print(f"  {stock:>6} - Mean: {mean:>10.6f}, Std: {std:>10.6f}")

        print(f"\n  {'Portfolio_EQ':>6} - Mean: {daily_data['Portfolio_EQ'].mean():>10.6f}, "
              f"Std: {daily_data['Portfolio_EQ'].std():>10.6f}\n")

        print("Fama-French Factors (Daily Mean Returns):")
        for factor in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
            print(f"  {factor:>8} - Mean: {daily_data[factor].mean():>10.6f}")

        print("\n" + "=" * 80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution: fetch all data and export to CSV"""

    # List of stocks to analyze
    stocks = ["BAC", "BLK", "JPM", "MS", "MET", "CG", "CME"]

    # Initialize fetcher
    fetcher = FamaFrenchDataFetcher(stocks=stocks, years_back=5)

    # Fetch all data
    logger.info("Starting data fetch process...\n")

    fetcher.fetch_fama_french_5_factors()
    fetcher.fetch_risk_free_rate_fred()
    fetcher.fetch_stock_data()
    fetcher.merge_all_data()

    # Export to CSV
    csv_path = "fama_french.csv"
    fetcher.export_to_csv(csv_path)

    # Print summary
    fetcher.print_summary()

    logger.info(f"✓ Complete! Data exported to {csv_path}")


if __name__ == "__main__":
    main()
