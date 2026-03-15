"""
Utility functions for options pricing models.

Provides helper functions for comparison, visualization, and analysis across
different pricing methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11


class PricingComparison:
    """
    Compare prices across different models.
    """
    
    def __init__(self, S0: float, K: float, r: float, sigma: float, T: float, q: float = 0.0):
        """
        Initialize comparison object.
        
        Parameters:
        -----------
        S0, K, r, sigma, T, q : float
            Option pricing parameters
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.q = q
        self.results = {}
    
    def add_result(self, model_name: str, call_price: float, put_price: float, 
                  exec_time: float = None):
        """
        Add pricing result from a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        call_price : float
            Call option price
        put_price : float
            Put option price
        exec_time : float
            Execution time in seconds
        """
        self.results[model_name] = {
            'call_price': call_price,
            'put_price': put_price,
            'exec_time': exec_time
        }
    
    def compare(self) -> pd.DataFrame:
        """
        Generate comparison summary.
        
        Returns:
        --------
        pd.DataFrame
            Comparison table
        """
        df_list = []
        
        for model_name, prices in self.results.items():
            df_list.append({
                'Model': model_name,
                'Call Price': prices['call_price'],
                'Put Price': prices['put_price'],
                'Exec Time (s)': prices['exec_time'] if prices['exec_time'] else 'N/A'
            })
        
        df = pd.DataFrame(df_list)
        
        logger.info(f"\n{df.to_string(index=False)}")
        return df
    
    def plot_comparison(self, option_type: str = 'call', save_path: Optional[str] = None):
        """
        Plot price comparison across models.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        models = list(self.results.keys())
        prices = [self.results[m][f'{option_type}_price'] for m in models]
        colors = sns.color_palette("husl", len(models))
        
        bars = ax.bar(models, prices, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, price in zip(bars, prices):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${price:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel(f'{option_type.capitalize()} Price ($)', fontsize=12, fontweight='bold')
        ax.set_title(f'{option_type.capitalize()} Price Comparison (S={self.S0}, K={self.K}, σ={self.sigma})',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        plt.show()


class VolatilityAnalysis:
    """
    Analyze and manipulate volatility structures.
    """
    
    @staticmethod
    def smile_profile(strikes: np.ndarray, S0: float, r: float, T: float,
                     market_prices: np.ndarray, option_type: str = 'call') -> pd.DataFrame:
        """
        Calculate implied volatility smile/skew.
        
        Parameters:
        -----------
        strikes : np.ndarray
            Strike prices
        S0 : float
            Spot price
        r : float
            Risk-free rate
        T : float
            Time to maturity
        market_prices : np.ndarray
            Market prices for options
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        pd.DataFrame
            IV profile with moneyness
        """
        from black_scholes import BlackScholesModel
        
        results = []
        
        for K, market_price in zip(strikes, market_prices):
            try:
                model = BlackScholesModel(S0, K, r, 0.2, T)  # Initial sigma guess
                
                if option_type.lower() == 'call':
                    iv = model.implied_volatility_call(market_price)
                else:
                    iv = model.implied_volatility_put(market_price)
                
                moneyness = np.log(S0 / K)
                
                results.append({
                    'strike': K,
                    'moneyness': moneyness,
                    'implied_vol': iv,
                    'market_price': market_price
                })
            except Exception as e:
                logger.warning(f"Could not calculate IV for K={K}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    @staticmethod
    def plot_volatility_smile(df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot implied volatility smile.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'moneyness' and 'implied_vol' columns
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.plot(df['moneyness'], df['implied_vol'], 'o-', linewidth=2, 
               markersize=8, label='Implied Volatility', color='navy', alpha=0.7)
        
        ax.set_xlabel('Moneyness (ln(S/K))', fontsize=12, fontweight='bold')
        ax.set_ylabel('Implied Volatility', fontsize=12, fontweight='bold')
        ax.set_title('Implied Volatility Smile/Skew', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        plt.show()


class SensitivityAnalysis:
    """
    Perform sensitivity analysis on option prices.
    """
    
    @staticmethod
    def calculate_price_surface(S_range: np.ndarray, sigma_range: np.ndarray,
                               K: float, r: float, T: float, q: float = 0.0,
                               pricing_func = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate option price surface across spot and volatility ranges.
        
        Parameters:
        -----------
        S_range : np.ndarray
            Range of spot prices
        sigma_range : np.ndarray
            Range of volatilities
        K, r, T, q : float
            Option parameters
        pricing_func : callable
            Pricing function that takes (S, K, r, sigma, T, q)
            
        Returns:
        --------
        Tuple
            (S_range, sigma_range, price_surface)
        """
        price_surface = np.zeros((len(sigma_range), len(S_range)))
        
        for i, sigma in enumerate(sigma_range):
            for j, S in enumerate(S_range):
                try:
                    price_surface[i, j] = pricing_func(S, K, r, sigma, T, q)
                except Exception as e:
                    logger.warning(f"Error calculating price at S={S}, sigma={sigma}: {e}")
                    price_surface[i, j] = np.nan
        
        return S_range, sigma_range, price_surface
    
    @staticmethod
    def plot_price_surface(S_range: np.ndarray, sigma_range: np.ndarray, 
                          price_surface: np.ndarray, title: str = "Option Price Surface",
                          save_path: Optional[str] = None):
        """
        Create 3D surface plot of option prices.
        
        Parameters:
        -----------
        S_range : np.ndarray
            Range of spot prices
        sigma_range : np.ndarray
            Range of volatilities
        price_surface : np.ndarray
            2D array of prices
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        S_mesh, sigma_mesh = np.meshgrid(S_range, sigma_range)
        
        surf = ax.plot_surface(S_mesh, sigma_mesh, price_surface, cmap='viridis', 
                              alpha=0.8, edgecolor='none')
        
        ax.set_xlabel('Spot Price ($)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Volatility', fontsize=11, fontweight='bold')
        ax.set_zlabel('Option Price ($)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        fig.colorbar(surf, ax=ax, label='Price ($)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_greeks_heatmap(strikes: np.ndarray, spot_range: np.ndarray,
                           greeks_func = None, greek_name: str = 'delta',
                           K_ref: float = 100, r: float = 0.05, T: float = 0.25,
                           save_path: Optional[str] = None):
        """
        Create heatmap of Greeks across strikes and spot prices.
        
        Parameters:
        -----------
        strikes : np.ndarray
            Strike prices
        spot_range : np.ndarray
            Range of spot prices
        greeks_func : callable
            Function to calculate Greek
        greek_name : str
            Name of Greek for title
        K_ref, r, T : float
            Reference parameters
        save_path : str, optional
            Path to save figure
        """
        greek_matrix = np.zeros((len(spot_range), len(strikes)))
        
        for i, S in enumerate(spot_range):
            for j, K in enumerate(strikes):
                try:
                    greek_matrix[i, j] = greeks_func(S, K, r, 0.2, T)
                except Exception as e:
                    greek_matrix[i, j] = np.nan
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        sns.heatmap(greek_matrix, xticklabels=[f'${K:.0f}' for K in strikes],
                   yticklabels=[f'${S:.0f}' for S in spot_range],
                   cmap='RdYlGn', center=0, ax=ax, cbar_kws={'label': greek_name.upper()})
        
        ax.set_xlabel('Strike Price', fontsize=12, fontweight='bold')
        ax.set_ylabel('Spot Price', fontsize=12, fontweight='bold')
        ax.set_title(f'{greek_name.upper()} Heatmap Across Strikes and Spot Prices',
                    fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        plt.show()


class ModelValidator:
    """
    Validate pricing models against market data.
    """
    
    @staticmethod
    def calculate_misprice(theoretical_price: np.ndarray, market_price: np.ndarray) -> Dict:
        """
        Calculate mispricing metrics.
        
        Parameters:
        -----------
        theoretical_price : np.ndarray
            Model prices
        market_price : np.ndarray
            Market prices
            
        Returns:
        --------
        Dict
            Mispricing statistics
        """
        abs_diff = np.abs(theoretical_price - market_price)
        rel_diff = np.abs((theoretical_price - market_price) / market_price) * 100
        
        return {
            'mean_abs_error': np.mean(abs_diff),
            'std_abs_error': np.std(abs_diff),
            'max_abs_error': np.max(abs_diff),
            'mean_rel_error': np.mean(rel_diff),
            'rmse': np.sqrt(np.mean((theoretical_price - market_price) ** 2))
        }
    
    @staticmethod
    def arbitrage_check(calls: pd.DataFrame, puts: pd.DataFrame, S0: float,
                       r: float, T: float) -> pd.DataFrame:
        """
        Check for put-call parity violations (arbitrage opportunities).
        
        Parameters:
        -----------
        calls : pd.DataFrame
            Call option data with 'strike' and 'price' columns
        puts : pd.DataFrame
            Put option data with 'strike' and 'price' columns
        S0 : float
            Current spot price
        r : float
            Risk-free rate
        T : float
            Time to maturity
            
        Returns:
        --------
        pd.DataFrame
            Arbitrage violations
        """
        results = []
        
        for strike in calls['strike'].unique():
            try:
                call_price = calls[calls['strike'] == strike]['price'].iloc[0]
                put_price = puts[puts['strike'] == strike]['price'].iloc[0]
                
                # Put-call parity: C - P = S - K*e^(-rT)
                lhs = call_price - put_price
                rhs = S0 - strike * np.exp(-r * T)
                
                violation = abs(lhs - rhs)
                
                results.append({
                    'strike': strike,
                    'call_price': call_price,
                    'put_price': put_price,
                    'parity_violation': violation
                })
            except Exception as e:
                logger.warning(f"Error checking parity for K={strike}: {e}")
                continue
        
        df = pd.DataFrame(results)
        
        # Highlight violations > threshold
        threshold = 0.1 * S0
        violations = df[df['parity_violation'] > threshold]
        
        if len(violations) > 0:
            logger.warning(f"Found {len(violations)} potential arbitrage opportunities")
        
        return df


def format_results_table(results: Dict, precision: int = 4) -> pd.DataFrame:
    """
    Format pricing results into a clean table.
    
    Parameters:
    -----------
    results : Dict
        Dictionary with model names as keys and price dicts as values
    precision : int
        Decimal places for rounding
        
    Returns:
    --------
    pd.DataFrame
        Formatted results table
    """
    rows = []
    
    for model, prices in results.items():
        rows.append({
            'Model': model,
            'Call Price': round(prices.get('call_price', np.nan), precision),
            'Put Price': round(prices.get('put_price', np.nan), precision),
            'Time (s)': prices.get('exec_time', 'N/A')
        })
    
    return pd.DataFrame(rows)


def calculate_greeks_table(S0: float, K: float, r: float, sigma: float, T: float,
                          q: float = 0.0, pricing_model = None) -> pd.DataFrame:
    """
    Generate comprehensive Greeks table.
    
    Parameters:
    -----------
    S0, K, r, sigma, T, q : float
        Option parameters
    pricing_model : module
        Pricing module with greeks calculation
        
    Returns:
    --------
    pd.DataFrame
        Greeks summary table
    """
    from black_scholes import BlackScholesModel
    
    model = BlackScholesModel(S0, K, r, sigma, T, q)
    
    call_greeks = model.greeks_call()
    put_greeks = model.greeks_put()
    
    return pd.DataFrame({
        'Greek': ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
        'Call': [call_greeks['delta'], call_greeks['gamma'], call_greeks['vega'],
                call_greeks['theta'], call_greeks['rho']],
        'Put': [put_greeks['delta'], put_greeks['gamma'], put_greeks['vega'],
               put_greeks['theta'], put_greeks['rho']]
    })


if __name__ == "__main__":
    print("=" * 70)
    print("Options Pricing Utilities - Examples")
    print("=" * 70)
    
    # Example: Calculate Greeks table
    print("\n[Example 1] Greeks Summary Table")
    print("-" * 70)
    
    try:
        greeks_table = calculate_greeks_table(S0=100, K=105, r=0.05, sigma=0.2, T=0.25)
        print(greeks_table.to_string(index=False))
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 70)
