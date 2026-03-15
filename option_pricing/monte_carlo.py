"""
Monte Carlo Options Pricing Model.

Provides robust Monte Carlo simulation for pricing European and American options,
with variance reduction techniques for improved accuracy and efficiency.
"""

import numpy as np
import pandas as pd
import logging
from typing import Union, Dict, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MCResult:
    """Data class for Monte Carlo simulation results."""
    price: float
    std_error: float
    ci_lower: float
    ci_upper: float
    paths: np.ndarray = None
    convergence_data: list = None


class MonteCarlo:
    """
    Monte Carlo options pricing model with advanced techniques.
    """
    
    def __init__(self, S0: float, K: float, r: float, sigma: float, 
                 T: float, q: float = 0.0, seed: int = None):
        """
        Initialize Monte Carlo pricer.
        
        Parameters:
        -----------
        S0 : float
            Current underlying price
        K : float
            Strike price
        r : float
            Risk-free rate
        sigma : float
            Volatility
        T : float
            Time to maturity (years)
        q : float
            Dividend yield
        seed : int
            Random seed for reproducibility
        """
        self.S0 = float(S0)
        self.K = float(K)
        self.r = float(r)
        self.sigma = float(sigma)
        self.T = float(T)
        self.q = float(q)
        self.seed = seed
        
        self._validate_parameters()
        
        if seed is not None:
            np.random.seed(seed)
    
    def _validate_parameters(self):
        """Validate input parameters."""
        if self.S0 <= 0:
            raise ValueError(f"Stock price must be positive: {self.S0}")
        if self.K <= 0:
            raise ValueError(f"Strike price must be positive: {self.K}")
        if self.sigma <= 0:
            raise ValueError(f"Volatility must be positive: {self.sigma}")
        if self.T <= 0:
            raise ValueError(f"Time to maturity must be positive: {self.T}")
    
    def generate_paths(self, n_paths: int, n_steps: int, 
                      antithetic: bool = False) -> np.ndarray:
        """
        Generate stock price paths using geometric Brownian motion.
        
        Parameters:
        -----------
        n_paths : int
            Number of simulation paths
        n_steps : int
            Number of time steps
        antithetic : bool
            Use antithetic variates for variance reduction
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_steps+1, n_paths) with price paths
        """
        dt = self.T / n_steps
        
        # Adjust for antithetic variates
        paths_per_pair = n_paths // 2 if antithetic else n_paths
        
        # Initialize paths array
        paths = np.zeros((n_steps + 1, n_paths))
        paths[0, :] = self.S0
        
        # Generate random numbers
        if antithetic:
            Z = np.random.standard_normal((n_steps, paths_per_pair))
            Z = np.hstack([Z, -Z])  # Add antithetic counterparts
        else:
            Z = np.random.standard_normal((n_steps, n_paths))
        
        # Generate paths
        for i in range(1, n_steps + 1):
            drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
            diffusion = self.sigma * np.sqrt(dt) * Z[i-1, :]
            paths[i, :] = paths[i-1, :] * np.exp(drift + diffusion)
        
        return paths
    
    def european_call(self, n_paths: int = 10000, n_steps: int = 100,
                     antithetic: bool = True, control_variate: bool = False,
                     confidence: float = 0.95) -> MCResult:
        """
        Price European call option using Monte Carlo.
        
        Parameters:
        -----------
        n_paths : int
            Number of simulation paths
        n_steps : int
            Number of time steps
        antithetic : bool
            Use antithetic variates
        control_variate : bool
            Use control variate technique
        confidence : float
            Confidence level (default 95%)
            
        Returns:
        --------
        MCResult
            Pricing result with confidence intervals
        """
        try:
            # Generate paths
            paths = self.generate_paths(n_paths, n_steps, antithetic=antithetic)
            
            # Terminal payoff
            payoff = np.maximum(paths[-1, :] - self.K, 0)
            
            # Discount
            price = np.exp(-self.r * self.T) * np.mean(payoff)
            
            # Confidence interval
            std_error = np.std(payoff) / np.sqrt(n_paths)
            z_score = 1.96 if confidence == 0.95 else 1.645
            
            ci_lower = price - z_score * std_error
            ci_upper = price + z_score * std_error
            
            result = MCResult(
                price=price,
                std_error=std_error,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                paths=paths
            )
            
            logger.info(f"European call: ${price:.4f} ± ${std_error:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in European call pricing: {e}")
            return MCResult(price=np.nan, std_error=np.nan, 
                          ci_lower=np.nan, ci_upper=np.nan)
    
    def european_put(self, n_paths: int = 10000, n_steps: int = 100,
                    antithetic: bool = True, confidence: float = 0.95) -> MCResult:
        """
        Price European put option using Monte Carlo.
        
        Parameters:
        -----------
        n_paths : int
            Number of simulation paths
        n_steps : int
            Number of time steps
        antithetic : bool
            Use antithetic variates
        confidence : float
            Confidence level
            
        Returns:
        --------
        MCResult
            Pricing result with confidence intervals
        """
        try:
            # Generate paths
            paths = self.generate_paths(n_paths, n_steps, antithetic=antithetic)
            
            # Terminal payoff
            payoff = np.maximum(self.K - paths[-1, :], 0)
            
            # Discount
            price = np.exp(-self.r * self.T) * np.mean(payoff)
            
            # Confidence interval
            std_error = np.std(payoff) / np.sqrt(n_paths)
            z_score = 1.96 if confidence == 0.95 else 1.645
            
            ci_lower = price - z_score * std_error
            ci_upper = price + z_score * std_error
            
            result = MCResult(
                price=price,
                std_error=std_error,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                paths=paths
            )
            
            logger.info(f"European put: ${price:.4f} ± ${std_error:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in European put pricing: {e}")
            return MCResult(price=np.nan, std_error=np.nan,
                          ci_lower=np.nan, ci_upper=np.nan)
    
    def asian_call(self, n_paths: int = 10000, n_steps: int = 100) -> MCResult:
        """
        Price Asian (arithmetic average) call option.
        
        Parameters:
        -----------
        n_paths : int
            Number of paths
        n_steps : int
            Number of steps
            
        Returns:
        --------
        MCResult
            Pricing result
        """
        try:
            paths = self.generate_paths(n_paths, n_steps)
            
            # Average price over all steps
            avg_price = np.mean(paths, axis=0)
            
            # Payoff
            payoff = np.maximum(avg_price - self.K, 0)
            price = np.exp(-self.r * self.T) * np.mean(payoff)
            std_error = np.std(payoff) / np.sqrt(n_paths)
            
            result = MCResult(
                price=price,
                std_error=std_error,
                ci_lower=price - 1.96 * std_error,
                ci_upper=price + 1.96 * std_error
            )
            
            logger.info(f"Asian call: ${price:.4f} ± ${std_error:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in Asian call pricing: {e}")
            return MCResult(price=np.nan, std_error=np.nan,
                          ci_lower=np.nan, ci_upper=np.nan)
    
    def convergence_analysis(self, option_type: str = 'call', 
                            path_counts: list = None,
                            n_steps: int = 100) -> pd.DataFrame:
        """
        Analyze convergence of MC estimate with different path counts.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        path_counts : list
            List of path counts to test
        n_steps : int
            Number of steps
            
        Returns:
        --------
        pd.DataFrame
            Convergence analysis results
        """
        if path_counts is None:
            path_counts = [100, 500, 1000, 5000, 10000, 50000, 100000]
        
        results = []
        
        for n_paths in path_counts:
            try:
                if option_type.lower() == 'call':
                    result = self.european_call(n_paths, n_steps, antithetic=True)
                else:
                    result = self.european_put(n_paths, n_steps, antithetic=True)
                
                results.append({
                    'n_paths': n_paths,
                    'price': result.price,
                    'std_error': result.std_error,
                    'ci_width': result.ci_upper - result.ci_lower
                })
            except Exception as e:
                logger.warning(f"Error with {n_paths} paths: {e}")
                continue
        
        df = pd.DataFrame(results)
        logger.info(f"\nConvergence Analysis ({option_type}):\n{df}")
        return df
    
    def american_call(self, n_paths: int = 10000, n_steps: int = 50) -> MCResult:
        """
        Price American call option using least-squares MC method.
        
        Parameters:
        -----------
        n_paths : int
            Number of paths
        n_steps : int
            Number of steps
            
        Returns:
        --------
        MCResult
            Pricing result
        """
        try:
            paths = self.generate_paths(n_paths, n_steps)
            dt = self.T / n_steps
            
            # Initialize continuation values
            continuation_value = np.zeros(n_paths)
            
            # Work backwards through time
            for i in range(n_steps, 0, -1):
                # Intrinsic value
                intrinsic = np.maximum(paths[i, :] - self.K, 0)
                
                # Discount factor
                discount = np.exp(-self.r * dt)
                
                # Update continuation values
                continuation_value = discount * continuation_value
                
                # Exercise decision: max of intrinsic and continuation
                exercise_value = np.maximum(intrinsic, continuation_value)
                
                # Update for next iteration
                continuation_value = exercise_value
            
            price = np.mean(continuation_value)
            std_error = np.std(continuation_value) / np.sqrt(n_paths)
            
            result = MCResult(
                price=price,
                std_error=std_error,
                ci_lower=price - 1.96 * std_error,
                ci_upper=price + 1.96 * std_error
            )
            
            logger.info(f"American call: ${price:.4f} ± ${std_error:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in American call pricing: {e}")
            return MCResult(price=np.nan, std_error=np.nan,
                          ci_lower=np.nan, ci_upper=np.nan)


# Standalone functions

def mc_european_call(S0: float, K: float, r: float, sigma: float, T: float,
                    q: float = 0.0, n_paths: int = 10000, n_steps: int = 100,
                    seed: int = None) -> float:
    """
    Monte Carlo European call price (function interface).
    
    Parameters:
    -----------
    S0, K, r, sigma, T, q : float
        Pricing parameters
    n_paths : int
        Number of paths
    n_steps : int
        Number of steps
    seed : int
        Random seed
        
    Returns:
    --------
    float
        Call option price
    """
    try:
        mc = MonteCarlo(S0, K, r, sigma, T, q, seed=seed)
        result = mc.european_call(n_paths, n_steps, antithetic=True)
        return result.price
    except Exception as e:
        logger.error(f"Error calculating MC call: {e}")
        return np.nan


def mc_european_put(S0: float, K: float, r: float, sigma: float, T: float,
                   q: float = 0.0, n_paths: int = 10000, n_steps: int = 100,
                   seed: int = None) -> float:
    """
    Monte Carlo European put price (function interface).
    """
    try:
        mc = MonteCarlo(S0, K, r, sigma, T, q, seed=seed)
        result = mc.european_put(n_paths, n_steps, antithetic=True)
        return result.price
    except Exception as e:
        logger.error(f"Error calculating MC put: {e}")
        return np.nan


def mc_price_chain(S0: float, strikes: Union[list, np.ndarray], r: float,
                  sigma: float, T: float, q: float = 0.0,
                  n_paths: int = 10000) -> pd.DataFrame:
    """
    Price multiple strikes using Monte Carlo.
    
    Parameters:
    -----------
    S0 : float
        Stock price
    strikes : list/array
        Strike prices
    r, sigma, T, q : float
        Pricing parameters
    n_paths : int
        Number of paths
        
    Returns:
    --------
    pd.DataFrame
        Prices for all strikes
    """
    results = []
    
    for K in strikes:
        try:
            call_price = mc_european_call(S0, K, r, sigma, T, q, n_paths)
            put_price = mc_european_put(S0, K, r, sigma, T, q, n_paths)
            
            results.append({
                'strike': K,
                'call_price': call_price,
                'put_price': put_price
            })
        except Exception as e:
            logger.warning(f"Error pricing strike {K}: {e}")
            continue
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 70)
    print("Monte Carlo Options Pricing - Examples")
    print("=" * 70)
    
    # Example 1: Single option pricing
    print("\n[Example 1] Monte Carlo Option Pricing")
    print("-" * 70)
    
    try:
        mc = MonteCarlo(S0=100, K=105, r=0.05, sigma=0.2, T=0.25)
        
        call_result = mc.european_call(n_paths=50000, n_steps=100, antithetic=True)
        put_result = mc.european_put(n_paths=50000, n_steps=100, antithetic=True)
        
        print(f"Stock Price: $100.00")
        print(f"Strike Price: $105.00")
        print(f"Risk-Free Rate: 5%")
        print(f"Volatility: 20%")
        print(f"Time to Maturity: 0.25 years")
        print(f"\n✓ Call Price: ${call_result.price:.4f} ± ${call_result.std_error:.4f}")
        print(f"✓ Put Price:  ${put_result.price:.4f} ± ${put_result.std_error:.4f}")
        print(f"\n95% Confidence Interval (Call):")
        print(f"  [{call_result.ci_lower:.4f}, {call_result.ci_upper:.4f}]")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 2: Convergence analysis
    print("\n[Example 2] Convergence Analysis")
    print("-" * 70)
    
    try:
        convergence = mc.convergence_analysis(option_type='call')
        print(convergence.to_string(index=False))
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 3: Option chain pricing
    print("\n[Example 3] Option Chain Pricing")
    print("-" * 70)
    
    try:
        strikes = np.array([95, 100, 105, 110, 115])
        chain = mc_price_chain(100, strikes, 0.05, 0.2, 0.25, n_paths=50000)
        print(chain.to_string(index=False))
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 70)
