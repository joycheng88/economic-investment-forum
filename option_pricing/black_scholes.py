"""
Black-Scholes Options Pricing Model.

Provides robust implementations of the Black-Scholes formula for European
options pricing, including Greeks calculation and implied volatility estimation.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd
import logging
from typing import Union, Tuple, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlackScholesModel:
    """
    Black-Scholes options pricing model with comprehensive functionality.
    """
    
    def __init__(self, S0: float, K: float, r: float, sigma: float, 
                 T: float, q: float = 0.0):
        """
        Initialize Black-Scholes model parameters.
        
        Parameters:
        -----------
        S0 : float
            Current underlying asset price
        K : float
            Strike price
        r : float
            Risk-free rate (annualized)
        sigma : float
            Volatility (annualized)
        T : float
            Time to maturity (in years)
        q : float
            Dividend yield (annualized)
        """
        self.S0 = float(S0)
        self.K = float(K)
        self.r = float(r)
        self.sigma = float(sigma)
        self.T = float(T)
        self.q = float(q)
        
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate input parameters."""
        if self.S0 <= 0:
            raise ValueError(f"Stock price must be positive, got {self.S0}")
        if self.K <= 0:
            raise ValueError(f"Strike price must be positive, got {self.K}")
        if self.r < -0.5 or self.r > 1.0:
            raise ValueError(f"Risk-free rate out of reasonable range: {self.r}")
        if self.sigma <= 0 or self.sigma > 3.0:
            raise ValueError(f"Volatility must be positive and reasonable, got {self.sigma}")
        if self.T <= 0:
            raise ValueError(f"Time to maturity must be positive, got {self.T}")
        if self.q < 0 or self.q > 0.5:
            raise ValueError(f"Dividend yield out of reasonable range: {self.q}")
    
    def _d1(self) -> float:
        """Calculate d1 from Black-Scholes formula."""
        return (np.log(self.S0 / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
    
    def _d2(self) -> float:
        """Calculate d2 from Black-Scholes formula."""
        return self._d1() - self.sigma * np.sqrt(self.T)
    
    def call_price(self) -> float:
        """
        Calculate European call option price.
        
        Formula:
        C = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
        
        Returns:
        --------
        float
            Call option price
        """
        d1 = self._d1()
        d2 = self._d2()
        
        call = (self.S0 * np.exp(-self.q * self.T) * norm.cdf(d1) - 
                self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        
        return max(call, 0.0)  # Guard against numerical errors
    
    def put_price(self) -> float:
        """
        Calculate European put option price.
        
        Formula:
        P = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)
        
        Returns:
        --------
        float
            Put option price
        """
        d1 = self._d1()
        d2 = self._d2()
        
        put = (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - 
               self.S0 * np.exp(-self.q * self.T) * norm.cdf(-d1))
        
        return max(put, 0.0)  # Guard against numerical errors
    
    def delta_call(self) -> float:
        """Calculate call delta (rate of change of option price w.r.t. stock price)."""
        d1 = self._d1()
        return np.exp(-self.q * self.T) * norm.cdf(d1)
    
    def delta_put(self) -> float:
        """Calculate put delta."""
        d1 = self._d1()
        return -np.exp(-self.q * self.T) * norm.cdf(-d1)
    
    def gamma(self) -> float:
        """Calculate gamma (rate of change of delta)."""
        d1 = self._d1()
        return (np.exp(-self.q * self.T) * norm.pdf(d1)) / (
            self.S0 * self.sigma * np.sqrt(self.T)
        )
    
    def vega(self) -> float:
        """Calculate vega (sensitivity to volatility)."""
        d1 = self._d1()
        return self.S0 * np.exp(-self.q * self.T) * norm.pdf(d1) * np.sqrt(self.T)
    
    def theta_call(self) -> float:
        """Calculate call theta (time decay)."""
        d1 = self._d1()
        d2 = self._d2()
        
        theta = (-self.S0 * np.exp(-self.q * self.T) * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) +
                 self.q * self.S0 * np.exp(-self.q * self.T) * norm.cdf(d1) -
                 self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        
        return theta / 365.0  # Convert to daily theta
    
    def theta_put(self) -> float:
        """Calculate put theta (time decay)."""
        d1 = self._d1()
        d2 = self._d2()
        
        theta = (-self.S0 * np.exp(-self.q * self.T) * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) -
                 self.q * self.S0 * np.exp(-self.q * self.T) * norm.cdf(-d1) +
                 self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
        
        return theta / 365.0  # Convert to daily theta
    
    def rho_call(self) -> float:
        """Calculate call rho (sensitivity to interest rate)."""
        d2 = self._d2()
        return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
    
    def rho_put(self) -> float:
        """Calculate put rho (sensitivity to interest rate)."""
        d2 = self._d2()
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
    
    def greeks_call(self) -> Dict[str, float]:
        """Calculate all Greeks for call option."""
        return {
            'delta': self.delta_call(),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta_call(),
            'rho': self.rho_call()
        }
    
    def greeks_put(self) -> Dict[str, float]:
        """Calculate all Greeks for put option."""
        return {
            'delta': self.delta_put(),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta_put(),
            'rho': self.rho_put()
        }
    
    def implied_volatility_call(self, market_price: float, 
                                tol: float = 1e-6, max_iter: int = 100) -> float:
        """
        Calculate implied volatility for call option using Newton-Raphson method.
        
        Parameters:
        -----------
        market_price : float
            Observed market price of call option
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        float
            Implied volatility
        """
        if market_price <= 0:
            return 0.0
        
        try:
            def objective(sigma):
                model = BlackScholesModel(self.S0, self.K, self.r, sigma, self.T, self.q)
                return model.call_price() - market_price
            
            # Use Brent's method for robustness
            iv = brentq(objective, 0.001, 5.0, xtol=tol, maxiter=max_iter)
            return max(iv, 0.0)
            
        except Exception as e:
            logger.warning(f"Failed to calculate implied volatility: {e}")
            return np.nan
    
    def implied_volatility_put(self, market_price: float, 
                               tol: float = 1e-6, max_iter: int = 100) -> float:
        """
        Calculate implied volatility for put option using Newton-Raphson method.
        
        Parameters:
        -----------
        market_price : float
            Observed market price of put option
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        float
            Implied volatility
        """
        if market_price <= 0:
            return 0.0
        
        try:
            def objective(sigma):
                model = BlackScholesModel(self.S0, self.K, self.r, sigma, self.T, self.q)
                return model.put_price() - market_price
            
            # Use Brent's method for robustness
            iv = brentq(objective, 0.001, 5.0, xtol=tol, maxiter=max_iter)
            return max(iv, 0.0)
            
        except Exception as e:
            logger.warning(f"Failed to calculate implied volatility: {e}")
            return np.nan
    
    def summary(self) -> Dict:
        """Get comprehensive pricing and Greeks summary."""
        return {
            'call_price': self.call_price(),
            'put_price': self.put_price(),
            'call_greeks': self.greeks_call(),
            'put_greeks': self.greeks_put(),
            'parameters': {
                'S0': self.S0,
                'K': self.K,
                'r': self.r,
                'sigma': self.sigma,
                'T': self.T,
                'q': self.q
            }
        }


# Standalone functions for convenience

def bs_call_price(S0: float, K: float, r: float, sigma: float, T: float, 
                  q: float = 0.0) -> float:
    """
    Black-Scholes European call option price (function interface).
    
    Parameters:
    -----------
    S0 : float
        Current stock price
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
        
    Returns:
    --------
    float
        Call option price
    """
    try:
        model = BlackScholesModel(S0, K, r, sigma, T, q)
        return model.call_price()
    except Exception as e:
        logger.error(f"Error calculating call price: {e}")
        return np.nan


def bs_put_price(S0: float, K: float, r: float, sigma: float, T: float, 
                 q: float = 0.0) -> float:
    """
    Black-Scholes European put option price (function interface).
    
    Parameters:
    -----------
    S0 : float
        Current stock price
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
        
    Returns:
    --------
    float
        Put option price
    """
    try:
        model = BlackScholesModel(S0, K, r, sigma, T, q)
        return model.put_price()
    except Exception as e:
        logger.error(f"Error calculating put price: {e}")
        return np.nan


def bs_greeks(S0: float, K: float, r: float, sigma: float, T: float, 
              q: float = 0.0, option_type: str = 'call') -> Dict[str, float]:
    """
    Calculate all Greeks for an option.
    
    Parameters:
    -----------
    S0, K, r, sigma, T, q : float
        Black-Scholes parameters
    option_type : str
        'call' or 'put'
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of Greeks (delta, gamma, vega, theta, rho)
    """
    try:
        model = BlackScholesModel(S0, K, r, sigma, T, q)
        if option_type.lower() == 'call':
            return model.greeks_call()
        else:
            return model.greeks_put()
    except Exception as e:
        logger.error(f"Error calculating Greeks: {e}")
        return {}


def calc_implied_vol(S0: float, K: float, r: float, market_price: float, 
                     T: float, q: float = 0.0, 
                     option_type: str = 'call') -> float:
    """
    Calculate implied volatility from market price.
    
    Parameters:
    -----------
    S0, K, r, T, q : float
        Black-Scholes parameters
    market_price : float
        Observed option price
    option_type : str
        'call' or 'put'
        
    Returns:
    --------
    float
        Implied volatility
    """
    try:
        model = BlackScholesModel(S0, K, r, 0.2, T, q)  # Initial guess for sigma
        if option_type.lower() == 'call':
            return model.implied_volatility_call(market_price)
        else:
            return model.implied_volatility_put(market_price)
    except Exception as e:
        logger.error(f"Error calculating implied volatility: {e}")
        return np.nan


def price_option_chain(S0: float, strikes: Union[list, np.ndarray], 
                      r: float, sigma: float, T: float, q: float = 0.0) -> pd.DataFrame:
    """
    Price an entire option chain (multiple strikes).
    
    Parameters:
    -----------
    S0 : float
        Stock price
    strikes : list or np.ndarray
        Array of strike prices
    r, sigma, T, q : float
        Black-Scholes parameters
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with call/put prices and Greeks
    """
    results = []
    
    for K in strikes:
        try:
            model = BlackScholesModel(S0, K, r, sigma, T, q)
            
            results.append({
                'strike': K,
                'call_price': model.call_price(),
                'put_price': model.put_price(),
                'call_delta': model.delta_call(),
                'put_delta': model.delta_put(),
                'gamma': model.gamma(),
                'vega': model.vega(),
                'call_theta': model.theta_call(),
                'put_theta': model.theta_put()
            })
        except Exception as e:
            logger.warning(f"Error pricing strike {K}: {e}")
            continue
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 70)
    print("Black-Scholes Options Pricing Model - Examples")
    print("=" * 70)
    
    # Example 1: Simple call and put pricing
    print("\n[Example 1] Simple Option Pricing")
    print("-" * 70)
    
    try:
        call_price = bs_call_price(S0=100, K=105, r=0.05, sigma=0.2, T=0.25)
        put_price = bs_put_price(S0=100, K=105, r=0.05, sigma=0.2, T=0.25)
        
        print(f"Stock Price: $100.00")
        print(f"Strike Price: $105.00")
        print(f"Risk-Free Rate: 5%")
        print(f"Volatility: 20%")
        print(f"Time to Maturity: 0.25 years (3 months)")
        print(f"\n✓ Call Price: ${call_price:.2f}")
        print(f"✓ Put Price: ${put_price:.2f}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 2: Greeks calculation
    print("\n[Example 2] Greeks Calculation")
    print("-" * 70)
    
    try:
        greeks = bs_greeks(100, 105, 0.05, 0.2, 0.25, option_type='call')
        print(f"Call Option Greeks:")
        for greek, value in greeks.items():
            print(f"  {greek.upper():6s}: {value:10.4f}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 3: Implied volatility
    print("\n[Example 3] Implied Volatility Calculation")
    print("-" * 70)
    
    try:
        market_price = 2.50
        iv = calc_implied_vol(100, 105, 0.05, market_price, 0.25, option_type='call')
        print(f"Market Price of Call: ${market_price:.2f}")
        print(f"✓ Implied Volatility: {iv:.2%}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 4: Option chain pricing
    print("\n[Example 4] Option Chain Pricing")
    print("-" * 70)
    
    try:
        strikes = np.array([95, 100, 105, 110, 115])
        chain = price_option_chain(100, strikes, 0.05, 0.2, 0.25)
        print(chain[['strike', 'call_price', 'put_price', 'call_delta', 'gamma']].to_string(index=False))
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 70)
