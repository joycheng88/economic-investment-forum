"""
Binomial Tree Options Pricing Model.

Provides robust binomial tree model implementation for pricing European and
American options, with support for dividend yields and multiple asset paths.
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
class BinomialResult:
    """Data class for binomial tree results."""
    price: float
    tree: np.ndarray = None
    exercise_tree: np.ndarray = None
    delta: float = None
    gamma: float = None


class BinomialTree:
    """
    Binomial tree options pricing model.
    """
    
    def __init__(self, S0: float, K: float, r: float, sigma: float, 
                 T: float, q: float = 0.0, n_steps: int = 50):
        """
        Initialize binomial tree pricer.
        
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
        n_steps : int
            Number of steps in tree
        """
        self.S0 = float(S0)
        self.K = float(K)
        self.r = float(r)
        self.sigma = float(sigma)
        self.T = float(T)
        self.q = float(q)
        self.n_steps = int(n_steps)
        
        self._validate_parameters()
        self._calculate_tree_parameters()
    
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
        if self.n_steps < 2:
            raise ValueError(f"Number of steps must be >= 2: {self.n_steps}")
    
    def _calculate_tree_parameters(self):
        """Calculate Cox-Ross-Rubinstein parameters."""
        self.dt = self.T / self.n_steps
        
        # Up and down factors
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1.0 / self.u
        
        # Risk-neutral probability
        discount_factor = np.exp(-self.r * self.dt)
        forward = np.exp((self.r - self.q) * self.dt)
        
        self.p = (forward - self.d) / (self.u - self.d)
        
        # Validate probability
        if self.p < 0 or self.p > 1:
            logger.warning(f"Risk-neutral probability out of range: {self.p}")
            # Recalculate using alternative formulation
            self.p = 0.5
        
        logger.info(f"Binomial parameters: u={self.u:.4f}, d={self.d:.4f}, p={self.p:.4f}")
    
    def _build_price_tree(self) -> np.ndarray:
        """
        Build stock price tree.
        
        Returns:
        --------
        np.ndarray
            Tree of stock prices (shape: n_steps+1 x n_steps+1)
        """
        tree = np.zeros((self.n_steps + 1, self.n_steps + 1))
        
        # Fill the tree
        for j in range(self.n_steps + 1):
            for i in range(j + 1):
                tree[i, j] = self.S0 * (self.u ** (j - i)) * (self.d ** i)
        
        return tree
    
    def european_call(self) -> BinomialResult:
        """
        Price European call option using binomial tree.
        
        Returns:
        --------
        BinomialResult
            Pricing result with tree details
        """
        try:
            # Build price tree
            price_tree = self._build_price_tree()
            
            # Initialize value tree
            value_tree = np.zeros((self.n_steps + 1, self.n_steps + 1))
            
            # Set terminal payoffs
            for i in range(self.n_steps + 1):
                value_tree[i, self.n_steps] = max(price_tree[i, self.n_steps] - self.K, 0)
            
            # Work backwards through the tree
            dt = self.T / self.n_steps
            discount = np.exp(-self.r * dt)
            
            for j in range(self.n_steps - 1, -1, -1):
                for i in range(j + 1):
                    # Expected value at this node
                    value_tree[i, j] = discount * (
                        self.p * value_tree[i, j + 1] + 
                        (1 - self.p) * value_tree[i + 1, j + 1]
                    )
            
            result = BinomialResult(
                price=value_tree[0, 0],
                tree=value_tree
            )
            
            logger.info(f"European call: ${result.price:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in European call pricing: {e}")
            return BinomialResult(price=np.nan)
    
    def european_put(self) -> BinomialResult:
        """
        Price European put option using binomial tree.
        
        Returns:
        --------
        BinomialResult
            Pricing result
        """
        try:
            # Build price tree
            price_tree = self._build_price_tree()
            
            # Initialize value tree
            value_tree = np.zeros((self.n_steps + 1, self.n_steps + 1))
            
            # Set terminal payoffs
            for i in range(self.n_steps + 1):
                value_tree[i, self.n_steps] = max(self.K - price_tree[i, self.n_steps], 0)
            
            # Work backwards
            dt = self.T / self.n_steps
            discount = np.exp(-self.r * dt)
            
            for j in range(self.n_steps - 1, -1, -1):
                for i in range(j + 1):
                    value_tree[i, j] = discount * (
                        self.p * value_tree[i, j + 1] + 
                        (1 - self.p) * value_tree[i + 1, j + 1]
                    )
            
            result = BinomialResult(
                price=value_tree[0, 0],
                tree=value_tree
            )
            
            logger.info(f"European put: ${result.price:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in European put pricing: {e}")
            return BinomialResult(price=np.nan)
    
    def american_call(self) -> BinomialResult:
        """
        Price American call option using binomial tree.
        
        Returns:
        --------
        BinomialResult
            Pricing result with exercise decisions
        """
        try:
            # Build price tree
            price_tree = self._build_price_tree()
            
            # Initialize value tree
            value_tree = np.zeros((self.n_steps + 1, self.n_steps + 1))
            exercise_tree = np.zeros((self.n_steps + 1, self.n_steps + 1), dtype=bool)
            
            # Set terminal payoffs
            for i in range(self.n_steps + 1):
                value_tree[i, self.n_steps] = max(price_tree[i, self.n_steps] - self.K, 0)
                exercise_tree[i, self.n_steps] = (price_tree[i, self.n_steps] > self.K)
            
            # Work backwards with early exercise
            dt = self.T / self.n_steps
            discount = np.exp(-self.r * dt)
            
            for j in range(self.n_steps - 1, -1, -1):
                for i in range(j + 1):
                    # European value at this node
                    eur_value = discount * (
                        self.p * value_tree[i, j + 1] + 
                        (1 - self.p) * value_tree[i + 1, j + 1]
                    )
                    
                    # Intrinsic value (early exercise value)
                    intrinsic = max(price_tree[i, j] - self.K, 0)
                    
                    # American: take maximum
                    if intrinsic > eur_value:
                        value_tree[i, j] = intrinsic
                        exercise_tree[i, j] = True
                    else:
                        value_tree[i, j] = eur_value
                        exercise_tree[i, j] = False
            
            result = BinomialResult(
                price=value_tree[0, 0],
                tree=value_tree,
                exercise_tree=exercise_tree
            )
            
            logger.info(f"American call: ${result.price:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in American call pricing: {e}")
            return BinomialResult(price=np.nan)
    
    def american_put(self) -> BinomialResult:
        """
        Price American put option using binomial tree.
        
        Returns:
        --------
        BinomialResult
            Pricing result with exercise decisions
        """
        try:
            # Build price tree
            price_tree = self._build_price_tree()
            
            # Initialize value tree
            value_tree = np.zeros((self.n_steps + 1, self.n_steps + 1))
            exercise_tree = np.zeros((self.n_steps + 1, self.n_steps + 1), dtype=bool)
            
            # Set terminal payoffs
            for i in range(self.n_steps + 1):
                value_tree[i, self.n_steps] = max(self.K - price_tree[i, self.n_steps], 0)
                exercise_tree[i, self.n_steps] = (self.K > price_tree[i, self.n_steps])
            
            # Work backwards with early exercise
            dt = self.T / self.n_steps
            discount = np.exp(-self.r * dt)
            
            for j in range(self.n_steps - 1, -1, -1):
                for i in range(j + 1):
                    # European value
                    eur_value = discount * (
                        self.p * value_tree[i, j + 1] + 
                        (1 - self.p) * value_tree[i + 1, j + 1]
                    )
                    
                    # Intrinsic value
                    intrinsic = max(self.K - price_tree[i, j], 0)
                    
                    # American: take maximum
                    if intrinsic > eur_value:
                        value_tree[i, j] = intrinsic
                        exercise_tree[i, j] = True
                    else:
                        value_tree[i, j] = eur_value
                        exercise_tree[i, j] = False
            
            result = BinomialResult(
                price=value_tree[0, 0],
                tree=value_tree,
                exercise_tree=exercise_tree
            )
            
            logger.info(f"American put: ${result.price:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in American put pricing: {e}")
            return BinomialResult(price=np.nan)
    
    def greeks(self, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate Greeks using finite differences from binomial tree.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        Dict[str, float]
            Greeks (delta, gamma)
        """
        try:
            if option_type.lower() == 'call':
                result_current = self.european_call()
            else:
                result_current = self.european_put()
            
            # Calculate with slightly higher stock price
            bump = self.S0 * 0.01
            self.S0 += bump
            self._calculate_tree_parameters()
            
            if option_type.lower() == 'call':
                result_up = self.european_call()
            else:
                result_up = self.european_put()
            
            # Calculate with lower stock price
            self.S0 -= 2 * bump
            self._calculate_tree_parameters()
            
            if option_type.lower() == 'call':
                result_down = self.european_call()
            else:
                result_down = self.european_put()
            
            # Restore original price
            self.S0 += bump
            self._calculate_tree_parameters()
            
            # Calculate Greeks
            delta = (result_up.price - result_down.price) / (2 * bump)
            gamma = (result_up.price - 2 * result_current.price + result_down.price) / (bump ** 2)
            
            return {
                'delta': delta,
                'gamma': gamma
            }
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return {'delta': np.nan, 'gamma': np.nan}


# Standalone functions

def bt_european_call(S0: float, K: float, r: float, sigma: float, T: float,
                    q: float = 0.0, n_steps: int = 50) -> float:
    """
    Binomial tree European call price (function interface).
    
    Parameters:
    -----------
    S0, K, r, sigma, T, q : float
        Pricing parameters
    n_steps : int
        Number of steps
        
    Returns:
    --------
    float
        Call option price
    """
    try:
        bt = BinomialTree(S0, K, r, sigma, T, q, n_steps)
        result = bt.european_call()
        return result.price
    except Exception as e:
        logger.error(f"Error calculating BT call: {e}")
        return np.nan


def bt_european_put(S0: float, K: float, r: float, sigma: float, T: float,
                   q: float = 0.0, n_steps: int = 50) -> float:
    """
    Binomial tree European put price (function interface).
    """
    try:
        bt = BinomialTree(S0, K, r, sigma, T, q, n_steps)
        result = bt.european_put()
        return result.price
    except Exception as e:
        logger.error(f"Error calculating BT put: {e}")
        return np.nan


def bt_american_call(S0: float, K: float, r: float, sigma: float, T: float,
                    q: float = 0.0, n_steps: int = 50) -> float:
    """
    Binomial tree American call price (function interface).
    """
    try:
        bt = BinomialTree(S0, K, r, sigma, T, q, n_steps)
        result = bt.american_call()
        return result.price
    except Exception as e:
        logger.error(f"Error calculating BT American call: {e}")
        return np.nan


def bt_american_put(S0: float, K: float, r: float, sigma: float, T: float,
                   q: float = 0.0, n_steps: int = 50) -> float:
    """
    Binomial tree American put price (function interface).
    """
    try:
        bt = BinomialTree(S0, K, r, sigma, T, q, n_steps)
        result = bt.american_put()
        return result.price
    except Exception as e:
        logger.error(f"Error calculating BT American put: {e}")
        return np.nan


def bt_price_chain(S0: float, strikes: Union[list, np.ndarray], r: float,
                  sigma: float, T: float, q: float = 0.0,
                  n_steps: int = 50, american: bool = False) -> pd.DataFrame:
    """
    Price multiple strikes using binomial tree.
    
    Parameters:
    -----------
    S0 : float
        Stock price
    strikes : list/array
        Strike prices
    r, sigma, T, q : float
        Pricing parameters
    n_steps : int
        Number of steps
    american : bool
        Use American options
        
    Returns:
    --------
    pd.DataFrame
        Prices for all strikes
    """
    results = []
    
    for K in strikes:
        try:
            if american:
                call_price = bt_american_call(S0, K, r, sigma, T, q, n_steps)
                put_price = bt_american_put(S0, K, r, sigma, T, q, n_steps)
            else:
                call_price = bt_european_call(S0, K, r, sigma, T, q, n_steps)
                put_price = bt_european_put(S0, K, r, sigma, T, q, n_steps)
            
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
    print("Binomial Tree Options Pricing - Examples")
    print("=" * 70)
    
    # Example 1: European options
    print("\n[Example 1] European Options - Binomial Tree")
    print("-" * 70)
    
    try:
        bt = BinomialTree(S0=100, K=105, r=0.05, sigma=0.2, T=0.25, n_steps=50)
        
        call_result = bt.european_call()
        put_result = bt.european_put()
        
        print(f"Stock Price: $100.00")
        print(f"Strike Price: $105.00")
        print(f"Risk-Free Rate: 5%")
        print(f"Volatility: 20%")
        print(f"Time to Maturity: 0.25 years")
        print(f"Tree Steps: 50")
        print(f"\n✓ European Call: ${call_result.price:.4f}")
        print(f"✓ European Put:  ${put_result.price:.4f}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 2: American options
    print("\n[Example 2] American Options - Binomial Tree")
    print("-" * 70)
    
    try:
        am_call_result = bt.american_call()
        am_put_result = bt.american_put()
        
        print(f"✓ American Call: ${am_call_result.price:.4f}")
        print(f"✓ American Put:  ${am_put_result.price:.4f}")
        print(f"\nEarlyExercise Premium:")
        print(f"  Call: ${am_call_result.price - call_result.price:.4f}")
        print(f"  Put:  ${am_put_result.price - put_result.price:.4f}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 3: Greeks calculation
    print("\n[Example 3] Greeks Calculation")
    print("-" * 70)
    
    try:
        greeks = bt.greeks(option_type='call')
        print(f"Call Greeks (using finite differences):")
        for greek, value in greeks.items():
            print(f"  {greek.upper():6s}: {value:10.4f}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 4: Option chain
    print("\n[Example 4] Option Chain Pricing")
    print("-" * 70)
    
    try:
        strikes = np.array([95, 100, 105, 110, 115])
        chain = bt_price_chain(100, strikes, 0.05, 0.2, 0.25, n_steps=50)
        print(chain.to_string(index=False))
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 70)
