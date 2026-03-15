"""
Stochastic Volatility Options Pricing Models.

Provides implementations of advanced volatility models for capturing volatility
smile/skew and more realistic option pricing:
- Heston: Semi-closed-form solution for equities/FX
- SABR: Volatility surface fitting for rates
- Hull-White: Stochastic correlation models
"""

import numpy as np
from scipy.integrate import quad, odeint
from scipy.optimize import minimize
from scipy.special import hyp1f1
import pandas as pd
import logging
from typing import Dict, Tuple, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SVResult:
    """Data class for stochastic volatility results."""
    price: float
    delta: float = None
    gamma: float = None
    vega: float = None
    theta: float = None


class HestonModel:
    """
    Heston Stochastic Volatility Model.
    
    Models volatility as mean-reverting CIR process:
    dS = μS dt + √v S dW_S
    dv = κ(θ - v) dt + σ_v √v dW_v
    
    Parameters:
    kappa: Mean reversion speed
    theta: Long-term variance
    sigma_v: Volatility of volatility
    rho: Correlation between S and v
    v0: Initial variance
    """
    
    def __init__(self, S0: float, K: float, r: float, T: float,
                 kappa: float = 3.0, theta: float = 0.04, 
                 sigma_v: float = 0.4, rho: float = -0.3, 
                 v0: float = None):
        """
        Initialize Heston model.
        
        Parameters:
        -----------
        S0 : float
            Initial stock price
        K : float
            Strike price
        r : float
            Risk-free rate
        T : float
            Time to maturity
        kappa : float
            Mean reversion speed (default 3.0)
        theta : float
            Long-term variance (default 0.04)
        sigma_v : float
            Volatility of volatility (default 0.4)
        rho : float
            Correlation with spot (default -0.3)
        v0 : float
            Initial variance (default: theta)
        """
        self.S0 = float(S0)
        self.K = float(K)
        self.r = float(r)
        self.T = float(T)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma_v = float(sigma_v)
        self.rho = float(rho)
        self.v0 = float(v0) if v0 is not None else self.theta
        
        self._validate_params()
    
    def _validate_params(self):
        """Validate Heston parameters (Feller condition check)."""
        if 2 * self.kappa * self.theta <= self.sigma_v ** 2:
            logger.warning(
                "Feller condition violated: 2κθ ≤ σ_v². "
                "Variance may become negative. Consider adjusting parameters."
            )
        if abs(self.rho) >= 1:
            raise ValueError(f"Correlation must be in (-1, 1), got {self.rho}")
    
    def characteristic_function(self, w: float, u: float = 0.0) -> complex:
        """
        Calculate characteristic function for calibration/pricing.
        
        Parameters:
        -----------
        w : float
            Integration variable
        u : float
            Heston parameter
        
        Returns:
        --------
        complex
            Characteristic function value
        """
        lam = np.sqrt((self.rho * self.sigma_v * w * 1j - self.kappa) ** 2 + 
                     self.sigma_v ** 2 * (w * 1j + w ** 2))
        
        d = (self.rho * self.sigma_v * w * 1j - self.kappa + lam) / (
            self.rho * self.sigma_v * w * 1j - self.kappa - lam
        )
        
        g = 1 - d * np.exp(self.T * lam)
        
        coeff = np.exp(1j * w * np.log(self.S0 / self.K) + self.r * self.T * 1j * w)
        
        term1 = (1 - d * np.exp(self.T * lam)) / (self.sigma_v ** 2 * (1 - d))
        term2 = self.kappa * self.theta / (self.sigma_v ** 2) * (
            (self.rho * self.sigma_v * w * 1j - self.kappa + lam) * self.T - 
            2 * np.log((1 - d * np.exp(self.T * lam)) / (1 - d))
        )
        
        return coeff * np.exp(term1 * self.v0 + term2)
    
    def call_price_quad(self, n_quad: int = 200) -> float:
        """
        Price European call using Heston formula via numerical integration.
        
        Parameters:
        -----------
        n_quad : int
            Number of quadrature points
        
        Returns:
        --------
        float
            Call option price
        """
        # Heston call price formula
        def integrand(x):
            z = 1j * x
            arg = (np.log(self.S0 / self.K) + self.r * self.T) * z - 0.5 * z * (z + 1j)
            
            # Characteristic function approach
            numerator = np.real(
                np.exp(-self.r * self.T) * 
                self.characteristic_function(x - 0.5j) / (z * (z + 1j))
            )
            denominator = x ** 2 + 0.25
            
            return numerator / denominator
        
        try:
            integral, _ = quad(integrand, 0, 100, limit=100)
            call = self.S0 - np.sqrt(self.S0 * self.K) * np.exp(-0.5 * self.r * self.T) / np.pi * integral
            return max(call, 0)
        except Exception as e:
            logger.warning(f"Quadrature failed: {str(e)}. Using Monte Carlo fallback.")
            return self.call_price_monte_carlo()
    
    def call_price_monte_carlo(self, n_paths: int = 10000, n_steps: int = 100) -> float:
        """
        Price European call using Monte Carlo simulation.
        
        Parameters:
        -----------
        n_paths : int
            Number of simulation paths
        n_steps : int
            Number of time steps
        
        Returns:
        --------
        float
            Call option price estimate
        """
        dt = self.T / n_steps
        paths_S = np.zeros((n_steps + 1, n_paths))
        paths_v = np.zeros((n_steps + 1, n_paths))
        
        paths_S[0, :] = self.S0
        paths_v[0, :] = self.v0
        
        # Generate correlated Brownian motions
        for step in range(n_steps):
            Z1 = np.random.standard_normal(n_paths)
            Z2 = np.random.standard_normal(n_paths)
            Z_v = Z2
            Z_S = self.rho * Z2 + np.sqrt(1 - self.rho ** 2) * Z1
            
            # Update variance (Euler scheme)
            paths_v[step + 1, :] = np.maximum(
                paths_v[step, :] + self.kappa * (self.theta - paths_v[step, :]) * dt +
                self.sigma_v * np.sqrt(np.maximum(paths_v[step, :], 0)) * np.sqrt(dt) * Z_v,
                0.001  # Prevent negative variance
            )
            
            # Update spot
            paths_S[step + 1, :] = paths_S[step, :] * np.exp(
                (self.r - 0.5 * paths_v[step, :]) * dt +
                np.sqrt(np.maximum(paths_v[step, :], 0) * dt) * Z_S
            )
        
        payoff = np.maximum(paths_S[-1, :] - self.K, 0)
        call_price = np.exp(-self.r * self.T) * np.mean(payoff)
        
        return call_price
    
    def put_price(self, method: str = 'mc') -> float:
        """
        Price European put using put-call parity.
        
        Parameters:
        -----------
        method : str
            'mc' for Monte Carlo, 'quad' for quadrature
        
        Returns:
        --------
        float
            Put option price
        """
        if method == 'quad':
            call = self.call_price_quad()
        else:
            call = self.call_price_monte_carlo()
        
        put = call - self.S0 + self.K * np.exp(-self.r * self.T)
        return max(put, 0)
    
    def call_price(self, method: str = 'mc') -> float:
        """Get call price using specified method."""
        if method == 'quad':
            return self.call_price_quad()
        else:
            return self.call_price_monte_carlo()


class SABRModel:
    """
    SABR (Stochastic Alpha Beta Rho) Model.
    
    Models forward rate dynamics:
    dF = α F^β dW_F
    dα = ν α dW_α
    d<W_F, W_α> = ρ dt
    
    Primarily used for interest rate and FX volatility surfaces.
    Provides closed-form approximation for option prices.
    """
    
    def __init__(self, F: float, K: float, T: float,
                 alpha: float, beta: float = 1.0, 
                 nu: float = 0.4, rho: float = -0.3,
                 r: float = 0.0):
        """
        Initialize SABR model.
        
        Parameters:
        -----------
        F : float
            Forward rate
        K : float
            Strike
        T : float
            Time to maturity
        alpha : float
            Initial volatility
        beta : float
            Elasticity (default 1.0 for lognormal)
        nu : float
            Volatility of volatility
        rho : float
            Correlation
        r : float
            Discount rate
        """
        self.F = float(F)
        self.K = float(K)
        self.T = float(T)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.nu = float(nu)
        self.rho = float(rho)
        self.r = float(r)
        
        if not (0 <= self.beta <= 1):
            logger.warning(f"Beta outside [0,1]: {self.beta}")
    
    def implied_vol_bbg(self) -> float:
        """
        Bloomberg-Bartlett-Goyal approximation for implied volatility.
        Simple and commonly used in practice.
        
        Returns:
        --------
        float
            Implied volatility
        """
        try:
            if abs(self.F - self.K) < 1e-6:
                # ATM volatility - simplified formula
                sigma_atm = self.alpha / ((self.F) ** (1 - self.beta))
                
                # Add corrections for vol of vol and correlation
                sigma_atm *= (1 + ((2 - 3 * self.rho ** 2) / 24) * 
                             (self.nu ** 2 / (self.F ** (2 * (1 - self.beta)))) * self.T)
                
                return max(sigma_atm, 0.001)  # Ensure positive
            else:
                # OTM volatility - more complex calculation
                moneyness = self.K / self.F
                
                # Compute z for the log-normal case
                z = (self.nu / self.alpha) * (self.F ** (1 - self.beta)) * np.log(moneyness)
                
                if abs(z) < 1e-8:
                    sigma = self.alpha / (self.F ** (1 - self.beta))
                else:
                    # Chi function
                    numerator = np.sqrt(1 - 2 * self.rho * z + z ** 2) + z - self.rho
                    denominator = 1 - self.rho
                    
                    if numerator <= 0 or denominator <= 0:
                        # Fallback for edge cases
                        sigma = self.alpha / (self.F ** (1 - self.beta))
                    else:
                        chi = np.log(numerator / denominator)
                        sigma = self.alpha * z / chi * (self.F ** (1 - self.beta)) ** (-1)
                
                # Apply smile adjustment
                smile_adjustment = (1 + 
                    ((1 - self.beta) ** 2 / 24) * (np.log(moneyness) ** 2) * self.T +
                    (self.rho * self.beta * self.nu / (4 * self.F ** (1 - self.beta))) * 
                    np.log(moneyness) * self.T)
                
                sigma = max(sigma * smile_adjustment, 0.001)
                return sigma
        except Exception as e:
            logger.warning(f"SABR IV calculation failed: {e}. Using base alpha.")
            return max(self.alpha, 0.001)
    
    def black_call_price(self) -> float:
        """
        Price call using Black's model with SABR implied volatility.
        
        Returns:
        --------
        float
            Call option price
        """
        from scipy.stats import norm
        sigma = self.implied_vol_bbg()
        d1 = (np.log(self.F / self.K) + 0.5 * sigma ** 2 * self.T) / (
            sigma * np.sqrt(self.T)
        )
        d2 = d1 - sigma * np.sqrt(self.T)
        
        call = np.exp(-self.r * self.T) * (
            self.F * norm.cdf(d1) - 
            self.K * norm.cdf(d2)
        )
        return call
    
    def call_price(self) -> float:
        """
        Get SABR call price using Black's model.
        
        Returns:
        --------
        float
            Call option price
        """
        from scipy.stats import norm
        sigma = self.implied_vol_bbg()
        d1 = (np.log(self.F / self.K) + 0.5 * sigma ** 2 * self.T) / (
            sigma * np.sqrt(self.T)
        )
        d2 = d1 - sigma * np.sqrt(self.T)
        
        call = np.exp(-self.r * self.T) * (
            self.F * norm.cdf(d1) - self.K * norm.cdf(d2)
        )
        return call
    
    def put_price(self) -> float:
        """
        Get SABR put price using Black's model.
        
        Returns:
        --------
        float
            Put option price
        """
        from scipy.stats import norm
        sigma = self.implied_vol_bbg()
        d1 = (np.log(self.F / self.K) + 0.5 * sigma ** 2 * self.T) / (
            sigma * np.sqrt(self.T)
        )
        d2 = d1 - sigma * np.sqrt(self.T)
        
        put = np.exp(-self.r * self.T) * (
            self.K * norm.cdf(-d2) - self.F * norm.cdf(-d1)
        )
        return put
    
    def volatility_surface(self, strikes: np.ndarray) -> np.ndarray:
        """
        Generate volatility surface across strikes.
        
        Parameters:
        -----------
        strikes : np.ndarray
            Array of strike prices
        
        Returns:
        --------
        np.ndarray
            Implied volatilities for each strike
        """
        vols = np.zeros_like(strikes, dtype=float)
        for i, K in enumerate(strikes):
            sabr_temp = SABRModel(self.F, K, self.T, self.alpha, 
                                 self.beta, self.nu, self.rho, self.r)
            vols[i] = sabr_temp.implied_vol_bbg()
        return vols


class HullWhiteModel:
    """
    Hull-White Stochastic Volatility Model.
    
    Extension combining stochastic rates and volatility:
    dS = (r - q)S dt + √v S dW_S
    dr = α(θ(t) - r) dt + σ_r dW_r
    dv = κ(v̄ - v) dt + σ_v √v dW_v
    
    Useful for caps, floors, swaptions and equity options under stochastic rates.
    """
    
    def __init__(self, S0: float, K: float, T: float,
                 r0: float = 0.03, alpha_r: float = 0.15,
                 sigma_r: float = 0.01, kappa_v: float = 2.0,
                 v_bar: float = 0.04, sigma_v: float = 0.3,
                 rho_sr: float = 0.1, rho_sv: float = -0.3,
                 q: float = 0.0):
        """
        Initialize Hull-White model.
        
        Parameters:
        -----------
        S0 : float
            Initial spot price
        K : float
            Strike price
        T : float
            Time to maturity
        r0 : float
            Initial short rate
        alpha_r : float
            Mean reversion speed for rates
        sigma_r : float
            Rate volatility
        kappa_v : float
            Mean reversion speed for variance
        v_bar : float
            Long-term variance level
        sigma_v : float
            Volatility of variance
        rho_sr : float
            Correlation spot-rate
        rho_sv : float
            Correlation spot-variance
        q : float
            Dividend yield
        """
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r0 = float(r0)
        self.alpha_r = float(alpha_r)
        self.sigma_r = float(sigma_r)
        self.kappa_v = float(kappa_v)
        self.v_bar = float(v_bar)
        self.sigma_v = float(sigma_v)
        self.rho_sr = float(rho_sr)
        self.rho_sv = float(rho_sv)
        self.q = float(q)
    
    def call_price_monte_carlo(self, n_paths: int = 10000, 
                               n_steps: int = 100) -> Tuple[float, float]:
        """
        Price European call using Monte Carlo with three factors.
        
        Parameters:
        -----------
        n_paths : int
            Number of simulation paths
        n_steps : int
            Number of time steps
        
        Returns:
        --------
        Tuple[float, float]
            (Call price, Standard error)
        """
        dt = self.T / n_steps
        
        prices = np.zeros((n_steps + 1, n_paths))
        rates = np.zeros((n_steps + 1, n_paths))
        vars = np.zeros((n_steps + 1, n_paths))
        
        prices[0, :] = self.S0
        rates[0, :] = self.r0
        vars[0, :] = self.v_bar
        
        # Simulate paths
        for step in range(n_steps):
            # Generate three independent Brownian motions
            Z_S = np.random.standard_normal(n_paths)
            Z_r = np.random.standard_normal(n_paths)
            Z_v = np.random.standard_normal(n_paths)
            
            # Make correlated
            Z_r_corr = self.rho_sr * Z_S + np.sqrt(1 - self.rho_sr ** 2) * Z_r
            Z_v_corr = self.rho_sv * Z_S + np.sqrt(1 - self.rho_sv ** 2) * Z_v
            
            # Update rates (Hull-White)
            rates[step + 1, :] = rates[step, :] + self.alpha_r * (
                0.03 - rates[step, :]
            ) * dt + self.sigma_r * np.sqrt(dt) * Z_r_corr
            
            # Update variance (CIR)
            vars[step + 1, :] = np.maximum(
                vars[step, :] + self.kappa_v * (self.v_bar - vars[step, :]) * dt +
                self.sigma_v * np.sqrt(np.maximum(vars[step, :], 0)) * np.sqrt(dt) * Z_v_corr,
                0.001
            )
            
            # Update spot
            drift = (rates[step, :] - self.q - 0.5 * vars[step, :])
            prices[step + 1, :] = prices[step, :] * np.exp(
                drift * dt + np.sqrt(np.maximum(vars[step, :], 0) * dt) * Z_S
            )
        
        # Discount payoffs by final rates (better approximation)
        payoff = np.maximum(prices[-1, :] - self.K, 0)
        discount_factors = np.exp(-np.sum(rates[1:, :], axis=0) * dt)
        call_prices = payoff * discount_factors
        
        call_price = np.mean(call_prices)
        std_error = np.std(call_prices) / np.sqrt(n_paths)
        
        return call_price, std_error
    
    def put_price_monte_carlo(self, n_paths: int = 10000,
                              n_steps: int = 100) -> Tuple[float, float]:
        """
        Price European put using Monte Carlo.
        
        Returns:
        --------
        Tuple[float, float]
            (Put price, Standard error)
        """
        dt = self.T / n_steps
        
        prices = np.zeros((n_steps + 1, n_paths))
        rates = np.zeros((n_steps + 1, n_paths))
        vars = np.zeros((n_steps + 1, n_paths))
        
        prices[0, :] = self.S0
        rates[0, :] = self.r0
        vars[0, :] = self.v_bar
        
        for step in range(n_steps):
            Z_S = np.random.standard_normal(n_paths)
            Z_r = np.random.standard_normal(n_paths)
            Z_v = np.random.standard_normal(n_paths)
            
            Z_r_corr = self.rho_sr * Z_S + np.sqrt(1 - self.rho_sr ** 2) * Z_r
            Z_v_corr = self.rho_sv * Z_S + np.sqrt(1 - self.rho_sv ** 2) * Z_v
            
            rates[step + 1, :] = rates[step, :] + self.alpha_r * (
                0.03 - rates[step, :]
            ) * dt + self.sigma_r * np.sqrt(dt) * Z_r_corr
            
            vars[step + 1, :] = np.maximum(
                vars[step, :] + self.kappa_v * (self.v_bar - vars[step, :]) * dt +
                self.sigma_v * np.sqrt(np.maximum(vars[step, :], 0)) * np.sqrt(dt) * Z_v_corr,
                0.001
            )
            
            drift = (rates[step, :] - self.q - 0.5 * vars[step, :])
            prices[step + 1, :] = prices[step, :] * np.exp(
                drift * dt + np.sqrt(np.maximum(vars[step, :], 0) * dt) * Z_S
            )
        
        payoff = np.maximum(self.K - prices[-1, :], 0)
        discount_factors = np.exp(-np.sum(rates[1:, :], axis=0) * dt)
        put_prices = payoff * discount_factors
        
        put_price = np.mean(put_prices)
        std_error = np.std(put_prices) / np.sqrt(n_paths)
        
        return put_price, std_error


# Utility functions
def compare_stochastic_vol_models(S0: float, K: float, r: float, T: float,
                                  sigma: float) -> pd.DataFrame:
    """
    Compare all three stochastic volatility models with Black-Scholes.
    
    Parameters:
    -----------
    S0 : float
        Initial spot price
    K : float
        Strike price
    r : float
        Risk-free rate
    T : float
        Time to maturity
    sigma : float
        Base volatility estimate
    
    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    results = {}
    
    # Black-Scholes (benchmark)
    from black_scholes import bs_call_price, bs_put_price
    bs_call = bs_call_price(S0, K, r, sigma, T)
    bs_put = bs_put_price(S0, K, r, sigma, T)
    results['Black-Scholes'] = {'Call': bs_call, 'Put': bs_put, 'Model Type': 'Constant Vol'}
    
    # Heston
    heston = HestonModel(S0, K, r, T, kappa=3.0, theta=sigma**2, 
                        sigma_v=0.4, rho=-0.3, v0=sigma**2)
    h_call = heston.call_price(method='mc')
    h_put = heston.put_price(method='mc')
    results['Heston'] = {'Call': h_call, 'Put': h_put, 'Model Type': 'Stochastic Vol'}
    
    # SABR
    sabr = SABRModel(S0, K, T, alpha=sigma, beta=1.0, nu=0.4, rho=-0.3, r=r)
    s_call = sabr.call_price()
    s_put = sabr.put_price()
    results['SABR'] = {'Call': s_call, 'Put': s_put, 'Model Type': 'Vol Surface'}
    
    # Hull-White
    hw = HullWhiteModel(S0, K, T, r0=r, alpha_r=0.15, sigma_r=0.01,
                       kappa_v=2.0, v_bar=sigma**2, sigma_v=0.3)
    hw_call, _ = hw.call_price_monte_carlo(n_paths=5000)
    hw_put, _ = hw.put_price_monte_carlo(n_paths=5000)
    results['Hull-White'] = {'Call': hw_call, 'Put': hw_put, 'Model Type': 'Multi-factor'}
    
    return pd.DataFrame(results).T


if __name__ == "__main__":
    # Quick test
    print("Testing Stochastic Volatility Models\n")
    
    S0, K, r, T, sigma = 100, 105, 0.05, 0.25, 0.20
    
    print("=" * 70)
    print("Heston Model Test")
    print("=" * 70)
    heston = HestonModel(S0, K, r, T)
    print(f"Call Price (MC): ${heston.call_price_monte_carlo():.4f}")
    print(f"Put Price (MC):  ${heston.put_price():.4f}")
    
    print("\n" + "=" * 70)
    print("SABR Model Test")
    print("=" * 70)
    sabr = SABRModel(S0, K, T, alpha=0.20)
    print(f"Implied Vol (ATM): {sabr.implied_vol_bbg():.4f}")
    print(f"Call Price: ${sabr.call_price():.4f}")
    print(f"Put Price:  ${sabr.put_price():.4f}")
    
    print("\n" + "=" * 70)
    print("Hull-White Model Test")
    print("=" * 70)
    hw = HullWhiteModel(S0, K, T, r0=r)
    hw_call, call_se = hw.call_price_monte_carlo(n_paths=5000)
    hw_put, put_se = hw.put_price_monte_carlo(n_paths=5000)
    print(f"Call Price: ${hw_call:.4f} ± ${call_se:.4f}")
    print(f"Put Price:  ${hw_put:.4f} ± ${put_se:.4f}")
    
    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)
    comparison = compare_stochastic_vol_models(S0, K, r, T, sigma)
    print(comparison)
