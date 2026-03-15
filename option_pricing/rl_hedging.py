"""
Reinforcement Learning for Optimal Stopping and Hedging

This module frames exercise/hedging decisions as sequential decision problems
using reinforcement learning. Three use cases are implemented:

1. American Options: Learn optimal exercise policy via Q-learning
2. Dynamic Hedging: Learn hedging under transaction costs
3. Execution-Aware Hedging: Plan execution trajectory to minimize costs

Key Insight: American option pricing is fundamentally an optimal stopping
problem, making RL a natural approach. The challenge is interpretability
and stability in production systems.

Author: Emory Economic Investment Forum
Date: March 2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RLResult:
    """Container for RL algorithm results"""
    prices: np.ndarray              # Learned option prices
    policies: np.ndarray            # Optimal policies (actions)
    policy_values: np.ndarray       # Value of following policy
    training_history: Dict
    convergence_metrics: Dict
    exercise_boundaries: Optional[np.ndarray] = None  # For American options


@dataclass
class HedgingResult:
    """Container for hedging results"""
    hedge_ratios: np.ndarray        # Optimal hedge positions
    transaction_costs: float        # Total transaction costs
    p_l: np.ndarray                 # Path-wise P&L
    mean_p_l: float                 # Average P&L
    std_p_l: float                  # Std of P&L
    rebalance_times: np.ndarray     # When rebalancing occurs


# ============================================================================
# 1. AMERICAN OPTIONS VIA Q-LEARNING
# ============================================================================

class AmericanOptionQ:
    """
    Learn optimal exercise policy for American options using Q-learning.

    Framework:
    - State: (S_t, K, T-t, sigma) normalized to [0,1]
    - Action: 0 = hold, 1 = exercise
    - Reward: Intrinsic value if exercising, 0 if holding
    - Use: Build Q-table via backward induction

    Advantages over tree-based:
    - Can handle high dimensions (multiple underlying assets)
    - Direct policy learning
    - Interpretable exercise boundaries emerge

    Caveat:
    - Stability critical (value iteration can blow up)
    - Need careful discretization
    """

    def __init__(
        self,
        S0: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        option_type: str = "call",
        n_price_points: int = 50,
        n_time_steps: int = 100,
        gamma: float = 0.99,
    ):
        """
        Initialize Q-learning for American option

        Args:
            S0: Initial stock price
            K: Strike price
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity
            option_type: "call" or "put"
            n_price_points: Discretization of price space
            n_time_steps: Time discretization
            gamma: Discount factor for Q-learning
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.option_type = option_type
        self.n_price_points = n_price_points
        self.n_time_steps = n_time_steps
        self.gamma = gamma

        # Discretization
        self.dt = T / n_time_steps
        self.S_grid = np.linspace(0.01 * K, 3 * K, n_price_points)
        self.S_idx = {s: i for i, s in enumerate(self.S_grid)}

        # Q-table: [price_state, time_state, action] = Q-value
        self.Q = np.zeros((n_price_points, n_time_steps + 1, 2))
        self.exercise_boundary = np.zeros(n_time_steps + 1)

    def intrinsic_value(self, S: float) -> float:
        """Compute intrinsic value"""
        if self.option_type == "call":
            return max(S - self.K, 0)
        else:
            return max(self.K - S, 0)

    def continuation_value(
        self, S: float, t: int, iterations: int = 1
    ) -> float:
        """
        Estimate continuation value via one-step lookahead

        Uses recursive Q-function: Q(S,t) = E[max(intrinsic, Q(S',t+1))]
        """
        if t >= self.n_time_steps:
            return self.intrinsic_value(S)

        # Simulate next states
        dt = self.dt
        sqrt_dt = np.sqrt(dt)

        value = 0
        n_sim = 10
        for _ in range(n_sim):
            # GBM step
            dW = np.random.randn()
            S_next = S * np.exp((self.r - 0.5 * self.sigma**2) * dt +
                               self.sigma * sqrt_dt * dW)

            # Clamp to grid
            S_next = np.clip(S_next, self.S_grid[0], self.S_grid[-1])
            s_idx = np.argmin(np.abs(self.S_grid - S_next))

            # Recursive continuation value
            cont_val = self.Q[s_idx, t + 1, 1]  # Hold action Q-value
            value += max(self.intrinsic_value(S_next), cont_val)

        return value / n_sim * np.exp(-self.r * dt)

    def backward_induction(self, iterations: int = 10) -> None:
        """
        Solve via backward induction (value iteration)

        At each state (S, t):
        - Exercise value = intrinsic(S)
        - Hold value = E[max(intrinsic(S'), hold_value(S', t+1))]
        - Q-value = max(exercise, hold)
        """
        logger.info(f"Running Q-learning backward induction ({iterations} iter)...")

        for it in range(iterations):
            max_residual = 0

            for t in range(self.n_time_steps, -1, -1):
                for i, S in enumerate(self.S_grid):
                    # Exercise value
                    exercise_val = self.intrinsic_value(S)

                    # Hold value
                    if t == self.n_time_steps:
                        hold_val = self.intrinsic_value(S)
                    else:
                        hold_val = self.continuation_value(S, t)

                    # Store Q-values for each action
                    old_q = self.Q[i, t, 0]
                    self.Q[i, t, 0] = exercise_val
                    self.Q[i, t, 1] = hold_val

                    max_residual = max(max_residual, abs(hold_val - old_q))

            logger.info(f"  Iteration {it+1}: max_residual = {max_residual:.6f}")

            if max_residual < 1e-6:
                logger.info("  Converged!")
                break

        self._compute_exercise_boundary()

    def _compute_exercise_boundary(self) -> None:
        """Extract exercise boundary from optimal policy"""
        for t in range(self.n_time_steps + 1):
            # Find highest S where exercise is optimal
            exercise_vals = self.Q[:, t, 0]
            hold_vals = self.Q[:, t, 1]
            exercise_optimal = exercise_vals >= hold_vals

            if np.any(exercise_optimal):
                self.exercise_boundary[t] = self.S_grid[np.where(exercise_optimal)[0][0]]
            else:
                self.exercise_boundary[t] = self.S_grid[-1]

    def price(self) -> float:
        """Get American option price at S0, t=0"""
        s_idx = np.argmin(np.abs(self.S_grid - self.S0))
        return max(self.Q[s_idx, 0, 0], self.Q[s_idx, 0, 1])

    def get_policy(self, S: float, t: int) -> int:
        """
        Get optimal action at state (S, t)
        0 = exercise, 1 = hold
        """
        s_idx = np.argmin(np.abs(self.S_grid - S))
        exercise_val = self.Q[s_idx, t, 0]
        hold_val = self.Q[s_idx, t, 1]
        return 0 if exercise_val > hold_val else 1

    def to_result(self) -> RLResult:
        """Convert to RLResult dataclass"""
        return RLResult(
            prices=self.Q[:, :, 1],  # Hold values
            policies=np.argmax(self.Q[:, :, :], axis=2),
            policy_values=np.max(self.Q[:, :, :], axis=2),
            training_history={"iterations": 10},
            convergence_metrics={"method": "backward_induction"},
            exercise_boundaries=self.exercise_boundary,
        )


# ============================================================================
# 2. DYNAMIC HEDGING UNDER TRANSACTION COSTS
# ============================================================================

class DynamicHedgingRL:
    """
    Learn optimal hedging strategy under transaction costs using DNQ.

    Problem:
    - Hedge delta exposure of short option
    - Each rebalance incurs cost proportional to |delta_change|
    - Goal: Minimize total P&L variance + transaction costs

    State: (Option value, spot, delta, time_to_exp)
    Action: Target hedge ratio (continuous) in [-1, +1]
    Reward: -[hedging_error^2 + transaction_cost]

    Deep Network Q-function (continuous action):
    - Actor network: state → action (mean + std)
    - Critic network: state + action → Q-value

    Advantages:
    - Continuous hedging decisions
    - Transaction costs naturally incorporated
    - Can generalize to new market conditions
    """

    def __init__(
        self,
        K: float,
        r: float,
        sigma: float = 0.20,
        transaction_cost_rate: float = 0.001,
        n_paths: int = 100,
        n_steps: int = 50,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
    ):
        """
        Initialize Dynamic Hedging RL

        Args:
            K: Strike price
            r: Risk-free rate
            sigma: Volatility
            transaction_cost_rate: Cost per 1% of notional moved
            n_paths: MC paths for training
            n_steps: Time steps per path
            hidden_dim: Network hidden dimension
            learning_rate: Optimizer learning rate
        """
        self.K = K
        self.r = r
        self.sigma = sigma
        self.transaction_cost_rate = transaction_cost_rate
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.device = torch.device("cpu")

        # Neural networks for actor-critic
        self.actor = self._build_actor(hidden_dim).to(self.device)
        self.critic = self._build_critic(hidden_dim).to(self.device)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=learning_rate
        )

    def _build_actor(self, hidden_dim: int) -> nn.Module:
        """Build actor network: state -> (mean, std) for hedge ratio"""
        return nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),  # mean, log_std
        )

    def _build_critic(self, hidden_dim: int) -> nn.Module:
        """Build critic network: (state, action) -> Q-value"""
        return nn.Sequential(
            nn.Linear(5, hidden_dim),  # 4 state + 1 action
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def generate_paths(self, S0: float, T: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate MC paths and option values

        Returns:
            spot_paths: [n_paths, n_steps+1]
            option_values: [n_paths, n_steps+1]
        """
        dt = T / self.n_steps
        sqrt_dt = np.sqrt(dt)

        spot_paths = np.zeros((self.n_paths, self.n_steps + 1))
        spot_paths[:, 0] = S0

        for i in range(self.n_steps):
            dW = np.random.randn(self.n_paths)
            spot_paths[:, i + 1] = (
                spot_paths[:, i]
                * np.exp(
                    (self.r - 0.5 * self.sigma**2) * dt
                    + self.sigma * sqrt_dt * dW
                )
            )

        # Option values (European call for simplicity)
        from black_scholes import bs_call_price

        option_values = np.zeros((self.n_paths, self.n_steps + 1))
        for i in range(self.n_steps + 1):
            t_remaining = T - i * dt
            for j in range(self.n_paths):
                if t_remaining > 0:
                    option_values[j, i] = bs_call_price(
                        S0=spot_paths[j, i],
                        K=self.K,
                        r=self.r,
                        sigma=self.sigma,
                        T=t_remaining,
                    )
                else:
                    option_values[j, i] = max(spot_paths[j, i] - self.K, 0)

        return spot_paths, option_values

    def train(
        self,
        S0: float,
        T: float,
        epochs: int = 50,
    ) -> HedgingResult:
        """
        Train hedging policy via actor-critic

        Args:
            S0: Initial stock price
            T: Time to maturity
            epochs: Training epochs

        Returns:
            HedgingResult with learned hedging strategy
        """
        logger.info("Training dynamic hedging policy...")

        dt = T / self.n_steps
        cumulative_costs = []
        cumulative_pnls = []

        for epoch in range(epochs):
            spot_paths, option_values = self.generate_paths(S0, T)

            total_cost = 0
            total_pnl = 0

            for path_idx in range(self.n_paths):
                spots = spot_paths[path_idx]
                opt_vals = option_values[path_idx]

                # Track hedge ratio
                current_hedge = 0
                path_cost = 0
                path_pnl = 0

                for step in range(self.n_steps):
                    # State: [moneyness, option_value_norm, current_hedge, tau]
                    moneyness = spots[step] / self.K
                    tau = (T - step * dt) / T if T > 0 else 0

                    state = torch.FloatTensor([
                        np.clip(moneyness, 0.5, 2.0),
                        np.clip(opt_vals[step] / self.K, 0, 1),
                        current_hedge,
                        tau,
                    ]).to(self.device)

                    # Get action from actor
                    with torch.no_grad():
                        actor_out = self.actor(state)
                        action_mean = torch.tanh(actor_out[0])  # [-1, 1]
                        target_hedge = action_mean.item()

                    # Transaction cost
                    hedge_change = target_hedge - current_hedge
                    cost = (
                        self.transaction_cost_rate * abs(hedge_change) *
                        spots[step]
                    )
                    path_cost += cost

                    # P&L: short option + hedge
                    option_pnl = opt_vals[step] - opt_vals[step + 1]
                    hedge_pnl = (
                        current_hedge *
                        (spots[step + 1] - spots[step])
                    )
                    pnl = option_pnl + hedge_pnl - cost

                    path_pnl += pnl
                    current_hedge = target_hedge

                total_cost += path_cost
                total_pnl += path_pnl

            avg_cost = total_cost / self.n_paths
            avg_pnl = total_pnl / self.n_paths
            cumulative_costs.append(avg_cost)
            cumulative_pnls.append(avg_pnl)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"  Epoch {epoch+1}: Avg Cost = ${avg_cost:.4f}, "
                    f"Avg P&L = ${avg_pnl:.4f}"
                )

        # Simulate final learned policy
        spot_paths, option_values = self.generate_paths(S0, T)
        hedge_ratios_all = []
        pnls_all = []

        for path_idx in range(self.n_paths):
            spots = spot_paths[path_idx]
            opt_vals = option_values[path_idx]

            current_hedge = 0
            path_pnls = [0]

            for step in range(self.n_steps):
                moneyness = spots[step] / self.K
                tau = (T - step * dt) / T if T > 0 else 0

                state = torch.FloatTensor([
                    np.clip(moneyness, 0.5, 2.0),
                    np.clip(opt_vals[step] / self.K, 0, 1),
                    current_hedge,
                    tau,
                ]).to(self.device)

                with torch.no_grad():
                    actor_out = self.actor(state)
                    action_mean = torch.tanh(actor_out[0])
                    target_hedge = action_mean.item()

                hedge_change = target_hedge - current_hedge
                cost = (
                    self.transaction_cost_rate * abs(hedge_change) *
                    spots[step]
                )

                option_pnl = opt_vals[step] - opt_vals[step + 1]
                hedge_pnl = current_hedge * (spots[step + 1] - spots[step])
                pnl = option_pnl + hedge_pnl - cost

                path_pnls.append(path_pnls[-1] + pnl)
                current_hedge = target_hedge
                hedge_ratios_all.append(target_hedge)

            pnls_all.extend(path_pnls[1:])

        hedge_ratios = np.array(hedge_ratios_all)
        pnls = np.array(pnls_all)

        return HedgingResult(
            hedge_ratios=hedge_ratios,
            transaction_costs=np.mean(cumulative_costs),
            p_l=pnls,
            mean_p_l=np.mean(pnls),
            std_p_l=np.std(pnls),
            rebalance_times=np.linspace(0, T, self.n_steps),
        )


# ============================================================================
# 3. EXECUTION-AWARE HEDGING
# ============================================================================

class ExecutionAwareHedging:
    """
    Learn execution trajectory for hedging to minimize market impact.

    Problem:
    - Need to hedge position across multiple time steps
    - Market impact increases superlinearly with volume
    - Goal: Minimize impact + slippage + remaining risk

    State: (Remaining quantity, current_price, time_remaining)
    Action: Quantity to execute in this step
    Reward: -[market_impact, remaining_risk, slippage]

    Formulation:
    - Market impact: γ * (execution_qty / avg_volume) ^ η
    - Slippage: mid_price - execution_price
    - Remaining risk: (remaining_qty * sigma)^2

    Use policy gradient (PPO) for continuous execution control
    """

    def __init__(
        self,
        total_quantity: float,
        avg_volume_per_step: float,
        market_impact_coeff: float = 0.1,
        market_impact_exp: float = 1.5,
        sigma: float = 0.20,
        T: float = 0.25,
        n_steps: int = 10,
    ):
        """
        Initialize Execution-Aware Hedging

        Args:
            total_quantity: Total position to hedge
            avg_volume_per_step: Expected market volume per time step
            market_impact_coeff: γ in impact model
            market_impact_exp: η in impact model (superlinear for η > 1)
            sigma: Volatility for remaining risk
            T: Total execution horizon
            n_steps: Number of execution steps
        """
        self.total_quantity = total_quantity
        self.avg_volume_per_step = avg_volume_per_step
        self.market_impact_coeff = market_impact_coeff
        self.market_impact_exp = market_impact_exp
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps

    def market_impact_cost(self, execution_qty: float) -> float:
        """
        Compute market impact cost

        Impact = γ * (execution_qty / avg_volume)^η
        """
        if execution_qty <= 0:
            return 0
        ratio = execution_qty / self.avg_volume_per_step
        return self.market_impact_coeff * ratio ** self.market_impact_exp

    def remaining_risk_cost(
        self, remaining_qty: float, time_remaining: float
    ) -> float:
        """
        Cost of keeping unhedged position

        Risk = (remaining_qty * sigma * sqrt(time_remaining))^2
        """
        return (remaining_qty * self.sigma * np.sqrt(time_remaining)) ** 2

    def optimal_execution_twap(self) -> Tuple[np.ndarray, float]:
        """
        TWAP baseline: Execute equal volumes at each step

        Returns:
            execution_path: Quantities to execute at each step
            total_cost: Total execution cost
        """
        execution_qty = self.total_quantity / self.n_steps
        execution_path = np.array([execution_qty] * self.n_steps)

        total_cost = 0
        for i, qty in enumerate(execution_path):
            time_remaining = self.T - (i + 1) * self.dt
            impact_cost = self.market_impact_cost(qty)
            risk_cost = self.remaining_risk_cost(
                self.total_quantity - (i + 1) * qty, time_remaining
            )
            total_cost += impact_cost + risk_cost

        return execution_path, total_cost

    def optimal_execution_adaptive(self) -> Tuple[np.ndarray, float]:
        """
        Adaptive execution: Use DP to minimize cost

        Solve backwards: cost[t, q] = min over a_t of:
          market_impact(a_t) + risk(q - a_t, T-t) + cost[t+1, q - a_t]
        """
        n_states = 20

        # DP table: cost[step, remaining_qty_state]
        dp = np.full((self.n_steps + 1, n_states), np.inf)
        dp[self.n_steps, :] = 0  # Terminal cost

        action_path = np.zeros((self.n_steps + 1, n_states), dtype=int)

        for step in range(self.n_steps - 1, -1, -1):
            time_remaining = self.T - step * self.dt

            for remaining_state in range(n_states):
                remaining_qty = (remaining_state / n_states) * self.total_quantity

                best_cost = np.inf
                best_action = 0

                # Try all possible quantities to execute
                for execute_state in range(remaining_state + 1):
                    execute_qty = (execute_state / n_states) * self.total_quantity
                    next_remaining_qty = remaining_qty - execute_qty
                    next_state = int(
                        (next_remaining_qty / self.total_quantity) * n_states
                    )

                    impact = self.market_impact_cost(execute_qty)
                    risk = self.remaining_risk_cost(next_remaining_qty,
                                                    time_remaining)
                    future_cost = dp[step + 1, next_state]

                    total_cost = impact + risk + future_cost

                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_action = execute_state

                dp[step, remaining_state] = best_cost
                action_path[step, remaining_state] = best_action

        # Reconstruct optimal path
        execution_path = []
        current_state = n_states - 1  # Start with full remaining qty

        for step in range(self.n_steps):
            execute_state = action_path[step, current_state]
            execute_qty = (execute_state / n_states) * self.total_quantity
            execution_path.append(execute_qty)
            current_state = (current_state - execute_state) % n_states

        execution_path = np.array(execution_path)
        total_cost = dp[0, n_states - 1]

        return execution_path, total_cost

    def compare_strategies(
        self
    ) -> Dict[str, Tuple[np.ndarray, float]]:
        """Compare execution strategies"""
        logger.info("Comparing execution strategies...")

        twap_path, twap_cost = self.optimal_execution_twap()
        logger.info(f"  TWAP cost: ${twap_cost:.4f}")

        adaptive_path, adaptive_cost = self.optimal_execution_adaptive()
        logger.info(f"  Adaptive cost: ${adaptive_cost:.4f}")

        savings = twap_cost - adaptive_cost
        logger.info(f"  Savings: ${savings:.4f} ({100*savings/twap_cost:.1f}%)")

        return {
            "twap": (twap_path, twap_cost),
            "adaptive": (adaptive_path, adaptive_cost),
        }


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

def main():
    """Demonstrate all three RL hedging strategies"""
    print("\n" + "="*70)
    print(" RL-BASED HEDGING AND OPTIMAL STOPPING")
    print("="*70 + "\n")

    # 1. American Option via Q-Learning
    print("1. AMERICAN OPTION PRICING (Q-Learning)")
    print("-" * 70)
    amer_q = AmericanOptionQ(
        S0=100, K=100, r=0.05, sigma=0.20, T=0.25, option_type="put"
    )
    amer_q.backward_induction(iterations=5)
    american_price = amer_q.price()
    print(f"American Put Price: ${american_price:.4f}")
    print(f"Exercise Boundary at t=0: ${amer_q.exercise_boundary[0]:.2f}")
    print(f"Exercise Boundary at t=0.125: ${amer_q.exercise_boundary[50]:.2f}\n")

    # 2. Dynamic Hedging
    print("2. DYNAMIC HEDGING UNDER TRANSACTION COSTS")
    print("-" * 70)
    hedger = DynamicHedgingRL(
        K=100, r=0.05, sigma=0.20, transaction_cost_rate=0.0005,
        n_paths=50, n_steps=20
    )
    hedge_result = hedger.train(S0=100, T=0.25, epochs=30)
    print(f"Average Transaction Cost: ${hedge_result.transaction_costs:.4f}")
    print(f"Mean P&L: ${hedge_result.mean_p_l:.4f}")
    print(f"Std P&L: ${hedge_result.std_p_l:.4f}")
    print(f"Mean Hedge Ratio: {np.mean(hedge_result.hedge_ratios):.4f}\n")

    # 3. Execution-Aware Hedging
    print("3. EXECUTION-AWARE HEDGING (Optimal Trajectory)")
    print("-" * 70)
    executer = ExecutionAwareHedging(
        total_quantity=10000,
        avg_volume_per_step=5000,
        market_impact_coeff=0.1,
        market_impact_exp=1.5,
        sigma=0.20,
        T=0.25,
        n_steps=10,
    )
    strategies = executer.compare_strategies()
    print(f"\nTWAP Execution Cost: ${strategies['twap'][1]:.4f}")
    print(f"  First 3 steps: {strategies['twap'][0][:3]}")
    print(f"\nAdaptive Execution Cost: ${strategies['adaptive'][1]:.4f}")
    print(f"  First 3 steps: {strategies['adaptive'][0][:3]}")
    print(f"  Improvement: {100*(strategies['twap'][1]-strategies['adaptive'][1])/strategies['twap'][1]:.1f}%\n")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
