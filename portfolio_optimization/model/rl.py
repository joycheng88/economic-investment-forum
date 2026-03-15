"""
Reinforcement Learning (Actor-Critic) for Portfolio Allocation

Formulates portfolio optimization as a Markov Decision Process:
- State: Macro factors (volatility regime, market conditions)
- Action: Portfolio weights (continuous, sum to 1)
- Reward: Utility-adjusted returns minus transaction costs
- Objective: Maximize expected discounted cumulative utility

Methods:
- Actor-Critic with policy gradient optimization
- Experience replay with mini-batch updates
- State normalization and reward shaping
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Optional: PyTorch for neural networks (can implement without if not available)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ==================== STATE EXTRACTION ====================

class StateExtractor:
    """
    Extract state features from returns data.
    
    State components:
    1. Volatility regime (rolling volatility, VIX-like measure)
    2. Macro factors (momentum, mean reversion signals)
    3. Latent factors (PCA of returns covariance)
    """
    
    def __init__(self, lookback_vol: int = 20, lookback_macro: int = 60, n_pca: int = 3):
        """
        Parameters:
        -----------
        lookback_vol : int
            Window for volatility calculation (default 20 days)
        lookback_macro : int
            Window for macro factor calculation (default 60 days)
        n_pca : int
            Number of PCA components for latent factors (default 3)
        """
        self.lookback_vol = lookback_vol
        self.lookback_macro = lookback_macro
        self.n_pca = n_pca
        self.pca = None
        self.scaler = StandardScaler()
        self.state_dim = None
        
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit PCA and scaler on historical returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns (T x N)
        """
        # Fit PCA on covariance matrix eigenvectors
        cov = returns.cov()
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        self.pca_basis = eigenvectors[:, -self.n_pca:]  # Top n_pca eigenvectors
        
        # Fit scaler (will normalize states) - need to collect multiple samples for scaler
        # Extract features from multiple time windows
        min_samples = min(10, len(returns) // self.lookback_macro)
        sample_states = []
        for i in range(min_samples):
            start_idx = max(0, len(returns) - (i + 1) * self.lookback_macro)
            end_idx = len(returns) - i * self.lookback_macro
            if end_idx - start_idx >= self.lookback_macro:
                window_returns = returns.iloc[start_idx:end_idx]
                state = self._extract_features(window_returns)
                sample_states.append(state)
        
        # Fit scaler on collected samples (2D array: n_samples x n_features)
        sample_states = np.array(sample_states)
        self.scaler.fit(sample_states)
        
        # State dimension: volatility (1) + momentum (N) + mean reversion (N) + PCA (n_pca)
        self.state_dim = 1 + returns.shape[1] + returns.shape[1] + self.n_pca
    
    def get_state(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Extract state features from returns history.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Recent asset returns (requires at least lookback_macro rows)
        
        Returns:
        --------
        state : np.ndarray
            Normalized state vector
        """
        if len(returns) < self.lookback_macro:
            raise ValueError(f"Need at least {self.lookback_macro} rows, got {len(returns)}")
        
        features = self._extract_features(returns)
        state = self.scaler.transform(features.reshape(1, -1))[0]
        return state
    
    def _extract_features(self, returns: pd.DataFrame) -> np.ndarray:
        """Extract raw state features."""
        n_assets = returns.shape[1]
        
        # 1. Volatility regime (1 feature)
        recent_vol = returns.iloc[-self.lookback_vol:].std().mean()
        macro_vol = returns.iloc[-self.lookback_macro:].std().mean()
        vol_regime = recent_vol / (macro_vol + 1e-6)  # Ratio of recent to long-term vol
        
        # 2. Momentum signals (N features): rate of change of returns
        momentum = returns.iloc[-self.lookback_macro:].mean() / (returns.iloc[-self.lookback_macro:].std() + 1e-6)
        
        # 3. Mean reversion signals (N features): z-score of volatility
        recent_volatility = returns.iloc[-self.lookback_vol:].std()
        macro_volatility = returns.iloc[-self.lookback_macro:].std()
        mean_reversion = (recent_volatility - macro_volatility) / (macro_volatility + 1e-6)
        
        # 4. PCA latent factors (n_pca features)
        cov_recent = returns.iloc[-self.lookback_vol:].cov().values
        pca_factors = cov_recent @ self.pca_basis  # Project onto PCA basis
        pca_factors = np.linalg.norm(pca_factors, axis=0)  # Take norms of each PCA direction
        
        # Concatenate: [vol_regime, momentum, mean_reversion, pca_factors]
        features = np.concatenate([
            [vol_regime],
            momentum.values,
            mean_reversion.values,
            pca_factors
        ])
        
        return features


# ==================== NEURAL NETWORK POLICIES ====================

if TORCH_AVAILABLE:
    class ActorNetwork(nn.Module):
        """
        Actor network: maps state -> portfolio weights.
        
        Policy: π_θ(a|s) = softmax(linear(relu(linear(s))))
        Outputs: logits for categorical distribution over weights
        """
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
            super(ActorNetwork, self).__init__()
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)  # Logits for softmax
            
            # Initialize with small weights
            for layer in [self.fc1, self.fc2, self.fc3]:
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            """
            Parameters:
            -----------
            state : torch.Tensor
                State vector (batch_size, state_dim) or (state_dim,)
            
            Returns:
            --------
            logits : torch.Tensor
                Action logits (batch_size, action_dim) or (action_dim,)
            """
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            logits = self.fc3(x)  # Logits before softmax
            return logits
        
        def get_weights(self, state: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
            """
            Sample portfolio weights via softmax.
            
            Parameters:
            -----------
            state : torch.Tensor
                State vector
            temperature : float
                Softmax temperature (higher = more uniform, default 1.0)
            
            Returns:
            --------
            weights : torch.Tensor
                Portfolio weights (sum to 1)
            """
            logits = self.forward(state)
            weights = torch.softmax(logits / temperature, dim=-1)
            return weights
    
    class CriticNetwork(nn.Module):
        """
        Critic network: maps state -> value estimate.
        
        V_φ(s) = critic(s)
        Estimates expected discounted cumulative reward
        """
        
        def __init__(self, state_dim: int, hidden_dim: int = 64):
            super(CriticNetwork, self).__init__()
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)  # Single value output
            
            for layer in [self.fc1, self.fc2, self.fc3]:
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            """
            Parameters:
            -----------
            state : torch.Tensor
                State vector (batch_size, state_dim) or (state_dim,)
            
            Returns:
            --------
            value : torch.Tensor
                Value estimate (batch_size, 1) or (1,)
            """
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            value = self.fc3(x)
            return value


# ==================== RL AGENT ====================

class PortfolioRLAgent:
    """
    Actor-Critic RL Agent for portfolio allocation.
    
    Maintains:
    - Actor network π_θ(w|s): policy for generating portfolio weights
    - Critic network V_φ(s): value function for state evaluation
    - Experience buffer for mini-batch training
    - Optimizer with policy gradient updates
    """
    
    def __init__(
        self,
        n_assets: int,
        state_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        hidden_dim: int = 64,
        device: str = 'cpu'
    ):
        """
        Parameters:
        -----------
        n_assets : int
            Number of assets in portfolio
        state_dim : int
            Dimension of state vector
        learning_rate : float
            Learning rate for optimizer (default 0.001)
        gamma : float
            Discount factor (default 0.99)
        hidden_dim : int
            Hidden layer dimension (default 64)
        device : str
            'cpu' or 'cuda' (default 'cpu')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RL agent. Install with: pip install torch")
        
        self.n_assets = n_assets
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = torch.device(device)
        
        # Networks
        self.actor = ActorNetwork(state_dim, n_assets, hidden_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
    def choose_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Generate portfolio weights for given state.
        
        Parameters:
        -----------
        state : np.ndarray
            State vector (state_dim,)
        deterministic : bool
            If True, return mode of policy (greedy)
            If False, sample from policy (stochastic)
        temperature : float
            Softmax temperature (default 1.0)
        
        Returns:
        --------
        weights : np.ndarray
            Portfolio weights (n_assets,), sum to 1
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                logits = self.actor(state_tensor)
                weights = torch.softmax(logits, dim=-1)
            else:
                weights = self.actor.get_weights(state_tensor, temperature)
        
        return weights[0].cpu().numpy()
    
    def store_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store transition in experience buffer.
        
        Parameters:
        -----------
        state : np.ndarray
            Current state
        action : np.ndarray
            Action taken (portfolio weights)
        reward : float
            Reward received
        next_state : np.ndarray
            Next state
        done : bool
            Whether episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Perform one training step on a mini-batch.
        
        Parameters:
        -----------
        batch_size : int
            Mini-batch size (default 32)
        
        Returns:
        --------
        metrics : dict
            Actor loss, Critic loss, entropy bonus
        """
        if len(self.states) < batch_size:
            return {}
        
        # Random batch
        indices = np.random.choice(len(self.states), size=batch_size, replace=False)
        
        states = torch.FloatTensor(np.array(self.states)[indices]).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)[indices]).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)[indices]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)[indices]).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)[indices]).unsqueeze(1).to(self.device)
        
        # Critic update: minimize TD error
        with torch.no_grad():
            next_values = self.critic(next_states)
            target_values = rewards + self.gamma * next_values * (1 - dones)
        
        current_values = self.critic(states)
        critic_loss = nn.MSELoss()(current_values, target_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Actor update: policy gradient with advantage
        advantages = (target_values - current_values).detach()
        
        logits = self.actor(states)
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Action indices for gathering log probs
        action_indices = torch.argmax(actions, dim=-1) if actions.sum(dim=-1).max() > 0 else torch.zeros(batch_size, dtype=torch.long)
        # Better: use actions as probabilities over weights
        action_log_probs = (log_probs * actions).sum(dim=-1, keepdim=True)
        
        # Entropy bonus (encourage exploration)
        entropy = -(torch.softmax(logits, dim=-1) * log_probs).sum(dim=-1, keepdim=True)
        entropy_weight = 0.01
        
        actor_loss = -(action_log_probs * advantages + entropy_weight * entropy).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.mean().item()
        }
    
    def clear_buffer(self) -> None:
        """Clear experience buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []


# ==================== REWARD FUNCTION ====================

def compute_reward(
    portfolio_return: float,
    portfolio_vol: float,
    prev_weights: np.ndarray,
    curr_weights: np.ndarray,
    transaction_cost_rate: float = 0.001,
    risk_aversion: float = 1.0,
    utility_type: str = 'mean_variance'
) -> float:
    """
    Compute reward: utility-adjusted return minus transaction costs.
    
    Parameters:
    -----------
    portfolio_return : float
        Return of portfolio over period
    portfolio_vol : float
        Realized volatility of portfolio
    prev_weights : np.ndarray
        Previous portfolio weights
    curr_weights : np.ndarray
        Current portfolio weights
    transaction_cost_rate : float
        One-way transaction cost (default 0.001 = 10 bps)
    risk_aversion : float
        Risk aversion coefficient (default 1.0)
    utility_type : str
        'mean_variance', 'log', or 'cvar' (default 'mean_variance')
    
    Returns:
    --------
    reward : float
        Utility-adjusted reward
    """
    # Transaction cost: turnover * cost_rate
    turnover = np.sum(np.abs(curr_weights - prev_weights)) / 2  # One-way
    transaction_cost = turnover * transaction_cost_rate
    
    # Utility function
    if utility_type == 'mean_variance':
        # U(R) = R - (λ/2) * σ²
        utility = portfolio_return - (risk_aversion / 2) * (portfolio_vol ** 2)
    elif utility_type == 'log':
        # Log utility (Kelly criterion): U(R) = log(1 + R)
        utility = np.log(1 + max(portfolio_return, -0.99))
    elif utility_type == 'cvar':
        # CVaR-based: penalize downside volatility
        downside_vol = portfolio_vol if portfolio_return < 0 else portfolio_vol / 2
        utility = portfolio_return - risk_aversion * downside_vol
    else:
        raise ValueError(f"Unknown utility_type: {utility_type}")
    
    # Reward = utility - transaction costs
    reward = utility - transaction_cost
    
    return reward


# ==================== MAIN INTERFACE ====================

def train_rl_agent(
    returns: pd.DataFrame,
    n_epochs: int = 10,
    episode_length: int = 20,
    learning_rate: float = 0.001,
    gamma: float = 0.99,
    transaction_cost_rate: float = 0.001,
    risk_aversion: float = 1.0,
    batch_size: int = 32,
    verbose: bool = True
) -> Tuple[PortfolioRLAgent, StateExtractor]:
    """
    Train RL agent on historical returns.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical asset returns (T x N)
    n_epochs : int
        Number of training epochs (default 10)
    episode_length : int
        Length of episode (rebalancing periods, default 20)
    learning_rate : float
        Learning rate for optimizer (default 0.001)
    gamma : float
        Discount factor (default 0.99)
    transaction_cost_rate : float
        One-way transaction cost (default 0.001)
    risk_aversion : float
        Risk aversion (default 1.0)
    batch_size : int
        Mini-batch size (default 32)
    verbose : bool
        Print training progress (default True)
    
    Returns:
    --------
    agent : PortfolioRLAgent
        Trained RL agent
    state_extractor : StateExtractor
        Fitted state extractor
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. Install with: pip install torch")
    
    n_assets = returns.shape[1]
    
    # Initialize state extractor
    state_extractor = StateExtractor(lookback_vol=20, lookback_macro=60, n_pca=3)
    state_extractor.fit(returns)
    state_dim = state_extractor.state_dim
    
    # Initialize agent
    agent = PortfolioRLAgent(
        n_assets=n_assets,
        state_dim=state_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        hidden_dim=64
    )
    
    if verbose:
        print(f"Training RL Agent")
        print(f"  Assets: {n_assets}")
        print(f"  State dim: {state_dim}")
        print(f"  Epochs: {n_epochs}, Episode length: {episode_length}")
    
    # Training loop
    for epoch in range(n_epochs):
        epoch_loss = {'actor': 0, 'critic': 0, 'entropy': 0}
        n_updates = 0
        
        # Generate episodes: random sliding windows through history
        for episode in range(len(returns) // episode_length):
            start_idx = np.random.randint(100, len(returns) - episode_length)
            episode_returns = returns.iloc[start_idx:start_idx + episode_length]
            
            # Initialize portfolio (equal weight)
            weights = np.ones(n_assets) / n_assets
            
            for t in range(episode_length - 1):
                # Get state
                hist_returns = episode_returns.iloc[:t + 1]
                if len(hist_returns) < 60:
                    continue
                
                state = state_extractor.get_state(hist_returns)
                
                # Choose action (portfolio weights)
                action = agent.choose_action(state, deterministic=False, temperature=1.0)
                
                # Compute reward
                period_return = episode_returns.iloc[t + 1].values
                portfolio_return = action @ period_return
                portfolio_vol = 0.05  # Estimated vol (simplified)
                
                reward = compute_reward(
                    portfolio_return=portfolio_return,
                    portfolio_vol=portfolio_vol,
                    prev_weights=weights,
                    curr_weights=action,
                    transaction_cost_rate=transaction_cost_rate,
                    risk_aversion=risk_aversion,
                    utility_type='mean_variance'
                )
                
                # Next state
                next_hist = episode_returns.iloc[:t + 2]
                if len(next_hist) >= 60:
                    next_state = state_extractor.get_state(next_hist)
                    done = (t == episode_length - 2)
                    
                    # Store experience
                    agent.store_experience(state, action, reward, next_state, done)
                
                weights = action
            
            # Mini-batch update
            metrics = agent.train_step(batch_size)
            if metrics:
                for key in metrics:
                    epoch_loss[key] += metrics[key]
                n_updates += 1
        
        if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
            avg_actor = epoch_loss['actor'] / max(n_updates, 1)
            avg_critic = epoch_loss['critic'] / max(n_updates, 1)
            print(f"  Epoch {epoch + 1}/{n_epochs}: actor_loss={avg_actor:.4f}, critic_loss={avg_critic:.4f}")
        
        agent.clear_buffer()
    
    return agent, state_extractor


def get_rl_weights(
    returns: pd.DataFrame,
    agent: Optional[PortfolioRLAgent] = None,
    state_extractor: Optional[StateExtractor] = None,
    n_epochs: int = 10,
    max_weight: float = 0.15,
    long_only: bool = True
) -> pd.Series:
    """
    Get optimal portfolio weights from trained RL agent.
    
    If agent not provided, trains one on historical returns.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical asset returns (will be used for training if agent=None)
    agent : PortfolioRLAgent, optional
        Pre-trained agent (if None, trains new agent)
    state_extractor : StateExtractor, optional
        Pre-fitted state extractor (if None, creates new one)
    n_epochs : int
        Training epochs if agent=None (default 10)
    max_weight : float
        Maximum single-asset weight (default 0.15)
    long_only : bool
        Enforce non-negative weights (default True)
    
    Returns:
    --------
    weights : pd.Series
        Portfolio weights
    """
    if agent is None or state_extractor is None:
        print("Training new RL agent (this may take 30-60 seconds)...")
        agent, state_extractor = train_rl_agent(
            returns,
            n_epochs=n_epochs,
            episode_length=20,
            learning_rate=0.001,
            gamma=0.99,
            transaction_cost_rate=0.001,
            risk_aversion=1.0,
            batch_size=32,
            verbose=False
        )
    
    # Get current state from full history
    state = state_extractor.get_state(returns)
    
    # Generate deterministic weights (greedy policy)
    weights = agent.choose_action(state, deterministic=True, temperature=0.5)
    
    # Apply constraints
    weights = np.maximum(weights, 0) if long_only else weights
    weights = np.minimum(weights, max_weight)
    weights = weights / weights.sum()  # Renormalize
    
    return pd.Series(weights, index=returns.columns)


# ==================== DIAGNOSTICS ====================

def get_agent_state_interpretation(
    state: np.ndarray,
    state_extractor: StateExtractor,
    n_assets: int
) -> Dict:
    """
    Interpret state vector components.
    
    Parameters:
    -----------
    state : np.ndarray
        Normalized state vector
    state_extractor : StateExtractor
        Fitted state extractor
    n_assets : int
        Number of assets
    
    Returns:
    --------
    interpretation : dict
        Breakdown of state components
    """
    interpretation = {
        'volatility_regime': state[0],
        'momentum_signals': state[1:n_assets+1],
        'mean_reversion_signals': state[n_assets+1:2*n_assets+1],
        'pca_factors': state[2*n_assets+1:]
    }
    return interpretation
