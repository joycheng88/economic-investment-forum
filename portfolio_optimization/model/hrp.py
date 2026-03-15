"""
Hierarchical Risk Parity (HRP) Portfolio

HRP is a risk-based allocation method that:
1. Clusters assets using hierarchical clustering (based on correlation distance)
2. Builds a tree structure to represent the clustering hierarchy
3. Recursively allocates risks top-down (risk parity within each cluster)

Advantages:
- Does not require inversion of large covariance matrices (numerically stable)
- More robust to estimation errors in expected returns
- Captures asset relationships through clustering structure
- No extreme concentration in few assets

Paper: Lopéz de Prado (2016), "Building Diversified Portfolios that Outperform"
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


@dataclass
class HRPConfig:
    """Configuration for Hierarchical Risk Parity."""
    linkage_method: str = "ward"  # "single", "complete", "average", "ward"
    long_only: bool = True
    max_weight: Optional[float] = None


def estimate_covariance(returns: pd.DataFrame, annualize: bool = True, periods_per_year: int = 252) -> pd.DataFrame:
    """
    Estimate covariance matrix using Ledoit-Wolf shrinkage.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        T x N asset returns
    annualize : bool
        If True, annualize the covariance
    periods_per_year : int
        Number of periods per year (default 252 for daily data)
        
    Returns:
    --------
    pd.DataFrame
        N x N covariance matrix
    """
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError:
        raise ImportError("Install scikit-learn: pip install scikit-learn")
    
    lw = LedoitWolf().fit(returns.values)
    cov = pd.DataFrame(
        lw.covariance_,
        index=returns.columns,
        columns=returns.columns
    )
    
    if annualize:
        cov = cov * periods_per_year
    
    return cov


def correlation_distance(corr: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Convert correlation matrix to distance matrix for clustering.
    
    Distance = sqrt(0.5 * (1 - correlation))
    This distance metric has nice properties for clustering.
    
    Parameters:
    -----------
    corr : pd.DataFrame
        Correlation matrix
        
    Returns:
    --------
    dist : np.ndarray
        Condensed distance matrix (upper triangle)
    tickers : list
        Asset tickers in the order of the distance matrix
    """
    tickers = corr.index.tolist()
    
    # Distance = sqrt(0.5 * (1 - correlation))
    dist_matrix = np.sqrt(0.5 * (1.0 - corr.values))
    
    # Ensure symmetry
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    
    # Set diagonal to 0
    np.fill_diagonal(dist_matrix, 0.0)
    
    # Convert to condensed distance matrix for linkage
    dist_condensed = squareform(dist_matrix)
    
    return dist_condensed, tickers


def hierarchical_clustering(
    returns: pd.DataFrame,
    cov: Optional[pd.DataFrame] = None,
    config: HRPConfig = HRPConfig()
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Perform hierarchical clustering on assets.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        T x N asset returns
    cov : pd.DataFrame, optional
        Covariance matrix (if None, estimated from returns)
    config : HRPConfig
        Configuration
        
    Returns:
    --------
    linkage_matrix : np.ndarray
        Linkage matrix from hierarchical clustering
    tickers : list
        Asset tickers
    corr : pd.DataFrame
        Correlation matrix
    """
    if cov is None:
        cov = estimate_covariance(returns, annualize=False)
    
    tickers = cov.index.tolist()
    
    # Compute correlation from covariance
    std_devs = np.sqrt(np.diag(cov.values))
    corr_matrix = cov.values / np.outer(std_devs, std_devs)
    corr = pd.DataFrame(corr_matrix, index=tickers, columns=tickers)
    
    # Convert to distance
    dist, tickers_ordered = correlation_distance(corr)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(dist, method=config.linkage_method)
    
    return linkage_matrix, tickers_ordered, corr


def tree_clustering(linkage_matrix: np.ndarray, tickers: List[str]) -> Dict[int, Any]:
    """
    Build a tree structure from linkage matrix for recursive risk allocation.
    
    Parameters:
    -----------
    linkage_matrix : np.ndarray
        Output from scipy.cluster.hierarchy.linkage
    tickers : list
        Asset tickers (in same order as used in clustering)
        
    Returns:
    --------
    tree : dict
        Tree structure representing the clustering hierarchy
    """
    n_assets = len(tickers)
    
    # Create node dictionary: nodes 0 to n_assets-1 are leaves (assets)
    nodes = {}
    for i, ticker in enumerate(tickers):
        nodes[i] = {"ticker": ticker, "children": [], "cluster": [ticker]}
    
    # Process linkage matrix to build tree
    # linkage_matrix[i] = [asset1_idx, asset2_idx, distance, n_samples_in_cluster]
    for i, (idx1, idx2, dist, n_samples) in enumerate(linkage_matrix):
        idx1, idx2 = int(idx1), int(idx2)
        parent_idx = n_assets + i
        
        # Get clusters from children
        cluster1 = nodes[idx1]["cluster"]
        cluster2 = nodes[idx2]["cluster"]
        
        nodes[parent_idx] = {
            "children": [idx1, idx2],
            "cluster": cluster1 + cluster2,
            "distance": dist
        }
    
    # Root is the last node
    root_idx = n_assets + len(linkage_matrix) - 1
    
    return nodes, root_idx


def risk_parity_weights(
    volatilities: Dict[str, float],
    cluster_tickers: List[str]
) -> Dict[str, float]:
    """
    Allocate weights using risk parity: w_i = (1/vol_i) / sum(1/vol_j)
    
    Parameters:
    -----------
    volatilities : dict
        Volatility (risk) for each ticker
    cluster_tickers : list
        Tickers in the cluster
        
    Returns:
    --------
    weights : dict
        Risk parity weights for assets in cluster
    """
    inv_vols = {t: 1.0 / volatilities[t] for t in cluster_tickers}
    total_inv_vol = sum(inv_vols.values())
    
    weights = {t: inv_vols[t] / total_inv_vol for t in cluster_tickers}
    
    return weights


def recursive_bisection(
    nodes: Dict[int, Any],
    node_idx: int,
    volatilities: Dict[str, float],
    parent_weight: float = 1.0
) -> Dict[str, float]:
    """
    Recursively allocate portfolio weight using risk parity.
    
    Top-down approach: at each node, split weight among children using risk parity,
    then recursively apply to subtrees.
    
    Parameters:
    -----------
    nodes : dict
        Tree structure from tree_clustering
    node_idx : int
        Current node index
    volatilities : dict
        Volatility for each ticker
    parent_weight : float
        Weight allocated to current node from parent
        
    Returns:
    --------
    weights : dict
        Weights for all assets
    """
    weights = {}
    node = nodes[node_idx]
    cluster = node["cluster"]
    
    if "ticker" in node:  # Leaf node (single asset)
        weights[node["ticker"]] = parent_weight
        return weights
    
    # Non-leaf node: allocate weight to children using risk parity
    children_idx = node["children"]
    
    child_clusters = [nodes[idx]["cluster"] for idx in children_idx]
    child_vols = {}
    
    # Compute cluster volatility (risk): risk of cluster = sqrt(weighted_var)
    # Simplified: use average volatility of assets in cluster
    for idx, cluster in zip(children_idx, child_clusters):
        avg_vol = np.mean([volatilities[t] for t in cluster])
        child_vols[idx] = avg_vol
    
    # Risk parity: allocate inverse proportion to risk
    inv_vols = {idx: 1.0 / child_vols[idx] for idx in children_idx}
    total_inv_vol = sum(inv_vols.values())
    child_weights = {idx: inv_vols[idx] / total_inv_vol for idx in children_idx}
    
    # Recursively allocate to subtrees
    for idx in children_idx:
        child_weight = child_weights[idx] * parent_weight
        weights.update(recursive_bisection(nodes, idx, volatilities, child_weight))
    
    return weights


def get_hrp_weights(
    returns: pd.DataFrame,
    config: HRPConfig = HRPConfig()
) -> pd.Series:
    """
    Main function to compute Hierarchical Risk Parity weights.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        T x N historical asset returns
    config : HRPConfig
        Configuration for clustering and constraints
        
    Returns:
    --------
    pd.Series
        HRP portfolio weights
    """
    # Estimate covariance and volatility
    cov = estimate_covariance(returns, annualize=False)
    vols = np.sqrt(np.diag(cov.values))
    volatilities = dict(zip(cov.index, vols))
    
    # Hierarchical clustering
    linkage_matrix, tickers, corr = hierarchical_clustering(returns, cov, config)
    
    # Build tree structure
    nodes, root_idx = tree_clustering(linkage_matrix, tickers)
    
    # Recursive risk parity allocation
    weights_dict = recursive_bisection(nodes, root_idx, volatilities, parent_weight=1.0)
    
    # Convert to Series, aligned with original ticker order
    weights = pd.Series(
        [weights_dict.get(t, 0.0) for t in returns.columns],
        index=returns.columns
    )
    
    # Apply constraints
    if config.long_only:
        weights = weights.clip(lower=0)
        weights = weights / weights.sum()  # renormalize
    
    if config.max_weight is not None:
        # Cap weights and renormalize
        weights = weights.clip(upper=config.max_weight)
        weights = weights / weights.sum()
    
    return weights


def portfolio_volatility(weights: pd.Series, cov: pd.DataFrame) -> float:
    """
    Compute portfolio volatility.
    
    Parameters:
    -----------
    weights : pd.Series
        Portfolio weights
    cov : pd.DataFrame
        Covariance matrix
        
    Returns:
    --------
    float
        Portfolio volatility
    """
    weights = weights[cov.index]
    portfolio_var = weights @ cov @ weights
    return np.sqrt(portfolio_var)
