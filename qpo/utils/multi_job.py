# Copyright (C) 2025 Postquant Labs Incorporated
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Multi-job optimization for portfolios with more clusters than template capacity.

When clustering produces more clusters than a fixed embedding template supports,
this module splits the portfolio into multiple optimization jobs and combines results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def cluster_with_constraints(clusters: Dict[str, List[str]],
                            returns: pd.DataFrame,
                            max_cluster_size: int,
                            max_clusters: int) -> List[Dict[str, List[str]]]:
    """
    Validate clustering constraints and split into jobs if needed.

    Args:
        clusters: Dictionary of {cluster_id: [tickers]}
        returns: Returns DataFrame for computing inter-cluster correlations
        max_cluster_size: Maximum assets per cluster (HARD constraint)
        max_clusters: Maximum clusters per job (SOFT - will split if exceeded)

    Returns:
        List of cluster dictionaries (one per job)

    Raises:
        ValueError: If max_cluster_size constraint cannot be satisfied
    """
    n_clusters = len(clusters)

    # Check max_cluster_size constraint (HARD - cannot be violated)
    cluster_sizes = [len(tickers) for tickers in clusters.values()]
    max_size = max(cluster_sizes)

    if max_size > max_cluster_size:
        violating = [(cid, len(t)) for cid, t in clusters.items() if len(t) > max_cluster_size]
        raise ValueError(
            f"Cannot satisfy max_cluster_size={max_cluster_size}. "
            f"{len(violating)} clusters exceed limit:\n" +
            "\n".join([f"  {cid}: {size} assets" for cid, size in violating[:5]]) +
            f"\nThis violates quantum hardware constraints. "
            f"Increase max_cluster_size or use more aggressive clustering."
        )

    # Check max_clusters constraint (SOFT - can split into jobs)
    if n_clusters <= max_clusters:
        print(f"✓ Single job: {n_clusters} clusters (within {max_clusters} limit)")
        return [clusters]

    # Need to split
    print(f"Portfolio has {n_clusters} clusters but template supports {max_clusters}.")
    print(f"Splitting into multiple optimization jobs...")

    return split_clusters_into_jobs(clusters, returns, max_clusters)


def split_clusters_into_jobs(clusters: Dict[str, List[str]],
                             returns: pd.DataFrame,
                             max_clusters_per_job: int,
                             method: str = 'hierarchical') -> List[Dict[str, List[str]]]:
    """
    Split clusters into multiple jobs.

    Args:
        clusters: Dictionary of {cluster_id: [tickers]}
        returns: Returns DataFrame for computing correlations
        max_clusters_per_job: Maximum clusters per job
        method: 'hierarchical' (recommended) or 'greedy' (fallback)

    Returns:
        List of cluster dictionaries, one per job
    """
    n_clusters = len(clusters)

    if n_clusters <= max_clusters_per_job:
        return [clusters]

    # Auto-select method based on cluster count
    if n_clusters > 100 and method == 'hierarchical':
        print(f"Warning: {n_clusters} clusters is large, using greedy split")
        method = 'greedy'

    if method == 'hierarchical':
        return _hierarchical_split(clusters, returns, max_clusters_per_job)
    else:
        return _greedy_split(clusters, max_clusters_per_job)


def _hierarchical_split(clusters: Dict[str, List[str]],
                       returns: pd.DataFrame,
                       max_per_job: int) -> List[Dict[str, List[str]]]:
    """
    Hierarchically re-cluster clusters based on inter-cluster correlations.

    Groups highly correlated clusters into the same job to minimize
    cross-job dependencies and improve combined solution quality.

    Uses vectorized correlation computation for speed.
    """
    cluster_items = list(clusters.items())
    n_clusters = len(cluster_items)

    print(f"  Using hierarchical split (vectorized correlation)...")

    # Compute cluster centroids (equal-weighted average)
    centroids = []
    for cluster_id, tickers in cluster_items:
        centroid = returns[tickers].mean(axis=1)
        centroids.append(centroid.values)

    # Vectorized correlation computation (FAST!)
    centroid_matrix = np.column_stack(centroids)  # Shape: (T, C)
    corr_matrix = np.corrcoef(centroid_matrix.T)  # Shape: (C, C)

    # Convert correlation to distance (ensure symmetry)
    distance = 1 - np.abs(corr_matrix)
    distance = (distance + distance.T) / 2  # Force perfect symmetry
    np.fill_diagonal(distance, 0)

    # Hierarchical clustering of clusters
    condensed = squareform(distance, checks=False)  # Skip check since we ensured symmetry
    Z = linkage(condensed, method='ward')

    # Calculate target number of jobs
    n_jobs = (n_clusters + max_per_job - 1) // max_per_job

    # Cut dendrogram
    job_assignments = fcluster(Z, n_jobs, criterion='maxclust')

    # Build job dictionaries
    jobs_dict = {i: {} for i in range(1, n_jobs + 1)}
    for (cluster_id, tickers), job_id in zip(cluster_items, job_assignments):
        jobs_dict[job_id][cluster_id] = tickers

    # Convert to list and validate sizes
    jobs = []
    for job_idx, job_dict in enumerate(jobs_dict.values()):
        if not job_dict:
            continue

        # If job somehow exceeds max (shouldn't happen), split it
        if len(job_dict) > max_per_job:
            items = list(job_dict.items())
            for i in range(0, len(items), max_per_job):
                chunk = dict(items[i:i+max_per_job])
                if chunk:
                    jobs.append(chunk)
        else:
            jobs.append(job_dict)

    # Print summary
    print(f"  Created {len(jobs)} jobs:")
    for i, job in enumerate(jobs):
        n_assets = sum(len(t) for t in job.values())
        print(f"    Job {i}: {len(job)} clusters, {n_assets} assets")

    return jobs


def _greedy_split(clusters: Dict[str, List[str]],
                 max_per_job: int) -> List[Dict[str, List[str]]]:
    """
    Greedy sequential split (fallback - ignores correlations).

    Simply partitions clusters in order without considering correlations.
    Fast but may produce sub-optimal job assignments.
    """
    print(f"  Using greedy sequential split...")

    cluster_items = list(clusters.items())
    jobs = []

    for i in range(0, len(cluster_items), max_per_job):
        job_dict = dict(cluster_items[i:i+max_per_job])
        jobs.append(job_dict)

    print(f"  Created {len(jobs)} jobs (sequential)")
    return jobs


def optimize_portfolio_multi_job(jobs: List[Dict[str, List[str]]],
                                 returns: pd.DataFrame,
                                 mu: pd.Series,
                                 Sigma: pd.DataFrame,
                                 optimizer,
                                 combination_method: str = 'equal') -> pd.Series:
    """
    Run optimization across multiple jobs and combine results.

    Args:
        jobs: List of cluster dictionaries from split_clusters_into_jobs()
        returns: Full returns DataFrame
        mu: Full expected returns
        Sigma: Full covariance matrix
        optimizer: Optimizer instance (must have optimize() method)
        combination_method: 'equal' or 'sharpe' or 'optimal'

    Returns:
        Combined portfolio weights (Series indexed by tickers)
    """
    if len(jobs) == 1:
        # Single job - normal optimization
        return optimizer.optimize(returns, mu, Sigma, jobs[0])

    # Multiple jobs
    print(f"\nRunning {len(jobs)} optimization jobs...")

    job_weights = {}
    job_metrics = {}

    for job_idx, clusters in enumerate(jobs):
        print(f"\n  Job {job_idx}: {len(clusters)} clusters...")

        # Get tickers in this job
        job_tickers = [t for cluster in clusters.values() for t in cluster]

        # Subset data for this job
        job_returns = returns[job_tickers]
        job_mu = mu[job_tickers]
        job_Sigma = Sigma.loc[job_tickers, job_tickers]

        # Optimize this job
        weights = optimizer.optimize(job_returns, job_mu, job_Sigma, clusters)
        job_weights[job_idx] = weights

        # Compute metrics for weighting
        job_return = (weights * job_mu).sum()
        job_risk = np.sqrt(weights.T @ job_Sigma @ weights)
        job_sharpe = job_return / job_risk if job_risk > 0 else 0

        job_metrics[job_idx] = {
            'return': job_return,
            'risk': job_risk,
            'sharpe': job_sharpe,
            'n_assets': len(job_tickers)
        }

        print(f"    Return: {job_return:.2%}, Risk: {job_risk:.2%}, Sharpe: {job_sharpe:.2f}")

    # Combine weights from all jobs
    combined = combine_job_weights(
        job_weights,
        job_metrics,
        returns.columns,
        method=combination_method
    )

    return combined


def combine_job_weights(job_weights: Dict[int, pd.Series],
                       job_metrics: Dict[int, Dict],
                       all_tickers: pd.Index,
                       method: str = 'equal') -> pd.Series:
    """
    Combine weights from multiple optimization jobs.

    Args:
        job_weights: {job_id: weights Series}
        job_metrics: {job_id: {'return', 'risk', 'sharpe', 'n_assets'}}
        all_tickers: All tickers in portfolio
        method: 'equal' (default), 'sharpe', or 'return'

    Returns:
        Combined weights (Series indexed by all_tickers)
    """
    combined = pd.Series(0.0, index=all_tickers)

    if method == 'equal':
        # Equal allocation to each job
        job_fraction = 1.0 / len(job_weights)

        for job_id, weights in job_weights.items():
            for ticker, weight in weights.items():
                combined[ticker] = weight * job_fraction

    elif method == 'sharpe':
        # Weight by Sharpe ratio
        sharpe_sum = sum(m['sharpe'] for m in job_metrics.values())

        if sharpe_sum > 0:
            for job_id, weights in job_weights.items():
                job_fraction = job_metrics[job_id]['sharpe'] / sharpe_sum
                for ticker, weight in weights.items():
                    combined[ticker] = weight * job_fraction
        else:
            # Fallback to equal if all Sharpe ratios are zero
            return combine_job_weights(job_weights, job_metrics, all_tickers, method='equal')

    elif method == 'return':
        # Weight by expected return
        return_sum = sum(m['return'] for m in job_metrics.values())

        if return_sum > 0:
            for job_id, weights in job_weights.items():
                job_fraction = job_metrics[job_id]['return'] / return_sum
                for ticker, weight in weights.items():
                    combined[ticker] = weight * job_fraction
        else:
            return combine_job_weights(job_weights, job_metrics, all_tickers, method='equal')

    else:
        raise ValueError(f"Unknown combination method: {method}")

    # Renormalize to ensure sum = 1.0 (handle floating point errors)
    combined = combined / combined.sum()

    print(f"\nCombined {len(job_weights)} jobs using '{method}' method")
    print(f"  Total assets: {(combined > 0).sum()}/{len(all_tickers)}")
    print(f"  Total weight: {combined.sum():.6f}")

    return combined
