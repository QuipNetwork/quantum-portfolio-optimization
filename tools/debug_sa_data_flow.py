#!/usr/bin/env python3
# Copyright (C) 2025 Postquant Labs Incorporated
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Debug Simulated Annealing optimizer data flow.

Traces inputs/outputs at each stage:
1. Input: mu, Sigma, clusters → BQM coefficients
2. Sampling: BQM → raw binary samples
3. Decoding: binary samples → weights
4. Aggregation: cluster weights + meta-cluster weights → portfolio weights
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import dimod

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qpo.qubo.discrete_levels import DiscreteLevelFormulator
from qpo.utils.data_prep import load_portfolio_data
from clustering import UniformClusterer


def analyze_bqm_structure(bqm: dimod.BinaryQuadraticModel, label: str = "BQM"):
    """Analyze and print BQM structure statistics."""
    print(f"\n{'='*80}")
    print(f"{label} STRUCTURE ANALYSIS")
    print(f"{'='*80}")

    n_vars = len(bqm.variables)
    n_linear = len(bqm.linear)
    n_quadratic = len(bqm.quadratic)

    # Compute coefficient statistics
    h_values = list(bqm.linear.values())
    Q_values = list(bqm.quadratic.values())

    print(f"Variables: {n_vars}")
    print(f"Linear terms (h): {n_linear}")
    if h_values:
        print(f"  Range: [{min(h_values):.6f}, {max(h_values):.6f}]")
        print(f"  Mean: {np.mean(h_values):.6f}")
        print(f"  Std: {np.std(h_values):.6f}")

    print(f"Quadratic terms (Q): {n_quadratic}")
    if Q_values:
        print(f"  Range: [{min(Q_values):.6f}, {max(Q_values):.6f}]")
        print(f"  Mean: {np.mean(Q_values):.6f}")
        print(f"  Std: {np.std(Q_values):.6f}")

    print(f"Offset: {bqm.offset:.6f}")

    # Show sample of variables
    sample_vars = list(bqm.variables)[:10]
    print(f"\nSample variables (first 10): {sample_vars}")

    # Show sample of linear coefficients
    print(f"\nSample linear coefficients:")
    for var in sample_vars[:5]:
        print(f"  {var}: {bqm.linear[var]:.6f}")

    # Show sample of quadratic coefficients
    print(f"\nSample quadratic coefficients (first 5):")
    for i, (edge, coeff) in enumerate(list(bqm.quadratic.items())[:5]):
        print(f"  {edge}: {coeff:.6f}")


def analyze_sample(sample: Dict[str, int], tickers: List[str], n_levels: int):
    """Analyze a binary sample."""
    print(f"\n{'='*80}")
    print(f"BINARY SAMPLE ANALYSIS")
    print(f"{'='*80}")

    print(f"Total variables in sample: {len(sample)}")
    print(f"Variables set to 1: {sum(sample.values())}")
    print(f"Variables set to 0: {len(sample) - sum(sample.values())}")

    # Check thermometer encoding for each asset
    print(f"\nThermometer encoding check (first 5 assets):")
    thermometer_violations = 0
    all_zero_assets = 0

    for ticker in tickers[:5]:
        bits = []
        for q in range(n_levels):
            var = f"{ticker}_{q}"
            bit = sample.get(var, 0)
            bits.append(bit)

        # Check thermometer constraint
        is_valid = True
        for q in range(1, n_levels):
            if bits[q] == 1 and bits[q-1] == 0:
                is_valid = False
                thermometer_violations += 1
                break

        # Check if all zeros
        if sum(bits) == 0:
            all_zero_assets += 1

        valid_str = "✓" if is_valid else "✗ VIOLATION"
        print(f"  {ticker}: {bits} {valid_str}")

    # Count violations and all-zero across all assets
    for ticker in tickers[5:]:
        bits = []
        for q in range(n_levels):
            var = f"{ticker}_{q}"
            bit = sample.get(var, 0)
            bits.append(bit)

        for q in range(1, n_levels):
            if bits[q] == 1 and bits[q-1] == 0:
                thermometer_violations += 1
                break

        if sum(bits) == 0:
            all_zero_assets += 1

    print(f"\nOverall statistics:")
    print(f"  Thermometer violations: {thermometer_violations}/{len(tickers)} assets")
    print(f"  All-zero assets: {all_zero_assets}/{len(tickers)} assets")


def analyze_weights(weights: pd.Series, label: str = "Weights"):
    """Analyze decoded weights."""
    print(f"\n{'='*80}")
    print(f"{label} ANALYSIS")
    print(f"{'='*80}")

    print(f"Number of assets: {len(weights)}")
    print(f"Sum of weights: {weights.sum():.6f}")
    print(f"Non-zero weights: {(weights > 1e-6).sum()}/{len(weights)}")

    if weights.sum() > 0:
        print(f"\nWeight statistics:")
        print(f"  Min: {weights.min():.6f}")
        print(f"  Max: {weights.max():.6f}")
        print(f"  Mean: {weights.mean():.6f}")
        print(f"  Std: {weights.std():.6f}")

        print(f"\nTop 10 weights:")
        top_weights = weights.nlargest(10)
        for ticker, weight in top_weights.items():
            print(f"  {ticker}: {weight:.6f}")
    else:
        print(f"\n⚠ WARNING: All weights are ZERO!")


def test_sa_data_flow(csv_path: str, n_levels: int = 6, max_cluster_size: int = 18):
    """
    Test SA optimizer with detailed logging at each stage.

    Args:
        csv_path: Path to portfolio CSV
        n_levels: Number of discrete levels
        max_cluster_size: Max assets per cluster
    """
    print("="*80)
    print("SIMULATED ANNEALING DATA FLOW DIAGNOSTIC")
    print("="*80)

    # Load data
    print(f"\nLoading data from: {csv_path}")
    prices, returns = load_portfolio_data(csv_path, preprocess=False)
    print(f"  Assets: {len(returns.columns)}")
    print(f"  Days: {len(returns)}")

    # Compute mu and Sigma
    mu = returns.mean() * 252
    Sigma = returns.cov() * 252

    print(f"\n{'='*80}")
    print(f"INPUT DATA ANALYSIS")
    print(f"{'='*80}")
    print(f"Expected returns (mu):")
    print(f"  Range: [{mu.min():.6f}, {mu.max():.6f}]")
    print(f"  Mean: {mu.mean():.6f}")
    print(f"  Std: {mu.std():.6f}")

    print(f"\nCovariance matrix (Sigma):")
    print(f"  Shape: {Sigma.shape}")
    print(f"  Diagonal (variance) range: [{Sigma.values.diagonal().min():.6f}, {Sigma.values.diagonal().max():.6f}]")
    print(f"  Off-diagonal (covariance) range: [{Sigma.values[np.triu_indices_from(Sigma.values, k=1)].min():.6f}, {Sigma.values[np.triu_indices_from(Sigma.values, k=1)].max():.6f}]")

    # Cluster assets
    print(f"\n{'='*80}")
    print(f"CLUSTERING")
    print(f"{'='*80}")
    clusterer = UniformClusterer(max_cluster_size=max_cluster_size, target_cluster_size=10)
    clusters = clusterer.cluster(returns)
    print(f"Number of clusters: {len(clusters)}")
    for cluster_id, tickers in clusters.items():
        print(f"  Cluster {cluster_id}: {len(tickers)} assets")

    # Test formulation for first cluster
    first_cluster_id = list(clusters.keys())[0]
    first_cluster_tickers = clusters[first_cluster_id]
    cluster_mu = mu[first_cluster_tickers]
    cluster_Sigma = Sigma.loc[first_cluster_tickers, first_cluster_tickers]

    print(f"\n{'='*80}")
    print(f"TESTING CLUSTER {first_cluster_id}")
    print(f"{'='*80}")
    print(f"Assets: {first_cluster_tickers}")
    print(f"\nCluster returns (mu):")
    print(f"  Range: [{cluster_mu.min():.6f}, {cluster_mu.max():.6f}]")

    # Test different parameter configurations
    param_configs = [
        {
            'name': 'Default (alpha=10, beta=2, budget=0, thermo=10)',
            'alpha': 10,
            'beta': 2,
            'budget_penalty': 0.0,
            'thermometer_penalty': 10.0
        },
        {
            'name': 'Strong budget (alpha=10, beta=2, budget=100, thermo=10)',
            'alpha': 10,
            'beta': 2,
            'budget_penalty': 100.0,
            'thermometer_penalty': 10.0
        },
        {
            'name': 'Aggressive return (alpha=50, beta=2, budget=0, thermo=10)',
            'alpha': 50,
            'beta': 2,
            'budget_penalty': 0.0,
            'thermometer_penalty': 10.0
        },
    ]

    for config in param_configs:
        print(f"\n\n{'#'*80}")
        print(f"CONFIGURATION: {config['name']}")
        print(f"{'#'*80}")

        # Create formulator
        formulator = DiscreteLevelFormulator(
            n_levels=n_levels,
            alpha=config['alpha'],
            beta=config['beta'],
            budget_penalty=config['budget_penalty'],
            thermometer_penalty=config['thermometer_penalty']
        )

        # Create BQM
        print(f"\nFormulating BQM...")
        bqm = formulator.formulate_cluster(
            first_cluster_tickers,
            cluster_mu,
            cluster_Sigma,
            include_budget_constraint=(config['budget_penalty'] > 0)
        )

        # Analyze BQM
        analyze_bqm_structure(bqm, label=f"CLUSTER BQM ({config['name']})")

        # Sample with SA
        print(f"\n{'='*80}")
        print(f"SIMULATED ANNEALING SAMPLING")
        print(f"{'='*80}")
        from neal import SimulatedAnnealingSampler
        sampler = SimulatedAnnealingSampler()

        print(f"Running SA with num_reads=256, num_sweeps=256...")
        response = sampler.sample(bqm, num_reads=256, num_sweeps=256)

        print(f"Samples returned: {len(response)}")

        # Analyze best sample
        best_sample = response.first.sample
        best_energy = response.first.energy

        print(f"Best sample energy: {best_energy:.6f}")
        analyze_sample(best_sample, first_cluster_tickers, n_levels)

        # Decode weights
        print(f"\n{'='*80}")
        print(f"DECODING WEIGHTS")
        print(f"{'='*80}")
        weights = formulator.decode_solution(best_sample, first_cluster_tickers)
        analyze_weights(weights, label=f"DECODED WEIGHTS ({config['name']})")

        # Analyze all samples (check for diversity)
        print(f"\n{'='*80}")
        print(f"SAMPLE DIVERSITY ANALYSIS")
        print(f"{'='*80}")
        energies = [sample.energy for sample in response.data()]
        print(f"Energy statistics across {len(energies)} samples:")
        print(f"  Min: {min(energies):.6f}")
        print(f"  Max: {max(energies):.6f}")
        print(f"  Mean: {np.mean(energies):.6f}")
        print(f"  Std: {np.std(energies):.6f}")
        print(f"  Unique energies: {len(set(energies))}")

        # Check if all samples are identical
        unique_samples = set()
        for sample in response.data():
            sample_tuple = tuple(sorted(sample.sample.items()))
            unique_samples.add(sample_tuple)

        print(f"  Unique samples: {len(unique_samples)}/{len(response)}")

        if len(unique_samples) == 1:
            print(f"\n⚠ WARNING: ALL SAMPLES ARE IDENTICAL! SA is not exploring solution space.")
        elif len(unique_samples) < 10:
            print(f"\n⚠ WARNING: Very few unique samples. SA may be converging too quickly.")


def test_meta_cluster_aggregation(csv_path: str, n_levels: int = 6, max_cluster_size: int = 18):
    """
    Test full multi-cluster + meta-cluster optimization.
    """
    print("\n\n" + "="*80)
    print("MULTI-CLUSTER + META-CLUSTER INTEGRATION TEST")
    print("="*80)

    # Load data
    prices, returns = load_portfolio_data(csv_path, preprocess=False)
    mu = returns.mean() * 252
    Sigma = returns.cov() * 252

    # Cluster
    clusterer = UniformClusterer(max_cluster_size=max_cluster_size, target_cluster_size=10)
    clusters = clusterer.cluster(returns)

    print(f"\nClusters: {len(clusters)}")
    for cluster_id, tickers in clusters.items():
        print(f"  Cluster {cluster_id}: {len(tickers)} assets")

    # Create formulator
    formulator = DiscreteLevelFormulator(
        n_levels=n_levels,
        alpha=10,
        beta=2,
        budget_penalty=0.0,
        thermometer_penalty=10.0
    )

    # Combine all cluster BQMs
    print(f"\n{'='*80}")
    print(f"COMBINING CLUSTER BQMs")
    print(f"{'='*80}")

    h_combined = {}
    Q_combined = {}
    cluster_info = []

    for cluster_id, cluster_tickers in clusters.items():
        cluster_mu = mu[cluster_tickers]
        cluster_Sigma = Sigma.loc[cluster_tickers, cluster_tickers]

        bqm = formulator.formulate_cluster(
            cluster_tickers,
            cluster_mu,
            cluster_Sigma,
            include_budget_constraint=False
        )

        # Merge
        for var, coeff in bqm.linear.items():
            h_combined[var] = h_combined.get(var, 0.0) + coeff

        for edge, coeff in bqm.quadratic.items():
            Q_combined[edge] = Q_combined.get(edge, 0.0) + coeff

        cluster_info.append({
            'id': cluster_id,
            'tickers': cluster_tickers,
            'variables': list(bqm.variables)
        })

    combined_bqm = dimod.BinaryQuadraticModel(h_combined, Q_combined, 0.0, dimod.BINARY)
    analyze_bqm_structure(combined_bqm, label="COMBINED ASSET CLUSTERS BQM")

    # Create meta-cluster
    print(f"\n{'='*80}")
    print(f"CREATING META-CLUSTER BQM")
    print(f"{'='*80}")

    # Compute cluster statistics
    cluster_ids = list(clusters.keys())
    cluster_mu = pd.Series(index=cluster_ids, dtype=float)
    cluster_Sigma = pd.DataFrame(0.0, index=cluster_ids, columns=cluster_ids)

    for cluster_id in cluster_ids:
        tickers = clusters[cluster_id]
        cluster_mu[cluster_id] = mu[tickers].mean()

    for i, cluster_i in enumerate(cluster_ids):
        tickers_i = clusters[cluster_i]
        for j, cluster_j in enumerate(cluster_ids):
            tickers_j = clusters[cluster_j]
            cluster_sigma_block = Sigma.loc[tickers_i, tickers_j]
            cluster_Sigma.loc[cluster_i, cluster_j] = cluster_sigma_block.values.mean()

    print(f"Meta-cluster statistics:")
    print(f"  Cluster returns (mu): {cluster_mu.values}")
    print(f"  Cluster covariances (diagonal): {cluster_Sigma.values.diagonal()}")

    meta_bqm = formulator.formulate_cluster(
        cluster_ids,
        cluster_mu,
        cluster_Sigma,
        include_budget_constraint=False
    )
    analyze_bqm_structure(meta_bqm, label="META-CLUSTER BQM")

    # Combine all BQMs
    print(f"\n{'='*80}")
    print(f"COMBINING ALL BQMs (ASSETS + META)")
    print(f"{'='*80}")

    for var, coeff in meta_bqm.linear.items():
        combined_bqm.add_variable(var, coeff)

    for edge, coeff in meta_bqm.quadratic.items():
        combined_bqm.add_interaction(edge[0], edge[1], coeff)

    cluster_info.append({
        'id': 'META_CLUSTER',
        'tickers': cluster_ids,
        'variables': list(meta_bqm.variables)
    })

    analyze_bqm_structure(combined_bqm, label="FULL COMBINED BQM (ASSETS + META)")

    # Sample
    print(f"\n{'='*80}")
    print(f"SAMPLING COMBINED BQM")
    print(f"{'='*80}")
    from neal import SimulatedAnnealingSampler
    sampler = SimulatedAnnealingSampler()

    print(f"Running SA with num_reads=256, num_sweeps=256...")
    response = sampler.sample(combined_bqm, num_reads=256, num_sweeps=256)

    best_sample = response.first.sample
    best_energy = response.first.energy
    print(f"Best sample energy: {best_energy:.6f}")

    # Decode all clusters
    print(f"\n{'='*80}")
    print(f"DECODING ALL CLUSTERS")
    print(f"{'='*80}")

    cluster_results = {}
    meta_cluster_weights = None

    for cluster_data in cluster_info:
        cluster_id = cluster_data['id']
        cluster_tickers = cluster_data['tickers']

        weights = formulator.decode_solution(best_sample, cluster_tickers, warn_budget_violation=False)

        if cluster_id == 'META_CLUSTER':
            meta_cluster_weights = weights
            analyze_weights(weights, label=f"META-CLUSTER WEIGHTS")
        else:
            cluster_results[cluster_id] = {
                'weights': weights,
                'tickers': cluster_tickers
            }
            analyze_weights(weights, label=f"CLUSTER {cluster_id} WEIGHTS")

    # Apply meta-cluster aggregation
    print(f"\n{'='*80}")
    print(f"APPLYING META-CLUSTER AGGREGATION")
    print(f"{'='*80}")

    print(f"Meta-cluster weights (unnormalized):")
    for cluster_id, weight in meta_cluster_weights.items():
        print(f"  Cluster {cluster_id}: {weight:.6f}")

    # Magnitude preservation aggregation
    n_clusters = len(meta_cluster_weights)
    original_sum = meta_cluster_weights.sum()

    if original_sum > 0:
        cluster_allocations = (meta_cluster_weights / original_sum) * n_clusters
    else:
        cluster_allocations = pd.Series(1.0, index=meta_cluster_weights.index)

    print(f"\nCluster allocations (after magnitude preservation):")
    for cluster_id, alloc in cluster_allocations.items():
        print(f"  Cluster {cluster_id}: {alloc:.6f}")

    # Aggregate
    all_weights = {}
    for cluster_id, result in cluster_results.items():
        cluster_alloc = cluster_allocations[cluster_id]
        cluster_weights = result['weights']
        cluster_weight_sum = cluster_weights.sum()

        if cluster_weight_sum > 0:
            for ticker in cluster_weights.index:
                all_weights[ticker] = cluster_weights[ticker] * cluster_alloc
        else:
            # Equal weight within cluster
            n_tickers = len(result['tickers'])
            for ticker in result['tickers']:
                all_weights[ticker] = cluster_alloc / n_tickers

    # Normalize
    total_weight = sum(all_weights.values())
    if total_weight > 0:
        final_weights = pd.Series({k: v/total_weight for k, v in all_weights.items()})
    else:
        final_weights = pd.Series(1.0 / len(mu), index=mu.index)

    analyze_weights(final_weights, label="FINAL PORTFOLIO WEIGHTS")

    # Compute portfolio metrics
    portfolio_return = np.dot(final_weights.values, mu.values)
    portfolio_risk = np.sqrt(np.dot(final_weights.values, np.dot(Sigma.values, final_weights.values)))
    sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

    print(f"\n{'='*80}")
    print(f"PORTFOLIO METRICS")
    print(f"{'='*80}")
    print(f"Expected return: {portfolio_return:.6f}")
    print(f"Risk (volatility): {portfolio_risk:.6f}")
    print(f"Sharpe ratio: {sharpe:.6f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Debug SA data flow')
    parser.add_argument('--portfolio-csv', type=str, required=True,
                       help='Path to portfolio CSV')
    parser.add_argument('--n-levels', type=int, default=6,
                       help='Number of discrete levels')
    parser.add_argument('--max-cluster-size', type=int, default=18,
                       help='Max assets per cluster')
    parser.add_argument('--test-aggregation', action='store_true',
                       help='Test full multi-cluster aggregation')

    args = parser.parse_args()

    # Test single cluster
    test_sa_data_flow(args.portfolio_csv, args.n_levels, args.max_cluster_size)

    # Test full aggregation if requested
    if args.test_aggregation:
        test_meta_cluster_aggregation(args.portfolio_csv, args.n_levels, args.max_cluster_size)


if __name__ == '__main__':
    main()
