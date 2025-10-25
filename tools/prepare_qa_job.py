#!/usr/bin/env python3
# Copyright (C) 2025 Postquant Labs Incorporated
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Prepare quantum annealing job payloads for D-Wave SAPI submission.

This tool generates job payload files (JSON) for backtest schedules that can be
submitted via curl to the D-Wave solver endpoint.

Example usage:
    python tools/prepare_qa_job.py \\
        --portfolio-csv portfolio.csv \\
        --output-dir backtest_jobs \\
        --train-days 252 \\
        --test-days 21 \\
        --step-days 21 \\
        --num-reads 100

This generates: train.json, period-1.json, period-2.json, ...
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import dimod

# Suppress D-Wave warnings
warnings.filterwarnings('ignore', message='All bqm biases are zero')
warnings.filterwarnings('ignore', message='.*Temperature range is set arbitrarily.*')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qpo.utils.data_prep import load_portfolio_data
from qpo.qubo.discrete_levels import DiscreteLevelFormulator
from qpo.utils.topology_selection import select_optimal_template
from clustering import UniformClusterer


def load_embedding_template(template_name: str) -> Dict[str, Any]:
    """Load precomputed embedding template from embeddings/templates/."""
    template_path = project_root / 'embeddings' / 'templates' / template_name

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, 'r') as f:
        template_data = json.load(f)

    return template_data


def create_clusters(
    returns: pd.DataFrame,
    max_cluster_size: int,
    target_cluster_size: int
) -> Dict[str, List[str]]:
    """Create uniform clusters from portfolio data."""
    clusterer = UniformClusterer(
        max_cluster_size=max_cluster_size,
        target_cluster_size=target_cluster_size
    )
    clusters = clusterer.cluster(returns)
    return clusters


def pad_clusters_to_template(
    clusters: Dict[str, List[str]],
    mu: pd.Series,
    Sigma: pd.DataFrame,
    expected_assets_per_cluster: int,
    expected_n_clusters: int
) -> tuple:
    """
    Pad clusters with dummy assets to match template dimensions.

    Returns:
        (padded_clusters, padded_mu, padded_Sigma, real_tickers)
    """
    padded_clusters = {}
    real_tickers = set(mu.index)
    dummy_counter = 0

    # Pad existing clusters to target size
    for cluster_id, tickers in clusters.items():
        n_real = len(tickers)
        n_padding = max(0, expected_assets_per_cluster - n_real)

        padded_tickers = list(tickers)

        for i in range(n_padding):
            dummy_ticker = f"_DUMMY_{dummy_counter}"
            padded_tickers.append(dummy_ticker)
            dummy_counter += 1

        padded_clusters[cluster_id] = padded_tickers

    # Add empty clusters if needed
    current_n_clusters = len(padded_clusters)
    if current_n_clusters < expected_n_clusters:
        for i in range(expected_n_clusters - current_n_clusters):
            empty_cluster_id = f"_EMPTY_CLUSTER_{i}"
            empty_cluster = []

            for j in range(expected_assets_per_cluster):
                dummy_ticker = f"_DUMMY_{dummy_counter}"
                empty_cluster.append(dummy_ticker)
                dummy_counter += 1

            padded_clusters[empty_cluster_id] = empty_cluster

    # Extend mu with small negative returns for dummy assets
    all_tickers = []
    for tickers in padded_clusters.values():
        all_tickers.extend(tickers)

    dummy_tickers = [t for t in all_tickers if t.startswith('_DUMMY_')]

    padded_mu = mu.copy()
    for dummy in dummy_tickers:
        padded_mu[dummy] = -1e-6

    # Extend Sigma with zeros for dummy assets
    if dummy_tickers:
        n_dummies = len(dummy_tickers)
        n_existing = len(Sigma)

        dummy_block = pd.DataFrame(
            0.0,
            index=dummy_tickers,
            columns=list(Sigma.columns) + dummy_tickers
        )

        existing_to_dummy = pd.DataFrame(
            0.0,
            index=Sigma.index,
            columns=dummy_tickers
        )

        for dummy in dummy_tickers:
            dummy_block.loc[dummy, dummy] = 1e-10

        padded_Sigma = pd.concat([
            pd.concat([Sigma, existing_to_dummy], axis=1),
            dummy_block
        ], axis=0)
    else:
        padded_Sigma = Sigma.copy()

    return padded_clusters, padded_mu, padded_Sigma, real_tickers


def build_combined_bqm(
    clusters: Dict[str, List[str]],
    mu: pd.Series,
    Sigma: pd.DataFrame,
    formulator: DiscreteLevelFormulator
) -> tuple:
    """
    Build combined BQM from all clusters + meta-cluster.

    The meta-cluster allocates weight between asset clusters,
    enabling concentration in high-performing clusters.

    Returns:
        (combined_bqm, cluster_info)
    """
    h_combined = {}
    Q_combined = {}
    cluster_info = []
    cluster_ids = []
    cluster_tickers_map = {}

    # 1. Create BQMs for each asset cluster
    for cluster_id, cluster_tickers in clusters.items():
        cluster_tickers = [t for t in cluster_tickers if t in mu.index]
        if len(cluster_tickers) == 0:
            continue

        cluster_mu = pd.Series(mu[cluster_tickers])
        cluster_Sigma = pd.DataFrame(Sigma.loc[cluster_tickers, cluster_tickers])

        bqm = formulator.formulate_cluster(
            cluster_tickers, cluster_mu, cluster_Sigma
        )

        # ADD BUDGET CONSTRAINT to each asset cluster: penalty for (Σw - 1)²
        # This forces each cluster to allocate exactly 100% total weight
        # Without this, optimizer prefers minimal investment (tiny uniform weights)
        budget_penalty = 50.0  # Strong penalty to enforce budget
        weight_values = formulator.weight_values

        # Build list of cluster variables
        cluster_vars = []
        for ticker in cluster_tickers:
            for q in range(formulator.n_levels):
                var = f"{ticker}_{q}"
                if var in bqm.variables:
                    cluster_vars.append((var, q))

        # Expand (Σw - 1)² = (Σw)² - 2(Σw) + 1
        # Linear terms: -2λ Σw_i
        # Quadratic diagonal (w_i² = w_i for binary): +λ Σw_i
        # Quadratic off-diagonal: +2λ Σᵢ<ⱼ w_i w_j
        for var, q in cluster_vars:
            # Combine linear and diagonal quadratic
            linear_coeff = -2 * budget_penalty * weight_values[q] + budget_penalty * weight_values[q]
            bqm.add_variable(var, linear_coeff)

        # Off-diagonal quadratic terms
        for i, (var_i, q_i) in enumerate(cluster_vars):
            for j, (var_j, q_j) in enumerate(cluster_vars):
                if i < j:  # Only upper triangle
                    coeff = 2 * budget_penalty * weight_values[q_i] * weight_values[q_j]
                    bqm.add_interaction(var_i, var_j, coeff)

        # Merge linear terms
        for var, coeff in bqm.linear.items():
            h_combined[var] = h_combined.get(var, 0.0) + coeff

        # Merge quadratic terms
        for edge, coeff in bqm.quadratic.items():
            Q_combined[edge] = Q_combined.get(edge, 0.0) + coeff

        cluster_info.append({
            'id': cluster_id,
            'tickers': cluster_tickers,
            'variables': list(bqm.variables)
        })

        cluster_ids.append(cluster_id)
        cluster_tickers_map[cluster_id] = cluster_tickers

    # 2. Create meta-cluster BQM (allocates weight between clusters)
    # Compute cluster-level statistics (equal-weighted)
    cluster_mu_series = pd.Series(index=cluster_ids, dtype=float)
    for cluster_id in cluster_ids:
        tickers = cluster_tickers_map[cluster_id]
        cluster_mu_series[cluster_id] = mu[tickers].mean()

    # Compute cluster covariance matrix
    cluster_Sigma = pd.DataFrame(0.0, index=cluster_ids, columns=cluster_ids)
    for i, cluster_i in enumerate(cluster_ids):
        tickers_i = cluster_tickers_map[cluster_i]
        n_i = len(tickers_i)

        for j, cluster_j in enumerate(cluster_ids):
            tickers_j = cluster_tickers_map[cluster_j]
            n_j = len(tickers_j)

            # Equal-weighted covariance between clusters
            cluster_sigma_block = Sigma.loc[tickers_i, tickers_j]
            cov_ij = cluster_sigma_block.values.mean()
            cluster_Sigma.loc[cluster_i, cluster_j] = cov_ij

    # Create meta-cluster BQM
    meta_bqm = formulator.formulate_cluster(
        cluster_ids, cluster_mu_series, cluster_Sigma
    )

    # ADD BUDGET CONSTRAINT to meta-cluster: penalty for (Σw - 1)²
    # This forces the meta-cluster to allocate exactly 100% total weight
    # Without this, the optimizer invests >100% in all clusters
    budget_penalty = 50.0  # Strong penalty to enforce budget
    weight_values = formulator.weight_values

    # Linear term: -2λ * Σw_i
    for cluster_id in cluster_ids:
        for q in range(formulator.n_levels):
            var = f"{cluster_id}_{q}"
            if var in meta_bqm.variables:
                coeff = -2 * budget_penalty * weight_values[q]
                meta_bqm.add_variable(var, coeff)

    # Quadratic terms: λ * (Σw_i)² = λ * Σᵢ Σⱼ wᵢwⱼ
    # For each pair of variables across all clusters
    for i, cluster_i in enumerate(cluster_ids):
        for q_i in range(formulator.n_levels):
            var_i = f"{cluster_i}_{q_i}"
            if var_i not in meta_bqm.variables:
                continue

            for j, cluster_j in enumerate(cluster_ids):
                for q_j in range(formulator.n_levels):
                    var_j = f"{cluster_j}_{q_j}"
                    if var_j not in meta_bqm.variables:
                        continue

                    # Coefficient for w_i * w_j term
                    coeff = budget_penalty * weight_values[q_i] * weight_values[q_j]

                    if i == j and q_i == q_j:
                        # Self term: add to linear
                        meta_bqm.add_variable(var_i, coeff)
                    elif var_i < var_j:  # Avoid double-counting
                        # Cross term: add to quadratic
                        meta_bqm.add_interaction(var_i, var_j, coeff)

    # Constant term: λ * 1² = λ (doesn't affect optimization)

    # Merge meta-cluster into combined BQM
    for var, coeff in meta_bqm.linear.items():
        h_combined[var] = h_combined.get(var, 0.0) + coeff

    for edge, coeff in meta_bqm.quadratic.items():
        Q_combined[edge] = Q_combined.get(edge, 0.0) + coeff

    # Add meta-cluster info
    cluster_info.append({
        'id': 'META_CLUSTER',
        'tickers': cluster_ids,
        'variables': list(meta_bqm.variables)
    })

    combined_bqm = dimod.BinaryQuadraticModel(h_combined, Q_combined, 0.0, dimod.BINARY)

    return combined_bqm, cluster_info


def relabel_bqm_for_template(
    bqm: dimod.BinaryQuadraticModel,
    cluster_info: List[Dict]
) -> tuple:
    """
    Relabel BQM variables to template format.

    Asset clusters: TICKER_LEVEL → C{i}_ASSET_{j}_{level}
    Meta-cluster: cluster_X_LEVEL → META_{X}_{level}

    Returns:
        (relabeled_bqm, reverse_mapping)
    """
    ticker_to_location = {}
    meta_cluster_tickers = []

    for cluster_idx, cluster_data in enumerate(cluster_info):
        cluster_id = cluster_data['id']
        tickers = cluster_data['tickers']

        if cluster_id == 'META_CLUSTER':
            # Meta-cluster: track cluster IDs
            meta_cluster_tickers = tickers
        else:
            # Asset cluster: map tickers to (cluster_idx, asset_idx)
            for asset_idx, ticker in enumerate(tickers):
                ticker_to_location[ticker] = (cluster_idx, asset_idx)

    relabeling = {}
    for var in bqm.variables:
        mapped = False

        # Check if it's a meta-cluster variable (cluster_X_level)
        for meta_idx, cluster_id in enumerate(meta_cluster_tickers):
            if var.startswith(str(cluster_id) + '_'):
                level_str = var[len(str(cluster_id)) + 1:]
                if level_str.isdigit():
                    template_var = f"META_{meta_idx}_{level_str}"
                    relabeling[var] = template_var
                    mapped = True
                    break

        if not mapped:
            # Check if it's an asset variable (TICKER_level)
            for ticker in ticker_to_location.keys():
                if var.startswith(ticker + '_'):
                    level_str = var[len(ticker) + 1:]
                    if level_str.isdigit():
                        cluster_idx, asset_idx = ticker_to_location[ticker]
                        template_var = f"C{cluster_idx}_ASSET_{asset_idx}_{level_str}"
                        relabeling[var] = template_var
                        mapped = True
                        break

        if not mapped:
            relabeling[var] = var

    relabeled_bqm = bqm.relabel_variables(relabeling, inplace=False)
    reverse_mapping = {v: k for k, v in relabeling.items()}

    return relabeled_bqm, reverse_mapping


def validate_bqm_topology(
    bqm: dimod.BinaryQuadraticModel,
    embedding: Dict[str, List[int]]
) -> dimod.BinaryQuadraticModel:
    """
    Ensure BQM topology matches template embedding.

    Removes edges and variables not present in the embedding.
    """
    embedded_vars = set(embedding.keys())

    # Build set of embeddable edges
    embeddable_edges = set()
    embedded_var_list = list(embedded_vars)
    for i, u in enumerate(embedded_var_list):
        for v in embedded_var_list[i+1:]:
            embeddable_edges.add(tuple(sorted([u, v])))

    # Remove edges not in template
    edges_to_remove = [edge for edge in bqm.quadratic if edge not in embeddable_edges]
    for edge in edges_to_remove:
        bqm.remove_interaction(edge[0], edge[1])

    # Remove variables not in embedding
    vars_to_remove = set(bqm.variables) - embedded_vars
    for var in vars_to_remove:
        bqm.remove_variable(var)

    # Add missing variables with zero bias
    missing_vars = embedded_vars - set(bqm.variables)
    for var in missing_vars:
        bqm.add_variable(var, 0.0)

    return bqm


def bqm_to_h_j_format(bqm: dimod.BinaryQuadraticModel) -> Dict[str, Any]:
    """
    Convert BQM to h/J format for SAPI submission.

    Returns dict with 'h' (list) and 'J' (list of [i, j, coupling]) suitable for JSON.
    """
    # Build variable index mapping
    variables = sorted(bqm.variables)
    var_to_idx = {var: idx for idx, var in enumerate(variables)}

    # Extract h (linear coefficients)
    h = [0.0] * len(variables)
    for var, bias in bqm.linear.items():
        h[var_to_idx[var]] = float(bias)

    # Extract J (quadratic coefficients)
    J = []
    for (u, v), coupling in bqm.quadratic.items():
        i = var_to_idx[u]
        j = var_to_idx[v]
        J.append([i, j, float(coupling)])

    return {
        'h': h,
        'J': J,
        'variable_mapping': {var: idx for var, idx in var_to_idx.items()}
    }


def generate_job_payload(
    returns_train: pd.DataFrame,
    template_params: Dict[str, Any],
    embedding: Dict,
    formulator: DiscreteLevelFormulator,
    num_reads: int
) -> tuple:
    """Generate a single job payload from training data."""
    # Compute mean returns and covariance from training data
    mu = returns_train.mean() * 252  # Annualized
    Sigma = returns_train.cov() * 252

    # Create clusters
    clusters = create_clusters(
        returns=returns_train,
        max_cluster_size=template_params['assets_per_cluster'],
        target_cluster_size=template_params['assets_per_cluster']
    )

    # Pad clusters to match template
    padded_clusters, padded_mu, padded_Sigma, real_tickers = pad_clusters_to_template(
        clusters=clusters,
        mu=mu,
        Sigma=Sigma,
        expected_assets_per_cluster=template_params['assets_per_cluster'],
        expected_n_clusters=template_params['cluster_size']
    )

    # Build BQM
    combined_bqm, cluster_info = build_combined_bqm(
        clusters=padded_clusters,
        mu=padded_mu,
        Sigma=padded_Sigma,
        formulator=formulator
    )

    # Relabel for template
    relabeled_bqm, reverse_mapping = relabel_bqm_for_template(
        bqm=combined_bqm,
        cluster_info=cluster_info
    )

    # Validate topology
    validated_bqm = validate_bqm_topology(
        bqm=relabeled_bqm,
        embedding=embedding
    )

    # Convert to h/J format
    hj_data = bqm_to_h_j_format(validated_bqm)

    # Create job payload
    job_payload = {
        'h': hj_data['h'],
        'J': hj_data['J'],
        'num_samples': num_reads
    }

    metadata = {
        'variable_mapping': hj_data['variable_mapping'],
        'reverse_mapping': reverse_mapping,
        'real_tickers': list(real_tickers),
        'cluster_info': cluster_info
    }

    return job_payload, metadata


def generate_backtest_schedule(args):
    """Generate multiple job payloads for backtest schedule."""
    print("="*80)
    print("QUANTUM ANNEALING BACKTEST JOB PREPARATION")
    print("="*80)

    # 1. Load portfolio data
    print(f"\n[1/4] Loading portfolio data from: {args.portfolio_csv}")
    prices, returns = load_portfolio_data(args.portfolio_csv, preprocess=False)
    n_assets = len(returns.columns)
    print(f"  ✓ Loaded {n_assets} assets, {len(prices)} days")
    print(f"  Date range: {prices.index[0]} to {prices.index[-1]}")

    # 2. Select optimal template
    print(f"\n[2/4] Selecting optimal embedding template...")
    template_params = select_optimal_template(
        num_assets=n_assets,
        n_levels=args.n_levels
    )

    if template_params is None:
        print("  ✗ No suitable template found for portfolio size")
        sys.exit(1)

    print(f"  ✓ Selected template: {template_params['template_name']}")
    print(f"    Clusters: {template_params['cluster_size']}")
    print(f"    Assets per cluster: {template_params['assets_per_cluster']}")
    print(f"    Levels: {template_params['num_levels']}")
    print(f"    Capacity: {template_params['capacity']} (waste: {template_params['waste']})")

    # Load template embedding
    template_data = load_embedding_template(template_params['template_name'])
    embedding = template_data['embedding']

    # 3. Create formulator
    print(f"\n[3/4] Creating BQM formulator...")
    formulator = DiscreteLevelFormulator(
        n_levels=template_params['num_levels'],
        alpha=args.alpha,
        beta=args.beta,
        thermometer_penalty=args.thermometer_penalty,
        l1_sparsity_penalty=args.l1_penalty
    )
    print(f"  ✓ Formulator configured (α={args.alpha}, β={args.beta})")

    # 4. Generate backtest schedule
    print(f"\n[4/4] Generating backtest schedule...")
    print(f"  Training window: {args.train_days} days")
    print(f"  Test period: {args.test_days} days")
    print(f"  Rebalancing: every {args.step_days} days")

    # Calculate periods
    total_days = len(returns)
    n_periods = (total_days - args.train_days - args.test_days) // args.step_days + 1
    print(f"  Expected periods: {n_periods}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate jobs for each period
    curl_commands = []
    period_idx = 0

    for start_idx in range(0, total_days - args.train_days - args.test_days + 1, args.step_days):
        train_end_idx = start_idx + args.train_days
        test_end_idx = train_end_idx + args.test_days

        if test_end_idx > total_days:
            break

        # Extract training data
        returns_train = returns.iloc[start_idx:train_end_idx]

        # Generate job payload
        job_payload, metadata = generate_job_payload(
            returns_train=returns_train,
            template_params=template_params,
            embedding=embedding,
            formulator=formulator,
            num_reads=args.num_reads
        )

        # Save job file
        if period_idx == 0:
            job_filename = "train.json"
        else:
            job_filename = f"period-{period_idx}.json"

        job_path = output_dir / job_filename
        with open(job_path, 'w') as f:
            json.dump(job_payload, f, indent=2)

        # Save metadata
        metadata_full = {
            'portfolio_csv': args.portfolio_csv,
            'period': period_idx,
            'train_start_date': str(returns.index[start_idx]),
            'train_end_date': str(returns.index[train_end_idx - 1]),
            'test_start_date': str(returns.index[train_end_idx]),
            'test_end_date': str(returns.index[test_end_idx - 1]),
            'n_assets': n_assets,
            'template': template_params['template_name'],
            **metadata,
            'parameters': {
                'n_levels': template_params['num_levels'],
                'alpha': args.alpha,
                'beta': args.beta,
                'thermometer_penalty': args.thermometer_penalty,
                'l1_penalty': args.l1_penalty,
                'num_reads': args.num_reads
            }
        }

        metadata_path = output_dir / job_filename.replace('.json', '.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata_full, f, indent=2)

        # Generate curl command
        curl_cmd = f"curl -X POST https://qpu-1.nodes.quip.network:20049/solve -H \"Content-Type: application/json\" -d @{job_path.name}"
        curl_commands.append((job_filename, curl_cmd))

        print(f"  ✓ Generated {job_filename} (period {period_idx}, train: {returns.index[start_idx].date()} to {returns.index[train_end_idx-1].date()})")

        period_idx += 1

    # Save curl commands script
    script_path = output_dir / "submit_all.sh"
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated submission script for backtest jobs\n\n")
        f.write(f"cd {output_dir.absolute()}\n\n")
        for job_filename, curl_cmd in curl_commands:
            f.write(f"echo 'Submitting {job_filename}...'\n")
            f.write(f"{curl_cmd}\n")
            f.write("echo ''\n\n")

    script_path.chmod(0o755)

    print(f"\n  ✓ Generated {period_idx} job payloads")
    print(f"  ✓ Output directory: {output_dir.absolute()}")

    print("\n" + "="*80)
    print("READY FOR SUBMISSION")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - train.json")
    for i in range(1, period_idx):
        print(f"  - period-{i}.json")
    print(f"  - *.metadata.json (metadata for each job)")
    print(f"  - submit_all.sh (batch submission script)")

    print(f"\nSubmit individual jobs:")
    print(f"  cd {output_dir}")
    for job_filename, curl_cmd in curl_commands[:3]:  # Show first 3
        print(f"  {curl_cmd}")
    if len(curl_commands) > 3:
        print(f"  ... ({len(curl_commands) - 3} more)")

    print(f"\nOr submit all at once:")
    print(f"  cd {output_dir} && ./submit_all.sh")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Prepare quantum annealing job payloads for D-Wave SAPI (backtest schedule)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python tools/prepare_qa_job.py \\
        --portfolio-csv portfolio.csv \\
        --output-dir backtest_jobs \\
        --train-days 252 \\
        --test-days 21 \\
        --step-days 21 \\
        --num-reads 100

This generates:
    backtest_jobs/train.json
    backtest_jobs/period-1.json
    backtest_jobs/period-2.json
    ...
    backtest_jobs/submit_all.sh
        """
    )

    parser.add_argument('--portfolio-csv', type=str, required=True,
                       help='Path to portfolio price CSV file')
    parser.add_argument('--portfolio-info-csv', type=str, default=None,
                       help='Path to portfolio info CSV (for sector data)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for job payloads')

    # Backtest parameters
    parser.add_argument('--train-days', type=int, default=252,
                       help='Training window size in days (default: 252 = 1 year)')
    parser.add_argument('--test-days', type=int, default=21,
                       help='Holding period in days (default: 21 = 1 month)')
    parser.add_argument('--step-days', type=int, default=21,
                       help='Rebalancing frequency in days (default: 21)')

    parser.add_argument('--num-reads', type=int, default=256,
                       help='Number of annealing samples (default: 256, matches quantum_classical_comparison.py)')
    parser.add_argument('--n-levels', type=int, default=None,
                       help='Number of discrete weight levels (auto-selected if not specified)')
    parser.add_argument('--alpha', type=float, default=20.0,
                       help='Return objective coefficient (default: 20.0, matches benchmark)')
    parser.add_argument('--beta', type=float, default=2.0,
                       help='Risk objective coefficient (default: 2.0, matches benchmark)')
    parser.add_argument('--thermometer-penalty', type=float, default=20.0,
                       help='Thermometer constraint penalty (default: 20.0, REQUIRED for valid encoding)')
    parser.add_argument('--l1-penalty', type=float, default=1.0,
                       help='L1 sparsity penalty (default: 1.0, promotes sparsity)')

    args = parser.parse_args()

    generate_backtest_schedule(args)


if __name__ == '__main__':
    main()
