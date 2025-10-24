#!/usr/bin/env python3
"""
Find optimal maximal template by testing different cluster configurations.

Strategy: For each num_levels in [4, 6, 8], incrementally search for the maximum
cluster_size where num_clusters = assets_per_cluster = cluster_size.
This means num_assets = cluster_size^2.

Run with: python tools/find_optimal_template.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dwave.system import DWaveSampler
import minorminer
from qpo.qubo.discrete_levels import DiscreteLevelFormulator
from dotenv import load_dotenv
import json
import time

load_dotenv()

def create_test_bqm(n_clusters, assets_per_cluster, n_levels=10):
    """Create a test BQM with specified configuration.

    Args:
        n_clusters: Number of asset clusters
        assets_per_cluster: Assets per cluster
        n_levels: Number of discrete levels
    """
    formulator = DiscreteLevelFormulator(n_levels=n_levels)

    # Generate dummy data
    np.random.seed(42)

    # Create asset clusters WITHOUT budget constraint (matches optimizer)
    combined_bqm = None
    all_assets = []

    for cluster_id in range(n_clusters):
        # Create dummy cluster
        cluster_tickers = [f"C{cluster_id}_ASSET_{i}" for i in range(assets_per_cluster)]
        all_assets.extend(cluster_tickers)

        # Dummy returns and covariance
        mu = pd.Series(np.random.randn(assets_per_cluster) * 0.1 + 0.1, index=cluster_tickers)

        corr = np.random.rand(assets_per_cluster, assets_per_cluster) * 0.6 - 0.3
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)

        vols = np.random.rand(assets_per_cluster) * 0.1 + 0.1
        Sigma = pd.DataFrame(
            corr * np.outer(vols, vols),
            index=cluster_tickers,
            columns=cluster_tickers
        )

        # No budget constraint (sparse graph)
        bqm = formulator.formulate_cluster(cluster_tickers, mu, Sigma, include_budget_constraint=False)

        if combined_bqm is None:
            combined_bqm = bqm
        else:
            combined_bqm.update(bqm)

    # Add meta-cluster WITHOUT budget constraint (matches optimizer)
    meta_tickers = [f"META_{i}" for i in range(n_clusters)]
    meta_mu = pd.Series(np.random.randn(n_clusters) * 0.08 + 0.1, index=meta_tickers)

    meta_corr = np.random.rand(n_clusters, n_clusters) * 0.4 - 0.2
    meta_corr = (meta_corr + meta_corr.T) / 2
    np.fill_diagonal(meta_corr, 1.0)

    meta_vols = np.random.rand(n_clusters) * 0.06 + 0.08
    meta_Sigma = pd.DataFrame(
        meta_corr * np.outer(meta_vols, meta_vols),
        index=meta_tickers,
        columns=meta_tickers
    )

    # No budget constraint (sparse graph)
    meta_bqm = formulator.formulate_cluster(meta_tickers, meta_mu, meta_Sigma, include_budget_constraint=False)
    combined_bqm.update(meta_bqm)

    return combined_bqm, len(all_assets)

def test_configuration(n_clusters, assets_per_cluster, n_levels, sampler, timeout=600):
    """Test if a configuration can be embedded.

    Args:
        n_clusters: Number of asset clusters
        assets_per_cluster: Assets per cluster
        n_levels: Number of discrete levels
        sampler: D-Wave sampler
        timeout: Embedding timeout in seconds
    """

    total_assets = n_clusters * assets_per_cluster

    print(f"\n{'='*70}", flush=True)
    print(f"TESTING: {n_clusters} clusters × {assets_per_cluster} assets/cluster = {total_assets} total assets × {n_levels} levels", flush=True)
    print(f"{'='*70}", flush=True)

    start = time.time()

    # Create BQM
    print(f"[{time.strftime('%H:%M:%S')}] Creating BQM...", flush=True)
    bqm, actual_assets = create_test_bqm(n_clusters, assets_per_cluster, n_levels=n_levels)

    print(f"  Variables: {len(bqm.variables)}", flush=True)
    print(f"  Quadratic: {len(bqm.quadratic)}", flush=True)
    print(f"  Total assets: {actual_assets}", flush=True)
    print(f"  n_levels: {n_levels}", flush=True)

    # Sanity checks before attempting embedding
    source_edgelist = list(bqm.quadratic.keys())
    target_edgelist = sampler.edgelist

    n_source_vars = len(bqm.variables)
    n_source_edges = len(bqm.quadratic)
    n_target_qubits = sampler.properties['num_qubits']
    n_target_couplers = len(sampler.edgelist)

    print(f"\n[{time.strftime('%H:%M:%S')}] Running sanity checks...", flush=True)

    # Check 1: Minimum qubit requirement (with best-case chain length of 1)
    if n_source_vars > n_target_qubits:
        print(f"  ✗ SANITY CHECK FAILED: Need at least {n_source_vars} qubits, but only {n_target_qubits} available", flush=True)
        print(f"  Skipping embedding attempt (impossible to embed)", flush=True)
        return None

    # Check 2: Graph density - if source is much denser than target, likely to fail
    source_density = n_source_edges / (n_source_vars * (n_source_vars - 1) / 2) if n_source_vars > 1 else 0
    target_density = n_target_couplers / (n_target_qubits * (n_target_qubits - 1) / 2) if n_target_qubits > 1 else 0

    print(f"  Source graph: {n_source_vars} vars, {n_source_edges} edges (density: {source_density:.4f})", flush=True)
    print(f"  Target graph: {n_target_qubits} qubits, {n_target_couplers} couplers (density: {target_density:.4f})", flush=True)

    # Heuristic: If we need average chain length > 10 just to fit all variables, likely too large
    estimated_min_chain_length = n_source_vars / n_target_qubits
    if estimated_min_chain_length > 10:
        print(f"  ⚠ WARNING: Would need avg chain length > {estimated_min_chain_length:.1f} just to fit variables", flush=True)
        print(f"  This is likely too large, but attempting anyway...", flush=True)

    # Check 3: Rough heuristic - if source has more edges than target has couplers, likely problematic
    if n_source_edges > n_target_couplers:
        print(f"  ⚠ WARNING: Source has {n_source_edges} edges but target only has {n_target_couplers} couplers", flush=True)
        print(f"  Embedding may be difficult or impossible", flush=True)

    print(f"  ✓ Basic sanity checks passed", flush=True)

    # Attempt embedding
    print(f"\n[{time.strftime('%H:%M:%S')}] Attempting embedding (timeout: {timeout}s = {timeout//60}min)...", flush=True)

    try:
        embedding = minorminer.find_embedding(
            source_edgelist,
            target_edgelist,
            timeout=timeout,
            max_no_improvement=timeout // 5,
            verbose=1,
            tries=10,
            chainlength_patience=500,
        )

        elapsed = time.time() - start

        if not embedding:
            print(f"[{time.strftime('%H:%M:%S')}] ✗ FAILED (no embedding found)", flush=True)
            return None

        # Success! Compute stats
        chain_lengths = [len(chain) for chain in embedding.values()]
        all_qubits = [q for chain in embedding.values() for q in chain]
        total_qubits = len(set(all_qubits))

        print(f"[{time.strftime('%H:%M:%S')}] ✓ SUCCESS! ({elapsed/60:.1f} minutes)", flush=True)
        print(f"\n  Embedding Statistics:", flush=True)
        print(f"    Qubits used: {total_qubits}/{sampler.properties['num_qubits']} ({100*total_qubits/sampler.properties['num_qubits']:.1f}%)", flush=True)
        print(f"    Avg chain: {np.mean(chain_lengths):.2f}", flush=True)
        print(f"    Max chain: {max(chain_lengths)}", flush=True)

        return {
            'n_clusters': n_clusters,
            'assets_per_cluster': assets_per_cluster,
            'total_assets': actual_assets,
            'n_levels': n_levels,
            'embedding': {str(k): list(v) for k, v in embedding.items()},  # JSON-serializable
            'n_variables': len(bqm.variables),
            'n_quadratic': len(bqm.quadratic),
            'total_qubits': total_qubits,
            'avg_chain_length': float(np.mean(chain_lengths)),
            'max_chain_length': int(max(chain_lengths)),
            'min_chain_length': int(min(chain_lengths)),
            'min_qubit': int(min(all_qubits)),
            'max_qubit': int(max(all_qubits)),
            'embedding_time': elapsed,
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"[{time.strftime('%H:%M:%S')}] ✗ ERROR: {e}", flush=True)
        return None

def find_max_cluster_size_for_levels(n_levels, sampler, timeout=600, start_cluster_size=8):
    """Two-phase search to find maximum cluster configuration for a given n_levels.

    Phase 1: Find max cluster_size where num_clusters = assets_per_cluster (square config)
    Phase 2: Keep optimal num_clusters, increase assets_per_cluster by 1 until failure

    Args:
        n_levels: Number of discrete levels to test
        sampler: D-Wave sampler
        timeout: Embedding timeout in seconds
        start_cluster_size: Starting cluster size for Phase 1

    Returns:
        list of result dicts for all successful configurations
    """
    print(f"\n{'='*70}", flush=True)
    print(f"SEARCHING FOR MAX CLUSTER CONFIGURATIONS WITH n_levels={n_levels}", flush=True)
    print(f"{'='*70}", flush=True)

    all_successful_configs = []

    # PHASE 1: Find optimal num_clusters (where num_clusters = assets_per_cluster)
    print(f"\n{'='*70}", flush=True)
    print(f"PHASE 1: Finding optimal num_clusters (square configuration)", flush=True)
    print(f"Constraint: num_clusters = assets_per_cluster = cluster_size", flush=True)
    print(f"{'='*70}", flush=True)

    cluster_size = start_cluster_size
    last_successful = None

    # Find upper bound by doubling
    print(f"\nPhase 1a: Finding upper bound...", flush=True)
    while True:
        result = test_configuration(cluster_size, cluster_size, n_levels, sampler, timeout)
        if result:
            total = cluster_size * cluster_size
            print(f"  ✓ cluster_size={cluster_size} ({total} total assets): SUCCESS", flush=True)
            last_successful = result
            all_successful_configs.append(result)
            cluster_size *= 2
        else:
            total = cluster_size * cluster_size
            print(f"  ✗ cluster_size={cluster_size} ({total} total assets): FAILED", flush=True)
            print(f"  Upper bound found: cluster_size={cluster_size}", flush=True)
            break

    if last_successful is None:
        print(f"\n✗ Could not embed even cluster_size={start_cluster_size}", flush=True)
        return []

    # Binary search for exact maximum
    lower = last_successful['n_clusters']
    upper = cluster_size

    print(f"\nPhase 1b: Binary search between {lower} and {upper}...", flush=True)

    while lower < upper - 1:
        mid = (lower + upper) // 2
        result = test_configuration(mid, mid, n_levels, sampler, timeout)

        if result:
            total = mid * mid
            print(f"  ✓ cluster_size={mid} ({total} total assets): SUCCESS", flush=True)
            last_successful = result
            all_successful_configs.append(result)
            lower = mid
        else:
            total = mid * mid
            print(f"  ✗ cluster_size={mid} ({total} total assets): FAILED", flush=True)
            upper = mid

    optimal_n_clusters = last_successful['n_clusters']
    optimal_assets_per_cluster = last_successful['assets_per_cluster']

    print(f"\n{'='*70}", flush=True)
    print(f"PHASE 1 COMPLETE: Optimal num_clusters = {optimal_n_clusters}", flush=True)
    print(f"  Configuration: {optimal_n_clusters} clusters × {optimal_assets_per_cluster} assets = {optimal_n_clusters * optimal_assets_per_cluster} total assets", flush=True)
    print(f"{'='*70}", flush=True)

    # PHASE 2: Keep num_clusters fixed, increase assets_per_cluster by 1 until failure
    print(f"\n{'='*70}", flush=True)
    print(f"PHASE 2: Increasing assets_per_cluster (keep num_clusters={optimal_n_clusters})", flush=True)
    print(f"Meta-cluster constraint: assets_per_cluster must not exceed num_clusters", flush=True)
    print(f"{'='*70}", flush=True)

    current_assets_per_cluster = optimal_assets_per_cluster + 1

    while True:
        # Meta-cluster cannot have more assets than there are clusters
        # (meta-cluster has num_clusters assets, one per cluster)
        if current_assets_per_cluster > optimal_n_clusters:
            print(f"\n✗ Stopping: assets_per_cluster={current_assets_per_cluster} would exceed num_clusters={optimal_n_clusters}", flush=True)
            print(f"  Meta-cluster cannot be encoded (needs {optimal_n_clusters} levels but assets have {current_assets_per_cluster} levels)", flush=True)
            break

        result = test_configuration(optimal_n_clusters, current_assets_per_cluster, n_levels, sampler, timeout)

        if result:
            total = optimal_n_clusters * current_assets_per_cluster
            print(f"  ✓ {optimal_n_clusters} clusters × {current_assets_per_cluster} assets ({total} total): SUCCESS", flush=True)
            all_successful_configs.append(result)
            current_assets_per_cluster += 1
        else:
            total = optimal_n_clusters * current_assets_per_cluster
            print(f"  ✗ {optimal_n_clusters} clusters × {current_assets_per_cluster} assets ({total} total): FAILED", flush=True)
            print(f"  Maximum assets_per_cluster found: {current_assets_per_cluster - 1}", flush=True)
            break

    print(f"\n{'='*70}", flush=True)
    print(f"PHASE 2 COMPLETE: Found {len(all_successful_configs)} embeddable configurations", flush=True)
    print(f"{'='*70}", flush=True)

    return all_successful_configs


def find_optimal():
    """Find optimal template configurations."""

    print("="*70, flush=True)
    print("FINDING OPTIMAL MAXIMAL TEMPLATES", flush=True)
    print("="*70, flush=True)
    print(f"\nStarted: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"\nStrategy: For each n_levels in [4, 6, 8], find maximum cluster_size", flush=True)
    print(f"Constraint: num_clusters = assets_per_cluster = cluster_size", flush=True)
    print(f"Timeout: 600 seconds (10 minutes) per embedding attempt", flush=True)

    # Get QPU
    print(f"\nConnecting to QPU...", flush=True)
    sampler = DWaveSampler(solver='Advantage2_system1.6')
    print(f"  Solver: {sampler.properties['chip_id']}", flush=True)
    print(f"  Topology: {sampler.properties.get('topology', {}).get('type', 'zephyr')}", flush=True)
    print(f"  Working qubits: {len(sampler.nodelist)}", flush=True)

    # Test n_levels in order: 4, 6, 8
    n_levels_list = [4, 6, 8]
    timeout = 600  # 10 minutes

    # Store all successful results
    all_results = []

    for n_levels in n_levels_list:
        configs = find_max_cluster_size_for_levels(n_levels, sampler, timeout, start_cluster_size=8)

        if configs:
            all_results.extend(configs)
        else:
            print(f"\n✗ Failed to find any embeddable configuration for n_levels={n_levels}", flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print(f"RESULTS SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    if not all_results:
        print(f"\n✗ No configurations succeeded.", flush=True)
        return []

    print(f"\nTotal successful configurations: {len(all_results)}", flush=True)

    # Group by n_levels for summary
    by_levels = {}
    for result in all_results:
        n_lev = result['n_levels']
        if n_lev not in by_levels:
            by_levels[n_lev] = []
        by_levels[n_lev].append(result)

    for n_levels in sorted(by_levels.keys()):
        configs = by_levels[n_levels]
        print(f"\n  n_levels = {n_levels}: {len(configs)} configurations", flush=True)
        for cfg in configs:
            print(f"    {cfg['n_clusters']} clusters × {cfg['assets_per_cluster']} assets = {cfg['total_assets']} total", flush=True)
            print(f"      Variables: {cfg['n_variables']}, Qubits: {cfg['total_qubits']} ({cfg['total_qubits']/sampler.properties['num_qubits']*100:.1f}%)", flush=True)


    # Save all successful templates
    print(f"\n{'='*70}", flush=True)
    print(f"SAVING TEMPLATES", flush=True)
    print(f"{'='*70}", flush=True)

    output_dir = Path("embeddings/templates")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute sparsification signature (hash of sparsification parameters)
    import hashlib
    sparsif_str = f"cov_cutoff=0.0,diag_risk=0.0,risk_abs=0.0,risk_pct=None,max_deg=None,preserve_budget=True"
    sparsif_sig = hashlib.sha256(sparsif_str.encode()).hexdigest()[:16]

    # Get topology information
    topology_type = sampler.properties.get('topology', {}).get('type', 'zephyr')
    topology_family = topology_type.lower() if topology_type else 'zephyr'

    saved_files = []
    for result in all_results:
        template_data = {
            'version': '1.0.0',
            'formulator': 'discrete_levels',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'generator_version': 'tools/find_optimal_template.py@v2.0',
            'configuration': {
                'n_asset_clusters': result['n_clusters'],
                'n_meta_clusters': 1,
                'n_clusters_total': result['n_clusters'] + 1,
                'assets_per_cluster': result['assets_per_cluster'],
                'total_assets': result['total_assets'],
                'n_levels': result['n_levels'],
                'alpha': 5.0,
                'beta': 2.5,
            },
            'topology': {
                'name': 'Advantage2_system1.6',
                'family': topology_family,
                'chip_id': sampler.properties['chip_id'],
                'num_qubits': len(sampler.nodelist),
                'num_couplers': len(sampler.edgelist),
            },
            'shape': {
                'n_clusters': result['n_clusters'] + 1,  # Including meta-cluster
                'assets_per_cluster': result['assets_per_cluster'],
                'n_levels': result['n_levels'],
            },
            'sparsification_signature': sparsif_sig,
            'embedding': result['embedding'],
            'statistics': {
                'n_variables': result['n_variables'],
                'n_quadratic': result['n_quadratic'],
                'total_qubits_used': result['total_qubits'],
                'qpu_utilization_pct': float(result['total_qubits'] / len(sampler.nodelist) * 100),
                'avg_chain_length': result['avg_chain_length'],
                'max_chain_length': result['max_chain_length'],
                'min_chain_length': result.get('min_chain_length', result['avg_chain_length']),
                'min_qubit': result['min_qubit'],
                'max_qubit': result['max_qubit'],
                'qubit_span': result['max_qubit'] - result['min_qubit'],
                'embedding_time_seconds': result['embedding_time'],
            }
        }

        # Standardized filename: portfolio_{C}c_{A}a_{L}l.json
        filename = f"portfolio_{result['n_clusters']}c_{result['total_assets']}a_{result['n_levels']}l.json"
        output_file = output_dir / filename

        with open(output_file, 'w') as f:
            json.dump(template_data, f, indent=2)

        saved_files.append(output_file)
        print(f"  ✓ Saved: {filename}", flush=True)
        print(f"      {result['total_assets']} assets, {result['total_qubits']} qubits ({result['total_qubits']/len(sampler.nodelist)*100:.1f}%)", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"COMPLETE", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\nFinished: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Total templates saved: {len(saved_files)}", flush=True)

    return all_results

if __name__ == '__main__':
    find_optimal()
