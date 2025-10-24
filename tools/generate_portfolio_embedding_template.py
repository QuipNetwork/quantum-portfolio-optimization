#!/usr/bin/env python3
"""
Generate portfolio optimization embedding template for D-Wave QPU.

This tool creates a pre-computed embedding template for multi-cluster portfolio
optimization with meta-cluster support. The template uses discrete-level encoding
with thermometer constraints.

The template includes:
  - N asset clusters (specified by user, each with M assets)
  - 1 meta-cluster (automatically added, with N "mega-assets")

Note: You specify the number of ASSET clusters. The meta-cluster is added automatically.

Usage:
    # Generate template from live QPU
    python tools/generate_portfolio_embedding_template.py \\
        --solver Advantage2_system1.6 \\
        --n-levels 10 \\
        --num-clusters 12 \\
        --cluster-size 10 \\
        --output embeddings/templates/portfolio_12c_10a_10l.json

    # Generate from saved topology
    python tools/generate_portfolio_embedding_template.py \\
        --topology topologies/advantage2.json \\
        --n-levels 10 \\
        --num-clusters 12 \\
        --cluster-size 10 \\
        --output embeddings/templates/portfolio_12c_10a_10l.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import minorminer
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from qpo.qubo.discrete_levels import DiscreteLevelFormulator

load_dotenv()


def load_topology(topology_path: Path) -> Tuple[List, List]:
    """
    Load topology from JSON file.

    Returns:
        (nodelist, edgelist)
    """
    with open(topology_path, 'r') as f:
        data = json.load(f)

    nodelist = data['nodes']
    edgelist = [tuple(edge) for edge in data['edges']]

    return nodelist, edgelist


def get_live_topology(solver_name: str) -> Tuple[List, List, Dict]:
    """
    Get topology from live D-Wave solver.

    Returns:
        (nodelist, edgelist, metadata)
    """
    from dwave.system import DWaveSampler

    print(f"Connecting to solver: {solver_name}...", flush=True)
    sampler = DWaveSampler(solver=solver_name)

    metadata = {
        'solver': solver_name,
        'chip_id': sampler.properties.get('chip_id'),
        'topology_type': sampler.properties.get('topology', {}).get('type'),
        'num_qubits': len(sampler.nodelist),
        'num_couplers': len(sampler.edgelist),
    }

    print(f"Connected: {metadata['chip_id']} ({metadata['topology_type']})", flush=True)
    print(f"  Working qubits: {metadata['num_qubits']}", flush=True)

    return sampler.nodelist, sampler.edgelist, metadata


def create_dummy_portfolio_bqm(
    n_asset_clusters: int,
    assets_per_cluster: int,
    n_levels: int,
    alpha: float = 5.0,
    beta: float = 2.5
) -> Tuple:
    """
    Create a dummy portfolio BQM for embedding (discrete levels formulator).

    This creates N asset clusters + 1 meta-cluster automatically.

    Args:
        n_asset_clusters: Number of asset clusters (user-facing count)
        assets_per_cluster: Assets per cluster
        n_levels: Number of discrete levels
        alpha: Return coefficient
        beta: Risk coefficient

    Returns:
        (bqm, asset_variables, meta_variables, total_clusters_including_meta)
    """
    formulator = DiscreteLevelFormulator(n_levels=n_levels, alpha=alpha, beta=beta)

    np.random.seed(42)  # Reproducible dummy data

    # Build combined BQM using manual dict merging (avoid .update() bugs)
    h_combined = {}
    Q_combined = {}
    asset_variables = []

    # Create asset clusters
    for cluster_id in range(n_asset_clusters):
        cluster_tickers = [f"ASSET_{cluster_id}_{i}" for i in range(assets_per_cluster)]
        asset_variables.extend(cluster_tickers)

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

        # NOTE: include_budget_constraint=False to match runtime optimizer
        # Budget constraints create dense O(N²×L²) couplings that are hard to embed.
        # Post-processing normalizes weights, so budget constraints aren't needed.
        bqm = formulator.formulate_cluster(cluster_tickers, mu, Sigma, include_budget_constraint=False)

        # Manually merge linear terms
        for var, coeff in bqm.linear.items():
            h_combined[var] = h_combined.get(var, 0.0) + coeff

        # Manually merge quadratic terms
        for edge, coeff in bqm.quadratic.items():
            Q_combined[edge] = Q_combined.get(edge, 0.0) + coeff

    # Create meta-cluster (automatically added)
    meta_tickers = [f"META_{i}" for i in range(n_asset_clusters)]
    meta_mu = pd.Series(np.random.randn(n_asset_clusters) * 0.08 + 0.1, index=meta_tickers)

    meta_corr = np.random.rand(n_asset_clusters, n_asset_clusters) * 0.4 - 0.2
    meta_corr = (meta_corr + meta_corr.T) / 2
    np.fill_diagonal(meta_corr, 1.0)

    meta_vols = np.random.rand(n_asset_clusters) * 0.06 + 0.08
    meta_Sigma = pd.DataFrame(
        meta_corr * np.outer(meta_vols, meta_vols),
        index=meta_tickers,
        columns=meta_tickers
    )

    meta_bqm = formulator.formulate_cluster(meta_tickers, meta_mu, meta_Sigma, include_budget_constraint=False)

    # Manually merge meta-cluster
    for var, coeff in meta_bqm.linear.items():
        h_combined[var] = h_combined.get(var, 0.0) + coeff
    for edge, coeff in meta_bqm.quadratic.items():
        Q_combined[edge] = Q_combined.get(edge, 0.0) + coeff

    # Create combined BQM from merged dicts
    import dimod
    combined_bqm = dimod.BinaryQuadraticModel(h_combined, Q_combined, 0.0, dimod.BINARY)

    # Total clusters = N asset clusters + 1 meta cluster
    total_clusters = n_asset_clusters + 1

    return combined_bqm, asset_variables, meta_tickers, total_clusters




def find_embedding(
    bqm,
    target_nodelist: List,
    target_edgelist: List,
    timeout: int = 3600,
    tries: int = 10,
    chainlength_patience: int = 500,
    verbose: bool = True
) -> Optional[Dict]:
    """
    Find embedding using minorminer.

    Returns:
        Embedding dict or None if failed
    """
    source_edgelist = list(bqm.quadratic.keys())

    if verbose:
        print(f"\nFinding embedding...", flush=True)
        print(f"  Source: {len(bqm.variables)} variables, {len(source_edgelist)} edges", flush=True)
        print(f"  Target: {len(target_nodelist)} qubits, {len(target_edgelist)} couplers", flush=True)
        print(f"  Timeout: {timeout}s, Tries: {tries}", flush=True)

    start = time.time()

    try:
        embedding = minorminer.find_embedding(
            source_edgelist,
            target_edgelist,
            timeout=timeout,
            max_no_improvement=timeout // 5,
            verbose=1 if verbose else 0,
            tries=tries,
            chainlength_patience=chainlength_patience,
        )

        elapsed = time.time() - start

        if not embedding:
            if verbose:
                print(f"✗ Embedding failed (no solution found)", flush=True)
            return None

        # Convert to plain dict with list chains
        embedding_dict = {}
        for var, chain in embedding.items():
            if isinstance(chain, (list, tuple)):
                embedding_dict[var] = list(chain)
            else:
                embedding_dict[var] = [chain]

        if verbose:
            chain_lengths = [len(chain) for chain in embedding_dict.values()]
            all_qubits = [q for chain in embedding_dict.values() for q in chain]
            total_qubits = len(set(all_qubits))

            print(f"\n✓ Embedding found! ({elapsed/60:.1f} minutes)", flush=True)
            print(f"  Qubits used: {total_qubits}/{len(target_nodelist)} ({100*total_qubits/len(target_nodelist):.1f}%)", flush=True)
            print(f"  Avg chain length: {np.mean(chain_lengths):.2f}", flush=True)
            print(f"  Max chain length: {max(chain_lengths)}", flush=True)

        return embedding_dict

    except Exception as e:
        if verbose:
            print(f"✗ Embedding error: {e}", flush=True)
        return None


def generate_template(
    formulator_type: str,
    n_clusters: int,
    assets_per_cluster: int,
    n_levels_or_bits: int,
    topology_source: str,  # 'live' or path to JSON
    solver_name: Optional[str] = None,
    timeout: int = 3600,
    tries: int = 10,
    chainlength_patience: int = 500
) -> Optional[Dict]:
    """
    Generate portfolio embedding template.

    Args:
        formulator_type: 'discrete_levels' or 'qubo_bits'
        n_clusters: Number of clusters (for discrete_levels: asset clusters, meta added automatically)
        assets_per_cluster: Assets per cluster
        n_levels_or_bits: Number of discrete levels (discrete_levels) or bits (qubo_bits)
        topology_source: 'live' or path to topology JSON
        solver_name: Solver name if using live topology
        timeout: Embedding timeout
        tries: Number of embedding attempts
        chainlength_patience: Patience parameter

    Returns:
        Template data dict or None if failed
    """
    print(f"\n{'='*70}")
    print(f"GENERATING PORTFOLIO EMBEDDING TEMPLATE")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  Formulator: {formulator_type}")

    # Get topology
    if topology_source == 'live':
        target_nodelist, target_edgelist, topology_metadata = get_live_topology(solver_name)
    else:
        print(f"\nLoading topology from: {topology_source}", flush=True)
        target_nodelist, target_edgelist = load_topology(Path(topology_source))
        topology_metadata = {
            'source': 'file',
            'file': str(topology_source),
            'num_qubits': len(target_nodelist),
            'num_couplers': len(target_edgelist),
        }

    # Create BQM based on formulator type
    print(f"\nCreating BQM...", flush=True)

    if formulator_type == 'discrete_levels':
        print(f"  Asset clusters: {n_clusters} (specified)")
        print(f"  Meta-cluster: 1 (automatically added)")
        print(f"  Total clusters in embedding: {n_clusters + 1}")
        print(f"  Assets per cluster: {assets_per_cluster}")
        print(f"  Total assets: {n_clusters * assets_per_cluster}")
        print(f"  Discrete levels: {n_levels_or_bits}")

        bqm, asset_vars, meta_vars, total_clusters = create_dummy_portfolio_bqm(
            n_clusters, assets_per_cluster, n_levels_or_bits
        )

        print(f"  Variables: {len(bqm.variables)}")
        print(f"  Edges: {len(bqm.quadratic)}")
        print(f"  Meta-cluster assets: {len(meta_vars)}")

        config = {
            'n_asset_clusters': n_clusters,
            'n_meta_clusters': 1,
            'n_clusters_total': total_clusters,
            'assets_per_cluster': assets_per_cluster,
            'total_assets': n_clusters * assets_per_cluster,
            'n_levels': n_levels_or_bits,
            'alpha': 5.0,
            'beta': 2.5,
        }

    else:
        raise ValueError(f"Unknown formulator type: {formulator_type}. Only 'discrete_levels' is supported.")

    # Find embedding
    embedding = find_embedding(
        bqm, target_nodelist, target_edgelist,
        timeout, tries, chainlength_patience
    )

    if not embedding:
        return None

    # Compute statistics
    chain_lengths = [len(chain) for chain in embedding.values()]
    all_qubits = [q for chain in embedding.values() for q in chain]
    total_qubits = len(set(all_qubits))

    # Compute sparsification signature (hash of sparsification parameters)
    import hashlib
    sparsif_str = f"cov_cutoff=0.0,diag_risk=0.0,risk_abs=0.0,risk_pct=None,max_deg=None,preserve_budget=True"
    sparsif_sig = hashlib.sha256(sparsif_str.encode()).hexdigest()[:16]

    # Create template data
    template_data = {
        'version': '1.0.0',
        'formulator': formulator_type,
        'created_at': datetime.now().isoformat(),
        'generator_version': 'tools/generate_portfolio_embedding_template.py@v1.0',
        'configuration': config,
        'topology': topology_metadata,
        'shape': config,  # Alias for compatibility
        'sparsification_signature': sparsif_sig,
        'embedding': embedding,
        'statistics': {
            'n_variables': len(embedding),
            'total_qubits_used': total_qubits,
            'qpu_utilization_pct': float(100 * total_qubits / len(target_nodelist)),
            'avg_chain_length': float(np.mean(chain_lengths)),
            'max_chain_length': int(max(chain_lengths)),
            'min_chain_length': int(min(chain_lengths)),
        }
    }

    return template_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate portfolio optimization embedding template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Topology source
    topology_group = parser.add_mutually_exclusive_group(required=True)
    topology_group.add_argument(
        '--solver',
        type=str,
        help='D-Wave solver name (e.g., Advantage2_system1.6)'
    )
    topology_group.add_argument(
        '--topology',
        type=Path,
        help='Path to topology JSON file (from dump_dwave_topology.py)'
    )

    # Formulator type
    parser.add_argument(
        '--formulator',
        type=str,
        choices=['discrete_levels', 'qubo_bits'],
        default='discrete_levels',
        help='QUBO formulator type (default: discrete_levels)'
    )

    # Configuration (now required)
    parser.add_argument(
        '--num-clusters',
        type=int,
        required=True,
        help='Number of clusters. For discrete_levels: asset clusters (meta-cluster is added automatically). '
             'For qubo_bits: total clusters. Example: --num-clusters 12 with discrete_levels creates 12 asset clusters + 1 meta-cluster = 13 total.'
    )

    parser.add_argument(
        '--cluster-size',
        type=int,
        required=True,
        help='Number of assets per cluster'
    )

    # Mutually exclusive: n-levels for discrete_levels, n-bits for qubo_bits
    encoding_group = parser.add_mutually_exclusive_group()
    encoding_group.add_argument(
        '--n-levels',
        type=int,
        help='Number of discrete weight levels (for discrete_levels formulator, default: 10)'
    )
    encoding_group.add_argument(
        '--n-bits',
        type=int,
        help='Number of bits per asset (for qubo_bits formulator, default: 10)'
    )

    # Embedding parameters
    parser.add_argument(
        '--timeout',
        type=int,
        default=3600,
        help='Embedding timeout in seconds (default: 3600)'
    )

    parser.add_argument(
        '--tries',
        type=int,
        default=10,
        help='Number of embedding attempts (default: 10)'
    )

    parser.add_argument(
        '--chainlength-patience',
        type=int,
        default=500,
        help='Chainlength patience parameter (default: 500)'
    )

    # Output
    parser.add_argument(
        '--output',
        type=Path,
        help='Output JSON file path (default: auto-generated name in embeddings/templates/)'
    )

    args = parser.parse_args()

    # Determine n_levels_or_bits based on formulator type
    if args.formulator == 'discrete_levels':
        n_levels_or_bits = args.n_levels if args.n_levels is not None else 10
        suffix = f"{n_levels_or_bits}l"
    elif args.formulator == 'qubo_bits':
        n_levels_or_bits = args.n_bits if args.n_bits is not None else 10
        suffix = f"{n_levels_or_bits}b"
    else:
        raise ValueError(f"Unknown formulator: {args.formulator}")

    # Auto-generate output filename if not specified
    if not args.output:
        args.output = Path(f'embeddings/templates/portfolio_{args.num_clusters}c_{args.cluster_size}a_{suffix}.json')

    # Generate template
    try:
        topology_source = 'live' if args.solver else str(args.topology)

        template = generate_template(
            formulator_type=args.formulator,
            n_clusters=args.num_clusters,
            assets_per_cluster=args.cluster_size,
            n_levels_or_bits=n_levels_or_bits,
            topology_source=topology_source,
            solver_name=args.solver,
            timeout=args.timeout,
            tries=args.tries,
            chainlength_patience=args.chainlength_patience
        )

        if not template:
            print(f"\n✗ Template generation failed", file=sys.stderr)
            return 1

        # Save to JSON
        args.output.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving template to: {args.output}", flush=True)
        with open(args.output, 'w') as f:
            json.dump(template, f, indent=2)

        print(f"\n{'='*70}")
        print(f"SUCCESS!")
        print(f"{'='*70}")
        print(f"\nTemplate Details:")
        print(f"  Formulator: {template['formulator']}")

        if template['formulator'] == 'discrete_levels':
            print(f"  Asset clusters: {template['configuration']['n_asset_clusters']}")
            print(f"  Meta-cluster: {template['configuration']['n_meta_clusters']}")
            print(f"  Total clusters: {template['configuration']['n_clusters_total']}")
            print(f"  Assets per cluster: {template['configuration']['assets_per_cluster']}")
            print(f"  Total assets: {template['configuration']['total_assets']}")
            print(f"  Discrete levels: {template['configuration']['n_levels']}")
        elif template['formulator'] == 'qubo_bits':
            print(f"  Clusters: {template['configuration']['n_clusters']}")
            print(f"  Assets per cluster: {template['configuration']['assets_per_cluster']}")
            print(f"  Total assets: {template['configuration']['total_assets']}")
            print(f"  Bits per asset: {template['configuration']['n_bits']}")

        print(f"  Variables: {template['statistics']['n_variables']}")
        print(f"  QPU utilization: {template['statistics']['qpu_utilization_pct']:.1f}%")
        print(f"\n✓ Template saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
