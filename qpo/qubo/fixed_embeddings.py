"""
Fixed embeddings for portfolio optimization QUBOs.

Pre-computed embeddings for common cluster sizes to avoid re-computing
minor-embeddings on every solve. Embeddings are topology-specific.
"""

import pickle
import hashlib
from pathlib import Path
from typing import Dict, Optional, List
import dimod


class EmbeddingCache:
    """Cache for minor-embeddings of portfolio QUBOs."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize embedding cache.

        Args:
            cache_dir: Directory to store cached embeddings (default: ./embeddings/)
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / 'embeddings'

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def get_structure_hash(self, bqm: dimod.BinaryQuadraticModel) -> str:
        """
        Compute hash of BQM structure (graph topology only, not coefficients).

        Args:
            bqm: Binary quadratic model

        Returns:
            Hash string identifying the graph structure
        """
        # Extract graph structure (variables and edges)
        variables = sorted(bqm.variables)
        edges = sorted(bqm.quadratic.keys())

        # Create deterministic string representation
        structure_str = f"{len(variables)}v_"
        structure_str += f"{len(edges)}e_"
        structure_str += hashlib.md5(str(edges).encode()).hexdigest()[:8]

        return structure_str

    def load_embedding(self, structure_hash: str, topology: str) -> Optional[Dict]:
        """
        Load cached embedding for given structure and topology.

        Args:
            structure_hash: Hash identifying BQM structure
            topology: Hardware topology ('zephyr', 'pegasus', 'chimera')

        Returns:
            Embedding dict or None if not cached
        """
        cache_file = self.cache_dir / f"{topology}_{structure_hash}.pkl"

        if not cache_file.exists():
            return None

        with open(cache_file, 'rb') as f:
            embedding = pickle.load(f)

        return embedding

    def save_embedding(self,
                      structure_hash: str,
                      topology: str,
                      embedding: Dict):
        """
        Save embedding to cache.

        Args:
            structure_hash: Hash identifying BQM structure
            topology: Hardware topology
            embedding: Embedding dict from minorminer
        """
        cache_file = self.cache_dir / f"{topology}_{structure_hash}.pkl"

        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)

    def find_and_cache_embedding(self,
                                 bqm: dimod.BinaryQuadraticModel,
                                 target_edgelist: List,
                                 topology: str,
                                 **embedding_params) -> Dict:
        """
        Find embedding for BQM and cache it.

        Args:
            bqm: Binary quadratic model
            target_edgelist: Hardware graph edges
            topology: Hardware topology name
            **embedding_params: Additional parameters for find_embedding

        Returns:
            Embedding dict
        """
        from minorminer import find_embedding

        structure_hash = self.get_structure_hash(bqm)

        # Check cache first
        embedding = self.load_embedding(structure_hash, topology)

        if embedding is not None:
            print(f"Loaded cached embedding for {structure_hash} ({topology})")
            return embedding

        # Compute new embedding
        print(f"Computing new embedding for {structure_hash} ({topology})...")
        print(f"  Variables: {len(bqm.variables)}, Edges: {len(bqm.quadratic)}")

        source_edgelist = list(bqm.quadratic.keys())

        embedding = find_embedding(
            source_edgelist,
            target_edgelist,
            **embedding_params
        )

        if not embedding:
            raise ValueError(f"Failed to find embedding for {structure_hash}")

        # Save to cache
        self.save_embedding(structure_hash, topology, embedding)
        print(f"  Embedding cached to {self.cache_dir}")

        # Print embedding stats
        chain_lengths = [len(chain) for chain in embedding.values()]
        print(f"  Chain length: avg={sum(chain_lengths)/len(chain_lengths):.1f}, "
              f"max={max(chain_lengths)}")

        return embedding


def create_fixed_embedding_sampler(sampler, bqm: dimod.BinaryQuadraticModel,
                                   topology: str = 'zephyr',
                                   cache_dir: Optional[Path] = None):
    """
    Create a FixedEmbeddingComposite with cached embedding.

    Args:
        sampler: D-Wave sampler (DWaveSampler)
        bqm: Binary quadratic model to embed
        topology: Hardware topology ('zephyr', 'pegasus', 'chimera')
        cache_dir: Directory for caching embeddings

    Returns:
        FixedEmbeddingComposite sampler with cached embedding
    """
    from dwave.system import FixedEmbeddingComposite

    # Get hardware structure
    _, target_edgelist, _ = sampler.structure

    # Find or load cached embedding
    cache = EmbeddingCache(cache_dir)
    embedding = cache.find_and_cache_embedding(bqm, target_edgelist, topology)

    # Create fixed embedding sampler
    return FixedEmbeddingComposite(sampler, embedding)


def load_template_embedding(formulator: str, shape: Dict, topology: str = 'zephyr') -> tuple:
    """
    Load a pre-computed embedding template.

    Args:
        formulator: 'discrete_levels' or 'qubo_bits'
        shape: Dict with keys like 'n_clusters', 'assets_per_cluster', 'n_levels' or 'n_bits'
        topology: Hardware topology family ('zephyr', 'pegasus', etc.)

    Returns:
        (embedding, metadata) tuple or (None, None) if not found

    Example:
        >>> embedding, metadata = load_template_embedding(
        ...     'qubo_bits',
        ...     {'n_clusters': 13, 'assets_per_cluster': 9, 'n_bits': 10},
        ...     'zephyr'
        ... )
    """
    import json

    # Construct template filename based on formulator and shape
    template_dir = Path(__file__).parent.parent.parent / 'embeddings' / 'templates'

    if formulator == 'discrete_levels':
        # portfolio_{C}c_{A}a_{L}l_{topo}.json or portfolio_{C}c_{A}a_{L}l.json
        C = shape.get('n_clusters') or shape.get('clusters')
        A = shape.get('assets_per_cluster')
        L = shape.get('n_levels')

        if C is None or A is None or L is None:
            return None, None

        # Try exact match first
        candidates = [
            template_dir / f"portfolio_{C}c_{A}a_{L}l_{topology}.json",
            template_dir / f"portfolio_{C}c_{A}a_{L}l.json",
        ]

        # Try templates with more clusters (can use subset)
        # Search for templates with C+1 to C+10 clusters
        for extra_clusters in range(1, 11):
            C_larger = C + extra_clusters
            candidates.extend([
                template_dir / f"portfolio_{C_larger}c_{A}a_{L}l_{topology}.json",
                template_dir / f"portfolio_{C_larger}c_{A}a_{L}l.json",
            ])

        # Fallback: single-cluster templates
        candidates.extend([
            template_dir / f"cluster_{A}a_{L}l_{topology}.json",
            template_dir / f"cluster_{A}a_{L}l.json",
        ])

    elif formulator == 'qubo_bits':
        # portfolio_{C}c_{A}a_{B}b_{topo}.json or portfolio_{C}c_{A}a_{B}b.json
        C = shape.get('n_clusters')
        A = shape.get('assets_per_cluster')
        B = shape.get('n_bits')

        if C is None or A is None or B is None:
            return None, None

        # Try with topology suffix first, then without
        candidates = [
            template_dir / f"portfolio_{C}c_{A}a_{B}b_{topology}.json",
            template_dir / f"portfolio_{C}c_{A}a_{B}b.json",
            template_dir / f"cluster_{A}a_{B}b_{topology}.json",
            template_dir / f"cluster_{A}a_{B}b.json",
        ]
    else:
        return None, None

    # Try to load from candidates
    for template_path in candidates:
        if template_path.exists():
            try:
                with open(template_path, 'r') as f:
                    data = json.load(f)

                # Extract embedding and metadata
                embedding = data.get('embedding')
                if embedding is None:
                    continue

                metadata = {
                    'template_file': str(template_path),
                    'formulator': data.get('formulator', formulator),
                    'topology': data.get('topology', {}),
                    'shape': data.get('shape', shape),
                    'sparsification_signature': data.get('sparsification_signature'),
                    'version': data.get('version'),
                    'created_at': data.get('created_at'),
                }

                return embedding, metadata

            except Exception:
                continue

    # No template found
    return None, None


def load_single_cluster_template(cluster_size: int, n_levels: int, formulator: str = 'discrete_levels', topology: str = 'zephyr') -> tuple:
    """
    Load template for a single cluster.

    Supports padding: If no exact match, uses a larger template and returns
    only the subset of variables needed for the actual cluster size.

    Args:
        cluster_size: Number of assets in cluster (actual assets, may be < template size)
        n_levels: Number of discrete levels
        formulator: 'discrete_levels' or 'qubo_bits'
        topology: Hardware topology

    Returns:
        (embedding, metadata) tuple or (None, None) if not found
    """
    # Try available template sizes in ascending order (prefer closest match)
    # Common template sizes: 8, 9, 16, plus larger ones for padding
    template_sizes = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 24, 32, 64, 144, 196, 289]

    for template_size in sorted(template_sizes):
        # Only try templates >= cluster_size (we can use subset of larger template)
        if template_size < cluster_size:
            continue

        # Try single-cluster template
        shape = {
            'n_clusters': 1,
            'assets_per_cluster': template_size,
            'n_levels': n_levels
        }

        embedding, metadata = load_template_embedding(formulator, shape, topology)

        if embedding is not None:
            # Filter to only ASSET_0 variables (first cluster)
            single_cluster_embedding = {
                k: v for k, v in embedding.items()
                if k.startswith('ASSET_0_')
            }

            if single_cluster_embedding:
                # Further filter: only use first cluster_size assets (ASSET_0_0_*, ASSET_0_1_*, ..., ASSET_0_(cluster_size-1)_*)
                filtered_embedding = {}
                for var, qubits in single_cluster_embedding.items():
                    # Parse: ASSET_0_asset_idx_level
                    parts = var.split('_')
                    if len(parts) == 4:
                        asset_idx = int(parts[2])
                        if asset_idx < cluster_size:
                            filtered_embedding[var] = qubits

                if filtered_embedding:
                    metadata['template_name'] = metadata.get('template_file', 'unknown')
                    metadata['template_size'] = template_size
                    metadata['actual_cluster_size'] = cluster_size
                    metadata['note'] = f'Using {cluster_size}/{template_size} assets from template'
                    return filtered_embedding, metadata

        # Try multi-cluster templates and extract first cluster
        for n_clusters in [12, 13, 14, 16, 17]:
            shape['n_clusters'] = n_clusters
            embedding, metadata = load_template_embedding(formulator, shape, topology)

            if embedding is not None:
                # Extract ASSET_0 variables
                single_cluster_embedding = {
                    k: v for k, v in embedding.items()
                    if k.startswith('ASSET_0_')
                }

                if single_cluster_embedding:
                    # Filter to first cluster_size assets
                    filtered_embedding = {}
                    for var, qubits in single_cluster_embedding.items():
                        parts = var.split('_')
                        if len(parts) == 4:
                            asset_idx = int(parts[2])
                            if asset_idx < cluster_size:
                                filtered_embedding[var] = qubits

                    if filtered_embedding:
                        metadata['template_name'] = metadata.get('template_file', 'unknown')
                        metadata['template_size'] = template_size
                        metadata['actual_cluster_size'] = cluster_size
                        metadata['note'] = f'Extracted cluster 0 from {n_clusters}-cluster template, using {cluster_size}/{template_size} assets'
                        return filtered_embedding, metadata

    return None, None


def create_fixed_embedding_sampler_from_template(sampler, embedding: Dict):
    """
    Create a FixedEmbeddingComposite from a pre-loaded template embedding.

    Args:
        sampler: D-Wave sampler (DWaveSampler)
        embedding: Pre-loaded embedding dict {variable: [qubits]}

    Returns:
        FixedEmbeddingComposite sampler with the given embedding
    """
    from dwave.system import FixedEmbeddingComposite

    return FixedEmbeddingComposite(sampler, embedding)
