#!/usr/bin/env python3
"""
Export D-Wave QPU topology to JSON format.

This tool connects to a D-Wave solver and exports its hardware topology
(nodes and edges) to a JSON file for offline use.

Usage:
    python tools/dump_dwave_topology.py --solver Advantage2_system1.6 --output topology.json
    python tools/dump_dwave_topology.py --list  # List available solvers
"""

import argparse
import json
import sys
from pathlib import Path
from dwave.system import DWaveSampler
from dotenv import load_dotenv

load_dotenv()


def list_available_solvers():
    """List all available D-Wave solvers."""
    from dwave.cloud import Client

    print("Connecting to D-Wave...", flush=True)
    with Client.from_config() as client:
        solvers = client.get_solvers()

        print("\nAvailable D-Wave Solvers:")
        print("=" * 80)

        for solver in solvers:
            props = solver.properties
            print(f"\nSolver: {solver.id}")
            print(f"  Topology: {props.get('topology', {}).get('type', 'unknown')}")
            print(f"  Qubits: {props.get('num_qubits', 'unknown')}")
            print(f"  Status: {solver.status}")
            if 'chip_id' in props:
                print(f"  Chip ID: {props['chip_id']}")


def dump_topology(solver_name: str, output_file: Path, include_properties: bool = False):
    """
    Dump D-Wave topology to JSON.

    Args:
        solver_name: Name of the D-Wave solver
        output_file: Path to output JSON file
        include_properties: Whether to include full solver properties
    """
    print(f"Connecting to solver: {solver_name}...", flush=True)
    sampler = DWaveSampler(solver=solver_name)

    print(f"Connected to: {sampler.properties.get('chip_id', 'unknown')}", flush=True)

    # Extract topology
    nodelist = sampler.nodelist
    edgelist = sampler.edgelist

    topology_data = {
        'solver': solver_name,
        'chip_id': sampler.properties.get('chip_id'),
        'topology_type': sampler.properties.get('topology', {}).get('type'),
        'num_qubits': len(nodelist),
        'num_couplers': len(edgelist),
        'nodes': list(nodelist),  # Convert to list for JSON
        'edges': [[int(u), int(v)] for u, v in edgelist],  # Convert to list of lists
    }

    # Add basic properties
    if include_properties:
        topology_data['properties'] = {
            'extended_j_range': sampler.properties.get('extended_j_range'),
            'h_range': sampler.properties.get('h_range'),
            'j_range': sampler.properties.get('j_range'),
            'num_qubits_total': sampler.properties.get('num_qubits'),
            'supported_problem_types': sampler.properties.get('supported_problem_types'),
            'topology': sampler.properties.get('topology'),
        }

    # Write to JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing topology to: {output_file}", flush=True)
    with open(output_file, 'w') as f:
        json.dump(topology_data, f, indent=2)

    print(f"\nTopology Summary:", flush=True)
    print(f"  Solver: {topology_data['solver']}", flush=True)
    print(f"  Chip ID: {topology_data['chip_id']}", flush=True)
    print(f"  Topology: {topology_data['topology_type']}", flush=True)
    print(f"  Working qubits: {topology_data['num_qubits']}", flush=True)
    print(f"  Couplers: {topology_data['num_couplers']}", flush=True)
    print(f"\n✓ Topology saved to: {output_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Export D-Wave QPU topology to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available solvers
  python tools/dump_dwave_topology.py --list

  # Export Advantage2 topology
  python tools/dump_dwave_topology.py --solver Advantage2_system1.6 --output topologies/advantage2.json

  # Include full properties
  python tools/dump_dwave_topology.py --solver Advantage2_system1.6 --output topology.json --properties
        """
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available D-Wave solvers'
    )

    parser.add_argument(
        '--solver',
        type=str,
        help='D-Wave solver name (e.g., Advantage2_system1.6)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        help='Output JSON file path'
    )

    parser.add_argument(
        '--properties',
        action='store_true',
        help='Include full solver properties in output'
    )

    args = parser.parse_args()

    if args.list:
        list_available_solvers()
        return 0

    if not args.solver:
        parser.error("--solver is required (or use --list to see available solvers)")

    if not args.output:
        parser.error("--output is required")

    try:
        dump_topology(args.solver, args.output, args.properties)
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
