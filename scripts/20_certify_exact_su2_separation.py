"""Exact SU(2) separation certificates for representatives 1 and 2.

Run from the project root:
    python scripts/20_certify_exact_su2_separation.py

All determinant computations use exact arithmetic. No floats are converted
into symbolic entries. A failed probe is recorded, never treated as success.
"""
from pathlib import Path
import json
import sys

import networkx as nx
import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.graphs.tanner_graphs import build_tanner_graph
from src.graphs.fundamental_holonomies import (
    deterministic_spanning_tree, fundamental_chords,
)

INPUT = PROJECT_ROOT / 'data/results/random_6x6_regular_tanner_search.json'
OUTPUT = PROJECT_ROOT / 'data/results/exact_collision_su2_certificate.json'
Q = 11
PROBES = {
    'quaternion_a': ('ix', 'iy', 'iz', 'ix', 'iy', 'iz', 'ix'),
    'quaternion_b': ('ix', 'iy', 'I', 'iz', 'ix', 'I', 'iy'),
    'quaternion_c': ('iz', 'ix', 'iy', 'I', 'iy', 'ix', 'iz'),
}


def exact_det(matrix):
    dm = DomainMatrix.from_Matrix(matrix)
    return sp.expand(dm.domain.to_sympy(dm.det()))


def palette():
    return {
        'I': sp.eye(2),
        'ix': sp.Matrix([[0, sp.I], [sp.I, 0]]),
        'iy': sp.Matrix([[0, 1], [-1, 0]]),
        'iz': sp.diag(sp.I, -sp.I),
    }


def exact_adjacency(graph, chords, specification):
    if len(chords) != len(specification):
        raise ValueError('Probe length does not match chord count.')
    choices = palette()
    for h in choices.values():
        if h.det() != 1 or h.conjugate().T * h != sp.eye(2):
            raise AssertionError('Invalid SU(2) palette entry.')
    transport = {}
    for x, y in graph.edges():
        transport[x, y] = sp.eye(2)
        transport[y, x] = sp.eye(2)
    for (x, y), label in zip(chords, specification):
        h = choices[label]
        transport[x, y] = h
        transport[y, x] = h.conjugate().T
    vertices = sorted(graph.nodes())
    index = {v: k for k, v in enumerate(vertices)}
    adjacency = sp.zeros(2 * len(vertices))
    for (x, y), h in transport.items():
        i, j = 2 * index[x], 2 * index[y]
        adjacency[i:i+2, j:j+2] = h
    if adjacency != adjacency.conjugate().T:
        raise AssertionError('Twisted adjacency must be Hermitian.')
    return adjacency


def record_for(records, index):
    matches = [r for r in records if r['index'] == index]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one representative {index}.')
    h = np.asarray(matches[0]['parity_check_matrix'])
    if h.shape != (6, 6) or not np.isin(h, [0, 1]).all():
        raise ValueError('Expected a binary 6 by 6 parity-check matrix.')
    return h.astype(int)


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(f'Missing collision search data: {INPUT}')
    records = json.loads(INPUT.read_text(encoding='utf-8'))['representatives']
    matrices = [record_for(records, i) for i in (1, 2)]
    graphs = [build_tanner_graph(h) for h in matrices]
    for g in graphs:
        if not nx.is_connected(g) or g.number_of_nodes() != 12 or g.number_of_edges() != 18:
            raise ValueError('Expected a connected 12-vertex, 18-edge graph.')
        if any(d != 3 for _, d in g.degree()):
            raise ValueError('Expected a 3-regular graph.')
    node_match = nx.algorithms.isomorphism.categorical_node_match('bipartite', None)
    tanner_isomorphic = nx.is_isomorphic(*graphs, node_match=node_match)
    if tanner_isomorphic:
        raise AssertionError('Pair is Tanner-isomorphic; stop and investigate.')

    # For 3-regular graphs the ordinary Bass polynomial is determined by
    # the adjacency characteristic polynomial. Verify this premise exactly.
    ordinary = []
    for g in graphs:
        vertices = sorted(g.nodes())
        a = sp.Matrix([[int(g.has_edge(x, y)) for y in vertices] for x in vertices])
        ordinary.append(a.charpoly().all_coeffs())
    if ordinary[0] != ordinary[1]:
        raise AssertionError('Ordinary adjacency characteristic polynomials differ.')
    print('Exact ordinary collision: verified')
    print('Tanner inequivalence: verified (bipartition-preserving)')

    chord_lists = [fundamental_chords(g, deterministic_spanning_tree(g)) for g in graphs]
    if any(len(chords) != 7 for chords in chord_lists):
        raise AssertionError('Expected seven chords per graph.')
    result = {
        'representatives': [1, 2],
        'parity_check_matrices': [h.tolist() for h in matrices],
        'ordinary_adjacency_charpoly_descending': [str(c) for c in ordinary[0]],
        'ordinary_collision_verified_exactly': True,
        'tanner_inequivalent_bipartition_preserving': True,
        'chords': [[[list(x), list(y)] for x, y in chords] for chords in chord_lists],
        'u': '1/11',
        'scaled_bass_matrix': 'K = 123 I_24 - 11 A_T',
        'edge_determinant_scale': 'det(I-u B_T) = 120^12 * det(K) / 11^72',
        'palette': {name: [[str(h[i,j]) for j in range(2)] for i in range(2)]
                    for name, h in palette().items()},
        'probes': [],
    }
    for name, spec in PROBES.items():
        print(f'\nProbe: {name}', flush=True)
        determinants = []
        for number, (g, chords) in enumerate(zip(graphs, chord_lists), 1):
            a = exact_adjacency(g, chords, spec)
            k = 123 * sp.eye(24) - Q * a
            value = exact_det(k)
            if value.is_integer is not True:
                raise AssertionError('Expected a real integer determinant.')
            determinants.append(value)
            print(f'  det K_{number} = {value}', flush=True)
        difference = determinants[0] - determinants[1]
        separated = difference != 0
        print(f'  Exact difference: {difference}')
        print(f'  Separated exactly: {separated}')
        result['probes'].append({
            'name': name, 'specification': list(spec),
            'scaled_bass_determinants': [str(v) for v in determinants],
            'exact_difference': str(difference),
            'separated_exactly': bool(separated),
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'\nSaved certificate: {OUTPUT}')
    if not any(r['separated_exactly'] for r in result['probes']):
        print('No exact separation found with these probes. Do not claim success.')


if __name__ == '__main__':
    main()
