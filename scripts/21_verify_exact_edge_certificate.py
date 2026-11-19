"""Independently verify an exact SU(2) certificate on the edge side.
Usage: python scripts/21_verify_exact_edge_certificate.py [certificate.json]
Does not import the project's adjacency or non-backtracking builders.
"""
import json
import sys
from pathlib import Path
import sympy as sp
import networkx as nx
from sympy.polys.matrices import DomainMatrix


def determinant(matrix):
    dm = DomainMatrix.from_Matrix(matrix)
    return sp.expand(dm.domain.to_sympy(dm.det()))


def verify(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    probe = next(p for p in data['probes'] if p['name'] == 'quaternion_a')
    palette = {'I': sp.eye(2), 'ix': sp.Matrix([[0,sp.I],[sp.I,0]]),
               'iy': sp.Matrix([[0,1],[-1,0]]), 'iz': sp.diag(sp.I,-sp.I)}
    records = []
    graphs = []
    expected_charpoly = [sp.Integer(c) for c in data['ordinary_adjacency_charpoly_descending']]
    for j, raw in enumerate(data['parity_check_matrices']):
        h = sp.Matrix(raw)
        assert h.shape == (6,6) and all(x in (0,1) for x in h)
        assert all(sum(h[i,k] for k in range(6)) == 3 for i in range(6))
        assert all(sum(h[k,i] for k in range(6)) == 3 for i in range(6))
        vertices = [('check',i) for i in range(6)] + [('variable',i) for i in range(6)]
        edges = [(('check',i),('variable',k)) for i in range(6) for k in range(6) if h[i,k]]
        g = nx.Graph()
        g.add_nodes_from(vertices)
        g.add_edges_from(edges)
        nx.set_node_attributes(g, {v:v[0] for v in vertices}, 'part')
        assert nx.is_connected(g)
        graphs.append(g)
        scalar_a = sp.zeros(12)
        index = {v:i for i,v in enumerate(vertices)}
        for x,y in edges:
            scalar_a[index[x],index[y]] = scalar_a[index[y],index[x]] = 1
        assert scalar_a.charpoly().all_coeffs() == expected_charpoly
        chords = [(tuple(x),tuple(y)) for x,y in data['chords'][j]]
        assert len(chords) == len(set(chords)) == 7
        assert all(e in edges for e in chords)
        tree = g.copy()
        tree.remove_edges_from(chords)
        assert nx.is_tree(tree)
        transport = {}
        for x,y in edges:
            transport[x,y] = sp.eye(2)
            transport[y,x] = sp.eye(2)
        for (x,y),label in zip(chords,probe['specification']):
            t = palette[label]
            assert t.det() == 1 and t.conjugate().T*t == sp.eye(2)
            transport[x,y],transport[y,x] = t,t.conjugate().T
        oriented = edges + [(y,x) for x,y in edges]
        b = sp.zeros(72)
        for i,(x,y) in enumerate(oriented):
            for k,(z,w) in enumerate(oriented):
                if y == z and w != x:
                    b[2*i:2*i+2,2*k:2*k+2] = transport[z,w]
        print(f'Representative {j+1}: computing exact 72 by 72 determinant...', flush=True)
        edge_det = determinant(11*sp.eye(72)-b)
        a = sp.zeros(24)
        for (x,y),t in transport.items():
            i,k = index[x],index[y]
            a[2*i:2*i+2,2*k:2*k+2] = t
        vertex_det = determinant(123*sp.eye(24)-11*a)
        assert vertex_det == sp.Integer(probe['scaled_bass_determinants'][j])
        assert edge_det == 120**12*vertex_det
        print('  Stored vertex determinant reproduced: True')
        print('  Exact edge/Bass identity: True', flush=True)
        records.append({'representative': j+1, 'det_11I_minus_B':str(edge_det),
                        'det_K':str(vertex_det), 'exact_bass_identity': True})
    match = nx.algorithms.isomorphism.categorical_node_match('part',None)
    assert not nx.is_isomorphic(*graphs,node_match=match)
    assert records[0]['det_11I_minus_B'] != records[1]['det_11I_minus_B']
    result = {'probe':'quaternion_a','ordinary_collision_rechecked':True,
              'tanner_inequivalence_rechecked':True,'records':records,
              'edge_side_exact_separation':True}
    return result


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    path = Path(sys.argv[1]) if len(sys.argv)>1 else root/'data/results/exact_collision_su2_certificate.json'
    result = verify(path)
    output = root/'data/results/exact_edge_certificate_verification.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'Saved: {output}')
