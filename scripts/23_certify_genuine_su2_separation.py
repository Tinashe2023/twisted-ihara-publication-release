"""Exact nine-chord separation of a genuine ordinary Ihara collision.
Run: python scripts/23_certify_genuine_su2_separation.py
Self-contained: exact matrices and deterministic BFS conventions are included.
"""
import json
from collections import deque
from pathlib import Path
import networkx as nx
import sympy as sp
from sympy.polys.matrices import DomainMatrix

MATRICES = [
 [[1,0,0,0,1,1,0,0],[0,1,0,0,0,1,0,1],
  [0,1,0,0,1,0,1,0],[0,0,0,1,0,0,1,1],
  [0,0,1,0,1,0,1,0],[1,0,1,1,0,0,0,0],
  [1,0,1,0,0,1,0,0],[0,1,0,1,0,0,0,1]],
 [[1,0,0,1,1,0,0,0],[0,0,0,0,1,1,0,1],
  [0,0,1,1,0,0,1,0],[1,1,0,0,1,0,0,0],
  [0,1,1,0,0,0,1,0],[0,0,0,1,0,0,1,1],
  [1,0,0,0,0,1,0,1],[0,1,1,0,0,1,0,0]]
]
PROBES = {
 'quaternion_9a': ('ix','iy','iz','ix','iy','iz','ix','iy','iz'),
 'quaternion_9b': ('ix','iy','I','iz','ix','I','iy','iz','ix'),
 'quaternion_9c': ('iz','ix','iy','I','iy','ix','iz','I','iy'),
}


def exact_det(matrix):
    dm=DomainMatrix.from_Matrix(matrix)
    return sp.expand(dm.domain.to_sympy(dm.det()))


def key(vertex):
    return type(vertex).__name__,repr(vertex)


def canonical(x,y):
    return (x,y) if key(x)<=key(y) else (y,x)


def build_graph(raw):
    h=sp.Matrix(raw)
    assert h.shape==(8,8) and all(x in (0,1) for x in h)
    assert all(sum(h[i,j] for j in range(8))==3 for i in range(8))
    assert all(sum(h[i,j] for i in range(8))==3 for j in range(8))
    g=nx.Graph()
    g.add_nodes_from([('check',i) for i in range(8)]+[('variable',i) for i in range(8)])
    g.add_edges_from((('check',i),('variable',j)) for i in range(8) for j in range(8) if h[i,j])
    assert nx.is_connected(g)
    root=min(g,key=key)
    seen={root};queue=deque([root]);tree=set()
    while queue:
        x=queue.popleft()
        for y in sorted(g.neighbors(x),key=key):
            if y not in seen:
                seen.add(y);queue.append(y);tree.add(canonical(x,y))
    chords=sorted({canonical(x,y) for x,y in g.edges()}-tree,
                  key=lambda e:(key(e[0]),key(e[1])))
    assert len(chords)==9
    return g,chords


def transport_for(g,chords,spec):
    palette={'I':sp.eye(2),'ix':sp.Matrix([[0,sp.I],[sp.I,0]]),
             'iy':sp.Matrix([[0,1],[-1,0]]),'iz':sp.diag(sp.I,-sp.I)}
    assert len(spec)==len(chords)
    t={}
    for x,y in g.edges():
        t[x,y]=sp.eye(2);t[y,x]=sp.eye(2)
    for (x,y),label in zip(chords,spec):
        h=palette[label]
        assert h.det()==1 and h.conjugate().T*h==sp.eye(2)
        t[x,y]=h;t[y,x]=h.conjugate().T
    return t


def adjacency(g,t):
    vertices=sorted(g);index={v:i for i,v in enumerate(vertices)}
    a=sp.zeros(2*len(vertices))
    for (x,y),h in t.items():
        i,j=2*index[x],2*index[y]
        a[i:i+2,j:j+2]=h
    assert a==a.conjugate().T
    return a


def edge_matrix(g,t):
    edges=list(g.edges())
    oriented=edges+[(y,x) for x,y in edges]
    b=sp.zeros(2*len(oriented))
    for i,(x,y) in enumerate(oriented):
        for j,(z,w) in enumerate(oriented):
            if y==z and w!=x:
                b[2*i:2*i+2,2*j:2*j+2]=t[z,w]
    return b


def certify():
    built=[build_graph(raw) for raw in MATRICES]
    assert not nx.is_isomorphic(built[0][0],built[1][0])
    polynomials=[(sp.Matrix(raw)*sp.Matrix(raw).T).charpoly().all_coeffs() for raw in MATRICES]
    assert polynomials[0]==polynomials[1]
    print('Exact ordinary collision: verified',flush=True)
    print('Uncolored graph nonisomorphism: verified',flush=True)
    result={'parity_check_matrices':MATRICES,
            'ordinary_collision_verified_exactly':True,
            'uncolored_nonisomorphism_verified':True,
            'gram_charpoly_descending':[str(c) for c in polynomials[0]],
            'chords':[[[list(x),list(y)] for x,y in chords] for g,chords in built],
            'u':'1/11','scaled_bass_matrix':'K = 123 I_32 - 11 A_T',
            'edge_identity':'det(11 I_96 - B_T) = 120^16 det(K)',
            'probes':[]}
    edge_checked=False
    for name,spec in PROBES.items():
        print(f'\nProbe: {name}',flush=True)
        dets=[];transports=[]
        for number,(g,chords) in enumerate(built,1):
            t=transport_for(g,chords,spec);transports.append(t)
            value=exact_det(123*sp.eye(32)-11*adjacency(g,t))
            assert value.is_integer is True
            dets.append(value)
            print(f'  det K_{number} = {value}',flush=True)
        difference=dets[0]-dets[1]
        separated=difference!=0
        print(f'  Exact difference: {difference}\n  Separated exactly: {separated}',flush=True)
        rec={'name':name,'specification':list(spec),
             'scaled_bass_determinants':[str(v) for v in dets],
             'exact_difference':str(difference),'separated_exactly':bool(separated)}
        if separated and not edge_checked:
            edge_values=[]
            for number,((g,chords),t,vertex_value) in enumerate(zip(built,transports,dets),1):
                print(f'  Checking exact 96 by 96 edge determinant {number}...',flush=True)
                edge_value=exact_det(11*sp.eye(96)-edge_matrix(g,t))
                assert edge_value==120**16*vertex_value
                edge_values.append(str(edge_value))
                print('  Exact edge/Bass identity: True',flush=True)
            rec['scaled_edge_determinants']=edge_values
            rec['edge_bass_verified_exactly']=True
            edge_checked=True
        result['probes'].append(rec)
    result['some_probe_separates_exactly']=any(r['separated_exactly'] for r in result['probes'])
    return result


if __name__=='__main__':
    result=certify()
    out=Path(__file__).resolve().parents[1]/'data/results/genuine_8x8_exact_su2_certificate.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite: {out}')
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'\nSaved: {out}')
