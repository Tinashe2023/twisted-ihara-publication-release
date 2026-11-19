"""Search exact ordinary collisions, rejecting ALL uncolored isomorphisms.
Usage: python scripts/22_search_genuine_collisions.py --size 8 --attempts 5000
Sampling is reproducible, not uniform or exhaustive.
"""
import argparse
import json
from pathlib import Path
import networkx as nx
import numpy as np
import sympy as sp


def search(size=8, attempts=5000, seed=20261016):
    rng=np.random.default_rng(seed)
    buckets={}
    distinct=0
    for trial in range(1,attempts+1):
        # A union of disjoint permutation matrices is simple and 3-regular.
        while True:
            h=np.zeros((size,size),dtype=int)
            for _ in range(3):
                h[np.arange(size),rng.permutation(size)]+=1
            if h.max()==1:
                break
        g=nx.Graph()
        g.add_nodes_from(range(2*size))
        g.add_edges_from((int(i),size+int(j)) for i,j in zip(*np.nonzero(h)))
        if not nx.is_connected(g):
            continue
        # det(xI-A)=det(x^2 I-HH^T), so this key is exact.
        gram=sp.Matrix((h@h.T).tolist())
        key=tuple(int(c) for c in gram.charpoly().all_coeffs())
        group=buckets.setdefault(key,[])
        if any(nx.is_isomorphic(g,old_g) for old_h,old_g in group):
            continue
        distinct+=1
        if group:
            old_h,old_g=group[0]
            assert not nx.is_isomorphic(g,old_g)
            result={'found':True,'size':size,'degree':3,'seed':seed,
                    'attempts_used':trial,'distinct_graphs_seen':distinct,
                    'gram_characteristic_polynomial_descending':list(key),
                    'ordinary_collision_verified_exactly':True,
                    'uncolored_nonisomorphism_verified':True,
                    'representatives':[
                        {'index':1,'parity_check_matrix':old_h.tolist()},
                        {'index':2,'parity_check_matrix':h.tolist()}]}
            return result
        group.append((h.copy(),g))
        if trial%500==0:
            print(f'Attempt {trial}; distinct uncolored graphs: {distinct}',flush=True)
    return {'found':False,'size':size,'degree':3,'seed':seed,
            'attempts_used':attempts,'distinct_graphs_seen':distinct,
            'note':'No collision found in this sample; not a nonexistence proof.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--size',type=int,default=8)
    parser.add_argument('--attempts',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=20261016)
    args=parser.parse_args()
    if args.size<3 or args.attempts<1:
        parser.error('size must be at least 3 and attempts must be positive')
    result=search(args.size,args.attempts,args.seed)
    root=Path(__file__).resolve().parents[1]
    out=root/'data/results'/f'genuine_{args.size}x{args.size}_collision_search.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing results: {out}')
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    print(f'Saved: {out}')
