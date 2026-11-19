"""Verify the manuscript's final 16-vertex example (not the older 12-vertex pair).
Run from the project root: python scripts/verify_16vertex_cycle_certificate.py
Dependencies: numpy, networkx, sympy. Uses the original project builder when available.
Outputs a JSON report beside this script; never overwrites an existing report.
"""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
from collections import Counter
import numpy as np
import networkx as nx
import sympy as sp

H1 = [[1,0,0,0,1,1,0,0],[0,1,0,0,0,1,0,1],
[0,1,0,0,1,0,1,0],[0,0,0,1,0,0,1,1],
[0,0,1,0,1,0,1,0],[1,0,1,1,0,0,0,0],
[1,0,1,0,0,1,0,0],[0,1,0,1,0,0,0,1]]
H2 = [[1,0,0,1,1,0,0,0],[0,0,0,0,1,1,0,1],
[0,0,1,1,0,0,1,0],[1,1,0,0,1,0,0,0],
[0,1,1,0,0,0,1,0],[0,0,0,1,0,0,1,1],
[1,0,0,0,0,1,0,1],[0,1,1,0,0,1,0,0]]
CHORDS = [[(1,1),(3,6),(3,7),(4,2),(4,6),(6,2),(6,5),(7,1),(7,7)],
[(1,5),(1,7),(3,4),(4,2),(4,6),(5,6),(5,7),(7,2),(7,5)]]
Q = [sp.Matrix([[0,sp.I],[sp.I,0]]),sp.Matrix([[0,1],[-1,0]]),sp.diag(sp.I,-sp.I)]

project = None
for root in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents, Path.cwd()):
    if (root / 'src/ihara/symbolic_su2_nonbacktracking.py').is_file():
        project = root
        break
if project is not None:
    sys.path.insert(0,str(project))
    from src.ihara.symbolic_su2_nonbacktracking import build_symbolic_su2_twisted_nonbacktracking_matrix as builder
    builder_name = 'src.ihara.symbolic_su2_nonbacktracking.build_symbolic_su2_twisted_nonbacktracking_matrix'
else:
    def builder(G,T):
        edges = [(a,b) for a,b in G.edges() for a,b in [(a,b),(b,a)]]
        B=sp.zeros(2*len(edges))
        for i,(a,b) in enumerate(edges):
            for k,(c,d) in enumerate(edges):
                if b==c and d!=a:
                    B[2*i:2*i+2,2*k:2*k+2]=T[c,d]
        return B
    builder_name = 'standalone builder included in this script'

def canonical(cycle):
    c=tuple(cycle); r=c[::-1]; n=len(c)
    return min([c[i:]+c[:i] for i in range(n)]+[r[i:]+r[:i] for i in range(n)])

def cycles_of_length(G,n):
    found=set()
    def visit(path):
        if len(path)==n:
            if G.has_edge(path[-1],path[0]): found.add(canonical(path))
            return
        for v in G[path[-1]]:
            if v not in path: visit(path+[v])
    for v in G: visit([v])
    return sorted(found)

def label(v): return ('c'+str(v)) if v<8 else ('v'+str(v-8))

report={'operator_convention':'B[e,f]=T[f] for t(e)=o(f), f!=reverse(e)',
        'builder':builder_name, 'arithmetic':'exact Gaussian integers using Python-integer arrays',
        'versions':{'numpy':np.__version__,'networkx':nx.__version__,'sympy':sp.__version__}, 'graphs':[]}
for j,H in enumerate([H1,H2]):
    G=nx.Graph(); G.add_nodes_from(range(16)); T={}
    for c in range(8):
        for v in range(8):
            if H[c][v]:
                G.add_edge(c,8+v)
                M=Q[CHORDS[j].index((c,v))%3] if (c,v) in CHORDS[j] else sp.eye(2)
                T[c,8+v]=M; T[8+v,c]=M.conjugate().T
    tree=G.copy();tree.remove_edges_from((c,8+v) for c,v in CHORDS[j])
    assert nx.is_connected(G) and all(d==3 for _,d in G.degree()) and nx.is_tree(tree)
    B=builder(G,T); assert B.shape==(96,96)
    parts=[[z.as_real_imag() for z in row] for row in B.tolist()]
    R=np.array([[int(z[0]) for z in row] for row in parts],dtype=object)
    J=np.array([[int(z[1]) for z in row] for row in parts],dtype=object)
    P=np.eye(96,dtype=object); Z=np.zeros((96,96),dtype=object); traces=[]
    for k in range(1,7):
        P,Z=P@R-Z@J,P@J+Z@R
        assert int(np.trace(Z))==0
        traces.append(int(np.trace(P)))
    assert traces==[0,0,0,0,0,48*j]
    coeff=[1]
    for k in range(1,7):
        total=-sum(coeff[k-r]*traces[r-1] for r in range(1,k+1))
        assert total%k==0
        coeff.append(total//k)
    cycle_report={}
    for n in [4,6]:
        records=[]; counts=Counter()
        for c in cycles_of_length(G,n):
            M=sp.eye(2)
            for a,b in zip(c,c[1:]+c[:1]): M=M*T[a,b]
            tr=int(sp.trace(M)); counts[tr]+=1
            records.append({'vertices':[label(v) for v in c],'trace':tr,'transport':str(M)})
        weighted_sum=sum(r['trace'] for r in records)
        assert 2*n*weighted_sum==traces[n-1]
        assert len(records)=={4:5,6:10}[n]
        cycle_report[str(n)]={'counts_by_trace':dict(counts),'sum_of_traces':weighted_sum,'cycles':records}
    report['graphs'].append({'name':f'X{j+1}','H':H,'chords':CHORDS[j],
       'matrix_shape':list(B.shape),'traces_powers_1_through_6':traces,
       'det_coefficients_degrees_0_through_6':coeff,'cycles':cycle_report})
    print(f'X{j+1}: traces = {traces}; coefficient of u^6 = {coeff[6]}',flush=True)
report['all_checks_passed']=True
out=Path(__file__).resolve().with_name('certificate_16vertex_cycle_traces.json')
# Exclusive creation prevents accidental replacement of an existing report.
with out.open('x') as f: json.dump(report,f,indent=2)
print('Builder:',builder_name)
print('All checks passed. Report:',out)
