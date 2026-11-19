import cmath

import networkx as nx
import sympy as sp

from src.ihara.oriented_edges import build_oriented_edge_index
from src.ihara.twisted_nonbacktracking import (
    build_u1_twisted_nonbacktracking_matrix,
)
from src.representations.u1_representations import trivial_u1_transport

graph = nx.cycle_graph(4)
oriented = build_oriented_edge_index(graph)

theta = sp.pi / 3
transport = trivial_u1_transport(oriented)

transport[(0, 1)] = complex(cmath.exp(1j * float(theta)))
transport[(1, 0)] = complex(cmath.exp(-1j * float(theta)))

B_twisted = sp.Matrix(
    build_u1_twisted_nonbacktracking_matrix(
        graph,
        transport,
        oriented,
    )
)

u = sp.symbols("u")
lhs = sp.expand((sp.eye(B_twisted.rows) - u * B_twisted).det())

print("det(I - u B_T):")
print(sp.N(lhs, 12))
