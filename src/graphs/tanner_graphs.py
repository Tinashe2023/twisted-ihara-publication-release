"""Construct Tanner graphs from parity-check matrices."""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike


def build_tanner_graph(parity_check_matrix: ArrayLike) -> nx.Graph:
	"""Build the bipartite Tanner graph of a parity-check matrix.

	Matrix rows become check nodes and columns become variable nodes. An edge
	joins check ``i`` to variable ``j`` exactly when the corresponding matrix
	entry is nonzero.
	"""
	matrix = np.asarray(parity_check_matrix)
	if matrix.ndim != 2:
		raise ValueError("parity_check_matrix must be a two-dimensional array")

	graph = nx.Graph()
	check_nodes = [("check", index) for index in range(matrix.shape[0])]
	variable_nodes = [("variable", index) for index in range(matrix.shape[1])]

	graph.add_nodes_from(check_nodes, bipartite=0)
	graph.add_nodes_from(variable_nodes, bipartite=1)

	for check_index, variable_index in zip(*np.nonzero(matrix)):
		graph.add_edge(
			("check", int(check_index)),
			("variable", int(variable_index)),
			value=matrix[check_index, variable_index],
		)

	return graph
