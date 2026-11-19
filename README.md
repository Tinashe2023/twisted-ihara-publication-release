# Exact SU(2)-Twisted Ihara Verification

This repository contains the focused computational materials accompanying the manuscript:

**An Exact Ihara Collision of Tanner Graphs Separated by Quaternion-Valued Transports**

The code provides an exact-arithmetic verification of the manuscript's 16-vertex Tanner-graph example. It constructs the twisted non-backtracking operators, checks the graph and spanning-tree conditions, computes traces through degree six, enumerates simple cycles of lengths four and six, and checks the cycle-trace identity and the resulting determinant coefficients.

## Main computational result

For the two specified Tanner graphs \(X_1\) and \(X_2\), the verifier checks the exact trace sequences

- \(X_1\): \((0,0,0,0,0,0)\);
- \(X_2\): \((0,0,0,0,0,48)\).

The corresponding coefficients of \(u^6\) in \(\det(I-uB\_{T,j})\) are \(0\) and \(-8\), respectively. The JSON certificate records the graph matrices, chord sets, cycle-by-cycle transport traces, matrix dimensions, software versions, and computed coefficients.

The computational script is a reproducibility check for the finite exact-arithmetic calculations reported in the manuscript. It does **not** replace the paper's mathematical proofs of ordinary cospectrality or graph nonisomorphism.

## Requirements

- Python 3.10 or later
- NumPy
- NetworkX
- SymPy

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The archived JSON certificate was generated with NumPy 2.4.6, NetworkX 3.6.1, and SymPy 1.14.0. Other compatible versions may work; the script records the versions used for each run.

## Reproduce the verification

From the repository root, run:

```bash
python scripts/verify_16vertex_cycle_certificate.py
```

By default, the script writes a fresh report to:

```text
results/certificate_16vertex_cycle_traces_recomputed.json
```

To choose another output location:

```bash
python scripts/verify_16vertex_cycle_certificate.py --output results/my_verification.json
```

The archived reference certificate is:

```text
scripts/certificate_16vertex_cycle_traces.json
```

The script does not overwrite that archived certificate. It writes a separate recomputed report.

A successful run prints the trace sequence and degree-six coefficient for each graph, followed by `All checks passed.`

## Repository layout

```text
publication-release/
├── README.md
├── CITATION.cff
├── .gitignore
├── requirements.txt
├── scripts/
│   ├── verify_16vertex_cycle_certificate.py
│   └── certificate_16vertex_cycle_traces.json
├── src/
│   ├── ihara/
│   │   ├── oriented_edges.py
│   │   └── symbolic_su2_nonbacktracking.py
│   └── representations/
│       └── symbolic_su2.py
└── results/
    └── (newly recomputed JSON reports are written here)
```

### File roles

- `scripts/verify_16vertex_cycle_certificate.py`: main reproducibility entry point.
- `scripts/certificate_16vertex_cycle_traces.json`: archived machine-readable certificate accompanying the manuscript.
- `src/ihara/oriented_edges.py`: deterministic oriented-edge indexing.
- `src/ihara/symbolic_su2_nonbacktracking.py`: exact symbolic SU(2)-twisted non-backtracking matrix construction.
- `src/representations/symbolic_su2.py`: exact \(2\times2\) SU(2) matrix and adjoint utilities.
- `requirements.txt`: Python dependencies.
- `CITATION.cff`: citation metadata for the software repository.

## Scope and limitations

The verifier checks the explicit computational example and its exact cycle/trace/determinant calculations. It does not claim that the underlying pair of graphs is new, and it does not establish coding-performance or engineering-system implications. Those claims are outside the scope of this computational package.

## Citation

Please cite the associated manuscript and this software repository when using these materials.

**Software repository:** https://github.com/Tinashe2023/twisted-ihara-publication-release

An archival DOI will be added when a version of this repository is deposited and published through an archival service such as Zenodo.

## License

This software is released under the MIT License. See the [LICENSE](LICENSE) file for details.
