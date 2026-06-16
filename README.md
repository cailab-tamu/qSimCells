# qSimCells

**Quantum-Inspired Single-Cell Data Simulation & Analysis Pipeline**  
[Preprint: arXiv:2510.12776](https://www.arxiv.org/abs/2510.12776)

This project provides a Python package and Jupyter notebook workflows for simulating, merging, and benchmarking cell type interactions using classical and quantum computational models. It also includes downstream single-cell analysis and R/CellChat validation tools.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Benchmark Packages](#benchmark-packages)
- [IBM Quantum Setup (Optional)](#ibm-quantum-setup-optional)
- [Running the Main Analysis & Simulation Workflow](#running-the-main-analysis--simulation-workflow)
- [Using the Python Package in Your Own Code](#using-the-python-package-in-your-own-code)
- [R Scripts](#r-scripts)
- [Package Hierarchy](#package-hierarchy)
- [Citation/Preprint](#citationpreprint)
- [Contact](#contact)
- [Contributing](#contributing)
- [License](#license)

---

## Project Structure

```
qSimCells/
├── environment.yml                     # Conda environment — core pipeline
├── pyproject.toml                      # Python package definition
├── qSim_cell_chat.ipynb                # Main pipeline: simulation, merging, CellChat
├── qSim_cell_benchmarks.ipynb          # GRN benchmark (4 methods × 2 cases × 10 seeds)
├── qSim_100k_convergence.ipynb         # Convergence analysis at N=100k shots
├── simulator_benchmark.ipynb           # Simulator comparison: SERGIO vs qSimCells vs scMultiSim
├── scmultisim_benchmark.R              # Official scMultiSim simulation script (outputs .mtx)
├── README.md                           # This file
├── qsim_cells/                         # Core Python package
│   ├── __init__.py
│   ├── generative.py                   # Quantum circuit and simulation functions
│   └── grn_utils.py                    # GRN utility functions
├── quantum_device_run/                 # Real IBM Quantum hardware run (read-only reference)
│   ├── qSimCells_hardware_analysis.ipynb   # Hardware analysis notebook (do not modify)
│   ├── hardware_report.json                # Calibration and job metadata
│   └── qSimCells_master_figure.pdf         # Hardware figure
├── r_cellchat_qsim/                    # R scripts and outputs for CellChat validation
│   ├── cellchat_test.R
│   └── ...other R outputs...
└── sim_merged_datasets_co_mo_quantum_*.h5ad    # Example output data files
```

> **Note:** `quantum_device_run/qSimCells_hardware_analysis.ipynb` contains real IBM Quantum job IDs and cannot be re-run without the original account credentials. Treat it as a read-only reference artifact.

---

## Installation

1. **Clone this repository or download as zip:**

    ```sh
    git clone git@github.com:cailab-tamu/qSimCells.git
    cd qSimCells
    ```

2. **Create (and activate) the conda environment:**

    ```sh
    conda env create -f environment.yml
    conda activate qsim_cells_env
    ```

3. **Install the package in "editable" (developer) mode:**

    ```sh
    pip install -e .
    ```

---

## Benchmark Packages

All benchmark dependencies are included in `environment.yml` — no separate install
step is needed. The standard `conda env create` command above installs everything,
including the GRN inference and simulator packages.

| Package | Purpose |
|---------|---------|
| `arboreto` (pip) | GRNBoost2 + GENIE3 inference |
| `sergio_rs` (pip) | Official SERGIO scRNA-seq simulator — Rust reimplementation, ~150× faster (Dibaeinia & Sinha 2020) |
| `scikit-learn` | Silhouette score, AUROC |
| `seaborn` | Benchmark figures |
| `umap-learn`, `leidenalg` | UMAP + Leiden clustering in scanpy |

> **Important — SERGIO install note:**  
> `sergio_rs` is installed automatically from PyPI via `environment.yml`.  
> Do **not** run `pip install sergio` manually — that installs an unrelated package.  
> Import in Python: `import sergio_rs`

> **scMultiSim (R, optional):**  
> `simulator_benchmark.ipynb` calls scMultiSim via `Rscript` and falls back to a
> Python CIF implementation automatically if R is unavailable.  
> To install scMultiSim in R:
> ```r
> install.packages("remotes")
> remotes::install_github("ZhangLabGT/scMultiSim")
> ```

---

## IBM Quantum Setup (Optional, For Hardware Usage)

If you wish to run on actual IBM Quantum devices:

1. Register for a free account at [https://quantum.ibm.com](https://quantum.ibm.com).
2. Find your API Token in your account/profile.
3. In Python, save your token once (run this once in a separate script or notebook cell):

    ```python
    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(token='YOUR_IBM_TOKEN_HERE')
    ```

4. After saving, `QiskitRuntimeService()` will load credentials automatically in subsequent sessions.

> **Security:** Never hardcode your IBM token directly in notebook code or commit it to version control. Use `save_account()` to store it in your local credentials file, or load it from an environment variable.

### Hardware Validation

A full hardware run on **IBM Marrakesh** (27-qubit Eagle r3, native 2Q gate: CZ) is included under `quantum_device_run/`. The circuit (Ry + CRX(π), 10 genes, 2000 shots) was executed using dynamical decoupling (XY4) and gate twirling for noise mitigation on probability estimates. The resulting cell profiles are stored in `sim_merged_datasets_co_mo_quantum_device.h5ad`.

Calibration at time of run (2026-06-15): T1 = 166 µs, T2 = 87 µs, readout error ≈ 1.02%, sx/X error ≈ 0.038%, CZ error ≈ 0.26%.

> **Note on error mitigation:** Noise mitigation (dynamical decoupling, gate twirling, TREX) applies to probability *estimates* used in Figure S3. The individual bitstring cell profiles saved to `.h5ad` come from raw 2000-shot hardware counts and are not error-mitigated (mitigation operates on aggregated statistics, not individual bitstrings). Full quantum error *correction* is not applied — it would require ~100 physical qubits per logical qubit, beyond current device capacity.

---

## Running the Main Analysis & Simulation Workflow

The main workflow is in the Jupyter notebook:

```sh
jupyter notebook qSim_cell_chat.ipynb
```
or
```sh
jupyter lab qSim_cell_chat.ipynb
```

This notebook demonstrates the end-to-end workflow: quantum circuit simulation, matrix reconstruction, AnnData merging, and visualization.

Other reproducible notebooks:
- **`qSim_cell_benchmarks.ipynb`** — GRN benchmarking (4 methods × 2 cases × 10 seeds)
- **`qSim_100k_convergence.ipynb`** — Convergence analysis at 100k shots
- **`simulator_benchmark.ipynb`** — SERGIO vs qSimCells vs scMultiSim comparison

---

## Using the Python Package in Your Own Code

You can use any function directly after install (if your conda env is activated and you ran `pip install -e .`).

### Simulated run (AerSimulator)

```python
from qsim_cells.generative import run_qsimcells_circuit, create_binary_matrix
import numpy as np

# Define angles and interaction map
ang_ct1 = np.array([0.35, 0.35, 0.35, 0.35, 0.35]) * np.pi
ang_ct2 = np.array([0.35, 0.35, 0.35, 0.35, 0.35]) * np.pi
interaction_map = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]

# Run circuit (defaults to AerSimulator, seed=42)
counts_ct1, counts_ct2 = run_qsimcells_circuit(
    ang_ct1, ang_ct2, interaction_map, n_shots=2000
)

# Convert to binary cell × gene matrices
mat_ct1 = create_binary_matrix(counts_ct1)  # shape (2000, 5)
mat_ct2 = create_binary_matrix(counts_ct2)  # shape (2000, 5)
```

### Real IBM Quantum device run

```python
from qsim_cells.generative import run_qsimcells_circuit, get_best_quantum_backend
import numpy as np

# Load credentials saved via QiskitRuntimeService.save_account()
backend = get_best_quantum_backend(required_qubits=10)

counts_ct1, counts_ct2 = run_qsimcells_circuit(
    ang_ct1, ang_ct2, interaction_map, n_shots=2000,
    backend=backend, optimization_level=3
)
```

### Low-level circuit construction

```python
from qsim_cells.generative import (
    create_rotation_circuit,
    concatenate_circuits_with_separate_measurements,
    add_crx_and_measurements_to_circuit,
)

circ1    = create_rotation_circuit(ang_ct1)
circ2    = create_rotation_circuit(ang_ct2)
combined = concatenate_circuits_with_separate_measurements(circ1, circ2)
final    = add_crx_and_measurements_to_circuit(combined, circ1.num_qubits, interaction_map)
```

---

## R Scripts

**CellChat validation** (`r_cellchat_qsim/cellchat_test.R`): ligand-receptor
communication analysis on qSimCells-simulated data. Run after the main Python pipeline.

**scMultiSim benchmark** (`scmultisim_benchmark.R`): simulates co-culture and
mono-culture data using the official scMultiSim package and saves outputs as 10x
Genomics sparse matrices (`.mtx` + `features.tsv` + `barcodes.tsv`) under
`scmultisim_simulation/`. Called automatically from `simulator_benchmark.ipynb`,
or run directly:

```sh
Rscript scmultisim_benchmark.R scmultisim_simulation
```

Read the outputs in Python:
```python
import scipy.io, pandas as pd
mat      = scipy.io.mmread("scmultisim_simulation/co_culture/matrix.mtx").T.toarray()
features = pd.read_csv("scmultisim_simulation/co_culture/features.tsv", header=None)[0].tolist()
barcodes = pd.read_csv("scmultisim_simulation/co_culture/barcodes.tsv", header=None)[0].tolist()
```

---

## Package Hierarchy

- Core code modules:
    - `qsim_cells/generative.py` — quantum circuit construction, simulation, and sampling
    - `qsim_cells/grn_utils.py`  — gene regulatory network analysis utilities

### Public API (`qsim_cells.generative`)

| Function | Description |
|----------|-------------|
| `run_qsimcells_circuit` | **Main entry point.** Build, transpile, and sample a two-register circuit. Returns `(counts_ct1, counts_ct2)`. |
| `create_rotation_circuit` | Build an Ry rotation circuit for one cell type. |
| `concatenate_circuits_with_separate_measurements` | Combine two cell-type circuits on disjoint registers. |
| `add_crx_and_measurements_to_circuit` | Add CRX(θ) entangling gates and measure both registers. |
| `create_binary_matrix` | Expand bitstring counts into a binary cell × gene matrix. |
| `create_count_matrix_nbinom` | Convert binary activation to overdispersed counts via Negative Binomial. |
| `plot_measurement_histograms` | Run a circuit and display side-by-side histograms for both registers. |
| `get_best_quantum_backend` | Return the least-busy operational IBM backend with sufficient qubits. |

---

## Citation/Preprint

If you use this pipeline in your research, please cite:

> Selim Romero, et al.  
> Quantum Generative Modeling of Single-Cell Transcriptomics: Capturing Gene-Gene and Cell-Cell Interactions.  
> [arXiv:2510.12776](https://www.arxiv.org/abs/2510.12776)

---

## Contact

Questions?  
Open an issue or email **James J. Cai** (jcai@tamu.edu) | **Selim Romero** (ssromerogon@tamu.edu).

---

## Contributing

Pull requests, bug reports, and suggestions are welcome!

---

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.
