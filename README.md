# qSimCells

**Quantum-Inspired Single-Cell Data Simulation & Analysis Pipeline**  
[Preprint: arXiv:2510.12776](https://www.arxiv.org/abs/2510.12776)

This project provides a Python package and Jupyter notebook workflows for simulating, merging, and benchmarking cell type interactions using classical and quantum computational models. It also includes downstream single-cell analysis and R/CellChat validation tools.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Simulator Benchmark — Extra Packages](#simulator-benchmark--extra-packages)
- [IBM Quantum Setup (Optional)](#ibm-quantum-setup-optional)
- [Running the Main Analysis & Simulation Workflow](#running-the-main-analysis--simulation-workflow)
- [Using the Python Package in Your Own Code](#using-the-python-package-in-your-own-code)
- [R Scripts](#r-scripts)
- [Package Hierarchy](#package-hierarchy)
- [Main Entry Point](#main-entry-point)
- [Citation/Preprint](#citationpreprint)
- [Contact](#contact)
- [Contributing](#contributing)
- [License](#license)

---

## Project Structure

```
qSimCells/
├── environment.yml              # Conda environment — core pipeline
├── pyproject.toml               # Python package definition
├── qSim_cell_chat.ipynb         # Main pipeline notebook
├── qSim_cell_benchmarks.ipynb   # GRN benchmark (4 methods × 2 cases × 10 seeds)
├── simulator_benchmark.ipynb    # Simulator comparison: SERGIO vs qSimCells vs scMultiSim
├── scmultisim_benchmark.R       # Official scMultiSim simulation script (outputs .mtx)
├── README.md                    # This file
├── qsim_cells/                  # Core Python package
│   ├── __init__.py
│   ├── generative.py            # Quantum circuit and simulation functions
│   └── grn_utils.py             # GRN utility functions
├── r_cellchat_qsim/             # R scripts and outputs for CellChat validation
│   ├── cellchat_test.R
│   └── ...other R outputs...
└── sim_merged_datasets_co_mo_quantum_*.h5ad    # Example output data files
```

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
3. In Python, save your token (run only once):

    ```python
    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(token='YOUR_IBM_TOKEN_HERE')
    ```

4. The package/notebooks will now be able to use `QiskitRuntimeService()` to submit jobs to IBM Quantum devices.

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

- To use a quantum device instead of simulator, pass a real Qiskit backend to `plot_measurement_histograms` inside the notebook.
- Results (e.g. `.h5ad` files) are saved as shown in the notebook.

---

## Using the Python Package in Your Own Code

You can use any function directly after install (if your conda env is activated and you ran `pip install -e .`):

```python
from qsim_cells.generative import create_rotation_circuit, plot_measurement_histograms
import numpy as np

# Create a circuit
angles = [0.3, 0.8, 1.27]
qc = create_rotation_circuit(angles)

# Optionally select a backend (real quantum device)
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
backend = service.least_busy([
    b for b in service.backends(simulator=False, operational=True)
    if b.configuration().n_qubits >= qc.num_qubits
])

# Run your circuit (simulated or device)
counts1, counts2 = plot_measurement_histograms(qc, backend=backend)
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
    - `qsim_cells/generative.py` — quantum simulation/synthetic data
    - `qsim_cells/grn_utils.py`  — gene regulatory/network analysis

- Main usage is shown in the notebook.
- R pipeline (`r_cellchat_qsim/`) provides downstream/validation analysis.

---

## Main Entry Point

**Main workflow:**  
Jupyter notebook `qSim_cell_chat.ipynb`, simulating, merging, and visualizing single-cell quantum-inspired data.

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

---

**Enjoy quantum-enhanced single-cell simulation and analysis! 🚀**