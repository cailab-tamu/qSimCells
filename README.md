# qSimCells

**Quantum-Inspired Single-Cell Data Simulation & Analysis Pipeline**  
[Preprint: arXiv:2510.12776](https://www.arxiv.org/abs/2510.12776)

This project provides a Python package and Jupyter notebook workflows for simulating, merging, and benchmarking cell type interactions using classical and quantum computational models. It also includes downstream single-cell analysis and R/CellChat validation tools.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [IBM Quantum Setup (Optional)](#ibm-quantum-setup-optional)
- [Running the Main Analysis & Simulation Workflow](#running-the-main-analysis--simulation-workflow)
- [Using the Python Package in Your Own Code](#using-the-python-package-in-your-own-code)
- [Verifying with R/CellChat](#verifying-with-rcellchat)
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
├── environment.yml              # Conda environment definition
├── pyproject.toml               # Python package/dependency definition
├── qSim_cell_chat.ipynb         # Main Jupyter notebook pipeline
├── README.md                    # This file
├── qsim_cells/                  # Main Python package
│   ├── __init__.py
│   ├── generative.py            # Quantum circuit/simulation functions
│   └── grn_utils.py             # Gene network utility functions
├── r_cellchat_qsim/             # R scripts/output for CellChat interaction validation
│   ├── cellchat_test.R
│   └── ...other R outputs...
└── sim_merged_datasets_co_mo_quantum_*.h5ad    # Example output data files
```

---

## Installation

1. **Clone this repository or download as zip:**

    ```sh
    git clone https://github.com/YOUR_GITHUB_USER/qSimCells.git
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

## Verifying with R/CellChat

- The `r_cellchat_qsim/` folder contains R scripts and results for CellChat-based ligand-receptor communication verification.
- After running the Python pipeline and exporting simulated data, use the R scripts in that folder for complementary cell interaction analysis.
- See `r_cellchat_qsim/cellchat_test.R` for details.

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
> Quantum-inspired synthetic single-cell analysis.  
> [arXiv:2510.12776](https://www.arxiv.org/abs/2510.12776)

---

## Contact

Questions?  
Open an issue or email **Selim Romero** (ssromerogon@tamu.edu).

---

## Contributing

Pull requests, bug reports, and suggestions are welcome!

---

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

**Enjoy quantum-enhanced single-cell simulation and analysis! 🚀**