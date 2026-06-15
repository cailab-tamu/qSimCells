import numpy as np
from scipy.stats import nbinom
from typing import Tuple, Union
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator


def create_rotation_circuit(angles: list) -> QuantumCircuit:
    """
    Initialises a quantum circuit by applying an Ry rotation gate to each qubit.
    The number of qubits equals len(angles).

    Parameters
    ----------
    angles : list[float]
        Rotation angles in radians, one per qubit.
        Tip: pass np.array([0.2, 0.5, ...]) * np.pi to keep angles as pi-fractions.

    Returns
    -------
    QuantumCircuit
        Circuit with Ry(angle) applied to each qubit. No measurements.
    """
    qr = QuantumRegister(len(angles), name='q')
    circuit = QuantumCircuit(qr)
    for i, angle in enumerate(angles):
        circuit.ry(angle, qr[i])
    return circuit


def concatenate_circuits_with_separate_measurements(
    circ1: QuantumCircuit,
    circ2: QuantumCircuit
) -> QuantumCircuit:
    """
    Places two circuits on disjoint qubit registers and adds two separate
    classical registers (c_measure1, c_measure2) for independent readout.

    Parameters
    ----------
    circ1, circ2 : QuantumCircuit
        The two cell-type circuits (no measurements required).

    Returns
    -------
    QuantumCircuit
        Combined circuit; no gates or measurements added beyond the two inputs.
    """
    n1, n2 = circ1.num_qubits, circ2.num_qubits
    qr_all      = QuantumRegister(n1 + n2, name='q')
    cr_measure1 = ClassicalRegister(n1, name='c_measure1')
    cr_measure2 = ClassicalRegister(n2, name='c_measure2')
    circ_all = QuantumCircuit(qr_all, cr_measure1, cr_measure2)
    circ_all.compose(circ1, qubits=range(n1),          inplace=True)
    circ_all.compose(circ2, qubits=range(n1, n1 + n2), inplace=True)
    return circ_all


def add_crx_and_measurements_to_circuit(
    base_circuit: QuantumCircuit,
    circ1_num_qubits: int,
    interaction_map: list,
    crx_angle: Union[float, list] = np.pi
) -> QuantumCircuit:
    """
    Applies CRX(theta) gates for each (control, target) pair in interaction_map,
    then measures both cell-type registers.

    CRX(theta) is the controlled-RX rotation gate:

      crx_angle = np.pi        -> CRX(pi) == CX  (real amplitudes, classically simulable)
      crx_angle = np.pi / 2    -> partial entanglement with complex amplitudes
      Any other value           -> complex amplitudes outside the non-negative real regime,
                                   going beyond the classically simulable Ry+CX family.

    Parameters
    ----------
    base_circuit : QuantumCircuit
        Output of concatenate_circuits_with_separate_measurements (no measurements yet).
    circ1_num_qubits : int
        Number of qubits belonging to Cell Type 1 (determines classical register split).
    interaction_map : list of (int, int)
        Ordered list of (control_qubit, target_qubit) pairs using global qubit indices.
        Gates are applied in the order listed -- ordering matters when qubits are shared.
    crx_angle : float or list of float, default np.pi
        Rotation angle(s) for the CRX gates in radians.
          - Single float  -> same angle applied to every gate.
          - List of float -> one angle per gate (must match len(interaction_map)).
        Tip: pass np.array([0.5, 1.0, ...]) * np.pi to specify as pi-fractions.

    Returns
    -------
    QuantumCircuit
        New circuit with CRX gates and measurements appended.

    Raises
    ------
    ValueError
        If qubit indices are out of range, control == target, or the angle list
        length does not match interaction_map.
    """
    # Normalise crx_angle to a per-gate list
    if isinstance(crx_angle, (int, float, np.floating)):
        angles = [float(crx_angle)] * len(interaction_map)
    else:
        angles = list(crx_angle)
        if len(angles) != len(interaction_map):
            raise ValueError(
                f"crx_angle list length ({len(angles)}) must match "
                f"interaction_map length ({len(interaction_map)})."
            )

    circuit  = base_circuit.copy()
    qr_all   = circuit.qregs[0]
    n_total  = circuit.num_qubits

    for (ctrl, tgt), angle in zip(interaction_map, angles):
        if not (0 <= ctrl < n_total and 0 <= tgt < n_total and ctrl != tgt):
            raise ValueError(
                f"Invalid gate pair ({ctrl}, {tgt}): indices must be in "
                f"[0, {n_total}) and control != target."
            )
        circuit.crx(angle, qr_all[ctrl], qr_all[tgt])

    # Measurements
    circuit.measure(qr_all[:circ1_num_qubits],  circuit.cregs[0])
    circuit.measure(qr_all[circ1_num_qubits:],  circuit.cregs[1])
    return circuit


def run_qsimcells_circuit(
    ang_ct1, ang_ct2, interaction_map, n_shots, seed: int = 42,
    backend=None, optimization_level: int = 3
) -> Tuple[dict, dict]:
    """
    Build, transpile, and sample a two-register qSimCells circuit.

    Parameters
    ----------
    ang_ct1 : array-like
        Ry rotation angles (radians) for Cell Type 1 qubits.
    ang_ct2 : array-like
        Ry rotation angles (radians) for Cell Type 2 qubits.
    interaction_map : list of (int, int)
        CRX(pi) gate pairs using global qubit indices.
    n_shots : int
        Number of measurement shots (= number of simulated cells).
    seed : int
        Seed for both np.random and AerSimulator.
    backend : Qiskit backend or None
        None -> AerSimulator(seed_simulator=seed).
    optimization_level : int, default 3
        Pass-manager optimization level (0-3). Use 1 for speed, 3 for depth.

    Returns
    -------
    (counts_ct1, counts_ct2) : (dict, dict)
        Bitstring -> count mappings for c_measure1 and c_measure2.
    """
    np.random.seed(seed)

    circ1    = create_rotation_circuit(ang_ct1)
    circ2    = create_rotation_circuit(ang_ct2)
    combined = concatenate_circuits_with_separate_measurements(circ1, circ2)
    final    = add_crx_and_measurements_to_circuit(combined, circ1.num_qubits, interaction_map)

    if backend is None:
        backend = AerSimulator(seed_simulator=seed)

    try:
        pm      = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
        qc_comp = pm.run(final)
    except Exception:
        qc_comp = final

    result = Sampler(mode=backend).run([qc_comp], shots=n_shots).result()[0]

    reg_names  = [cr.name for cr in final.cregs]
    counts_ct1 = result.data.c_measure1.get_counts() if 'c_measure1' in reg_names else None
    counts_ct2 = result.data.c_measure2.get_counts() if 'c_measure2' in reg_names else None

    return counts_ct1, counts_ct2


def create_binary_matrix(joint_counts: dict) -> np.ndarray:
    """
    Expands a Qiskit bitstring-count dictionary into a binary cell x gene matrix.

    Each row is one simulated cell (one shot); each column is a gene.
    Bit order is reversed so that qubit 0 maps to column 0 (gene 0).

    Parameters
    ----------
    joint_counts : dict
        Bitstring -> count mapping from result.data.<register>.get_counts().

    Returns
    -------
    np.ndarray of shape (total_shots, n_genes), dtype int
    """
    if not joint_counts:
        return np.array([], dtype=int).reshape(0, 0)
    n_genes = len(next(iter(joint_counts)))
    rows = []
    for bitstring, count in joint_counts.items():
        row = [int(b) for b in reversed(bitstring)]   # qubit-0 -> column 0
        rows.extend([row] * count)
    return np.array(rows, dtype=int)


def create_count_matrix_nbinom(
    binary_matrix: np.ndarray,
    mu_vector: np.ndarray,
    r_vector: np.ndarray
) -> np.ndarray:
    """
    Converts a binary activation matrix into overdispersed count data by
    sampling from a Negative Binomial distribution for each active (1) entry.

      X_ij = NB(r_j, p_j)  if binary_matrix[i,j] == 1,  else 0
      where p_j = r_j / (mu_j + r_j).

    Parameters
    ----------
    binary_matrix : np.ndarray, shape (n_cells, n_genes)
    mu_vector     : np.ndarray, shape (n_genes,)  -- mean expression per gene
    r_vector      : np.ndarray, shape (n_genes,)  -- dispersion parameter per gene

    Returns
    -------
    np.ndarray of shape (n_cells, n_genes), dtype int32
    """
    n_cells, n_genes = binary_matrix.shape
    if len(mu_vector) != n_genes or len(r_vector) != n_genes:
        raise ValueError("mu_vector and r_vector must have length equal to n_genes.")

    counts = np.zeros((n_cells, n_genes), dtype=np.int32)
    for j in range(n_genes):
        on_idx = np.where(binary_matrix[:, j] == 1)[0]
        if len(on_idx) == 0:
            continue
        p_j = r_vector[j] / (mu_vector[j] + r_vector[j])
        counts[on_idx, j] = nbinom.rvs(n=r_vector[j], p=p_j, size=len(on_idx))
    return counts


def plot_measurement_histograms(
    circuit: QuantumCircuit,
    nshots: int = 1000,
    backend=None,
    title_prefix: str = "",
    figure_save_name: str = None,
    figsize: Tuple[int, int] = (12, 5),
    seed: int = None
) -> Tuple[dict, dict]:
    """
    Executes the circuit and plots side-by-side histograms for c_measure1 / c_measure2.

    Parameters
    ----------
    circuit : QuantumCircuit
        Must contain classical registers named 'c_measure1' and 'c_measure2'.
    nshots : int
        Number of measurement shots.
    backend : Qiskit backend or None
        None -> AerSimulator(seed_simulator=seed).
    title_prefix : str
        Prefix for the figure suptitle.
    figure_save_name : str or None
        If given, saves the figure to this path.
    figsize : tuple
        Matplotlib figure size.
    seed : int or None
        Seed for AerSimulator when backend is None.

    Returns
    -------
    (counts_measure1, counts_measure2) : (dict, dict)
        Bitstring -> count mappings for each classical register.
    """
    print(f"\n--- Running circuit: {title_prefix} ---")

    if backend is None:
        backend = AerSimulator(seed_simulator=seed)

    try:
        pm      = generate_preset_pass_manager(backend=backend, optimization_level=3)
        qc_comp = pm.run(circuit)
    except Exception:
        qc_comp = circuit

    result = Sampler(mode=backend).run([qc_comp], shots=nshots).result()[0]

    reg_names       = [cr.name for cr in circuit.cregs]
    counts_measure1 = result.data.c_measure1.get_counts() if 'c_measure1' in reg_names else None
    counts_measure2 = result.data.c_measure2.get_counts() if 'c_measure2' in reg_names else None

    if counts_measure1 is not None or counts_measure2 is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f"{title_prefix} -- Measurement counts ({nshots} shots)", fontsize=14)
        for ax, counts, label in zip(axes,
                                     [counts_measure1, counts_measure2],
                                     ['c_measure1', 'c_measure2']):
            if counts is not None:
                plot_histogram(counts, ax=ax, title=label)
            else:
                ax.set_title(f"{label} (not found)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if figure_save_name:
            fig.savefig(figure_save_name)
            print(f"Figure saved to {figure_save_name}")
        plt.show()

    return counts_measure1, counts_measure2


def get_best_quantum_backend(required_qubits: int = 5):
    """
    Returns the least-busy operational IBM Quantum backend with at least
    required_qubits qubits. Requires a saved QiskitRuntimeService account.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService
    service    = QiskitRuntimeService()
    candidates = [
        b for b in service.backends(simulator=False, operational=True)
        if b.configuration().n_qubits >= required_qubits and b.status().operational
    ]
    if not candidates:
        raise RuntimeError(f"No operational IBM backend with >= {required_qubits} qubits found.")
    candidates.sort(key=lambda b: b.status().pending_jobs)
    return candidates[0]
