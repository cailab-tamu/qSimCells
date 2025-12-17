import sys
import os
import math
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import nbinom
import scanpy as sc
from typing import Tuple
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator


def create_rotation_circuit(angles_list: list[float]) -> QuantumCircuit:
    """
    Initializes a quantum circuit by applying a rotation gate with a
    specified angle to each qubit. The number of qubits is determined
    by the length of the angles_list.

    Args:
        angles_list (list[float]): A list of rotation angles in radians,
                                   one for each qubit.

    Returns:
        QuantumCircuit: The initialized quantum circuit.
    """
    # The number of qubits is determined by the length of the angles list.
    num_qubits = len(angles_list)
    
    # Create a quantum register with the specified number of qubits.
    qr = QuantumRegister(num_qubits, name='q')
    
    # Create a quantum circuit with the quantum register.
    circuit = QuantumCircuit(qr)

    for i in range(num_qubits):
        circuit.ry(angles_list[i], qr[i])
        
    return circuit

def concatenate_circuits_with_separate_measurements(circ1: QuantumCircuit, circ2: QuantumCircuit) -> QuantumCircuit:
    """
    Concatenates two QuantumCircuit objects onto disjoint sets of qubits
    within a larger circuit and adds separate classical registers for measurement
    of each original circuit's qubits.

    Args:
        circ1 (QuantumCircuit): The first quantum circuit.
        circ2 (QuantumCircuit): The second quantum circuit.

    Returns:
        QuantumCircuit: A new circuit combining circ1 and circ2 on separate
                        qubits, with two distinct classical registers for measurements.
    """
    ng_circ1 = circ1.num_qubits
    ng_circ2 = circ2.num_qubits
    num_total_qubits = ng_circ1 + ng_circ2

    qr_all = QuantumRegister(num_total_qubits, name='q')
    cr_measure1 = ClassicalRegister(ng_circ1, name='c_measure1')
    cr_measure2 = ClassicalRegister(ng_circ2, name='c_measure2')

    circ_all = QuantumCircuit(qr_all, cr_measure1, cr_measure2)

    # Compose circ1 onto the first set of qubits
    circ_all.compose(circ1, qubits=range(ng_circ1), inplace=True)

    # Compose circ2 onto the next set of qubits
    circ_all.compose(circ2, qubits=range(ng_circ1, num_total_qubits), inplace=True)

    return circ_all



def add_cnots_and_measurements_to_circuit(
    base_circuit: QuantumCircuit,
    circ1_num_qubits: int,
    global_cnot_configurations: list[tuple[int, int]]
) -> QuantumCircuit:
    """
    Applies a specified list of CNOT gates (using global qubit indices)
    and then adds measurements to the circuit.

    Args:
        base_circuit (QuantumCircuit): The circuit already containing the two
                                        chunks composed on disjoint qubits.
                                        This circuit should NOT have measurements yet.
        circ1_num_qubits (int): The number of qubits in the first chunk.
                                This is used to determine the classical register split.
        global_cnot_configurations (list[tuple[int, int]]): A list of tuples, where each tuple
                                                      (global_control_idx, global_target_idx)
                                                      specifies a CNOT gate using global qubit indices.

    Returns:
        QuantumCircuit: A new circuit with the specified CNOTs and measurements added.
    """
    circuit_with_cnots = base_circuit.copy()

    qr_all = circuit_with_cnots.qregs[0]
    cr_measure1 = circuit_with_cnots.cregs[0]
    cr_measure2 = circuit_with_cnots.cregs[1]

    for control_q, target_q in global_cnot_configurations:
        # Add checks to ensure indices are valid within the combined circuit
        if not (0 <= control_q < circuit_with_cnots.num_qubits and
                0 <= target_q < circuit_with_cnots.num_qubits and
                control_q != target_q):
            raise ValueError(f"Invalid CNOT indices: ({control_q}, {target_q}). Qubits must be valid and distinct.")

        circuit_with_cnots.cx(qr_all[control_q], qr_all[target_q])
        #circuit_with_cnots.cy(qr_all[control_q], qr_all[target_q])

    # Add measurements after all CNOTs are applied
    circuit_with_cnots.measure(qr_all[0:circ1_num_qubits], cr_measure1)
    circuit_with_cnots.measure(qr_all[circ1_num_qubits:circuit_with_cnots.num_qubits], cr_measure2)

    return circuit_with_cnots

def add_crx_gates_and_measurements_to_circuit(
    base_circuit: QuantumCircuit,
    circ1_num_qubits: int,
    crx_configurations: list[tuple[int, int]], # List of (control, target) global indices
    angles: list[float] # List of angles corresponding to each CRX
) -> QuantumCircuit:
    """
    Applies a specified list of CRX gates (controlled-RX) with given angles
    and then adds measurements to the circuit.

    Args:
        base_circuit (QuantumCircuit): The circuit already containing the two
                                        chunks composed on disjoint qubits.
                                        This circuit should NOT have measurements yet.
        circ1_num_qubits (int): The number of qubits in the first chunk.
                                This is used to determine the classical register split.
        crx_configurations (list[tuple[int, int]]): A list of (control_q, target_q) global qubit
                                                     indices defining where CRX gates will be placed.
        angles (list[float]): A list of rotation angles for each CRX gate,
                              corresponding to the order in `crx_configurations`.

    Returns:
        QuantumCircuit: A new circuit with the specified CRX gates and measurements added.
    """
    circuit_with_crx = base_circuit.copy()
    qr_all = circuit_with_crx.qregs[0]
    cr_measure1 = circuit_with_crx.cregs[0]
    cr_measure2 = circuit_with_crx.cregs[1]

    if len(crx_configurations) != len(angles):
        raise ValueError("Number of CRX configurations must match the number of angles.")

    for i, (control_q, target_q) in enumerate(crx_configurations):
        # Add checks for valid indices
        if not (0 <= control_q < circuit_with_crx.num_qubits and
                0 <= target_q < circuit_with_crx.num_qubits and
                control_q != target_q):
            raise ValueError(f"Invalid CRX indices: ({control_q}, {target_q}). Qubits must be valid and distinct.")
        
        #circuit_with_crx.append(CRXGate(angles[i]), [qr_all[control_q], qr_all[target_q]])
        circuit_with_crx.crx(angles[i], qr_all[control_q], qr_all[target_q]) 

    # Add measurements after all CRX gates are applied
    circuit_with_crx.measure(qr_all[0:circ1_num_qubits], cr_measure1)
    circuit_with_crx.measure(qr_all[circ1_num_qubits:circuit_with_crx.num_qubits], cr_measure2)

    return circuit_with_crx

def create_binary_matrix(joint_counts: dict):
    """
    Generates a binary matrix from a joint histogram dictionary with 0s and 1s.

    Args:
        joint_counts (dict): A dictionary where keys are bit strings
                             (representing rows) and values are their counts.

    Returns:
        np.ndarray: A reconstructed binary matrix with integer values.
    """
    if not joint_counts:
        return np.array([], dtype=int).reshape(0, 0)
    
    # Get the number of genes (columns) from the length of the first key
    first_key = next(iter(joint_counts.keys()))
    num_genes = len(first_key)
    
    reconstructed_rows = []

    # Iterate through the joint counts
    for bit_string, count in joint_counts.items():
        # Reverse the bit string to align with the original g0, g1, ... order
        reversed_bit_string = bit_string[::-1]
        
        # Convert the reversed bit string to a list of integer values (0 or 1)
        row_values = [int(char) for char in reversed_bit_string]
        
        # Repeat the row 'count' number of times
        for _ in range(count):
            reconstructed_rows.append(row_values)
            
    # Convert the list of lists into a NumPy array with integer dtype
    return np.array(reconstructed_rows, dtype=int)

# --- Re-using the gene count matrix function for demonstration ---
def create_count_matrix_nbinom(binary_matrix: np.ndarray, mu_vector: np.ndarray, r_vector: np.ndarray):
    """
    Creates a count matrix from a binary matrix using a Negative Binomial distribution
    with gene-specific mean and dispersion parameters.
    """
    num_cells, num_genes = binary_matrix.shape
    
    if len(mu_vector) != num_genes or len(r_vector) != num_genes:
        raise ValueError("The length of mu_vector and r_vector must match the number of genes.")
        
    count_matrix = np.zeros_like(binary_matrix, dtype=np.int32)
    
    for j in range(num_genes):
        on_indices = np.where(binary_matrix[:, j] == 1)[0]
        mu_j = mu_vector[j]
        r_j = r_vector[j]
        p_j = r_j / (mu_j + r_j)
        random_counts = nbinom.rvs(n=r_j, p=p_j, size=len(on_indices))
        count_matrix[on_indices, j] = random_counts
        
    return count_matrix

import numpy as np



def plot_measurement_histograms(circuit: QuantumCircuit, nshots: int = 1000, title_prefix: str = "", figure_save_name: str = None, figsize: Tuple[int, int] = (12, 5)
):
    """
    Simulates the given circuit and plots histograms for its classical registers
    (c_measure1 and c_measure2) side-by-side. The plots are displayed interactively
    and can optionally be saved to a file.

    Args:
        circuit (QuantumCircuit): The Qiskit QuantumCircuit to simulate and plot.
                                  It must have classical registers named 'c_measure1' and 'c_measure2'.
        nshots (int): The number of shots for the simulation. Defaults to 1000.
        title_prefix (str): A prefix to add to the overall figure title (e.g., "Final Circuit").
        figure_save_name (str, optional): If provided, the figure will be saved to this filename.
                                          Defaults to None (figure only displayed interactively).
    """
    print(f"\n--- Simulating and Plotting Histograms for: {title_prefix} ---")

    # 1. Simulate the circuit
    backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    qc_comp = pm.run(circuit)

    sampler = Sampler(mode=backend)
    job = sampler.run([qc_comp], shots=nshots)

    # 2. Access results and plot histograms
    try:
        result = job.result()[0]
        
        counts_measure1 = None
        counts_measure2 = None

        # Check if classical registers exist and get counts
        if 'c_measure1' in [creg.name for creg in circuit.cregs]:
            counts_measure1 = result.data.c_measure1.get_counts()
            print(f"Counts for c_measure1: {counts_measure1}")
        else:
            print("Warning: Classical register 'c_measure1' not found in circuit. Skipping histogram for c_measure1.")

        if 'c_measure2' in [creg.name for creg in circuit.cregs]:
            counts_measure2 = result.data.c_measure2.get_counts()
            print(f"Counts for c_measure2: {counts_measure2}")
        else:
            print("Warning: Classical register 'c_measure2' not found in circuit. Skipping histogram for c_measure2.")

        # Create a figure with two subplots
        if counts_measure1 is not None or counts_measure2 is not None:
            fig, axes = plt.subplots(1, 2, figsize=figsize) # 1 row, 2 columns
            fig.suptitle(f"{title_prefix} - Measurement Counts ({nshots} shots)", fontsize=16)

            if counts_measure1 is not None:
                plot_histogram(counts_measure1, ax=axes[0], title="c_measure1")
                axes[0].set_title("c_measure1") # Manually set the title
            else:
                axes[0].set_title("c_measure1 (Not Found)")
                axes[0].text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)

            if counts_measure2 is not None:
                plot_histogram(counts_measure2, ax=axes[1], title="c_measure2")
                axes[1].set_title("c_measure2") # Manually set the title
            else:
                axes[1].set_title("c_measure2 (Not Found)")
                axes[1].text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
            plt.show() # Display the figure

            if figure_save_name:
                try:
                    fig.savefig(figure_save_name)
                    print(f"Histogram figure saved to {figure_save_name}")
                except Exception as save_e:
                    print(f"Error saving figure to {figure_save_name}: {save_e}")
                finally:
                    plt.close(fig) # Close the figure after showing/saving to free memory
        else:
            print("No classical register data available to plot histograms.")

    except AttributeError as e:
        print(f"Error accessing classical register counts: {e}")
        print("Please ensure your circuit has classical registers named 'c_measure1' and 'c_measure2' and measurements are applied.")
    except Exception as e:
        print(f"An unexpected error occurred during simulation or plotting: {e}")
    
    return counts_measure1, counts_measure2