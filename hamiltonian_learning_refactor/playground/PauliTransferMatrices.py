import sys, os

import jax
import jax.numpy as jnp


sys.path.append("/root/projects/HamiltonianLearning/hamiltonian_learning_refactor")
from hamiltonian_learning import Solver
from itertools import product
from functools import partial

from superkets import density_matrix_to_superket, superket_to_density_matrix
from utils.operators import sigma_x, sigma_y, sigma_z, identity
from utils.operators import gate_y_90, gate_y_270, gate_x_90, gate_x_270
from utils.tensor import tensor_product


# New items to be developed
# - State <-> Pauli super operator representation. Requires just N qubits
# - Lindblad -> Chi -> Pauli Operator Generator R

# General quantum channel solver

# Then combine with the tools we already have. Measurements can be added on
# if we just convert to and from the super operatorfs formalism


# Generate pauli representation
single_qubit_pauli = jnp.stack([identity(2), sigma_x(2), sigma_y(2), sigma_z(2)])


@partial(jax.jit, static_argnums=2)
def apply_chi_matrix(density_matrices, chi_matrix, nqubits):
    """
    Apply a chi matrix to a superket
    """
    single_qubit_states = single_qubit_pauli
    multi_qubit_paulis = jnp.stack(
        [
            tensor_product(states)
            for states in product(single_qubit_states, repeat=nqubits)
        ]
    ).reshape(-1, 2**nqubits, 2**nqubits)

    # print(multi_qubit_paulis.shape)

    return jnp.einsum(
        "ij, ikl, ...lm, jmn->...kn",
        chi_matrix,
        multi_qubit_paulis,
        density_matrices,
        multi_qubit_paulis,
    )


@partial(jax.jit, static_argnums=0)
def _chi_to_pauli_matrix(nqubits):
    """
    Return the Pauli transfer matrix

    TODO: This can be simplified significantly by finding the matrix from looking at indicies
    """
    single_qubit_states = single_qubit_pauli
    multi_qubit_paulis = jnp.stack(
        [
            tensor_product(states)
            for states in product(single_qubit_states, repeat=nqubits)
        ]
    ).reshape(-1, 2**nqubits, 2**nqubits)

    all_traces = jnp.einsum(
        "iab, jbc, kcd, lda->ijkl",
        multi_qubit_paulis,
        multi_qubit_paulis,
        multi_qubit_paulis,
        multi_qubit_paulis,
    )

    return all_traces


def appply_pauli_transfer_matrix(pauli_transfer_matrix, superket):
    """
    Apply a Pauli transfer matrix to a superket
    """

    return pauli_transfer_matrix @ superket


@partial(jax.jit, static_argnums=1)
def chi_matrix_to_pauli_transfer_matrix(transfer_matrix, nqubits):
    """
    Convert a transfer matrix to a Pauli transfer matrix
    """
    basis_change_matrix = _chi_to_pauli_matrix(nqubits) / 2**nqubits

    return jnp.einsum("ijkl, ...jl->...ik", basis_change_matrix, transfer_matrix)


@partial(jax.jit, static_argnums=1)
def pauli_transfer_matrix_to_chi_matrix(pauli_transfer_matrix, nqubits):
    """
    Convert a Pauli transfer matrix to a chi matrix
    """
    basis_change_matrix = _chi_to_pauli_matrix(nqubits) / 2 ** (3 * nqubits)

    return jnp.einsum("ijkl, ...ki->...jl", basis_change_matrix, pauli_transfer_matrix)


@partial(jax.jit, static_argnums=0)
def _pauli_state_combinations(n_qubits):
    """
    Return the initial states for the Pauli operators
    """
    single_qubit_states = single_qubit_pauli
    multi_qubit_states = [
        tensor_product(states)
        for states in product(single_qubit_states, repeat=n_qubits)
    ]

    return jnp.stack(multi_qubit_states, axis=0) / jnp.sqrt(2**n_qubits)


if __name__ == "__main__":

    # First test, one qubit x gate should flip 0 to 1
    nqubits = 1
    test_state = (identity(2) + sigma_z(2)) / 2
    test_gate = jnp.zeros((4, 4)).at[1, 1].set(1.0)

    apply_chi_matrix(test_state, test_gate, nqubits)

    # Two qubit, apply x gate on qubit 0
    nqubits = 2
    test_state = (identity(2) + sigma_z(2)) / 2
    test_state = tensor_product([test_state, test_state]).reshape(
        2**nqubits, 2**nqubits
    )

    chi_identity = jnp.zeros((4, 4)).at[0, 0].set(1.0)

    test_gate = tensor_product([test_gate, chi_identity]).reshape(16, 16)

    apply_chi_matrix(test_state, test_gate, nqubits)

    # Test the translation between transfer matrices
    test_gate = jnp.zeros((4, 4)).at[1, 1].set(1.0)

    x_gate_PTM = chi_matrix_to_pauli_transfer_matrix(test_gate, 1)

    pauli_transfer_matrix_to_chi_matrix(x_gate_PTM, 1)