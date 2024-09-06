import sys, os

import jax
import jax.numpy as jnp

from utils.operators import sigma_x, sigma_y, sigma_z, identity
from utils.operators import gate_y_90, gate_y_270, gate_x_90, gate_x_270, identity
from utils.tensor import tensor_product


sys.path.append("/root/projects/HamiltonianLearning/hamiltonian_learning_refactor")
from hamiltonian_learning import Solver
from itertools import product
from functools import partial


# New items to be developed
# - State <-> Pauli super operator representation. Requires just N qubits
# - Lindblad -> Chi -> Pauli Operator Generator R

# General quantum channel solver


# Then combine with the tools we already have. Measurements can be added on
# if we just convert to and from the super operatorfs formalism


# Generate pauli representation
single_qubit_pauli = jnp.stack([identity(2), sigma_x(2), sigma_y(2), sigma_z(2)])


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


@partial(jax.jit, static_argnums=1)
def density_matrix_to_superket(
    density_matrices: jnp.ndarray, nqubits: int
) -> jnp.ndarray:
    """
    Convert a density matrix to its Pauli representation
    """
    # TODO: prefer not to use the reshape here
    dimension = 2**nqubits
    density_matrices = density_matrices.reshape(-1, dimension, dimension)
    pauli_states = _pauli_state_combinations(nqubits).reshape(-1, dimension, dimension)

    return jnp.expand_dims(
        jnp.einsum("...ij, kji->...k", density_matrices, pauli_states), -1
    )


@partial(jax.jit, static_argnums=1)
def superket_to_density_matrix(superkets: jnp.ndarray, nqubits: int) -> jnp.ndarray:
    """
    Convert a superket to a density matrix
    """
    dimension = 2**nqubits
    # superkets = superkets.reshape(-1, dimension)
    pauli_states = _pauli_state_combinations(nqubits).reshape(-1, dimension, dimension)

    return jnp.einsum("...kl, kji->...ji", superkets, pauli_states)


if __name__ == "__main__":

    # First test, two qubits
    nqubits = 2
    test_state = (identity(2) + sigma_z(2)) / 2
    test_state = tensor_product([test_state, test_state]).reshape(
        2**nqubits, 2**nqubits
    )

    pauli_representation = density_matrix_to_superket(test_state, nqubits=nqubits)

    superket_to_density_matrix(pauli_representation, nqubits=nqubits)

    # print(pauli_representation)

    # Second test, multiple states just a signle qubit
    nqubits = 1
    test_states = [(identity(2) + op) / 2 for op in [sigma_x(), sigma_y(), sigma_z()]]
    test_states = jnp.stack(test_states)

    pauli_representation = density_matrix_to_superket(test_states, nqubits=nqubits)
    superket_to_density_matrix(pauli_representation, nqubits=nqubits)

    # print(pauli_representation)

    # Test the other way
    test_state = jnp.zeros((4, 1)).at[0, 0].set(1 / jnp.sqrt(2)).at[3, 0].set(0.0)

    superket_to_density_matrix(test_state, nqubits=1)

    density_matrix_to_superket(jnp.eye(2) / 2, nqubits=1)
