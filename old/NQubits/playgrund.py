# This documents serves as a proof of concept that we can run a simulation with the following
# Any Qubits with local hamiltonians and lindbladian operators
# Two qubit interaction. Here we do all to all connections but complexity can be reduced by doing nearest neighbor
import sys
import numpy as np
import jax.numpy as jnp
from jax import jit

sys.path.append("../")
from utils import *
from hamiltonian_learning_utils import *


# CONSTANTS
NQUBITS = 3


# Parameters
LOCAL_HAMILONIAN_PARAMS_SHAPE = (NQUBITS, 3)  # Which qubit, x, y or z
LOCAL_LINDBLADIAN_PARAMS_SHAPE = (NQUBITS, 9)  # Which qubit, 3x3 cholesky decomposition

COUPLED_HAMILTONIAN_PARAMS_SHAPE = (
    NQUBITS * (NQUBITS - 1) // 2,  # Two qubit permutations
    3,  # x, y, z
    3,  # x, y, z
)

# Random parameters
key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), 3)
local_hamiltonian_params = jax.random.normal(key1, LOCAL_HAMILONIAN_PARAMS_SHAPE) * 1e-2
local_lindbladian_params = (
    jax.random.normal(key2, LOCAL_LINDBLADIAN_PARAMS_SHAPE) * 1e-2
)
coupled_hamiltonian_params = (
    jax.random.normal(key3, COUPLED_HAMILTONIAN_PARAMS_SHAPE) * 1e-2
)


sys.path.append("../tensor_utils")
from kronecker import *


# Build the operators
# @jit
def build_local_hamiltonian(params):
    local_hamiltonians = jnp.zeros((NQUBITS, 2, 2), dtype=jnp.complex128)
    for qubit in range(NQUBITS):
        H_i = jnp.einsum("i, ijk -> jk", params[qubit], pauli_gate_set)
        local_hamiltonians = local_hamiltonians.at[qubit].set(H_i)

    return tensor_sum(jnp.array(local_hamiltonians))


build_local_hamiltonian(local_hamiltonian_params).shape


# @jit
def build_local_lindbladian(params):
    lindbladian_matrices = jnp.zeros((NQUBITS, 3, 3), dtype=jnp.complex128)

    for qubit in range(NQUBITS):
        L_i = jnp.zeros((3, 3), dtype=jnp.complex128)
        L_i = L_i.at[jnp.tril_indices(3)].set(params[qubit, :6])
        L_i = L_i.at[jnp.tril_indices(3, k=-1)].add(1j * params[qubit, 6:])

        lindbladian_matrices = lindbladian_matrices.at[qubit].set(L_i)

    return lindbladian_matrices


build_local_lindbladian(local_lindbladian_params).shape, pauli_gate_set.shape


# @jit
def jump_operators(params):
    cholesko = build_local_lindbladian(params)
    pauli_matrices = jnp.einsum(
        "qij, jkl -> qikl",
        cholesko,
        pauli_gate_set,
    )

    return pauli_matrices


# @jit
def packed_jump_operators(params):
    cholesko = build_local_lindbladian(params)
    pauli_matrices = jnp.einsum(
        "qij, jkl -> qikl",
        cholesko,
        pauli_gate_set,
    )

    compact_jump_operators = jnp.zeros(
        (3, 2**NQUBITS, 2**NQUBITS), dtype=jnp.complex128
    )

    for i in range(3):
        compact_jump_operators = compact_jump_operators.at[i].set(
            tensor_sum(*pauli_matrices[:, i])
        )

    return compact_jump_operators


packed_jump_operators(local_lindbladian_params).shape


# Build the cross terms in the Hamiltonian


# @jit
def build_coupled_hamiltonian(params):
    coupled_hamiltoninan = jnp.zeros((2**NQUBITS, 2**NQUBITS), dtype=jnp.complex128)
    indices = jnp.triu_indices(NQUBITS, k=1)
    for n in range(params.shape[0]):
        i, j = indices[0][n], indices[1][n]
        # Insert the parts
        His = [jnp.einsum("ij, ijkl -> kl", params[n], two_qubit_pauli_gate_set)]
        His += [jnp.eye(2)] * (NQUBITS - 2)

        # Combine to a tensor
        H_total = tensor_sum(*His)

        # Reorder indices to match the qubit order
        order = jnp.arange(NQUBITS)
        order = order.at[jnp.array([j, 1])].set(jnp.array([1, order[j]]))
        order = order.at[jnp.array([i, 0])].set(jnp.array([0, order[i]]))
        H_total = reorder_tensor(H_total, 2 * jnp.ones(NQUBITS, dtype=int), order)

        # Add to the total
        coupled_hamiltoninan += H_total

    return coupled_hamiltoninan


build_coupled_hamiltonian(coupled_hamiltonian_params)
