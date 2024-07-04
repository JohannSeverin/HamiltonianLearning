import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.operators import sigma_x, sigma_y, sigma_z, identity
from utils.operators import gate_y_90, gate_x_270, identity
from utils.tensor import tensor_product
from functools import partial

from itertools import product

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from typing import List


# Operator to define states from strings
_pauli_operator_names = {
    "I": identity(2),
    "X": sigma_x(2),
    "Y": sigma_y(2),
    "Z": sigma_z(2),
    "-X": -sigma_x(2),
    "-Y": -sigma_y(2),
    "-Z": -sigma_z(2),
}


# Combinations to get states which should be used as initial states
def _get_states(state_strings: List[str]):
    """
    Return a list of states given a list of state strings
    """
    return [
        identity(2) / 2 + _pauli_operator_names[state] / 2 for state in state_strings
    ]


def _pauli_state_combinations(states, n_qubits):
    """
    Return the initial states for the Pauli operators
    """
    single_qubit_states = _get_states(states)
    multi_qubit_states = [
        tensor_product(states)
        for states in product(single_qubit_states, repeat=n_qubits)
    ]

    return jnp.stack(multi_qubit_states, axis=0)


# Basise Change Operators
# To measure along x we rotate with y90
# To measure along y we rotate with x-90
# To measure along z we apply Identity

_basis_transformations_names = {
    "X": gate_y_90(2),
    "Y": gate_x_270(2),
    "Z": identity(2),
}


def _get_basis_transformations(basis_strings: List[str]):
    """
    Return a list of basis transformations given a list of basis strings
    """
    return [_basis_transformations_names[basis] for basis in basis_strings]


def _basis_transformations_combinations(basis, n_qubits):
    """
    Return the basis transformations for the Pauli operators
    """
    single_qubit_basis = _get_basis_transformations(basis)
    multi_qubit_basis = [
        tensor_product(basis) for basis in product(single_qubit_basis, repeat=n_qubits)
    ]

    return jnp.stack(multi_qubit_basis, axis=0)


# Function to be used in loop to change the basis
# It will compile the operators on first run-through and afterwards only apply the transformations
@partial(jax.jit, static_argnums=(1, 2))
def _change_to_measurement_basis(rho, basis, n_qubits):
    transformation = basis_transformations_combinations(basis, n_qubits).reshape(
        len(basis) ** n_qubits, 2**n_qubits, 2**n_qubits
    )
    return jnp.einsum(
        "bik, ...kl, bjl -> ...bij", transformation, rho, transformation.conj()
    )
