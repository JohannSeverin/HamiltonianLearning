import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.operators import sigma_x, sigma_y, sigma_z, identity
from utils.operators import gate_y_90, gate_y_270, gate_x_90, gate_x_270, identity
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

_gates_to_initilize = {
    "X": gate_y_90(2),
    "Y": gate_x_270(2),
    "Z": identity(2),
    "-X": gate_y_270(2),
    "-Y": gate_x_90(2),
    "-Z": sigma_x(2),
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


def _get_basis_transformations(basis_strings: List[str]):
    """
    Return a list of basis transformations given a list of basis strings
    """
    single_qubit_operations = [_gates_to_initilize[basis] for basis in basis_strings]
    multi_qubit_operations = [
        tensor_product(operations) for operations in product(single_qubit_operations)
    ]
    return jnp.stack(multi_qubit_operations, axis=0)


# @partial(jax.jit, static_argnames=("basis_strings", "n_qubits"))
def _basis_transformations_combinations(basis_strings, n_qubits):
    """
    Return the basis transformations for the Pauli operators
    """
    single_qubit_basis = _get_basis_transformations(basis_strings)
    multi_qubit_basis = [
        tensor_product(basis).reshape(2**n_qubits, 2**n_qubits)
        for basis in product(single_qubit_basis, repeat=n_qubits)
    ]

    return jnp.stack(multi_qubit_basis, axis=0)


def _apply_gates(states, transformations):
    """
    Apply the gates to the states
    """
    return jnp.einsum(
        "ijk, ...kl, ilm -> ...ijm",
        transformations,
        states,
        transformations.conj().transpose((0, 2, 1)),
    )


# State Preparation and Measurement Class
# Only added local noise for now
# TODO: Add multiqubit state preparation errors and measurements


class StatePreparation:

    def __init__(
        self,
        n_qubits: int = 1,
        initial_states=["Z"],
        perfect_state_preparation: bool = True,
    ):
        self.n_qubits = n_qubits
        self.initial_states = initial_states
        self.perfect_state_preparation = perfect_state_preparation

        if not perfect_state_preparation:
            self.state_preparation_params = jnp.zeros((self.n_qubits))

    def get_initial_state_generator(self):

        if self.perfect_state_preparation:

            @jax.jit
            def generator():
                return _pauli_state_combinations(self.initial_states, self.n_qubits)

        else:
            transformations = _basis_transformations_combinations(
                basis_strings=self.initial_states, n_qubits=self.n_qubits
            )

            @jax.jit
            def generator(state_preparation_params):
                # Find the mixed state contributions using temperature like softmax
                ground_states_diag = jnp.zeros((self.n_qubits, 2))
                ground_states_diag = ground_states_diag.at[:, 0].set(1.0)
                ground_states_diag = ground_states_diag.at[:, 1].set(
                    state_preparation_params
                )
                ground_states_diag = jax.nn.softmax(ground_states_diag, axis=-1)

                # Get the ground states for individual qubits
                ground_states = jax.vmap(jnp.diag)(ground_states_diag)
                ground_states = tensor_product(ground_states).reshape(
                    (2**self.n_qubits, 2**self.n_qubits)
                )

                # Apply the gates corresponding to initial gates
                initial_states = _apply_gates(ground_states, transformations)
                return initial_states

        return generator
