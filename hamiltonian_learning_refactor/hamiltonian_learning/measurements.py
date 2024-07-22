import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.operators import sigma_x, sigma_y, sigma_z, identity
from utils.operators import gate_y_90, gate_x_270, identity, gate_y_270, gate_x_90
from utils.tensor import tensor_product
from functools import partial

from itertools import product

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from typing import List


# Operator to define states from strings
_basis_transformations_names = {
    "X": gate_y_270(2),
    "Y": gate_x_90(2),
    "Z": identity(2),
}


def _get_basis_transformations(basis_strings: List[str]):
    """
    Return a list of basis transformations given a list of basis strings
    """
    return [_basis_transformations_names[basis] for basis in basis_strings]


def _get_basis_transformation_combinations(basis_strings, n_qubits):
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


#### Measurement class ####
from pathlib import Path


# First iteration only supports perfect measurements
from tensorflow_probability.substrates.jax.distributions import multinomial


class Measurements:

    def __init__(
        self,
        n_qubits: int,
        basis: list[str] = ["Z"],
        perfect_measurement: bool = True,
        clip: float = 1e-10,
    ):
        self.n_qubits = n_qubits
        self.basis = basis
        self.perfect_measurement = perfect_measurement

        self.params = None  # EMPTY FOR NOW
        self.clip = clip

    def get_squared_difference_function(self, equal_weights: bool = False):

        if self.perfect_measurement:

            basis_transformations = _get_basis_transformation_combinations(
                self.basis, self.n_qubits
            )

        def _basis_change(states):
            return _apply_gates(states, basis_transformations)

        def _extract_diag(states):
            return jnp.einsum("...ii -> ...i", states)

        @partial(jax.jit, static_argnums=(2))
        def _squared_diffs(states, data, samples):
            states = _basis_change(states)
            diag = _extract_diag(states)
            probs = jnp.clip(diag.real, self.clip, 1 - self.clip)

            std_estimate = jnp.sqrt(probs * (1 - probs)) / jnp.sqrt(samples)

            if equal_weights:
                diffs = (probs - data / samples) ** 2
            else:
                diffs = (probs - data / samples) ** 2 / std_estimate**2

            return jnp.sum(diffs)

        return _squared_diffs

    def get_log_likelihood_function(self):

        if self.perfect_measurement:

            basis_transformations = _get_basis_transformation_combinations(
                self.basis, self.n_qubits
            )

            def _basis_change(states):
                return _apply_gates(states, basis_transformations)

            def _extract_diag(states):
                return jnp.einsum("...ii -> ...i", states)

            @partial(jax.jit, static_argnums=(2))
            def _log_prob(states, data, samples):
                states = _basis_change(states)
                diag = _extract_diag(states)
                probs = jnp.clip(diag.real, self.clip, 1 - self.clip)

                log_prob_for_measurement = multinomial.Multinomial(
                    total_count=samples,
                    probs=probs,
                ).log_prob(data)

                return -jnp.sum(log_prob_for_measurement)

            return _log_prob

    def generate_samples(self, states, samples: int, key: int = 0):

        if self.perfect_measurement:

            basis_transformations = _get_basis_transformation_combinations(
                self.basis, self.n_qubits
            )

            states = _apply_gates(states, basis_transformations)
            diag = jnp.einsum("...ii -> ...i", states).real
            probs = jnp.clip(diag, self.clip, 1 - self.clip)

            return multinomial.Multinomial(total_count=samples, probs=probs).sample(
                seed=jax.random.PRNGKey(key)
            )

    def calculate_measurement_probabilities(self, states):

        if self.perfect_measurement:

            basis_transformations = _get_basis_transformation_combinations(
                self.basis, self.n_qubits
            )

            states = _apply_gates(states, basis_transformations)
            diag = jnp.einsum("...ii -> ...i", states)
            probs = jnp.clip(diag, self.clip, 1 - self.clip)

            return probs.real
