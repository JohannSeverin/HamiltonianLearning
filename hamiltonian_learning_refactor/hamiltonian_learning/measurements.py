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

#### Measurement class ####
from pathlib import Path


# First iteration only supports perfect measurements
from tensorflow_probability.substrates.jax.distributions import multinomial
from typing import Literal


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


def _extract_diag_from_changed_basis(states, basis_transformations):
    return jnp.einsum(
        "ijk, ...kl, ilj -> ...ij",
        basis_transformations,
        states,
        basis_transformations.conj().transpose((0, 2, 1)),
    )


# Loss functions. Should take probabilities, data and number of samples and ret


# Squared difference loss
def squared_loss(probs, data, samples):
    """
    Calculate the squared loss between the probabilities and the data
    """
    return jnp.sum((probs - data / samples) ** 2)


def multinomial_likelihood(probs, data, samples):
    """
    Calculate the multinomial likelihood
    """
    return (
        multinomial.Multinomial(
            total_count=samples,
            probs=probs,
        )
        .log_prob(data)
        .sum()
    )


available_losses = {
    "squared_difference": squared_loss,
    "multinomial": multinomial_likelihood,
}


class Measurements:

    def __init__(
        self,
        n_qubits: int,
        samples: int,
        basis: list[str] = ["Z"],
        perfect_measurement: bool = True,
        clip: float = 1e-10,
        loss: Literal["squared_difference", "multinomial"] = "squared_difference",
        batch_size: int = 1,
    ):
        self.n_qubits = n_qubits
        self.basis = basis
        self.perfect_measurement = perfect_measurement
        self.samples = samples

        self.params = None  # EMPTY FOR NOW
        self.clip = clip

        self.loss = loss
        self.batch_size = batch_size

    def get_loss_fn(self):
        if self.batch_size > 1:
            return self._get_looped_loss_fn()
        else:
            inner_func = available_losses[self.loss]

            # Defined for perfect measurements
            if self.perfect_measurement:

                # Get basis transformation
                basis_transformations = _get_basis_transformation_combinations(
                    self.basis, self.n_qubits
                )

                # The function to return
                @partial(jax.jit)
                def loss_fn(states, data):
                    probs = _extract_diag_from_changed_basis(
                        states, basis_transformations
                    ).real
                    return inner_func(probs, data, self.samples)

                # Return the function to be used
                return loss_fn

    def _get_looped_loss_fn(self):
        inner_func = available_losses[self.loss]

        if self.perfect_measurement:

            # Function allowing us to use jax.scan to separate data and measurement basis into batches
            def batched_function(carry, xs, states):
                # Unpack
                cumulative_loss = carry
                transformations, data = xs

                # Calculate the new loss
                state_diagonals = _extract_diag_from_changed_basis(
                    states, transformations
                ).real
                state_diagonals = jnp.moveaxis(state_diagonals, -2, 0)
                new_loss = inner_func(state_diagonals, data, self.samples)
                cumulative_loss += new_loss

                # Return cumulative loss and the new one for storing
                return cumulative_loss, new_loss

            basis_transformations = _get_basis_transformation_combinations(
                self.basis, self.n_qubits
            )

            @partial(jax.jit)
            def loss_fn(states, data):
                # move measurement basis to first position and split in batch size
                data_batched = jnp.moveaxis(data, -2, 0)

                data_batched = data_batched.reshape(
                    -1, self.batch_size, *data_batched.shape[1:]
                )
                basis_transformations_batched = basis_transformations.reshape(
                    -1, self.batch_size, *basis_transformations.shape[1:]
                )

                # Run the scan function
                cumulative_loss, _ = jax.lax.scan(
                    partial(batched_function, states=states),
                    0,
                    (basis_transformations_batched, data_batched),
                )

                return cumulative_loss

            return loss_fn

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
        if self.batch_size > 1:
            return self._looped_measurement_probabilities(states)

        if self.perfect_measurement:

            basis_transformations = _get_basis_transformation_combinations(
                self.basis, self.n_qubits
            )

            states = _apply_gates(states, basis_transformations)
            diag = jnp.einsum("...ii -> ...i", states)
            probs = jnp.clip(diag, self.clip, 1 - self.clip)

            return probs.real

    # @partial(jax.jit)
    def _looped_measurement_probabilities(self, states):

        def batched_function(carry, xs, states):
            # Unpack
            carry = None
            transformations = xs

            # Calculate the new loss
            state_diagonals = _extract_diag_from_changed_basis(
                states, transformations
            ).real

            probs = jnp.clip(state_diagonals, self.clip, 1 - self.clip)

            # No need for carry here. Return probs to be stored in collection
            return None, probs

        basis_transformations = _get_basis_transformation_combinations(
            self.basis, self.n_qubits
        )
        basis_transformations = basis_transformations.reshape(
            -1, self.batch_size, *basis_transformations.shape[1:]
        )

        _, probs = jax.lax.scan(
            partial(batched_function, states=states),
            None,
            basis_transformations,
        )
        probs = jnp.squeeze(jnp.moveaxis(probs, 0, -3))
        probs = probs.reshape(*probs.shape[:-3], -1, probs.shape[-1])

        return probs
