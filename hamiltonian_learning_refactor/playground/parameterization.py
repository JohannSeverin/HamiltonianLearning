import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax import jit
from optax import adam, apply_updates
from functools import partial

from pauli_string_product import letter_product, string_product, lookup


# One qubit example
hamiltonian_params = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64)
dissipative_params = (
    jnp.array([[0.25, -0.25, 0], [0.25, 0.25, 0], [0, 0, 0]], dtype=jnp.float64) / 10
)


@partial(jit, static_argnums=(1, 2, 3))
def _to_cholesky(tensors, connections, n_qubits, locality):
    """
    Generates the jump operators based on the Lindbladian parameters.
    """
    number_of_generators = 4**locality - 1
    tensors = tensors.reshape(
        (len(connections), number_of_generators, number_of_generators)
    )

    # The calzone
    output_cholesky = jnp.zeros(
        (len(connections), number_of_generators, number_of_generators),
        dtype=jnp.complex128,
    )

    output_cholesky += jnp.tril(tensors)
    output_cholesky += 1j * jnp.swapaxes(jnp.triu(tensors), -2, -1)

    return output_cholesky @ jnp.conj(output_cholesky).swapaxes(-2, -1)


# def to_1_qubit_chi(params):
params = hamiltonian_params, dissipative_params

hamiltonian_params = params[0]
lindbladian_params = params[1]


def to_chi_matrix(params):
    hamiltonian_params, lindbladian_params = params

    dissipative_block_matrix = _to_cholesky(
        tensors=lindbladian_params,
        connections=(0,),
        n_qubits=1,
        locality=1,
    )[0]

    chi_matrix = jnp.zeros((4, 4), dtype=jnp.complex128)

    # Hamiltonian part
    chi_matrix = chi_matrix.at[0, 1:].set(1j * hamiltonian_params)
    chi_matrix = chi_matrix.at[1:, 0].set(-1j * hamiltonian_params)

    # Dissipative part
    chi_matrix = chi_matrix.at[1:, 1:].set(dissipative_block_matrix)

    # Normalize trace
    chi_matrix = chi_matrix.at[0, 0].set(-jnp.trace(chi_matrix[1:, 1:]))

    # Contribution from dissipative part on the hamiltnonian terms are found by
    normlization_terms = (
        jnp.einsum(
            "ij, kij->k", jnp.imag(dissipative_block_matrix), lookup()[1:, 1:, 1:]
        )
        / 2
    )

    chi_matrix = chi_matrix.at[0, 1:].add(-normlization_terms)

    chi_matrix = chi_matrix.at[1:, 0].add(-normlization_terms)

    return chi_matrix


### Do experiment with 1 qubit
import xarray

dataset = xarray.open_dataarray(
    "/root/projects/HamiltonianLearning/old/illustrations/1 Qubit Hamiltonian/DatasetLindblad.nc"
)

data = dataset.values[..., 1]

# Load hyper params
NQUBITS = 1
SAMPLES = 100
DURATION = 10000
TIMESTEPS = 20
LEARNING_RATE = 1e-4
ITERATIONS = 200

times = dataset.time.values

# Initial states


# Write the loss function
from tensorflow_probability.substrates.jax.distributions import Binomial

from super_operators import chi_matrix_to_pauli_transfer_matrix
from super_states import density_matrix_to_superket, superket_to_density_matrix


def negative_log_likelihood(params):
    chi_matrix = to_chi_matrix(params)
    pauli_tranfer_matrix_generator = chi_matrix_to_pauli_transfer_matrix(
        chi_matrix, NQUBITS
    )

    generators = jnp.einsum("ij, t-> tij", pauli_tranfer_matrix_generator, times)
    channels = jax.scipy.linalg.expm(generators)

    probs = get_probability_from_states(result_in_measurement_basis)[..., 1]

    LLH = Binomial(total_count=SAMPLES, probs=probs).log_prob(data).sum()

    return -LLH
