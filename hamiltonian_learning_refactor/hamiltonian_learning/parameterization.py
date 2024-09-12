# Sketch at the moment to generate the parameterization of the Lindblad Master Equation
from typing import Union, List, Tuple
import jax.numpy as jnp
import itertools
import os.path as osp
from functools import partial

from jax import jit
import jax
import jaxtyping

jax.config.update("jax_enable_x64", True)

import sys

sys.path.append(
    osp.join(osp.dirname(osp.abspath(__file__)), "..")
)  # Add the path to the parent directory
from utils.tensor import tensor, _add_matrix_to_tensor
from utils.operators import sigma_x, sigma_y, sigma_z, identity


### FUNCTIONS FOR GENERATING HAMILTONIAN GIVEN THE STRUCTURE ###
@partial(jit, static_argnums=0)
def _pauli_product(locality: int):
    """
    Generate the Pauli product for a given locality.
    """
    return jnp.array(
        [
            tensor(*pauli)
            for pauli in itertools.product(
                [sigma_x(), sigma_y(), sigma_z()], repeat=locality
            )
        ]
    )  # .reshape(*shape)


@partial(jit, static_argnums=(1, 2, 3))
def _convert_pauli_to_hamiltonians(tensors, connections, n_qubits, locality):
    """
    Takes the Pauli product representation and converts to hamiltonians describing the dynamics
    """

    number_of_connections = len(connections)
    pauli_matrices = _pauli_product(locality)

    tensors = tensors.reshape((number_of_connections, 3**locality))

    return jnp.einsum("cp, pij -> cij", tensors, pauli_matrices)


@partial(jit, static_argnums=(1, 2, 3, 4))
def _convert_pauli_to_hamiltonians_with_time(
    tensors, connections, n_qubits, locality, timesteps
):

    number_of_connections = len(connections)
    pauli_matrices = _pauli_product(locality)

    tensors = tensors.reshape((timesteps, number_of_connections, 3**locality))

    return jnp.einsum("tcp, pij -> tcij", tensors, pauli_matrices)


@partial(jit, static_argnums=(1, 2, 3))
def _sum_interaction_hamiltonian_to_vector(tensors, connections, n_qubits, locality):
    shape = (4,) * n_qubits
    output = jnp.zeros(shape, dtype=jnp.float64)

    # Loop over connection elements
    for i in range(len(connections)):

        # Need to calculate the index order for permuting the tensor so the desired qubits interact
        order = [-1] * n_qubits
        connection = connections[i]

        for j, conn in enumerate(connection):
            order[conn] = j

        fill_with = list(range(locality, n_qubits))

        for j, o in enumerate(order):
            if o == -1:
                order[j] = fill_with.pop(0)

        # create the Hamiltonian contribution

        H = jnp.zeros((4,) * locality, dtype=jnp.float64)

        slices = [slice(1, None)] * locality

        H = H.at[*slices].set(tensors[i].reshape((3,) * locality))

        # Fill with zeros to have proper hilbert space
        for j in range(n_qubits - locality):
            dummy_H = jnp.zeros_like(H)
            H = jnp.stack([H, dummy_H, dummy_H, dummy_H], axis=-1)

        H = H.transpose(order)

        output += H

    return output


@partial(jit, static_argnums=(1, 2, 3))
def _sum_interaction_hamiltonian(tensors, connections, n_qubits, locality):
    shape = (2,) * (2 * n_qubits)
    output = jnp.zeros(shape, dtype=jnp.complex128)

    # Loop over connection elements
    for i in range(len(connections)):

        # Need to calculate the index order for permuting the tensor so the desired qubits interact
        order = [-1] * n_qubits
        connection = connections[i]

        for j, conn in enumerate(connection):
            order[conn] = j

        fill_with = list(range(locality, n_qubits))

        for j, o in enumerate(order):
            if o == -1:
                order[j] = fill_with.pop(0)

        order = order + [n_qubits + i for i in order]

        # Create the Hamiltonian contribution
        H = tensors[i].reshape((2,) * locality * 2)

        # Fill with identity to have proper hilbert space
        for j in range(n_qubits - locality):
            H = _add_matrix_to_tensor(H, jnp.eye(2))

        H = H.transpose(order)

        output += H

    return output


@partial(jit, static_argnums=(1, 2, 3, 4))
def _sum_interaction_hamiltonian_with_time(
    tensors, connections, n_qubits, locality, timesteps
):
    shape = (timesteps,) + (2,) * (2 * n_qubits)
    output = jnp.zeros(shape, dtype=jnp.complex128)

    # Loop over connection elements
    for i in range(len(connections)):

        # Need to calculate the index order for permuting the tensor so the desired qubits interact
        order = [-1] * n_qubits
        connection = connections[i]

        for j, conn in enumerate(connection):
            order[conn] = j

        fill_with = list(range(locality, n_qubits))

        for j, o in enumerate(order):
            if o == -1:
                order[j] = fill_with.pop(0)

        order = order + [n_qubits + i for i in order]
        order = [0] + [i + 1 for i in order]

        # Create the Hamiltonian contribution
        H = tensors[:, i].reshape((timesteps,) + (2,) * locality * 2)

        # Fill with identity to have proper hilbert space
        for j in range(n_qubits - locality):
            H = jax.vmap(_add_matrix_to_tensor, in_axes=(0, None))(H, jnp.eye(2))

        H = H.transpose(order)

        output += H

    return output


### FUNCTIONS FOR GENERATING JUMP OPERATORS GIVEN THE STRUCTURE ###
def _calculate_number_of_generators(lindblad_graph: dict):
    """
    Calculates the number of generators for the Lindbladian.
    """
    # TODO: CORRECT FOR IDENTITY OVERCOUNTING

    number_of_jumps = 0

    for locality, connections in lindblad_graph.items():
        number_of_jumps += len(connections) * 4**locality

    return number_of_jumps


@partial(jit, static_argnums=(0,))
def _pauli_product_with_id(locality: int):
    """
    Generate the Pauli product for a given locality.
    """
    return jnp.array(
        [
            tensor(*pauli)
            for pauli in itertools.product(
                [identity(), sigma_x(), sigma_y(), sigma_z()], repeat=locality
            )
        ]
    )


@partial(jit, static_argnums=(1, 2, 3))
def _to_cholesky_no_dummy(tensors, connections, n_qubits, locality):
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

    return output_cholesky


@partial(jit, static_argnums=(1, 2, 3))
def _to_cholesky(tensors, connections, n_qubits, locality):
    """
    Generates the jump operators based on the Lindbladian parameters.
    """
    number_of_generators = 4**locality
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

    return output_cholesky


@partial(jit, static_argnums=(1, 2, 3))
def _cholesky_to_jumps(tensors, connections, n_qubits, locality):
    """
    Converts the Cholesky decomposition to jump operators.
    """
    number_of_connections = len(connections)
    tensors = tensors.reshape(number_of_connections, 4**locality, 4**locality)
    paulis = _pauli_product_with_id(locality)

    jump_operators = jnp.einsum("cij, jkl -> cikl", tensors, paulis).reshape(
        (number_of_connections, 4**locality)
        + (
            2,
            2,
        )
        * locality
    )

    return jump_operators


_add_matrix_to_tensor_vmap = jax.vmap(_add_matrix_to_tensor, in_axes=(0, None))


@partial(jit, static_argnums=(1, 2, 3))
def _sum_interaction_jump_operators(tensors, connections, n_qubits, locality):
    """
    Sums the interaction jump operators.
    """

    # Counts of connections and jump operators
    number_of_connections = len(connections)
    number_of_jumps_per_connection = 4**locality
    number_of_jumps = number_of_connections * number_of_jumps_per_connection

    # TODO: Strip identities from tensors
    # tensors = tensors[..., 1:, 1:]

    # Full shape
    shape = (number_of_jumps,) + (2**n_qubits, 2**n_qubits)
    output = jnp.zeros(shape, dtype=jnp.complex128)

    for i in range(len(connections)):
        # Next few lines are sorting the indices for transposing the final tensor.
        order = [-1] * n_qubits
        connection = connections[i]

        for j, conn in enumerate(connection):
            order[conn] = j

        fill_with = list(range(locality, n_qubits))

        for j, o in enumerate(order):
            if o == -1:
                order[j] = fill_with.pop(0)

        # Add second matrix idx
        order = order + [n_qubits + i for i in order]

        # Push with one, because we have multiple jump operators
        order = [
            0,
        ] + [o + 1 for o in order]

        # Create the Lindbladian jumps
        Ls = tensors[i].reshape((number_of_jumps_per_connection,) + (2,) * locality * 2)

        # Upconvert matrix to include all other qubits
        for j in range(n_qubits - locality):
            Ls = _add_matrix_to_tensor_vmap(Ls, jnp.eye(2))

        # Sort the order of the qubits
        Ls = Ls.transpose(order)
        Ls = Ls.reshape((number_of_jumps_per_connection, 2**n_qubits, 2**n_qubits))

        # Add to the returning array
        output = output.at[
            i
            * number_of_jumps_per_connection : (i + 1)
            * number_of_jumps_per_connection
        ].set(Ls)

    # Out put the reshaped
    return output


def _sum_interaction_dissipation_matrices(tensors, connections, n_qubits, locality):
    shape = (4,) * (2 * n_qubits)
    output = jnp.zeros(shape, dtype=jnp.complex128)

    # Loop over connection elements
    for i in range(len(connections)):

        # Need to calculate the index order for permuting the tensor so the desired qubits interact
        order = [-1] * n_qubits
        connection = connections[i]

        for j, conn in enumerate(connection):
            order[conn] = j

        fill_with = list(range(locality, n_qubits))

        for j, o in enumerate(order):
            if o == -1:
                order[j] = fill_with.pop(0)

        order = order + [n_qubits + i for i in order]

        # Create the Hamiltonian contribution
        local_dissipation_matrix = jnp.zeros(
            (4**n_qubits, 4**n_qubits), dtype=jnp.complex128
        )
        local_dissipation_matrix = local_dissipation_matrix.at[
            1 : 4**locality, 1 : 4**locality
        ].set(tensors[i][1:, 1:])
        local_dissipation_matrix = local_dissipation_matrix.reshape(
            (4,) * (n_qubits * 2)
        )

        local_dissipation_matrix = local_dissipation_matrix.transpose(order)

        output += local_dissipation_matrix

    return output


# Transformation between chi and PTM matrices
from utils.operators import sigma_x, sigma_y, sigma_z, identity

single_qubit_pauli = jnp.stack([identity(2), sigma_x(2), sigma_y(2), sigma_z(2)])


@jax.jit
def levi_cevita():
    epsilon = jnp.zeros((3, 3, 3), dtype=jnp.int8)
    epsilon = epsilon.at[0, 1, 2].set(1)
    epsilon = epsilon.at[1, 2, 0].set(1)
    epsilon = epsilon.at[2, 0, 1].set(1)
    epsilon = epsilon.at[0, 2, 1].set(-1)
    epsilon = epsilon.at[1, 0, 2].set(-1)
    epsilon = epsilon.at[2, 1, 0].set(-1)
    return epsilon


@jax.jit
def lookup():
    lookup = jnp.zeros((4, 4, 4), dtype=jnp.int8)

    # Setup i not equal to j
    lookup = lookup.at[1:, 1:, 1:].set(levi_cevita())

    # Setup i equal to j
    lookup = lookup.at[0, 0, 0].set(1)
    lookup = lookup.at[1, 1, 0].set(1)
    lookup = lookup.at[2, 2, 0].set(1)
    lookup = lookup.at[3, 3, 0].set(1)

    # Setup i = 0 and j = 1, 2, 3
    lookup = lookup.at[0, 1, 1].set(1)
    lookup = lookup.at[0, 2, 2].set(1)
    lookup = lookup.at[0, 3, 3].set(1)

    # Setup j = 0 and i = 1, 2, 3
    lookup = lookup.at[1, 0, 1].set(1)
    lookup = lookup.at[2, 0, 2].set(1)
    lookup = lookup.at[3, 0, 3].set(1)

    return lookup


def lookup_for_qubit_number(n_qubits: int):
    """
    Returns the lookup table for a given number of qubits.
    """
    lookup_multiple_qubits = lookup()
    for i in range(n_qubits - 1):
        lookup_multiple_qubits = jnp.kron(lookup_multiple_qubits, lookup())

    return lookup_multiple_qubits


def _chi_normalization(dissipation_matrix: jnp.ndarray, n_qubits: int):
    """
    Normalizes the chi matrix.
    """
    dissipation_matrix = dissipation_matrix.at[0, 0].set(
        -jnp.trace(dissipation_matrix[1:, 1:])
    )

    return dissipation_matrix


@partial(jax.jit, static_argnums=0)
def _chi_to_pauli_matrix(nqubits):
    """
    Return the Pauli transfer matrix

    TODO: This can be simplified significantly by finding the matrix from looking at indicies
    """
    single_qubit_states = single_qubit_pauli

    single_qubit_trace_combination = jnp.einsum(
        "iab, jbc, kcd, lda->ijkl",
        single_qubit_states,
        single_qubit_states,
        single_qubit_states,
        single_qubit_states,
    )

    if nqubits == 1:
        return single_qubit_trace_combination
    else:
        multi_qubit_trace_combination = single_qubit_trace_combination.copy()
        for qubit in range(1, nqubits):
            dimension = 4**qubit

            multi_qubit_trace_combination = multi_qubit_trace_combination.reshape(
                [dimension, 1] * 4
            ) * single_qubit_trace_combination.reshape([1, 4] * 4)

            multi_qubit_trace_combination = multi_qubit_trace_combination.reshape(
                [4 * dimension] * 4
            )
    return multi_qubit_trace_combination


@partial(jax.jit, static_argnums=1)
def chi_matrix_to_pauli_transfer_matrix(transfer_matrix, nqubits):
    """
    Convert a transfer matrix to a Pauli transfer matrix
    """
    basis_change_matrix = _chi_to_pauli_matrix(nqubits) / 2**nqubits

    return jnp.einsum("ijkl, ...jl->...ik", basis_change_matrix, transfer_matrix)


### PARAMETRIZATION CLASS ###
class SuperOperatorParameterization:

    def __init__(
        self,
        n_qubits: int,
        qubit_levels: int = 2,
        hamiltonian_locality: int = 0,
        lindblad_locality: int = 0,
        hamiltonian_graph: dict = {},
        lindblad_graph: dict = {},
        hamiltonian_amplitudes: list[float] = [],
        lindblad_amplitudes: list[float] = [],
    ):

        # Load Params
        self.n_qubits = n_qubits
        self.qubit_levels = qubit_levels
        self.hilbert_size = self.qubit_levels**n_qubits
        self.generators = self.qubit_levels**2  # not including identity

        # Set values to locality
        self.hamiltonian_locality = hamiltonian_locality
        self.lindblad_locality = lindblad_locality

        # Or overwrite with graph
        self.hamiltonian_graph = hamiltonian_graph
        self.lindblad_graph = lindblad_graph

        # Generate graphs if not given
        if len(hamiltonian_graph) == 0:
            self.hamiltonian_graph = {
                i: tuple(itertools.combinations(range(n_qubits), r=i))
                for i in range(1, self.hamiltonian_locality + 1)
            }

        if len(lindblad_graph) == 0:
            self.lindblad_graph = {
                i: tuple(itertools.combinations(range(n_qubits), r=i))
                for i in range(1, self.lindblad_locality + 1)
            }

        self.number_of_jump_operators = _calculate_number_of_generators(
            lindblad_graph=self.lindblad_graph
        )

        # Generate Hamiltonian Parameters

        self.hamiltonian_params = self._generate_hamiltonian_params(
            hamiltonian_amplitudes
        )
        self.lindbladian_params = self._generate_lindbladian_params(lindblad_amplitudes)

    def _generate_hamiltonian_params(self, amplitude={}, seed=0):
        """
        Generates the Hamiltonian parameters based on the Hamiltonian graph.

        Returns:
            dict: A dictionary containing the Hamiltonian parameters for each locality.
        """
        hamiltonian_params = {
            1: jnp.zeros((self.n_qubits, self.generators), dtype=jnp.complex128)
        }

        hamiltonian_connections = {
            locality: len(graph) for locality, graph in self.hamiltonian_graph.items()
        }
        hamiltonian_params_shape = {
            locality: [hamiltonian_connections[locality]] + [self.generators] * locality
            for locality in hamiltonian_connections
        }
        hamiltonian_params.update(
            {
                locality: jnp.zeros(
                    hamiltonian_params_shape[locality], dtype=jnp.complex128
                )
                for locality in hamiltonian_connections
            }
        )

        if len(amplitude) > 0:
            key = jax.random.PRNGKey(seed)
            keys = jax.random.split(key, self.hamiltonian_locality)

            for locality in hamiltonian_params:
                hamiltonian_params[locality] = amplitude[locality] * jax.random.normal(
                    keys[locality], hamiltonian_params[locality].shape
                )

        return hamiltonian_params

    def _generate_lindbladian_params(self, amplitudes=None, seed=0):
        """
        Generates the Lindbladian parameters based on the Lindbladian graph.

        Returns:
            dict: A dictionary containing the Lindbladian parameters for each locality.
        """
        lindbladian_params = {
            1: jnp.zeros(
                (self.n_qubits, self.generators, self.generators),
                dtype=jnp.float64,
            )
        }

        lindbladian_connections = {
            locality: len(graph) for locality, graph in self.lindblad_graph.items()
        }
        lindbladian_params_shape = {
            locality: [lindbladian_connections[locality]]
            + [self.generators] * 2 * locality
            for locality in lindbladian_connections
        }
        lindbladian_params.update(
            {
                locality: jnp.zeros(
                    lindbladian_params_shape[locality], dtype=jnp.float64
                )
                for locality in lindbladian_connections
            }
        )

        if amplitudes:
            key = jax.random.PRNGKey(seed)
            keys = jax.random.split(key, self.hamiltonian_locality)

            for locality in lindbladian_params:
                lindbladian_params[locality] = amplitudes[locality] * jax.random.normal(
                    keys[locality], lindbladian_params[locality].shape
                )

        return lindbladian_params

    def set_hamiltonian_params(self, hamiltonian_params: dict):
        """
        Sets the Hamiltonian parameters.
        """

        for locality, params_at_locality in hamiltonian_params.items():
            if isinstance(params_at_locality, dict):

                print("Settings params will assume only one connection")
                for key, param in params_at_locality.items():
                    # Convert key to index
                    idx = ["xyz".index(key[i]) for i in range(len(key))]
                    self.hamiltonian_params[locality] = (
                        self.hamiltonian_params[locality].at[0, *idx].set(param)
                    )

            else:
                self.hamiltonian_params[locality] = params_at_locality

    def set_lindbladian_params(self, lindbladian_params: dict):
        """
        Sets the Lindbladian parameters.
        """

        for locality, params_at_locality in lindbladian_params.items():
            if isinstance(params_at_locality, dict):

                for connection, params_at_connection in params_at_locality.items():

                    print("Settings params will assume only one connection")
                    for key, param in params_at_locality.items():
                        # Convert key to index
                        idx = ["ixyz".index(key[i]) for i in range(len(key))]
                        self.lindbladian_params[locality] = (
                            self.lindbladian_params[locality].at[0, *idx].set(param)
                        )

            else:
                self.lindbladian_params[locality] = params_at_locality

    def _get_hamiltonian_generator(self):

        def _hamiltonian_generator(parameters):
            """
            Return a list of all the coefficients in front of paulis in the Hamiltonian
            """
            hamiltonian_coefficients = jnp.zeros(
                (4,) * self.n_qubits, dtype=jnp.float64
            )

            for locality in range(1, self.hamiltonian_locality + 1):
                hamiltonian_coefficients += _sum_interaction_hamiltonian_to_vector(
                    parameters[locality],
                    self.hamiltonian_graph[locality],
                    self.n_qubits,
                    locality,
                )

            return hamiltonian_coefficients.reshape(4**self.n_qubits)

        return _hamiltonian_generator

    def _get_dissipation_matrix_generator(self):

        def lindbladian_generator(lindbladian_params: dict):
            """
            Creates the jump operators based on the Lindbladian parameters.
            """
            dissipation_matrix = jnp.zeros(
                (4, 4) * self.n_qubits,
                dtype=jnp.complex128,
            )

            for locality in range(1, self.lindblad_locality + 1):

                choleskies = _to_cholesky(
                    lindbladian_params[locality],
                    self.lindblad_graph[locality],
                    self.n_qubits,
                    locality,
                )

                local_dissipation_matrix = jnp.einsum(
                    "...ij, ...il -> ...jl",
                    choleskies,
                    jnp.conj(choleskies),
                )

                inflated_dissipation_matrix = _sum_interaction_dissipation_matrices(
                    local_dissipation_matrix,
                    self.lindblad_graph[locality],
                    self.n_qubits,
                    locality,
                )

                dissipation_matrix += inflated_dissipation_matrix

            dissipation_matrix = dissipation_matrix.reshape(
                4**self.n_qubits, 4**self.n_qubits
            )[1:, 1:]

            return dissipation_matrix

        return lindbladian_generator

    def get_generator_for_pauli_transfer_matrix(self):

        hamiltonian_vector_generator = self._get_hamiltonian_generator()
        dissipation_matrix_generator = self._get_dissipation_matrix_generator()

        lookup_table = lookup_for_qubit_number(self.n_qubits)

        def create_dchi(hamiltonian_params, lindbladian_params):
            """
            Create the differential chi matrix
            """

            dchi = jnp.zeros((4**self.n_qubits, 4**self.n_qubits), dtype=jnp.complex128)

            # Set the Hamiltonian part
            hamiltonian_vector = hamiltonian_vector_generator(hamiltonian_params)
            dchi = dchi.at[1:, 0].set(1j * hamiltonian_vector[1:])
            dchi = dchi.at[0, 1:].set(-1j * hamiltonian_vector[1:])

            # Dissipation part
            dissipation_matrix = dissipation_matrix_generator(lindbladian_params)
            dchi = dchi.at[1:, 1:].set(dissipation_matrix)

            # Trace preservation
            dchi = dchi.at[0, 0].set(-jnp.trace(dissipation_matrix[1:, 1:]))

            # Normalization
            normalization_contribution = jnp.einsum(
                "ij, kij->k", dissipation_matrix, lookup_table[1:, 1:, 1:]
            )

            dchi = dchi.at[0, 1:].add(-normalization_contribution)
            dchi = dchi.at[1:, 0].add(-normalization_contribution)

            return dchi

        def transfer_generator(hamiltonian_params, lindbladian_params):
            """
            Returns the Pauli transfer matrix
            """
            dchi = create_dchi(hamiltonian_params, lindbladian_params)

            return chi_matrix_to_pauli_transfer_matrix(dchi, self.n_qubits)

        return transfer_generator


class Parameterization:
    # TODO: Add docstrings
    # TODO: Add type hints
    # TODO: Add random initialization to make guesses
    # TODO: Add function to set certain parameters and ease the extraction of results

    def __init__(
        self,
        n_qubits: int,
        qubit_levels: int = 2,
        hamiltonian_locality: int = 0,
        lindblad_locality: int = 0,
        hamiltonian_graph: dict = {},
        lindblad_graph: dict = {},
        hamiltonian_amplitudes: list[float] = [],
        lindblad_amplitudes: list[float] = [],
        seed: int = 0,
    ):
        assert hamiltonian_locality is not None or hamiltonian_graph is not None
        assert lindblad_locality is not None or lindblad_graph is not None

        # Load Params
        self.n_qubits = n_qubits
        self.qubit_levels = qubit_levels
        self.hilbert_size = self.qubit_levels**n_qubits
        self.generators = self.qubit_levels**2 - 1  # not including identity

        # Set values to locality
        self.hamiltonian_locality = hamiltonian_locality
        self.lindblad_locality = lindblad_locality

        # Or overwrite with graph
        self.hamiltonian_graph = hamiltonian_graph
        self.lindblad_graph = lindblad_graph

        # Generate graphs if not given
        if len(hamiltonian_graph) == 0:
            self.hamiltonian_graph = {
                i: tuple(itertools.combinations(range(n_qubits), r=i))
                for i in range(1, self.hamiltonian_locality + 1)
            }

        if len(lindblad_graph) == 0:
            self.lindblad_graph = {
                i: tuple(itertools.combinations(range(n_qubits), r=i))
                for i in range(1, self.lindblad_locality + 1)
            }

        self.number_of_jump_operators = _calculate_number_of_generators(
            lindblad_graph=self.lindblad_graph
        )

        # Generate Hamiltonian Parameters
        self.seed = seed
        self.hamiltonian_params = self._generate_hamiltonian_params(
            hamiltonian_amplitudes
        )
        self.lindbladian_params = self._generate_lindbladian_params(lindblad_amplitudes)

    def _generate_hamiltonian_params(self, amplitude={}):
        """
        Generates the Hamiltonian parameters based on the Hamiltonian graph.

        Returns:
            dict: A dictionary containing the Hamiltonian parameters for each locality.
        """
        hamiltonian_params = {
            1: jnp.zeros((self.n_qubits, self.generators), dtype=jnp.complex128)
        }

        hamiltonian_connections = {
            locality: len(graph) for locality, graph in self.hamiltonian_graph.items()
        }
        hamiltonian_params_shape = {
            locality: [hamiltonian_connections[locality]] + [self.generators] * locality
            for locality in hamiltonian_connections
        }
        hamiltonian_params.update(
            {
                locality: jnp.zeros(
                    hamiltonian_params_shape[locality], dtype=jnp.complex128
                )
                for locality in hamiltonian_connections
            }
        )

        if len(amplitude) > 0:
            key = jax.random.PRNGKey(self.seed)
            keys = jax.random.split(key, self.hamiltonian_locality)

            for locality in hamiltonian_params:
                hamiltonian_params[locality] = amplitude[
                    locality - 1
                ] * jax.random.normal(
                    keys[locality], hamiltonian_params[locality].shape
                )

        return hamiltonian_params

    def _generate_lindbladian_params(self, amplitudes=None):
        """
        Generates the Lindbladian parameters based on the Lindbladian graph.

        Returns:
            dict: A dictionary containing the Lindbladian parameters for each locality.
        """
        lindbladian_params = {
            1: jnp.zeros(
                (self.n_qubits, self.generators + 1, self.generators + 1),
                dtype=jnp.float64,
            )
        }

        lindbladian_connections = {
            locality: len(graph) for locality, graph in self.lindblad_graph.items()
        }
        lindbladian_params_shape = {
            locality: [lindbladian_connections[locality]]
            + [self.generators + 1] * 2 * locality
            for locality in lindbladian_connections
        }
        lindbladian_params.update(
            {
                locality: jnp.zeros(
                    lindbladian_params_shape[locality], dtype=jnp.float64
                )
                for locality in lindbladian_connections
            }
        )

        if amplitudes:
            key = jax.random.PRNGKey(self.seed)
            keys = jax.random.split(key, self.hamiltonian_locality)

            for locality in lindbladian_params:
                lindbladian_params[locality] = amplitudes[
                    locality - 1
                ] * jax.random.normal(
                    keys[locality], lindbladian_params[locality].shape
                )

        return lindbladian_params

    def set_hamiltonian_params(self, hamiltonian_params: dict):
        """
        Sets the Hamiltonian parameters.
        """

        for locality, params_at_locality in hamiltonian_params.items():
            if isinstance(params_at_locality, dict):

                print("Settings params will assume only one connection")
                for key, param in params_at_locality.items():
                    # Convert key to index
                    idx = ["xyz".index(key[i]) for i in range(len(key))]
                    self.hamiltonian_params[locality] = (
                        self.hamiltonian_params[locality].at[0, *idx].set(param)
                    )

            else:
                self.hamiltonian_params[locality] = params_at_locality

    def set_lindbladian_params(self, lindbladian_params: dict):
        """
        Sets the Lindbladian parameters.
        """

        for locality, params_at_locality in lindbladian_params.items():
            if isinstance(params_at_locality, dict):

                for connection, params_at_connection in params_at_locality.items():

                    print("Settings params will assume only one connection")
                    for key, param in params_at_locality.items():
                        # Convert key to index
                        idx = ["ixyz".index(key[i]) for i in range(len(key))]
                        self.lindbladian_params[locality] = (
                            self.lindbladian_params[locality].at[0, *idx].set(param)
                        )

            else:
                self.lindbladian_params[locality] = params_at_locality

    def get_hamiltonian_generator(self):
        """
        Creates a jitted function that converts the Hamiltonian parameters to a Hamiltonian matrix.
        """

        @partial(jit)
        def hamiltonian_generator(hamiltonian_params: dict):
            """
            Creates the Hamiltonian based on the Hamiltonian parameters.
            """
            hamiltonian = jnp.zeros((self.hilbert_size, self.hilbert_size))

            for locality in range(1, self.hamiltonian_locality + 1):
                hamiltonians = _convert_pauli_to_hamiltonians(
                    hamiltonian_params[locality],
                    self.hamiltonian_graph[locality],
                    self.n_qubits,
                    locality,
                )
                hamiltonian += _sum_interaction_hamiltonian(
                    hamiltonians,
                    self.hamiltonian_graph[locality],
                    self.n_qubits,
                    locality,
                ).reshape((self.hilbert_size, self.hilbert_size))

            return hamiltonian

        return hamiltonian_generator

    def get_jump_operator_generator(self):
        """
        Creates the jump operators based on the Lindbladian parameters.
        The structure is jitted out.
        """

        @partial(jit)
        def lindbladian_generator(lindbladian_params: dict):
            """
            Creates the jump operators based on the Lindbladian parameters.
            """
            jump_operators = jnp.zeros(
                (self.number_of_jump_operators, self.hilbert_size, self.hilbert_size),
                dtype=jnp.complex128,
            )
            filled_untill_index = 0

            for locality in range(1, self.lindblad_locality + 1):
                choleskies = _to_cholesky(
                    lindbladian_params[locality],
                    self.lindblad_graph[locality],
                    self.n_qubits,
                    locality,
                )
                jumps = _cholesky_to_jumps(
                    choleskies, self.lindblad_graph[locality], self.n_qubits, locality
                )

                operators = _sum_interaction_jump_operators(
                    jumps, self.lindblad_graph[locality], self.n_qubits, locality
                )

                number_jumps_at_locality = len(operators)

                jump_operators = jump_operators.at[
                    filled_untill_index : filled_untill_index + number_jumps_at_locality
                ].set(operators)
                filled_untill_index += number_jumps_at_locality

            return jump_operators

        return lindbladian_generator


# Time dependent Hamiltonian
class InterpolatedParameterization(Parameterization):

    def __init__(
        self,
        n_qubits: int,
        qubit_levels: int = 2,
        times: jaxtyping.Array = jnp.array([0.0, 1.0]),
        time_dependent_hamiltonian_locality: int = 0,
        hamiltonian_locality: int = 0,
        lindblad_locality: int = 0,
        hamiltonian_graph: dict = {},
        lindblad_graph: dict = {},
        hamiltonian_amplitudes: dict[int, float] = {},
        lindblad_amplitudes: dict[int, float] = {},
    ):
        self.number_of_interpolation_points = len(times)

        super().__init__(
            n_qubits=n_qubits,
            qubit_levels=qubit_levels,
            hamiltonian_locality=hamiltonian_locality,
            lindblad_locality=lindblad_locality,
            hamiltonian_graph=hamiltonian_graph,
            lindblad_graph=lindblad_graph,
            hamiltonian_amplitudes=hamiltonian_amplitudes,
            lindblad_amplitudes=lindblad_amplitudes,
        )

    def _generate_hamiltonian_params(self, amplitude=None, seed=0):
        """
        Generates the Hamiltonian parameters based on the Hamiltonian graph.

        Returns:
            dict: A dictionary containing the Hamiltonian parameters for each locality.
        """
        hamiltonian_params = {
            1: jnp.zeros(
                (self.number_of_interpolation_points, self.n_qubits, self.generators),
                dtype=jnp.complex128,
            )
        }

        hamiltonian_connections = {
            locality: len(graph) for locality, graph in self.hamiltonian_graph.items()
        }
        hamiltonian_params_shape = {
            locality: [self.number_of_interpolation_points]
            + [hamiltonian_connections[locality]]
            + [self.generators] * locality
            for locality in hamiltonian_connections
        }
        hamiltonian_params.update(
            {
                locality: jnp.zeros(
                    hamiltonian_params_shape[locality], dtype=jnp.complex128
                )
                for locality in hamiltonian_connections
            }
        )

        if amplitude:
            key = jax.random.PRNGKey(seed)
            keys = jax.random.split(key, self.hamiltonian_locality)

            for locality in hamiltonian_params:
                hamiltonian_params[locality] = amplitude[locality] * jax.random.normal(
                    keys[locality], hamiltonian_params[locality].shape
                )

        return hamiltonian_params

    def set_hamiltonian_params(self, hamiltonian_params: dict):
        print("Setting Hamiltonian Parameters is not optimzied to time yet")
        return super().set_hamiltonian_params(hamiltonian_params)

    def get_hamiltonian_generator(self):
        """
        Creates a jitted function that converts the Hamiltonian parameters to a Hamiltonian matrix.
        """

        @partial(jit)
        def hamiltonian_generator(hamiltonian_params: dict):
            """
            Creates the Hamiltonian based on the Hamiltonian parameters.
            """
            hamiltonian = jnp.zeros(
                (
                    self.number_of_interpolation_points,
                    self.hilbert_size,
                    self.hilbert_size,
                )
            )

            for locality in range(1, self.hamiltonian_locality + 1):
                hamiltonians = _convert_pauli_to_hamiltonians_with_time(
                    hamiltonian_params[locality],
                    self.hamiltonian_graph[locality],
                    self.n_qubits,
                    locality,
                    self.number_of_interpolation_points,
                )
                hamiltonian += _sum_interaction_hamiltonian_with_time(
                    hamiltonians,
                    self.hamiltonian_graph[locality],
                    self.n_qubits,
                    locality,
                    self.number_of_interpolation_points,
                ).reshape(
                    (
                        self.number_of_interpolation_points,
                        self.hilbert_size,
                        self.hilbert_size,
                    )
                )

            return hamiltonian

        return hamiltonian_generator


if __name__ == "__main__":
    NQUBITS = 2
    H_LOCALITY = 2
    L_LOCALITY = 2

    parameters = SuperOperatorParameterization(
        NQUBITS,
        hamiltonian_locality=H_LOCALITY,
        lindblad_locality=L_LOCALITY,
        # times=jnp.arange(0, 40, 4),
        hamiltonian_amplitudes=[],
    )

    # params = parameters.hamiltonian_params
    # hamiltonian_generator = parameters.get_hamiltonian_generator()

    # hamiltonian_generator(params)[0].imag

    hparams = parameters.hamiltonian_params
    hparams[1] = jnp.array([[1, 0, 0], [0, 1, 0]], dtype=jnp.float64)
    hparams[2] = jnp.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=jnp.float64)

    hamiltonian_generator = parameters._get_hamiltonian_generator()

    hamiltonian = hamiltonian_generator(hparams)

    _sum_interaction_hamiltonian_to_vector(
        hparams[2], parameters.hamiltonian_graph[2], NQUBITS, 2
    )

    dissipation_generator = parameters._get_dissipation_matrix_generator()

    lparams = parameters.lindbladian_params
    # lparams[1] = jnp.array(
    #     [[[[0, 0, 0], [0, 0, 0], [0, 0, 1]]], [[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]],
    #     dtype=jnp.float64,
    # )

    transfer_generator = parameters.get_generator_for_pauli_transfer_matrix()

    transfer_generator(hparams, lparams)
