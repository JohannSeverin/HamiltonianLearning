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

        print(order, H.shape)
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


### PARAMETRIZATION CLASS ###
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

        self.hamiltonian_params = self._generate_hamiltonian_params(
            hamiltonian_amplitudes
        )
        self.lindbladian_params = self._generate_lindbladian_params(lindblad_amplitudes)

    def _generate_hamiltonian_params(self, amplitude=None, seed=0):
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

        if amplitude:
            key = jax.random.PRNGKey(seed)
            keys = jax.random.split(key, self.hamiltonian_locality)

            for locality in hamiltonian_params:
                hamiltonian_params[locality] = amplitude * jax.random.normal(
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
            key = jax.random.PRNGKey(seed)
            keys = jax.random.split(key, self.hamiltonian_locality)

            for locality in lindbladian_params:
                lindbladian_params[locality] = amplitudes * jax.random.normal(
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
        hamiltonian_amplitudes: list[float] = [],
        lindblad_amplitudes: list[float] = [],
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
                hamiltonian_params[locality] = amplitude * jax.random.normal(
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
    L_LOCALITY = 0

    parameters = InterpolatedParameterization(
        NQUBITS,
        hamiltonian_locality=H_LOCALITY,
        lindblad_locality=L_LOCALITY,
        times=jnp.arange(0, 40, 4),
        hamiltonian_amplitudes=1.0,
    )

    params = parameters.hamiltonian_params
    hamiltonian_generator = parameters.get_hamiltonian_generator()

    hamiltonian_generator(params)[0].imag

    # jump_operator_generator = parameters.get_jump_operator_generator()

    # hamiltonian_generator(parameters.hamiltonian_params)
    # jump_operator_generator(parameters.lindbladian_params)

    # %timeit hamiltonian = hamiltonian_generator(parameters.hamiltonian_params)
    # %timeit jump_operators = jump_operator_generator(parameters.lindbladian_params)
