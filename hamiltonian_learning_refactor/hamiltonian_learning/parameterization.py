# Skets at the moment to generate the parameterization of the Lindblad Master Equation
from typing import Union, List, Tuple
import jax.numpy as jnp
import itertools


# Locality -> Set if all to all connections
# Graph -> For each locality > 2 a tuple of sets of connections
# Can be given as a dict {2: ((0, 1), (1, 2), (2, 3), (3, 0)), 3: ((0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1))}


class Parameterization:
    """
    A class representing the parameterization of a quantum system.

    Attributes:
        n_qubits (int): The number of qubits in the system.
        hamiltonian_locality (int): The locality of the Hamiltonian terms.
        lindblad_locality (int): The locality of the Lindblad terms.
        hamiltonian_graph (tuple, None or 'full): Tuple containing sets of connections defining a structure, None for no
        lindblad_graph (optional): The graph representing the connectivity of the Lindblad terms.
    """

    def __init__(
        self,
        n_qubits: int,
        qubit_levels: int = 2,
        hamiltonian_locality: int = None,
        lindblad_locality: int = None,
        hamiltonian_graph: dict = None,
        lindblad_graph: dict = None,
    ):
        assert hamiltonian_locality is not None or hamiltonian_graph is not None
        assert lindblad_locality is not None or lindblad_graph is not None

        # Load Params
        self.n_qubits = n_qubits
        self.qubit_levels = qubit_levels
        self.hilbert_size = self.qubit_levels**n_qubits
        self.generators = self.qubit_levels**2 - 1  # Minus idenity

        # Set values to locality
        self.hamiltonian_locality = hamiltonian_locality
        self.lindblad_locality = lindblad_locality

        # Or overwrite with graph
        self.hamiltonian_graph = hamiltonian_graph
        self.lindblad_graph = lindblad_graph

        # Generate graphs if not given
        if hamiltonian_graph is None:
            self.hamiltonian_graph = {
                i: tuple(itertools.product(range(n_qubits), repeat=i))
                for i in range(self.hamiltonian_locality)
            }

        if lindblad_graph is None:
            self.lindblad_graph = {
                i: tuple(itertools.product(range(n_qubits), repeat=i))
                for i in range(self.lindblad_locality)
            }

        # Generate Hamiltonian Parameters
        self.hamiltonian_params = self._generate_hamiltonian_params()
        self.lindbladian_params = self._generate_lindbladian_params()

    def _generate_hamiltonian_params(self):
        """
        Generates the Hamiltonian parameters based on the Hamiltonian graph.

        Returns:
            dict: A dictionary containing the Hamiltonian parameters for each locality.
        """
        hamiltonian_connections = {
            locality: len(graph) for locality, graph in self.hamiltonian_graph.items()
        }
        hamiltonian_params_shape = {
            locality: [hamiltonian_connections[locality]] + [self.generators] * locality
            for locality in hamiltonian_connections
        }
        hamiltonian_params = {
            locality: jnp.zeros(hamiltonian_params_shape[locality])
            for locality in hamiltonian_connections
        }
        return hamiltonian_params

    def _generate_lindbladian_params(self):
        """
        Generates the Lindbladian parameters based on the Lindbladian graph.

        Returns:
            dict: A dictionary containing the Lindbladian parameters for each locality.
        """
        lindbladian_connections = {
            locality: len(graph) for locality, graph in self.lindblad_graph.items()
        }
        lindbladian_params_shape = {
            locality: lindbladian_connections[locality] * [self.qubit_levels] * 2
            for locality in lindbladian_connections
        }
        lindbladian_params = {
            locality: jnp.zeros(lindbladian_params_shape[locality])
            for locality in lindbladian_connections
        }
        return lindbladian_params

    def get_hamiltonian(self):
        return None  # hilbert_size x hilbert_size matrix

    def get_jump_operators(self):
        return None  # List of hilbert_size x hilbert_size matrices


class TimeEvolution:
    parameterization: Parameterization
