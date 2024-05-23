import jax.numpy as jnp
import sys

sys.path.append("..")
from utils import *

pauli_gates = jnp.array([sigma_x(), sigma_y(), sigma_z()])


def jump_operators_to_pauli_matrix(jump_operators):
    """
    Converts a set of jump operators to a Lindbladian matrix in the Pauli basis.

    Args:
        jump_operators (ndarray): A set of jump operators represented as a 3-dimensional array.

    Returns:
        ndarray: The Lindbladian matrix in the Pauli basis.

    """
    pauli_basis = jnp.einsum("ijk, pjk -> ip", jump_operators, pauli_gates)
    lindbladian_matrix = jnp.einsum("ik, il -> kl", pauli_basis, pauli_basis.conj())
    return lindbladian_matrix


def pauli_matrix_to_jump_operators(pauli_matrix):
    """
    Converts a Pauli matrix to jump operators.

    Args:
        pauli_matrix: The Pauli matrix to be converted.

    Returns:
        The jump operators corresponding to the given Pauli matrix.
    """
    return jnp.einsum("ij, jkl -> ikl", pauli_matrix, pauli_gates)
