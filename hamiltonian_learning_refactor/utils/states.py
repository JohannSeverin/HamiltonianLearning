# Document for saving all common states used in the project
import jax.numpy as jnp


def basis(state_index, n):
    """
    Return the basis vector of the state_index-th state in a Hilbert space of dimension n
    """
    return jnp.zeros(n, dtype=jnp.complex128).at[state_index].set(1.0)


def basis_dm(state_index, n):
    """
    Return the density matrix of the state_index-th state in a Hilbert space of dimension n
    """
    return basis(state_index, n)[:, None] @ basis(state_index, n)[None, :]


def pauli_state_x(n=2):
    """
    Return the X eigenstate in a Hilbert space of dimension n
    """
    return (basis(0, n) + basis(1, n)) / jnp.sqrt(2)


def pauli_state_minus_x(n=2):
    """
    Return the X eigenstate in a Hilbert space of dimension n
    """
    return (basis(0, n) - basis(1, n)) / jnp.sqrt(2)


def pauli_state_y(n=2):
    """
    Return the Y eigenstate in a Hilbert space of dimension n
    """
    return (basis(0, n) + 1j * basis(1, n)) / jnp.sqrt(2)


def pauli_state_minus_y(n=2):
    """
    Return the Y eigenstate in a Hilbert space of dimension n
    """
    return (basis(0, n) - 1j * basis(1, n)) / jnp.sqrt(2)


def pauli_state_x_dm(n=2):
    """
    Return the density matrix of the X eigenstate in a Hilbert space of dimension n
    """
    return pauli_state_x(n)[:, None] @ pauli_state_x(n)[None, :]


def pauli_state_minus_x_dm(n=2):
    """
    Return the density matrix of the X eigenstate in a Hilbert space of dimension n
    """
    return pauli_state_minus_x(n)[:, None] @ pauli_state_minus_x(n)[None, :]


def pauli_state_y_dm(n=2):
    """
    Return the density matrix of the Y eigenstate in a Hilbert space of dimension n
    """
    return pauli_state_y(n)[:, None] @ jnp.conj(pauli_state_y(n)[None, :])


def pauli_state_minus_y_dm(n=2):
    """
    Return the density matrix of the Y eigenstate in a Hilbert space of dimension n
    """
    return pauli_state_minus_y(n)[:, None] @ jnp.conj(pauli_state_minus_y(n)[None, :])
