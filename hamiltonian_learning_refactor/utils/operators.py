# Document for saving all common operators used in the project
import jax.numpy as jnp


# Common operators
def identity(n=2) -> jnp.ndarray:
    """
    Return the identity matrix
    """
    return jnp.eye(n, dtype=jnp.complex128)


def sigma_x(n=2) -> jnp.ndarray:
    """
    Return the Pauli X matrix
    """
    return jnp.zeros((n, n), dtype=jnp.complex128).at[0, 1].set(1.0).at[1, 0].set(1.0)


def sigma_y(n=2) -> jnp.ndarray:
    """
    Return the Pauli Y matrix
    """
    return jnp.zeros((n, n), dtype=jnp.complex128).at[0, 1].set(-1j).at[1, 0].set(1j)


def sigma_z(n=2) -> jnp.ndarray:
    """
    Return the Pauli Z matrix
    """
    return jnp.zeros((n, n), dtype=jnp.complex128).at[0, 0].set(1.0).at[1, 1].set(-1.0)


def gate_x_90(n=2):
    """
    Return the X gate with a 90 degree rotation
    """
    return jnp.array([[1, -1j], [-1j, 1]], dtype=jnp.complex128) / jnp.sqrt(2)


def gate_x_270(n=2):
    """
    Return the X gate with a 90 degree rotation
    """
    return jnp.array([[1, 1j], [1j, 1]], dtype=jnp.complex128) / jnp.sqrt(2)


def gate_y_90(n=2):
    """
    Return the Y gate with a 90 degree rotation
    """
    return jnp.array([[1, -1], [1, 1]], dtype=jnp.complex128) / jnp.sqrt(2)


def gate_y_270(n=2):
    """
    Return the Y gate with a 90 degree rotation
    """
    return jnp.array([[1, 1], [-1, 1]], dtype=jnp.complex128) / jnp.sqrt(2)


def gate_z_90(n=2):
    """
    Return the Z gate with a 90 degree rotation
    """
    return jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)


# Ladder operators


def destroy(n=2):
    """
    Return the annihilation operator
    """
    return jnp.diag(jnp.sqrt(jnp.arange(1, n, dtype=jnp.complex128)), k=1)


def create(n=2):
    """
    Return the creation operator
    """
    return jnp.diag(jnp.sqrt(jnp.arange(1, n, dtype=jnp.complex128)), k=-1)


# Common dissipation operators
def t1_decay(t1, n=2):
    """
    Return the T1 decay dissipator
    """
    return jnp.sqrt(1 / t1) * destroy(n)


def t2_decay(decay_time, t1=None, n=2):
    if t1 is None:
        gamma_phase = 1 / decay_time
        print(f"assuming Tphase = {decay_time:.2e}")
    else:
        gamma_phase = 1 / decay_time - 1 / t1 / 2

    return jnp.sqrt(gamma_phase) * sigma_z(n)
