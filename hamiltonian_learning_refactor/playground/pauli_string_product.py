import jax
import jax.numpy as jnp


# Let strings be the Pauli strings with 0 identity and 1, 2, 3 being the Pauli matrices
# We build strings as a list of pauli letters

# All elements in string multiplication are done separately


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


def letter_product(letter1, letter2):
    return lookup()[letter1, letter2]


def string_product(string1, string2):
    return jnp.einsum("i, ij->j", string1, string2)


letter_z = jnp.array([0, 0, 0, 1], dtype=jnp.bool)
letter_x = jnp.array([0, 1, 0, 0], dtype=jnp.bool)

letter_product(letter_x, letter_z)

jnp.kron(lookup(), lookup()).shape

jnp.kron(jnp.kron(lookup(), lookup()), lookup())[2, 1]


if __name__ == "__main__":
    DT1 = jnp.array([[0.25, -0.25j, 0], [0.25j, 0.25, 0], [0, 0, 0]])

    normalization = jnp.einsum("mnk, mn -> k", lookup(), chi)


jnp.nonzero(jnp.kron(lookup(), lookup()).reshape(4, 4, 4, 4, 4, 4)[..., 2, 2])
