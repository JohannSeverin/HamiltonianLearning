import jax.numpy as jnp
from jax import jit


@jit
def _add_matrix_to_tensor(tensor, matrix):
    """
    This functions make the tensorproduct of a tensor and a matrix
    """
    Ndims = (jnp.ndim(tensor) + jnp.ndim(matrix)) // 2
    # print(Ndims)
    product = jnp.einsum("...,jk->...jk", tensor, matrix)
    product = jnp.moveaxis(product, -1, Ndims - 1)
    return product


@jit
def tensor_product(tensors):
    """
    First dimension should correspond to the number of tensors to tensor product
    """
    product = tensors[0]
    for tensor in tensors[1:]:
        product = _add_matrix_to_tensor(product, tensor)
    return product


@jit
def tensor_sum(tensors):
    """
    First dimension should correspond to the number of tensors to sum
    """
    number_tensors = tensors.shape[0]
    shape = (2,) * 2 * number_tensors
    output = jnp.zeros(shape)

    for i in range(number_tensors):
        Hs = [jnp.eye(tensors.shape[1]) for _ in range(number_tensors)]
        Hs[i] = tensors[i]
        output += tensor_product(Hs)

    return output


from functools import partial


@partial(jit, static_argnames=("Nqubits", "connections"))
def sum_two_qubit_interaction_tensors(
    connections: jnp.ndarray, tensors: jnp.ndarray, Nqubits: int
):
    """
    This function is used to sum connecting tensors.
    - Connections is a matrix with the connections between the qubit indices which should be connected
    - Tensors is a list of tensors which should be connected
    """
    shape = (2,) * (2 * Nqubits)
    output = jnp.zeros(shape)
    where_from, where_to = connections[0], connections[1]

    for i in range(len(where_from)):
        H = tensors[i]

        for j in range(Nqubits - 2):
            H = _add_matrix_to_tensor(H, jnp.eye(2))

        H = jnp.swapaxes(H, where_from[i], 0)
        H = jnp.swapaxes(H, where_to[i], 1)

        H = jnp.swapaxes(H, Nqubits + where_from[i], Nqubits + 0)
        H = jnp.swapaxes(H, Nqubits + where_to[i], Nqubits + 1)

        output += H

    return output


# sum_two_qubit_interaction_tensors = jit(
#     sum_two_qubit_interaction_tensors, static_argnames=("Nqubits", "connections")
# )


if __name__ == "__main__":
    Nqubits = 10
    z = jnp.array([[1, 0], [0, -1]])
    x = jnp.array([[0, 1], [1, 0]])

    one_qubit_hamil = tensor_sum(jnp.array([z] * Nqubits))

    connections = ((0, 1), (1, 2))
    interaction = tensor_product(jnp.array([x, x]))

    interactions = jnp.stack([interaction, interaction], axis=0)

    out = sum_two_qubit_interaction_tensors(connections, interactions, Nqubits)
