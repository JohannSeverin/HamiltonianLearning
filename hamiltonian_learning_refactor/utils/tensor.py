from functools import partial
import jax.numpy as jnp
from jax import jit


# Tensor Product and Partial Trace
def tensor(*args, return_dimensions=False):
    """
    Compute the tensor product of a list of matrices
    """
    result = args[0]
    dims = [result.shape[0]] if return_dimensions else None

    for i in range(1, len(args)):
        result = jnp.kron(result, args[i])

        if return_dimensions:
            dims.append(args[i].shape[0])

    if return_dimensions:
        return result, jnp.array(dims, dtype=jnp.int32)
    else:
        return result


@jit
def _add_matrix_to_tensor(tensor, matrix):
    """
    This functions make the tensorproduct of a tensor and a matrix
    """
    Ndims = (jnp.ndim(tensor) + jnp.ndim(matrix)) // 2
    product = jnp.einsum("...,jk->...jk", tensor, matrix)
    product = jnp.moveaxis(product, -2, Ndims - 1)
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


@partial(jit, static_argnames=("NQubits"))
def tensor_at_index(tensor, index, NQubits):
    """
    Compute the tensor product of a list of matrices
    """
    Is = jnp.array([jnp.eye(2) for _ in range(NQubits)])

    Is = Is.at[index].set(tensor)

    return tensor_product(jnp.array(Is))


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

        if where_to[i] == 0 and where_from[i] == 1:
            pass
        elif where_to[i] == 1 and where_from[i] == 0:
            H = jnp.swapaxes(H, 0, 1)
            H = jnp.swapaxes(H, Nqubits, Nqubits + 1)
        elif where_from[i] == 1:
            H = jnp.swapaxes(H, 1, where_to[i])
            H = jnp.swapaxes(H, Nqubits + 1, where_to[i] + Nqubits)
            H = jnp.swapaxes(H, 0, where_from[i])
            H = jnp.swapaxes(H, Nqubits, where_from[i] + Nqubits)
        else:
            H = jnp.swapaxes(H, 0, where_from[i])
            H = jnp.swapaxes(H, Nqubits, where_from[i] + Nqubits)
            H = jnp.swapaxes(H, 1, where_to[i])
            H = jnp.swapaxes(H, Nqubits + 1, where_to[i] + Nqubits)

        output += H

    return output


@partial(jit, static_argnames=("Nqubits", "connections", "Ninteractions"))
def sum_k_local_interaction_tensors(connections, tensors, Nqubits, k_locality):
    shape = (2,) * (2 * Nqubits)
    output = jnp.zeros(shape)

    for i in range(len(connections[0])):
        H = tensors[i]

        for j in range(Nqubits - k_locality):
            H = _add_matrix_to_tensor(H, jnp.eye(2))

        order = [-1] * Nqubits
        connection = [conn[i] for conn in connections]  # Indices for interactions

        for j, conn in enumerate(connection):
            order[conn] = j

        fill_with = list(range(k_locality, Nqubits))

        for j, o in enumerate(order):
            if o == -1:
                order[j] = fill_with.pop(0)

        order = order + [Nqubits + i for i in order]

        H = H.transpose(order)

        output += H

    return output
