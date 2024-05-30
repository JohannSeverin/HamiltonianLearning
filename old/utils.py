import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)

# Standard Solver Options for Quantum Mechanical Simulations
from diffrax import (
    Dopri5,
    Dopri8,
    SaveAt,
    diffeqsolve,
    DirectAdjoint,
    ODETerm,
    PIDController,
    RecursiveCheckpointAdjoint,
)


def create_solver(
    t1,
    t0=0,
    adjoint=False,
    stepsize_controller=PIDController(1e-5, 1e-5),
    solver=Dopri5(),
    tlist=None,
    initial_stepsize=1e-1,
    number_of_jump_operators=0,
    max_steps=10000,
):
    adjoint = (
        DirectAdjoint() if adjoint else RecursiveCheckpointAdjoint(checkpoints=None)
    )
    saveat = SaveAt(ts=tlist) if tlist is not None else None

    @jit
    def dynamics(t, rho, args):
        """
        Differential equation governing the system
        """
        drho = get_unitary_term(rho, args["hamiltonian"])

        for i in range(number_of_jump_operators):
            drho += get_dissipation_term(rho, args["jump_operators"][i])

        return drho

    term = ODETerm(dynamics)

    def evolve_states(initial_state, hamiltonian, jump_operators=None):
        return diffeqsolve(
            terms=term,
            solver=solver,
            y0=initial_state,
            t0=t0,
            t1=t1,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            dt0=initial_stepsize,
            args=dict(hamiltonian=hamiltonian, jump_operators=jump_operators),
            adjoint=adjoint,
            max_steps=max_steps,
        )

    return evolve_states


def create_solver_with_pauli_structure(
    t1,
    number_of_qubits,
    t0=0,
    adjoint=False,
    stepsize_controller=PIDController(1e-10, 1e-10),
    solver=Dopri8(),
    tlist=None,
    initial_stepsize=1e-1,
):
    adjoint = (
        DirectAdjoint() if adjoint else RecursiveCheckpointAdjoint(checkpoints=None)
    )
    saveat = SaveAt(ts=tlist) if tlist is not None else None

    @jit
    def dynamics(t, rho, args):
        """
        Differential equation governing the system
        """
        drho = get_unitary_term(rho, args["hamiltonian"])
        drho += dissipation_from_pauli_matrix(
            rho, args["dissipation_matrix"], number_of_qubits
        )

        return drho

    term = ODETerm(dynamics)

    def evolve_states(initial_state, hamiltonian, dissipation_matrix=None):
        return diffeqsolve(
            terms=term,
            solver=solver,
            y0=initial_state,
            t0=t0,
            t1=t1,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            dt0=initial_stepsize,
            args=dict(hamiltonian=hamiltonian, dissipation_matrix=dissipation_matrix),
            adjoint=adjoint,
        )

    return evolve_states


def propagate_uncertainties(function, arguments, errors):
    """
    Compute the uncertainty of a function given the uncertainties of its arguments
    """
    value = function(arguments)
    jacobian = jax.jacfwd(function)(arguments)
    covariance = jnp.diag(errors**2) if errors.ndim == 1 else errors

    return value, jnp.sqrt(
        jnp.einsum("ij, jk, ki->i", jacobian, covariance, jacobian.T)
    )


# Functions to compute time evolutions of quantum systems
def get_unitary_term(rho, hamiltonian):
    """
    Compute the unitary term of the Master equation.
    Has to be jax numpy arrays ending with a square matrix as the last term
    """
    return -1j * (
        jnp.einsum("...ij, ...jk -> ...ik", hamiltonian, rho)
        - jnp.einsum("...ij, ...jk -> ...ik", rho, hamiltonian)
    )


def get_dissipation_term(rho, dissipator):
    """
    Compute the dissipator term of the Lindblad equation
    """
    drho = jnp.einsum(
        "...ij, ...jk, ...kl -> ...il", dissipator, rho, dissipator.conj().T
    )

    drho -= 0.5 * (
        jnp.einsum("...ij, ...jk -> ...ik", dissipator.conj().T @ dissipator, rho)
        + jnp.einsum("...ij, ...jk -> ...ik", rho, dissipator.conj().T @ dissipator)
    )

    return drho


from itertools import product


## FOLLOWING TWO FUNCTIONS ARE NOT FINISHED. IF THEY ARE USED THEY SHOULD BE UPDATED ACCORDINGLY
@partial(jax.jit, static_argnames=("Nqubits"))
def _get_pauli_matrices(Nqubits):
    """
    Compute the Pauli matrices for a system of Nqubits
    """
    pauli_matrices = jnp.zeros(
        (4**Nqubits - 1, 2**Nqubits, 2**Nqubits), dtype=jnp.complex128
    )

    single_qubit_pauli_matrices = jnp.array(
        [jnp.eye(2), sigma_x(), sigma_y(), sigma_z()]
    )

    for i, qubit_operators in enumerate(
        product(single_qubit_pauli_matrices, repeat=Nqubits)
    ):
        if i == 0:
            continue
        pauli_matrix = tensor_product(qubit_operators).reshape(2**Nqubits, 2**Nqubits)
        pauli_matrices = pauli_matrices.at[i - 1].set(pauli_matrix)

    return pauli_matrices


@partial(jax.jit, static_argnames=("Nqubits"))
def dissipation_from_pauli_matrix(rho, dissipation_matrix, Nqubits):
    pauli_matrices = _get_pauli_matrices(Nqubits)
    drho = jnp.einsum(
        "pq, pkl, ...lm, qmn -> ...kn",
        dissipation_matrix,
        pauli_matrices,
        rho,
        pauli_matrices,
    )

    drho -= 0.5 * jnp.einsum(
        "pq, qkl, pkm, ...mn -> ...kn ",
        dissipation_matrix,
        pauli_matrices,
        pauli_matrices,
        rho,
    )

    drho -= 0.5 * jnp.einsum(
        "pq, ...kl, qlm, pmn -> ...kn ",
        dissipation_matrix,
        rho,
        pauli_matrices,
        pauli_matrices,
    )

    return drho


# Expectation Values, Probabilities and Projections
def expectation_values(rhos, observable):
    """
    Compute the expectation value of an observable
    """
    if observable.ndim == 2:
        return jnp.einsum("...ij, ji -> ...", rhos, observable)
    elif observable.ndim == 3:
        return jnp.einsum("...ij, kji -> ...k", rhos, observable)


def rotating_unitary(t, hamiltonian):
    """
    Compute the unitary operator of a time-dependent Hamiltonian
    """
    unitary_rotation = jax.scipy.linalg.expm(
        -1j * hamiltonian[None, ...] * t[:, None, None]
    )

    return unitary_rotation


def rotating_expectation_values(rhos, observable, t, hamiltonian):
    """
    Compute the expectation value of an observable
    """
    unitary_rotation = jax.scipy.linalg.expm(
        -1j * hamiltonian[None, ...] * t[:, None, None]
    )

    unitary_rotation_dagger = unitary_rotation.conj().transpose((0, 2, 1))

    rotating_observable = (
        unitary_rotation @ observable[None, ...] @ unitary_rotation_dagger
    )

    return jnp.einsum("...tij, tji -> ...t", rhos, rotating_observable)


# Common States
def basis_ket(state_index, n):
    """
    Return the basis vector of the state_index-th state in a Hilbert space of dimension n
    """
    return jnp.zeros(n, dtype=jnp.complex128).at[state_index].set(1.0)


def basis_dm(state_index, n):
    """
    Return the density matrix of the state_index-th state in a Hilbert space of dimension n
    """
    return basis_ket(state_index, n)[:, None] @ basis_ket(state_index, n)[None, :]


def pauli_state_x_ket(n=2):
    """
    Return the X eigenstate in a Hilbert space of dimension n
    """
    return (basis_ket(0, n) + basis_ket(1, n)) / jnp.sqrt(2)


def pauli_state_minus_x_ket(n=2):
    """
    Return the X eigenstate in a Hilbert space of dimension n
    """
    return (basis_ket(0, n) - basis_ket(1, n)) / jnp.sqrt(2)


def pauli_state_y_ket(n=2):
    """
    Return the Y eigenstate in a Hilbert space of dimension n
    """
    return (basis_ket(0, n) + 1j * basis_ket(1, n)) / jnp.sqrt(2)


def pauli_state_minus_y_ket(n=2):
    """
    Return the Y eigenstate in a Hilbert space of dimension n
    """
    return (basis_ket(0, n) - 1j * basis_ket(1, n)) / jnp.sqrt(2)


def pauli_state_x_dm(n=2):
    """
    Return the density matrix of the X eigenstate in a Hilbert space of dimension n
    """
    return pauli_state_x_ket(n)[:, None] @ pauli_state_x_ket(n)[None, :]


def pauli_state_minus_x_dm(n=2):
    """
    Return the density matrix of the X eigenstate in a Hilbert space of dimension n
    """
    return pauli_state_minus_x_ket(n)[:, None] @ pauli_state_minus_x_ket(n)[None, :]


def pauli_state_y_dm(n=2):
    """
    Return the density matrix of the Y eigenstate in a Hilbert space of dimension n
    """
    return pauli_state_y_ket(n)[:, None] @ jnp.conj(pauli_state_y_ket(n)[None, :])


def pauli_state_minus_y_dm(n=2):
    """
    Return the density matrix of the Y eigenstate in a Hilbert space of dimension n
    """
    return pauli_state_minus_y_ket(n)[:, None] @ jnp.conj(
        pauli_state_minus_y_ket(n)[None, :]
    )


# Common Operators
def identity(n=2):
    """
    Return the identity matrix
    """
    return jnp.eye(n, dtype=jnp.complex128)


def sigma_x(n=2):
    """
    Return the Pauli X matrix
    """
    return jnp.zeros((n, n), dtype=jnp.complex128).at[0, 1].set(1.0).at[1, 0].set(1.0)


def sigma_y(n=2):
    """
    Return the Pauli Y matrix
    """
    return jnp.zeros((n, n), dtype=jnp.complex128).at[0, 1].set(-1j).at[1, 0].set(1j)


def sigma_z(n=2):
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


# Common dissipators
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


from functools import partial


# @partial(jit, static_argnames=("Nqubits", "connections"))
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

        # print(i, H.shape, where_from[i], where_to[i])
        # print(H.reshape(4, 4).real)

        for j in range(Nqubits - 2):
            H = _add_matrix_to_tensor(H, jnp.eye(2))
        # print(H.reshape(8, 8).real)

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

        # print(H.reshape(8, 8).real)

        output += H

    return output


@partial(jit, static_argnames=("Nqubits", "connections", "Ninteractions"))
def sum_N_qubit_interaction_tensors(connections, tensors, Nqubits, Ninteractions):
    shape = (2,) * (2 * Nqubits)
    output = jnp.zeros(shape)

    for i in range(len(connections[0])):
        H = tensors[i]

        for j in range(Nqubits - Ninteractions):
            H = _add_matrix_to_tensor(H, jnp.eye(2))

        order = [-1] * Nqubits
        connection = [conn[i] for conn in connections]  # Indices for interactions

        for j, conn in enumerate(connection):
            order[conn] = j

        fill_with = list(range(Ninteractions, Nqubits))

        for j, o in enumerate(order):
            if o == -1:
                order[j] = fill_with.pop(0)

        order = order + [Nqubits + i for i in order]

        H = H.transpose(order)

        output += H

    return output


# TESTING
def main():
    pass


if __name__ == "__main__":
    main()
