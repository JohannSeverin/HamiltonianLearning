import sys, os

import jax.numpy as jnp

from itertools import product

sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils import *

from itertools import product


# Different useful constants and lists
pauli_gate_set = jnp.array(
    [
        sigma_x(),
        sigma_y(),
        sigma_z(),
    ]
)


def N_qubit_pauli_gate_set(N):
    return jnp.array([tensor(*pauli) for pauli in product(pauli_gate_set, repeat=N)])


pauli_gate_set_with_id = jnp.array(
    [
        identity(2),
        sigma_x(),
        sigma_y(),
        sigma_z(),
    ]
)


def N_qubit_pauli_gate_set_with_identity(N):
    return jnp.array(
        [tensor(*pauli) for pauli in product(pauli_gate_set_with_id, repeat=N)]
    )


two_qubit_pauli_gate_set = jnp.array(
    [tensor(*pauli) for pauli in product(pauli_gate_set, repeat=2)]
).reshape(3, 3, 4, 4)


# Functions for formatting Hamiltonian if it is packed
two_qubit_paulis = two_qubit_pauli_gate_set.reshape(3, 3, 2, 2, 2, 2)


@partial(jax.jit, static_argnames=("NQUBITS"))
def build_local_hamiltonian(params, NQUBITS):
    local_hamiltonians = jnp.zeros((NQUBITS, 2, 2), dtype=jnp.complex128)
    for qubit in range(NQUBITS):
        H_i = jnp.einsum("i, ijk -> jk", params[qubit], pauli_gate_set)
        local_hamiltonians = local_hamiltonians.at[qubit].set(H_i)

    return tensor_sum(jnp.array(local_hamiltonians))


@partial(jax.jit, static_argnames=("connections", "NQUBITS"))
def build_interaction_hamiltonian(connections, params, NQUBITS):
    two_local_hamiltonians = jnp.zeros((NQUBITS - 1, 2, 2, 2, 2), dtype=jnp.complex128)
    for i in range(NQUBITS - 1):
        two_local_hamiltonians = two_local_hamiltonians.at[i].set(
            jnp.einsum("ij, ij... -> ...", params[i], two_qubit_paulis)
        )
    # return two_local_hamiltonians

    return sum_N_qubit_interaction_tensors(
        connections, two_local_hamiltonians, NQUBITS, 2
    )


from string import ascii_lowercase


@partial(jax.jit, static_argnames=("connections", "NQubits", "NInteractions"))
def build_N_interaction_hamiltonian(connections, tensors, NQubits, NInteractions):
    paulis = N_qubit_pauli_gate_set(NInteractions).reshape(
        (3,) * NInteractions + (2,) * 2 * NInteractions
    )
    Nconnections = len(connections[0])
    shape = (Nconnections,) + (2,) * NInteractions * 2
    hamiltonian = jnp.zeros(
        shape,
        dtype=jnp.complex128,
    )

    to_sum = "".join(ascii_lowercase[:NInteractions])
    for i in range(Nconnections):
        h_here = jnp.einsum(f"{to_sum}, {to_sum}... -> ...", tensors[i], paulis)
        hamiltonian = hamiltonian.at[i].set(h_here)

    # print(connections, tensors.shape, NQubits, NInteractions)

    return sum_N_qubit_interaction_tensors(
        connections, hamiltonian, NQubits, NInteractions
    )


# Functions for building the local jump operators
@partial(jax.jit, static_argnames=("NQUBITS"))
def build_local_dissipation(tensors, NQUBITS):
    """
    tensors should be Nqubits x 3 x 3 elements. The lower triagular matrix will be the real components and the upper the imaginary
    """
    cholesko = _build_local_cholesko(tensors, NQUBITS)

    single_qubit_jump_operators = jnp.einsum(
        "...ij, jkl -> ...ikl", cholesko, pauli_gate_set_with_id[1:]
    )
    shape = (NQUBITS * 3,) + (2, 2) * NQUBITS

    jump_operators = jnp.zeros(
        shape,
        dtype=jnp.complex128,
    )

    if NQUBITS == 1:
        return single_qubit_jump_operators
    else:

        for qubit in range(NQUBITS):
            jump_operators = jump_operators.at[qubit * 3 : (qubit + 1) * 3].set(
                tensor_sum(single_qubit_jump_operators[qubit])
            )

        return jump_operators


@partial(jax.jit, static_argnames=("NQubits"))
def _build_local_cholesko(tensors, NQubits):
    """
    tensors should be Nqubits x 3 x 3 elements. The lower triagular matrix will be the real components and the upper the imaginary
    """
    cholesko_matrices = jnp.zeros((NQubits, 3, 3), dtype=jnp.complex128)
    for qubit in range(NQubits):
        cholesko = jnp.zeros((3, 3), dtype=jnp.complex128)
        cholesko = cholesko.at[jnp.tril_indices(3)].set(
            tensors[qubit][jnp.tril_indices(3)]
        )
        cholesko = cholesko.at[jnp.tril_indices(3, k=-1)].add(
            1j * tensors[qubit][jnp.triu_indices(3, k=+1)]
        )
        cholesko_matrices = cholesko_matrices.at[qubit].set(tensors[qubit])

    return cholesko_matrices


# Functions for building the interacting jump operators


@partial(jax.jit, static_argnames=("NQubits", "NConnections", "set_local_to_zero"))
def _build_interaction_cholesko(
    tensors, NConnections, NQubits, set_local_to_zero=False
):
    """
    tensors should be #connections x 15 x 15 elements. The lower triagular matrix will be the real components and the upper the imaginary
    """
    cholesko_matrices = jnp.zeros(NConnections, 15, 15, dtype=jnp.complex128)
    for connection in range(NConnections):
        cholesko = jnp.zeros((15, 15), dtype=jnp.complex128)
        cholesko = cholesko.at[jnp.tril_indices(16)].set(
            tensors[connection][jnp.tril_indices(16)]
        )
        cholesko = cholesko.at[jnp.tril_indices(16, k=-1)].add(
            1j * tensors[connection][jnp.triu_indices(16, k=+1)]
        )
        cholesko_matrices = cholesko_matrices.at[connection].set(cholesko)


# Functions for setup
def pauli_strings(
    number_of_qubits=1,
    include_identity=True,
):
    pauli_letters = "ixyz" if include_identity else "xyz"
    return [
        "".join(indices) for indices in product(pauli_letters, repeat=number_of_qubits)
    ]


def pauli_operators(number_of_qubits=1):
    operators = [identity(2), sigma_x(), sigma_y(), sigma_z()]
    pauli_letters = pauli_strings(
        number_of_qubits=number_of_qubits, include_identity=True
    )

    if number_of_qubits == 1:
        return operators

    multi_qubit_operators = list(product(operators, repeat=number_of_qubits))
    multi_qubit_operators = jnp.array([tensor(*ops) for ops in multi_qubit_operators])

    return multi_qubit_operators, pauli_letters


def generate_initial_states(
    number_of_qubits=1, with_mixed_states=False, include_negative_states=False
):
    """
    Returns the initial states for the Hamiltonian learning problem.

    Args:
        number_of_qubits (int): The number of qubits. Default is 1.

    Returns:
        tuple: A tuple containing the initial states and the corresponding Pauli indices.
            - initial_states (list): A list of initial states for the Hamiltonian learning problem.
            - pauli_index (list): A list of Pauli indices corresponding to the initial states.

    Notes:
        The initial states are generated as follows:
        - For a single qubit, the initial states are all combinations of three Pauli states: X, Y, and Z.
        - For multiple qubits, the initial states are all combinations of the Pauli states for each qubit and the possible products between them.
    """
    single_qubit_pauli_states = [
        pauli_state_x_dm(2),
        pauli_state_y_dm(2),
        basis_dm(0, 2),
    ]
    pauli_index = ["x", "y", "z"]

    if include_negative_states:
        single_qubit_pauli_states += [
            pauli_state_minus_x_dm(2),
            pauli_state_minus_y_dm(2),
            basis_dm(1, 2),
        ]
        pauli_index += ["~x", "~y", "~z"]

    if with_mixed_states:
        single_qubit_pauli_states.insert(0, basis_dm(1, 2) / 2 + basis_dm(0, 2) / 2)
        pauli_index = ["m"] + pauli_index

    if number_of_qubits == 1:
        return jnp.stack(single_qubit_pauli_states, axis=0), pauli_index

    multi_qubit_pauli_states = list(
        product(single_qubit_pauli_states, repeat=number_of_qubits)
    )

    multi_qubit_pauli_states = jnp.array(
        [tensor(*states) for states in multi_qubit_pauli_states]
    )

    pauli_index = [
        "".join(indices) for indices in product(pauli_index, repeat=number_of_qubits)
    ]

    return jnp.stack(multi_qubit_pauli_states, axis=0), pauli_index


def generate_basis_transformations(number_of_qubits=2, invert=False):
    """
    Change the states to the measurement basis
    """
    single_qubit_transformations = [
        gate_y_270(2),
        gate_x_90(2),
        identity(2),
    ]

    if invert:
        single_qubit_transformations = [
            jnp.linalg.inv(op) for op in single_qubit_transformations
        ]

    pauli_index = "xyz"

    if number_of_qubits == 1:
        return jnp.stack(single_qubit_transformations, axis=0), pauli_index

    multi_qubit_transformations = list(
        product(single_qubit_transformations, repeat=number_of_qubits)
    )

    multi_qubit_transformations = jnp.array(
        [tensor(*states) for states in multi_qubit_transformations]
    )

    pauli_index = [
        "".join(indices) for indices in product(pauli_index, repeat=number_of_qubits)
    ]

    return jnp.stack(multi_qubit_transformations, axis=0), pauli_index


def apply_basis_transformations(states, transforms):
    return jnp.einsum("...ij, bjk -> ...bik", states, transforms)


def apply_basis_transformations_dm(states, transforms):
    return jnp.einsum(
        "bik, ...kl, bjl -> ...bij", transforms, states, transforms.conj()
    )


def hamiltonian_from_dict(
    hamiltonian_params, number_of_qubits=1, return_filled_dict=False
):
    hamiltonian_params_ = {
        element: 0.0
        for element in pauli_strings(
            number_of_qubits=number_of_qubits, include_identity=True
        )
    }
    hamiltonian_params_.update(hamiltonian_params)

    operators, letters = pauli_operators(number_of_qubits=number_of_qubits)

    coefficients = jnp.array([hamiltonian_params_[idx] for idx in letters])

    if return_filled_dict:
        return jnp.einsum("i, ijk -> jk", coefficients, operators), hamiltonian_params_
    else:
        return jnp.einsum("i, ijk -> jk", coefficients, operators)


# Function for running optimization
def jump_operators_from_t1_and_t2(t1, t2):
    number_of_qubits = len(t1) if hasattr(t1, "__len__") else 1

    if number_of_qubits == 1:
        return [t1_decay(t1), t2_decay(t2, t1)]
    else:
        qubit_decays = [(t1_decay(t1_), t2_decay(t2_, t1_)) for t1_, t2_ in zip(t1, t2)]
        all_decays = []
        for qubit in range(number_of_qubits):
            for i in range(2):
                list_of_operators = [identity(2)] * number_of_qubits
                list_of_operators[qubit] = qubit_decays[qubit][i]
                decay = tensor(*list_of_operators)
                all_decays.append(decay)
    return jnp.array(all_decays)


def cholesko_matrix_from_params(cholesko_params, number_of_qubits=1):
    if number_of_qubits == 1:
        params = params.reshape(1, -1)

    choleskos = []
    for qubit in range(number_of_qubits):
        empty_matrix = jnp.zeros((3, 3), dtype=jnp.complex128)
        empty_matrix = empty_matrix.at[jnp.tril_indices(3)].set(
            cholesko_params[qubit, :6]
        )
        empty_matrix = empty_matrix.at[jnp.tril_indices(3, k=-1)].add(
            1j * cholesko_params[qubit, 6:]
        )

        choleskos.append(empty_matrix)
    return jnp.array(choleskos)


def pauli_matrix_from_cholesko_params(cholesko_params, number_of_qubits=1):
    paulis = []

    if number_of_qubits == 1:
        cholesko_params = cholesko_params.reshape(1, -1)

    for qubit in range(number_of_qubits):
        empty_matrix = jnp.zeros((3, 3), dtype=jnp.complex128)
        empty_matrix = empty_matrix.at[jnp.tril_indices(3)].set(
            cholesko_params[qubit, :6]
        )
        empty_matrix = empty_matrix.at[jnp.tril_indices(3, k=-1)].add(
            1j * cholesko_params[qubit, 6:]
        )

        paulis.append(empty_matrix)

    cholesko = jnp.array(paulis)
    pauli_matrix = jnp.einsum(
        "ijk, ikl -> ijl", cholesko, cholesko.conj().transpose(0, 2, 1)
    )
    return pauli_matrix


pauli_gates = jnp.array([sigma_x(), sigma_y(), sigma_z()])


def pauli_matrix_to_jump_operators(pauli_matrix):
    """
    Converts a Pauli matrix to jump operators.

    Args:
        pauli_matrix: The Pauli matrix to be converted.

    Returns:
        The jump operators corresponding to the given Pauli matrix.
    """
    single_qubit_jump_operators = jnp.einsum(
        "...ij, jkl -> ...ikl", pauli_matrix, pauli_gates
    )

    I = identity(2)
    multi_qubit_operators = []
    for qubit in range(pauli_matrix.shape[0]):
        for jump_operator in range(single_qubit_jump_operators.shape[1]):
            multi_qubit_operators.append(
                tensor(
                    *[
                        (
                            I
                            if i != qubit
                            else single_qubit_jump_operators[qubit, jump_operator]
                        )
                        for i in range(pauli_matrix.shape[0])
                    ]
                )
            )
    return jnp.array(multi_qubit_operators)


# Measurement analysis
from tensorflow_probability.substrates.jax.distributions import multinomial


def get_probability_from_states(states, epsilon=1e-10):
    probs = jnp.einsum("...kk->...k", states)
    probs = jnp.clip(jnp.real(probs), epsilon, 1 - epsilon)

    return probs


def get_measurements_from_states(states, samples=1000, seed=0):
    probs = jnp.einsum("...kk->...k", states)
    probs = jnp.real(probs)

    key = jax.random.PRNGKey(seed)
    outcomes = multinomial.Multinomial(total_count=samples, probs=probs).sample(
        seed=key
    )

    return outcomes
