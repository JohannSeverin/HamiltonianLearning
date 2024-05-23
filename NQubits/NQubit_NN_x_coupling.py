import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from functools import partial

sys.path.append("..")
# sys.path.append("../tensor_utils")

from utils import *
from hamiltonian_learning_utils import *

# from kronecker import *


# PARAMETERS
NQUBITS = 3
T1 = jnp.inf
DURATION = 1000
SAVE_EVERY = 10
INTERACTION_STRENGTH = 10e-3

# Setting up the parameters for the Hamiltonian
local_hamiltoian_params = jnp.zeros((NQUBITS, 3), dtype=jnp.float64)

connections = (
    tuple(range(NQUBITS - 1)),
    tuple(range(1, NQUBITS)),
)
two_local_hamiltonnian_params = jnp.zeros((NQUBITS - 1, 3, 3))
two_local_hamiltonnian_params = two_local_hamiltonnian_params.at[:, 0, 0].set(
    INTERACTION_STRENGTH
)

jump_operators = jnp.repeat(t1_decay(T1, 2)[None, :, :], NQUBITS, axis=0)


# Calculating the Hamiltonian and the jump operators
# Local
@jax.jit
def build_local_hamiltonian(params):
    local_hamiltonians = jnp.zeros((NQUBITS, 2, 2), dtype=jnp.complex128)
    for qubit in range(NQUBITS):
        H_i = jnp.einsum("i, ijk -> jk", params[qubit], pauli_gate_set)
        local_hamiltonians = local_hamiltonians.at[qubit].set(H_i)

    return tensor_sum(jnp.array(local_hamiltonians))


hamiltonian = build_local_hamiltonian(local_hamiltoian_params)

# Interactions
two_qubit_paulis = two_qubit_pauli_gate_set.reshape(3, 3, 2, 2, 2, 2)


@partial(jax.jit, static_argnames=("connections"))
def build_interaction_hamiltonian(connections, params):
    two_local_hamiltonians = jnp.zeros((NQUBITS - 1, 2, 2, 2, 2), dtype=jnp.complex128)
    for i in range(NQUBITS - 1):
        two_local_hamiltonians = two_local_hamiltonians.at[i].set(
            jnp.einsum("ij, ij... -> ...", params[i], two_qubit_paulis)
        )
    # return two_local_hamiltonians

    return sum_two_qubit_interaction_tensors(
        connections, two_local_hamiltonians, NQUBITS
    )


interaction_hamiltonian = build_interaction_hamiltonian(
    connections, two_local_hamiltonnian_params
)
hamiltonian += interaction_hamiltonian

# Adding the interaction terms
lindblad = tensor_sum(jump_operators)


# Reshape everything to be run on the simulators
H = hamiltonian.reshape(2**NQUBITS, 2**NQUBITS)
L = lindblad.reshape(1, 2**NQUBITS, 2**NQUBITS)

# Running the simulation
solver = create_solver(
    DURATION,
    0,
    adjoint=True,
    number_of_jump_operators=1,
    tlist=jnp.arange(0, DURATION + SAVE_EVERY, SAVE_EVERY),
)

initial_states = jnp.array([basis_dm(1, 2)] * NQUBITS)
initial_state = tensor_product(initial_states).reshape(1, 2**NQUBITS, 2**NQUBITS)


states = solver(initial_state=initial_state, hamiltonian=H, jump_operators=L)

excited_state = basis_dm(1, 2)
occupations = jnp.array(
    [tensor_at_index(excited_state, i, NQUBITS) for i in range(NQUBITS)]
).reshape(NQUBITS, 2**NQUBITS, 2**NQUBITS)

# Calculating the expectation values
exp = expectation_values(states.ys, occupations)


import matplotlib.pyplot as plt

# for i in range(NQUBITS - 4):
plt.plot(exp[:, 0, 0], label=f"Qubit {0}")
# plt.plot(exp[:, 0, 1], label=f"Qubit {1}")
# plt.plot(exp[:, 0, 2], label=f"Qubit {2}")
# plt.plot(exp[:, 0, 3], label=f"Qubit {3}")
# plt.plot(exp[:, 0, 3], label=f"Qubit {3}")

plt.legend()
