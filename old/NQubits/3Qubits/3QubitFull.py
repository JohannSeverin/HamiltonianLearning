import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
sys.path.append("../..")

from functools import partial
from utils import *
from hamiltonian_learning_utils import *


# PARAMETERS
NQUBITS = 3
DURATION = 10000
SAVE_EVERY = 10
INTERACTION_STRENGTH = 100e-6
LOCAL_STRENGTH = 1e-3
SEED = 42

# Setting up the parameters for the Hamiltonian
key = jax.random.PRNGKey(SEED)

key_local, key_interaction, key_jump = jax.random.split(key, 3)

local_hamiltoian_params = (
    jax.random.normal(key_local, (NQUBITS, 3), dtype=jnp.float64) * LOCAL_STRENGTH
)

connections = (
    tuple([0, 0, 1]),
    tuple([1, 2, 2]),
)

two_local_hamiltonnian_params = (
    jax.random.normal(key_interaction, (3, 3, 3)) * INTERACTION_STRENGTH
)

dissipation_matrix = 1e-4 * jax.random.normal(
    key_jump, (4**NQUBITS - 1, 4**NQUBITS - 1)
)


# Test 3 local hamiltonian
three_local_hamiltonian_params = (
    jax.random.normal(key_interaction, (1, 3, 3, 3)) * INTERACTION_STRENGTH**2
)

conn = ((0,), (1,), (2,))

three_local_hamiltonian = build_N_interaction_hamiltonian(
    conn, three_local_hamiltonian_params, 3, 3
)

hamiltonian = build_local_hamiltonian(local_hamiltoian_params, NQUBITS)
hamiltonian += build_N_interaction_hamiltonian(
    connections, two_local_hamiltonnian_params, NQUBITS, 2
)
hamiltonian += three_local_hamiltonian

# Reshaping to be a 2D array
hamiltonian = hamiltonian.reshape(2**NQUBITS, 2**NQUBITS)

# Creating the solver
solver = create_solver(
    t0=0,
    t1=DURATION,
    adjoint=True,
    tlist=jnp.arange(0, DURATION + SAVE_EVERY, SAVE_EVERY),
    number_of_jump_operators=4**NQUBITS - 1,
)

solver = jax.jit(solver)


# Initial States
states, states_name = generate_initial_states(NQUBITS, with_mixed_states=True)
transforms, transforms_name = generate_basis_transformations(NQUBITS)

pauli_dissipation_matrix_cholesky = jnp.zeros(
    (4**NQUBITS - 1, 4**NQUBITS - 1), dtype=jnp.complex128
)
pauli_dissipation_matrix_cholesky = pauli_dissipation_matrix_cholesky.at[
    jnp.tril_indices(4**NQUBITS - 1)
].set(dissipation_matrix[jnp.tril_indices(4**NQUBITS - 1)])

pauli_dissipation_matrix_cholesky = pauli_dissipation_matrix_cholesky.at[
    jnp.tril_indices(4**NQUBITS - 1, -1)
].add(1j * dissipation_matrix[jnp.triu_indices(4**NQUBITS - 1, 1)])


pauli_matrices = pauli_operators(NQUBITS)[0][1:]
jump_operators = jnp.einsum(
    "ij, jkl -> ikl", pauli_dissipation_matrix_cholesky, pauli_matrices
)


# pauli_dissipation_matrix = jnp.einsum(
#     "ij, kj -> ik",
#     pauli_dissipation_matrix_cholesky,
#     pauli_dissipation_matrix_cholesky.conj(),
# )

solution = solver(
    states,
    hamiltonian,
    jump_operators,
)


results = apply_basis_transformations_dm(solution.ys, transforms)

# Collect data in xarray
import xarray as xr

shape = [-1] + NQUBITS * [4] + NQUBITS * [3] + [2**NQUBITS, 2**NQUBITS]
shape_names = (
    ["time"]
    + [f"initial_state_{i}" for i in range(NQUBITS)]
    + [f"measurement_basis_{i}" for i in range(NQUBITS)]
    + ["rho_i", "rho_j"]
)

shape_coords = (
    [solution.ts]
    + [["m", "x", "y", "z"] for _ in range(NQUBITS)]
    + [["x", "y", "z"] for _ in range(NQUBITS)]
    + [jnp.arange(2**NQUBITS), jnp.arange(2**NQUBITS)]
)

shape_coords = dict(zip(shape_names, shape_coords))


# data = xr.DataArray(
#     results.reshape(shape),
#     dims=shape_names,
#     coords=shape_coords,
# )


# Simulate the measurements
from tensorflow_probability.substrates.jax import distributions as tfd

measurement_probs = jnp.einsum("...ii->...i", results).real


# Plot data
import matplotlib.pyplot as plt
from matplotlib import style

style.use(
    "/mnt/c/Users/msk377/OneDrive - University of Copenhagen/Desktop/jax_playground/presentation_style.mplstyle"
)


# plt.plot(data.time, data.sel(states="xxx", measurement_basis="zzz", rho_i=0, rho_j=0))
