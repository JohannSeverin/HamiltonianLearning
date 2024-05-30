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
T1 = jnp.inf
DURATION = 1000
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

jump_operators = jnp.repeat(t1_decay(T1, 2)[None, :, :], NQUBITS, axis=0)

# Add together to simulated params
# hamiltonian = build_local_hamiltonian(local_hamiltoian_params, NQUBITS)

# interaction_hamiltonian = build_interaction_hamiltonian(
#     connections, two_local_hamiltonnian_params, NQUBITS
# )
# hamiltonian += interaction_hamiltonian


# Test 3 local hamiltonian
three_local_hamiltonian_params = (
    jax.random.normal(key_interaction, (1, 3, 3, 3)) * INTERACTION_STRENGTH
)

conn = ((0,), (1,), (2,))

build_N_interaction_hamiltonian(conn, three_local_hamiltonian_params, 3, 3)


# Adding the interaction terms
lindblad = tensor_sum(jump_operators)


# Reshaping to be a 2D array
hamiltonian = hamiltonian.reshape(2**NQUBITS, 2**NQUBITS)
lindblad = lindblad.reshape(1, 2**NQUBITS, 2**NQUBITS)

# Creating the solver
solver = create_solver(
    t0=0,
    t1=DURATION,
    adjoint=True,
    tlist=jnp.arange(0, DURATION + SAVE_EVERY, SAVE_EVERY),
    number_of_jump_operators=1,
)


# Initial States
states, states_name = generate_initial_states(NQUBITS, with_mixed_states=True)
transforms, transforms_name = generate_basis_transformations(NQUBITS)

solution = solver(states, hamiltonian, lindblad)


results = apply_basis_transformations(solution.ys, transforms)

# Collect data in xarray
import xarray as xr

data = xr.DataArray(
    results,
    dims=("time", "states", "measurement_basis", "rho_i", "rho_j"),
    coords={
        "time": solution.ts,
        "states": states_name,
        "measurement_basis": transforms_name,
        "rho_i": jnp.arange(2**NQUBITS),
        "rho_j": jnp.arange(2**NQUBITS),
    },
)

# Plot data
import matplotlib.pyplot as plt
from matplotlib import style

style.use(
    "/mnt/c/Users/msk377/OneDrive - University of Copenhagen/Desktop/jax_playground/presentation_style.mplstyle"
)
