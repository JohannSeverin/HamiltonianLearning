import xarray as xr
import sys
import itertools

sys.path.append("../../..")
from utils import *
from hamiltonian_learning_utils import *

data = xr.open_dataset("dataset_for_johann_v5_1.nc")


# Load hyper params
NQUBITS = 3
SAMPLES = 1000
DURATION = 200
TIMESTEPS = 4


# Guess amplitude and seed
seed = 42
amplitude_local = 1e-3
amplitude_two_local = 1e-4
amplitude_three_local = 1e-4
amplitude_dissipation = 1e-3

# Give connections
two_local_connections = ((0, 0, 1), (1, 2, 2))
three_local_connections = ((0,), (1,), (2,))

dissipation_jump_operators = 4**NQUBITS - 1  # This is general for any NQUBITS

# Function for creating guesses
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

local_key, two_local_key, three_local_key, dissipation_key = jax.random.split(
    jax.random.PRNGKey(seed), 4
)


#### MOVE TO UTILS ####
def random_local_hamiltonian_parmas(key, nqubits):
    return jax.random.normal(key, (nqubits, 3), dtype=jnp.float64)


def random_n_local_hamiltoian_params(key, order, number_of_connections):
    shape = [number_of_connections] + [3] * order
    return jax.random.normal(key, shape, dtype=jnp.float64)


def random_general_dissipation_matrix(key, nqubits):
    return jax.random.normal(key, (4**nqubits - 1, 4**nqubits - 1))


#######################

# Create guesses
local_hamiltoian_params = amplitude_local * random_local_hamiltonian_parmas(
    local_key, NQUBITS
)
two_local_hamiltonnian_params = amplitude_two_local * random_n_local_hamiltoian_params(
    two_local_key, 2, len(two_local_connections[0])
)
three_local_hamiltonian_params = (
    amplitude_three_local
    * random_n_local_hamiltoian_params(
        three_local_key, 3, len(three_local_connections[0])
    )
)
dissipation_matrix = amplitude_dissipation * random_general_dissipation_matrix(
    dissipation_key, NQUBITS
)

# Additional Guesses For example if we know something
# dissipation_matrix = dissipation_matrix.at[-1, -1].set(1e-3)


#### Define Solver ####
solver = create_solver(
    t1=DURATION,
    t0=0,
    adjoint=True,
    tlist=jnp.arange(0, DURATION + TIMESTEPS, TIMESTEPS),
    number_of_jump_operators=dissipation_jump_operators,
)


### Initial states ###
states, states_name = generate_initial_states(NQUBITS, include_negative_states=True)
measurement_basis_transformations, basis_names = generate_basis_transformations(
    NQUBITS, invert=True
)
observable = N_qubit_pauli_gate_set_with_identity(NQUBITS)


##### Define loss function #####
# Will do it a few sub steps


# Get physical operators from the current params
def hamilton_from_params(
    local_hamiltoian_params,
    two_local_hamiltonnian_params,
    three_local_hamiltonian_params,
):
    # Local Hamiltonian
    hamiltonian = build_local_hamiltonian(local_hamiltoian_params, NQUBITS)

    # Two local Hamiltonian
    hamiltonian += build_N_interaction_hamiltonian(
        two_local_connections, two_local_hamiltonnian_params, NQUBITS, 2
    )

    # Three local Hamiltonian
    three_local_hamiltonian = build_N_interaction_hamiltonian(
        three_local_connections, three_local_hamiltonian_params, NQUBITS, 3
    )
    hamiltonian += three_local_hamiltonian

    # reshape and return
    return 2 * jnp.pi * hamiltonian.reshape(2**NQUBITS, 2**NQUBITS)


pauli_gates = N_qubit_pauli_gate_set_with_identity(NQUBITS)[1:]


def pauli_matrix_from_params(dissipation_params):
    # Define matrix
    pauli_dissipation_matrix_cholesky = jnp.zeros(
        (4**NQUBITS - 1, 4**NQUBITS - 1), dtype=jnp.complex128
    )
    # Set real part
    pauli_dissipation_matrix_cholesky = pauli_dissipation_matrix_cholesky.at[
        jnp.tril_indices(4**NQUBITS - 1)
    ].set(dissipation_params[jnp.tril_indices(4**NQUBITS - 1)])

    # Set imaginary part
    pauli_dissipation_matrix_cholesky = pauli_dissipation_matrix_cholesky.at[
        jnp.tril_indices(4**NQUBITS - 1, -1)
    ].add(1j * dissipation_params[jnp.triu_indices(4**NQUBITS - 1, 1)])

    return (
        pauli_dissipation_matrix_cholesky @ pauli_dissipation_matrix_cholesky.T.conj()
    )


def jump_operators_from_params(dissipation_params):
    # Define matrix
    pauli_dissipation_matrix_cholesky = jnp.zeros(
        (4**NQUBITS - 1, 4**NQUBITS - 1), dtype=jnp.complex128
    )
    # Set real part
    pauli_dissipation_matrix_cholesky = pauli_dissipation_matrix_cholesky.at[
        jnp.tril_indices(4**NQUBITS - 1)
    ].set(dissipation_params[jnp.tril_indices(4**NQUBITS - 1)])

    # Set imaginary part
    pauli_dissipation_matrix_cholesky = pauli_dissipation_matrix_cholesky.at[
        jnp.tril_indices(4**NQUBITS - 1, -1)
    ].add(1j * dissipation_params[jnp.triu_indices(4**NQUBITS - 1, 1)])

    # Multiply with pauli matrices and return jump operators
    return jnp.einsum("ij, jkl -> ikl", pauli_dissipation_matrix_cholesky, pauli_gates)


hamiltonian = hamilton_from_params(
    local_hamiltoian_params,
    two_local_hamiltonnian_params,
    three_local_hamiltonian_params,
)

jump_operators = jump_operators_from_params(dissipation_matrix)

final_solution = solver(states, hamiltonian, jump_operators)

final_solution_in_measurement_basis = apply_basis_transformations_dm(
    final_solution.ys, measurement_basis_transformations
)

# Expectation values
exp_values = expectation_values(final_solution.ys, observable).real
exp_values = jnp.moveaxis(exp_values, 0, -1)
exp_values = exp_values.reshape(
    (6,) * NQUBITS + (4,) * NQUBITS + (len(final_solution.ts),)
)


# Sample outcomes
outcome_states = final_solution.ys
outcome_states = apply_basis_transformations_dm(
    outcome_states, measurement_basis_transformations
)
outcome_probs = jnp.einsum("...ii->...i", outcome_states).real
outcome_probs = outcome_probs.reshape(
    (len(final_solution.ts),) + (6,) * NQUBITS + (3,) * NQUBITS + (2**NQUBITS,)
)

# Sample outcomes
from tensorflow_probability.substrates.jax import distributions as tfd

sampled_outcome = tfd.Multinomial(total_count=SAMPLES, probs=outcome_probs).sample(
    seed=jax.random.PRNGKey(42)
)
sampled_outcome = sampled_outcome.reshape(
    (len(final_solution.ts),) + (6,) * NQUBITS + (3,) * NQUBITS + (2,) * NQUBITS
)

# Place values in xarrays.
import xarray as xr

xr_data = xr.DataArray(
    sampled_outcome,
    dims=["time"]
    + [f"initial_state_{i}" for i in range(NQUBITS)]
    + [f"measurement_basis_{i}" for i in range(NQUBITS)]
    + [f"outcome_{i}" for i in range(NQUBITS)],
    coords=dict(
        time=final_solution.ts,
        initial_state_0=("x, y, z, -x, -y, -z".split(", ")),
        initial_state_1=("x, y, z, -x, -y, -z".split(", ")),
        initial_state_2=("x, y, z, -x, -y, -z".split(", ")),
        measurement_basis_0=("x, y, z".split(", ")),
        measurement_basis_1=("x, y, z".split(", ")),
        measurement_basis_2=("x, y, z".split(", ")),
        outcome_0=[0, 1],
        outcome_1=[0, 1],
        outcome_2=[0, 1],
    ),
)

xr_expectation = xr.DataArray(
    jnp.moveaxis(exp_values, -1, 0),
    dims=[
        "time",
        "initial_state_0",
        "initial_state_1",
        "initial_state_2",
        "observable_0",
        "observable_1",
        "observable_2",
    ],
    coords=dict(
        time=final_solution.ts,
        initial_state_0=("x, y, z, -x, -y, -z".split(", ")),
        initial_state_1=("x, y, z, -x, -y, -z".split(", ")),
        initial_state_2=("x, y, z, -x, -y, -z".split(", ")),
        observable_0=["I", "X", "Y", "Z"],
        observable_1=["I", "X", "Y", "Z"],
        observable_2=["I", "X", "Y", "Z"],
    ),
)


# Save data to dataset
dataset = xr.Dataset(dict(outcomes=xr_data, expectation_values=xr_expectation))
dataset.to_netcdf("3QubitFullDataset.nc")


# Save parameters
xr_local_hamiltonian = xr.DataArray(
    local_hamiltoian_params,
    dims=["qubit", "parameter"],
    coords=dict(
        qubit=[f"q{i}" for i in range(NQUBITS)],
        parameter=["x", "y", "z"],
    ),
)

xr_two_local_hamiltonian = xr.DataArray(
    two_local_hamiltonnian_params,
    dims=["two_connection", "parameter_1", "parameter_2"],
    coords=dict(
        two_connection=[f"q{i}-q{j}" for i, j in zip(*two_local_connections)],
        parameter_1=["x", "y", "z"],
        parameter_2=["x", "y", "z"],
    ),
)

xr_three_local_hamiltonian = xr.DataArray(
    three_local_hamiltonian_params,
    dims=["three_connection", "parameter_1", "parameter_2", "parameter_3"],
    coords=dict(
        three_connection=[
            f"q{i}-q{j}-q{k}" for i, j, k in zip(*three_local_connections)
        ],
        parameter_1=["x", "y", "z"],
        parameter_2=["x", "y", "z"],
        parameter_3=["x", "y", "z"],
    ),
)

xr_dissipation_matrix = xr.DataArray(
    dissipation_matrix,
    dims=["i", "j"],
    coords=dict(
        i=["".join(x) for x in (itertools.product("ixyz", repeat=3))][1:],
        j=["".join(x) for x in (itertools.product("ixyz", repeat=3))][1:],
    ),
)

dataset_params = xr.Dataset(
    dict(
        local_hamiltonian=xr_local_hamiltonian,
        two_local_hamiltonian=xr_two_local_hamiltonian,
        three_local_hamiltonian=xr_three_local_hamiltonian,
        dissipation_matrix=xr_dissipation_matrix,
    )
)

dataset_params.to_netcdf("3QubitFullParams.nc")
