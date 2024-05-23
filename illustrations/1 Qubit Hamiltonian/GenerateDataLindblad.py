import xarray as xr
import sys
import itertools

sys.path.append("../..")
from utils import *
from hamiltonian_learning_utils import *


# Load hyper params
NQUBITS = 1
SAMPLES = 100
DURATION = 10000
TIMESTEPS = 20
SEED = 1


# Setup the True Hamiltnonian
X, Y, Z = 1e-4, -2e-4, 1e-4

# Hamiltonian Parmas
hamiltonian_params = jnp.array([X, Y, Z]).reshape((NQUBITS, 3))
hamiltonian = build_local_hamiltonian(hamiltonian_params, NQUBITS)

jump_operator = jump_operators_from_t1_and_t2(t1=1e6, t2=1e6)


# Initial States
initial_states = generate_initial_states(NQUBITS, include_negative_states=True)[0]
basis_transformation = generate_basis_transformations(NQUBITS)[0]

# Define the solver
solver = create_solver(
    t1=DURATION,
    t0=0,
    adjoint=True,
    tlist=jnp.arange(0, DURATION + TIMESTEPS, TIMESTEPS),
    number_of_jump_operators=2,
)

# Solve the system
result = solver(
    initial_states,
    hamiltonian,
    jump_operator,
)

# Get the expectation values and samples from the result
observables = N_qubit_pauli_gate_set_with_identity(NQUBITS)[1:]
exp_values = expectation_values(result.ys, observables)

# Transform the result to the measurement basis
result_in_measurement_basis = apply_basis_transformations_dm(
    result.ys, basis_transformation
)

# Extract Probabilities and a sample of the true states
probs = get_probability_from_states(result_in_measurement_basis)
samples = get_measurements_from_states(result_in_measurement_basis, SAMPLES, seed=SEED)


reconstructed_exp_vals = 2 * samples[..., 0] / SAMPLES - 1


# Plot the Dataset
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)

fig.suptitle("Expectation Values")

for i, j in itertools.product(range(3), range(3)):

    ax[i, j].set_title(f"Init {['X', 'Y', 'Z'][i]} - Measure {['X', 'Y', 'Z'][j]}")

    ax[i, j].plot(
        result.ts,
        reconstructed_exp_vals[:, i, j],
        ".",
        label="Expectation Value [+]",
        color="C0",
        alpha=0.75,
    )
    ax[i, j].plot(
        result.ts,
        reconstructed_exp_vals[:, i + 3, j],
        ".",
        label="Expectation Value [-]",
        color="C1",
        alpha=0.75,
    )
    ax[i, j].plot(
        result.ts,
        exp_values[:, i, j],
        linewidth=2,
        color="k",
    )
    ax[i, j].plot(
        result.ts,
        exp_values[:, i + 3, j],
        linewidth=2,
        color="k",
    )

    if i == 2:
        ax[i, j].set_xlabel("Time (ns)")

    ax[i, j].set_ylabel(f"$<{['X', 'Y', 'Z'][j]}>$")


ax[0, -1].legend(loc="upper right")
fig.tight_layout()
fig.savefig("SampledDataLindblad.png")

# Save the data
sampled_outcome = samples.reshape(
    (len(result.ts),) + (6,) * NQUBITS + (3,) * NQUBITS + (2,) * NQUBITS
)

import xarray as xr

dataset = xr.DataArray(
    sampled_outcome,
    dims=["time"]
    + [f"initial_state_{i}" for i in range(NQUBITS)]
    + [f"measurement_basis_{i}" for i in range(NQUBITS)]
    + [f"outcome_{i}" for i in range(NQUBITS)],
    coords=dict(
        time=result.ts,
        initial_state_0=("x, y, z, -x, -y, -z".split(", ")),
        measurement_basis_0=("x, y, z".split(", ")),
        outcome_0=[0, 1],
    ),
)

dataset.to_netcdf("DatasetLindblad.nc")
