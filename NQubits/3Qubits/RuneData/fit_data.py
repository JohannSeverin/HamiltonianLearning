import xarray as xr
import sys
import itertools

sys.path.append("../../..")
from utils import *
from hamiltonian_learning_utils import *

data = xr.open_dataset("dataset_for_johann_v4.nc")


data.initial_state1


# Load hyper params
NQUBITS = 3
time = data.time.values

data_expectation = data.expectations.values
data_error = data.uncertainties.values

data_expectation += jax.random.normal(jax.random.PRNGKey(42), data_expectation.shape) * data_error

# Fitting params
ITERATIONS = 500
LEARNING_RATE = 1e-3

# Guess amplitude and seed
seed = 41
amplitude_local = 1e-6
amplitude_two_local = 1e-6
amplitude_three_local = 1e-6
amplitude_dissipation = 1e-6

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
    two_local_key, 2, len(two_local_connections)
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
dissipation_matrix = dissipation_matrix.at[-1, -1].set(1e-3)


#### Define Solver ####
solver = create_solver(
    t1=time.max(),
    t0=0,
    adjoint=True,
    tlist=time,
    number_of_jump_operators=dissipation_jump_operators,
)


### Initial states ###
states, states_name = generate_initial_states(NQUBITS, with_mixed_states=True)
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


def loss_fn(
    local_hamiltoian_params,
    two_local_hamiltonnian_params,
    three_local_hamiltonian_params,
    dissipation_params,
    data_values=data_expectation,
    data_error=data_error,
):
    # Get Hamiltonian
    hamiltonian = hamilton_from_params(
        local_hamiltoian_params,
        two_local_hamiltonnian_params,
        three_local_hamiltonian_params,
    )

    # Get jump operators
    jump_operators = jump_operators_from_params(dissipation_params)

    # Solve
    solution = solver(states, hamiltonian, jump_operators)

    # Calculate expectation values and set in right format
    exp_values = expectation_values(solution.ys, observable).real
    exp_values = jnp.moveaxis(exp_values, 0, -1)
    exp_values = exp_values.reshape((4,) * 2 * NQUBITS + (len(time),))

    return jnp.sum((exp_values - data_values) ** 2 / data_error**2)


loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, (0, 1, 2, 3)))

### Setup the optimizer ###
from optax import adam, apply_updates

opt = adam(LEARNING_RATE)
opt_state = opt.init(
    (
        local_hamiltoian_params,
        two_local_hamiltonnian_params,
        three_local_hamiltonian_params,
        dissipation_matrix,
    )
)

for i in range(ITERATIONS):
    loss_val, grad = loss_and_grad(
        local_hamiltoian_params,
        two_local_hamiltonnian_params,
        three_local_hamiltonian_params,
        dissipation_matrix,
    )
    updates, opt_state = opt.update(grad, opt_state)
    (
        local_hamiltoian_params,
        two_local_hamiltonnian_params,
        three_local_hamiltonian_params,
        dissipation_matrix,
    ) = apply_updates(
        (
            local_hamiltoian_params,
            two_local_hamiltonnian_params,
            three_local_hamiltonian_params,
            dissipation_matrix,
        ),
        updates,
    )
    if i % 1 == 0:
        print(f"Iteration {i}, Loss {loss_val:.3e}")


hamiltonian = hamilton_from_params(
    local_hamiltoian_params,
    two_local_hamiltonnian_params,
    three_local_hamiltonian_params,
)

jump_operators = jump_operators_from_params(dissipation_matrix)

final_solution = solver(states, hamiltonian, jump_operators)

exp_values = expectation_values(final_solution.ys, observable).real
exp_values = jnp.moveaxis(exp_values, 0, -1)
exp_values = exp_values.reshape((4,) * 2 * NQUBITS + (len(time),))


# Lindblad analysis
pauli_matrix = jnp.zeros((4**NQUBITS, 4**NQUBITS), dtype=jnp.complex128)
pauli_matrix = pauli_matrix.at[1:, 1:].set(pauli_matrix_from_params(dissipation_matrix))
eigvals, eigvecs = jnp.linalg.eigh(pauli_matrix)


%matplotlib widget
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display



# Local Lindbladian parts
pauli_matrix = pauli_matrix.reshape(4, 4, 4, 4, 4, 4)

# Local
fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

labels = ["".join(i) for i in itertools.product("IXYZ", repeat=1)]

ax[0].set(title="Qubit 0")
ax[1].set(title="Qubit 1")
ax[2].set(title="Qubit 2")

for i in range(3):
    ax[i].set_xticks(jnp.arange(4))
    ax[i].set_xticklabels(labels)
    ax[i].set_yticks(jnp.arange(4))
    ax[i].set_yticklabels(labels)


im0 = ax[0].imshow(jnp.abs(pauli_matrix[:, 0, 0, :, 0, 0]), cmap="plasma")
im1 = ax[1].imshow(jnp.abs(pauli_matrix[0, :, 0, 0, :, 0]), cmap="plasma")
im2 = ax[2].imshow(jnp.abs(pauli_matrix[0, 0, :, 0, 0, :]), cmap="plasma")

fig.colorbar(im0, ax=ax[0])
fig.colorbar(im1, ax=ax[1])
fig.colorbar(im2, ax=ax[2])

fig.tight_layout()


# Two local
fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

ax[0].set(title="Qubit 0-1")
ax[1].set(title="Qubit 0-2")
ax[2].set(title="Qubit 1-2")


labels = ["".join(i) for i in itertools.product("IXYZ", repeat=2)]

for i in range(3):
    ax[i].set_xticks(jnp.arange(16))
    ax[i].set_xticklabels(labels)
    ax[i].set_yticks(jnp.arange(16))
    ax[i].set_yticklabels(labels)


im0 = ax[0].imshow(
    jnp.abs(pauli_matrix[:, :, 0, :, :, 0]).reshape(16, 16), cmap="plasma"
)
im1 = ax[1].imshow(
    jnp.abs(pauli_matrix[:, 0, :, :, 0, :]).reshape(16, 16), cmap="plasma"
)
im2 = ax[2].imshow(
    jnp.abs(pauli_matrix[0, :, :, 0, :, :]).reshape(16, 16), cmap="plasma"
)

fig.colorbar(im0, ax=ax[0])
fig.colorbar(im1, ax=ax[1])
fig.colorbar(im2, ax=ax[2])

fig.tight_layout()



# Lindbladian Terms Plots

fig, ax = plt.subplots(1, figsize=(10, 10))

im = ax.imshow(jnp.abs(pauli_matrix).reshape(64, 64), cmap="plasma")
fig.colorbar(im, ax=ax)
labels = ["".join(i) for i in itertools.product("IXYZ", repeat=3)]

ax.set_xticks(jnp.arange(64))
ax.set_xticklabels(labels)
ax.set_yticks(jnp.arange(64))
ax.set_yticklabels(labels)



# Check some examples and compare with data
# T2
dropdown_widgets = {}

for i in range(NQUBITS):
    dropdown_widgets[f"Qubit {i} Initial"] = widgets.Dropdown(
        options=["I", "X", "Y", "Z"],
        value="I",
        description=f"Qubit {i} Initial:",
        layout=widgets.Layout(width="200px"),
    )

    dropdown_widgets[f"Qubit {i} Measure"] = widgets.Dropdown(
        options=["I", "X", "Y", "Z"],
        value="I",
        description=f"Qubit {i} Measure:",
        layout=widgets.Layout(width="200px"),
    )

# Create hbox for the first row of dropdown widgets
hbox1 = widgets.HBox([dropdown_widgets[f"Qubit {i} Initial"] for i in range(NQUBITS)])

# Create hbox for the second row of dropdown widgets
hbox2 = widgets.HBox([dropdown_widgets[f"Qubit {i} Measure"] for i in range(NQUBITS)])

#display(hbox1, hbox2)

key_mapping = {key: value for key, value in zip("IXYZ", range(4))}

fig, ax = plt.subplots(ncols=1)
# Create interactive plot function
def plot_interactive(**kwargs):
    init = tuple(key_mapping[k] for k in [kwargs[f"Qubit {i} Initial"] for i in range(NQUBITS)])
    measure = tuple(key_mapping[k] for k in [kwargs[f"Qubit {i} Measure"] for i in range(NQUBITS)])

    data_index = init + measure

    #print(init, measure)

    ax.clear()
    
    init_title = "".join([f"{k}" for k in [kwargs[f"Qubit {i} Initial"] for i in range(NQUBITS)]])
    measure_title = "".join([f"{k}" for k in [kwargs[f"Qubit {i} Measure"] for i in range(NQUBITS)]])

    ax.set(title=f"Initial: {init_title} - Measured: {measure_title}", xlabel="Time (ns)", ylabel="Expectation of X")
    ax.plot(time, exp_values[*data_index], label="Fit", c="k")
    ax.plot(time, data_expectation[*data_index], "r.", label="Data")
    ax.errorbar(
        time,
        data_expectation[*data_index],
        yerr=data_error[*data_index],
        fmt="none",
        c="r",
    )

# Create interactive widget
interactive_plot = widgets.interactive_output(
    plot_interactive, dropdown_widgets
)

# Display the interactive widget
display(hbox1, hbox2, interactive_plot)
