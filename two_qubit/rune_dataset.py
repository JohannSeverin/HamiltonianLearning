import sys
sys.path.append("../")
from utils import *
from hamiltonian_learning_utils import *
from tensorflow_probability.substrates.jax.distributions import multinomial
from ipywidgets import widgets
%matplotlib widget

# CONSTANTS
NQUBITS = 2
DURATION = 400
STORE_EVERY = 4
SAMPLES = 100

EPOCHS = 750
LEARNING_RATE = 2e-3

# Dataset
import xarray as xr

data = xr.load_dataarray("dataset_for_johann_v3.nc")
data_jax = jnp.array(data.values)


# Initial guesses
two_qubit_pauli_strings = pauli_strings(NQUBITS)
initial_guess_hamiltonian = (
    jax.random.normal(jax.random.PRNGKey(3), (len(two_qubit_pauli_strings),)) * 1e-2
)

initial_guess_lindbladian = (
    jax.random.normal(jax.random.PRNGKey(1), (NQUBITS, 9)) * 1e-2
).flatten()

hamiltonian = hamiltonian_from_dict(
    {
        key: value
        for key, value in zip(two_qubit_pauli_strings, initial_guess_hamiltonian)
    },
    number_of_qubits=NQUBITS,
)

pauli_matrices = pauli_matrix_from_cholesko_params(initial_guess_lindbladian.reshape(NQUBITS, 9), NQUBITS)
jump_operators = pauli_matrix_to_jump_operators(pauli_matrices)


# Setup for the simulation
initial_states, init_index = generate_initial_states(NQUBITS, with_mixed_states=True )
initial_states = initial_states.reshape(4, 4, 4, 4)


# Evolution function
solver = create_solver(
    t1=DURATION,
    t0=0,
    tlist=data.time.values,
    number_of_jump_operators=jump_operators.shape[0],
    adjoint=True,
)

# Check simulation
results = solver(
    initial_state=initial_states,
    hamiltonian=hamiltonian,
    jump_operators=jump_operators,
)


# Training Setup
observation_operators = pauli_operators(NQUBITS)[0].reshape(4, 4, 4, 4)


def approximate_standard_deviation(expectation_values, epsilon=1e-6):
    probs = jnp.clip((expectation_values + 1) / 2, epsilon, 1 - epsilon)
    return jnp.sqrt(probs * (1 - probs) / SAMPLES)


def loss_fn(
    hamiltonian_params,
    lindbladian_params,
    initial_states,
    observation_operators=observation_operators,
    data=data_jax,
):
    hamiltonian = hamiltonian_from_dict(
        {key: value for key, value in zip(two_qubit_pauli_strings, hamiltonian_params)},
        number_of_qubits=NQUBITS,
    )
    # pauli_matrices = pauli_matrix_from_cholesko_params(lindbladian_params, NQUBITS)
    # jump_operators = pauli_matrix_to_jump_operators(pauli_matrices)
    lindbladian_params = lindbladian_params.reshape(NQUBITS, 9)
    cholesko = cholesko_matrix_from_params(lindbladian_params, NQUBITS)
    jump_operators = pauli_matrix_to_jump_operators(cholesko)

    results = solver(
        initial_state=initial_states,
        hamiltonian=hamiltonian,
        jump_operators=jump_operators,
        
    )

    expectation_values = jnp.einsum(
        "tijkl, mnlk -> ijmnt", results.ys, observation_operators
    ).real

    approximate_errs = approximate_standard_deviation(expectation_values)

    return jnp.sum((data - expectation_values) ** 2 / approximate_errs**2) / 2


from jax import value_and_grad

loss_and_grad = value_and_grad(loss_fn, argnums=(0, 1))

# Optimization Loop


hamiltonian_params = initial_guess_hamiltonian
lindbladian_params = initial_guess_lindbladian

from optax import adam, apply_updates

opt = adam(LEARNING_RATE)
opt_state = opt.init((hamiltonian_params, lindbladian_params))

for epoch in range(EPOCHS):
    loss_val, grad = loss_and_grad(
        hamiltonian_params,
        lindbladian_params,
        initial_states,
        observation_operators,
        data_jax,
    )
    updates, opt_state = opt.update(grad, opt_state)
    hamiltonian_params, lindbladian_params = apply_updates(
        (hamiltonian_params, lindbladian_params), updates
    )

    print(f"Epoch: {epoch}, Loss: {loss_val}")

    # if epoch % 25 == 0:
    #     plot_hamiltonian_guesses(hamiltonian_params / 2 / jnp.pi, title=f"Epoch{epoch}")


# Evolve the states with final parameters
hamiltonian = hamiltonian_from_dict(
    {key: value for key, value in zip(two_qubit_pauli_strings, hamiltonian_params)},
    number_of_qubits=NQUBITS,
)
# pauli_matrices = pauli_matrix_from_cholesko_params(lindbladian_params, NQUBITS)
# jump_operators = pauli_matrix_to_jump_operators(pauli_matrices)

cholesko = cholesko_matrix_from_params(lindbladian_params.reshape(NQUBITS, 9), NQUBITS)
jump_operators = pauli_matrix_to_jump_operators(cholesko)

results = solver(
    initial_state=initial_states,
    hamiltonian=hamiltonian,
    jump_operators=jump_operators,
)

expectation_values = jnp.einsum(
    "tijkl, mnlk -> ijmnt", results.ys, observation_operators
).real

# Plot guesses

from matplotlib import style
import matplotlib.pyplot as plt

style.use(
    "/mnt/c/Users/msk377/OneDrive - University of Copenhagen/Desktop/jax_playground/presentation_style.mplstyle"
)


def plot_hamiltonian_guesses(hamiltonian_params, title="InitialGuesses", errs=None):
    hamiltonian_params = {
        key: value for key, value in zip(two_qubit_pauli_strings, hamiltonian_params)
    }

    fig, ax = plt.subplots(figsize=(10, 5), sharex=True)
    ax.plot(
        jnp.arange(len(hamiltonian_params)),
        list(hamiltonian_params.values()),
        "o",
        label="Parameters",
    )

    if errs is not None:
        ax.errorbar(
            jnp.arange(len(hamiltonian_params)),
            list(hamiltonian_params.values()),
            yerr=errs,
            fmt=".",
            label="Parameters",
            color="C0",
        )

    ax.set_xticks(jnp.arange(len(hamiltonian_params)))
    ax.set_xticklabels(hamiltonian_params.keys())
    ax.set_title(title)


from jax import hessian

hess = hessian(loss_fn, argnums=(0))
cov = jnp.linalg.inv(
    hess(
        hamiltonian_params,
        lindbladian_params,
        initial_states,
        observation_operators,
        data_jax,
    )
)

errs = jnp.sqrt(jnp.diag(cov))

hess_lindblad = hessian(loss_fn, argnums=(1))
cov_lindblad = jnp.linalg.inv(
    hess_lindblad(
        hamiltonian_params,
        lindbladian_params,
        initial_states,
        observation_operators,
        data_jax,
    )
)

errs_lindblad = jnp.sqrt(jnp.diag(cov_lindblad))


# Interpret the lindbladian result 
cholesko = cholesko_matrix_from_params(lindbladian_params.reshape(NQUBITS, 9), NQUBITS)
pauli_matrices = cholesko.conj().transpose([0, 2, 1]) @ cholesko
pauli_matrices[0]

eigvals_0, eigvecs_0 = jnp.linalg.eig(pauli_matrices[0])
eigvals_1, eigvecs_1 = jnp.linalg.eig(pauli_matrices[1])


plot_hamiltonian_guesses(hamiltonian_params / 2 / jnp.pi, errs=errs / 2 / jnp.pi, title = "Hamiltonian Guesses")
plt.ylim(-1e-4, 1e-4)
plt.show()


# Plot the lindbladian result
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

lindbladian_params 
errs_lindblad = jnp.nan_to_num(errs_lindblad, nan=0)

plt.errorbar(jnp.arange(2 * 9), lindbladian_params, yerr=errs_lindblad, fmt="o")
plt.show()


def get_decay_rates(lindbladian_params):
    cholesko = cholesko_matrix_from_params(lindbladian_params.reshape(NQUBITS, 9), NQUBITS)
    pauli_matrices = cholesko.conj().transpose([0, 2, 1]) @ cholesko
    decay_rates = jnp.zeros(NQUBITS * 3)
    for i in range(NQUBITS):
        eigvals = jnp.linalg.eigvalsh(pauli_matrices[i])
        decay_rates = decay_rates.at[i * 3 : (i + 1) * 3].set(eigvals)
    return decay_rates

def get_decay_jump_operators(lindbladian_params):
    cholesko = cholesko_matrix_from_params(lindbladian_params.reshape(NQUBITS, 9), NQUBITS)
    pauli_matrices = cholesko.conj().transpose([0, 2, 1]) @ cholesko
    decay_rates = jnp.zeros(NQUBITS * 3)
    jump_operators = jnp.zeros((NQUBITS, 3, 3), dtype=jnp.complex128)
    for i in range(NQUBITS):
        eigvals, eigvecs = jnp.linalg.eigh(pauli_matrices[i])
        decay_rates = decay_rates.at[i * 3 : (i + 1) * 3].set(eigvals)
        jump_operators = jump_operators.at[i].set(eigvecs)

    return decay_rates, jump_operators


rates, rates_errs = propagate_uncertainties(get_decay_rates, lindbladian_params, errs_lindblad)

decay_rates, decay_jumps = get_decay_jump_operators(lindbladian_params)



fig, ax = plt.subplots()
ax.errorbar(jnp.arange(2 * 3), rates, yerr=rates_errs, fmt="o")
ax.set(
    title 
)


# ppp 



# Plot some examples
import matplotlib.pyplot as plt
from ipywidgets import interactive
import xarray as xr
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

# Create dropdown widgets for initial state and measurement basis
init_dropdown_1 = widgets.Dropdown(
    options=data.initial_state1.values,
    description="Initial State:",
    value=data.initial_state1[0],
)
init_dropdown_2 = widgets.Dropdown(
    options=data.initial_state2.values,
    description="Initial State:",
    value=data.initial_state2[0],
)
observable_dropdown_1 = widgets.Dropdown(
    options=data.observable1.values,
    description="Observable:",
    value=data.observable1[0],
)
observable_dropdown_2 = widgets.Dropdown(
    options=data.observable2.values,
    description="Observable:",
    value=data.observable2[0],
)


# Create a button widget for plotting
plot_button = widgets.Button(description="Plot")

key_to_idx = {
    "id": 0,
    "x": 1,
    "y": 2,
    "z": 3,
}


fig, ax = plt.subplots(figsize=(8, 6))


# Define a function to handle button click event
def plot_button_clicked(init_1, init_2, obs_1, obs_2):
    # init_1 = init_dropdown_1.value
    # init_2 = init_dropdown_2.value
    # obs_1 = observable_dropdown_1.value
    # obs_2 = observable_dropdown_2.value

    # Clear the current figure
    ax.cla()

    # Plot the selected data
    data.sel(
        initial_state1=init_1,
        initial_state2=init_2,
        observable1=obs_1,
        observable2=obs_2,
    ).plot(ax=ax, label="Data")

    ax.plot(
        data.time.values,
        expectation_values[
            key_to_idx[init_1],
            key_to_idx[init_2],
            key_to_idx[obs_1],
            key_to_idx[obs_2],
            :,
        ],
        label="Fit",
    )
    ax.set_ylim(-1, 1)
    ax.legend()

from ipywidgets import interact_manual

interact_manual(
    plot_button_clicked,
    # button=plot_button,
    init_1=init_dropdown_1,
    init_2=init_dropdown_2,
    obs_1=observable_dropdown_1,
    obs_2=observable_dropdown_2,
)
