import sys, os

import jax
import jax.numpy as jnp
import xarray as xr

from functools import partial

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# jax.config.update("jax_platform_name", 'cpu')

dataset = xr.load_dataset("dataset.zarr", engine="zarr")

data = dataset.sampled_outcome
data = jnp.array(data)

# System parameters
NQUBITS = int(dataset.attrs["NQUBITS"])

# Experiment parameters
TIMES = jnp.array(dataset.time)
INIT_STATES = ["Z", "X", "Y", "-Z"]
MEASUREMENT_BASIS = ["Z", "X", "Y"]
SAMPLES = int(dataset.attrs["SAMPLES"])

# Model parameters
PERFECT_PREPARATION = True
# MEASUREMENT_LOCALITY = 1
HAMILTONIAN_LOCALITY = 2
LINDLAD_LOCALITY = 1

CLIP = 1e-3 # Clip harder if no dissipation or SPAM

# Solver parameters
INITIAL_STEPSIZE = 1.0
MAX_STEPS = 1000
ODE_SOLVER = "Dopri5"
STEPSIZE_CONTROLLER = "adaptive"
ADJOINT = True
TOLERANCE = 1e-6

# Optimizer Params
LOSS = "squared_difference" # "squared_diffrence" OR "likelihood"
USE_WEIGHTS = False # If squared differences this determines if we should weigh the points according to the estimated std


LEARNING_RATE = 1e-4
ITERATIONS = 100
HAMILTONIAN_GUESS_ORDER = 1e-3
LINDLADIAN_GUESS_ORDER = 1e-5

# Memory to computational time trade-offs
BATCH_SIZE_INIT_STATES = 32 # Number of initial states to use in the batch
SCAN_TIMES = True # If true timesteps are looped to save memory


# Import the necessary modules
sys.path.append(
    "../../../hamiltonian_learning_refactor"
)
from hamiltonian_learning import (
    Measurements,
    StatePreparation,
    Solver,
    Parameterization,
)

dynamics = Parameterization(
    NQUBITS,
    hamiltonian_locality=HAMILTONIAN_LOCALITY,
    lindblad_locality=LINDLAD_LOCALITY,
    hamiltonian_amplitudes=HAMILTONIAN_GUESS_ORDER,
    lindblad_amplitudes=LINDLADIAN_GUESS_ORDER,
)

state_preparation = StatePreparation(
    NQUBITS,
    initial_states=INIT_STATES,
    perfect_state_preparation=PERFECT_PREPARATION,
)

measurements = Measurements(
    NQUBITS,
    basis=MEASUREMENT_BASIS,
    perfect_measurement=True,
    clip=CLIP,
    samples=SAMPLES,
    loss=LOSS,
)

# Define the solver for time dynamics
solver = Solver(
    times=TIMES,
    initial_stepsize=INITIAL_STEPSIZE,
    max_steps=MAX_STEPS,
    ode_solver=ODE_SOLVER,
    stepsize_controller=STEPSIZE_CONTROLLER,
    adjoint=ADJOINT,
    tolerance=TOLERANCE,
)


# Get the initial parameters to define solve the problem
hamiltonian_params = dynamics.hamiltonian_params
lindbladian_params = dynamics.lindbladian_params
state_preparation_params = - 3e-3 * jnp.ones_like(state_preparation.state_preparation_params)


# Get the generators to convert the parameters to states or operators
generate_initial_states = state_preparation.get_initial_state_generator()
generate_hamiltonian = dynamics.get_hamiltonian_generator()
generate_lindbladian = dynamics.get_jump_operator_generator()
calculate_log_likelihood = measurements.get_loss_fn()
evolve_states = solver.create_solver()

# Define Helper Functions 
def body_func(current_sum, states):
    evolved_states, data = states
    log_probability = calculate_log_likelihood(evolved_states, data)
    return current_sum + log_probability, log_probability


# Define the loss function
@jax.jit
def loss_fn(params):
    """
    Loss function to be minimized
    """
    state_preparation_params, hamiltonian_params, lindbladian_params = params

    initial_states = generate_initial_states(3e3 * state_preparation_params) # .reshape(-1, 4, 4)
    hamiltonian = generate_hamiltonian(hamiltonian_params)
    lindblad_operators = generate_lindbladian(lindbladian_params)

    # Evolve the states
    evolved_states = evolve_states(initial_states, hamiltonian, lindblad_operators)

    # # Calculate the log probability    
    if SCAN_TIMES:
        # Use scan to loop over the times and save memory
        log_probability, _ = jax.lax.scan(
            body_func,
            0,
            (evolved_states, data),
        )
    # Otherwise calculate the log probability for all times at once
    else:
        log_probability = calculate_log_likelihood(evolved_states, data)

    return log_probability / data.size # Normalize to regularize gradient

@jax.jit
def loss_fn_batched(params, batch_indices):
    """
    Loss function to be minimized
    """
    state_preparation_params, hamiltonian_params, lindbladian_params = params

    initial_states = generate_initial_states(3e3 * state_preparation_params)[batch_indices]
    hamiltonian = generate_hamiltonian(hamiltonian_params)
    lindblad_operators = generate_lindbladian(lindbladian_params)

    data_batched = data[:, batch_indices]

    # Evolve the states
    evolved_states = evolve_states(initial_states, hamiltonian, lindblad_operators)

    # # Calculate the log probability    
    if SCAN_TIMES:
        # Use scan to loop over the times and save memory
        log_probability, _ = jax.lax.scan(
            body_func,
            0,
            (evolved_states, data_batched),
        )
    # Otherwise calculate the log probability for all times at once
    else:
        log_probability = calculate_log_likelihood(evolved_states, data_batched)

    return log_probability / data_batched.size # Normalize to regularize gradient


# Generate the gradient function
loss_and_grad = jax.value_and_grad(loss_fn, argnums=(0)) if BATCH_SIZE_INIT_STATES == None else jax.value_and_grad(loss_fn_batched, argnums=(0))

# Fit the model
from optax import adam, apply_updates

# Init
params = (state_preparation_params, hamiltonian_params, lindbladian_params)
opt = adam(LEARNING_RATE)
state = opt.init(params)

# ASSUME BATCHED VERSION FOR NOW!
key = jax.random.PRNGKey(0)

from tqdm import tqdm

# Test 
def update_func(params, state, batch_indices):
    loss, grads = loss_and_grad(params, batch_indices)
    updates, state = opt.update(grads, state, params)
    params = apply_updates(params, updates)
    return params, state, loss

for i in range(ITERATIONS):
    batch_key, key = jax.random.split(key) 
    index_list = jax.random.permutation(batch_key, len(dataset.initial_gate))

    for j in tqdm(range(0, len(dataset.initial_gate), BATCH_SIZE_INIT_STATES)):
        batch_indices = index_list[j:j+BATCH_SIZE_INIT_STATES]
        params, state, loss = update_func(params, state, batch_indices)
        print(f"Iteration {i:03d}  - Loss: {loss:.3e}")




# Test the functions
state_preparation_params, hamiltonian_params, lindbladian_params = params
states = generate_initial_states(3e3 * state_preparation_params)#.reshape(-1, 4, 4)
hamiltonian = generate_hamiltonian(hamiltonian_params)
lindblad_operators = generate_lindbladian(lindbladian_params)
evolved_states = evolve_states(states, hamiltonian, lindblad_operators)

measurements.generate_samples(evolved_states, SAMPLES)

probs = measurements.calculate_measurement_probabilities(evolved_states)

simulated = dataset.measurement_probabilities.copy()
simulated.values = probs

print("final loss: " + str(loss_fn(params))) # , data

##########################
# Plot the results       #
##########################

%matplotlib widget 
import matplotlib.pyplot as plt
from ipywidgets import interact, Dropdown
from IPython.display import display


# We plot the z1, z2, and z1z2 values of the states
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

x = TIMES

# Get the initial states and measurement basis options
initial_states_options = simulated.initial_gate.values
measurement_basis_options = simulated.final_gate.values

# Define the interactive function
def update_plot(initial_state, measurement_basis):
    # Get the corresponding outcome values
    title = f"{initial_state} - {measurement_basis}"
    ax.clear()

    for i, outcome in enumerate(simulated.outcome.values):


        y = (
            dataset.sel(initial_gate = initial_state, final_gate=measurement_basis, outcome=outcome).sampled_outcome.values
            / SAMPLES
        )
        ax.scatter(x, y, c = f"C{i}")

        # Get the corresponding simulated values
        y = simulated.sel(initial_gate=initial_state, final_gate=measurement_basis, outcome=outcome).values
        ax.plot(x, y, f"C{i}", label=outcome)
    
    ax.legend()
    ax.set_title(title)

# Create the dropdown widgets
initial_state_dropdown = Dropdown(options=initial_states_options, description="Initial State:")
measurement_basis_dropdown = Dropdown(options=measurement_basis_options, description="Measurement Basis:")

# Define the callback function for the dropdowns
def dropdown_callback(change):
    initial_state = initial_state_dropdown.value
    measurement_basis = measurement_basis_dropdown.value
    update_plot(initial_state, measurement_basis)

# Register the callback function to the dropdowns
initial_state_dropdown.observe(dropdown_callback, names="value")
measurement_basis_dropdown.observe(dropdown_callback, names="value")

# Display the dropdowns
display(initial_state_dropdown)
display(measurement_basis_dropdown)

# Initialize the plot
initial_state = initial_state_dropdown.value
measurement_basis = measurement_basis_dropdown.value
update_plot(initial_state, measurement_basis)

# Add legend and labels to the plot
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("Probability")
ax.set_title("Measurement Probabilities")
