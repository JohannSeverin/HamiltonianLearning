import sys, os

import jax
import jax.numpy as jnp
import xarray as xr

from functools import partial

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
jax.default_device(jax.devices("cpu")[0])

dataset = xr.load_dataset("dataset.zarr", engine="zarr")

data = dataset.sampled_outcome
data = jnp.array(data)

# System parameters
NQUBITS = dataset.attrs["NQUBITS"]

# Experiment parameters
TIMES = jnp.array(dataset.time)
INIT_STATES = ["Z", "X", "Y", "-Z"]
MEASUREMENT_BASIS = ["Z", "X", "Y"]
SAMPLES = dataset.attrs["SAMPLES"]

# Model parameters
# PREPARATION_LOCALITY = 1
# MEASUREMENT_LOCALITY = 1
HAMILTONIAN_LOCALITY = 2
LINDLAD_LOCALITY = 0

CLIP = 1e-3

# Solver parameters
INITIAL_STEPSIZE = 1.0
MAX_STEPS = 10000
ODE_SOLVER = "Dopri5"
STEPSIZE_CONTROLLER = "adaptive"
ADJOINT = False
TOLERANCE = 1e-6

# Optimizer Params
LOSS = "likelihood" # "squared_diffrence" OR "likelihood"

LEARNING_RATE = 1e-5
ITERATIONS = 1000
HAMILTONIAN_GUESS_ORDER = 1e-4
LINDLADIAN_GUESS_ORDER = 1e-5




# Define the parameterization classes

# Something Like this  should be imported - all which contains classes that generates functions
# given the system and model parameters
sys.path.append(
    "/home/archi1/projects/HamiltonianLearning/hamiltonian_learning_refactor"
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
    perfect_state_preparation=False,
)

measurements = Measurements(
    NQUBITS,
    basis=MEASUREMENT_BASIS,
    perfect_measurement=True,
    clip=CLIP,
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
# state_preparation_params = state_preparation.params
# measurement_params = measurements.params

# hamiltonian_params = dynamics.hamiltonian_params
lindbladian_params = dynamics.lindbladian_params
state_preparation_params = - 3e-3 * jnp.ones_like(state_preparation.state_preparation_params)

hamiltonian_params

# hamiltonian_params[1] = hamiltonian_params[1].at[0, -1].set(- 2 * jnp.pi * 450e-6/2)
# hamiltonian_params[1] = hamiltonian_params[1].at[1, -1].set(2 * jnp.pi * 150e-6/2)
# # dynamics.set_hamiltonian_params({1: dict(z)})
# dynamics.set_hamiltonian_params({2: dict(xx = 2 * jnp.pi * 140e-6, zz = 2 * jnp.pi *  100e-6)})
# lindbladian_params[1] = lindbladian_params[1].at[0, -1, -1].set(0.5 * 3.3333e-5 ** 0.5)
# lindbladian_params[1] = lindbladian_params[1].at[1, -1, -1].set(0.5 * 3.3333e-5 ** 0.5)
# lindbladian_params


# Get the generators to convert the parameters to states or operators
generate_initial_states = state_preparation.get_initial_state_generator()
generate_hamiltonian = dynamics.get_hamiltonian_generator()
generate_lindbladian = dynamics.get_jump_operator_generator()
calculate_log_likelihood = measurements.get_squared_difference_function() if LOSS == "squared_difference" else measurements.get_log_likelihood_function()
evolve_states = solver.create_solver()


# Define the loss function
@partial(
    jax.jit,
)
def loss_fn(
    params,
    data,
):
    """
    Loss function to be minimized
    """
    state_preparation_params, hamiltonian_params, lindbladian_params = params

    initial_states = generate_initial_states(state_preparation_params * 1e3)
    hamiltonian = generate_hamiltonian(hamiltonian_params)
    lindblad_operators = generate_lindbladian(lindbladian_params)

    # Evolve the states
    evolved_states = evolve_states(initial_states, hamiltonian, lindblad_operators)

    # # Calculate the log probability
    log_probability = calculate_log_likelihood(evolved_states, data, samples=SAMPLES)

    return log_probability


loss_and_grad = jax.value_and_grad(loss_fn, argnums=(0))

# Fit the model

from optax import adam, apply_updates

# Init
params = (state_preparation_params, hamiltonian_params, lindbladian_params)
opt = adam(LEARNING_RATE)
state = opt.init(params)

# Update
for i in range(ITERATIONS):
    loss, grads = loss_and_grad(params, data)
    updates, state = opt.update(grads, state)
    params = apply_updates(params, updates)

    print(f"Iteration {i:03d} - Loss: {loss:.3e}")



# Test the functions
state_preparation_params, hamiltonian_params, lindbladian_params = params
states = generate_initial_states(state_preparation_params  * 1e3)
hamiltonian = generate_hamiltonian(hamiltonian_params)
lindblad_operators = generate_lindbladian(lindbladian_params)
evolved_states = evolve_states(states, hamiltonian, lindblad_operators)

measurements.generate_samples(evolved_states, SAMPLES)

probs = measurements.calculate_measurement_probabilities(evolved_states)

simulated = dataset.measurement_probabilities.copy()
simulated.values = probs

loss_fn(params, data)

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
