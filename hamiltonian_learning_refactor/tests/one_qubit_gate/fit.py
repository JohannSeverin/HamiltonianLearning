import sys, os

import jax
import jax.numpy as jnp
import xarray as xr

from functools import partial

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
PERFECT_PREPARATION = True
# MEASUREMENT_LOCALITY = 1
HAMILTONIAN_LOCALITY = 1
LINDLAD_LOCALITY = 0

CLIP = 3e-2 # Clip harder if no dissipation or SPAM

# Solver parameters
INITIAL_STEPSIZE = 1.0
MAX_STEPS = 10000
ODE_SOLVER = "Dopri5"
STEPSIZE_CONTROLLER = "adaptive"
ADJOINT = False
TOLERANCE = 1e-6

# Optimizer Params
LOSS = "squared_difference" # "squared_difference" OR "multinomial"
USE_WEIGHTS = True # If squared differences this determines if we should weigh the points according to the estimated std

LEARNING_RATE = 1e-4
ITERATIONS = 20
HAMILTONIAN_GUESS_ORDER = {1: 1e-4}
LINDLADIAN_GUESS_ORDER ={1: 1e-5}


# Define the parameterization classes

# Something Like this  should be imported - all which contains classes that generates functions
# given the system and model parameters
sys.path.append(
    "/home/archi1/projects/HamiltonianLearning/hamiltonian_learning_refactor"
)
sys.path.append(
    "/root/projects/HamiltonianLearning/hamiltonian_learning_refactor/"
)

from hamiltonian_learning import (
    Measurements,
    StatePreparation,
    Solver,
    TimeDependentSolver,
    Parameterization,
    InterpolatedParameterization,
)

dynamics = InterpolatedParameterization(
    NQUBITS,
    times = TIMES,
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
    samples=SAMPLES,
    perfect_measurement=True,
    clip=CLIP,
    loss=LOSS,
)

# Define the solver for time dynamics
solver = TimeDependentSolver(
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

hamiltonian_params = dynamics.hamiltonian_params
lindbladian_params = dynamics.lindbladian_params
state_preparation_params = - 3e-3 * jnp.ones_like(state_preparation.state_preparation_params)

# hamiltonian_params

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
calculate_log_likelihood = measurements.get_loss_fn()
evolve_states = solver.create_solver(TIMES)


# Define the loss function
@jax.jit
def loss_fn(
    params,
):  
    """
    Loss function to be minimized
    """
    state_preparation_params, hamiltonian_params, lindbladian_params = params

    initial_states = generate_initial_states(3e3 * state_preparation_params)
    hamiltonian = generate_hamiltonian(hamiltonian_params)
    lindblad_operators = generate_lindbladian(lindbladian_params)

    # Evolve the states
    evolved_states = evolve_states(initial_states, TIMES, hamiltonian, lindblad_operators)

    # # Calculate the log probability
    log_probability = calculate_log_likelihood(evolved_states, data)

    return log_probability / data.size


loss_and_grad = jax.value_and_grad(loss_fn, argnums=(0))

# Fit the model

from optax import adam, apply_updates
from optax import lbfgs, value_and_grad_from_state
from optax import adaptive_grad_clip

loss_and_grad_from_state = value_and_grad_from_state(loss_fn)

# Init
params = (state_preparation_params, hamiltonian_params, lindbladian_params)
opt = lbfgs(LEARNING_RATE)
clipper = adaptive_grad_clip(10)
clip_state = clipper.init(params)
state = opt.init(params)

# Update
for i in range(ITERATIONS):
    loss, grads = loss_and_grad(params) #, data)
    updates, state = opt.update(grads, state, params, value = loss, grad = grads, value_fn = loss_fn, has_aux = True)
    params = apply_updates(params, updates)

    print(f"Iteration {i:03d} - Loss: {loss:.3e}")



# Test the functions
state_preparation_params, hamiltonian_params, lindbladian_params = params
initial_states = generate_initial_states(3e3 * state_preparation_params)#.reshape(-1, 4, 4)
hamiltonian = generate_hamiltonian(hamiltonian_params)
lindblad_operators = generate_lindbladian(lindbladian_params)
evolved_states = evolve_states(initial_states, TIMES, hamiltonian, lindblad_operators)

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

x = TIMES


# Show the hamiltonian parameters as function of time 
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

for i, key in enumerate("xyz"):
    y = hamiltonian_params[1][:, 0, i]
    ax.plot(x, y, label=key, color = "C" + str(i))
    ax.plot(x, y, "o", label=key, color = "C" + str(i))

ax.set(xlabel="Time", ylabel="Amplitude")
ax.set_title("Hamiltonian Parameters")


ax.legend()


# We plot the z1, z2, and z1z2 values of the states
fig, ax = plt.subplots(1, 1, figsize=(8, 5))


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

