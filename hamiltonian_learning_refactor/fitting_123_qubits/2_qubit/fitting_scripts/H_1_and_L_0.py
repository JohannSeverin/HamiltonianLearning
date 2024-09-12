import xarray
import sys, os
import jax
import jax.numpy as jnp

from functools import partial


dataset = xarray.open_dataset("../dataset.zarr", engine="zarr")
data = dataset.sampled_outcome.values

# System parameters
NQUBITS = 2

# Experiment parameters
TIMES = dataset.time.values
INIT_STATES = ["X", "Y", "Z", "-Z"]
MEASUREMENT_BASIS = ["X", "Y", "Z"]
SAMPLES = dataset.attrs["SAMPLES"]

# Model parameters
HAMILTONIAN_LOCALITY = 1
LINDLAD_LOCALITY = 0

# Solver parameters
INITIAL_STEPSIZE = 1.0
MAX_STEPS = 100000
ODE_SOLVER = "Dopri5"
STEPSIZE_CONTROLLER = "adaptive"
ADJOINT = False
TOLERANCE = 1e-6

# Optimzier parameters
LEARNING_RATE_SCAN = 1e-4
LEARNING_RATE_FINE = 5e-5
ITERATIONS_SME = 250
ITERATIONS_MLE = 750

loss = "squared_difference"


# Define the parameterization classes
sys.path.append("/root/projects/HamiltonianLearning/hamiltonian_learning_refactor")
from hamiltonian_learning import (
    Measurements,
    Parameterization,
    Solver,
    StatePreparation,
)

# Define the parameterization classes

dynamics = Parameterization(
    NQUBITS,
    hamiltonian_locality=HAMILTONIAN_LOCALITY,
    lindblad_locality=LINDLAD_LOCALITY,
    hamiltonian_amplitudes=[1e-4, 1e-5],
    lindblad_amplitudes=[1e-3],
    seed = 1
)

state_preparation = StatePreparation(
    NQUBITS, perfect_state_preparation=True, initial_states=["Z", "X", "Y", "-Z"]
)

measurements = Measurements(
    NQUBITS,
    samples=SAMPLES,
    basis=["Z", "X", "Y"],
    perfect_measurement=True,
    loss=loss,
    clip=1e-5,
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
state_preparation_params = state_preparation.state_preparation_params
measurement_params = measurements.params
hamiltonian_params = dynamics.hamiltonian_params
lindbladian_params = dynamics.lindbladian_params

# Load guesses
import pickle
with open("parameters_H2L1.pickle", "rb") as f:
    parameters = pickle.load(f)

hamiltonian_params[1] = parameters["hamiltonian_params"][1]
# lindbladian_params[1] = parameters["lindbladian_params"][1]


# Get the generators to convert the parameters to states or operators
generate_initial_states = state_preparation.get_initial_state_generator()
generate_hamiltonian = dynamics.get_hamiltonian_generator()
generate_lindbladian = dynamics.get_jump_operator_generator()
calculate_log_likelihood = measurements.get_loss_fn()
evolve_states = solver.create_solver()


# Define the loss function
def loss(params):
    state_preparation_params, hamiltonian_params, lindbladian_params = params

    initial_states = generate_initial_states(state_preparation_params)
    hamiltonian = generate_hamiltonian(hamiltonian_params)
    lindbladian = generate_lindbladian(lindbladian_params)

    states = evolve_states(initial_states, hamiltonian, lindbladian)

    return calculate_log_likelihood(states, data)


# Define the gradient of the loss function
value_and_grad = jax.jit(jax.value_and_grad(loss))

# Setup the optimizer
from optax import adam, apply_updates

params = (state_preparation_params, hamiltonian_params, lindbladian_params)
opt = adam(learning_rate=LEARNING_RATE_SCAN)
state = opt.init(params)

# Update
for i in range(ITERATIONS_SME):
    loss, grads = value_and_grad(params)
    updates, state = opt.update(grads, state)
    params = apply_updates(params, updates)

    print(f"Iteration {i:03d} - Loss: {loss:.3e}")


# Change to multinomial
measurements.loss = "multinomial"
calculate_log_likelihood = measurements.get_loss_fn()



# Define the loss function
def loss(params):
    state_preparation_params, hamiltonian_params, lindbladian_params = params

    initial_states = generate_initial_states(state_preparation_params)
    hamiltonian = generate_hamiltonian(hamiltonian_params)
    lindbladian = generate_lindbladian(lindbladian_params)

    states = evolve_states(initial_states, hamiltonian, lindbladian)

    return - calculate_log_likelihood(states, data)


# Define the gradient of the loss function
value_and_grad = jax.jit(jax.value_and_grad(loss))

# Setup the optimizer
from optax import adam, apply_updates

opt = adam(learning_rate=LEARNING_RATE_FINE)
state = opt.init(params)

# Update
for i in range(ITERATIONS_MLE):
    loss, grads = value_and_grad(params)
    updates, state = opt.update(grads, state)
    params = apply_updates(params, updates)

    print(f"Iteration {i:03d} - Loss: {loss:.3e}")


# Evolve states using the fitted parameters
state_preparation_params, hamiltonian_params, lindbladian_params = params
initial_states = generate_initial_states(state_preparation_params)
hamiltonian = generate_hamiltonian(hamiltonian_params)
lindbladian = generate_lindbladian(lindbladian_params)

states = evolve_states(initial_states, hamiltonian, lindbladian)

probs = measurements.calculate_measurement_probabilities(states)

# Plot the results along with the data points using ipywidgets interactive
simulated = dataset.measurement_probabilities.copy()
simulated.values = probs

# Save the dataset 
simulated.to_zarr("simulated_H1L0.zarr", mode="w")

# Save the parameters from the fit
import numpy as np
parameters = dict(
    state_preparation_params=state_preparation_params,
    hamiltonian_params=hamiltonian_params,
    lindbladian_params=lindbladian_params,
    times=TIMES,
    final_loss=loss,
)

import pickle
with open("parameters_H1L0.pickle", "wb") as f:
    pickle.dump(parameters, f)

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
    print(initial_state, measurement_basis)
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



# # Pvalue test 
# from tensorflow_probability.substrates.jax import distributions as tfd

# nllh = tfd.Multinomial(total_count = SAMPLES, probs = probs).log_prob(data).sum()

# # Average nllh 
# samples = tfd.Multinomial(total_count = SAMPLES, probs = probs).sample(1000)
# nllh_sampled = tfd.Multinomial(total_count = SAMPLES, probs = probs).log_prob(samples)