import sys, os
import jax
import jax.numpy as jnp

from functools import partial

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..", "hamiltonian_learning")
)

# System parameters
NQUBITS = 2

# Experiment parameters
TIMES = jnp.linspace(0, 1000, 100)
INIT_STATES = ["X", "Y", "Z", "-Z"]
MEASUREMENT_BASIS = ["X", "Y", "Z"]

# Model parameters
PREPARATION_LOCALITY = 1
MEASUREMENT_LOCALITY = 1
HAMILTONIAN_LOCALITY = 2
LINDLAD_LOCALITY = 2

# Solver parameters
INITIAL_STEPSIZE = 1.0
MAX_STEPS = 1000
ODE_SOLVER = "Dopri5"
STEPSIZE_CONTROLLER = "adaptive"
ADJOINT = False
TOLERANCE = 1e-6

# Define the parameterization classes

# Something Like this  should be imported - all which contains classes that generates functions
# given the system and model parameters
from state_preparation import StatePreparation
from measurements import Measurements
from parameterization import Parameterization
from solvers import Solver

dynamics = Parameterization(
    NQUBITS,
    hamiltonian_locality=HAMILTONIAN_LOCALITY,
    lindblad_locality=LINDLAD_LOCALITY,
)

state_preparation = StatePreparation(
    NQUBITS, initial_state_method="temperature", states=["X", "Y", "Z", "-Z"]
)

measurements = Measurements(
    NQUBITS,
    basis=["X", "Y", "Z"],
    measurement_locality=MEASUREMENT_LOCALITY,
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
state_preparation_params = state_preparation.params
measurement_params = measurements.params
hamiltonian_params = dynamics.hamiltonian_params
lindbladian_params = dynamics.lindbladian_params


# Get the generators to convert the parameters to states or operators
generate_initial_states = state_preparation.get_initial_state_generator()
generate_hamiltonian = dynamics.get_hamiltonian_generator()
generate_lindbladian = dynamics.get_lindbladian_generator()
generate_measurement_operators = measurements.get_measurement_generator()
calculate_log_probability = measurements.get_log_probability_calculator()
evolve_states = solver.create_solver()


# Define the loss function
@partial(
    jax.jit,
    static_argnums=(
        0,
        1,
        2,
        3,
    ),
)
def loss_fn(
    hamiltonian_params,
    lindbladian_params,
    state_preparation_params,
    measurement_params,
    data,
):
    """
    Loss function to be minimized
    """
    initial_states = generate_initial_states(state_preparation_params)
    hamiltonian = generate_hamiltonian(hamiltonian_params)
    lindblad_operators = generate_lindbladian(lindbladian_params)

    # Evolve the states
    evolved_states = evolve_states(initial_states, hamiltonian, lindblad_operators)

    # Calculate the log probability
    log_probability = jnp.sum(
        calculate_log_probability(
            measurement_params,
            evolved_states,
            data,
        )
    )

    return -log_probability


# Define the gradient function
grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2, 3))


# Do the optimization loop
