# Going to rotating frame
import jax
import jax.numpy as jnp
import xarray as xr
from diffrax import Dopri5, ODETerm, PIDController, diffeqsolve, SaveAt
from matplotlib import style
import matplotlib.pyplot as plt
from itertools import product
from utils import rotating_unitary
from utils import get_unitary_term, get_dissipation_term
from utils import sigma_z
from utils import tensor, identity
from utils import destroy, create
from utils import sigma_x, sigma_y, sigma_z, identity
from utils import get_unitary_term, get_dissipation_term
from utils import rotating_unitary

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)

style.use(
    "/mnt/c/Users/msk377/OneDrive - University of Copenhagen/Desktop/styles/presentation_style.mplstyle"
)


# Simulation parameters
DURATION = 200
STORE_EVERY = 4
INITIAL_STEPSIZE = 0.1
MAX_SOLVER_STEPS = 10000
SIMULATED_SINGLE_SHOTS = 100

# Lindbladian parameters
QUBIT_ENERGIES = [5.0, 5.2, 5.4]
T1 = 10000
T2 = 5000
UNWANTED_COUPLING = 0.1
FRACTION_EXCITATION_RATE = 0.1

lindbladian_parameters = dict(
    qubit_energies=jnp.array(QUBIT_ENERGIES),
    T1=T1,
    T2=T2,
    unwanted_coupling=UNWANTED_COUPLING,
    fraction_excitation_rate=FRACTION_EXCITATION_RATE,
)

# Hamiltonian terms to consider
id = identity(2)
hamiltonian_operators = dict(
    z_1=tensor(sigma_z(), id, id),
    z_2=tensor(id, sigma_z(), id),
    z_3=tensor(id, id, sigma_z()),
    x_1=tensor(sigma_x(), id, id),
    x_2=tensor(id, sigma_x(), id),
    x_3=tensor(id, id, sigma_x()),
    y_1=tensor(sigma_y(), id, id),
    y_2=tensor(id, sigma_y(), id),
    y_3=tensor(id, id, sigma_y()),
    zz_12=tensor(sigma_z(), sigma_z(), id),
    zz_23=tensor(id, sigma_z(), sigma_z()),
)

hamiltonian_operators_array = jnp.array(
    list(hamiltonian_operators.values()), dtype=jnp.complex128
)

# hamiltonian_coefficients = dict(
#     z_1=0.0,
#     z_2=0.0,
#     z_3=0.0,
#     x_1=0.0,
#     x_2=0.0,
#     x_3=0.0,
#     y_1=0.0,
#     y_2=0.0,
#     y_3=0.0,
#     zz_12=0.1,
#     zz_23=0.1,
# )
hamiltonian_coefficients = dict(
    z_1=0.0,
    z_2=0.0,
    z_3=0.0,
    x_1=0.0,
    x_2=0.0,
    x_3=0.0,
    y_1=0.0,
    y_2=0.0,
    y_3=0.0,
    zz_12=2 * jnp.pi * 0.1,
    zz_23=2 * jnp.pi * 0.1,
)


hamiltonian_coefficients_array = jnp.array(
    list(hamiltonian_coefficients.values()), dtype=jnp.float64
)

# Lindbladian terms to consider
dissipators = dict(
    down_1=tensor(destroy(), id, id),
    down_2=tensor(id, destroy(), id),
    down_3=tensor(id, id, destroy()),
    up_1=tensor(create(), id, id),
    up_2=tensor(id, create(), id),
    up_3=tensor(id, id, create()),
    pure_dephasing_1=tensor(sigma_z(), id, id),
    pure_dephasing_2=tensor(id, sigma_z(), id),
    pure_dephasing_3=tensor(id, id, sigma_z()),
)

dissipators_array = jnp.array(list(dissipators.values()), dtype=jnp.complex128)

dissipators_coefficients = dict(
    down_1=0.00948683,
    down_2=0.00948683,
    down_3=0.00948683,
    up_1=0.00316228,
    up_2=0.00316228,
    up_3=0.00316228,
    pure_dephasing_1=0.01224745,
    pure_dephasing_2=0.01224745,
    pure_dephasing_3=0.01224745,
)
dissipators_coefficients = dict(
    down_1=0.1,
    down_2=0.1,
    down_3=0.1,
    up_1=0.01,
    up_2=0.01,
    up_3=0.01,
    pure_dephasing_1=0.1,
    pure_dephasing_2=0.1,
    pure_dephasing_3=0.1,
)

dissipators_coefficients_array = jnp.array(
    list(dissipators_coefficients.values()), dtype=jnp.float64
)

# Params
params = jnp.concatenate(
    (hamiltonian_coefficients_array, dissipators_coefficients_array), axis=0
)


# This is the ordinary differential equation (ODE) that we want to solve
def Lindbladian(t, rho, args):
    """
    Calculate the Lindbladian operator for a given time `t`, density matrix `rho`, and additional arguments `args`.

    Parameters:
        t (float): The time at which to evaluate the Lindbladian.
        rho (numpy.ndarray): The density matrix.
        args (dict): Additional arguments.

    Returns:
        numpy.ndarray: The Lindbladian operator.

    """
    drho = get_unitary_term(rho, args["Hamiltonian"])
    for i in range(len(dissipators)):
        drho += get_dissipation_term(rho, args["Dissipators"][i])
    return drho


# Time with data
tlist = jnp.arange(0, DURATION + STORE_EVERY, STORE_EVERY)

# Setup states
from utils import *

pauli_letters = ["x", "y", "z"]
pauli_states = dict(x=pauli_state_x_ket(), y=pauli_state_y_ket(), z=basis_ket(0, 2))
pauli_states_dm = dict(x=pauli_state_x_dm(), y=pauli_state_y_dm(), z=basis_dm(0, 2))
pauli_operators = dict(x=sigma_x(), y=sigma_y(), z=sigma_z(), I=id)

# Setup the init states and the operators to measure
pauli_strings = list(i + j + k for i, j, k in product(pauli_letters, repeat=3))

# Initial States
initial_state = jnp.array(
    list(
        tensor(*[pauli_states_dm[state] for state in state_string])
        for state_string in pauli_strings
    )
)

# Operators to measure
states_to_measure = jnp.array(
    list(
        tensor(*[pauli_states[state] for state in state_string])
        for state_string in pauli_strings
    )
)


# Setup of the ODE solver
from diffrax import Tsit5, DirectAdjoint

term = ODETerm(Lindbladian)
solver = Dopri5()
pid_controller = PIDController(1e-5, 1e-5)
saveat = SaveAt(ts=tlist)
adjoint = DirectAdjoint()


def evolve_states(params):
    """
    Calculate the time derivative of the states for a given time `t`, parameters `params`, and lindbladian parameters `lindbladian_parameters`.

    Parameters:
        t (float): The time at which to evaluate the Lindbladian.
        params (dict): The parameters for the Hamiltonian.
        lindbladian_parameters (dict): The parameters for the Lindbladian.

    Returns:
        dict: The time derivative of the states.

    """
    hamiltonian_coefficients = params[: len(hamiltonian_operators)]
    Hamiltonian = jnp.sum(
        hamiltonian_coefficients[:, None, None] * hamiltonian_operators_array,
        axis=0,
    )

    dissipators_coefficients = params[len(hamiltonian_operators) :]
    Dissipators = dissipators_coefficients[:, None, None] * dissipators_array

    return diffeqsolve(
        terms=term,
        solver=solver,
        y0=initial_state,
        t0=0,
        t1=DURATION,
        stepsize_controller=pid_controller,
        saveat=saveat,
        dt0=INITIAL_STEPSIZE,
        max_steps=MAX_SOLVER_STEPS,
        args=dict(Hamiltonian=Hamiltonian, Dissipators=Dissipators),
        adjoint=adjoint,
    )


# Jit the function
from jax import jit

evolve_states = jit(evolve_states)

trial_results = evolve_states(params)

# Load the data from the "real world experiment"
import xarray as xr

data = xr.open_dataarray("data.nc")
data_jax = jnp.array(data, dtype=jnp.int32)


# Probability of measuring the state given the right parameters
from jax.scipy.stats import binom


def results_to_probs(states, clip_epsilon=0.00001):
    """
    Calculate the probabilities for the results of the states.

    Parameters:
        results (dict): The results of the states.

    Returns:
        numpy.ndarray: The probabilities for the results of the states.

    """
    # Probability of measuring the states
    probs = jnp.einsum(
        "ni, tmij, nj -> tnm",
        states_to_measure,
        states,
        jnp.conj(states_to_measure),
    )
    probs = probs.real
    return jnp.clip(probs, clip_epsilon, 1 - clip_epsilon)


def LLH(data, probs):
    """
    Calculate the log likelihood for the data and the probabilities.

    Parameters:
        data (numpy.ndarray): The data from the "real world experiment".
        probs (numpy.ndarray): The probabilities for the results of the states.

    Returns:
        numpy.ndarray: The log likelihood for the data and the probabilities.

    """
    return -jax.scipy.stats.binom.logpmf(data, n=SIMULATED_SINGLE_SHOTS, p=probs).sum()


def evolve_and_calculate_LLH(params):
    """
    Calculate the log likelihood for the data and the probabilities for a given set of parameters.

    Parameters:
        params (dict): The parameters for the Hamiltonian.

    Returns:
        numpy.ndarray: The log likelihood for the data and the probabilities.

    """
    results = evolve_states(params).ys
    return LLH(data_jax, results_to_probs(results))


# Optimize the parameters
from jax import value_and_grad

evolve_and_calculate_LLH_and_grad = value_and_grad(evolve_and_calculate_LLH)

LLH_grad = value_and_grad(LLH, argnums=1)

from optax import adam, apply_updates


# Optimization parameters
LEARNING_RATE = 1e-3
EPOCHS = 50


# Initialize the optimizer

params = jnp.concat(
    (hamiltonian_coefficients_array, dissipators_coefficients_array), axis=0
)
opt = adam(LEARNING_RATE)
opt_state = opt.init(params)

for epoch in range(EPOCHS):
    value, grads = evolve_and_calculate_LLH_and_grad(params)
    updates, opt_state = opt.update(grads, opt_state)
    params = apply_updates(params, updates)

    print(f"Epoch {epoch}, LLH: {value}")


# Uncertainty quantification
print("calculating hessian, this may take a while")
hessian = jax.hessian(evolve_and_calculate_LLH)(params)


# Check how correct this was
# unpack
hamiltonian_coefficients_final = params[: len(hamiltonian_operators)]
dissipators_coefficients_final = params[len(hamiltonian_operators) :]

hamiltonian_coefficients = {
    key: value
    for key, value in zip(hamiltonian_operators.keys(), hamiltonian_coefficients)
}

dissipators_coefficients = {
    key: value for key, value in zip(dissipators.keys(), dissipators_coefficients)
}


inverse_hessian = jnp.linalg.inv(hessian)

errors = jnp.sqrt(jnp.diag(inverse_hessian))

# Check performance
# Correct params
correct_hamiltonian_coefficients = dict(
    z_1=0.0,
    z_2=0.0,
    z_3=0.0,
    x_1=0.0,
    x_2=0.0,
    x_3=0.0,
    y_1=0.0,
    y_2=0.0,
    y_3=0.0,
    zz_12=2 * jnp.pi * 0.1,
    zz_23=2 * jnp.pi * 0.1,
)

correct_dissipators_coefficients = dict(
    down_1=0.00948683,
    down_2=0.00948683,
    down_3=0.00948683,
    up_1=0.00316228,
    up_2=0.00316228,
    up_3=0.00316228,
    pure_dephasing_1=0.01224745,
    pure_dephasing_2=0.01224745,
    pure_dephasing_3=0.01224745,
)


# Plot the results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Hamiltonian params
ax[0].plot(
    hamiltonian_coefficients_final,
    label="Final",
    marker=".",
    linestyle="none",
    color="black",
)
ax[0].errorbar(
    range(len(hamiltonian_coefficients_final)),
    hamiltonian_coefficients_final,
    errors[: len(hamiltonian_coefficients_final)],
    ls="none",
)
ax[0].plot(
    list(correct_hamiltonian_coefficients.values()),
    label="Correct",
    marker="o",
    linestyle="none",
    color="red",
)
ax[0].set_xticks(range(len(hamiltonian_coefficients_final)))
ax[0].set_xticklabels(hamiltonian_coefficients.keys(), rotation=90)

# Dissipator params
ax[1].plot(
    dissipators_coefficients_final,
    label="Final",
    marker="o",
    linestyle="none",
    color="black",
)
ax[1].errorbar(
    range(len(dissipators_coefficients_final)),
    dissipators_coefficients_final,
    errors[len(hamiltonian_coefficients_final) :],
    ls="none",
)
ax[1].plot(
    list(correct_dissipators_coefficients.values()),
    label="Correct",
    marker="o",
    linestyle="none",
    color="red",
)
ax[1].set_xticks(range(len(dissipators_coefficients_final)))
ax[1].set_xticklabels(dissipators_coefficients.keys(), rotation=90)

ax[1].legend()
