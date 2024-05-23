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
style.use(
    "/mnt/c/Users/msk377/OneDrive - University of Copenhagen/Desktop/styles/presentation_style.mplstyle"
)


# Simulation parameters
DURATION = 200
STORE_EVERY = 4
INITIAL_STEPSIZE = 0.1
MAX_SOLVER_STEPS = 100000
SIMULATED_SINGLE_SHOTS = 100

# Lindbladian parameters
QUBIT_ENERGIES = [5.0, 5.2, 5.4]
T1 = 10000
T2 = 5000
UNWANTED_COUPLING = 0.1
FRACTION_EXCITATION_RATE = 0.1

# Times to store the data
tlist = jnp.arange(0, DURATION + STORE_EVERY, STORE_EVERY)

# Setup operators
single_qubit_hamiltonians = [energy * sigma_z() for energy in QUBIT_ENERGIES]
id = identity(2)

# This is the known part of the Hamiltonian
H0 = (
    2
    * jnp.pi
    * (
        tensor(single_qubit_hamiltonians[0], id, id)
        + tensor(id, single_qubit_hamiltonians[1], id)
        + tensor(id, id, single_qubit_hamiltonians[2])
    )
)

H0 = jnp.zeros_like(H0)

# Setup the interaction term
H1 = jnp.array(
    UNWANTED_COUPLING
    * 2
    * jnp.pi
    * (tensor(sigma_z(), sigma_z(), id) + tensor(id, sigma_z(), sigma_z()))
)

# Setup dissipators for T1
rate = 1 / T1 * (1 - FRACTION_EXCITATION_RATE)
dissipators = [jnp.sqrt(rate) * destroy() for _ in QUBIT_ENERGIES]
Ls = [
    tensor(dissipators[0], id, id),
    tensor(id, dissipators[1], id),
    tensor(id, id, dissipators[2]),
]

rate = 1 / T1 * FRACTION_EXCITATION_RATE
dissipators = [jnp.sqrt(rate) * create() for _ in QUBIT_ENERGIES]
Ls += [
    tensor(dissipators[0], id, id),
    tensor(id, dissipators[1], id),
    tensor(id, id, dissipators[2]),
]

# Setup dissipators for T2
pure_dephasing_rate = 1 / T2 - 1 / T1 / 2

dissipators = [jnp.sqrt(pure_dephasing_rate) * sigma_z() for _ in QUBIT_ENERGIES]

Ls += [
    tensor(dissipators[0], id, id),
    tensor(id, dissipators[1], id),
    tensor(id, id, dissipators[2]),
]

### List of Pauli Strings
pauli_letters = ["x", "y", "z"]
from utils import (
    pauli_state_x_dm,
    pauli_state_y_dm,
    basis_dm,
    basis_ket,
    pauli_state_x_ket,
    pauli_state_y_ket,
)

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

# Projector
projections = jnp.einsum(
    "ni, mij, nj -> nm", states_to_measure, initial_state, jnp.conj(states_to_measure)
)

data = xr.DataArray(
    jnp.nan,
    coords=[tlist, pauli_strings, pauli_strings],
    dims=["time", "initial_states", "final_states"],
)


# Setup of the ODE
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
    drho = get_unitary_term(rho, H0 + H1)
    for L in Ls:
        drho += get_dissipation_term(rho, L)
    return drho


# Setup of the ODE solver
term = ODETerm(Lindbladian)
solver = Dopri5()
pid_controller = PIDController(1e-5, 1e-5)
saveat = SaveAt(ts=tlist)

# Solve the ODE
results = diffeqsolve(
    terms=term,
    solver=solver,
    y0=initial_state,
    t0=0,
    t1=DURATION,
    stepsize_controller=pid_controller,
    saveat=saveat,
    dt0=INITIAL_STEPSIZE,
    max_steps=MAX_SOLVER_STEPS,
)

# Going to rotating frame
Us = rotating_unitary(tlist, H0)
Us_dag = jnp.conj(Us).transpose((0, 2, 1))
rotating_states = jnp.einsum("tij, tnjk, tkl -> tnil", Us_dag, results.ys, Us)

# Probability of measuring the states
probs = jnp.einsum(
    "ni, tmij, nj -> tnm",
    states_to_measure,
    rotating_states,
    jnp.conj(states_to_measure),
)
probs = jnp.clip(probs, 0, 1)

# Samples frm the probabilities
key = jax.random.PRNGKey(0)
samples = jax.random.binomial(key, n=SIMULATED_SINGLE_SHOTS, p=probs.real)
plt.imshow(samples[-1].real)

# Save the results as probabilities in data
data.loc[:, :, :] = samples

data.to_netcdf("data.nc")

# Plot the results for some tests
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

y = data.sel(initial_states="xxx", final_states="zzz").values
yerr = jnp.sqrt(y * (1 - y / SIMULATED_SINGLE_SHOTS))

ax.errorbar(
    tlist,
    y,
    yerr=yerr,
    fmt="o",
    label="xxx",
)


from jax.scipy.stats import binom


def results_to_probs(states, clip_epsilon=0.0001):
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


results_to_probs(results.ys)

LLH(samples, results_to_probs(results.ys))
