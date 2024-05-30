import sys

sys.path.append("..")
from utils import *

import jax.numpy as jnp

from diffrax import Dopri5, SaveAt, diffeqsolve, DirectAdjoint, ODETerm, PIDController

# Experiment name
EXPERIMENT_NAME = "one_qubit_test_lindbladian"

# Experiement parameters
SAMPLES = 100

# Define the Hamiltonian parameters
HAMILTONIAN_PARAMETERS = 2 * jnp.pi * jnp.array([0.0, 0.00, 0.0, 0.0])  # [Id, x, y, z]

# Define the Hamiltonian
DURATION = 200
STORE_EVERY = 4
INITIAL_STEPSIZE = 0.1
MAX_SOLVER_STEPS = 10000
SIMULATED_SINGLE_SHOTS = 100
INIT_TIMESTEP = 0.1

# Define the Lindbladian parameters
T1 = 100
T2 = 100


# Learning parameters
LEARNING_RATE = 0.01
EPOCHS = 1000

dephasing_rate = 1 / T2 - 1 / (2 * T1)

# Setup the lindbladian jump operators
lindblad_jumps = jnp.array(
    [
        1 / jnp.sqrt(T1) * destroy(),
        1 / jnp.sqrt(T1) * create(),
        jnp.sqrt(dephasing_rate) * sigma_z(),
    ]
)

# Define ways of going frm the pauli matrix representation of the lindbladian to the jump operators
pauli_gates = jnp.array([sigma_x(), sigma_y(), sigma_z()])


def jump_operators_to_pauli_matrix(jump_operators):
    """
    Converts a set of jump operators to a Lindbladian matrix in the Pauli basis.

    Args:
        jump_operators (ndarray): A set of jump operators represented as a 3-dimensional array.

    Returns:
        ndarray: The Lindbladian matrix in the Pauli basis.

    """
    pauli_basis = jnp.einsum("ijk, pjk -> ip", jump_operators, pauli_gates)
    lindbladian_matrix = jnp.einsum("ik, il -> kl", pauli_basis, pauli_basis.conj())
    return lindbladian_matrix


def pauli_matrix_to_jump_operators(pauli_matrix):
    """
    Converts a Pauli matrix to jump operators.

    Args:
        pauli_matrix: The Pauli matrix to be converted.

    Returns:
        The jump operators corresponding to the given Pauli matrix.
    """
    return jnp.einsum("ij, jkl -> ikl", pauli_matrix, pauli_gates)


# Find guess in cholesky form from the initial guesses
pauli_matrix = jump_operators_to_pauli_matrix(lindblad_jumps)
reduced_jump_opreators = pauli_matrix_to_jump_operators(pauli_matrix)
cholesky = jax.scipy.linalg.cholesky(pauli_matrix + 1e-10, lower=False)
cholesky = cholesky.at[jnp.isclose(cholesky, 0, atol=1e-9, rtol=1e-9)].set(0)

lindbladian_params = jnp.concat(
    [cholesky[jnp.tril_indices(3)].real, cholesky[jnp.tril_indices(3, k=-1)].imag]
)


def system_dynamics(t, rho, args):
    """
    Differential equation governing the system
    """
    drho = get_unitary_term(rho, args["hamiltonian"])

    for i in range(3):
        drho += get_dissipation_term(rho, args["jump_operators"][i])

    return drho


# SETUP LINDBLADIAN
term = ODETerm(system_dynamics)
solver = Dopri5()
saveat = SaveAt(ts=jnp.arange(0, DURATION + STORE_EVERY, STORE_EVERY))
pid_controller = PIDController(1e-5, 1e-5)
adjoint = DirectAdjoint()


def evolve_system(args):
    """
    Evolve the system using the Runge-Kutta method
    Need following in args
    - init_states
    - hamiltonian_params
    """
    init_states = args["init_states"]
    hamiltonian_params = args["hamiltonian_params"]
    lindbladian_params = args["lindbladian_params"]

    hamiltonian = (
        identity()
        + hamiltonian_params[1] * sigma_x()
        + hamiltonian_params[2] * sigma_y()
        + hamiltonian_params[3] * sigma_z()
    )

    lindbladian_matrix = jnp.zeros((3, 3), dtype=jnp.complex128)
    lindbladian_matrix = lindbladian_matrix.at[jnp.tril_indices(3)].set(
        lindbladian_params[:6]
    )
    lindbladian_matrix = lindbladian_matrix.at[jnp.tril_indices(3, k=-1)].add(
        1j * lindbladian_params[6:]
    )

    lindbladian_pauli_matrix = lindbladian_matrix @ lindbladian_matrix.T.conj()

    print(lindbladian_pauli_matrix, lindbladian_pauli_matrix.shape)

    jump_operators = pauli_matrix_to_jump_operators(lindbladian_pauli_matrix)

    print(jump_operators, jump_operators.shape)

    return diffeqsolve(
        terms=term,
        solver=solver,
        y0=init_states,
        t0=0,
        t1=DURATION,
        stepsize_controller=pid_controller,
        saveat=saveat,
        dt0=INITIAL_STEPSIZE,
        max_steps=MAX_SOLVER_STEPS,
        args=dict(hamiltonian=hamiltonian, jump_operators=jump_operators),
        adjoint=adjoint,
    )


from jax.scipy.linalg import expm

# pauli_gates = [identity(), sigma_x(), sigma_y(), sigma_z()]

pauli_transforms = jnp.array(
    [expm(1j * gate * jnp.pi / 4) for gate in [-sigma_y(), +sigma_x()]] + [identity()]
)

# Define ground state
ground_state = basis_dm(0, 2)


# Initial States
initial_states = jnp.einsum(
    "...ij, jk, ...kl -> ...il",
    pauli_transforms,
    ground_state,
    pauli_transforms.conj().transpose((0, 2, 1)),
)

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def measurement(args):
    """
    Measure the system a certain number of times at different times
    """
    init_states = args["init_states"]
    hamiltonian_params = args["hamiltonian_params"]
    lindbladian_params = args["lindbladian_params"]

    # Evolve the system
    evolved_states = evolve_system(
        dict(
            init_states=init_states,
            hamiltonian_params=hamiltonian_params,
            lindbladian_params=lindbladian_params,
        )
    )

    # Measurement basis
    measurement_runs = jnp.einsum(
        "pij, tqjk, pkl -> tqpil",
        pauli_transforms.conj().transpose((0, 2, 1)),
        evolved_states.ys,
        pauli_transforms,
    )

    prob_1 = measurement_runs[..., 1, 1].real

    key = jax.random.PRNGKey(0)
    return tfd.Binomial(total_count=SAMPLES, probs=prob_1).sample(seed=key)


data = measurement(
    dict(
        init_states=initial_states,
        hamiltonian_params=HAMILTONIAN_PARAMETERS,
        lindbladian_params=lindbladian_params,
    )
)

import xarray as xr

data_xr = xr.DataArray(
    data,
    dims=("time", "initial_state", "measurement_basis"),
    coords=dict(
        time=jnp.arange(0, DURATION + STORE_EVERY, STORE_EVERY),
        initial_state=["x", "y", "z"],
        measurement_basis=["x", "y", "z"],
    ),
)

# Save data
data_xr.to_netcdf(f"{EXPERIMENT_NAME}.nc")


# def NLLH(probs, data):
#     """
#     Log likelihood function
#     """
#     probs = jnp.clip(probs, 1e-6, 1 - 1e-6)
#     return -jnp.sum(tfd.Binomial(total_count=SAMPLES, probs=probs).log_prob(data))


# def loss_fn(params, data):
#     """
#     Loss function
#     """
#     evolved_states = evolve_system(
#         dict(init_states=initial_states, hamiltonian_params=params)
#     )

#     measured_states = jnp.einsum(
#         "pij, tqjk, pkl -> tqpil",
#         pauli_transforms.conj().transpose((0, 2, 1)),
#         evolved_states.ys,
#         pauli_transforms,
#     )
#     probs = measured_states[..., 1, 1].real
#     return NLLH(probs, data)


# from jax import value_and_grad

# print(f"Optimal Loss: {loss_fn(HAMILTONIAN_PARAMETERS, data)}")
# loss_and_grad = value_and_grad(loss_fn, argnums=0)

# from optax import adam, apply_updates

# key = jax.random.PRNGKey(0)
# init_guess = HAMILTONIAN_PARAMETERS + jax.random.normal(key, (4,)) * 0.1
# guess = init_guess

# # Initialize the optimizer
# opt = adam(LEARNING_RATE)
# opt_state = opt.init(guess)


# # Run training loop

# for epoch in range(EPOCHS):
#     loss, grad = loss_and_grad(guess, data)
#     updates, opt_state = opt.update(grad, opt_state)
#     guess = apply_updates(guess, updates)

#     print(f"Epoch: {epoch}, Loss: {loss}")


# # Get the hessian
# hessian = jax.hessian(loss_fn, argnums=0)(guess, data)

# # Get the covariance matrix
# cov = jnp.linalg.inv(hessian)

# # Get the standard deviation
# std = jnp.sqrt(jnp.diag(cov))

# # Plot
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec

# gs = GridSpec(3, 3)

# fig = plt.figure(figsize=(8, 8))

# ax = fig.add_subplot(gs[:2, :])

# ax.errorbar(
#     jnp.arange(4),
#     guess,
#     yerr=std,
#     fmt="o",
#     label="Estimated Parameters",
# )

# ax.plot(jnp.arange(4), init_guess, "o", label="Initial Guess")


# ax.plot(HAMILTONIAN_PARAMETERS, "o", label="True Parameters")
# ax.legend()

# ax = fig.add_subplot(gs[2, :], sharex=ax)

# ax.errorbar(
#     jnp.arange(4),
#     guess - HAMILTONIAN_PARAMETERS,
#     yerr=std,
#     fmt="o",
#     label="Estimated Parameters",
# )
# ax.axhline(0, color="black", linestyle="--")
# ax.set_xticks(jnp.arange(4))
# ax.set_xticklabels(["Id", "X", "Y", "Z"])
