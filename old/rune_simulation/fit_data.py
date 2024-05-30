import sys

sys.path.append("..")
from utils import *
from lindblad_util import jump_operators_to_pauli_matrix, pauli_matrix_to_jump_operators

import jax
import jax.numpy as jnp
import xarray as xr

from diffrax import Dopri5, SaveAt, diffeqsolve, DirectAdjoint, ODETerm, PIDController

# Load data
data = xr.open_dataset("dataset_for_Johann.nc")
data_np = data["data"].values

# Solver options
DURATION = data.time.max().values
STORE_EVERY = (data.time[1] - data.time[0]).values
INITIAL_STEPSIZE = 0.1
MAX_SOLVER_STEPS = 10000


# Guess
rand = jax.random.PRNGKey(42)
hamilton_params = jax.random.normal(rand, (4,)) * 0.01
lindbladian_params = jax.random.normal(rand, (9,)) * 0.01


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


from jax.scipy.linalg import expm

# pauli_gates = [identity(), sigma_x(), sigma_y(), sigma_z()]
pauli_gates = jnp.array([sigma_x(), sigma_y(), sigma_z()])
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


def evolve_system(args):
    """
    Evolve the system using the Runge-Kutta method
    Need following in args
    - init_states
    - hamiltonian_params
    """
    init_states = initial_states
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

    # print(lindbladian_pauli_matrix, lindbladian_pauli_matrix.shape)

    jump_operators = pauli_matrix_to_jump_operators(lindbladian_matrix)

    # print(jump_operators, jump_operators.shape)

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


evolved_states = evolve_system(
    dict(hamiltonian_params=hamilton_params, lindbladian_params=lindbladian_params)
)


expectation = jnp.einsum(
    "tikl, plk -> tpi",
    evolved_states.ys,
    pauli_gates,
)

expectation.shape

# Plot initial guess
times = evolved_states.ts

import matplotlib.pyplot as plt
from matplotlib import style

style.use(
    "/mnt/c/Users/msk377/OneDrive - University of Copenhagen/Desktop/jax_playground/presentation_style.mplstyle"
)

fig, ax = plt.subplots(3, 3, figsize=(16, 16), sharex=True, sharey=True)

for i, state in enumerate(data.state.values):
    for j, basis in enumerate(data.measurmement.values):
        ax[i, j].plot(times, expectation[:, i, j])
        ax[i, j].set_title(f"State: {state}, Measurement: {basis}")

    ax[i, 0].set_ylabel("Expectation Value")
    ax[-1, i].set_xlabel("Time (ns)")

fig.tight_layout()


### Define optimization

data_np = jnp.transpose(data_np, (2, 0, 1))


def loss(hamiltonian_params, lindbladian_params):
    """
    Loss function to minimize
    """
    evolved_states = evolve_system(
        dict(
            hamiltonian_params=hamiltonian_params, lindbladian_params=lindbladian_params
        )
    )

    expectation = jnp.einsum(
        "tikl, plk -> tpi",
        evolved_states.ys,
        pauli_gates,
    ).real

    return jnp.mean((expectation - data_np) ** 2)


from jax import value_and_grad
from optax import adam, apply_updates

learning_rate = 0.0025


loss_and_grad = value_and_grad(loss, argnums=(0, 1))

loss_and_grad(hamilton_params, lindbladian_params)


# Initialize the optimizer
params = (hamilton_params, lindbladian_params)
opt = adam(learning_rate)

opt_state = opt.init((hamilton_params, lindbladian_params))


# Run training loop
EPOCHS = 1000

for epoch in range(EPOCHS):
    loss_val, grad = loss_and_grad(*params)
    updates, opt_state = opt.update(grad, opt_state)
    params = apply_updates(params, updates)

    print(f"Epoch: {epoch}, Loss: {loss_val}")

    if epoch % 100 == 0:
        fig, ax = plt.subplots(3, 3, figsize=(16, 16), sharex=True, sharey=True)

        evolved_states = evolve_system(
            dict(hamiltonian_params=params[0], lindbladian_params=params[1])
        )

        expectation = jnp.einsum(
            "tikl, plk -> tpi",
            evolved_states.ys,
            pauli_gates,
        ).real

        for i, state in enumerate(data.state.values):
            for j, basis in enumerate(data.measurmement.values):
                ax[i, j].plot(times, expectation[:, i, j])
                ax[i, j].plot(times, data_np[:, i, j], "--", alpha=0.5)
                ax[i, j].set_title(f"State: {state}, Measurement: {basis}")

            ax[i, 0].set_ylabel("Expectation Value")
            ax[-1, i].set_xlabel("Time (ns)")

        fig.tight_layout()


lindbladian_params = params[1]

lindbladian_matrix = jnp.zeros((3, 3), dtype=jnp.complex128)
lindbladian_matrix = lindbladian_matrix.at[jnp.tril_indices(3)].set(
    lindbladian_params[:6]
)
lindbladian_matrix = lindbladian_matrix.at[jnp.tril_indices(3, k=-1)].add(
    1j * lindbladian_params[6:]
)

lindbladian_pauli_matrix = lindbladian_matrix @ lindbladian_matrix.T.conj()

# print(lindbladian_pauli_matrix, lindbladian_pauli_matrix.shape)

jump_operators = pauli_matrix_to_jump_operators(lindbladian_pauli_matrix)
