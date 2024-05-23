import sys

sys.path.append("../")
from utils import *
from hamiltonian_learning_utils import *
from tensorflow_probability.substrates.jax.distributions import multinomial


# CONSTANTS
NQUBITS = 2
DURATION = 400
STORE_EVERY = 4
SAMPLES = 1000

# Dataset
import xarray as xr

data = xr.load_dataarray("initial_dataset_with_dissipation.nc").sel(time=slice(0, 204))
data_jax = jnp.array(data.values)


# Initial guesses
two_qubit_pauli_strings = pauli_strings(NQUBITS)
initial_guess_hamiltonian = (
    jax.random.normal(jax.random.PRNGKey(0), (len(two_qubit_pauli_strings),)) * 1e-2
)

initial_guess_lindbladian = (
    jax.random.normal(jax.random.PRNGKey(1), (NQUBITS, 9)) * 1e-3
)

hamiltonian = hamiltonian_from_dict(
    {
        key: value
        for key, value in zip(two_qubit_pauli_strings, initial_guess_hamiltonian)
    },
    number_of_qubits=NQUBITS,
)

pauli_matrices = pauli_matrix_from_cholesko_params(initial_guess_lindbladian, NQUBITS)
jump_operators = pauli_matrix_to_jump_operators(pauli_matrices)


# Setup for the simulation
initial_states, init_index = generate_initial_states(NQUBITS, with_mixed_states=True)
measurement_basis, basis_index = generate_basis_transformations(
    number_of_qubits=NQUBITS
)


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

# Visualization step
import matplotlib.pyplot as plt

HAMILTONIAN_PARAMS = dict(zi=8e-3, iz=-10e-3, xx=1e-5, zz=1e-5)  # Correct values

# Update guesses to be closer to the correct values
true_results = {key: 0.0 for key in two_qubit_pauli_strings}
true_results.update(HAMILTONIAN_PARAMS)

initial_guess_hamiltonian += jnp.array(list(true_results.values())) * 2 * jnp.pi


def plot_hamiltonian_guesses(hamiltonian_params, title="InitialGuesses", errs=None):
    hamiltonian_params = {
        key: value for key, value in zip(two_qubit_pauli_strings, hamiltonian_params)
    }

    fig, ax = plt.subplots(
        nrows=2 if errs is not None else 1, figsize=(10, 5), sharex=True
    )
    if errs is not None:
        ax, ax_err = ax
    ax.plot(
        jnp.arange(len(hamiltonian_params)),
        list(hamiltonian_params.values()),
        "o",
        label="Parameters",
    )
    true_results = {key: 0.0 for key in two_qubit_pauli_strings}
    true_results.update(HAMILTONIAN_PARAMS)
    ax.plot(
        jnp.arange(len(hamiltonian_params)),
        list(true_results.values()),
        "x",
        label="True Values",
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
        residual = jnp.array(list(hamiltonian_params.values())) - jnp.array(
            list(true_results.values())
        )

        ax_err.errorbar(
            jnp.arange(len(hamiltonian_params))[1:],
            residual[1:],
            yerr=errs[1:],
            fmt=".",
            label="Residuals",
            color="C1",
        )

        ax_err.hlines(0, 0, len(hamiltonian_params), linestyles="--", color="black")

    ax.set_xticks(jnp.arange(len(hamiltonian_params)))
    ax.set_xticklabels(hamiltonian_params.keys())
    ax.set_title(title)
    fig.savefig(f"{title}.png")


plot_hamiltonian_guesses(initial_guess_hamiltonian / 2 / jnp.pi)


# Training Setup
def loss_fn(
    hamiltonian_params, lindbladian_params, initial_states, measurement_basis, data
):
    hamiltonian = hamiltonian_from_dict(
        {key: value for key, value in zip(two_qubit_pauli_strings, hamiltonian_params)},
        number_of_qubits=NQUBITS,
    )
    pauli_matrices = pauli_matrix_from_cholesko_params(lindbladian_params, NQUBITS)
    jump_operators = pauli_matrix_to_jump_operators(pauli_matrices)

    results = solver(
        initial_state=initial_states,
        hamiltonian=hamiltonian,
        jump_operators=jump_operators,
    )

    transformed_states = jnp.einsum(
        "mkl, tiln, mno ->imtko",
        measurement_basis,
        results.ys,
        jnp.conj(measurement_basis).transpose(0, 2, 1),
    )

    probs = get_probability_from_states(transformed_states)

    return -jnp.sum(
        multinomial.Multinomial(probs=probs, total_count=SAMPLES).log_prob(data)
    )


from jax import value_and_grad

loss_and_grad = value_and_grad(loss_fn, argnums=(0, 1))

# Optimization Loop
EPOCHS = 1000
LEARNING_RATE = 1e-3

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
        measurement_basis,
        data_jax,
    )
    updates, opt_state = opt.update(grad, opt_state)
    hamiltonian_params, lindbladian_params = apply_updates(
        (hamiltonian_params, lindbladian_params), updates
    )

    print(f"Epoch: {epoch}, Loss: {loss_val}")

    if epoch % 25 == 0:
        plot_hamiltonian_guesses(hamiltonian_params / 2 / jnp.pi, title=f"Epoch{epoch}")


from jax import hessian

hess = hessian(loss_fn, argnums=(0))(
    hamiltonian_params, lindbladian_params, initial_states, measurement_basis, data_jax
)

errs = jnp.sqrt(jnp.diag(jnp.linalg.inv(hess)))
plot_hamiltonian_guesses(
    hamiltonian_params / 2 / jnp.pi, title=f"Epoch{epoch}", errs=errs / 2 / jnp.pi
)
