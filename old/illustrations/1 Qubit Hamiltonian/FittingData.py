import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import itertools
import jax
import xarray as xr

jax.config.update("jax_enable_x64", True)

sys.path.append("../..")
from utils import *
from hamiltonian_learning_utils import *
import matplotlib.pyplot as plt 
plt.style.use("/home/johann/projects/HamiltonianLearning/hamiltonian_learning_refactor/utils/presentation_figure_template.mplstyle")

# Load Data
xarray = xr.open_dataarray("DatasetLindblad.nc")

data = xarray.values[..., 1]

# Load hyper params
NQUBITS = 1
SAMPLES = 100
DURATION = 10000
TIMESTEPS = 20
LEARNING_RATE = 1e-4
ITERATIONS = 200


# Hamiltonian Parmas
hamiltonian_params = jax.random.normal(jax.random.PRNGKey(0), (NQUBITS, 3)) * 1e-4


# Initial States
initial_states = generate_initial_states(NQUBITS, include_negative_states=True)[0]
basis_transformation = generate_basis_transformations(NQUBITS)[0]

# Define the solver
solver = create_solver(
    t1=DURATION,
    t0=0,
    adjoint=True,
    tlist=jnp.arange(0, DURATION + TIMESTEPS, TIMESTEPS),
    number_of_jump_operators=0,
    max_steps=10000,
)


# Define Loss
from tensorflow_probability.substrates.jax.distributions import Binomial


def negative_log_likelihood(hamiltonian_params):
    hamiltonian = build_local_hamiltonian(hamiltonian_params, NQUBITS)

    result = solver(
        initial_states,
        hamiltonian,
    )

    result_in_measurement_basis = apply_basis_transformations_dm(
        result.ys, basis_transformation
    )

    probs = get_probability_from_states(result_in_measurement_basis)[..., 1]

    LLH = Binomial(total_count=SAMPLES, probs=probs).log_prob(data).sum()

    return -LLH


# Optimize
from jax import value_and_grad
from optax import adam, apply_updates


loss_and_grad = jax.jit(value_and_grad(negative_log_likelihood, (0)))

logs = {
    "Loss": {},
    "Parameters": {},
}


### Setup the optimizer ###
from optax import adam, apply_updates

opt = adam(LEARNING_RATE)
opt_state = opt.init(hamiltonian_params)

for i in range(ITERATIONS):
    loss, grad = loss_and_grad(hamiltonian_params)
    updates, opt_state = opt.update(grad, opt_state)
    hamiltonian_params = apply_updates(hamiltonian_params, updates)

    print(f"Iteration {i}, Loss: {loss}")

    logs["Loss"][i] = loss

    if i % 10 == 0:
        logs["Parameters"][i] = hamiltonian_params


# Analyze the results
hamiltonian = build_local_hamiltonian(hamiltonian_params, NQUBITS)
result = solver(initial_states, hamiltonian)

# Get the expectation values and samples from the result
observables = N_qubit_pauli_gate_set_with_identity(NQUBITS)[1:]
exp_values = expectation_values(result.ys, observables)

# Transform the result to the measurement basis
result_in_measurement_basis = apply_basis_transformations_dm(
    result.ys, basis_transformation
)

reconstructed_exp_vals = -2 * data / SAMPLES + 1

# Extract Probabilities and a sample of the true states
probs = get_probability_from_states(result_in_measurement_basis)


### PLOTTING THE FIT ###
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)

fig.suptitle("Expectation Values")

for i, j in itertools.product(range(3), range(3)):

    ax[i, j].set_title(f"Init {['X', 'Y', 'Z'][i]} - Measure {['X', 'Y', 'Z'][j]}")

    ax[i, j].plot(
        result.ts,
        reconstructed_exp_vals[:, i, j],
        ".",
        label="Expectation Value [+]",
        color="C0",
        alpha=0.75,
    )
    ax[i, j].plot(
        result.ts,
        reconstructed_exp_vals[:, i + 3, j],
        ".",
        label="Expectation Value [-]",
        color="C1",
        alpha=0.75,
    )
    ax[i, j].plot(
        result.ts,
        exp_values[:, i, j],
        linewidth=2,
        color="k",
    )
    ax[i, j].plot(
        result.ts,
        exp_values[:, i + 3, j],
        linewidth=2,
        color="k",
    )

    if i == 2:
        ax[i, j].set_xlabel("Time (ns)")

    ax[i, j].set_ylabel(f"$<{['X', 'Y', 'Z'][j]}>$")


ax[0, -1].legend(loc="upper right")
fig.tight_layout()
# fig.savefig("SampledData.png")


### ANALYZE ERRORS ###
hess = jax.hessian(negative_log_likelihood)(hamiltonian_params)
cov = jnp.linalg.inv(hess.squeeze())

errors = jnp.sqrt(jnp.diag(cov))


# Scan over the parameter space
offset_range = jnp.linspace(-7e-7, 7e-7, 100)
X, Y, Z = 1e-4, -2e-4, 1e-4

# Hamiltonian Parmas
negative_log_likelihood = jax.jit(negative_log_likelihood)

losses = dict(
    X=jnp.zeros_like(offset_range),
    Y=jnp.zeros_like(offset_range),
    Z=jnp.zeros_like(offset_range),
)

for param_idx, param in enumerate(["X", "Y", "Z"]):
    for i, offset in enumerate(offset_range):
        test_hamiltonian_params = hamiltonian_params.at[0, param_idx].add(offset)
        loss = negative_log_likelihood(test_hamiltonian_params)
        losses[param] = losses[param].at[i].set(loss)
        if i % 10 == 0:
            print(f"Iteration {i} loss {loss:.2f}")

# Plot the results
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

for i, param in enumerate(["X", "Y", "Z"]):
    ax[i].plot(
        (hamiltonian_params[0, i] + offset_range) * 1e6,
        losses[param] - losses[param].min(),
    )
    ax[i].set_title(f"Loss vs offset {param}")
    ax[i].set_xlabel(f"{param} Value (kHz)")
    ax[i].set_ylabel("Loss - Min Loss")

    ax[i].axvline(
        hamiltonian_params[0, i] * 1e6,
        color="k",
        linestyle="--",
        label="Predicted Value",
    )

    ax[i].axvline([X, Y, Z][i] * 1e6, color="red", linestyle="--", label="True Value")

    ax[i].axhline(0.5, color="k", linestyle="-", alpha=0.5, label="1/2 LLH")

    ax[i].axvline(
        (hamiltonian_params[0, i] - errors[i]) * 1e6,
        color="k",
        linestyle="--",
        alpha=0.5,
        label="Error Bounds",
    )
    ax[i].axvline(
        (hamiltonian_params[0, i] + errors[i]) * 1e6,
        color="k",
        linestyle="--",
        alpha=0.5,
    )

    # ax[i].axhline(negative_log_likelihood(hamiltonian_params) + 0.5)

# ax[0].legend(loc="upper left")

fig.tight_layout()
fig.savefig("ParameterScan.svg")

uniform_probs = jnp.ones_like(probs) / 2

# P value estimate for the fit
sampled_from_fit = Binomial(total_count=SAMPLES, probs=probs[..., 1]).sample(
    1000, seed=jax.random.key(0)
)

sampled_from_uniform = Binomial(
    total_count=SAMPLES, probs=uniform_probs[..., 1]
).sample(1000, seed=jax.random.PRNGKey(0))

sampled_LLH = -(
    Binomial(total_count=SAMPLES, probs=probs[..., 1])
    .log_prob(sampled_from_fit)
    .sum(axis=(1, 2, 3))
)

sampled_LLH_uniform = -(
    Binomial(total_count=SAMPLES, probs=uniform_probs[..., 1]).log_prob(
        sampled_from_uniform
    )
    # .sum(axis=(1, 2, 3))
)

uniform_LLH = (
    -Binomial(total_count=SAMPLES, probs=uniform_probs[..., 1]).log_prob(data).sum()
)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.set(
    title="P-Value Estimate",
    xlabel="Negative Log Likelihood",
    ylabel="Density",
)

ax.hist(sampled_LLH, bins=30, alpha=0.5, label="Sampled LLH", density=True)
ax.hist(sampled_LLH_uniform, bins=30, alpha=0.5, label="Sampled LLH", density=True)

ax.axvline(loss, color="r", linestyle="--", label="Data LLH")

quantiles = jnp.quantile(sampled_LLH, jnp.array([0.05, 0.95]))

ax.axvline(quantiles[0], color="k", linestyle="--", label="5%, 95% quantiles")
ax.axvline(quantiles[1], color="k", linestyle="--")

pval = jnp.mean(sampled_LLH > loss)

ax.text(0.5, 0.65, transform=ax.transAxes, s=f"P-value: {pval:.3f}", ha="center")

ax.legend()

# fig.savefig("PValueEstimate.svg")
