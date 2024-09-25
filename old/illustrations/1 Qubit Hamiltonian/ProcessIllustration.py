import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import itertools
import jax
import xarray as xr

sys.path.append("../..")
from utils import *
from hamiltonian_learning_utils import *


import matplotlib.pyplot as plt

plt.style.use(
    "/home/archi1/projects/HamiltonianLearning/hamiltonian_learning_refactor/utils/presentation_figure_template.mplstyle"
)


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

times = jnp.arange(0, DURATION + TIMESTEPS, TIMESTEPS)

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
    tlist=times,
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
    "Loss": {0: negative_log_likelihood(hamiltonian_params)},
    "parameters": {0: hamiltonian_params},
}


### Setup the optimizer ###
from optax import adam, apply_updates

opt = adam(LEARNING_RATE)
opt_state = opt.init(hamiltonian_params)

for i in range(ITERATIONS):
    loss, grad = loss_and_grad(hamiltonian_params)
    updates, opt_state = opt.update(grad, opt_state)
    hamiltonian_params = apply_updates(hamiltonian_params, updates)

    print(f"Iteration {i+1}, Loss: {loss}")

    logs["Loss"][i + 1] = loss

    if (i + 1) % 5 == 0:
        logs["parameters"][i + 1] = hamiltonian_params


# Analyze the results


def simulate_given_parameters(hamiltonian_params):

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

    return probs


# Data
example_data = data[:, 0, 0]
data_fig, data_ax = plt.subplots(1, 1, figsize=(6, 4))

data_ax.scatter(
    times, example_data, label="Data", color="C0", alpha=1, s=15, marker="o"
)


data_ax.set(title="Initial Z - Measure Z", ylabel="Counts of 1", xlabel="Time (µs)")

data_ax.set(title="Initial Z - Measure Z", ylabel="Counts of 1", xlabel="Time (µs)")

data_ax.set_xticklabels(data_ax.get_xticks() * 1e-3)

## Do Illustration plot of the z - z result versus log likelihood
fig, axes = plt.subplots(
    1, 3, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 2, 1]}, sharey=True
)
example_data = data[:, 0, 0]
axes[0].set_title("Initial Z - Measure Z")


axes[0].scatter(
    times, example_data, label="Data", color="C0", alpha=1, s=15, marker="o"
)

axes[0].set(title="Initial Z - Measure Z", ylabel="Counts of 1", xlabel="Time (µs)")

axes[0].set_xticklabels(axes[0].get_xticks() * 1e-3)

# Show zoomin
test_time = 5.5e3
idx = jnp.argmin(jnp.abs(times - test_time))

probs = simulate_given_parameters(logs["parameters"][0])

axes[0].plot(times, 100 * probs[:, 0, 0, 1], color="C1")
axes[1].plot(times, 100 * probs[:, 0, 0, 1], color="C1")

axes[1].set_xlim(test_time - 0.05e3, test_time + 0.05e3)

axes[1].scatter(times[idx], data[idx, 0, 0], s=100)

axes[2].plot(
    Binomial(total_count=SAMPLES, probs=probs[idx, 0, 0, 1]).log_prob(
        jnp.arange(0, SAMPLES + 1)
    ),
    jnp.arange(0, SAMPLES + 1),
    color="C1",
)

axes[2].scatter(
    Binomial(total_count=SAMPLES, probs=probs[idx, 0, 0, 1]).log_prob(data[idx, 0, 0]),
    data[idx, 0, 0],
    zorder=10,
    s=100,
)

axes[1].scatter(times, data[:, 0, 0], color="C0", alpha=0.5, s=15, marker="o")
axes[0].indicate_inset_zoom(axes[1])

axes[1].set(title="Example Simulation", xlabel="Time (µs)")
axes[2].set(title="Log Likelihood", xlabel="Log Likelihood")

fig.tight_layout()

fig.savefig("figs/LindbladianIllustration.svg")

# Plot the progress of the optimization
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

example_data = data[:, 0, 0]

ax[0].scatter(times, example_data, label="Data", color="C0", alpha=1, s=15, marker="o")

ax[0].set(title="Initial Z - Measure Z", ylabel="Counts of 1", xlabel="Time (µs)")

ax[0].set_xticklabels(ax[0].get_xticks() * 1e-3)


###
import numpy as np

losses_iteration = np.array(list(logs["Loss"].keys()))
losses = np.array(list(logs["Loss"].values()))

ax[1].plot(
    list(logs["Loss"].keys()), list(logs["Loss"].values()), label="Loss", color="white"
)
ax[1].set_yscale("log")
ax[1].set(
    title="Loss vs Iteration",
    xlabel="Iteration",
    ylabel="Negative Log Likelihood",
)

for i, iteration_index in enumerate([0, 5, 50, 200]):
    probs = simulate_given_parameters(logs["parameters"][iteration_index])

    ax[0].plot(
        times,
        100 * probs[:, 0, 0, 1],
        label=f"Iteration {iteration_index}",
        alpha=1.0,
        linewidth=2,
        color=f"C{i+1}",
    )

    ax[1].plot(
        losses_iteration[losses_iteration <= iteration_index],
        losses[losses_iteration <= iteration_index],
        color="black",
    )

    ax[1].scatter(
        iteration_index,
        logs["Loss"][iteration_index],
        color=f"C{i+1}",
        marker="o",
        s=100,
        zorder=10,
    )

    fig.tight_layout()

    fig.savefig(f"figs/optimization_progress_iteration_{iteration_index}.svg")


# Add zoomins
import pickle

with open("lindbladian_logs.pkl", "rb") as f:
    lindbladian_logs = pickle.load(f)

ax[1].plot(
    list(lindbladian_logs["Loss"].keys()),
    list(lindbladian_logs["Loss"].values()),
)

zoom_in = ax[1].inset_axes([0.65, 0.65, 0.3, 0.3])

zoom_in.plot(
    list(lindbladian_logs["Loss"].keys()),
    list(lindbladian_logs["Loss"].values()),
)

zoom_in.plot(
    list(logs["Loss"].keys()), list(logs["Loss"].values()), label="Loss", color="k"
)

zoom_in.set(
    xlim=(190, 200),
    ylim=(min(logs["Loss"].values()) - 150, min(logs["Loss"].values()) + 100),
)

min_loss = min(logs["Loss"].values())
min_lindbladian_loss = min(lindbladian_logs["Loss"].values())

# Add an arrow between the two minimum values
zoom_in.annotate(
    "",
    xy=(195, min_loss),
    xytext=(195, min_lindbladian_loss),
    arrowprops=dict(arrowstyle="<->", color="red"),
)

# Annotate the difference
zoom_in.text(
    196,
    (min_loss + min_lindbladian_loss) / 2,
    f"{int((abs(min_loss - min_lindbladian_loss)))}",
    ha="left",
    va="center",
    color="red",
)

fig.savefig("figs/Added_Lindblad_Model.svg")
fig.show()

# # Show the likelihood of the simulation
# test_time = 5.5e3
# idx = jnp.argmin(jnp.abs(times - test_time))

# fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# probs = simulate_given_parameters(logs["parameters"][0])

# ax[0].plot(times, 100 * probs[:, 0, 0, 1], color="C1")

# ax[0].set_xlim(test_time - 0.05e3, test_time + 0.05e3)

# ax[0].scatter(times[idx], data[idx, 0, 0], s=100)

# ax[1].plot(
#     Binomial(total_count=SAMPLES, probs=probs[idx, 0, 0, 1]).log_prob(
#         jnp.arange(0, SAMPLES + 1)
#     ),
#     jnp.arange(0, SAMPLES + 1),
#     color="C1",
# )

# ax[1].scatter(
#     Binomial(total_count=SAMPLES, probs=probs[idx, 0, 0, 1]).log_prob(data[idx, 0, 0]),
#     data[idx, 0, 0],
#     zorder=10,
#     s=100,
# )

# ax[0].scatter(times, data[:, 0, 0], color="C0", alpha=0.5, s=15, marker="o")

# ax[0].set(title="Example Simulation", xlabel="Time (µs)", ylabel="Counts of 1")
# ax[1].set(title="Log Likelihood", ylabel="Counts of 1", xlabel="Log Likelihood")


# fig.tight_layout()
# ### PLOTTING THE FIT ###
# fig, ax = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)

# fig.suptitle("Expectation Values")

# for i, j in itertools.product(range(3), range(3)):

#     ax[i, j].set_title(f"Init {['X', 'Y', 'Z'][i]} - Measure {['X', 'Y', 'Z'][j]}")

#     ax[i, j].plot(
#         result.ts,
#         reconstructed_exp_vals[:, i, j],
#         ".",
#         label="Expectation Value [+]",
#         color="C0",
#         alpha=0.75,
#     )
#     ax[i, j].plot(
#         result.ts,
#         reconstructed_exp_vals[:, i + 3, j],
#         ".",
#         label="Expectation Value [-]",
#         color="C1",
#         alpha=0.75,
#     )
#     ax[i, j].plot(
#         result.ts,
#         exp_values[:, i, j],
#         linewidth=2,
#         color="k",
#     )
#     ax[i, j].plot(
#         result.ts,
#         exp_values[:, i + 3, j],
#         linewidth=2,
#         color="k",
#     )

#     if i == 2:
#         ax[i, j].set_xlabel("Time (ns)")

#     ax[i, j].set_ylabel(f"$<{['X', 'Y', 'Z'][j]}>$")


# ax[0, -1].legend(loc="upper right")
# fig.tight_layout()
# # fig.savefig("SampledData.png")


# ### ANALYZE ERRORS ###
# hess = jax.hessian(negative_log_likelihood)(hamiltonian_params)
# cov = jnp.linalg.inv(hess.squeeze())

# errors = jnp.sqrt(jnp.diag(cov))


# # Scan over the parameter space
# offset_range = jnp.linspace(-7e-7, 7e-7, 100)
# X, Y, Z = 1e-4, -2e-4, 1e-4

# # Hamiltonian Parmas
# negative_log_likelihood = jax.jit(negative_log_likelihood)

# losses = dict(
#     X=jnp.zeros_like(offset_range),
#     Y=jnp.zeros_like(offset_range),
#     Z=jnp.zeros_like(offset_range),
# )

# for param_idx, param in enumerate(["X", "Y", "Z"]):
#     for i, offset in enumerate(offset_range):
#         test_hamiltonian_params = hamiltonian_params.at[0, param_idx].add(offset)
#         loss = negative_log_likelihood(test_hamiltonian_params)
#         losses[param] = losses[param].at[i].set(loss)
#         if i % 10 == 0:
#             print(f"Iteration {i} loss {loss:.2f}")

# # Plot the results
# fig, ax = plt.subplots(1, 3, figsize=(12, 4))

# for i, param in enumerate(["X", "Y", "Z"]):
#     ax[i].plot(
#         (hamiltonian_params[0, i] + offset_range) * 1e6,
#         losses[param] - losses[param].min(),
#     )
#     ax[i].set_title(f"Loss vs offset {param}")
#     ax[i].set_xlabel(f"{param} Value (kHz)")
#     ax[i].set_ylabel("Loss - Min Loss")

#     ax[i].axvline(
#         hamiltonian_params[0, i] * 1e6,
#         color="k",
#         linestyle="--",
#         label="Predicted Value",
#     )

#     ax[i].axvline([X, Y, Z][i] * 1e6, color="red", linestyle="--", label="True Value")

#     ax[i].axhline(0.5, color="k", linestyle="-", alpha=0.5, label="1/2 LLH")

#     ax[i].axvline(
#         (hamiltonian_params[0, i] - errors[i]) * 1e6,
#         color="k",
#         linestyle="--",
#         alpha=0.5,
#         label="Error Bounds",
#     )
#     ax[i].axvline(
#         (hamiltonian_params[0, i] + errors[i]) * 1e6,
#         color="k",
#         linestyle="--",
#         alpha=0.5,
#     )

#     # ax[i].axhline(negative_log_likelihood(hamiltonian_params) + 0.5)

# # ax[0].legend(loc="upper left")

# fig.tight_layout()
# fig.savefig("ParameterScan.svg")

# uniform_probs = jnp.ones_like(probs) / 2

# # P value estimate for the fit
# sampled_from_fit = Binomial(total_count=SAMPLES, probs=probs[..., 1]).sample(
#     1000, seed=jax.random.PRNGKey(0)
# )

# sampled_from_uniform = Binomial(
#     total_count=SAMPLES, probs=uniform_probs[..., 1]
# ).sample(1000, seed=jax.random.PRNGKey(0))

# sampled_LLH = -(
#     Binomial(total_count=SAMPLES, probs=probs[..., 1])
#     .log_prob(sampled_from_fit)
#     .sum(axis=(1, 2, 3))
# )

# sampled_LLH_uniform = -(
#     Binomial(total_count=SAMPLES, probs=uniform_probs[..., 1]).log_prob(
#         sampled_from_uniform
#     )
#     # .sum(axis=(1, 2, 3))
# )

# uniform_LLH = (
#     -Binomial(total_count=SAMPLES, probs=uniform_probs[..., 1]).log_prob(data).sum()
# )

# fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# ax.set(
#     title="P-Value Estimate",
#     xlabel="Negative Log Likelihood",
#     ylabel="Density",
# )

# ax.hist(sampled_LLH, bins=30, alpha=0.5, label="Sampled LLH", density=True)
# ax.hist(sampled_LLH_uniform, bins=30, alpha=0.5, label="Sampled LLH", density=True)

# ax.axvline(loss, color="r", linestyle="--", label="Data LLH")

# quantiles = jnp.quantile(sampled_LLH, jnp.array([0.05, 0.95]))

# ax.axvline(quantiles[0], color="k", linestyle="--", label="5%, 95% quantiles")
# ax.axvline(quantiles[1], color="k", linestyle="--")

# pval = jnp.mean(sampled_LLH > loss)

# ax.text(0.5, 0.65, transform=ax.transAxes, s=f"P-value: {pval:.3f}", ha="center")

# ax.legend()

# # fig.savefig("PValueEstimate.svg")
