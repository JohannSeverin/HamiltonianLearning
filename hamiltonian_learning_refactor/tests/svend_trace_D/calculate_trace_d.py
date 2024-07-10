import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from iminuit import Minuit
from iminuit.cost import LeastSquares


NAME = "Test_reshaped"
POLY_ORDER = 3


### Load dataset
dataset = xr.open_zarr(f"data/{NAME}.zarr")

# Decompose to only the parts quired for virtual circuits
xyz = list("xyz")
single_qubit_dataset = [
    dataset.expectation_values.mean("init1").sel(exp1="i"),
    dataset.expectation_values.mean("init0").sel(exp0="i"),
]

dataset_two_qubit = dataset.expectation_values.sel(exp1=xyz, exp0=xyz)

# Combine qubits to the proper circuits

# Loop over qubits
single_qubit_virtual_circuits = []


for qubit_number in range(2):
    # Setup virtual qubit circuits
    virtual_circuits = {}

    # Loop over initial states
    for init_state_index, init_state in enumerate(xyz):
        # Measurements are done in the same basis as initial state preparation
        measurement_basis = init_state

        # Take the difference between positve and negative initial states
        exp_val = single_qubit_dataset[qubit_number].sel(
            {f"exp{qubit_number}": measurement_basis}
        )

        positve = 1 / 3 + exp_val.sel({f"init{qubit_number}": init_state}).values
        negative = 1 / 3 - exp_val.sel({f"init{qubit_number}": f"-{init_state}"}).values

        # Calculate the difference and save in the virtual circuit
        virtual_circuits[init_state] = positve + negative

    # Save the virtual circuit
    single_qubit_virtual_circuits.append(virtual_circuits)


# Combine the experiments to two qubit virutal circuits
two_qubit_virtual_circuits = {}
for i in xyz:
    for j in xyz:
        dataset_xyz_xyz = dataset_two_qubit.sel(exp1=i, exp0=j)
        two_qubit_virtual_circuits[f"{i}{j}"] = (
            dataset_xyz_xyz.sel(init1=i, init0=j).values
            + dataset_xyz_xyz.sel(init1=f"-{i}", init0=j).values
            + dataset_xyz_xyz.sel(init1=i, init0=f"-{j}").values
            + dataset_xyz_xyz.sel(init1=f"-{i}", init0=f"-{j}").values
        )


# Plot large gitter
fig, ax = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)

ax_qubit_0 = ax[1:, 0]
ax_qubit_1 = ax[0, 1:]

linear_coefficients_qubit_0 = {}
linear_coefficients_qubit_1 = {}

# Plot the single qubit examples

for i, key in enumerate(xyz):
    ax_qubit_0[i].scatter(
        dataset.time, single_qubit_virtual_circuits[0][key], label=key
    )

    x = dataset.time
    y = single_qubit_virtual_circuits[0][key]

    fit_params = np.polyfit(x, y, POLY_ORDER)

    ax_qubit_0[i].plot(x, np.poly1d(fit_params)(x), label="Fit")
    ax_qubit_0[i].plot(x, np.poly1d(fit_params[-2:])(x), label="1st order")

    ax_qubit_0[i].set_xlabel("Time [ns]")
    ax_qubit_0[i].set_title(f"Qubit 0: {key}")

    ax_qubit_0[i].text(
        0.1,
        0.2,
        f"rate: {fit_params[-2]:.2e}",
        ha="left",
        va="center",
        transform=ax_qubit_0[i].transAxes,
    )


for i, key in enumerate(xyz):
    ax_qubit_1[i].scatter(
        dataset.time, single_qubit_virtual_circuits[1][key], label=key
    )

    x = dataset.time
    y = single_qubit_virtual_circuits[1][key]

    fit_params = np.polyfit(x, y, POLY_ORDER)

    ax_qubit_1[i].plot(x, np.poly1d(fit_params)(x), label="Fit")
    ax_qubit_1[i].plot(x, np.poly1d(fit_params[-2:])(x), label="1st order")

    ax_qubit_1[i].set_xlabel("Time [ns]")
    ax_qubit_1[i].set_title(f"Qubit 1: {key}")

    ax_qubit_1[i].text(
        0.1,
        0.2,
        f"rate: {fit_params[-2]:.2e}",
        ha="left",
        va="center",
        transform=ax_qubit_1[i].transAxes,
    )

# Repeat for the two qubit examples
ax_two_qubit = ax[1:, 1:]

for i, key_i in enumerate(xyz):
    for j, key_j in enumerate(xyz):
        key = f"{key_i}{key_j}"
        ax_two_qubit[i, j].scatter(
            dataset.time, two_qubit_virtual_circuits[key], label=key
        )

        x = dataset.time
        y = two_qubit_virtual_circuits[key]

        fit_params = np.polyfit(x, y, POLY_ORDER)

        ax_two_qubit[i, j].plot(x, np.poly1d(fit_params)(x), label="Fit")
        ax_two_qubit[i, j].plot(x, np.poly1d(fit_params[-2:])(x), label="1st order")

        ax_two_qubit[i, j].set_xlabel("Time [ns]")
        ax_two_qubit[i, j].set_title(f"Two Qubit: {key}")

        ax_two_qubit[i, j].text(
            0.1,
            0.2,
            f"rate: {fit_params[-2]:.2e}",
            ha="left",
            va="center",
            transform=ax_two_qubit[i, j].transAxes,
        )


# Main figure
Qubit0 = np.array([single_qubit_virtual_circuits[0][key] for key in xyz])
Qubit1 = np.array([single_qubit_virtual_circuits[1][key] for key in xyz])
TwoQubit = np.array(
    [two_qubit_virtual_circuits[key] for key in two_qubit_virtual_circuits]
)

TwoQubit = np.concatenate([TwoQubit, Qubit0, Qubit1], axis=0)

x = dataset.time
y = (Qubit0.sum(axis=0) / 3 + Qubit1.sum(axis=0) / 3) / 2

fit_params = np.polyfit(x, y, POLY_ORDER)
ax[0, 0].scatter(dataset.time, y, label="sum / -3")
ax[0, 0].plot(x, np.poly1d(fit_params)(x), color="C0", label="A + B")
ax[0, 0].plot(x, np.poly1d(fit_params[-2:])(x), color="C0", label="A + B 1st order")

ax[0, 0].text(
    0.1,
    0.2,
    f"A + B rate: {fit_params[-2]:.2e}",
    ha="left",
    va="center",
    transform=ax[0, 0].transAxes,
)

# y = Qubit1.sum(axis=0) / 3

# fit_params = np.polyfit(x, y, POLY_ORDER)
# ax[0, 0].scatter(dataset.time, y, label="sum / -3")
# ax[0, 0].plot(x, np.poly1d(fit_params)(x), label="Fit")
# ax[0, 0].plot(x, np.poly1d(fit_params[-2:])(x), label="1st order")

# ax[0, 0].text(
#     0.1,
#     0.3,
#     f"rate: {fit_params[-2]:.2e}",
#     ha="left",
#     va="center",
#     transform=ax[0, 0].transAxes,
# )

y = TwoQubit.sum(axis=0) / 15

fit_params = np.polyfit(x, y, POLY_ORDER)
ax[0, 0].scatter(dataset.time, y, label="sum / -15")
ax[0, 0].plot(x, np.poly1d(fit_params)(x), color="C1", label="AB")
ax[0, 0].plot(x, np.poly1d(fit_params[-2:])(x), color="C1", label="AB 1st order")

ax[0, 0].text(
    0.1,
    0.4,
    f"AB rate: {fit_params[-2]:.2e}",
    ha="left",
    va="center",
    transform=ax[0, 0].transAxes,
)

ax[0, 0].set_xlabel("Time [ns]")


fig.savefig(f"figures/{NAME}.png")
