import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt


# Load the data
data = np.load("two_qubit_gate_single_shot.npz")

# coords
ts, state, basis, sample = (
    data["ts"],
    data["state"],
    data["qubit_basis"],
    np.array(range(data["n"] + 1)),
)

# Data
I1, Q1, I2, Q2 = data["I1"], data["Q1"], data["I2"], data["Q2"]
I1 = I1.flatten().reshape(1000, 8, 4, 4, 3, 3)
Q1 = Q1.flatten().reshape(1000, 8, 4, 4, 3, 3)
I2 = I2.flatten().reshape(1000, 8, 4, 4, 3, 3)
Q2 = Q2.flatten().reshape(1000, 8, 4, 4, 3, 3)

# Convert to xarrays
import xarray as xr

data = {
    "I1": xr.DataArray(
        I1,
        dims=[
            "sample",
            "ts",
            "state1",
            "state2",
            "basis1",
            "basis2",
        ],
    ),
    "Q1": xr.DataArray(
        Q1,
        dims=[
            "sample",
            "ts",
            "state1",
            "state2",
            "basis1",
            "basis2",
        ],
    ),
    "I2": xr.DataArray(
        I2,
        dims=[
            "sample",
            "ts",
            "state1",
            "state2",
            "basis1",
            "basis2",
        ],
    ),
    "Q2": xr.DataArray(
        Q2,
        dims=[
            "sample",
            "ts",
            "state1",
            "state2",
            "basis1",
            "basis2",
        ],
    ),
}

data = xr.Dataset(
    data,
    coords={
        "sample": sample,
        "ts": ts,
        "state1": state,
        "state2": state,
        "basis1": basis,
        "basis2": basis,
    },
)

data.to_zarr("two_qubit_gate_single_shot.zarr", mode="w")


ro_1, ro_2 = data["I1"] + 1j * data["Q1"], data["I2"] + 1j * data["Q2"]

ground_1 = ro_1.sel(basis1="I", basis2="I", state1="I", state2="I", ts=0)
excited_1 = ro_1.sel(basis1="I", basis2="I", state1="x180", state2="x180", ts=0)
ground_2 = ro_2.sel(basis1="I", basis2="I", state1="I", state2="I", ts=0)
excited_2 = ro_2.sel(basis1="I", basis2="I", state1="x180", state2="x180", ts=0)


# Plot the data
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].plot(ground_1.real, ground_1.imag, "o")
ax[0].plot(excited_1.real, excited_1.imag, "o")
ax[0].set_title("Qubit 1")

ax[1].plot(ground_2.real, ground_2.imag, "o")
ax[1].plot(excited_2.real, excited_2.imag, "o")
ax[1].set_title("Qubit 2")


# Hists over I
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# ax[0].hist(ground_1.real, bins=30, alpha=0.5, label="ground")
ax[0].hist(ground_1.real, bins=30, alpha=0.5, label="ground")
ax[0].hist(excited_1.real, bins=30, alpha=0.5, label="excited")

ax[1].hist(ground_2.real, bins=30, alpha=0.5, label="ground")
ax[1].hist(excited_2.real, bins=30, alpha=0.5, label="excited")
