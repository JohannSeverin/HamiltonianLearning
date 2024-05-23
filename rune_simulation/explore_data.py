import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from matplotlib import style
import matplotlib.pyplot as plt

style.use(
    "/mnt/c/Users/msk377/OneDrive - University of Copenhagen/Desktop/jax_playground/presentation_style.mplstyle"
)

# Load data

data = xr.open_dataset("dataset_for_Johann.nc")
data_np = data["data"].values

times = data.time.values

fig, ax = plt.subplots(3, 3, figsize=(16, 16), sharex=True, sharey=True)

for i, state in enumerate(data.state.values):
    for j, basis in enumerate(data.measurmement.values):
        ax[i, j].plot(times, data_np[i, j])
        ax[i, j].set_title(f"State: {state}, Measurement: {basis}")

    ax[i, 0].set_ylabel("Expectation Value")
    ax[-1, i].set_xlabel("Time (ns)")

fig.tight_layout()
