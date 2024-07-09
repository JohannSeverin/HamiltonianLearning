import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from iminuit import Minuit
from iminuit.cost import LeastSquares


NAME = "OneQubitDataset"
POLY_ORDER = 3


### Load dataset
dataset = xr.open_zarr(f"data/{NAME}.zarr")

xyz = list("xyz")
summed_dict = {}

for i in xyz:
    summed_dict[i] = (
        dataset.expectation_values.sel(initial_gate=i).values
        - dataset.expectation_values.sel(initial_gate=f"-{i}").values
    ) / 2


def poly_fit(x, y, order=3):

    fit = np.polyfit(x, y, order)

    return fit
    # poly_func = lambda coffs: (np.poly1d(coffs)(x) - y) ** 2

    # cost = LeastSquares(x, y, np.ones_like(y), poly_func)

    # m = Minuit(poly_func, coffs=np.ones(order + 1, dtype=float))
    # m.limits = [(None, None)] * (order + 1)
    # m.migrad()

    # return poly_func(x, m.values["coffs"])

linear_order = {}

%matplotlib inline
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for i, key in enumerate(xyz):
    ax[i].scatter(dataset.time, summed_dict[key][:, i + 1], label=key)

    x = dataset.time
    y = summed_dict[key][:, i + 1]

    fit_params = poly_fit(x, y, POLY_ORDER)

    ax[i].plot(x, np.poly1d(fit_params)(x), label="Fit")
    ax[i].plot(x, np.poly1d(fit_params[-2:])(x), label="1st order")

    ax[i].set_xlabel("Time [ns]")
    ax[i].set_title(f"Expectation Value of {key}")

    linear_order[key] = fit_params[-2]


ax[i].set_ylabel("Expectation Value")
ax[i].legend()
fig.tight_layout()

fig, ax = plt.subplots(1)
x = dataset.time
y = (summed_dict["x"][:, 1] + summed_dict["y"][:, 2] + summed_dict["z"][:, 3]) / -3


ax.scatter(dataset.time, y, label="sum / -3")


fit_params = poly_fit(x, y, POLY_ORDER)

ax.plot(x, np.poly1d(fit_params)(x), label="Fit")
ax.plot(x, np.poly1d(fit_params[-2:])(x), label="1st order")

ax.set_xlabel("Time [ns]")
ax.set_title(f"Expectation Value of {key}")

linear_order[key] = fit_params[-2]