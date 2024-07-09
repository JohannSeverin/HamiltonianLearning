import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from iminuit import Minuit
from iminuit.cost import LeastSquares


NAME = "Test"
POLY_ORDER = 3


### Load dataset
dataset = xr.open_zarr(f"{NAME}.zarr")


# Processing
xyz = list("xyz")
xyz_minus = "x, y, z, -x, -y, -z".split(", ")


# Sum over identity configurations


# Qubit 1
sum_to_get_qubit_1_exp = {i: [f"{i}{j}" for j in xyz_minus] for i in xyz_minus}
sum_to_get_qubit_2_exp = {i: [f"{j}{i}" for j in xyz_minus] for i in xyz_minus}


# Qubit 1 expectation
qubit_1_exp = {
    i: (
        dataset.expectation_values.sel(
            initial_gate=sum_to_get_qubit_1_exp[i], expectation="i" + i
        ).sum("initial_gate")
        - dataset.expectation_values.sel(
            initial_gate=sum_to_get_qubit_2_exp["-" + i], expectation="i" + i
        ).sum("initial_gate")
    )
    / 2
    for i in xyz
}

qubit_2_exp = {
    i: (
        dataset.expectation_values.sel(
            initial_gate=sum_to_get_qubit_2_exp[i], expectation=i + "i"
        ).sum("initial_gate")
        - dataset.expectation_values.sel(
            initial_gate=sum_to_get_qubit_1_exp["-" + i], expectation=i + "i"
        ).sum("initial_gate")
    )
    / 2
    for i in xyz
}

# Sum over non-identity configurations
