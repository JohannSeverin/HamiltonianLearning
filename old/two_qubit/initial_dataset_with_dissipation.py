import sys

sys.path.append("../")
from utils import *
from hamiltonian_learning_utils import *


# CONSTANTS
NQUBITS = 2
DURATION = 2000
STORE_EVERY = 4
SAMPLES = 1000

# Hamiltonian parameters - ZI = 8e-3, IZ = -10e-3, XX = 1e-5, ZZ = 1e-5
HAMILTONIAN_PARAMS = dict(zi=8e-3, iz=-10e-3, xx=1e-5, zz=1e-5)

# Qubit 1 - T1 = 55 µs and T2 = 20µs. For Qubit 2 - T1 = 27.5 µs and T2 = 10µs
LINDBLADIAN_PARAMS = [dict(t1=55000, t2=20000), dict(t1=27500, t2=10000)]


# Derived quantities
tlist = jnp.arange(0, DURATION + STORE_EVERY, STORE_EVERY)
initial_states, init_index = generate_initial_states(NQUBITS, with_mixed_states=True)

# hamiltonian
hamiltonian, hamiltoninan_params = hamiltonian_from_dict(
    HAMILTONIAN_PARAMS, number_of_qubits=NQUBITS, return_filled_dict=True
)
hamiltonian *= 2 * jnp.pi

# Lindbladian
jump_operators = jump_operators_from_t1_and_t2(
    t1=[qubit["t1"] for qubit in LINDBLADIAN_PARAMS],
    t2=[qubit["t2"] for qubit in LINDBLADIAN_PARAMS],
)

# Measurement basis
measurement_basis, basis_index = generate_basis_transformations(
    number_of_qubits=NQUBITS
)

# Evolution function
solver = create_solver(
    t1=DURATION,
    t0=0,
    tlist=tlist,
    number_of_jump_operators=jump_operators.shape[0],
    adjoint=True,
)

results = solver(
    initial_state=initial_states,
    hamiltonian=hamiltonian,
    jump_operators=jump_operators,
)

# Change the basis
transformed_states = jnp.einsum(
    "mkl, tiln, mno ->imtko",
    measurement_basis,
    results.ys,
    jnp.conj(measurement_basis).transpose(0, 2, 1),
)


# Generate the samples
outcomes = get_measurements_from_states(transformed_states, samples=SAMPLES)
outcome_index = "00, 01, 10, 11".split(", ")

import xarray as xr
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

data = xr.DataArray(
    outcomes,
    dims=["initial_state", "measurement_basis", "time", "outcome"],
    coords=dict(
        time=tlist,
        initial_state=init_index,
        measurement_basis=basis_index,
        outcome=outcome_index,
    ),
)

stored_density_matrices = xr.DataArray(
    results.ys,
    dims=["time", "initial_state", "density_matrix_row", "density_matrix_col"],
    coords=dict(
        time=tlist,
        initial_state=init_index,
        density_matrix_row=[0, 1, 2, 3],
        density_matrix_col=[0, 1, 2, 3],
    ),
)

dataset = xr.Dataset({"density_matrices": stored_density_matrices, "data": data})

dataset = dataset.assign_attrs(intended_dimension_order = list(dataset._dims.keys()))

dataset.to_zarr("zarr_database/initial_dataset_with_dissipation.zarr", mode="w")

dataset = xr.open_zarr("zarr_database/initial_dataset_with_dissipation.zarr")


stored_density_matrices.to_zarr("zarr_database/initial_dataset_with_dissipation.zarr", mode="w")

# data.to_netcdf("initial_dataset_with_dissipation.nc")

# Create dropdown widgets for initial state and measurement basis
init_dropdown = widgets.Dropdown(
    options=data.initial_state.values,
    description="Initial State:",
    value=data.initial_state[0],
)

meas_dropdown = widgets.Dropdown(
    options=data.measurement_basis.values,
    description="Measurement Basis:",
    value=data.measurement_basis[0],
)

# Create a button widget for plotting
plot_button = widgets.Button(description="Plot")


# Define a function to handle button click event
def plot_button_clicked(button):
    init = init_dropdown.value
    meas = meas_dropdown.value

    # Plot the selected data
    fig, ax = plt.subplots(figsize=(8, 6))
    for outcome in data.outcome.values:
        data.sel(initial_state=init, measurement_basis=meas, outcome=outcome).plot(
            ax=ax, label=f"{outcome}"
        )
    ax.set_ylim(0, SAMPLES)
    plt.legend()
    plt.show()


# Register the button click event handler
plot_button.on_click(plot_button_clicked)

# Display the widgets
display(init_dropdown, meas_dropdown, plot_button)
