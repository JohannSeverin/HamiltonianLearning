path_data = "/root/projects/HamiltonianLearning/hamiltonian_learning_refactor/fitting_123_qubits/2_qubit/"
path_simulated = "/root/projects/HamiltonianLearning/hamiltonian_learning_refactor/fitting_123_qubits/2_qubit/fitting_scripts/"

%matplotlib widget 
import matplotlib.pyplot as plt
from ipywidgets import interact, Dropdown
from IPython.display import display
import xarray as xr
import numpy as np 


dataset = xr.open_dataset(path_data + "dataset.zarr", engine="zarr")
simulated = xr.open_dataset(path_simulated + "simulated_H2L0.zarr", engine="zarr").measurement_probabilities

SAMPLES = dataset.attrs["SAMPLES"] 


# We plot the z1, z2, and z1z2 values of the states
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

x = dataset.time.values

# Get the initial states and measurement basis options
initial_states_options = simulated.initial_gate.values
measurement_basis_options = simulated.final_gate.values

# Define the interactive function
def update_plot(initial_state, measurement_basis):
    # Get the corresponding outcome values
    title = f"{initial_state} - {measurement_basis}"
    ax.clear()

    for i, outcome in enumerate(simulated.outcome.values):
        

        y = (
            dataset.sel(initial_gate = initial_state, final_gate=measurement_basis, outcome=outcome).sampled_outcome.values
            / SAMPLES
        )
        ax.scatter(x, y, c = f"C{i}")

        # Get the corresponding simulated values
        y = simulated.sel(initial_gate=initial_state, final_gate=measurement_basis, outcome=outcome).values
        
        ax.plot(x, np.array(y), f"C{i}", label=outcome)
    
    ax.legend()
    ax.set_title(title)

# Create the dropdown widgets
initial_state_dropdown = Dropdown(options=initial_states_options, description="Initial State:")
measurement_basis_dropdown = Dropdown(options=measurement_basis_options, description="Measurement Basis:")

# Define the callback function for the dropdowns
def dropdown_callback(change):
    initial_state = initial_state_dropdown.value
    measurement_basis = measurement_basis_dropdown.value
    update_plot(initial_state, measurement_basis)

# Register the callback function to the dropdowns
initial_state_dropdown.observe(dropdown_callback, names="value")
measurement_basis_dropdown.observe(dropdown_callback, names="value")

# Display the dropdowns
display(initial_state_dropdown)
display(measurement_basis_dropdown)

# Initialize the plot
initial_state = initial_state_dropdown.value
measurement_basis = measurement_basis_dropdown.value
update_plot(initial_state, measurement_basis)

# Add legend and labels to the plot
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("Probability")
ax.set_title("Measurement Probabilities")
