"""
Simulate a dataset using qutip.

This script simulates a dataset using the qutip library. It sets up parameters for the qubits, defines the initial state based on temperature, and specifies readout error probabilities. It also defines initial gates for the simulation.

Author: Johann Severin
"""

import numpy as np
import qutip

################################################
# Parameters for the simulation of the dataset #
################################################

NAME = "OneQubitDataset"

# Setup parameters for the qubits
NQUBITS = 1

# Simulation parameters
DURATION = 100e-6
NPOINTS = 100
SAMPLES = 1000 # Not used can be introduced later

TIME_UNIT = 1e-9

# Define single qubit parameters
QUBIT_FREQUENCY = [5.2e9]  # Hz
QUBIT_FREQUENCY_OFFSET = [0] # -4e5]  # Hz
QUBIT_ANHARMONICITY = [-300e6]  # Hz


# Decoherence channels. There will be no two-local decoherences in this simulation
QUBIT_T1 = [10e-6]  # s
QUBIT_T2 = [20e-6]  # s


# Define the initial state on the basis of temperature
TEMPERATURE = [0]  # [50e-3, 60e-3]  # K

# Define readout error probabilities. This should be enough to reconstruct the matrix
P0_GIVEN_0 = [1]  # [0.98, 0.95]
P1_GIVEN_1 = [1]  # [0.97, 0.94]

# Initial Gates
INITIAL_GATES = {
    "z": qutip.qeye(2),  # Starts in z
    "x": qutip.qip.ry(np.pi / 2),  # Starts in x
    "y": qutip.qip.rx(-np.pi / 2),  # Starts in y
    "-z": qutip.sigmax(),  # Start in -z
    "-x": qutip.qip.ry(-np.pi / 2),  # Start in -x
    "-y": qutip.qip.rx(np.pi / 2),  # Start in -y
}

# Transformations to measurement basis
EXPECTATION_OPERATORS = {
    "i": qutip.qeye(2),
    "x": qutip.sigmax(),
    "y": qutip.sigmay(),
    "z": qutip.sigmaz(),
}


attributes = {
    "NQUBITS": NQUBITS,
    "DURATION": DURATION,
    "NPOINTS": NPOINTS,
    "SAMPLES": SAMPLES,
    "TIME_UNIT": TIME_UNIT,
    "QUBIT_FREQUENCY": QUBIT_FREQUENCY,
    "QUBIT_FREQUENCY_OFFSET": QUBIT_FREQUENCY_OFFSET,
    "QUBIT_ANHARMONICITY": QUBIT_ANHARMONICITY,
    "QUBIT_T1": QUBIT_T1,
    "QUBIT_T2": QUBIT_T2,
    "TEMPERATURE": TEMPERATURE,
    "P0_GIVEN_0": P0_GIVEN_0,
    # "P1_GIVEN_1": P1_GIVEN_1,
}


#####################################
# Construction of Simulation Params #
#####################################

### Initial state of the qubit ###
from qutip import fock_dm, tensor
from scipy.constants import Boltzmann, Planck

qubit_occupations = [
    (
        np.array(
            [1, np.exp(-Planck * QUBIT_FREQUENCY[0] / (TEMPERATURE[0] * Boltzmann))]
        )
        if TEMPERATURE[0] > 0
        else np.array([1, 0])
    ),
]

if TEMPERATURE[0] > 0:
    for i in range(NQUBITS):
        qubit_occupations[i] /= np.sum(qubit_occupations[i])

initial_state = qubit_occupations[0][0] * fock_dm(2, 0) + qubit_occupations[0][1] * fock_dm(2, 1)

### Define the Hamiltonian ###

# Single qubit terms
from qutip import sigmaz, identity, tensor

# Remove single qubit hamiltonian such that we are in the rotating frame
single_qubit_hamiltonian = 0 * (
    2 * np.pi * TIME_UNIT * QUBIT_FREQUENCY[0] * sigmaz() / 2
)

# From offset
single_qubit_hamiltonian += (
    2
    * np.pi
    * TIME_UNIT
    * QUBIT_FREQUENCY_OFFSET[0]
    * sigmaz()
    / 2
)

### Define the collapse operators ###
from qutip import destroy

collapse_operators = []

collapse_operators += [
    np.sqrt(1 / (2 * QUBIT_T1[0]) * TIME_UNIT) * destroy(2),
]

dephasing_rates = [1 / QUBIT_T2[i] - 1 / (2 * QUBIT_T1[i]) for i in range(NQUBITS)]

collapse_operators += [
    np.sqrt(dephasing_rates[0] * TIME_UNIT) * sigmaz(), identity(2),
]

### Corating transformation ###
H0 = single_qubit_hamiltonian

### Define the initial and final gates ###
from itertools import product

initial_gates = {}

for keys in product(INITIAL_GATES, repeat=NQUBITS):
    initial_gates["".join(keys)] = tensor(*[INITIAL_GATES[key] for key in keys])

expectation_operators = {}

for keys in product(EXPECTATION_OPERATORS, repeat=NQUBITS):
    expectation_operators["".join(keys)] = tensor(
        *[EXPECTATION_OPERATORS[key] for key in keys]
    )


#######################
# Run the Simulations #
#######################
from qutip import mesolve
from qutip.measurement import measurement_statistics_povm
from tqdm import tqdm
from scipy.stats import multinomial

# Setup xarray for the simulations
import xarray as xr

dataset = xr.Dataset()

dims = ["time", "initial_gate", "expectation"]
shape = (NPOINTS, len(initial_gates), len(expectation_operators))

coords = {
    "time": np.linspace(0, DURATION / TIME_UNIT, NPOINTS),
    "initial_gate": list(initial_gates),
    "expectation": list(expectation_operators),
}

expectation_values = np.zeros(shape)


for init_index, key_init in tqdm(enumerate(initial_gates)):
    initial_state_i = (
        initial_gates[key_init] * initial_state * initial_gates[key_init].dag()
    )

    result = mesolve(
        single_qubit_hamiltonian,
        initial_state_i,
        np.linspace(0, DURATION / TIME_UNIT, NPOINTS),
        collapse_operators,
        options=qutip.Options(store_states=True),
    )

    for time_index, time in enumerate(result.times):
        # state_rotated = corotating_transformation(result.states[time_index], time)
        state = result.states[time_index]

        for operator_index, operator in enumerate(expectation_operators):

            operator_to_measure = expectation_operators[operator]

            # Find expvals
            exp_val = qutip.expect(operator_to_measure, state)

            expectation_values[time_index, init_index, operator_index] = exp_val


expectation_dataset = xr.DataArray(
    expectation_values,
    dims=dims,
    coords=coords,
)

dataset = xr.Dataset(
    {
        "expectation_values": expectation_dataset,
    }
)

for key, value in attributes.items():
    dataset.attrs[key] = value

dataset.to_zarr(f"data/{NAME}.zarr", mode="w")

####################
### Plot Results ###
####################
%matplotlib widget
import matplotlib.pyplot as plt
from ipywidgets import interact, Dropdown
from IPython.display import display


# We plot the z1, z2, and z1z2 values of the states
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

x = dataset.time

# Get the initial states and measurement basis options
initial_states_options = dataset.initial_gate.values
expectation_options = dataset.expectation.values

# Define the interactive function
def update_plot(initial_state, expectation_operator):
    # Get the corresponding outcome values
    title = f"{initial_state} - {expectation_operator}"
    ax.clear()


    y = (
        dataset.sel(initial_gate = initial_state, expectation=expectation_operator).expectation_values.values
    )

    ax.scatter(x, y, c = None, label = "Expectation Value")

    ax.set(
        xlabel="Time [ns]",
        ylabel="Expectation Value",
    )
    ax.legend()
    ax.set_title(title)

# Create the dropdown widgets
initial_state_dropdown = Dropdown(options=initial_states_options, description="Initial State:")
expectation_operator_dropdown = Dropdown(options=expectation_options, description="Expectation Value:")

# Define the callback function for the dropdowns
def dropdown_callback(change):
    initial_state = initial_state_dropdown.value
    expectation_operator = expectation_operator_dropdown.value
    update_plot(initial_state, expectation_operator)

# Register the callback function to the dropdowns
initial_state_dropdown.observe(dropdown_callback, names="value")
expectation_operator_dropdown.observe(dropdown_callback, names="value")

# Display the dropdowns
display(initial_state_dropdown)
display(expectation_operator_dropdown)

# Initialize the plot
initial_state = initial_state_dropdown.value
expectation_operator = expectation_operator_dropdown.value
update_plot(initial_state, expectation_operator)

# Add legend and labels to the plot
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("Probability")
ax.set_title("Measurement Probabilities")
