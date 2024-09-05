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

# Setup parameters for the qubits
NQUBITS = 1

# Simulation parameters
DURATION = 100e-9
TIMESTEP = 4e-9
SAMPLES = 1000

TIME_UNIT = 1e-9

# Define single qubit parameters
QUBIT_FREQUENCY = [5.2e9]  # Hz
QUBIT_FREQUENCY_OFFSET = [-450e3]  # Hz
QUBIT_ANHARMONICITY = [-300e6]  # Hz

# Decoherence channels. There will be no two-local decoherences in this simulation
QUBIT_T1 = [30e-6]  # s
QUBIT_T2 = [30e-6]  # s

# Define the initial state on the basis of temperature
TEMPERATURE = [50e-3]  # K

# Define readout error probabilities. This should be enough to reconstruct the matrix
P0_GIVEN_0 = [0.98]
P1_GIVEN_1 = [0.97]

from qutip.qip.operations import ry, rx



AMPLITUDE = 1.5e0
CENTER = 50
SIGMA = 20


# Gate to check
from scipy.stats import norm

def time_dependent_I(t, args):
    return (norm.pdf(t, loc=CENTER, scale=SIGMA) - norm.pdf(0, loc = CENTER, scale = SIGMA))* AMPLITUDE

def time_dependent_Q(t, args):
    return 0.0 * np.ones_like(t)


time_dependent_H = [
    [qutip.sigmax(), time_dependent_I],
    [qutip.sigmay(), time_dependent_Q],
]


# Initial Gates
INITIAL_GATES = {
    "z": qutip.qeye(2),  # Starts in z
    "x": ry(np.pi / 2),  # Starts in x
    "y": rx(-np.pi / 2),  # Starts in y
    "-z": qutip.sigmax(),  # Start in -z
}

# Transformations to measurement basis
PRE_MEASUREMENT_GATES = {
    "z": qutip.qeye(2),  # Measurement in z
    "x": ry(-np.pi / 2),  # Measurement in x
    "y": rx(np.pi / 2),  # Measurement in y
}

NPOINTS = int(DURATION / TIMESTEP) + 2

attributes = {
    "NQUBITS": NQUBITS,
    "DURATION": DURATION,
    "NPOINTS":  int(DURATION / TIMESTEP) + 2,
    "SAMPLES": SAMPLES,
    "TIME_UNIT": TIME_UNIT,
    "QUBIT_FREQUENCY": QUBIT_FREQUENCY,
    "QUBIT_FREQUENCY_OFFSET": QUBIT_FREQUENCY_OFFSET,
    "QUBIT_ANHARMONICITY": QUBIT_ANHARMONICITY,
    # "QUBIT_QUBIT_COUPLING_X": QUBIT_QUBIT_COUPLING_X,
    # "QUBIT_QUBIT_COUPLING_Z": QUBIT_QUBIT_COUPLING_Z,
    "QUBIT_T1": QUBIT_T1,
    "QUBIT_T2": QUBIT_T2,
    "TEMPERATURE": TEMPERATURE,
    "P0_GIVEN_0": P0_GIVEN_0,
    "P1_GIVEN_1": P1_GIVEN_1,
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

initial_state = tensor(
    qubit_occupations[0][0] * fock_dm(2, 0) + qubit_occupations[0][1] * fock_dm(2, 1),
)

### Define the Hamiltonian ###

# Single qubit terms
from qutip import sigmaz, identity, tensor

# Remove single qubit hamiltonian such that we are in the rotating frame
single_qubit_hamiltonian = 0 * (
    2 * np.pi * TIME_UNIT * QUBIT_FREQUENCY[0] * sigmaz() / 2
)

# From offset
single_qubit_hamiltonian += (
    2 * np.pi * TIME_UNIT * QUBIT_FREQUENCY_OFFSET[0] * sigmaz() / 2
)


# Two qubit terms
from qutip import sigmax

### Define the collapse operators ###
from qutip import destroy

collapse_operators = []

collapse_operators += [
    np.sqrt(1 / (2 * QUBIT_T1[0]) * TIME_UNIT) * destroy(2),
]


dephasing_rates = [1 / QUBIT_T2[i] - 1 / (2 * QUBIT_T1[i]) for i in range(NQUBITS)]

collapse_operators += [
    np.sqrt(dephasing_rates[0] * TIME_UNIT) * sigmaz(),
]


### Define readout POVMs ###
M0s = [
    np.sqrt(P0_GIVEN_0[0]) * fock_dm(2, 0) + np.sqrt(1 - P1_GIVEN_1[0]) * fock_dm(2, 1),
]

M1s = [
    np.sqrt(1 - P0_GIVEN_0[0]) * fock_dm(2, 0) + np.sqrt(P1_GIVEN_1[0]) * fock_dm(2, 1),
]

Ms = [
    M0s[0],
    M1s[0],
]


### Corating transformation ###
H0 = single_qubit_hamiltonian


### Define the initial and final gates ###
from itertools import product

initial_gates = {}

for keys in product(INITIAL_GATES, repeat=NQUBITS):
    initial_gates["".join(keys)] = tensor(*[INITIAL_GATES[key] for key in keys])

final_gates = {}

for keys in product(PRE_MEASUREMENT_GATES, repeat=NQUBITS):
    final_gates["".join(keys)] = tensor(*[PRE_MEASUREMENT_GATES[key] for key in keys])


#######################
# Run the Simulations #
#######################
from qutip import mesolve
from qutip.measurement import measurement_statistics_povm
from tqdm import tqdm
from scipy.stats import multinomial

# Setup xarray for the simulations
import xarray as xr
import ipywidgets as widgets
from IPython.display import display

dataset = xr.Dataset()

dims = ["time", "initial_gate", "final_gate", "outcome"]
shape = (NPOINTS, len(initial_gates), len(final_gates), 2**NQUBITS)

coords = {
    "time": np.arange(0, DURATION / TIME_UNIT + TIMESTEP / TIME_UNIT,  TIMESTEP / TIME_UNIT),
    "initial_gate": list(initial_gates),
    "final_gate": list(final_gates),
    "outcome": ["0", "1"],
}

measurement_probabilities = np.zeros(shape)
outcome = np.zeros(shape)


for init_index, key_init in tqdm(enumerate(initial_gates)):
    initial_state_i = (
        initial_gates[key_init] * initial_state * initial_gates[key_init].dag()
    )

    result = mesolve(
        [single_qubit_hamiltonian] + time_dependent_H,
        initial_state_i,
        np.arange(0, DURATION / TIME_UNIT + TIMESTEP / TIME_UNIT,  TIMESTEP / TIME_UNIT),
        collapse_operators,
        options=qutip.Options(store_states=True),
    )

    for time_index, time in enumerate(result.times):
        # state_rotated = corotating_transformation(result.states[time_index], time)
        state_rotated = result.states[time_index]

        for measure_index, key_measure in enumerate(final_gates):
            state_to_measure = (
                final_gates[key_measure]
                * state_rotated
                * final_gates[key_measure].dag()
            )
            # _, measurement_probs = measurement_statistics_povm(state_to_measure, Ms)
            measurement_probs = np.clip(state_to_measure.diag().real, 0, 1)

            measurement_probabilities[time_index, init_index, measure_index] = (
                measurement_probs
            )

            # Sample the measurement
            outcome[time_index, init_index, measure_index] = multinomial.rvs(
                SAMPLES, measurement_probs
            )

measurement_probabilities = xr.DataArray(
    measurement_probabilities,
    dims=dims,
    coords=coords,
)

sampled_outcome = xr.DataArray(
    outcome,
    dims=dims,
    coords=coords,
)

dataset = xr.Dataset(
    {
        "measurement_probabilities": measurement_probabilities,
        "sampled_outcome": sampled_outcome,
    }
)

for key, value in attributes.items():
    dataset.attrs[key] = value

dataset.to_zarr("dataset.zarr", mode="w")

####################
### Plot Results ###
####################
%matplotlib widget
import matplotlib.pyplot as plt

# Get the dimensions and coordinates from the dataset
initial_gates = dataset["initial_gate"].values
final_gates = dataset["final_gate"].values

# Create dropdown widgets for each dimension
dropdowns = []

dropdowns.append(
    widgets.Dropdown(
        options=coords["initial_gate"],
        description="initial_gate",
        layout=widgets.Layout(width="200px"),
    )
)

dropdowns.append(
    widgets.Dropdown(
        options=coords["final_gate"],
        description="final_gate",
        layout=widgets.Layout(width="200px"),
    )
)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
times = dataset["time"].values

ax.plot(times, [time_dependent_I(t, {}) for t in times], label = "I")
ax.plot(times, [time_dependent_Q(t, {}) for t in times], label = "Q")

ax.set(
    title = "Pulse Shape",
    xlabel = "Time",
    ylabel = "Amplitude"
)

ax.legend()
fig

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# Create the interactive plot function
def interactive_plot(*args):
    # Get the selected values from the dropdowns
    initial_state = dropdowns[0].value
    final_state = dropdowns[1].value

    # Select the corresponding data from the dataset
    selected_data = dataset.sel(initial_gate=initial_state, final_gate=final_state)

    ax.cla()

    # Plot the selected data
    ax.plot(selected_data["time"], selected_data["measurement_probabilities"])
    ax.scatter(selected_data["time"], selected_data["sampled_outcome"].sel(outcome = "0") / SAMPLES)
    ax.scatter(selected_data["time"], selected_data["sampled_outcome"].sel(outcome = "1") / SAMPLES)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Measurement Probabilities")
    ax.set_title("Selected Data")
    # plt.legend(selected_data["final_gate"])


dropdowns[0].observe(interactive_plot, names="value")
dropdowns[1].observe(interactive_plot, names="value")

# Register the interactive plot function with the dropdowns
display(*dropdowns)
widgets.interact(interactive_plot)
