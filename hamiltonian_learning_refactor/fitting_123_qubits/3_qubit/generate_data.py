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
NQUBITS = 3

# Simulation parameters
DURATION = 10e-6
NPOINTS = 41
SAMPLES = 1000

TIME_UNIT = 1e-9

# Define single qubit parameters
QUBIT_FREQUENCY = [5.2e9, 5.8e9, 6.2e9]  # Hz
QUBIT_FREQUENCY_OFFSET = [-450e3, +150e3, 1e6]  # Hz
QUBIT_ANHARMONICITY = [-300e6, -300e6, -300e6]  # Hz

# Two qubit parameters. This will be an x - x coupling in rotating frame
QUBIT_QUBIT_COUPLING_X = 45e3  # Hz
QUBIT_QUBIT_COUPLING_Z = 15e3  # Hz

# Decoherence channels. There will be no two-local decoherences in this simulation
QUBIT_T1 = [30e-6, 30e-6, 30e-6]  # s
QUBIT_T2 = [30e-6, 30e-6, 30e-6]  # s

# Define the initial state on the basis of temperature
TEMPERATURE = [0, 0, 0]  # [50e-3, 60e-3]  # K

# Define readout error probabilities. This should be enough to reconstruct the matrix
P0_GIVEN_0 = [1.00, 1.00, 1.00]  # [0.98, 0.95]
P1_GIVEN_1 = [1.00, 1.00, 1.00]  # [0.97, 0.94]

from qutip.qip.operations import ry, rx

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


attributes = {
    "NQUBITS": NQUBITS,
    "DURATION": DURATION,
    "NPOINTS": NPOINTS,
    "SAMPLES": SAMPLES,
    "TIME_UNIT": TIME_UNIT,
    "QUBIT_FREQUENCY": QUBIT_FREQUENCY,
    "QUBIT_FREQUENCY_OFFSET": QUBIT_FREQUENCY_OFFSET,
    "QUBIT_ANHARMONICITY": QUBIT_ANHARMONICITY,
    "QUBIT_QUBIT_COUPLING_X": QUBIT_QUBIT_COUPLING_X,
    "QUBIT_QUBIT_COUPLING_Z": QUBIT_QUBIT_COUPLING_Z,
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
    (
        np.array(
            [1, np.exp(-Planck * QUBIT_FREQUENCY[1] / (TEMPERATURE[1] * Boltzmann))]
        )
        if TEMPERATURE[0] > 0
        else np.array([1, 0])
    ),
    (
        np.array(
            [1, np.exp(-Planck * QUBIT_FREQUENCY[2] / (TEMPERATURE[2] * Boltzmann))]
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
    qubit_occupations[1][0] * fock_dm(2, 0) + qubit_occupations[1][1] * fock_dm(2, 1),
    qubit_occupations[2][0] * fock_dm(2, 0) + qubit_occupations[2][1] * fock_dm(2, 1),
)

### Define the Hamiltonian ###

# Single qubit terms
from qutip import sigmaz, identity, tensor

# Remove single qubit hamiltonian such that we are in the rotating frame

# From offset
single_qubit_hamiltonian = (
    2
    * np.pi
    * TIME_UNIT
    * QUBIT_FREQUENCY_OFFSET[0]
    * tensor(sigmaz(), identity(2), identity(2))
    / 2
    + 2
    * np.pi
    * TIME_UNIT
    * QUBIT_FREQUENCY_OFFSET[1]
    * tensor(identity(2), sigmaz(), identity(2))
    / 2
    + 2
    * np.pi
    * TIME_UNIT
    * QUBIT_FREQUENCY_OFFSET[2]
    * tensor(identity(2), identity(2), sigmaz())
    / 2
)


# Two qubit terms
from qutip import sigmax

# X-X coupling
two_qubit_hamiltonian = (
    2
    * np.pi
    * TIME_UNIT
    * QUBIT_QUBIT_COUPLING_X
    * tensor(sigmax(), sigmax(), identity(2))
)

two_qubit_hamiltonian += (
    2
    * np.pi
    * TIME_UNIT
    * QUBIT_QUBIT_COUPLING_X
    * tensor(sigmax(), identity(2), sigmax())
)

# Z-Z coupling
two_qubit_hamiltonian += (
    2
    * np.pi
    * TIME_UNIT
    * QUBIT_QUBIT_COUPLING_Z
    * tensor(sigmaz(), identity(2), sigmaz())
)

### Define the collapse operators ###
from qutip import destroy

collapse_operators = []

collapse_operators += [
    np.sqrt(1 / (2 * QUBIT_T1[0]) * TIME_UNIT)
    * tensor(destroy(2), identity(2), identity(2)),
    np.sqrt(1 / (2 * QUBIT_T1[1]) * TIME_UNIT)
    * tensor(identity(2), destroy(2), identity(2)),
    np.sqrt(1 / (2 * QUBIT_T1[1]) * TIME_UNIT)
    * tensor(identity(2), identity(2), destroy(2)),
]

dephasing_rates = [1 / QUBIT_T2[i] - 1 / (2 * QUBIT_T1[i]) for i in range(NQUBITS)]

collapse_operators += [
    np.sqrt(dephasing_rates[0] * TIME_UNIT)
    * tensor(sigmaz(), identity(2), identity(2)),
    np.sqrt(dephasing_rates[1] * TIME_UNIT)
    * tensor(identity(2), sigmaz(), identity(2)),
    np.sqrt(dephasing_rates[2] * TIME_UNIT)
    * tensor(identity(2), identity(2), sigmaz()),
]


### Define readout POVMs ###
M0s = [
    np.sqrt(P0_GIVEN_0[0]) * fock_dm(2, 0) + np.sqrt(1 - P1_GIVEN_1[0]) * fock_dm(2, 1),
    np.sqrt(P0_GIVEN_0[1]) * fock_dm(2, 0) + np.sqrt(1 - P1_GIVEN_1[1]) * fock_dm(2, 1),
    np.sqrt(P0_GIVEN_0[2]) * fock_dm(2, 0) + np.sqrt(1 - P1_GIVEN_1[2]) * fock_dm(2, 1),
]

M1s = [
    np.sqrt(1 - P0_GIVEN_0[0]) * fock_dm(2, 0) + np.sqrt(P1_GIVEN_1[0]) * fock_dm(2, 1),
    np.sqrt(1 - P0_GIVEN_0[1]) * fock_dm(2, 0) + np.sqrt(P1_GIVEN_1[1]) * fock_dm(2, 1),
    np.sqrt(1 - P0_GIVEN_0[2]) * fock_dm(2, 0) + np.sqrt(P1_GIVEN_1[2]) * fock_dm(2, 1),
]

Ms = [
    tensor(M0s[0], M0s[1], M0s[2]),
    tensor(M0s[0], M0s[1], M1s[2]),
    tensor(M0s[0], M1s[1], M0s[2]),
    tensor(M0s[0], M1s[1], M1s[2]),
    tensor(M1s[0], M0s[1], M0s[2]),
    tensor(M1s[0], M0s[1], M1s[2]),
    tensor(M1s[0], M1s[1], M0s[2]),
    tensor(M1s[0], M1s[1], M1s[2]),
]


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

dataset = xr.Dataset()

dims = ["time", "initial_gate", "final_gate", "outcome"]
shape = (NPOINTS, len(initial_gates), len(final_gates), 2**NQUBITS)

coords = {
    "time": np.linspace(0, DURATION / TIME_UNIT, NPOINTS),
    "initial_gate": list(initial_gates),
    "final_gate": list(final_gates),
    "outcome": ["000", "001", "010", "011", "100", "101", "110", "111"],
}

measurement_probabilities = np.zeros(shape)
outcome = np.zeros(shape)


for init_index, key_init in tqdm(enumerate(initial_gates)):
    initial_state_i = (
        initial_gates[key_init] * initial_state * initial_gates[key_init].dag()
    )

    result = mesolve(
        single_qubit_hamiltonian + two_qubit_hamiltonian,
        initial_state_i,
        np.linspace(0, DURATION / TIME_UNIT, NPOINTS),
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
import matplotlib.pyplot as plt

measurement_probabilities = np.array(measurement_probabilities)
outcome = np.array(outcome)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

for i in range(4):
    ax.plot(dataset.time, measurement_probabilities[:, 0, 0, i], label=f"Outcome {i}")
    ax.plot(dataset.time, outcome[:, 0, 0, i] / SAMPLES, label=f"Sampled Outcome {i}")
