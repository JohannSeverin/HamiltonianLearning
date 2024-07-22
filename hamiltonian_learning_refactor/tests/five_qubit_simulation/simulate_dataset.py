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
NQUBITS = 5

# Simulation parameters
DURATION = 10e-6
NPOINTS = 41
SAMPLES = 1000

TIME_UNIT = 1e-9

# Define single qubit parameters
QUBIT_FREQUENCY = [5.2e9, 5.8e9, 4.2e9, 5.4e9, 4.9e9]  # Hz
QUBIT_FREQUENCY_OFFSET = [-450e3, +150e3, 50e3, 5e3, -100e3]  # Hz
QUBIT_ANHARMONICITY = [-300e6, -300e6, -300e6, -300e6, -300e6]  # Hz

# Two qubit parameters. This will be an x - x coupling in rotating frame
QUBIT_QUBIT_Z_COUPLINGS = {
    (0, 2): 100e3,
    (1, 2): 40e3,
    (3, 2): 50e3,
    (4, 2): 60e3,
}

QUBIT_QUBIT_X_COUPLINGS = {
    (0, 2): 15e3,
    (1, 2): 20e3,
    (3, 2): 25e3,
    (4, 2): 30e3,
}

# Decoherence channels. There will be no two-local decoherences in this simulation
QUBIT_T1 = [30e-6, 30e-6, 30e-6, 30e-6, 30e-6]  # s
QUBIT_T2 = [30e-6, 30e-6, 30e-6, 30e-6, 30e-6]  # s

# Define the initial state on the basis of temperature
TEMPERATURE = [50e-3, 60e-3, 50e-3, 40e-3, 25e-3]  # K

# Define readout error probabilities. This should be enough to reconstruct the matrix
P0_GIVEN_0 = [0.98, 0.95, 0.95, 0.95, 0.95]
P1_GIVEN_1 = [0.97, 0.94, 0.94, 0.94, 0.94]

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
    "QUBIT_QUBIT_COUPLING_X": QUBIT_QUBIT_X_COUPLINGS,
    "QUBIT_QUBIT_COUPLING_Z": QUBIT_QUBIT_Z_COUPLINGS,
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
            [1, np.exp(-Planck * QUBIT_FREQUENCY[i] / (TEMPERATURE[i] * Boltzmann))]
        )
        if TEMPERATURE[i] > 0
        else np.array([1, 0])
    )
    for i in range(NQUBITS)
]


if TEMPERATURE[0] > 0:
    for i in range(NQUBITS):
        qubit_occupations[i] /= np.sum(qubit_occupations[i])

initial_state = tensor(
    *[
        qubit_occupations[i][0] * fock_dm(2, 0)
        + qubit_occupations[i][1] * fock_dm(2, 1)
        for i in range(NQUBITS)
    ]
)

### Define the Hamiltonian ###

# Single qubit terms
from qutip import sigmaz, identity, tensor

# Remove single qubit hamiltonian such that we are in the rotating frame
single_qubit_hamiltonian = 0  # Rotating frame


# From offset
for i in range(NQUBITS):
    operator = [identity(2) for _ in range(NQUBITS)]
    operator[i] = sigmaz()
    single_qubit_hamiltonian += (
        2 * np.pi * TIME_UNIT * QUBIT_FREQUENCY_OFFSET[i] * tensor(*operator) / 2
    )


# Two qubit terms
from qutip import sigmax

# X-X coupling
two_qubit_hamiltonian = qutip.qzero([2] * NQUBITS)

for (i, j), coupling in QUBIT_QUBIT_X_COUPLINGS.items():
    operator = [identity(2) for _ in range(NQUBITS)]
    operator[i] = sigmax()
    operator[j] = sigmax()

    two_qubit_hamiltonian += 2 * np.pi * TIME_UNIT * coupling * tensor(operator)

for (i, j), coupling in QUBIT_QUBIT_Z_COUPLINGS.items():
    operator = [identity(2) for _ in range(NQUBITS)]
    operator[i] = sigmaz()
    operator[j] = sigmaz()

    two_qubit_hamiltonian += 2 * np.pi * TIME_UNIT * coupling * tensor(operator)


### Define the collapse operators ###
from qutip import destroy

collapse_operators = []

for i in range(NQUBITS):
    operators = [identity(2) for _ in range(NQUBITS)]
    operators[i] = destroy(2)

    collapse_operators.append(
        np.sqrt(1 / (2 * QUBIT_T1[i]) * TIME_UNIT) * tensor(operators)
    )

for i in range(NQUBITS):
    operators = [identity(2) for _ in range(NQUBITS)]
    operators[i] = sigmaz()

    collapse_operators.append(np.sqrt(1 / QUBIT_T2[i] * TIME_UNIT) * tensor(operators))


### Define readout POVMs ###
M0s = [
    np.sqrt(P0_GIVEN_0[i]) * fock_dm(2, 0) + np.sqrt(1 - P1_GIVEN_1[i]) * fock_dm(2, 1)
    for i in range(NQUBITS)
]

M1s = [
    np.sqrt(1 - P0_GIVEN_0[i]) * fock_dm(2, 0) + np.sqrt(P1_GIVEN_1[i]) * fock_dm(2, 1)
    for i in range(NQUBITS)
]

Ms_single = [[M0, M1] for M0, M1 in zip(M0s, M1s)]
Ms = []

from itertools import product

for bit_string in product([0, 1], repeat=NQUBITS):
    Ms.append(tensor(*[Ms_single[i][bit_string[i]] for i in range(NQUBITS)]))


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

dataset = xr.Dataset()

dims = ["time", "initial_gate", "final_gate", "outcome"]
shape = (NPOINTS, len(initial_gates), len(final_gates), 2**NQUBITS)

coords = {
    "time": np.linspace(0, DURATION / TIME_UNIT, NPOINTS),
    "initial_gate": list(initial_gates),
    "final_gate": list(final_gates),
    "outcome": [
        "".join(bit_string) for bit_string in product(["0", "1"], repeat=NQUBITS)
    ],
}

measurement_probabilities = np.zeros(shape)
outcome = np.zeros(shape)


for init_index, key_init in tqdm(enumerate(initial_gates), total=len(initial_gates)):
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
