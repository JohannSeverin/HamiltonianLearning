import sys

sys.path.append("../")
from utils import *
from hamiltonian_learning_utils import *


# CONSTANTS
NQUBITS = 2
DURATION = 1000
STORE_EVERY = 20

HAMILTONIAN_PARAMS = dict(xi=0.02, zz=0.00, iy=-0.00)


# Derived quantities
tlist = jnp.arange(0, DURATION + STORE_EVERY, STORE_EVERY)
initial_states, init_index = generate_initial_states(NQUBITS)
hamiltonian, hamiltoninan_params = hamiltonian_from_dict(
    HAMILTONIAN_PARAMS, number_of_qubits=NQUBITS, return_filled_dict=True
)


# Evolution function
solver = create_solver(t1=DURATION, t0=0, tlist=tlist)

results = solver(initial_state=initial_states, hamiltonian=hamiltonian)


measurement_basis, basis_index = generate_basis_transformations(
    number_of_qubits=NQUBITS
)


results.ys.shape

states = jnp.einsum(
    "mkl, tiln, mno ->imtko",
    measurement_basis,
    results.ys,
    jnp.conj(measurement_basis).transpose(0, 2, 1),
)

probs_index = "00, 01, 10, 11".split(", ")
probs = jnp.einsum("imtkk->imtk", states)

SAMPLES = 1000

from tensorflow_probability.substrates import jax as tfp

key = jax.random.PRNGKey(0)
outcomes = tfp.distributions.Multinomial(total_count=SAMPLES, probs=probs.real).sample(
    seed=key
)


import xarray as xr

data = xr.DataArray(
    outcomes,
    dims=["initial_state", "measurement_basis", "time", "outcome"],
    coords=dict(
        time=tlist,
        initial_state=init_index,
        measurement_basis=basis_index,
        outcome=probs_index,
    ),
)


# Plot examples
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].plot(
    data.time, data.sel(initial_state="xz", measurement_basis="xz", outcome="00")
)
ax[0, 1].plot(
    data.time, data.sel(initial_state="yz", measurement_basis="yz", outcome="00")
)
ax[1, 0].plot(
    data.time, data.sel(initial_state="zz", measurement_basis="zz", outcome="00")
)
